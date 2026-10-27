"""Stage 4a — hardware scaling benchmark.

phase single:  python  ddp_benchmark.py --phase single     (1 GPU)
phase ddp:     torchrun --nproc_per_node=2 ddp_benchmark.py --phase ddp
phase pick:    python  ddp_benchmark.py --phase pick       (merge + choose config)

Per config (micro-batch, torch.compile, ZeRO-1) it measures compute time per
micro-step (no_sync), all-reduce time, optimizer time and peak memory, then
derives the effective throughput at the real global batch (grad accumulation
=> one all-reduce per optimizer step) and the naive "sync every micro-step"
throughput, plus 2-GPU scaling efficiency and MFU.
"""
import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import torch.distributed as tdist  # noqa: E402

from common import dist  # noqa: E402
from common.config import PretrainConfig, model_preset, VOCAB_SIZE, PRE_DIR, SMOKE, stage_dir  # noqa: E402
from common.flops import flops_per_token, mfu  # noqa: E402
from common.logger import save_json, load_json, p  # noqa: E402
from common.model import GPT  # noqa: E402
from common.energy import EnergyMeter  # noqa: E402

MICROS = [4, 8] if SMOKE else [2, 4, 6, 8]
N_TIME = 3 if SMOKE else 8


def bench_one(cfg, seq, micro, use_compile, zero1, world):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = GPT(cfg).cuda()
    # memory probe without DDP so an OOM on one rank cannot hang the other in all-reduce
    ok = True
    try:
        xp = torch.randint(0, cfg.vocab_size, (micro, seq), device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            _, lp = model(xp, xp)
        lp.backward()
        model.zero_grad(set_to_none=True)
        del xp, lp
    except torch.OutOfMemoryError:
        ok = False
    if world > 1:
        ok = dist.all_sum(float(ok)) == world
    if not ok:
        del model
        gc.collect(); torch.cuda.empty_cache()
        return dict(ok=False, oom=True)
    opt = dist.make_optimizer(model, 1e-4, 0.1, (0.9, 0.95), zero1)
    fwd = torch.compile(model) if use_compile else model
    ddp = dist.wrap(fwd, torch.cuda.current_device()) if world > 1 else fwd
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randint(0, cfg.vocab_size, (micro, seq), device="cuda")

    def micro_step(sync):
        ctx = ddp.no_sync() if (world > 1 and not sync) else torch.enable_grad()
        with ctx:
            with torch.autocast("cuda", dtype=torch.float16):
                _, loss = ddp(x, x)
            scaler.scale(loss).backward()

    try:
        t_c0 = time.time()
        for _ in range(3):                       # warm-up (+ compile)
            micro_step(sync=True)
        torch.cuda.synchronize()
        warm = time.time() - t_c0
        t0 = time.time()
        for _ in range(N_TIME):
            micro_step(sync=False)
        torch.cuda.synchronize()
        t_micro = (time.time() - t0) / N_TIME
        t0 = time.time()
        for _ in range(N_TIME):
            micro_step(sync=True)
        torch.cuda.synchronize()
        t_sync = (time.time() - t0) / N_TIME
        t0 = time.time()
        for _ in range(3):
            scaler.unscale_(opt)
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize()
        t_opt = (time.time() - t0) / 3
        # pure all-reduce of a gradient-sized buffer (fp32)
        t_ar = 0.0
        if world > 1:
            buf = torch.zeros(sum(q.numel() for q in model.parameters()), device="cuda")
            tdist.all_reduce(buf); torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(3):
                tdist.all_reduce(buf)
            torch.cuda.synchronize()
            t_ar = (time.time() - t0) / 3
            del buf
        mem = torch.cuda.max_memory_allocated() / 2**30
        res = dict(ok=True, t_micro=t_micro, t_sync_micro=t_sync, t_opt=t_opt, t_allreduce=t_ar,
                   peak_mem_gb=mem, warmup_s=warm)
    except torch.OutOfMemoryError:
        res = dict(ok=False, oom=True)
    del model, opt, ddp, fwd
    gc.collect()
    torch.cuda.empty_cache()
    if world > 1:
        res = {k: (dist.all_mean(v) if isinstance(v, float) else v) for k, v in res.items()}
        ok_all = dist.all_sum(float(res["ok"])) == world
        res["ok"] = ok_all
    return res


def derive(r, micro, seq, world, global_batch, fpt):
    per_micro = micro * seq * world
    accum = max(1, round(global_batch / per_micro))
    step = accum * r["t_micro"] + (r["t_sync_micro"] - r["t_micro"]) + r["t_opt"]
    naive = accum * r["t_sync_micro"] + r["t_opt"]
    tps = accum * per_micro / step
    r.update(accum=accum, tok_per_s=tps, tok_per_s_naive_sync=accum * per_micro / naive,
             step_s=step, comm_share=max(0.0, r["t_sync_micro"] - r["t_micro"]) / step,
             mfu=mfu(tps, fpt, world))
    return r


def run_phase(phase):
    d = PretrainConfig()
    rank, local, world = dist.init()
    cfg = model_preset(d.model)
    cfg.vocab_size, cfg.max_seq_len = VOCAB_SIZE, d.seq_len
    n_ne = GPT(cfg).num_params(True)
    fpt = flops_per_token(cfg, d.seq_len, n_ne)
    meter = EnergyMeter(stage_dir("4_pre_train"), name=f"energy_bench_{phase}").start() if dist.is_main() else None
    results = []
    combos = [(m, False, False) for m in MICROS]
    for (m, c, z) in combos:
        r = bench_one(cfg, d.seq_len, m, c, z, world)
        r.update(micro_batch=m, compile=c, zero1=z, world=world)
        if r["ok"]:
            derive(r, m, d.seq_len, world, d.global_batch_tokens, fpt)
        results.append(r)
        if dist.is_main():
            p(f"[{phase}] micro {m} compile {c} zero1 {z}: " +
              (f"{r['tok_per_s']/1e3:.1f}k tok/s, mfu {r['mfu']*100:.1f}%, mem {r['peak_mem_gb']:.2f}G, "
               f"comm {r['comm_share']*100:.1f}%" if r["ok"] else "OOM"))
    ok = [r for r in results if r["ok"]]
    if ok:
        best_m = max(ok, key=lambda r: r["tok_per_s"])["micro_batch"]
        extra = [(best_m, True, False)]
        if world > 1:
            extra += [(best_m, False, True), (best_m, True, True)]
        for (m, c, z) in extra:
            r = bench_one(cfg, d.seq_len, m, c, z, world)
            r.update(micro_batch=m, compile=c, zero1=z, world=world)
            if r["ok"]:
                derive(r, m, d.seq_len, world, d.global_batch_tokens, fpt)
            results.append(r)
            if dist.is_main():
                p(f"[{phase}] micro {m} compile {c} zero1 {z}: " +
                  (f"{r['tok_per_s']/1e3:.1f}k tok/s, mem {r['peak_mem_gb']:.2f}G" if r["ok"] else "OOM"))
    if dist.is_main():
        save_json(os.path.join(PRE_DIR, f"bench_{phase}.json"),
                  {"results": results, "flops_per_token": fpt, "params_nonemb": n_ne, "energy": meter.stop()})
    dist.cleanup()


def pick():
    d = PretrainConfig()
    single = load_json(os.path.join(PRE_DIR, "bench_single.json"), {"results": []})
    ddp = load_json(os.path.join(PRE_DIR, "bench_ddp.json"), {"results": []})
    ok2 = [r for r in ddp["results"] if r["ok"]]
    ok1 = [r for r in single["results"] if r["ok"]]
    if not ok2:
        raise SystemExit("no working 2-GPU configuration")
    # keep ~0.6 GB headroom below the per-process cap (machine stability)
    safe = [r for r in ok2 if r["peak_mem_gb"] < 8 * 0.85 - 0.6] or ok2
    best = max(safe, key=lambda r: r["tok_per_s"])
    best1 = max(ok1, key=lambda r: r["tok_per_s"]) if ok1 else None
    out = {"best": best, "best_single": best1, "single": single["results"], "ddp": ddp["results"],
           "flops_per_token": ddp.get("flops_per_token"), "params_nonemb": ddp.get("params_nonemb"),
           "energy": {"single": single.get("energy"), "ddp": ddp.get("energy")}}
    if best1:
        out["scaling_efficiency"] = best["tok_per_s"] / (2 * best1["tok_per_s"])
        same = [r for r in ok1 if r["micro_batch"] == best["micro_batch"] and r["compile"] == best["compile"]]
        if same:
            out["scaling_efficiency_naive_sync"] = best["tok_per_s_naive_sync"] / (2 * same[0]["tok_per_s"])
    tokens = best["tok_per_s"] * 0.93 * d.hours * 3600
    out["planned_tokens"] = tokens
    out["planned_flops"] = tokens * ddp.get("flops_per_token", 0)
    save_json(os.path.join(PRE_DIR, "ddp_benchmark.json"), out)
    p(f"best: micro {best['micro_batch']} compile {best['compile']} zero1 {best['zero1']} "
      f"{best['tok_per_s']/1e3:.1f}k tok/s, mfu {best['mfu']*100:.1f}%, scaling eff "
      f"{out.get('scaling_efficiency', float('nan')):.2f}; plan {tokens/1e9:.2f}B tokens in {d.hours} h")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["single", "ddp", "pick"], required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    stage_dir("4_pre_train")
    pick() if a.phase == "pick" else run_phase(a.phase)
