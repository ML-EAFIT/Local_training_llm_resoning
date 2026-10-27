"""Stage 4 — pre-training (next-token prediction, teacher forcing, Lecture 3).

L = -(1/T) sum_t log p(x_t | x_<t);   BPB = L / ln 2 / (bytes per token)

Run on both GPUs:  torchrun --nproc_per_node=2 4_pre_train/pretrain.py
Also used (single GPU, small presets, --out) by the scaling sweep and ablations.
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from common import dist, checkpoint  # noqa: E402
from common.config import PretrainConfig, model_preset, TOK_DIR, PRE_DIR, VOCAB_SIZE, SMOKE  # noqa: E402
from common.data import ShardLoader, sequential_batches, Tok  # noqa: E402
from common.energy import EnergyMeter, NullMeter  # noqa: E402
from common.logger import save_json, load_json, p  # noqa: E402
from common.model import GPT  # noqa: E402
from common.trainer import TrainArgs, train  # noqa: E402


def parse():
    d = PretrainConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=d.model)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tokens", type=float, default=d.max_tokens, help="0 = from benchmark x hours")
    ap.add_argument("--hours", type=float, default=d.hours)
    ap.add_argument("--seq_len", type=int, default=d.seq_len)
    ap.add_argument("--micro_batch", type=int, default=0, help="0 = from benchmark")
    ap.add_argument("--global_batch", type=int, default=d.global_batch_tokens)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--warmup", type=int, default=d.warmup_steps)
    ap.add_argument("--eval_every", type=int, default=d.eval_every)
    ap.add_argument("--eval_tokens", type=int, default=d.eval_tokens)
    ap.add_argument("--bench_every", type=int, default=d.bench_every)
    ap.add_argument("--bench_limit", type=int, default=200 if SMOKE else 1000)
    ap.add_argument("--compile", type=int, default=-1)
    ap.add_argument("--zero1", type=int, default=-1)
    ap.add_argument("--snapshots", default="" if SMOKE else "0.1,0.25,0.5,0.75")
    # architecture overrides (ablations)
    ap.add_argument("--pos_emb", default=None)
    ap.add_argument("--norm", default=None)
    ap.add_argument("--prenorm", type=int, default=None)
    ap.add_argument("--n_kv_head", type=int, default=None)
    ap.add_argument("--n_experts", type=int, default=None)
    ap.add_argument("--moe_aux_coef", type=float, default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--quiet_bench", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def domain_bpt():
    tk = load_json(os.path.join(TOK_DIR, "tokenize.json"), {})
    return {d: s["val_bytes"] / max(s["val_tokens"], 1) for d, s in tk.get("pre", {}).items()}


def make_val_fn(seq_len, micro, eval_tokens, bench_every, bench_limit, quiet_bench):
    bpt = domain_bpt()
    tok = Tok()
    from eval.benchmarks import multiple_choice

    @torch.no_grad()
    def val(model, step):
        out = {}
        r, w = dist.rank(), dist.world()
        tot_nats = tot_bytes = 0.0
        for d in bpt:
            path = os.path.join(TOK_DIR, f"val_{d}.bin")
            losses = []
            for i, (x, y) in enumerate(sequential_batches(path, seq_len, micro, eval_tokens)):
                if i % w != r:
                    continue
                with torch.autocast("cuda", dtype=torch.float16):
                    _, l = model(x, y)
                losses.append(l.item())
            n = dist.all_sum(len(losses))
            loss = dist.all_sum(sum(losses)) / max(n, 1)
            out[f"val_loss_{d}"] = loss
            out[f"bpb_{d}"] = loss / math.log(2) / bpt[d]
            tot_nats += loss
            tot_bytes += bpt[d]
        out["val_loss"] = tot_nats / max(len(bpt), 1)
        out["bpb"] = sum(out[f"bpb_{d}"] for d in bpt) / max(len(bpt), 1)
        if not quiet_bench and bench_every and step % bench_every == 0:
            out["hellaswag"] = multiple_choice(model, tok, "hellaswag", bench_limit)["acc_norm"]
        if dist.is_main():
            p(f"eval step {step}: val_loss {out['val_loss']:.4f} bpb {out['bpb']:.4f}"
              + (f" hellaswag {out['hellaswag']:.3f}" if "hellaswag" in out else ""))
        return out
    return val


def main():
    args = parse()
    rank, local, world = dist.init()
    torch.manual_seed(1337 + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    d = PretrainConfig()
    out = args.out or PRE_DIR
    os.makedirs(out, exist_ok=True)
    bench = load_json(os.path.join(PRE_DIR, "ddp_benchmark.json"), {})
    micro = args.micro_batch or bench.get("best", {}).get("micro_batch", d.micro_batch)
    use_compile = bool(args.compile) if args.compile >= 0 else bench.get("best", {}).get("compile", d.compile)
    zero1 = bool(args.zero1) if args.zero1 >= 0 else bench.get("best", {}).get("zero1", d.zero1)

    cfg = model_preset(args.model)
    cfg.vocab_size = VOCAB_SIZE
    cfg.max_seq_len = args.seq_len
    cfg.rope_orig_len = args.seq_len
    for k in ("pos_emb", "norm", "n_kv_head", "n_experts", "moe_aux_coef", "window"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    if args.prenorm is not None:
        cfg.prenorm = bool(args.prenorm)
    model = GPT(cfg).cuda()

    # token budget: explicit, or fill the time slot using the measured throughput
    tokens = args.tokens
    if not tokens:
        tps = bench.get("best", {}).get("tok_per_s")
        if not tps:
            raise SystemExit("no --tokens and no ddp_benchmark.json: run ddp_benchmark.py first")
        tokens = tps * 0.93 * args.hours * 3600   # ~7% for evals / checkpoints
    per_micro = micro * args.seq_len * world
    accum = max(1, round(args.global_batch / per_micro))
    tps_step = accum * per_micro
    steps = max(1, int(tokens // tps_step))
    a = TrainArgs(steps=steps, accum=accum, tokens_per_step=tps_step, lr=args.lr, warmup=min(args.warmup, steps // 10 + 1),
                  eval_every=args.eval_every, compile=use_compile, zero1=zero1, seq_len=args.seq_len,
                  snapshot_fracs=tuple(float(s) for s in args.snapshots.split(",") if s),
                  ckpt_minutes=d.ckpt_minutes, weight_decay=d.weight_decay, decay_frac=d.decay_frac,
                  min_lr_frac=d.min_lr_frac, log_every=1 if steps < 200 else 10)
    if dist.is_main():
        p(f"model {args.model}: {model.num_params()/1e6:.1f}M params ({model.num_params(True)/1e6:.1f}M non-emb), "
          f"{steps} steps x {tps_step} tok = {steps*tps_step/1e9:.3f}B tokens, micro {micro} accum {accum} "
          f"world {world} compile {use_compile} zero1 {zero1}")
        save_json(os.path.join(out, "run_config.json"), {"model_cfg": cfg.to_dict(), "train": vars(args),
                                                          "steps": steps, "accum": accum, "micro": micro})
    loader = ShardLoader(os.path.join(TOK_DIR, "train.bin"), args.seq_len, micro, rank, world)
    meter = EnergyMeter(out).start() if dist.is_main() else NullMeter()
    val_fn = make_val_fn(args.seq_len, micro, args.eval_tokens, args.bench_every, args.bench_limit, args.quiet_bench)
    t0 = time.time()
    summ = train(model, cfg, a, loader.next, out, eval_fn=val_fn, meter=meter, state_obj=loader)
    final_eval = val_fn(model.eval(), 0 if args.quiet_bench else args.bench_every)
    if dist.is_main():
        checkpoint.save(os.path.join(out, "final.pt"), model, cfg, step=steps)
        summ.update(final_eval=final_eval, seconds=time.time() - t0, energy=meter.stop(), model=args.model,
                    world=world, micro=micro)
        summ["tokens_per_param"] = summ["tokens"] / summ["params"]
        if cfg.n_experts:   # expert utilisation on validation data (routing collapse check)
            counts = [b.ffn.counts.tolist() for b in model.blocks]
            ent = [float(-(torch.tensor(c) * torch.tensor(c).clamp(min=1e-9).log()).sum() / math.log(len(c)))
                   for c in counts]
            summ["moe_counts"] = counts
            summ["moe_norm_entropy"] = ent
        save_json(os.path.join(out, "summary.json"), summ)
        p(f"done: val_loss {final_eval['val_loss']:.4f} bpb {final_eval['bpb']:.4f} "
          f"in {(time.time()-t0)/3600:.2f} h, {summ['energy'].get('total_kwh', 0):.3f} kWh")
    dist.barrier()
    dist.cleanup()


if __name__ == "__main__":
    main()
