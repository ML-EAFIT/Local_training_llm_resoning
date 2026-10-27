"""Stage 4b — scaling-law sweep + architecture ablations (Lecture 2).

IsoFLOP grid (Chinchilla approach 2): for each compute budget C, train several
model sizes N with D = C / FLOPs-per-token tokens and record the final loss.
Ablations at a fixed small budget compare the design choices of the modern
recipe. Two jobs run at once, one per GPU (small models do not need DDP).
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import PRE_DIR, SMOKE, VOCAB_SIZE, model_preset, stage_dir  # noqa: E402
from common.flops import flops_per_token  # noqa: E402
from common.logger import save_json, load_json, p  # noqa: E402
from common.model import GPT  # noqa: E402
from common.energy import EnergyMeter  # noqa: E402

SWEEP = os.path.join(PRE_DIR, "sweep")
SEQ = 256 if SMOKE else 1024
GLOBAL_BATCH = 8192 if SMOKE else 131072
SCALE = float(os.environ.get("SWEEP_SCALE", "1.0"))   # shrink/grow every budget

if SMOKE:
    ISOFLOP = {2e12: ["nano", "micro"], 6e12: ["nano", "micro", "tiny"]}
    ABL_MODEL, ABL_BUDGET = "micro", 3e12
else:
    ISOFLOP = {2e15: ["micro", "tiny", "small"],
               6e15: ["tiny", "small", "medium"],
               2e16: ["small", "medium", "large"]}
    ABL_MODEL, ABL_BUDGET = "small", 3e15

ABLATIONS = {
    "baseline": {},
    "pos_learned": {"pos_emb": "learned"},
    "pos_none": {"pos_emb": "none"},
    "attn_mha": {"n_kv_head": -1},          # -1 -> n_head
    "attn_mqa": {"n_kv_head": 1},
    "postnorm_layernorm": {"norm": "layernorm", "prenorm": 0},
    "sliding_window": {"window": 64 if SMOKE else 256},
    "moe_8x_top2": {"n_experts": 8},
    "moe_8x_top2_noaux": {"n_experts": 8, "moe_aux_coef": 0.0},
}


def fpt_of(name, **over):
    cfg = model_preset(name)
    cfg.vocab_size, cfg.max_seq_len = VOCAB_SIZE, SEQ
    for k, v in over.items():
        setattr(cfg, k, v)
    m = GPT(cfg)
    n_ne = m.num_params(True)
    # MoE: active params only (top-k experts + shared), approximated by dense equivalent
    return flops_per_token(cfg, SEQ, n_ne if cfg.n_experts == 0 else model_n_dense(name)), n_ne, cfg


def model_n_dense(name):
    cfg = model_preset(name)
    cfg.vocab_size, cfg.max_seq_len = VOCAB_SIZE, SEQ
    return GPT(cfg).num_params(True)


def lr_for(name):
    d = model_preset(name).d_model
    return min(3e-3, 1.2e-3 * (768 / d) ** 0.5)


def jobs():
    js = []
    for C, sizes in ISOFLOP.items():
        C = C * SCALE
        for s in sizes:
            fpt, n, _ = fpt_of(s)
            js.append({"name": f"iso_{C:.0e}_{s}", "kind": "isoflop", "model": s, "budget": C,
                       "tokens": C / fpt, "params_nonemb": n, "args": {}})
    for a, over in ABLATIONS.items():
        args = dict(over)
        if args.get("n_kv_head") == -1:
            args["n_kv_head"] = model_preset(ABL_MODEL).n_head
        fpt, n, _ = fpt_of(ABL_MODEL, **{k: v for k, v in args.items() if k in ("n_experts",)})
        js.append({"name": f"abl_{a}", "kind": "ablation", "ablation": a, "model": ABL_MODEL,
                   "budget": ABL_BUDGET * SCALE, "tokens": ABL_BUDGET * SCALE / fpt, "params_nonemb": n,
                   "args": args})
    return js


def launch(job, gpu):
    out = os.path.join(SWEEP, job["name"])
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "pretrain.py"), "--model", job["model"],
           "--out", out, "--tokens", str(int(job["tokens"])), "--seq_len", str(SEQ),
           "--global_batch", str(GLOBAL_BATCH), "--micro_batch", "8" if SMOKE else "16",
           "--lr", str(lr_for(job["model"])), "--warmup", "20" if SMOKE else "100",
           "--eval_every", "100000", "--quiet_bench", "--snapshots", "", "--zero1", "0",
           "--compile", "0" if SMOKE or job["args"].get("n_experts") else "1",
           "--eval_tokens", "32768" if SMOKE else "1048576"]
    for k, v in job["args"].items():
        cmd += [f"--{k}", str(v)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
               LLM_ENERGY_SHARE="0.5")
    env.pop("RANK", None); env.pop("WORLD_SIZE", None); env.pop("LOCAL_RANK", None)
    log = open(os.path.join(out, "stdout.log"), "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def main():
    stage_dir("4_pre_train")
    os.makedirs(SWEEP, exist_ok=True)
    meter = EnergyMeter(SWEEP, name="energy_sweep").start()
    todo = [j for j in jobs() if not os.path.exists(os.path.join(SWEEP, j["name"], "summary.json"))]
    total = sum(j["budget"] for j in jobs())
    p(f"sweep: {len(jobs())} runs ({len(todo)} to do), total {total:.2e} FLOPs")
    # biggest first so the two GPUs finish together
    todo.sort(key=lambda j: -j["budget"])
    running = {}
    t0 = time.time()
    while todo or running:
        for gpu in (0, 1):
            if gpu not in running and todo:
                j = todo.pop(0)
                running[gpu] = (j, launch(j, gpu))
                p(f"GPU{gpu}: start {j['name']} ({j['tokens']/1e6:.1f}M tokens, {j['params_nonemb']/1e6:.2f}M params)")
        time.sleep(5)
        for gpu in list(running):
            j, proc = running[gpu]
            if proc.poll() is not None:
                ok = proc.returncode == 0
                p(f"GPU{gpu}: {'done' if ok else 'FAILED'} {j['name']} after {(time.time()-t0)/60:.1f} min")
                del running[gpu]
    rows = []
    for j in jobs():
        s = load_json(os.path.join(SWEEP, j["name"], "summary.json"))
        if not s:
            continue
        rows.append({**{k: v for k, v in j.items() if k != "args"}, "args": j["args"],
                     "val_loss": s["final_eval"]["val_loss"], "bpb": s["final_eval"]["bpb"],
                     "tokens_trained": s["tokens"], "flops": s["flops"], "seconds": s["seconds"],
                     "kwh": s.get("energy", {}).get("total_kwh"),
                     "moe_counts": s.get("moe_counts")})
    save_json(os.path.join(PRE_DIR, "sweep.json"), {"runs": rows, "energy": meter.stop(), "seq_len": SEQ,
                                                    "global_batch": GLOBAL_BATCH})
    p(f"sweep finished: {len(rows)} runs in {(time.time()-t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
