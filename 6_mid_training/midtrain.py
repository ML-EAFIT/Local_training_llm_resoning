"""Stage 6b — mid-training: same next-token objective, higher-quality data, long context.

Curriculum 1024 -> 2048 -> 4096 tokens. At each phase RoPE frequencies are
rescaled with YaRN (scale = L / 1024) so long-range position differences fall in
ranges seen during short-context pre-training (Lecture 3). Gradient
checkpointing keeps 4k sequences inside 8 GB.
Run: torchrun --nproc_per_node=2 6_mid_training/midtrain.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import dist, checkpoint  # noqa: E402
from common.config import PRE_DIR, MID_DIR, TOK_DIR, SMOKE  # noqa: E402
from common.data import sequential_batches  # noqa: E402
from common.energy import EnergyMeter, NullMeter  # noqa: E402
from common.logger import save_json, load_json, p  # noqa: E402
from common.model import load_model  # noqa: E402
from common.trainer import TrainArgs, train  # noqa: E402
from build_mid_data import PHASES  # noqa: E402

LR = 3e-3 if SMOKE else 4e-4
GLOBAL_TOKENS = 16384 if SMOKE else 262144


class PackedLoader:
    def __init__(self, L, micro, rank, world, seed=0):
        self.rows = np.memmap(os.path.join(MID_DIR, f"mid_L{L}.bin"), dtype=np.uint16, mode="r").reshape(-1, L + 1)
        self.mask = np.memmap(os.path.join(MID_DIR, f"mid_L{L}.mask.bin"), dtype=np.uint8, mode="r").reshape(-1, L + 1)
        self.micro, self.rank, self.world = micro, rank, world
        self.order = np.arange(rank, len(self.rows), world)
        self.cursor = 0

    def state(self):
        return {"cursor": self.cursor}

    def load_state(self, s):
        self.cursor = s["cursor"]

    def next(self):
        if self.cursor + self.micro > len(self.order):
            self.cursor = 0
        idx = self.order[self.cursor:self.cursor + self.micro]
        self.cursor += self.micro
        r = torch.from_numpy(self.rows[idx].astype(np.int64)).cuda(non_blocking=True)
        m = torch.from_numpy(self.mask[idx].astype(np.float32)).cuda(non_blocking=True)
        return r[:, :-1], r[:, 1:], m[:, 1:]


def short_val(model, seq_len=1024 if not SMOKE else 256):
    out = {}
    for d in ("web", "code", "math", "wiki"):
        path = os.path.join(TOK_DIR, f"val_{d}.bin")
        if not os.path.exists(path):
            continue
        ls = []
        for i, (x, y) in enumerate(sequential_batches(path, seq_len, 4, 262144 if not SMOKE else 16384)):
            if i % dist.world() != dist.rank():
                continue
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                ls.append(model(x, y)[1].item())
        out[f"val_loss_{d}"] = dist.all_sum(sum(ls)) / max(dist.all_sum(len(ls)), 1)
    return out


def main():
    rank, local, world = dist.init()
    torch.manual_seed(7 + rank)
    meter = EnergyMeter(MID_DIR).start() if dist.is_main() else NullMeter()
    model, _ = load_model(os.path.join(PRE_DIR, "final.pt"))
    cfg = model.cfg
    base_len = cfg.rope_orig_len
    before = short_val(model.eval())
    if dist.is_main():
        p(f"short-context val before mid-training: {before}")
    phases = []
    t0 = time.time()
    for k, (L, frac) in enumerate(PHASES):
        done_marker = os.path.join(MID_DIR, f"phase_L{L}.done.pt")
        model.set_rope(scale=max(1.0, L / base_len), orig_len=base_len, max_seq_len=L)
        if os.path.exists(done_marker):
            checkpoint.load(done_marker, model)
            phases.append(load_json(os.path.join(MID_DIR, f"phase_L{L}.json")))
            continue
        cfg.grad_ckpt = L >= 2048 and not SMOKE
        micro = max(1, (4096 if not SMOKE else 2048) // L)
        info = load_json(os.path.join(MID_DIR, "mid_data.json"))["phases"][str(L)]
        accum = max(1, GLOBAL_TOKENS // (micro * L * world))
        tps = accum * micro * L * world
        steps = max(1, info["rows"] * L // tps)
        last = k == len(PHASES) - 1
        a = TrainArgs(steps=steps, accum=accum, tokens_per_step=tps, lr=LR, warmup=min(50, steps // 10 + 1),
                      schedule="wsd" if last else "constant", decay_frac=0.3, min_lr_frac=0.1,
                      eval_every=10**9, log_every=1 if steps < 200 else 5, seq_len=L,
                      compile=False, ckpt_minutes=20)
        loader = PackedLoader(L, micro, rank, world)
        if dist.is_main():
            p(f"mid phase L={L} rope_scale={cfg.rope_scale}: {steps} steps x {tps} tokens, micro {micro} accum {accum}")
        s = train(model, cfg, a, loader.next, MID_DIR, meter=meter, state_obj=loader,
                  ckpt_name=f"ckpt_L{L}.pt", log_name=f"metrics_L{L}")
        s["seq_len"], s["rope_scale"] = L, cfg.rope_scale
        phases.append(s)
        if dist.is_main():
            checkpoint.save(done_marker, model, cfg, step=steps)
            save_json(os.path.join(MID_DIR, f"phase_L{L}.json"), s)
        dist.barrier()
    cfg.grad_ckpt = False
    after = short_val(model.eval())
    if dist.is_main():
        checkpoint.save(os.path.join(MID_DIR, "final.pt"), model, cfg, step=0)
        summ = {"phases": phases, "short_val_before": before, "short_val_after": after,
                "seconds": time.time() - t0, "energy": meter.stop(), "rope_scale": cfg.rope_scale,
                "max_seq_len": cfg.max_seq_len,
                "tokens": sum(ph["tokens"] for ph in phases)}
        save_json(os.path.join(MID_DIR, "summary.json"), summ)
        p(f"mid-training done; short val before {before} after {after}")
    dist.barrier()
    dist.cleanup()


if __name__ == "__main__":
    main()
