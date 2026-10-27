"""Stage 6c — long-context evaluation.

needle-in-a-haystack: accuracy over (context length x needle depth)
loss vs. token position on long held-out text (does the model use far context?)
Usage: python 6_mid_training/needle_eval.py --ckpt X --tag before|after [--rope_scale 4]
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from common.config import TOK_DIR, MID_DIR, SMOKE, stage_dir  # noqa: E402
from common.data import Tok  # noqa: E402
from common.logger import save_json, p  # noqa: E402
from common.model import load_model  # noqa: E402
from needles import make_needle_doc, EVAL_FIRST  # noqa: E402

LENGTHS = [128, 256, 512, 1024] if SMOKE else [512, 1024, 2048, 3072, 4096]
DEPTHS = [0.0, 0.25, 0.5, 0.75, 1.0]
TRIALS = 3 if SMOKE else 12


@torch.no_grad()
def answer_correct(model, tok, ids, tail, val):
    prompt = ids[:-tail]
    x = torch.tensor([prompt], device="cuda")
    out = []
    cache = model.new_cache(1, len(prompt) + 8)
    inp = x
    for _ in range(4):
        with torch.autocast("cuda", dtype=torch.float16):
            lg, _ = model(inp, cache=cache)
        nt = int(lg[0, -1].argmax())
        out.append(nt)
        inp = torch.tensor([[nt]], device="cuda")
    return val in tok.decode(out).replace(" ", "")


@torch.no_grad()
def loss_by_position(model, filler, L, n_seq=4):
    buckets = np.unique(np.geomspace(1, L, 12).astype(int))
    tot = np.zeros(len(buckets) - 1)
    for s in range(n_seq):
        st = s * L
        if st + L + 1 > len(filler):
            break
        x = torch.tensor(np.asarray(filler[st:st + L + 1], dtype=np.int64)[None], device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            lg, _ = model(x[:, :-1])
        ce = F.cross_entropy(lg[0].float(), x[0, 1:], reduction="none").cpu().numpy()
        for i, (a, b) in enumerate(zip(buckets[:-1], buckets[1:])):
            tot[i] += ce[a:b].mean()
    return {"bucket_start": buckets[:-1].tolist(), "loss": (tot / n_seq).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rope_scale", type=float, default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    stage_dir("6_mid_training")
    tok = Tok()
    model, _ = load_model(a.ckpt)
    model.half().eval()
    if a.rope_scale:
        model.set_rope(scale=a.rope_scale, max_seq_len=max(LENGTHS))
    model.set_rope(max_seq_len=max(model.cfg.max_seq_len, max(LENGTHS)))
    filler = np.memmap(os.path.join(TOK_DIR, "val_web.bin"), dtype=np.uint16, mode="r")
    filler = np.asarray(filler[:3_000_000]).astype(np.int64)
    filler = filler[filler != tok.eos]
    rng = random.Random(123)
    grid = np.zeros((len(LENGTHS), len(DEPTHS)))
    for i, L in enumerate(LENGTHS):
        for j, d in enumerate(DEPTHS):
            ok = 0
            for _ in range(TRIALS):
                ids, tail, val = make_needle_doc(tok, rng, L, filler_ids=filler, depth=d, names=EVAL_FIRST)
                ok += answer_correct(model, tok, ids, tail, val)
            grid[i, j] = ok / TRIALS
        p(f"[{a.tag}] L={L}: " + " ".join(f"{v:.2f}" for v in grid[i]))
    res = {"tag": a.tag, "ckpt": a.ckpt, "rope_scale": model.cfg.rope_scale, "lengths": LENGTHS, "depths": DEPTHS,
           "accuracy": grid.tolist(), "mean_accuracy": float(grid.mean()),
           "loss_by_position": loss_by_position(model, filler, max(LENGTHS))}
    save_json(os.path.join(MID_DIR, f"needle_{a.tag}.json"), res)


if __name__ == "__main__":
    main()
