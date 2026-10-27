"""Best-of-N with the reward model (no RL training, old Lecture 5).

Sample N responses per held-out prompt from the final policy, keep the one the
reward model scores highest. Reports mean RM score of the selected response vs
N, compared with picking a random sample (the expected reward of plain sampling).
"""
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "5_generation"))
import torch  # noqa: E402

from common.config import POST_DIR, SMOKE  # noqa: E402
from common.data import Tok, chat_prompt  # noqa: E402
from common.energy import EnergyMeter  # noqa: E402
from common.logger import save_json, p  # noqa: E402
from common.model import load_model  # noqa: E402
from engine import Engine  # noqa: E402
from post_common import read_jsonl, ckpt  # noqa: E402
from reward_model import load_rm, score  # noqa: E402

N_MAX = 16
N_PROMPTS = 8 if SMOKE else 64


def main():
    out = os.path.join(POST_DIR, "best_of_n")
    os.makedirs(out, exist_ok=True)
    meter = EnergyMeter(out).start()
    tok = Tok()
    model, _ = load_model(os.path.join(POST_DIR, "final.pt"))
    model.half().eval()
    eng = Engine(model, tok)
    rm = load_rm()
    prompts = [r["prompt"] for r in read_jsonl("pref_test.jsonl")[:N_PROMPTS]]
    scores = []
    example = None
    for q in prompts:
        outs = [tok.decode(o) for o in eng.generate(chat_prompt(tok, q)[-512:], 160, temperature=0.8, top_p=0.95, n=N_MAX)]
        s = score(rm, tok, [q] * N_MAX, outs)
        scores.append(s)
        if example is None:
            order = sorted(range(N_MAX), key=lambda i: -s[i])
            example = {"prompt": q, "best": outs[order[0]], "best_score": s[order[0]],
                       "worst": outs[order[-1]], "worst_score": s[order[-1]]}
    rng = random.Random(0)
    curve = {}
    for n in (1, 2, 4, 8, 16):
        best, rand = [], []
        for s in scores:
            for _ in range(8):            # average over random subsets of size n
                sub = rng.sample(s, n)
                best.append(max(sub)); rand.append(sub[0])
        curve[n] = {"best_of_n": sum(best) / len(best), "random": sum(rand) / len(rand)}
    save_json(os.path.join(out, "summary.json"), {"curve": curve, "example": example, "n_prompts": len(prompts),
                                                   "energy": meter.stop()})
    p("best-of-N RM score:", {n: round(v["best_of_n"], 3) for n, v in curve.items()})


if __name__ == "__main__":
    with torch.no_grad():
        main()
