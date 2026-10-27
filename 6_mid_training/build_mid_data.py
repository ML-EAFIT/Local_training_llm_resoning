"""Stage 6a — build the mid-training mixture (Lecture 3, GLM-5 case study).

- high-quality pool from Stage 1/2 (edu score >= 4, finemath-4plus, documented code, long wiki)
- synthetic needle documents to teach long-range retrieval (fights context rot)
- replay of pre-training data to limit forgetting
- best-fit packing per curriculum phase (1024 -> 2048 -> 4096 tokens)
Output: mid_L{len}.bin (uint16 rows of len+1) + mid_L{len}.mask.bin
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from common.config import TOK_DIR, MID_DIR, SMOKE, stage_dir  # noqa: E402
from common.data import Tok, best_fit_pack  # noqa: E402
from common.logger import save_json, p  # noqa: E402
from needles import make_needle_doc  # noqa: E402

TOTAL_TOKENS = 1_500_000 if SMOKE else 200_000_000
PHASES = [(256, 0.3), (512, 0.35), (1024, 0.35)] if SMOKE else [(1024, 0.3), (2048, 0.35), (4096, 0.35)]
MIX = {"hq": 0.75, "needle": 0.05, "replay": 0.20}


def load_docs(name):
    data = np.memmap(os.path.join(TOK_DIR, f"mid_{name}.bin"), dtype=np.uint16, mode="r")
    off = np.load(os.path.join(TOK_DIR, f"mid_{name}.idx.npy"))
    return [np.asarray(data[off[i]:off[i + 1]]).tolist() for i in range(len(off) - 1)]


def main():
    out = stage_dir("6_mid_training")
    rng = random.Random(0)
    tok = Tok()
    hq = []
    srcs = {}
    for f in sorted(os.listdir(TOK_DIR)):
        if f.startswith("mid_") and f.endswith(".bin") and not f.startswith("mid_L"):
            name = f[4:-4]
            d = load_docs(name)
            srcs[name] = sum(len(x) for x in d)
            hq += d
    rng.shuffle(hq)
    hq_tokens = sum(len(d) for d in hq)
    train = np.memmap(os.path.join(TOK_DIR, "train.bin"), dtype=np.uint16, mode="r")
    stats = {"sources_tokens": srcs, "phases": {}}
    hq_i = 0
    for L, frac in PHASES:
        budget = TOTAL_TOKENS * frac
        docs, n_hq, n_needle, n_replay = [], 0, 0, 0
        # high-quality docs (cycled if the pool is smaller than the budget: <=4 epochs is fine, Lecture 3)
        while n_hq < budget * MIX["hq"]:
            d = hq[hq_i % len(hq)]
            hq_i += 1
            docs.append(d)
            n_hq += len(d)
        # needle documents, filler taken from high-quality docs
        while n_needle < budget * MIX["needle"]:
            target = rng.randint(L // 4, L)
            ids = make_needle_doc(tok, rng, target, filler_docs=hq)
            docs.append(ids)
            n_needle += len(ids)
        # replay chunks of pre-training data
        while n_replay < budget * MIX["replay"]:
            s = rng.randrange(0, len(train) - L - 1)
            docs.append(np.asarray(train[s:s + L]).tolist())
            n_replay += L
        rows, mask = best_fit_pack(docs, L + 1, tok.pad)
        perm = np.random.default_rng(L).permutation(len(rows))
        rows, mask = rows[perm], mask[perm]
        rows.tofile(os.path.join(MID_DIR, f"mid_L{L}.bin"))
        mask.tofile(os.path.join(MID_DIR, f"mid_L{L}.mask.bin"))
        st = {"rows": len(rows), "tokens": int(mask.sum()), "fill": float(mask.mean()), "hq": n_hq,
              "needle": n_needle, "replay": n_replay, "docs": len(docs)}
        stats["phases"][L] = st
        p(f"phase L={L}: {st}")
    stats["hq_pool_tokens"] = hq_tokens
    stats["hq_epochs"] = hq_i / max(len(hq), 1)
    save_json(os.path.join(out, "mid_data.json"), stats)


if __name__ == "__main__":
    main()
