"""Stage 3b — tokenize the cleaned corpus into uint16 memmaps.

pre-train:  train.bin (all domains, docs separated by <|eos|>) + val_<domain>.bin
mid-train:  mid_<domain>.bin + mid_<domain>.idx (doc offsets, needed for best-fit packing)
Also records bytes and tokens per domain -> bytes/token for bits-per-byte (Lecture 3).
"""
import glob
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import CLEAN_DIR, TOK_DIR, MAX_WORKERS, SMOKE, stage_dir  # noqa: E402

os.environ.setdefault("RAYON_NUM_THREADS", str(MAX_WORKERS))
import numpy as np  # noqa: E402
from common.data import Tok  # noqa: E402
from common.logger import save_json, p  # noqa: E402
from common.energy import EnergyMeter  # noqa: E402

VAL_TOKENS = 30_000 if SMOKE else 2_000_000   # held-out tokens per domain
BATCH = 2000


def docs_of(files):
    import pyarrow.parquet as pq
    for f in files:
        for t in pq.read_table(f, columns=["text"]).column("text").to_pylist():
            yield t


def batched(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def main():
    out = stage_dir("3_tokenization")
    meter = EnergyMeter(out, name="energy_tokenize").start()
    tok = Tok()
    assert tok.vocab_size <= 65535
    t0 = time.time()
    stats = {"pre": {}, "mid": {}}
    rng = random.Random(0)

    # ---- pre-training
    train_f = open(os.path.join(TOK_DIR, "train.bin.tmp"), "wb")
    for dom_dir in sorted(glob.glob(os.path.join(CLEAN_DIR, "pre", "*"))):
        dom = os.path.basename(dom_dir)
        files = sorted(glob.glob(os.path.join(dom_dir, "*.parquet")))
        rng.shuffle(files)
        val_f = open(os.path.join(TOK_DIR, f"val_{dom}.bin"), "wb")
        s = {"docs": 0, "tokens": 0, "bytes": 0, "val_tokens": 0, "val_bytes": 0}
        for batch in batched(docs_of(files), BATCH):
            enc = tok.encode_batch(batch)
            for text, ids in zip(batch, enc):
                ids.append(tok.eos)
                arr = np.asarray(ids, dtype=np.uint16)
                nb = len(text.encode("utf-8", "ignore"))
                if s["val_tokens"] < VAL_TOKENS:
                    val_f.write(arr.tobytes()); s["val_tokens"] += len(arr); s["val_bytes"] += nb
                else:
                    train_f.write(arr.tobytes()); s["tokens"] += len(arr); s["bytes"] += nb
                s["docs"] += 1
            if s["docs"] % (BATCH * 100) == 0:
                p(f"{dom}: {s['docs']} docs, {s['tokens']/1e6:.0f}M tokens, {time.time()-t0:.0f}s")
        val_f.close()
        s["bytes_per_token"] = s["bytes"] / max(s["tokens"], 1)
        stats["pre"][dom] = s
        p(f"pre {dom}: {s}")
    train_f.close()
    os.replace(os.path.join(TOK_DIR, "train.bin.tmp"), os.path.join(TOK_DIR, "train.bin"))

    # ---- mid-training pool (keep document boundaries)
    for dom_dir in sorted(glob.glob(os.path.join(CLEAN_DIR, "mid", "*"))):
        dom = os.path.basename(dom_dir)
        files = sorted(glob.glob(os.path.join(dom_dir, "*.parquet")))
        offsets, s = [0], {"docs": 0, "tokens": 0, "bytes": 0}
        with open(os.path.join(TOK_DIR, f"mid_{dom}.bin"), "wb") as f:
            for batch in batched(docs_of(files), BATCH):
                for text, ids in zip(batch, tok.encode_batch(batch)):
                    ids.append(tok.eos)
                    f.write(np.asarray(ids, dtype=np.uint16).tobytes())
                    offsets.append(offsets[-1] + len(ids))
                    s["docs"] += 1; s["tokens"] += len(ids); s["bytes"] += len(text.encode("utf-8", "ignore"))
        np.save(os.path.join(TOK_DIR, f"mid_{dom}.idx.npy"), np.asarray(offsets, dtype=np.int64))
        s["bytes_per_token"] = s["bytes"] / max(s["tokens"], 1)
        stats["mid"][dom] = s
        p(f"mid {dom}: {s}")

    tot_t = sum(v["tokens"] for v in stats["pre"].values())
    tot_b = sum(v["bytes"] for v in stats["pre"].values())
    stats["train_tokens"] = tot_t
    stats["bytes_per_token"] = tot_b / max(tot_t, 1)
    stats["domain_token_share"] = {k: v["tokens"] / max(tot_t, 1) for k, v in stats["pre"].items()}
    stats["seconds"] = round(time.time() - t0, 1)
    meter.add_tokens(tot_t)
    stats["energy"] = meter.stop()
    save_json(os.path.join(out, "tokenize.json"), stats)
    p(f"Stage 3b done: {tot_t/1e6:.1f}M train tokens, {stats['bytes_per_token']:.2f} bytes/token")


if __name__ == "__main__":
    main()
