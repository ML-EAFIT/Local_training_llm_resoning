"""Stage 3c — evaluate the tokenizer.

Compression (bytes/token) and fertility (tokens/word) per domain vs. GPT-2 and
cl100k, round-trip exactness, OOV (always 0 for byte-level BPE), vocabulary
utilisation, longest tokens, and examples. Lecture 1: vocab size trades off
against sequence length.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
from common.config import TOK_DIR, CLEAN_DIR, stage_dir  # noqa: E402
from common.data import Tok  # noqa: E402
from common.logger import save_json, load_json, p  # noqa: E402

EXAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "Photosynthesis converts light energy into chemical energy.",
    "What is 1234 + 5678? Let's think step by step.",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
    "Unbelievably, the antidisestablishmentarianism debate continued.",
    "El rápido zorro marrón salta sobre el perro perezoso.",
]


def val_docs(dom, n=300):
    import glob
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(CLEAN_DIR, "pre", dom, "*.parquet")))
    out = []
    for f in files:
        out += pq.read_table(f, columns=["text"]).column("text").to_pylist()
        if len(out) >= n:
            break
    return out[:n]


def main():
    out = stage_dir("3_tokenization")
    t0 = time.time()
    tok = Tok()
    import tiktoken
    refs = {"gpt2": tiktoken.get_encoding("gpt2"), "cl100k": tiktoken.get_encoding("cl100k_base")}
    tk_stats = load_json(os.path.join(out, "tokenize.json"), {})
    res = {"vocab_size": tok.vocab_size, "domains": {}, "ref_vocab": {k: v.n_vocab for k, v in refs.items()}}
    roundtrip_ok = roundtrip_n = 0
    for dom in tk_stats.get("pre", {}):
        docs = val_docs(dom)
        if not docs:
            continue
        nbytes = sum(len(d.encode("utf-8", "ignore")) for d in docs)
        nwords = sum(len(d.split()) for d in docs)
        ours = tok.encode_batch(docs)
        nt = sum(len(x) for x in ours)
        for d, ids in zip(docs[:200], ours[:200]):
            roundtrip_n += 1
            roundtrip_ok += tok.decode(ids) == d
        r = {"bytes_per_token": nbytes / nt, "tokens_per_word": nt / max(nwords, 1), "tokens": nt}
        for name, enc in refs.items():
            n_ref = sum(len(enc.encode_ordinary(d)) for d in docs)
            r[f"{name}_bytes_per_token"] = nbytes / n_ref
            r[f"{name}_tokens_per_word"] = n_ref / max(nwords, 1)
        res["domains"][dom] = r
    res["roundtrip_exact"] = roundtrip_ok / max(roundtrip_n, 1)
    res["oov_rate"] = 0.0  # byte-level alphabet: every string is representable

    # vocab utilisation on a slice of the training stream
    data = np.memmap(os.path.join(TOK_DIR, "train.bin"), dtype=np.uint16, mode="r")
    sl = np.asarray(data[: min(len(data), 20_000_000)])
    counts = np.bincount(sl, minlength=tok.vocab_size)
    res["vocab_used_frac"] = float((counts > 0).mean())
    srt = np.sort(counts)[::-1]
    res["top10_token_share"] = float(srt[:10].sum() / max(srt.sum(), 1))
    res["rank_freq"] = [int(srt[i]) for i in np.unique(np.logspace(0, np.log10(len(srt)), 40).astype(int) - 1)]
    res["most_common"] = [[tok.token_str(i), int(counts[i])] for i in np.argsort(-counts)[:25]]
    vocab = tok.tk.get_vocab()
    longest = sorted(vocab, key=lambda s: -len(s))[:20]
    res["longest_tokens"] = [tok.decode([vocab[s]]) for s in longest]
    res["examples"] = []
    for s in EXAMPLES:
        ids = tok.encode(s)
        res["examples"].append({"text": s, "n_ours": len(ids), "pieces": [tok.decode([i]) for i in ids],
                                "n_gpt2": len(refs["gpt2"].encode_ordinary(s)),
                                "n_cl100k": len(refs["cl100k"].encode_ordinary(s))})
    res["seconds"] = round(time.time() - t0, 1)
    tk_stats.pop("energy", None)
    res["tokenize"] = tk_stats
    res["energy"] = load_json(os.path.join(out, "tokenize.json"), {}).get("energy", {})
    save_json(os.path.join(out, "summary.json"), res)
    p("tokenizer eval:", {d: round(v["bytes_per_token"], 2) for d, v in res["domains"].items()},
      "roundtrip", res["roundtrip_exact"], "vocab used", round(res["vocab_used_frac"], 3))


if __name__ == "__main__":
    main()
