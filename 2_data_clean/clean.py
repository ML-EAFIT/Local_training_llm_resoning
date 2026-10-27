"""Stage 2 — Data cleaning.

pass 1 (MAX_WORKERS): boilerplate strip -> quality filters -> PII scrub ->
          13-gram decontamination vs eval sets -> exact hash + MinHash signature
pass 2 (main):     global exact dedup + MinHash-LSH near dedup (Jaccard >= 0.8)
pass 3 (workers):  write final cleaned shards
Statistics for every rule, per source, go to summary.json for the report.
"""
import glob
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.config import RAW_DIR, CLEAN_DIR, MAX_WORKERS, stage_dir  # noqa: E402
from common.logger import save_json, p  # noqa: E402
from common.energy import EnergyMeter  # noqa: E402
import filters  # noqa: E402
import minhash  # noqa: E402

TMP = os.path.join(CLEAN_DIR, "_tmp")
CONTAM_N = 13
LEN_BINS = [0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 1e12]


def build_eval_ngrams():
    """13-gram hashes of every evaluation set (decontamination, Lecture 3 data hygiene)."""
    aux = os.path.join(RAW_DIR, "aux")
    texts = []
    for path in glob.glob(os.path.join(aux, "eval_*.jsonl")):
        if "gsm8k_train" in path:
            continue
        for line in open(path):
            r = json.loads(line)
            for k in ("ctx", "question", "goal", "text", "prompt"):
                if isinstance(r.get(k), str):
                    texts.append(r[k])
            if isinstance(r.get("endings"), (list, str)):
                texts.append(" ".join(r["endings"]) if isinstance(r["endings"], list) else r["endings"])
            if isinstance(r.get("answer"), str):
                texts.append(r["answer"])
    hs = [minhash.ngram_hashes(minhash.word_hashes(t), CONTAM_N) for t in texts]
    arr = np.unique(np.concatenate(hs)) if hs else np.zeros(0, np.uint64)
    np.save(os.path.join(TMP, "eval_ngrams.npy"), arr)
    return len(arr)


_EVAL = None


def _eval_ngrams():
    global _EVAL
    if _EVAL is None:
        _EVAL = np.load(os.path.join(TMP, "eval_ngrams.npy"))
    return _EVAL


def pass1(args):
    path, source, kind, out_i = args
    import pyarrow.parquet as pq
    tbl = pq.read_table(path).to_pydict()
    rej = defaultdict(lambda: [0, 0])
    examples = defaultdict(list)
    len_before, len_after = np.zeros(len(LEN_BINS) - 1, int), np.zeros(len(LEN_BINS) - 1, int)
    kept_text, kept_meta, sigs, exact = [], [], [], []
    pii = {"email": 0, "ip": 0, "boiler_lines": 0}
    evn = _eval_ngrams()
    is_code = source == "code"
    for text, meta_s in zip(tbl["text"], tbl["meta"]):
        meta = json.loads(meta_s)
        nb = len(text.encode("utf-8", "ignore"))
        len_before[np.searchsorted(LEN_BINS, len(text), "right") - 1] += 1
        if len(text) > filters.MAX_CHARS:
            text = text[:filters.MAX_CHARS]
        if not is_code:
            text, nboil = filters.strip_boilerplate(text)
            pii["boiler_lines"] += nboil
            reason = filters.text_filter(text, source)
        else:
            reason = filters.code_filter(text, meta)
        if reason is None:
            text, ne, ni = filters.scrub_pii(text, ip=not is_code)
            pii["email"] += ne
            pii["ip"] += ni
            g = minhash.ngram_hashes(minhash.word_hashes(text), CONTAM_N)
            contaminated = False
            if len(evn) and len(g):
                pos = np.searchsorted(evn, g).clip(max=len(evn) - 1)   # evn is sorted + unique
                contaminated = bool((evn[pos] == g).any())
            if contaminated:
                reason = "eval_contamination"
        if reason is not None:
            rej[reason][0] += 1
            rej[reason][1] += nb
            if len(examples[reason]) < 3:
                examples[reason].append(text[:300])
            continue
        len_after[np.searchsorted(LEN_BINS, len(text), "right") - 1] += 1
        norm = " ".join(text.lower().split())
        exact.append(hashlib.md5(norm.encode()).hexdigest()[:16])
        sigs.append(minhash.signature(text))
        kept_text.append(text)
        kept_meta.append(json.dumps(meta))
    import pyarrow as pa
    stem = os.path.join(TMP, f"{source}_{kind}_{out_i:05d}")
    pq.write_table(pa.table({"text": kept_text, "meta": kept_meta}), stem + ".parquet")
    np.save(stem + ".sig.npy", np.stack(sigs) if sigs else np.zeros((0, minhash.NPERM), np.uint32))
    with open(stem + ".exact.json", "w") as f:
        json.dump(exact, f)
    return {"source": source, "kind": kind, "stem": stem, "docs_in": len(tbl["text"]),
            "bytes_in": sum(len(t.encode("utf-8", "ignore")) for t in tbl["text"]),
            "rej": dict(rej), "examples": dict(examples), "pii": pii,
            "len_before": len_before.tolist(), "len_after": len_after.tolist(), "kept": len(kept_text)}


def pass3(args):
    stem, drop_idx, source, kind = args
    import pyarrow.parquet as pq
    import pyarrow as pa
    tbl = pq.read_table(stem + ".parquet").to_pydict()
    drop = set(drop_idx)
    text = [t for i, t in enumerate(tbl["text"]) if i not in drop]
    meta = [m for i, m in enumerate(tbl["meta"]) if i not in drop]
    out = os.path.join(CLEAN_DIR, kind, source)
    os.makedirs(out, exist_ok=True)
    pq.write_table(pa.table({"text": text, "meta": meta}),
                   os.path.join(out, os.path.basename(stem) + ".parquet"), compression="zstd")
    for ext in (".parquet", ".sig.npy", ".exact.json"):
        os.remove(stem + ext)
    return len(text), sum(len(t.encode("utf-8", "ignore")) for t in text)


def main():
    out = stage_dir("2_data_clean")
    os.makedirs(TMP, exist_ok=True)
    meter = EnergyMeter(out).start()
    t0 = time.time()
    n_eval = build_eval_ngrams()
    p(f"eval 13-grams: {n_eval}")
    jobs, i = [], 0
    for src_dir in sorted(glob.glob(os.path.join(RAW_DIR, "*"))):
        source = os.path.basename(src_dir)
        if source == "aux" or not os.path.isdir(src_dir):
            continue
        for f in sorted(glob.glob(os.path.join(src_dir, "*.parquet"))):
            kind = os.path.basename(f).split("_")[0]
            jobs.append((f, source, kind, i))
            i += 1
    random.Random(0).shuffle(jobs)  # balance big/small shards across workers
    with Pool(MAX_WORKERS) as pool:
        res = []
        for r in pool.imap_unordered(pass1, jobs):
            res.append(r)
            if len(res) % 10 == 0:
                p(f"pass1 {len(res)}/{len(jobs)} shards")
    p(f"pass1 done in {time.time()-t0:.0f}s")

    # ---- pass 2: global exact + near dedup (mid and pre pools together, so mid never repeats pre)
    res.sort(key=lambda r: r["stem"])
    sigs = [np.load(r["stem"] + ".sig.npy") for r in res]
    exact = [json.load(open(r["stem"] + ".exact.json")) for r in res]
    offsets = np.cumsum([0] + [len(s) for s in sigs])
    allsig = np.concatenate(sigs) if sigs else np.zeros((0, minhash.NPERM), np.uint32)
    allex = [h for e in exact for h in e]
    seen, exact_dup = set(), np.zeros(len(allex), bool)
    for k, h in enumerate(allex):
        if h in seen:
            exact_dup[k] = True
        seen.add(h)
    near = minhash.find_near_dups(allsig) & ~exact_dup
    p(f"dedup: {exact_dup.sum()} exact, {near.sum()} near duplicates out of {len(allex)}")

    # ---- pass 3: write final
    jobs3, dedup_stats = [], defaultdict(lambda: {"exact_dup": 0, "near_dup": 0})
    for k, r in enumerate(res):
        a, b = offsets[k], offsets[k + 1]
        ed, nd = exact_dup[a:b], near[a:b]
        drop = np.nonzero(ed | nd)[0].tolist()
        dedup_stats[r["source"]]["exact_dup"] += int(ed.sum())
        dedup_stats[r["source"]]["near_dup"] += int(nd.sum())
        jobs3.append((r["stem"], drop, r["source"], r["kind"]))
    with Pool(MAX_WORKERS) as pool:
        written = pool.map(pass3, jobs3)

    # ---- statistics
    per = {}
    for r, (nd, nb) in zip(res, written):
        key = f'{r["source"]}/{r["kind"]}'
        s = per.setdefault(key, {"docs_in": 0, "bytes_in": 0, "docs_out": 0, "bytes_out": 0, "rej": {},
                                 "pii": defaultdict(int), "len_before": np.zeros(len(LEN_BINS) - 1, int),
                                 "len_after": np.zeros(len(LEN_BINS) - 1, int), "examples": {}})
        s["docs_in"] += r["docs_in"]; s["bytes_in"] += r["bytes_in"]
        s["docs_out"] += nd; s["bytes_out"] += nb
        for reason, (c, b) in r["rej"].items():
            s["rej"].setdefault(reason, [0, 0])
            s["rej"][reason][0] += c; s["rej"][reason][1] += b
        for kk, v in r["pii"].items():
            s["pii"][kk] += v
        s["len_before"] += np.array(r["len_before"]); s["len_after"] += np.array(r["len_after"])
        for reason, ex in r["examples"].items():
            s["examples"].setdefault(reason, ex)
    for key, s in per.items():
        src = key.split("/")[0]
        s["len_before"] = s["len_before"].tolist(); s["len_after"] = s["len_after"].tolist()
        s["pii"] = dict(s["pii"])
        s["kept_frac_docs"] = s["docs_out"] / max(s["docs_in"], 1)
        s["kept_frac_bytes"] = s["bytes_out"] / max(s["bytes_in"], 1)
    summary = {"per_source": per, "dedup": dict(dedup_stats), "eval_ngrams": n_eval,
               "len_bins": LEN_BINS[:-1], "seconds": round(time.time() - t0, 1)}
    tot_in = sum(s["bytes_in"] for s in per.values()); tot_out = sum(s["bytes_out"] for s in per.values())
    summary["total"] = {"bytes_in": tot_in, "bytes_out": tot_out, "kept_frac": tot_out / max(tot_in, 1)}
    summary["energy"] = meter.stop()
    save_json(os.path.join(out, "summary.json"), summary)
    import shutil
    shutil.rmtree(TMP, ignore_errors=True)
    p(f"Stage 2 done: kept {tot_out/1e9:.2f}/{tot_in/1e9:.2f} GB ({100*tot_out/max(tot_in,1):.1f}%)")


if __name__ == "__main__":
    main()
