"""Vectorised MinHash + LSH for near-duplicate detection, and rolling n-gram hashes.

Words are hashed with crc32, n-gram shingles with a polynomial rolling hash
(uint64, wrap-around arithmetic), and K permutations with a*h+b (a odd).
"""
import re
import zlib
import numpy as np

WORD = re.compile(r"\w+")
NPERM = 128
BANDS, ROWS = 16, 8            # LSH: candidate threshold ~ (1/16)^(1/8) ~= 0.71
_rng = np.random.default_rng(42)
A = (_rng.integers(1, 2**63 - 1, NPERM, dtype=np.uint64) | np.uint64(1))
B = _rng.integers(0, 2**63 - 1, NPERM, dtype=np.uint64)
POW = np.uint64(1099511628211)


def word_hashes(text, max_words=None):
    words = WORD.findall(text.lower())
    if max_words:
        words = words[:max_words]
    return np.fromiter((zlib.crc32(w.encode()) for w in words), dtype=np.uint64, count=len(words))


def ngram_hashes(wh, n):
    """Rolling polynomial hash of every n-gram of word hashes."""
    if len(wh) < n:
        return np.zeros(0, dtype=np.uint64)
    m = len(wh) - n + 1
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            h = h * POW + wh[j:j + m]
    return h


def signature(text, n=5, max_words=3000):
    sh = ngram_hashes(word_hashes(text, max_words), n)
    if len(sh) == 0:
        sh = word_hashes(text, max_words)[:1]
        if len(sh) == 0:
            return np.full(NPERM, np.iinfo(np.uint32).max, dtype=np.uint32)
    sh = np.unique(sh)
    with np.errstate(over="ignore"):
        hv = (A[:, None] * sh[None, :] + B[:, None]) >> np.uint64(32)
    return hv.min(1).astype(np.uint32)


def band_hashes(sigs):
    """(N, NPERM) uint32 -> (N, BANDS) uint64 band keys."""
    s = sigs.reshape(len(sigs), BANDS, ROWS).astype(np.uint64)
    h = np.zeros((len(sigs), BANDS), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for r in range(ROWS):
            h = h * POW + s[:, :, r]
    return h


def find_near_dups(sigs, threshold=0.8):
    """Return boolean array: True if doc i is a near-duplicate of an earlier doc."""
    N = len(sigs)
    dup = np.zeros(N, dtype=bool)
    if N == 0:
        return dup
    bh = band_hashes(sigs)
    for b in range(BANDS):
        col = bh[:, b]
        order = np.argsort(col, kind="stable")
        sc = col[order]
        starts = np.r_[True, sc[1:] != sc[:-1]]
        first = order[np.maximum.accumulate(np.where(starts, np.arange(N), 0))]  # representative per bucket
        cand = np.nonzero(first != order)[0]
        if len(cand) == 0:
            continue
        i, j = order[cand], first[cand]           # j is earliest in sort order (stable -> smaller index)
        lo, hi = np.minimum(i, j), np.maximum(i, j)
        est = (sigs[lo] == sigs[hi]).mean(1)
        dup[hi[est >= threshold]] = True
    return dup


def jaccard_exact(a, b, n=5):
    sa = set(ngram_hashes(word_hashes(a), n).tolist())
    sb = set(ngram_hashes(word_hashes(b), n).tolist())
    return len(sa & sb) / max(len(sa | sb), 1)
