import os, sys, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "2_data_clean"))
import numpy as np
import minhash


def test_minhash_estimate_close():
    random.seed(0)
    words = [f"w{random.randint(0, 5000)}" for _ in range(600)]
    a = " ".join(words)
    b = " ".join(words[:450] + [f"x{i}" for i in range(150)])
    est = (minhash.signature(a) == minhash.signature(b)).mean()
    assert abs(est - minhash.jaccard_exact(a, b)) < 0.1


def test_near_dup_detection():
    random.seed(1)
    base = " ".join(f"w{random.randint(0, 9999)}" for _ in range(500))
    docs = [base, base.replace("w1", "zz", 1), " ".join(f"q{random.randint(0, 9999)}" for _ in range(500))]
    dup = minhash.find_near_dups(np.stack([minhash.signature(d) for d in docs]))
    assert dup.tolist() == [False, True, False]
