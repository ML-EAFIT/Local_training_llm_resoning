import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "5_generation"))
import torch
from common.config import ModelConfig
from common.model import GPT
from engine import Engine, Pattern, sample_next


class FakeTok:
    eos, end, pad, bos = 1, 6, 2, 0
    vocab_size = 97
    def decode(self, ids, skip_special=False):
        return "".join(chr(48 + (i % 40)) for i in ids)


def model(seed):
    torch.manual_seed(seed)
    return GPT(ModelConfig(vocab_size=97, n_layer=2, n_head=4, n_kv_head=2, d_model=64, d_ff=128,
                           max_seq_len=128)).cuda().eval()


def test_cache_equals_nocache_greedy():
    e = Engine(model(0), FakeTok())
    p = [5, 9, 11, 30, 40]
    a = e.generate(p, 20, use_cache=True)[0]
    b = e.generate(p, 20, use_cache=False)[0]
    assert a == b


def test_speculative_matches_greedy():
    tgt = model(0)
    e = Engine(tgt, FakeTok())
    p = [5, 9, 11, 30, 40]
    greedy = e.generate(p, 30, stop=set())[0]
    e.stop = set()
    for draft in (model(1), tgt):
        spec, st = e.speculative(p, draft, k=4, max_new=30)
        n = min(len(spec), len(greedy))
        assert spec[:n] == greedy[:n] and n >= 25
    assert st["acceptance_rate"] == 1.0          # draft == target accepts everything


def test_top_p_top_k():
    lg = torch.tensor([[10.0, 9.0, 0.0, -5.0]]).cuda()
    for _ in range(20):
        assert sample_next(lg, 1.0, top_k=2).item() in (0, 1)
        assert sample_next(lg, 1.0, top_p=0.5).item() == 0
    assert sample_next(lg, 0).item() == 0


def test_pattern():
    pt = Pattern([("lit", '{"answer": '), ("digits", (1, 4)), ("lit", "}")])
    assert pt.check('{"ans') == "prefix" and pt.check('{"answer": 12') == "prefix"
    assert pt.check('{"answer": 12}') == "complete" and pt.check('{"answer": x') is None
    assert pt.check('{"answer": 12345') is None
