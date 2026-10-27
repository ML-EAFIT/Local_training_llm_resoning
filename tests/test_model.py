import math
import torch
from common.config import ModelConfig
from common.model import GPT, apply_rope, rope_cos_sin, MoE

dev = "cuda" if torch.cuda.is_available() else "cpu"


def small(**kw):
    c = dict(vocab_size=97, n_layer=2, n_head=4, n_kv_head=2, d_model=64, d_ff=128, max_seq_len=64)
    c.update(kw)
    return ModelConfig(**c)


def test_causal_no_leak():
    torch.manual_seed(0)
    m = GPT(small()).to(dev).eval()
    x = torch.randint(0, 97, (1, 20), device=dev)
    y = x.clone(); y[0, 15:] = (y[0, 15:] + 1) % 97
    a, _ = m(x); b, _ = m(y)
    assert torch.allclose(a[0, :15], b[0, :15], atol=1e-5)
    assert not torch.allclose(a[0, 15:], b[0, 15:])


def test_rope_relative():
    cfg = small()
    cos, sin = rope_cos_sin(cfg, 16, 64, "cpu")
    q = torch.randn(1, 1, 1, 16); k = torch.randn(1, 1, 1, 16)
    def score(m, n):
        qm = apply_rope(q, cos[m:m+1], sin[m:m+1]); kn = apply_rope(k, cos[n:n+1], sin[n:n+1])
        return (qm * kn).sum().item()
    assert abs(score(10, 3) - score(30, 23)) < 1e-4
    assert abs(score(5, 5) - (q * k).sum().item()) < 1e-4


def test_yarn_keeps_high_freq():
    from common.model import rope_frequencies
    base, _ = rope_frequencies(small(), 16)
    sc, ms = rope_frequencies(small(rope_scale=4.0, rope_orig_len=1024), 16)
    assert torch.allclose(base[0], sc[0]) and sc[-1] < base[-1] and ms > 1


def test_gqa_mha_mqa_shapes():
    x = torch.randint(0, 97, (2, 12), device=dev)
    for kv in (4, 2, 1):
        m = GPT(small(n_kv_head=kv)).to(dev)
        lg, loss = m(x, x)
        assert lg.shape == (2, 12, 97) and math.isfinite(loss.item())


def test_kv_cache_matches_full():
    torch.manual_seed(0)
    for window in (0, 4):
        m = GPT(small(window=window)).to(dev).eval()
        x = torch.randint(0, 97, (2, 16), device=dev)
        full, _ = m(x)
        cache = m.new_cache(2, 32)
        out = [m(x[:, :5], cache=cache)[0]]
        for t in range(5, 16):
            out.append(m(x[:, t:t+1], cache=cache)[0])
        inc = torch.cat(out, 1)
        assert torch.allclose(full, inc, atol=1e-4), (window, (full - inc).abs().max())


def test_window_and_attn_probs():
    m = GPT(small(window=4)).to(dev).eval()
    x = torch.randint(0, 97, (1, 12), device=dev)
    a, _ = m(x); b, _ = m(x, return_attn=True)
    assert torch.allclose(a, b, atol=1e-4)
    att = m.blocks[0].attn.last_attn[0, 0]
    assert att[10, :7].abs().max() < 1e-6           # local layer: window 4
    assert torch.allclose(att.sum(-1), torch.ones(12, device=dev), atol=1e-5)


def test_moe_aux_and_ablations():
    m = GPT(small(n_experts=4, moe_top_k=2)).to(dev).train()
    x = torch.randint(0, 97, (2, 16), device=dev)
    _, loss = m(x, x); loss.backward()
    assert m.aux_loss().item() > 0 and m.blocks[0].ffn.router.weight.grad is not None
    for cfg in (small(pos_emb="learned"), small(pos_emb="none"), small(norm="layernorm", prenorm=False)):
        _, l = GPT(cfg).to(dev)(x, x); assert math.isfinite(l.item())
