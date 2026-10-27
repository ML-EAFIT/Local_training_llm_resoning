"""Decoder-only Transformer with the "modern recipe" from Lecture 2:

pre-norm + RMSNorm, RoPE (+ YaRN scaling for long context), GQA, SwiGLU FFN,
optional sliding-window layers and optional sparse MoE FFN with a shared expert
and load-balancing auxiliary loss. Ablation switches (learned/no positions,
post-norm LayerNorm, MHA/MQA) exist so the scaling sweep can measure each idea.

Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V   (Lecture 1)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from common.config import ModelConfig


# ---------------------------------------------------------------- norms
class RMSNorm(nn.Module):
    """Rescale by root-mean-square, one learned gain, no recentring (Lecture 2)."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


def make_norm(cfg, d):
    return RMSNorm(d) if cfg.norm == "rmsnorm" else nn.LayerNorm(d)


# ---------------------------------------------------------------- RoPE / YaRN
def rope_frequencies(cfg: ModelConfig, head_dim, device=None):
    """Inverse frequencies and attention temperature.

    With rope_scale > 1 applies YaRN "NTK-by-parts": high-frequency dims (many
    rotations within the original context) are kept, low-frequency dims are
    interpolated by 1/scale, dims in between are blended (Lecture 3).
    """
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    mscale = 1.0
    s = cfg.rope_scale
    if s > 1.0:
        alpha, beta = 1.0, 32.0
        wavelen = 2 * math.pi / inv
        r = cfg.rope_orig_len / wavelen                      # rotations inside original context
        gamma = ((r - alpha) / (beta - alpha)).clamp(0, 1)  # 0 -> interpolate, 1 -> keep
        inv = inv / s * (1 - gamma) + inv * gamma
        mscale = 0.1 * math.log(s) + 1.0
    return inv, mscale


def rope_cos_sin(cfg, head_dim, seq_len, device):
    inv, mscale = rope_frequencies(cfg, head_dim, device)
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)                              # (T, hd/2)
    return (freqs.cos() * mscale), (freqs.sin() * mscale)


def apply_rope(x, cos, sin):
    """Rotate consecutive 2-D pairs of x (B, H, T, hd) by position-dependent angles."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


# ---------------------------------------------------------------- KV cache
class KVCache:
    """Pre-allocated per-layer K/V buffers. Only K and V are cached, never Q (Lecture 2)."""

    def __init__(self, cfg, batch, max_len, device, dtype):
        hd = cfg.d_model // cfg.n_head
        shape = (batch, cfg.n_kv_head, max_len, hd)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]
        self.pos = 0
        self.max_len = max_len

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.k + self.v)

    def reorder(self, idx):
        self.k = [t.index_select(0, idx) for t in self.k]
        self.v = [t.index_select(0, idx) for t in self.v]

    def crop(self, n):
        self.pos = n


def build_mask(q_start, T_q, T_k, window, device):
    """Boolean mask (True = attend): causal + optional sliding window."""
    qp = torch.arange(q_start, q_start + T_q, device=device)[:, None]
    kp = torch.arange(T_k, device=device)[None, :]
    m = kp <= qp
    if window > 0:
        m = m & (qp - kp < window)
    return m


# ---------------------------------------------------------------- attention
class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx):
        super().__init__()
        self.cfg = cfg
        self.n_head, self.n_kv = cfg.n_head, cfg.n_kv_head
        self.hd = cfg.d_model // cfg.n_head
        self.wq = nn.Linear(cfg.d_model, cfg.n_head * self.hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_head * self.hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_head * self.hd, bias=False)
        self.wo = nn.Linear(cfg.n_head * self.hd, cfg.d_model, bias=False)
        is_global = cfg.window == 0 or (layer_idx % cfg.local_every == cfg.local_every - 1)
        self.window = 0 if is_global else cfg.window
        self.head_mask = None     # (n_head,) multiplicative mask for head ablation
        self.last_attn = None     # filled when return_attn=True

    def forward(self, x, cos, sin, cache=None, layer_idx=0, return_attn=False):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        start = cache.pos if cache is not None else 0
        if self.cfg.pos_emb == "rope":
            q = apply_rope(q, cos[start:start + T], sin[start:start + T])
            k = apply_rope(k, cos[start:start + T], sin[start:start + T])
        if cache is not None:
            cache.k[layer_idx][:, :, start:start + T] = k
            cache.v[layer_idx][:, :, start:start + T] = v
            k = cache.k[layer_idx][:, :, :start + T]
            v = cache.v[layer_idx][:, :, :start + T]
        T_k = k.shape[2]
        rep = self.n_head // self.n_kv
        if rep > 1:  # GQA: groups of query heads share one K/V head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        if return_attn:
            att = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(self.hd)
            att = att.masked_fill(~build_mask(start, T, T_k, self.window, x.device), float("-inf"))
            att = att.softmax(-1)
            self.last_attn = att.detach()
            y = (att.to(v.dtype) @ v)
        else:
            if cache is None and self.window == 0:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            elif T == 1 and self.window == 0:
                y = F.scaled_dot_product_attention(q, k, v)
            else:
                mask = build_mask(start, T, T_k, self.window, x.device)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        if self.head_mask is not None:
            y = y * self.head_mask.view(1, -1, 1, 1).to(y.dtype)
        y = y.transpose(1, 2).reshape(B, T, self.n_head * self.hd)
        return self.wo(y)


# ---------------------------------------------------------------- FFN / MoE
class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w3 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """Sparse top-k MoE replacing the FFN: y = sum_{i in topk} G(x)_i E_i(x)  (+ shared expert).

    Load-balancing loss  N * sum_i f_i * P_i  with f_i (token fraction, no grad) and
    P_i (mean router prob, with grad) prevents routing collapse (Lecture 2).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n, self.k = cfg.n_experts, cfg.moe_top_k
        d_exp = max(64, cfg.d_ff // self.k)  # keep active params ~ dense FFN
        self.router = nn.Linear(cfg.d_model, self.n, bias=False)
        self.experts = nn.ModuleList([SwiGLU(cfg.d_model, d_exp) for _ in range(self.n)])
        self.shared = SwiGLU(cfg.d_model, d_exp) if cfg.moe_shared else None
        self.aux_loss = torch.tensor(0.0)
        self.counts = torch.zeros(self.n)

    def forward(self, x):
        B, T, D = x.shape
        xf = x.reshape(-1, D)
        logits = self.router(xf).float()
        probs = logits.softmax(-1)
        topv, topi = probs.topk(self.k, dim=-1)
        gates = (topv / topv.sum(-1, keepdim=True)).to(x.dtype)
        out = torch.zeros_like(xf)
        for e in range(self.n):
            sel = (topi == e)
            rows = sel.any(-1).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            g = (gates * sel.to(gates.dtype)).sum(-1)[rows].unsqueeze(-1)
            out.index_add_(0, rows, self.experts[e](xf[rows]) * g)
        if self.shared is not None:
            out = out + self.shared(xf)
        f = F.one_hot(topi, self.n).sum(1).float().mean(0) / self.k   # token fraction (const)
        P = probs.mean(0)
        self.aux_loss = self.n * (f.detach() * P).sum()
        self.counts = f.detach().cpu()
        return out.view(B, T, D)


# ---------------------------------------------------------------- block / model
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, i):
        super().__init__()
        self.cfg, self.i = cfg, i
        self.n1 = make_norm(cfg, cfg.d_model)
        self.attn = Attention(cfg, i)
        self.n2 = make_norm(cfg, cfg.d_model)
        self.ffn = MoE(cfg) if cfg.n_experts > 0 else SwiGLU(cfg.d_model, cfg.d_ff)
        self.ablate_attn = False
        self.ablate_ffn = False

    def forward(self, x, cos, sin, cache=None, return_attn=False):
        a = lambda h: self.attn(h, cos, sin, cache, self.i, return_attn)
        if self.cfg.prenorm:  # x + Sublayer(LN(x)) keeps a clean residual path
            if not self.ablate_attn:
                x = x + a(self.n1(x))
            if not self.ablate_ffn:
                x = x + self.ffn(self.n2(x))
        else:                 # original post-norm: LN(x + Sublayer(x))
            x = self.n1(x + (0 if self.ablate_attn else a(x)))
            x = self.n2(x + (0 if self.ablate_ffn else self.ffn(x)))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model) if cfg.pos_emb == "learned" else None
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.norm_f = make_norm(cfg, cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight   # tied embeddings
        self._rope_len = 0
        self.apply(self._init)
        for n, p in self.named_parameters():        # GPT-2 style scaled residual init
            if n.endswith("wo.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def set_rope(self, scale=None, orig_len=None, max_seq_len=None):
        if scale is not None:
            self.cfg.rope_scale = scale
        if orig_len is not None:
            self.cfg.rope_orig_len = orig_len
        if max_seq_len is not None:
            self.cfg.max_seq_len = max_seq_len
        self._rope_len = 0

    def _rope(self, T, device):
        if self._rope_len < T or self._cos.device != device:
            n = max(T, self.cfg.max_seq_len)
            cos, sin = rope_cos_sin(self.cfg, self.cfg.d_model // self.cfg.n_head, n, device)
            self._cos, self._sin, self._rope_len = cos, sin, n
        return self._cos, self._sin

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if self.pos_emb is not None:
                n -= self.pos_emb.weight.numel()
        return n

    def aux_loss(self):
        if self.cfg.n_experts == 0:
            return None
        return sum(b.ffn.aux_loss for b in self.blocks) / self.cfg.n_layer

    def forward(self, idx, targets=None, loss_mask=None, cache=None,
                return_hidden=False, return_attn=False, reduction="mean"):
        """Returns (logits, loss). loss = -(1/T) sum_t log p(x_t | x_<t) over masked tokens."""
        B, T = idx.shape
        start = cache.pos if cache is not None else 0
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(start, start + T, device=idx.device).clamp(max=self.cfg.max_seq_len - 1)
            x = x + self.pos_emb(pos)
        cos, sin = self._rope(start + T, idx.device)
        hidden = [x] if return_hidden else None
        for blk in self.blocks:
            if self.cfg.grad_ckpt and self.training:
                x = checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                x = blk(x, cos, sin, cache, return_attn)
            if return_hidden:
                hidden.append(x)
        if cache is not None:
            cache.pos += T
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1),
                                 reduction="none", ignore_index=-100).view(B, T)
            if loss_mask is None:
                loss_mask = (targets != -100).float()
            if reduction == "none":
                loss = ce * loss_mask
            else:
                loss = (ce * loss_mask).sum() / loss_mask.sum().clamp(min=1)
                aux = self.aux_loss()
                if aux is not None and self.training:
                    loss = loss + self.cfg.moe_aux_coef * aux
        if return_hidden:
            return logits, loss, hidden
        return logits, loss

    def features(self, idx):
        """Final normalised hidden states (B, T, d), without the LM head (reward-model backbone)."""
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(idx.shape[1], device=idx.device).clamp(max=self.cfg.max_seq_len - 1))
        cos, sin = self._rope(idx.shape[1], idx.device)
        for blk in self.blocks:
            x = checkpoint(blk, x, cos, sin, use_reentrant=False) if (self.cfg.grad_ckpt and self.training) \
                else blk(x, cos, sin)
        return self.norm_f(x)

    def lens_logits(self, h):
        """Logit lens: decode an intermediate residual state with the final norm + head."""
        return self.lm_head(self.norm_f(h))

    def new_cache(self, batch, max_len, device=None, dtype=None):
        p = next(self.parameters())
        return KVCache(self.cfg, batch, max_len, device or p.device, dtype or p.dtype)


def token_logprobs(model, idx, targets):
    """Per-token log p(target | prefix), shape (B, T). Used by DPO, GRPO, distillation."""
    logits, _ = model(idx)
    return torch.gather(F.log_softmax(logits.float(), -1), -1, targets.unsqueeze(-1)).squeeze(-1)


def load_model(path, device="cuda", dtype=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["model_cfg"])
    m = GPT(cfg)
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ck["model"].items()}
    m.load_state_dict(sd, strict=False)
    m.to(device)
    if dtype is not None:
        m.to(dtype)
    return m, ck
