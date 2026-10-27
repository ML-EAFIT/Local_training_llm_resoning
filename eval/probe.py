"""Per-component analysis of a checkpoint ("evaluate each part of the model").

- parameter / FLOP breakdown (embedding, attention, FFN, norms)
- layer ablation: delta val loss when a block's attention or FFN is skipped
- head ablation: delta val loss when one attention head is zeroed
- attention entropy, mean attention distance, attention vs. relative distance (RoPE locality)
- sample attention maps
- logit lens: next-token accuracy / loss decoded from every layer's residual stream
- weight norms, activation RMS and outliers per layer
- embedding nearest neighbours and analogies
Usage: python eval/probe.py --ckpt path --tag name [--light]
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from common.config import TOK_DIR, EVAL_DIR, SMOKE  # noqa: E402
from common.data import Tok, sequential_batches  # noqa: E402
from common.logger import save_json, p  # noqa: E402
from common.model import load_model  # noqa: E402
from common.trainer import component_of  # noqa: E402

SEQ = 256 if SMOKE else 512
TOKENS = 16384 if SMOKE else 65536


def val_batches(tokens=TOKENS, domain="web"):
    path = os.path.join(TOK_DIR, f"val_{domain}.bin")
    return list(sequential_batches(path, SEQ, 8, tokens))


@torch.no_grad()
def val_loss(model, batches):
    tot = 0.0
    for x, y in batches:
        with torch.autocast("cuda", dtype=torch.float16):
            _, l = model(x, y)
        tot += l.item()
    return tot / len(batches)


def param_breakdown(model):
    out = {}
    seen = set()
    for n, prm in model.named_parameters():
        if id(prm) in seen:
            continue
        seen.add(id(prm))
        c = component_of(n)
        out[c] = out.get(c, 0) + prm.numel()
    cfg = model.cfg
    hd = cfg.d_model // cfg.n_head
    fl = {  # forward FLOPs per token (2 x matmul params) at sequence length SEQ
        "attention_proj": 2 * cfg.n_layer * (2 * cfg.d_model * cfg.n_head * hd + 2 * cfg.d_model * cfg.n_kv_head * hd),
        "attention_scores": 2 * 2 * cfg.n_layer * SEQ * cfg.d_model / 2,
        "ffn": 2 * cfg.n_layer * 3 * cfg.d_model * cfg.d_ff,
        "lm_head": 2 * cfg.d_model * cfg.vocab_size,
    }
    return {"params": out, "fwd_flops_per_token": fl}


@torch.no_grad()
def layer_ablation(model, batches, base):
    res = []
    for i, blk in enumerate(model.blocks):
        blk.ablate_attn = True
        la = val_loss(model, batches)
        blk.ablate_attn = False
        blk.ablate_ffn = True
        lf = val_loss(model, batches)
        blk.ablate_ffn = False
        res.append({"layer": i, "attn_delta": la - base, "ffn_delta": lf - base})
    return res


@torch.no_grad()
def head_ablation(model, batches, base):
    H = model.cfg.n_head
    mat = np.zeros((model.cfg.n_layer, H))
    for i, blk in enumerate(model.blocks):
        for h in range(H):
            m = torch.ones(H, device="cuda")
            m[h] = 0
            blk.attn.head_mask = m
            mat[i, h] = val_loss(model, batches) - base
        blk.attn.head_mask = None
    return mat.tolist()


@torch.no_grad()
def attention_stats(model, x, tok, sample_text):
    _ = model(x, return_attn=True)
    ent, dist_mean, by_dist = [], [], []
    T = x.shape[1]
    rel = (torch.arange(T)[:, None] - torch.arange(T)[None, :]).clamp(min=0).float().cuda()
    buckets = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    for blk in model.blocks:
        a = blk.attn.last_attn.float()                      # B, H, T, T
        e = -(a * a.clamp(min=1e-12).log()).sum(-1)          # entropy per query
        ent.append(e.mean(dim=(0, 2)).tolist())
        dist_mean.append((a * rel).sum(-1).mean(dim=(0, 2)).tolist())
        row = []
        for lo, hi in zip(buckets[:-1], buckets[1:]):
            m = (rel >= lo) & (rel < hi)
            row.append(float((a * m).sum(-1).mean()))
        by_dist.append(row)
        blk.attn.last_attn = None
    # attention maps for one short sentence
    ids = [tok.bos] + tok.encode(sample_text)
    xs = torch.tensor([ids], device="cuda")
    _ = model(xs, return_attn=True)
    L = model.cfg.n_layer
    maps = {}
    for li in sorted({0, L // 2, L - 1}):
        a = model.blocks[li].attn.last_attn[0].float().cpu()
        maps[str(li)] = [a[h].tolist() for h in range(min(4, a.shape[0]))]
    for blk in model.blocks:
        blk.attn.last_attn = None
    return {"entropy": ent, "mean_distance": dist_mean, "mass_by_distance": by_dist,
            "distance_buckets": buckets[:-1], "map_tokens": [tok.decode([i]) for i in ids], "maps": maps}


@torch.no_grad()
def logit_lens(model, batches):
    L = model.cfg.n_layer
    acc, loss = np.zeros(L + 1), np.zeros(L + 1)
    rms, outlier = np.zeros(L + 1), np.zeros(L + 1)
    n = 0
    for x, y in batches[:4]:
        with torch.autocast("cuda", dtype=torch.float16):
            _, _, hidden = model(x, return_hidden=True)
            for i, h in enumerate(hidden):
                lg = model.lens_logits(h).float()
                loss[i] += F.cross_entropy(lg.view(-1, lg.size(-1)), y.reshape(-1)).item()
                acc[i] += (lg.argmax(-1) == y).float().mean().item()
                hf = h.float()
                r = hf.pow(2).mean(-1).sqrt()
                rms[i] += r.mean().item()
                outlier[i] += (hf.abs().amax(-1) / r.clamp(min=1e-6)).mean().item()
        n += 1
    return {"acc": (acc / n).tolist(), "loss": (loss / n).tolist(), "resid_rms": (rms / n).tolist(),
            "max_over_rms": (outlier / n).tolist()}


@torch.no_grad()
def weight_norms(model):
    out = {}
    for i, blk in enumerate(model.blocks):
        row = {}
        for name, prm in blk.named_parameters():
            if prm.dim() == 2:
                key = name.split(".")[-2]
                row[key] = float(prm.float().norm() / math.sqrt(prm.numel()))
        out[i] = row
    return out


@torch.no_grad()
def embedding_probe(model, tok):
    E = F.normalize(model.tok_emb.weight.float(), dim=-1)

    def tid(w):
        ids = tok.encode(w)
        return ids[0] if len(ids) == 1 else None

    def neighbours(v, k=8, exclude=()):
        sims = E @ F.normalize(v, dim=-1)
        for e in exclude:
            sims[e] = -1
        top = sims.topk(k)
        return [[tok.decode([int(i)]), round(float(s), 3)] for s, i in zip(top.values, top.indices)]

    words = [" king", " Paris", " dog", " three", " happy", " python", " water", " January"]
    nn_ = {w: neighbours(E[tid(w)], exclude=[tid(w)]) for w in words if tid(w) is not None}
    analog = {}
    for a, b, c, d in [(" king", " man", " woman", " queen"), (" Paris", " France", " Germany", " Berlin"),
                       (" walk", " walked", " run", " ran"), (" big", " bigger", " small", " smaller")]:
        ids = [tid(w) for w in (a, b, c, d)]
        if None in ids:
            continue
        v = E[ids[0]] - E[ids[1]] + E[ids[2]]
        top = neighbours(v, 5, exclude=ids[:3])
        analog[f"{a.strip()}-{b.strip()}+{c.strip()}"] = {"expected": d.strip(), "top": top,
                                                           "hit@5": any(t[0] == d for t in top)}
    return {"neighbours": nn_, "analogies": analog}


def probe(ckpt, tag, light=False):
    model, _ = load_model(ckpt)
    model.eval()
    tok = Tok()
    batches = val_batches()
    base = val_loss(model, batches)
    res = {"tag": tag, "ckpt": ckpt, "val_loss_web": base, "breakdown": param_breakdown(model),
           "layer_ablation": layer_ablation(model, batches, base), "logit_lens": logit_lens(model, batches)}
    if not light:
        small = batches[: max(1, len(batches) // 4)]
        base_s = val_loss(model, small)
        res["head_ablation"] = head_ablation(model, small, base_s)
        res["attention"] = attention_stats(model, batches[0][0][:2], tok,
                                           "The cat sat on the mat because it was tired.")
        res["weight_norms"] = weight_norms(model)
        res["embeddings"] = embedding_probe(model, tok)
    os.makedirs(EVAL_DIR, exist_ok=True)
    save_json(os.path.join(EVAL_DIR, f"probe_{tag}.json"), res)
    p(f"probe {tag}: loss {base:.3f}; lens acc last {res['logit_lens']['acc'][-1]:.3f}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--light", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    probe(a.ckpt, a.tag, a.light)
