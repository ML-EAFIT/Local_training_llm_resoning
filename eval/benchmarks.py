"""Zero-shot benchmarks scored by log-likelihood (no generation needed):
HellaSwag, ARC-Easy, PIQA (multiple choice, length-normalised log-prob) and
LAMBADA (last-word exact match + perplexity). Works sharded across DDP ranks.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from common.config import RAW_DIR  # noqa: E402
from common import dist  # noqa: E402


def load_eval(name):
    path = os.path.join(RAW_DIR, "aux", f"eval_{name}.jsonl")
    return [json.loads(line) for line in open(path)]


def _items(name):
    rows = load_eval(name)
    out = []
    if name == "hellaswag":
        for r in rows:
            ends = r["endings"] if isinstance(r["endings"], list) else eval(r["endings"])
            out.append((r["ctx"], [" " + e for e in ends], int(r["label"])))
    elif name == "arc_easy":
        for r in rows:
            ch = r["choices"] if isinstance(r["choices"], dict) else eval(r["choices"])
            labels = list(ch["label"])
            if r["answerKey"] not in labels:
                continue
            out.append((f"Question: {r['question']}\nAnswer:", [" " + t for t in ch["text"]],
                        labels.index(r["answerKey"])))
    elif name == "piqa":
        for r in rows:
            out.append((f"Question: {r['goal']}\nAnswer:", [" " + r["sol1"], " " + r["sol2"]], int(r["label"])))
    return out


@torch.no_grad()
def score_choices(model, tok, ctx, choices, device="cuda", max_len=1024):
    """Return per-choice (sum logprob, n tokens) of the continuation given ctx."""
    c_ids = tok.encode(ctx)
    seqs, spans = [], []
    for ch in choices:
        e = tok.encode(ch)
        ids = ([tok.bos] + c_ids + e)[-max_len:]
        seqs.append(ids)
        spans.append(len(e))
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), tok.pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    x = x.to(device)
    with torch.autocast("cuda", dtype=torch.float16):
        logits, _ = model(x)
    lp = F.log_softmax(logits.float(), -1)
    res = []
    for i, s in enumerate(seqs):
        n, e = spans[i], len(s)
        tgt = x[i, e - n:e]
        res.append((lp[i, e - n - 1:e - 1].gather(-1, tgt[:, None]).sum().item(), n))
    return res


@torch.no_grad()
def multiple_choice(model, tok, name, limit=None, device="cuda"):
    items = _items(name)
    if limit:
        items = items[:limit]
    r, w = dist.rank(), dist.world()
    correct = correct_norm = n = 0
    for ctx, choices, label in items[r::w]:
        sc = score_choices(model, tok, ctx, choices, device)
        correct += int(max(range(len(sc)), key=lambda i: sc[i][0]) == label)
        correct_norm += int(max(range(len(sc)), key=lambda i: sc[i][0] / sc[i][1]) == label)
        n += 1
    n, correct, correct_norm = dist.all_sum(n), dist.all_sum(correct), dist.all_sum(correct_norm)
    return {"acc": correct / max(n, 1), "acc_norm": correct_norm / max(n, 1), "n": int(n),
            "chance": sum(1 / len(c) for _, c, _ in items) / max(len(items), 1)}


@torch.no_grad()
def lambada(model, tok, limit=None, device="cuda"):
    rows = load_eval("lambada")
    if limit:
        rows = rows[:limit]
    r, w = dist.rank(), dist.world()
    hit = n = 0
    nll = ntok = 0.0
    for row in rows[r::w]:
        text = row["text"]
        ctx, last = text.rsplit(" ", 1)
        c = [tok.bos] + tok.encode(ctx)
        t = tok.encode(" " + last)
        x = torch.tensor([c + t], device=device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits, _ = model(x)
        lp = F.log_softmax(logits[0, len(c) - 1:-1].float(), -1)
        tgt = torch.tensor(t, device=device)
        nll -= lp.gather(-1, tgt[:, None]).sum().item()
        ntok += len(t)
        hit += int((lp.argmax(-1) == tgt).all().item())
        n += 1
    n, hit, nll, ntok = dist.all_sum(n), dist.all_sum(hit), dist.all_sum(nll), dist.all_sum(ntok)
    return {"acc": hit / max(n, 1), "ppl": math.exp(nll / max(ntok, 1)), "n": int(n)}


def run_all(model, tok, limit=None, device="cuda"):
    was = model.training
    model.eval()
    out = {name: multiple_choice(model, tok, name, limit, device) for name in ("hellaswag", "arc_easy", "piqa")}
    out["lambada"] = lambada(model, tok, limit, device)
    model.train(was)
    return out
