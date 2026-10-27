"""Shared helpers for post-training: data loading, chat rendering, sequence log-probs,
and a small DDP loop for losses that do not fit the next-token trainer."""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from common.config import RAW_DIR, POST_DIR  # noqa: E402
from common.data import render_chat  # noqa: E402


def read_jsonl(name, limit=None):
    rows = []
    with open(os.path.join(RAW_DIR, "aux", name)) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rows.append(json.loads(line))
    return rows


def pair_ids(tok, prompt, response, max_len, max_prompt=None):
    """Render (user prompt, assistant response); return ids and index where the response starts."""
    p_ids, _ = render_chat(tok, [{"role": "user", "content": prompt}], add_generation_prompt=True)
    if max_prompt and len(p_ids) > max_prompt:
        p_ids = p_ids[:1] + p_ids[-(max_prompt - 1):]
    r_ids = tok.encode(response) + [tok.end]
    ids = (p_ids + r_ids)[:max_len]
    return ids, len(p_ids)


def pad_batch(seqs, pad):
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    return x.cuda()


def response_logprobs(model, seqs, starts, pad, return_logits=False):
    """Sum of log p over response tokens for each sequence, plus per-token values and mask."""
    x = pad_batch(seqs, pad)
    with torch.autocast("cuda", dtype=torch.float16):
        logits, _ = model(x[:, :-1])
    lp = torch.gather(F.log_softmax(logits.float(), -1), -1, x[:, 1:, None]).squeeze(-1)
    mask = torch.zeros_like(lp)
    for i, (s, st) in enumerate(zip(seqs, starts)):
        mask[i, st - 1:len(s) - 1] = 1.0
    if return_logits:
        return (lp * mask).sum(-1), lp, mask, logits
    return (lp * mask).sum(-1), lp, mask


def ckpt(name):
    return os.path.join(POST_DIR, name, "final.pt")


def shuffled(rows, seed):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows
