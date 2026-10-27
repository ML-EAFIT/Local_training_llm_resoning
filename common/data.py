"""Data utilities: tokenizer wrapper, token-shard loader, chat template, best-fit packing."""
import os
import numpy as np
import torch

from common.config import TOK_DIR, SPECIAL_TOKENS


# ---------------------------------------------------------------- tokenizer
class Tok:
    def __init__(self, path=None):
        from tokenizers import Tokenizer
        self.tk = Tokenizer.from_file(path or os.path.join(TOK_DIR, "tokenizer.json"))
        self.special = {t: self.tk.token_to_id(t) for t in SPECIAL_TOKENS}
        self.bos, self.eos, self.pad = self.special["<|bos|>"], self.special["<|eos|>"], self.special["<|pad|>"]
        self.end = self.special["<|end|>"]
        self.user, self.assistant, self.system = (self.special["<|user|>"], self.special["<|assistant|>"],
                                                  self.special["<|system|>"])
        self.think, self.think_end = self.special["<|think|>"], self.special["<|/think|>"]
        self.vocab_size = self.tk.get_vocab_size()

    def encode(self, s):
        return self.tk.encode(s, add_special_tokens=False).ids

    def encode_batch(self, ss):
        return [e.ids for e in self.tk.encode_batch(ss, add_special_tokens=False)]

    def decode(self, ids, skip_special=False):
        return self.tk.decode(list(ids), skip_special_tokens=skip_special)

    def token_str(self, i):
        return self.tk.id_to_token(int(i))


# ---------------------------------------------------------------- chat template
ROLE = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}


def render_chat(tok: Tok, messages, add_generation_prompt=False, max_len=None):
    """Return (ids, loss_mask). Loss only on assistant content + its <|end|> (SFT, Lecture 3)."""
    ids, mask = [tok.bos], [0]
    for m in messages:
        role_id = tok.special[ROLE[m["role"]]]
        body = tok.encode(m["content"]) + [tok.end]
        ids += [role_id] + body
        is_asst = m["role"] == "assistant"
        mask += [0] + [1 if is_asst else 0] * len(body)
    if add_generation_prompt:
        ids.append(tok.assistant)
        mask.append(0)
    if max_len is not None:
        ids, mask = ids[:max_len], mask[:max_len]
    return ids, mask


def chat_prompt(tok: Tok, user, system=None, history=None):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs += history or []
    msgs.append({"role": "user", "content": user})
    return render_chat(tok, msgs, add_generation_prompt=True)[0]


def collate_sft(batch, pad_id, max_len):
    """batch: list of (ids, mask) -> x, y, loss_mask tensors (next-token shifted)."""
    L = min(max(len(b[0]) for b in batch), max_len + 1)
    X = torch.full((len(batch), L - 1), pad_id, dtype=torch.long)
    Y = torch.full((len(batch), L - 1), -100, dtype=torch.long)
    for i, (ids, m) in enumerate(batch):
        ids, m = ids[:L], m[:L]
        n = len(ids) - 1
        if n <= 0:
            continue
        X[i, :n] = torch.tensor(ids[:-1])
        y = torch.tensor(ids[1:])
        mm = torch.tensor(m[1:]).bool()
        y[~mm] = -100
        Y[i, :n] = y
    return X, Y


# ---------------------------------------------------------------- pre-training loader
class ShardLoader:
    """Iterates a uint16 token memmap in shuffled (seq_len+1)-chunks, disjoint per rank.

    State (epoch, cursor) is checkpointable so a 15 h run resumes exactly.
    """

    def __init__(self, path, seq_len, batch, rank=0, world=1, seed=1234, device="cuda"):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len, self.batch = seq_len, batch
        self.rank, self.world, self.seed, self.device = rank, world, seed, device
        self.n_chunks = (len(self.data) - 1) // seq_len
        self.epoch, self.cursor = 0, 0
        self._perm()

    def _perm(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        perm = rng.permutation(self.n_chunks)
        self.order = perm[self.rank::self.world]

    def state(self):
        return {"epoch": self.epoch, "cursor": self.cursor}

    def load_state(self, s):
        self.epoch, self.cursor = s["epoch"], s["cursor"]
        self._perm()

    def next(self):
        if self.cursor + self.batch > len(self.order):
            self.epoch += 1
            self.cursor = 0
            self._perm()
        idx = self.order[self.cursor:self.cursor + self.batch]
        self.cursor += self.batch
        L = self.seq_len
        arr = np.stack([self.data[i * L: i * L + L + 1] for i in idx]).astype(np.int64)
        t = torch.from_numpy(arr)
        if self.device == "cuda":
            t = t.pin_memory().cuda(non_blocking=True)
        return t[:, :-1], t[:, 1:]


def sequential_batches(path, seq_len, batch, max_tokens, device="cuda"):
    """Deterministic validation batches from the start of a val shard."""
    data = np.memmap(path, dtype=np.uint16, mode="r")
    n = min(len(data) - 1, max_tokens) // seq_len
    for s in range(0, n - batch + 1, batch):
        arr = np.stack([data[i * seq_len: i * seq_len + seq_len + 1] for i in range(s, s + batch)]).astype(np.int64)
        t = torch.from_numpy(arr).to(device)
        yield t[:, :-1], t[:, 1:]


# ---------------------------------------------------------------- best-fit packing
def best_fit_pack(docs, L, pad_id):
    """Best-fit-decreasing packing of token lists into rows of length L (GLM-5 mid-training).

    Documents longer than L are split into L-sized pieces first. Returns
    (rows uint16 [N, L], loss mask uint8 [N, L]) where padding is masked out.
    """
    pieces = []
    for d in docs:
        for i in range(0, len(d), L):
            pieces.append(d[i:i + L])
    pieces.sort(key=len, reverse=True)
    import bisect
    bins, free = [], []          # free: sorted list of (remaining, bin_idx)
    for p in pieces:
        n = len(p)
        j = bisect.bisect_left(free, (n, -1))
        if j < len(free):
            rem, b = free.pop(j)
            bins[b].append(p)
            rem -= n
        else:
            bins.append([p])
            b, rem = len(bins) - 1, L - n
        if rem > 0:
            bisect.insort(free, (rem, b))
    rows = np.full((len(bins), L), pad_id, dtype=np.uint16)
    mask = np.zeros((len(bins), L), dtype=np.uint8)
    for i, b in enumerate(bins):
        flat = [t for p in b for t in p]
        rows[i, :len(flat)] = flat
        mask[i, :len(flat)] = 1
    return rows, mask
