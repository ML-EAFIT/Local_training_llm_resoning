"""Stage 5 — inference engine (Lecture 2 decoding + old Lecture 3 inference efficiency).

- autoregressive decoding with a pre-allocated KV cache (only K/V cached, GQA-aware)
- greedy, temperature, top-k, top-p (nucleus), repetition penalty, stop tokens
- beam search
- guided (constrained) decoding: only tokens that keep the output a valid prefix
- speculative decoding with a small draft model (greedy acceptance)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def sample_next(logits, temperature=1.0, top_k=0, top_p=1.0):
    """logits (B, V) -> next ids (B,). temperature 0 = greedy (argmax)."""
    if temperature <= 0:
        return logits.argmax(-1)
    logits = logits.float() / temperature          # softmax(x / T): T>1 flattens, T<1 sharpens
    if top_k:
        kth = logits.topk(min(top_k, logits.size(-1)), -1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:                                 # smallest set with cumulative prob >= p
        srt, idx = logits.sort(-1, descending=True)
        cum = srt.softmax(-1).cumsum(-1)
        remove = cum - srt.softmax(-1) > top_p
        srt = srt.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, srt)
    return torch.multinomial(logits.softmax(-1), 1).squeeze(-1)


class Pattern:
    """Tiny prefix-checkable grammar: a list of ("lit", str) and ("digits", (min, max)) segments.

    Example JSON answer:  [("lit", '{"answer": '), ("digits", (1, 8)), ("lit", "}")]
    """

    def __init__(self, segments):
        self.segments = segments

    def check(self, text):
        """Return 'complete', 'prefix' or None (invalid)."""
        i = 0
        for k, (kind, val) in enumerate(self.segments):
            if kind == "lit":
                rest = text[i:]
                if len(rest) < len(val):
                    return "prefix" if val.startswith(rest) else None
                if not rest.startswith(val):
                    return None
                i += len(val)
            else:
                lo, hi = val
                j = i
                if j < len(text) and text[j] == "-":
                    j += 1
                while j < len(text) and text[j].isdigit():
                    j += 1
                n = j - i
                if n > hi:
                    return None
                if j == len(text):
                    return "prefix"
                if n < lo:
                    return None
                i = j
        return "complete" if i == len(text) else None


class Engine:
    def __init__(self, model, tok, device="cuda"):
        self.model, self.tok, self.device = model, tok, device
        self.stop = {tok.eos, tok.end}
        self._tok_strs = None

    # ------------------------------------------------------------------ core sampling
    @torch.no_grad()
    def generate(self, prompt, max_new=128, temperature=0.0, top_k=0, top_p=1.0, n=1,
                 repetition_penalty=1.0, use_cache=True, stop=None, return_logprobs=False):
        """Sample n continuations of one prompt (list of ids). Returns list of token lists
        (stop token excluded) and optionally per-token log-probs under the sampling policy."""
        m = self.model
        stop = self.stop if stop is None else stop
        max_len = min(len(prompt) + max_new, m.cfg.max_seq_len)
        max_new = max_len - len(prompt)
        x = torch.tensor([prompt] * n, device=self.device)
        out = torch.zeros((n, 0), dtype=torch.long, device=self.device)
        done = torch.zeros(n, dtype=torch.bool, device=self.device)
        lps = []
        cache = m.new_cache(n, max_len) if use_cache else None
        inp = x
        for t in range(max_new):
            with torch.autocast("cuda", dtype=torch.float16):
                logits, _ = m(inp, cache=cache) if use_cache else m(torch.cat([x, out], 1))
            lg = logits[:, -1].float()
            if repetition_penalty != 1.0 and out.shape[1]:
                prev = torch.cat([x, out], 1)
                sc = lg.gather(1, prev)
                lg.scatter_(1, prev, torch.where(sc > 0, sc / repetition_penalty, sc * repetition_penalty))
            nxt = sample_next(lg, temperature, top_k, top_p)
            if return_logprobs:
                lps.append(F.log_softmax(lg / max(temperature, 1e-6), -1).gather(1, nxt[:, None]).squeeze(1))
            nxt = torch.where(done, torch.full_like(nxt, self.tok.pad), nxt)
            out = torch.cat([out, nxt[:, None]], 1)
            done |= torch.isin(nxt, torch.tensor(list(stop), device=self.device))
            if done.all():
                break
            inp = nxt[:, None]
        seqs = []
        for i in range(n):
            s = out[i].tolist()
            cut = next((j for j, v in enumerate(s) if v in stop), len(s))
            seqs.append(s[:cut])
        if return_logprobs:
            return seqs, torch.stack(lps, 1) if lps else None
        return seqs

    def chat(self, user, system=None, history=None, **kw):
        from common.data import chat_prompt
        ids = chat_prompt(self.tok, user, system, history)
        return self.tok.decode(self.generate(ids, **kw)[0])

    # ------------------------------------------------------------------ beam search
    @torch.no_grad()
    def beam_search(self, prompt, beam=4, max_new=64, length_penalty=1.0):
        m = self.model
        max_len = min(len(prompt) + max_new, m.cfg.max_seq_len)
        cache = m.new_cache(beam, max_len)
        x = torch.tensor([prompt] * beam, device=self.device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits, _ = m(x, cache=cache)
        lp = F.log_softmax(logits[:, -1].float(), -1)
        scores = torch.full((beam,), float("-inf"), device=self.device)
        scores[0] = 0.0                                  # all beams identical at start
        seqs = [[] for _ in range(beam)]
        finished = []
        for t in range(max_len - len(prompt)):
            cand = (scores[:, None] + lp).view(-1)
            top = cand.topk(2 * beam)
            new_seqs, new_scores, src, nxt = [], [], [], []
            for s, idx in zip(top.values.tolist(), top.indices.tolist()):
                b, tkn = divmod(idx, lp.size(-1))
                if tkn in self.stop:
                    finished.append((s / (len(seqs[b]) + 1) ** length_penalty, seqs[b]))
                    continue
                new_seqs.append(seqs[b] + [tkn]); new_scores.append(s); src.append(b); nxt.append(tkn)
                if len(new_seqs) == beam:
                    break
            if len(finished) >= beam or not new_seqs:
                break
            while len(new_seqs) < beam:
                new_seqs.append(new_seqs[0]); new_scores.append(float("-inf")); src.append(src[0]); nxt.append(nxt[0])
            cache.reorder(torch.tensor(src, device=self.device))
            seqs, scores = new_seqs, torch.tensor(new_scores, device=self.device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, _ = m(torch.tensor(nxt, device=self.device)[:, None], cache=cache)
            lp = F.log_softmax(logits[:, -1].float(), -1)
        for s, q in zip(scores.tolist(), seqs):
            finished.append((s / max(len(q), 1) ** length_penalty, q))
        finished.sort(key=lambda z: -z[0])
        return finished[0][1], finished[0][0]

    # ------------------------------------------------------------------ guided decoding
    def _strings(self):
        if self._tok_strs is None:
            self._tok_strs = [self.tok.decode([i]) for i in range(self.tok.vocab_size)]
        return self._tok_strs

    @torch.no_grad()
    def generate_guided(self, prompt, pattern: Pattern, max_new=32):
        """Greedy decoding restricted to tokens that keep the text a valid prefix of `pattern`.
        Guarantees syntactically valid output (Lecture 2: guided decoding)."""
        m, strs = self.model, self._strings()
        cache = m.new_cache(1, len(prompt) + max_new + 1)
        inp = torch.tensor([prompt], device=self.device)
        text, out = "", []
        for _ in range(max_new):
            with torch.autocast("cuda", dtype=torch.float16):
                logits, _ = m(inp, cache=cache)
            order = logits[0, -1].float().argsort(descending=True).tolist()
            if pattern.check(text) == "complete":
                break
            chosen = None
            for tkn in order[:5000]:            # most likely first; stop at the first legal token
                s = strs[tkn]
                if not s or tkn in self.stop:
                    continue
                if pattern.check(text + s) is not None:
                    chosen = tkn
                    break
            if chosen is None:
                break
            out.append(chosen)
            text += strs[chosen]
            inp = torch.tensor([[chosen]], device=self.device)
        return out, text

    # ------------------------------------------------------------------ speculative decoding
    @torch.no_grad()
    def speculative(self, prompt, draft, k=4, max_new=128):
        """Greedy speculative decoding: the draft proposes k tokens, the target verifies them in
        one parallel pass and keeps the longest agreeing prefix plus its own next token.

        Invariant: the target cache holds seq[:-1] (every committed token except the last).
        """
        m = self.model
        max_len = min(len(prompt) + max_new + k + 2, m.cfg.max_seq_len, draft.cfg.max_seq_len)
        tc, dc = m.new_cache(1, max_len), draft.new_cache(1, max_len)
        with torch.autocast("cuda", dtype=torch.float16):
            tl, _ = m(torch.tensor([prompt], device=self.device), cache=tc)
            draft(torch.tensor([prompt[:-1]], device=self.device), cache=dc)
        seq = list(prompt) + [int(tl[0, -1].argmax())]
        proposed = accepted = 0
        while len(seq) - len(prompt) < max_new and len(seq) + k + 1 < max_len and seq[-1] not in self.stop:
            d_in = torch.tensor([seq[dc.pos:]], device=self.device)
            props = []
            for _ in range(k):
                with torch.autocast("cuda", dtype=torch.float16):
                    dl, _ = draft(d_in, cache=dc)
                props.append(int(dl[0, -1].argmax()))
                d_in = torch.tensor([[props[-1]]], device=self.device)
            with torch.autocast("cuda", dtype=torch.float16):
                vl, _ = m(torch.tensor([[seq[-1]] + props], device=self.device), cache=tc)
            tgt = vl[0].argmax(-1).tolist()          # k+1 target predictions
            n_ok = 0
            while n_ok < k and props[n_ok] == tgt[n_ok]:
                n_ok += 1
            proposed += k
            accepted += n_ok
            n_before = len(seq)
            seq += props[:n_ok] + [tgt[n_ok]]
            tc.crop(n_before + n_ok)                  # = len(seq) - 1
            dc.crop(min(dc.pos, n_before + min(n_ok, k - 1)))
        out = seq[len(prompt):]
        stop_at = next((i for i, t in enumerate(out) if t in self.stop), len(out))
        return out[:stop_at], {"proposed": proposed, "accepted": accepted,
                               "acceptance_rate": accepted / max(proposed, 1)}


def timed(fn, *a, **kw):
    torch.cuda.synchronize()
    t0 = time.time()
    r = fn(*a, **kw)
    torch.cuda.synchronize()
    return r, time.time() - t0
