"""Stage 5 — evaluate generation / inference of a checkpoint.

perplexity, KV-cache speed-up and memory (MHA vs GQA vs MQA), cache == no-cache
check, speculative decoding (acceptance + speed-up), decoding-strategy sweep
(diversity / repetition / length), beam vs greedy log-prob, guided decoding
validity, and text samples.
Usage: python 5_generation/eval_generation.py --ckpt X --tag base [--draft Y]
"""
import argparse
import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

from common.config import TOK_DIR, GEN_DIR, PRE_DIR, SMOKE, stage_dir  # noqa: E402
from common.data import Tok, sequential_batches  # noqa: E402
from common.logger import save_json, load_json, p  # noqa: E402
from common.model import load_model, token_logprobs  # noqa: E402
from common.energy import EnergyMeter  # noqa: E402
from engine import Engine, Pattern, timed  # noqa: E402

PROMPTS = [
    "The history of the Roman Empire",
    "In mathematics, a prime number is",
    "def quicksort(arr):",
    "Water boils at 100 degrees Celsius because",
    "The most important thing about learning is",
    "Once upon a time, in a small village,",
]
SETTINGS = {
    "greedy": dict(temperature=0.0),
    "T0.7_p0.9": dict(temperature=0.7, top_p=0.9),
    "T1.0": dict(temperature=1.0),
    "T1.0_k40": dict(temperature=1.0, top_k=40),
    "T1.5": dict(temperature=1.5),
    "greedy_rep1.3": dict(temperature=0.0, repetition_penalty=1.3),
}


def distinct_n(seqs, n):
    grams = [tuple(s[i:i + n]) for s in seqs for i in range(len(s) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


def repetition_rate(s, n=4):
    grams = [tuple(s[i:i + n]) for i in range(len(s) - n + 1)]
    return 1 - len(set(grams)) / max(len(grams), 1)


@torch.no_grad()
def perplexity(model, seq_len):
    out = {}
    for path in sorted(glob.glob(os.path.join(TOK_DIR, "val_*.bin"))):
        d = os.path.basename(path)[4:-4]
        tot = n = 0
        for x, y in sequential_batches(path, seq_len, 4, 131072 if not SMOKE else 16384):
            with torch.autocast("cuda", dtype=torch.float16):
                _, l = model(x, y)
            tot += l.item(); n += 1
        out[d] = math.exp(tot / max(n, 1))
    return out


def kv_memory(cfg, seq_len, batch=1, bytes_per=2):
    hd = cfg.d_model // cfg.n_head
    per = lambda kv: 2 * cfg.n_layer * kv * hd * seq_len * batch * bytes_per
    return {"gqa_current": per(cfg.n_kv_head), "mha": per(cfg.n_head), "mqa": per(1),
            "n_head": cfg.n_head, "n_kv_head": cfg.n_kv_head, "seq_len": seq_len}


def find_draft():
    cands = []
    for f in glob.glob(os.path.join(PRE_DIR, "sweep", "iso_*", "final.pt")):
        s = load_json(os.path.join(os.path.dirname(f), "summary.json"), {})
        cands.append((s.get("params", 1e18), s.get("final_eval", {}).get("val_loss", 99), f))
    if not cands:
        return None
    # smallest model among the best-trained third
    cands.sort(key=lambda c: c[1])
    best = cands[: max(1, len(cands) // 3)]
    return min(best)[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(PRE_DIR, "final.pt"))
    ap.add_argument("--tag", default="base")
    ap.add_argument("--draft", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out_dir = stage_dir("5_generation")
    meter = EnergyMeter(out_dir, name=f"energy_{a.tag}").start()
    tok = Tok()
    model, _ = load_model(a.ckpt)
    model.half().eval()
    eng = Engine(model, tok)
    res = {"tag": a.tag, "ckpt": a.ckpt}
    seq_len = model.cfg.max_seq_len
    res["perplexity"] = perplexity(model, min(seq_len, 1024))
    p("perplexity", res["perplexity"])

    # ---- KV cache: speed and memory
    prompt = [tok.bos] + tok.encode(PROMPTS[0])
    new = 64 if SMOKE else 256
    speed = {}
    for bs in (1, 8):
        eng.generate(prompt, 8, n=bs)   # warm-up
        _, t_c = timed(eng.generate, prompt, new, n=bs, stop=set())
        _, t_n = timed(eng.generate, prompt, new, n=bs, stop=set(), use_cache=False)
        speed[f"batch{bs}"] = {"tok_s_cache": bs * new / t_c, "tok_s_nocache": bs * new / t_n,
                               "speedup": t_n / t_c, "ms_per_token_cache": 1000 * t_c / new}
    res["speed"] = speed
    cache = model.new_cache(1, seq_len)
    res["kv_cache"] = {"measured_bytes_fp16": cache.nbytes(), **kv_memory(model.cfg, seq_len)}
    del cache
    g1 = eng.generate(prompt, 48, use_cache=True)[0]
    g2 = eng.generate(prompt, 48, use_cache=False)[0]
    res["cache_matches_nocache"] = (g1 == g2)
    p("speed", speed, "cache==nocache", res["cache_matches_nocache"])

    # ---- speculative decoding
    draft_path = a.draft or find_draft()
    if draft_path and os.path.exists(draft_path):
        draft, _ = load_model(draft_path)
        draft.half().eval()
        if draft.cfg.vocab_size == model.cfg.vocab_size:
            spec = {"draft": draft_path, "draft_params": draft.num_params(), "target_params": model.num_params(),
                    "by_k": {}}
            for k in (2, 4, 6):
                acc = []
                t_spec = t_greedy = 0.0
                for pr in PROMPTS[:4]:
                    ids = [tok.bos] + tok.encode(pr)
                    (o, st), t1 = timed(eng.speculative, ids, draft, k, new)
                    _, t2 = timed(eng.generate, ids, len(o) or 1, stop=set())
                    acc.append(st["acceptance_rate"]); t_spec += t1; t_greedy += t2
                spec["by_k"][k] = {"acceptance_rate": sum(acc) / len(acc), "speedup": t_greedy / t_spec}
            res["speculative"] = spec
            p("speculative", spec["by_k"])

    # ---- decoding strategies
    strat = {}
    samples = {}
    for name, kw in SETTINGS.items():
        seqs = []
        for pr in PROMPTS:
            ids = [tok.bos] + tok.encode(pr)
            n = 1 if kw.get("temperature", 0) == 0 else 4
            outs = eng.generate(ids, 96 if not SMOKE else 32, n=n, **kw)
            seqs += outs
            samples.setdefault(name, []).append(pr + tok.decode(outs[0]))
        strat[name] = {"distinct_1": distinct_n(seqs, 1), "distinct_2": distinct_n(seqs, 2),
                       "repetition_4gram": sum(repetition_rate(s) for s in seqs) / len(seqs),
                       "mean_len": sum(len(s) for s in seqs) / len(seqs)}
    res["strategies"] = strat
    res["samples"] = samples

    # ---- beam vs greedy: sequence log-prob under the model
    bvg = []
    for pr in PROMPTS[:3]:
        ids = [tok.bos] + tok.encode(pr)
        g = eng.generate(ids, 24, stop=set())[0]
        b, _ = eng.beam_search(ids, beam=4, max_new=24, length_penalty=0.0)
        def lp(c):
            x = torch.tensor([ids + c], device="cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                return token_logprobs(model, x[:, :-1], x[:, 1:])[0, len(ids) - 1:].sum().item()
        bvg.append({"prompt": pr, "greedy_logprob": lp(g), "beam_logprob": lp(b),
                    "greedy": tok.decode(g), "beam": tok.decode(b)})
    res["beam_vs_greedy"] = bvg

    # ---- guided decoding validity
    pat = Pattern([("lit", '{"answer": '), ("digits", (1, 6)), ("lit", "}")])
    valid_free = valid_guided = 0
    qs = [(a_, b_) for a_ in (3, 12, 25, 47) for b_ in (4, 9, 18)]
    for x_, y_ in qs:
        ids = [tok.bos] + tok.encode(f'Q: What is {x_} + {y_}? Reply in JSON like {{"answer": 7}}.\nA: ')
        free = tok.decode(eng.generate(ids, 16)[0]).strip()
        valid_free += pat.check(free.split("\n")[0].strip()) == "complete"
        _, gtxt = eng.generate_guided(ids, pat)
        valid_guided += pat.check(gtxt) == "complete"
    res["guided"] = {"n": len(qs), "valid_unconstrained": valid_free / len(qs),
                     "valid_guided": valid_guided / len(qs)}
    p("guided", res["guided"])
    res["energy"] = meter.stop()
    save_json(os.path.join(out_dir, f"summary_{a.tag}.json"), res)
    if a.tag == "base":
        save_json(os.path.join(out_dir, "summary.json"), res)


if __name__ == "__main__":
    main()
