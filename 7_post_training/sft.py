"""Stage 7a — supervised fine-tuning (instruction tuning, Lecture 3).

L = -(1/T) sum_t log p(y_t | x, y_<t)   computed ONLY on assistant tokens.
Data: smol-smoltalk chat + Python instructions + verifiable math with
<|think|> chains + GSM8K train. Also trains a LoRA (attention + MLP, r=16)
run for the same number of steps to compare cost and quality.
Run: torchrun --nproc_per_node=2 7_post_training/sft.py [--lora]
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

from common import dist, checkpoint  # noqa: E402
from common.config import MID_DIR, POST_DIR, SMOKE  # noqa: E402
from common.data import Tok, render_chat, collate_sft  # noqa: E402
from common.energy import EnergyMeter, NullMeter  # noqa: E402
from common.logger import save_json, p  # noqa: E402
from common.lora import apply_lora, merge_lora, trainable_params  # noqa: E402
from common.model import load_model  # noqa: E402
from common.trainer import TrainArgs, train  # noqa: E402
import math_tasks  # noqa: E402
from post_common import read_jsonl, shuffled  # noqa: E402

MAX_LEN = 512 if SMOKE else 1024
N_MATH = 2000 if SMOKE else 40000
EPOCHS = 1
MICRO = 4
GLOBAL_SEQS = 16 if SMOKE else 64


def build_examples(tok):
    ex = []
    for r in read_jsonl("sft_chat.jsonl"):
        msgs = [m for m in r["messages"] if m["role"] in ("system", "user", "assistant") and m["content"]]
        if msgs and msgs[-1]["role"] == "assistant":
            ex.append(("chat", msgs))
    for r in read_jsonl("sft_code.jsonl"):
        ex.append(("code", r["messages"]))
    for pb in math_tasks.dataset(N_MATH, seed=1):
        ex.append(("math", [{"role": "user", "content": pb["question"]},
                            {"role": "assistant", "content": math_tasks.solution_text(pb)}]))
    for r in read_jsonl("eval_gsm8k_train.jsonl"):
        pb = math_tasks.gsm8k_to_sft(r)
        ex.append(("gsm8k", [{"role": "user", "content": pb["question"]},
                             {"role": "assistant", "content": math_tasks.solution_text(pb)}]))
    ex = shuffled(ex, 0)
    rendered = []
    for kind, msgs in ex:
        ids, mask = render_chat(tok, msgs, max_len=MAX_LEN + 1)
        if sum(mask) > 0:
            rendered.append((kind, ids, mask))
    n_val = 64 if SMOKE else 1000
    return rendered[n_val:], rendered[:n_val]


class SFTLoader:
    def __init__(self, data, micro, rank, world, pad):
        self.data = [d for i, d in enumerate(data) if i % world == rank]
        self.micro, self.pad, self.cursor, self.epoch = micro, pad, 0, 0
        # length-grouped batches reduce padding: sort chunks of 50 batches by length
        self._order()

    def _order(self):
        idx = list(range(len(self.data)))
        random.Random(self.epoch).shuffle(idx)
        chunk = self.micro * 50
        self.idx = []
        for s in range(0, len(idx), chunk):
            self.idx += sorted(idx[s:s + chunk], key=lambda i: len(self.data[i][1]))
        batches = [self.idx[i:i + self.micro] for i in range(0, len(self.idx), self.micro)]
        random.Random(self.epoch + 99).shuffle(batches)
        self.batches = batches

    def state(self):
        return {"cursor": self.cursor, "epoch": self.epoch}

    def load_state(self, s):
        self.cursor, self.epoch = s["cursor"], s["epoch"]
        self._order()

    def next(self):
        if self.cursor >= len(self.batches):
            self.epoch += 1; self.cursor = 0; self._order()
        b = [self.data[i][1:] for i in self.batches[self.cursor]]
        self.cursor += 1
        x, y = collate_sft(b, self.pad, MAX_LEN)
        return x.cuda(non_blocking=True), y.cuda(non_blocking=True)


def make_val(val, pad):
    @torch.no_grad()
    def fn(model, step):
        out, tot = {}, {}
        for kind, ids, mask in val[dist.rank()::dist.world()]:
            x, y = collate_sft([(ids, mask)], pad, MAX_LEN)
            with torch.autocast("cuda", dtype=torch.float16):
                _, l = model(x.cuda(), y.cuda())
            tot.setdefault(kind, []).append(l.item())
        for kind in ("chat", "code", "math", "gsm8k"):
            v = tot.get(kind, [])
            out[f"val_loss_{kind}"] = dist.all_sum(sum(v)) / max(dist.all_sum(len(v)), 1)
        out["val_loss"] = sum(out.values()) / len(out)
        if dist.is_main():
            p(f"sft eval step {step}: {out}")
        return out
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora_frac", type=float, default=0.25, help="LoRA run length as fraction of full SFT steps")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    rank, local, world = dist.init()
    torch.manual_seed(11 + rank)
    name = "sft_lora" if args.lora else "sft"
    out = os.path.join(POST_DIR, name)
    os.makedirs(out, exist_ok=True)
    tok = Tok()
    train_ex, val_ex = build_examples(tok)
    model, _ = load_model(os.path.join(MID_DIR, "final.pt"))
    cfg = model.cfg
    cfg.grad_ckpt = not SMOKE
    n_total = model.num_params()
    if args.lora:
        apply_lora(model, r=args.rank, alpha=2 * args.rank)
        model.cuda()
    n_train = trainable_params(model)
    accum = max(1, GLOBAL_SEQS // (MICRO * world))
    steps = max(1, EPOCHS * len(train_ex) // (MICRO * world * accum))
    if args.lora:
        steps = max(1, int(steps * args.lora_frac))
    mean_len = sum(len(e[1]) for e in train_ex[:5000]) / min(len(train_ex), 5000)
    a = TrainArgs(steps=steps, accum=accum, tokens_per_step=int(mean_len * MICRO * world * accum),
                  lr=(2e-4 if args.lora else 5e-5) * (10 if SMOKE else 1), warmup=max(1, steps // 30),
                  schedule="cosine", min_lr_frac=0.1, weight_decay=0.0, eval_every=max(steps // 8, 1),
                  log_every=1 if steps < 200 else 10, seq_len=MAX_LEN, ckpt_minutes=20)
    if dist.is_main():
        kinds = {}
        for k, _, _ in train_ex:
            kinds[k] = kinds.get(k, 0) + 1
        p(f"{name}: {len(train_ex)} examples {kinds}, {steps} steps, trainable {n_train/1e6:.2f}M / {n_total/1e6:.1f}M")
    meter = EnergyMeter(out).start() if dist.is_main() else NullMeter()
    loader = SFTLoader(train_ex, MICRO, rank, world, tok.pad)
    t0 = time.time()
    summ = train(model, cfg, a, loader.next, out, eval_fn=make_val(val_ex, tok.pad), meter=meter,
                 state_obj=loader)
    if args.lora:
        merge_lora(model)
    final_eval = make_val(val_ex, tok.pad)(model.eval(), steps)
    if dist.is_main():
        checkpoint.save(os.path.join(out, "final.pt"), model, cfg, step=steps)
        summ.update(final_eval=final_eval, seconds=time.time() - t0, energy=meter.stop(), trainable_params=n_train,
                    total_params=n_total, lora=args.lora, lora_rank=args.rank if args.lora else None,
                    n_examples=len(train_ex), sec_per_step=(time.time() - t0) / steps)
        save_json(os.path.join(out, "summary.json"), summ)
    dist.barrier()
    dist.cleanup()


if __name__ == "__main__":
    main()
