"""Stage 7e — on-policy self-distillation (Lecture 4).

The student (current policy) samples a response to a plain prompt. The teacher is
a frozen copy of the same model that additionally sees a privileged reference
answer in its prompt; it scores every student token in one teacher-forced pass.
Loss per token = reverse KL(S || T) = sum_v S log(S / T)  (mode-seeking, dense,
on-policy, nothing to reward-hack). Mixed chat / code / math prompts aim to
recover capabilities that drifted during RL (GLM-5 cross-stage distillation idea).
Run: torchrun --nproc_per_node=2 7_post_training/self_distill.py
"""
import copy
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "5_generation"))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from common import dist, checkpoint  # noqa: E402
from common.config import POST_DIR, SMOKE  # noqa: E402
from common.data import Tok, chat_prompt  # noqa: E402
from common.energy import EnergyMeter, NullMeter  # noqa: E402
from common.logger import Logger, save_json, p  # noqa: E402
from common.model import load_model  # noqa: E402
from engine import Engine  # noqa: E402
import math_tasks  # noqa: E402
from post_common import read_jsonl, ckpt  # noqa: E402

STEPS = 10 if SMOKE else 120
PER_RANK = 4 if SMOKE else 8
MAX_NEW = 64 if SMOKE else 160
LR = 1e-5 if SMOKE else 1e-6
REF_CHARS = 1200


def reverse_kl(s_logits, t_logits):
    """Per-token KL(S || T) over the full vocabulary. (T, V) -> (T,)"""
    ls, lt = F.log_softmax(s_logits.float(), -1), F.log_softmax(t_logits.float(), -1)
    return (ls.exp() * (ls - lt)).sum(-1)


def prompt_pool(seed):
    rng = random.Random(seed)
    chat = [r for r in read_jsonl("sft_chat.jsonl", 20000)
            if len(r["messages"]) >= 2 and r["messages"][-2]["role"] == "user"]
    code = read_jsonl("sft_code.jsonl")
    while True:
        u = rng.random()
        if u < 0.5:
            r = rng.choice(chat)
            yield "chat", r["messages"][-2]["content"], r["messages"][-1]["content"]
        elif u < 0.75:
            r = rng.choice(code)
            yield "code", r["messages"][0]["content"], r["messages"][1]["content"]
        else:
            pb = math_tasks.make_problem(rng, rng.randint(1, 5))
            yield "math", pb["question"], math_tasks.solution_text(pb)


def main():
    rank, local, world = dist.init()
    torch.manual_seed(31 + rank)
    out = os.path.join(POST_DIR, "distill")
    os.makedirs(out, exist_ok=True)
    tok = Tok()
    student, _ = load_model(ckpt("grpo"))
    teacher = copy.deepcopy(student).half().eval()
    for q in teacher.parameters():
        q.requires_grad_(False)
    ddp = dist.wrap(student, local)
    opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.95), fused=True)
    scaler = torch.amp.GradScaler("cuda")
    eng = Engine(student, tok)
    pool = prompt_pool(50 + rank)
    log = Logger(out, enabled=dist.is_main())
    meter = EnergyMeter(out).start() if dist.is_main() else NullMeter()
    t0 = time.time()
    for step in range(STEPS):
        batch = [next(pool) for _ in range(PER_RANK)]
        student.eval()
        rolls = []
        for kind, user, refans in batch:
            sp = chat_prompt(tok, user)[-512:]
            priv = f"{user}\n\n[Reference answer, use it to judge but answer naturally]: {refans[:REF_CHARS]}"
            tp = chat_prompt(tok, priv)[-768:]
            o = eng.generate(sp, MAX_NEW, temperature=1.0)[0] + [tok.end]
            rolls.append((kind, sp, tp, o))
        student.train()
        kls, by_kind = [], {}
        n_tok = sum(len(r[3]) for r in rolls)
        for j, (kind, sp, tp, o) in enumerate(rolls):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                tl, _ = teacher(torch.tensor([tp + o[:-1]], device="cuda"))
            t_logits = tl[0, len(tp) - 1:]
            ctx = ddp.no_sync() if (world > 1 and j < len(rolls) - 1) else torch.enable_grad()
            with ctx:
                with torch.autocast("cuda", dtype=torch.float16):
                    sl, _ = ddp(torch.tensor([sp + o[:-1]], device="cuda"))
                kl = reverse_kl(sl[0, len(sp) - 1:], t_logits)
                loss = kl.sum() / n_tok                      # token-level average over the batch
                scaler.scale(loss).backward()
            kls.append(kl.mean().item())
            by_kind.setdefault(kind, []).append(kl.mean().item())
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        row = dict(step=step, reverse_kl=dist.all_mean(sum(kls) / len(kls)), grad_norm=float(gn),
                   resp_len=dist.all_mean(n_tok / len(rolls)), kl_by_kind={k: sum(v) / len(v) for k, v in by_kind.items()})
        log.log(**row)
        if dist.is_main() and (step % 10 == 0 or step == STEPS - 1):
            p(f"distill step {step}/{STEPS} reverse KL {row['reverse_kl']:.4f} len {row['resp_len']:.1f}")
    if dist.is_main():
        checkpoint.save(os.path.join(out, "final.pt"), student, student.cfg, step=STEPS)
        checkpoint.save(os.path.join(POST_DIR, "final.pt"), student, student.cfg, step=STEPS)
        save_json(os.path.join(out, "summary.json"), {"steps": STEPS, "seconds": time.time() - t0,
                                                       "energy": meter.stop()})
    dist.barrier()
    dist.cleanup()


if __name__ == "__main__":
    main()
