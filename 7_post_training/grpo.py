"""Stage 7d — reasoning RL with verifiable rewards: GRPO with Dr.GRPO / DAPO fixes (Lecture 4).

For each prompt sample G responses (T = 1), reward r_i = correct + 0.1 * format,
advantage A_i = r_i - mean(r)  (no std division: Dr. GRPO), shared by every token.
Objective per token (maximised):
    min(rho * A, clip(rho, 1 - eps_low, 1 + eps_high) * A) - beta * KL_k3(pi || pi_ref)
with token-level normalisation over the whole batch (DAPO, no 1/|o_i| length bias)
and asymmetric clipping (eps_high > eps_low) to keep exploration alive.
mu inner epochs reuse each batch of rollouts through importance ratios rho = pi / pi_old.
Run: torchrun --nproc_per_node=2 7_post_training/grpo.py
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

from common import dist, checkpoint  # noqa: E402
from common.config import POST_DIR, SMOKE  # noqa: E402
from common.data import Tok, chat_prompt  # noqa: E402
from common.energy import EnergyMeter, NullMeter  # noqa: E402
from common.logger import Logger, save_json, p  # noqa: E402
from common.model import load_model  # noqa: E402
from engine import Engine  # noqa: E402
import math_tasks  # noqa: E402
from post_common import response_logprobs, ckpt  # noqa: E402

G = 8
PROMPTS_PER_RANK = 4 if SMOKE else 16
STEPS = 12 if SMOKE else 160
MU = 2
EPS_LOW, EPS_HIGH = 0.2, 0.28
BETA = 0.04
LR = 1e-5 if SMOKE else 1e-6
MAX_NEW = 64 if SMOKE else 160
MICRO = 16


def grpo_advantages(rewards, group, std_norm=False):
    r = rewards.view(-1, group)
    a = r - r.mean(1, keepdim=True)
    if std_norm:
        a = a / (r.std(1, keepdim=True) + 1e-4)
    return a.view(-1)


def grpo_token_loss(lp, old_lp, ref_lp, adv, mask, eps_low=EPS_LOW, eps_high=EPS_HIGH, beta=BETA):
    """Returns (loss to minimise, stats). All tensors (B, T); adv (B,)."""
    ratio = (lp - old_lp).exp()
    A = adv[:, None]
    pg = torch.min(ratio * A, ratio.clamp(1 - eps_low, 1 + eps_high) * A)
    d = ref_lp - lp
    kl = d.exp() - d - 1                         # k3 estimator of KL(pi || pi_ref), >= 0
    per_tok = -(pg - beta * kl)
    n = mask.sum().clamp(min=1)
    loss = (per_tok * mask).sum() / n            # DAPO token-level normalisation
    clipped = (((ratio < 1 - eps_low) & (A < 0)) | ((ratio > 1 + eps_high) & (A > 0))).float()
    return loss, {"kl": ((kl * mask).sum() / n).item(), "clip_frac": ((clipped * mask).sum() / n).item()}


class Curriculum:
    """Sample difficulty levels; unlock harder levels once accuracy on the frontier passes 60%."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.max_level = 2
        self.acc = {lvl: [] for lvl in range(1, 6)}

    def sample(self):
        lvl = self.rng.randint(1, self.max_level)
        return math_tasks.make_problem(self.rng, lvl)

    def update(self, level, correct):
        self.acc[level].append(correct)
        self.acc[level] = self.acc[level][-200:]
        f = self.acc[self.max_level]
        if self.max_level < 5 and len(f) >= 40 and sum(f) / len(f) > 0.6:
            self.max_level += 1


def main():
    rank, local, world = dist.init()
    torch.manual_seed(21 + rank)
    out = os.path.join(POST_DIR, "grpo")
    os.makedirs(out, exist_ok=True)
    tok = Tok()
    policy, _ = load_model(ckpt("dpo"))
    ref = copy.deepcopy(policy).half().eval()
    for q in ref.parameters():
        q.requires_grad_(False)
    ddp = dist.wrap(policy, local)
    opt = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.95), fused=True)
    scaler = torch.amp.GradScaler("cuda")
    eng = Engine(policy, tok)
    cur = Curriculum(100 + rank)
    log = Logger(out, enabled=dist.is_main())
    meter = EnergyMeter(out).start() if dist.is_main() else NullMeter()
    t0 = time.time()
    for step in range(STEPS):
        # ---------------- rollouts (on-policy, current weights)
        policy.eval()
        seqs, starts, rewards, corrects, levels, lens = [], [], [], [], [], []
        t_gen = time.time()
        for _ in range(PROMPTS_PER_RANK):
            pb = cur.sample()
            prompt = chat_prompt(tok, pb["question"])
            outs = eng.generate(prompt, MAX_NEW, temperature=1.0, n=G)
            for o in outs:
                text = tok.decode(o)
                r, c = math_tasks.reward(text, pb)
                seqs.append(prompt + o + [tok.end]); starts.append(len(prompt))
                rewards.append(r); corrects.append(c); levels.append(pb["level"]); lens.append(len(o))
                cur.update(pb["level"], c)
        t_gen = time.time() - t_gen
        rewards_t = torch.tensor(rewards, device="cuda")
        adv = grpo_advantages(rewards_t, G)
        zero_var = (rewards_t.view(-1, G).std(1) == 0).float().mean().item()
        # ---------------- old / ref log-probs (no grad)
        with torch.no_grad():
            old_lp, ref_lp, masks = [], [], []
            for i in range(0, len(seqs), MICRO):
                _, lp, m = response_logprobs(policy, seqs[i:i + MICRO], starts[i:i + MICRO], tok.pad)
                _, rl, _ = response_logprobs(ref, seqs[i:i + MICRO], starts[i:i + MICRO], tok.pad)
                old_lp.append(lp); ref_lp.append(rl); masks.append(m)
        # ---------------- policy update: mu epochs over the rollouts
        policy.train()
        stats = []
        n_tok_total = sum(m.sum().item() for m in masks)
        for epoch in range(MU):
            chunks = list(range(0, len(seqs), MICRO))
            for ci, i in enumerate(chunks):
                ctx = ddp.no_sync() if (world > 1 and ci < len(chunks) - 1) else torch.enable_grad()
                with ctx:
                    _, lp, m = response_logprobs(ddp, seqs[i:i + MICRO], starts[i:i + MICRO], tok.pad)
                    k = ci
                    loss, st = grpo_token_loss(lp, old_lp[k], ref_lp[k], adv[i:i + MICRO], m)
                    w = m.sum().item() / max(n_tok_total, 1)   # token-level weighting across chunks
                    scaler.scale(loss * w).backward()
                stats.append(st)
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        acc = sum(corrects) / len(corrects)
        by_level = {}
        for lv, c in zip(levels, corrects):
            by_level.setdefault(lv, []).append(c)
        row = dict(step=step, reward=dist.all_mean(sum(rewards) / len(rewards)), acc=dist.all_mean(acc),
                   resp_len=dist.all_mean(sum(lens) / len(lens)), kl=dist.all_mean(sum(s["kl"] for s in stats) / len(stats)),
                   clip_frac=dist.all_mean(sum(s["clip_frac"] for s in stats) / len(stats)),
                   zero_var_groups=dist.all_mean(zero_var), max_level=cur.max_level, grad_norm=float(gn),
                   gen_s=t_gen, acc_by_level={k: sum(v) / len(v) for k, v in by_level.items()})
        log.log(**row)
        if dist.is_main() and (step % 10 == 0 or step == STEPS - 1):
            p(f"grpo step {step}/{STEPS} reward {row['reward']:.3f} acc {row['acc']:.3f} len {row['resp_len']:.1f} "
              f"kl {row['kl']:.4f} clip {row['clip_frac']:.3f} level<={cur.max_level} gen {t_gen:.1f}s")
    if dist.is_main():
        checkpoint.save(os.path.join(out, "final.pt"), policy, policy.cfg, step=STEPS)
        save_json(os.path.join(out, "summary.json"), {"steps": STEPS, "G": G, "prompts_per_step": PROMPTS_PER_RANK * world,
                                                       "beta": BETA, "eps": [EPS_LOW, EPS_HIGH], "mu": MU,
                                                       "seconds": time.time() - t0, "energy": meter.stop()})
    dist.barrier()
    dist.cleanup()


if __name__ == "__main__":
    main()
