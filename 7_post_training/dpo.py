"""Stage 7c — Direct Preference Optimization (Lecture 3/5).

implicit reward r = beta * log(pi_theta(y|x) / pi_ref(y|x))
L = -E[ log sigma( r(x, y_w) - r(x, y_l) ) ]        (beta = 0.1, ref = frozen SFT model)
No reward model, no RL; off-policy (fixed preference pairs).
Run: torchrun --nproc_per_node=2 7_post_training/dpo.py
"""
import copy
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from common import dist, checkpoint  # noqa: E402
from common.config import POST_DIR, SMOKE  # noqa: E402
from common.data import Tok  # noqa: E402
from common.energy import EnergyMeter, NullMeter  # noqa: E402
from common.logger import Logger, save_json, p  # noqa: E402
from common.model import load_model  # noqa: E402
from post_common import read_jsonl, pair_ids, response_logprobs, ckpt, shuffled  # noqa: E402

BETA = 0.1
MAX_LEN = 384 if SMOKE else 768
N_PAIRS = 800 if SMOKE else 16000
MICRO = 2
ACCUM = 8
LR = 1e-5 if SMOKE else 1e-6


def dpo_loss(pc, pr, rc, rr, beta=BETA):
    """pc/pr: policy log-probs of chosen/rejected, rc/rr: reference. Returns loss, rewards."""
    r_w, r_l = beta * (pc - rc), beta * (pr - rr)
    return -F.logsigmoid(r_w - r_l).mean(), r_w, r_l


def main():
    rank, local, world = dist.init()
    torch.manual_seed(5 + rank)
    out = os.path.join(POST_DIR, "dpo")
    os.makedirs(out, exist_ok=True)
    tok = Tok()
    policy, _ = load_model(ckpt("sft"))
    policy.cfg.grad_ckpt = not SMOKE
    ref = copy.deepcopy(policy).half().eval()
    for q in ref.parameters():
        q.requires_grad_(False)
    ddp = dist.wrap(policy, local)
    opt = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.95), fused=True)
    scaler = torch.amp.GradScaler("cuda")
    rows = shuffled(read_jsonl("pref_train.jsonl"), 1)[:N_PAIRS][rank::world]
    test = read_jsonl("pref_test.jsonl")[: 100 if SMOKE else 500][rank::world]
    steps = len(rows) // (MICRO * ACCUM)
    log = Logger(out, enabled=dist.is_main())
    meter = EnergyMeter(out).start() if dist.is_main() else NullMeter()

    def encode(batch):
        seqs, starts = [], []
        for r in batch:
            for resp in (r["chosen"], r["rejected"]):
                ids, st = pair_ids(tok, r["prompt"], resp, MAX_LEN, MAX_LEN // 2)
                seqs.append(ids); starts.append(st)
        return seqs, starts

    @torch.no_grad()
    def evaluate():
        policy.eval()
        acc, marg, kl = [], [], []
        for i in range(0, len(test), MICRO):
            seqs, starts = encode(test[i:i + MICRO])
            lp, tok_lp, mask = response_logprobs(policy, seqs, starts, tok.pad)
            rlp, rtok, _ = response_logprobs(ref, seqs, starts, tok.pad)
            _, rw, rl = dpo_loss(lp[0::2], lp[1::2], rlp[0::2], rlp[1::2])
            acc += (rw > rl).float().tolist(); marg += (rw - rl).tolist()
            kl += (((tok_lp - rtok) * mask).sum(-1) / mask.sum(-1)).tolist()  # mean log-ratio on data
        policy.train()
        n = dist.all_sum(len(acc))
        return {"pref_acc": dist.all_sum(sum(acc)) / n, "margin": dist.all_sum(sum(marg)) / n,
                "logratio": dist.all_sum(sum(kl)) / n}

    before = evaluate()
    if dist.is_main():
        p(f"DPO: {steps} steps, beta {BETA}; before {before}")
    t0 = time.time()
    policy.train()
    for step in range(steps):
        lr = LR * min(1.0, (step + 1) / max(1, steps // 10))
        for g in opt.param_groups:
            g["lr"] = lr
        stats = []
        for m in range(ACCUM):
            b = rows[(step * ACCUM + m) * MICRO:(step * ACCUM + m + 1) * MICRO]
            seqs, starts = encode(b)
            with torch.no_grad():
                rlp, _, _ = response_logprobs(ref, seqs, starts, tok.pad)
            ctx = ddp.no_sync() if (world > 1 and m < ACCUM - 1) else torch.enable_grad()
            with ctx:
                lp, _, _ = response_logprobs(ddp, seqs, starts, tok.pad)
                loss, rw, rl = dpo_loss(lp[0::2], lp[1::2], rlp[0::2], rlp[1::2])
                scaler.scale(loss / ACCUM).backward()
            stats.append((loss.item(), rw.mean().item(), rl.mean().item(), (rw > rl).float().mean().item(),
                          (lp[0::2] - rlp[0::2]).mean().item(), (lp[1::2] - rlp[1::2]).mean().item()))
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if step % 5 == 0 or step == steps - 1:
            s = [dist.all_mean(sum(v) / len(v)) for v in zip(*stats)]
            log.log(step=step, loss=s[0], reward_chosen=s[1], reward_rejected=s[2], acc=s[3],
                    logp_chosen_delta=s[4], logp_rejected_delta=s[5], margin=s[1] - s[2], grad_norm=float(gn), lr=lr)
            if dist.is_main() and step % 50 == 0:
                p(f"dpo step {step}/{steps} loss {s[0]:.4f} margin {s[1]-s[2]:.3f} acc {s[3]:.2f}")
        if step and step % max(1, steps // 4) == 0:
            log.log(step=step, eval=evaluate())
    after = evaluate()
    if dist.is_main():
        checkpoint.save(os.path.join(out, "final.pt"), policy, policy.cfg, step=steps)
        save_json(os.path.join(out, "summary.json"), {"before": before, "after": after, "beta": BETA, "steps": steps,
                                                       "seconds": time.time() - t0, "energy": meter.stop()})
        p(f"DPO done: {before} -> {after}")
    dist.barrier()
    dist.cleanup()


if __name__ == "__main__":
    main()
