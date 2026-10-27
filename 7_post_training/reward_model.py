"""Stage 7b — reward model (Bradley-Terry, Lecture 3/5).

P(y_w > y_l) = sigma(r(x, y_w) - r(x, y_l));   L = -E[log sigma(r_w - r_l)]
Backbone: the SFT model; head: scalar linear layer on the last token's hidden state.
Used to evaluate DPO / GRPO outputs, for Best-of-N, and as a cheap judge.
Run: torchrun --nproc_per_node=2 7_post_training/reward_model.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from common import dist  # noqa: E402
from common.config import POST_DIR, SMOKE  # noqa: E402
from common.data import Tok  # noqa: E402
from common.energy import EnergyMeter, NullMeter  # noqa: E402
from common.logger import Logger, save_json, p  # noqa: E402
from common.model import GPT, load_model  # noqa: E402
from common.config import ModelConfig  # noqa: E402
from post_common import read_jsonl, pair_ids, pad_batch, ckpt, shuffled  # noqa: E402

MAX_LEN = 384 if SMOKE else 768
N_PAIRS = 800 if SMOKE else 16000
MICRO = 4
LR = 1e-5 if not SMOKE else 1e-4


class RewardModel(nn.Module):
    def __init__(self, backbone: GPT):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.cfg.d_model, 1)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, last):
        h = self.backbone.features(x)
        return self.head(h[torch.arange(len(x), device=x.device), last].float()).squeeze(-1)


def load_rm(path=None):
    path = path or os.path.join(POST_DIR, "reward_model", "final.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    rm = RewardModel(GPT(ModelConfig(**ck["model_cfg"])))
    rm.load_state_dict(ck["model"])
    return rm.cuda().eval()


@torch.no_grad()
def score(rm, tok, prompts, responses, bs=8):
    out = []
    for i in range(0, len(prompts), bs):
        seqs = [pair_ids(tok, pr, rs, MAX_LEN, MAX_LEN // 2)[0] for pr, rs in zip(prompts[i:i + bs], responses[i:i + bs])]
        x = pad_batch(seqs, tok.pad)
        last = torch.tensor([len(s) - 1 for s in seqs], device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            out += rm(x, last).float().tolist()
    return out


def batch_of(tok, rows):
    seqs, last = [], []
    for r in rows:
        for resp in (r["chosen"], r["rejected"]):
            ids, _ = pair_ids(tok, r["prompt"], resp, MAX_LEN, MAX_LEN // 2)
            seqs.append(ids)
            last.append(len(ids) - 1)
    return pad_batch(seqs, tok.pad), torch.tensor(last, device="cuda")


def main():
    rank, local, world = dist.init()
    torch.manual_seed(3 + rank)
    out = os.path.join(POST_DIR, "reward_model")
    os.makedirs(out, exist_ok=True)
    tok = Tok()
    backbone, _ = load_model(ckpt("sft"))
    backbone.cfg.grad_ckpt = not SMOKE
    rm = RewardModel(backbone).cuda()
    ddp = dist.wrap(rm, local)
    opt = torch.optim.AdamW(rm.parameters(), lr=LR, weight_decay=0.0, fused=True)
    scaler = torch.amp.GradScaler("cuda")
    train_rows = shuffled(read_jsonl("pref_train.jsonl"), 0)[:N_PAIRS]
    test_rows = read_jsonl("pref_test.jsonl")[: 200 if SMOKE else 1000]
    mine = train_rows[rank::world]
    steps = len(mine) // MICRO
    log = Logger(out, enabled=dist.is_main())
    meter = EnergyMeter(out).start() if dist.is_main() else NullMeter()
    t0 = time.time()

    @torch.no_grad()
    def evaluate():
        rm.eval()
        hit = n = 0
        margins = []
        for i in range(rank * MICRO, len(test_rows), MICRO * world):
            x, last = batch_of(tok, test_rows[i:i + MICRO])
            with torch.autocast("cuda", dtype=torch.float16):
                r = rm(x, last).float().view(-1, 2)
            hit += (r[:, 0] > r[:, 1]).sum().item(); n += len(r)
            margins += (r[:, 0] - r[:, 1]).tolist()
        rm.train()
        return {"pairwise_acc": dist.all_sum(hit) / max(dist.all_sum(n), 1),
                "mean_margin": dist.all_sum(sum(margins)) / max(dist.all_sum(len(margins)), 1)}

    before = evaluate()
    if dist.is_main():
        p(f"reward model: {steps} steps; test acc before training {before['pairwise_acc']:.3f}")
    rm.train()
    for step in range(steps):
        lr = LR * min(1.0, (step + 1) / max(1, steps // 20)) * (0.1 + 0.9 * (1 - step / steps))
        for g in opt.param_groups:
            g["lr"] = lr
        x, last = batch_of(tok, mine[step * MICRO:(step + 1) * MICRO])
        with torch.autocast("cuda", dtype=torch.float16):
            r = ddp(x, last).float().view(-1, 2)
        loss = -F.logsigmoid(r[:, 0] - r[:, 1]).mean()     # Bradley-Terry
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if step % 20 == 0:
            acc = (r[:, 0] > r[:, 1]).float().mean().item()
            log.log(step=step, loss=dist.all_mean(loss.item()), acc=dist.all_mean(acc), lr=lr)
        if step % max(1, steps // 5) == 0 and step:
            ev = evaluate()
            log.log(step=step, eval=ev)
            if dist.is_main():
                p(f"rm step {step}/{steps} loss {loss.item():.3f} test acc {ev['pairwise_acc']:.3f}")
    after = evaluate()
    if dist.is_main():
        torch.save({"model": rm.state_dict(), "model_cfg": backbone.cfg.to_dict()}, os.path.join(out, "final.pt"))
        save_json(os.path.join(out, "summary.json"), {"before": before, "after": after, "steps": steps,
                                                       "pairs": len(train_rows), "seconds": time.time() - t0,
                                                       "energy": meter.stop()})
        p(f"reward model test pairwise accuracy {before['pairwise_acc']:.3f} -> {after['pairwise_acc']:.3f}")
    dist.barrier()
    dist.cleanup()


if __name__ == "__main__":
    main()
