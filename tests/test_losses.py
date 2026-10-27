import math
import math, random
import torch
from common.data import best_fit_pack, collate_sft


def test_sft_mask_only_assistant():
    ids = [1, 4, 10, 11, 7, 5, 20, 21, 7]
    mask = [0, 0, 0, 0, 0, 0, 1, 1, 1]
    X, Y = collate_sft([(ids, mask)], 2, 64)
    assert Y[0].tolist() == [-100, -100, -100, -100, -100, 20, 21, 7]
    assert X[0].tolist() == ids[:-1]


def test_best_fit_pack():
    docs = [[1] * 5, [2] * 3, [3] * 7, [4] * 2, [5] * 12]
    rows, mask = best_fit_pack(docs, 8, 0)
    assert mask.sum() == 29 and rows.shape[1] == 8
    assert rows.shape[0] == 4   # 29 tokens -> optimal 4 bins of 8


def test_reverse_kl_nonneg():
    s = torch.randn(5, 11).log_softmax(-1); t = torch.randn(5, 11).log_softmax(-1)
    kl = (s.exp() * (s - t)).sum(-1)
    assert (kl >= -1e-6).all()


def _post():
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "7_post_training"))


def test_dpo_loss_log2_at_reference():
    _post()
    from dpo import dpo_loss
    lp = torch.tensor([-12.0, -30.0])
    loss, rw, rl = dpo_loss(lp[:1], lp[1:], lp[:1], lp[1:])
    assert abs(loss.item() - math.log(2)) < 1e-6
    loss2, _, _ = dpo_loss(lp[:1] + 1, lp[1:] - 1, lp[:1], lp[1:])   # chosen up, rejected down -> lower loss
    assert loss2 < loss


def test_grpo_advantages_and_clip():
    _post()
    from grpo import grpo_advantages, grpo_token_loss
    r = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.1, 0.1, 0.1, 0.1])
    a = grpo_advantages(r, 4)
    assert torch.allclose(a.view(2, 4).mean(1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(a[4:], torch.zeros(4))          # zero-variance group -> no signal
    lp = torch.zeros(2, 3, requires_grad=True)
    loss, st = grpo_token_loss(lp, torch.zeros(2, 3), torch.zeros(2, 3), torch.tensor([1.0, -1.0]), torch.ones(2, 3))
    assert abs(st["kl"]) < 1e-7 and st["clip_frac"] == 0
    loss.backward()
    assert lp.grad[0].sum() < 0 and lp.grad[1].sum() > 0  # descent raises positive-advantage tokens
    # beyond the upper clip bound a positive advantage gives no gradient
    lp2 = torch.full((1, 2), 0.5, requires_grad=True)
    l2, st2 = grpo_token_loss(lp2, torch.zeros(1, 2), torch.full((1, 2), 0.5), torch.tensor([1.0]), torch.ones(1, 2))
    l2.backward()
    assert st2["clip_frac"] == 1.0 and lp2.grad.abs().max() < 1e-6


def test_math_reward():
    _post()
    import random as _r
    import math_tasks
    for lvl in range(1, 6):
        pb = math_tasks.make_problem(_r.Random(lvl), lvl)
        assert math_tasks.reward(math_tasks.solution_text(pb), pb) == (1.1, 1.0)
        assert math_tasks.reward("The answer is 999999.", pb)[1] == 0.0
