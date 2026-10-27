"""Shared training loop (pre-training, mid-training, SFT).

- DDP over NCCL, gradient accumulation with no_sync (one all-reduce per step)
- fp16 autocast + GradScaler (Turing has no bf16), master weights fp32
- WSD / cosine / constant LR schedules with warmup
- gradient clipping, per-component gradient norms, skipped-step counting
- throughput, MFU, memory, energy (rank 0) logging
- periodic atomic checkpoints + exact resume
"""
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, asdict

import torch

from common import dist, checkpoint
from common.flops import flops_per_token, mfu
from common.logger import Logger, save_json, p


@dataclass
class TrainArgs:
    steps: int
    accum: int
    tokens_per_step: int
    lr: float = 1e-3
    min_lr_frac: float = 0.1
    warmup: int = 100
    decay_frac: float = 0.15
    schedule: str = "wsd"           # wsd | cosine | constant
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    log_every: int = 10
    eval_every: int = 250
    ckpt_minutes: float = 30.0
    snapshot_fracs: tuple = ()      # save extra checkpoints at these fractions of training
    compile: bool = False
    zero1: bool = False
    seq_len: int = 1024


def lr_at(step, a: TrainArgs):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    lo = a.lr * a.min_lr_frac
    if a.schedule == "constant":
        return a.lr
    if a.schedule == "cosine":
        t = (step - a.warmup) / max(1, a.steps - a.warmup)
        return lo + 0.5 * (a.lr - lo) * (1 + math.cos(math.pi * min(t, 1.0)))
    decay_start = int(a.steps * (1 - a.decay_frac))   # WSD: warmup -> stable -> linear decay
    if step < decay_start:
        return a.lr
    t = (step - decay_start) / max(1, a.steps - decay_start)
    return a.lr + (lo - a.lr) * min(t, 1.0)


def component_of(name):
    if "tok_emb" in name or "pos_emb" in name or "lm_head" in name:
        return "embedding"
    if ".attn." in name:
        return "attention"
    if ".ffn." in name:
        return "ffn"
    if "n1" in name or "n2" in name or "norm" in name:
        return "norm"
    return "other"


def component_grad_norms(model):
    sq = {}
    for n, prm in dist.unwrap(model).named_parameters():
        if prm.grad is None:
            continue
        c = component_of(n)
        sq[c] = sq.get(c, 0.0) + prm.grad.detach().float().pow(2).sum().item()
    return {k: math.sqrt(v) for k, v in sq.items()}


def train(model, model_cfg, a: TrainArgs, get_batch, out_dir, eval_fn=None, meter=None,
          state_obj=None, ckpt_name="ckpt.pt", log_name="metrics", trainable=None):
    """get_batch() -> (x, y) or (x, y, loss_mask). eval_fn(model, step) -> dict (called on all ranks)."""
    rank, world = dist.rank(), dist.world()
    local = torch.cuda.current_device()
    main = rank == 0
    raw = model
    ck_path = os.path.join(out_dir, ckpt_name)
    opt = dist.make_optimizer(raw, a.lr, a.weight_decay, (a.beta1, a.beta2), a.zero1)
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 14)
    start_step = 0
    if os.path.exists(ck_path):
        ck = checkpoint.load(ck_path, raw, opt, scaler)
        start_step = ck["step"]
        if state_obj is not None and "loader" in ck["extra"]:
            state_obj.load_state(ck["extra"]["loader"])
        if main:
            p(f"resumed from {ck_path} at step {start_step}")
    fwd = torch.compile(raw) if a.compile else raw
    ddp = dist.wrap(fwd, local) if world > 1 else fwd
    log = Logger(out_dir, log_name, enabled=main, append=start_step > 0)
    n_ne = raw.num_params(non_embedding=True)
    fpt = flops_per_token(model_cfg, a.seq_len, n_ne)
    params = trainable or [q for q in raw.parameters() if q.requires_grad]
    last_ck = time.time()
    snap_steps = {int(f * a.steps): f for f in a.snapshot_fracs}
    skipped = 0
    t_last, tok_since = time.time(), 0
    loss_hist = []
    raw.train()
    torch.cuda.reset_peak_memory_stats()
    for step in range(start_step, a.steps):
        lr = lr_at(step, a)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_acc = 0.0
        for micro in range(a.accum):
            b = get_batch()
            x, y = b[0], b[1]
            mask = b[2] if len(b) > 2 else None
            sync = nullcontext() if (world == 1 or micro == a.accum - 1) else ddp.no_sync()
            with sync:
                with torch.autocast("cuda", dtype=torch.float16):
                    _, loss = ddp(x, y, loss_mask=mask)
                scaler.scale(loss / a.accum).backward()
            loss_acc += loss.detach()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(params, a.grad_clip)
        comp = component_grad_norms(raw) if (main and step % a.eval_every == 0) else None
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if scaler.get_scale() < scale_before:
            skipped += 1
        opt.zero_grad(set_to_none=True)
        tok_since += a.tokens_per_step
        if meter is not None:
            meter.add_tokens(a.tokens_per_step)
        loss_v = None
        if step % a.log_every == 0 or step == a.steps - 1:
            loss_v = dist.all_mean(loss_acc / a.accum)
            torch.cuda.synchronize()
            dt = time.time() - t_last
            tps = tok_since / dt
            t_last, tok_since = time.time(), 0
            row = dict(step=step, tokens=(step + 1) * a.tokens_per_step, loss=loss_v, lr=lr,
                       grad_norm=float(gnorm), loss_scale=scaler.get_scale(), skipped=skipped,
                       tok_per_s=tps, mfu=mfu(tps, fpt, world),
                       mem_gb=torch.cuda.max_memory_allocated() / 2**30)
            if comp:
                row["grad_norm_components"] = comp
            if main and meter is not None and hasattr(meter, "summary"):
                e = meter.summary()
                row["kwh"] = e.get("total_kwh")
                row["gpu_w"] = e.get("avg_gpu_w")
            log.log(**row)
            loss_hist.append(loss_v)
            if main and step % (a.log_every * 10) == 0:
                p(f"step {step}/{a.steps} loss {loss_v:.4f} lr {lr:.2e} gn {float(gnorm):.2f} "
                  f"{tps/1e3:.1f}k tok/s mfu {row['mfu']*100:.1f}% mem {row['mem_gb']:.2f}G")
        if eval_fn is not None and (step % a.eval_every == 0 or step == a.steps - 1) and step > 0:
            raw.eval()
            ev = eval_fn(raw, step)
            raw.train()
            if main:
                log.log(step=step, tokens=(step + 1) * a.tokens_per_step, eval=ev)
        if step + 1 in snap_steps:
            frac = snap_steps[step + 1]
            if main:
                checkpoint.save(os.path.join(out_dir, f"snap_{int(frac*100):03d}.pt"), raw, model_cfg, step=step + 1)
        want_ck = time.time() - last_ck > a.ckpt_minutes * 60
        want_ck = dist.all_sum(float(want_ck)) > 0 if world > 1 else want_ck
        if want_ck:
            extra = {"loader": state_obj.state()} if state_obj is not None else {}
            if main or a.zero1:
                checkpoint.save(ck_path, raw, model_cfg, opt, scaler, step + 1, extra)
            last_ck = time.time()
            dist.barrier()
    summary = dict(steps=a.steps, tokens=a.steps * a.tokens_per_step, skipped_steps=skipped,
                   final_train_loss=loss_hist[-1] if loss_hist else None,
                   flops=fpt * a.steps * a.tokens_per_step, flops_per_token=fpt,
                   params=raw.num_params(), params_nonemb=n_ne, peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30,
                   args=asdict(a))
    if os.path.exists(ck_path) and main:
        os.remove(ck_path)   # training finished: the final model is saved by the caller
    log.close()
    return summary
