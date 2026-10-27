"""Distributed helpers: torchrun/NCCL init, DDP wrapping, ZeRO-1, all-reduce utils.

Data parallelism (Lecture old4): each GPU keeps a full model copy, processes a
slice of the batch, and gradients are averaged with an all-reduce. ZeRO-1
shards the Adam states across ranks to save memory.
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def init():
    from common.config import limit_gpu_memory
    r = _init()
    limit_gpu_memory()
    return r


def _init():
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        dist.init_process_group("nccl", device_id=torch.device("cuda", local))
        return dist.get_rank(), local, dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    return 0, 0, 1


def is_dist():
    return dist.is_available() and dist.is_initialized()


def rank():
    return dist.get_rank() if is_dist() else 0


def world():
    return dist.get_world_size() if is_dist() else 1


def is_main():
    return rank() == 0


def barrier():
    if is_dist():
        dist.barrier()


def all_mean(x):
    """Average a python float / tensor across ranks."""
    if not is_dist():
        return float(x) if not torch.is_tensor(x) else x
    t = torch.tensor(float(x), device="cuda") if not torch.is_tensor(x) else x.detach().clone().float()
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t.item() if t.numel() == 1 else t


def all_sum(x):
    if not is_dist():
        return x
    t = torch.tensor(float(x), device="cuda") if not torch.is_tensor(x) else x.detach().clone().float()
    dist.all_reduce(t)
    return t.item() if t.numel() == 1 else t


def wrap(model, local_rank):
    if not is_dist():
        return model
    return DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True, bucket_cap_mb=50)


def unwrap(model):
    m = model.module if hasattr(model, "module") else model
    return getattr(m, "_orig_mod", m)


def make_optimizer(model, lr, wd, betas, zero1=False):
    """AdamW with weight decay only on matrices (not norms / gains)."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]
    fused = torch.cuda.is_available()
    if zero1 and is_dist():
        from torch.distributed.optim import ZeroRedundancyOptimizer
        opt = ZeroRedundancyOptimizer(decay, optimizer_class=torch.optim.AdamW, lr=lr,
                                      betas=betas, weight_decay=wd)
        opt.add_param_group({"params": no_decay, "weight_decay": 0.0})
        return opt
    return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)


def params_checksum(model):
    """Sum of parameters; equal across ranks if DDP keeps replicas in sync."""
    with torch.no_grad():
        return float(sum(p.detach().float().sum() for p in unwrap(model).parameters()))


def cleanup():
    if is_dist():
        dist.destroy_process_group()
