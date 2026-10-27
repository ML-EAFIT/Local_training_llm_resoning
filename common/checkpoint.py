"""Atomic checkpoint save / resume (model, optimizer, scaler, step, RNG)."""
import os
import random
import numpy as np
import torch

from common.dist import unwrap


def save(path, model, model_cfg, optimizer=None, scaler=None, step=0, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model": unwrap(model).state_dict(),
        "model_cfg": model_cfg.to_dict() if hasattr(model_cfg, "to_dict") else model_cfg,
        "step": step,
        "extra": extra or {},
        "rng": {"py": random.getstate(), "np": np.random.get_state(),
                "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
    }
    if optimizer is not None:
        if hasattr(optimizer, "consolidate_state_dict"):   # ZeRO-1
            optimizer.consolidate_state_dict(to=0)
            if torch.distributed.get_rank() != 0:
                return
        state["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load(path, model=None, optimizer=None, scaler=None, restore_rng=False):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if model is not None:
        sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ck["model"].items()}
        unwrap(model).load_state_dict(sd, strict=False)
    if optimizer is not None and "optimizer" in ck:
        optimizer.load_state_dict(ck["optimizer"])
    if scaler is not None and "scaler" in ck:
        scaler.load_state_dict(ck["scaler"])
    if restore_rng and "rng" in ck:
        random.setstate(ck["rng"]["py"])
        np.random.set_state(ck["rng"]["np"])
        torch.set_rng_state(ck["rng"]["torch"])
    return ck
