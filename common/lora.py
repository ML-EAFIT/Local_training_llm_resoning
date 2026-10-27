"""LoRA: W = W0 + (alpha/r) * B A with W0 frozen (Lecture 3).

Applied to attention AND MLP matrices (Thinking Machines recommendation).
"""
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=32):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))  # zero -> starts equal to W0
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        return self.base(x) + (x.to(self.A.dtype) @ self.A.t() @ self.B.t()).to(x.dtype) * self.scale

    def merged_weight(self):
        return self.base.weight + (self.B @ self.A).to(self.base.weight.dtype) * self.scale


TARGETS = ("wq", "wk", "wv", "wo", "w1", "w2", "w3")


def apply_lora(model, r=16, alpha=32, targets=TARGETS):
    for p in model.parameters():
        p.requires_grad_(False)
    for blk in model.blocks:
        for parent in (blk.attn, blk.ffn):
            for name in targets:
                if hasattr(parent, name) and isinstance(getattr(parent, name), nn.Linear):
                    setattr(parent, name, LoRALinear(getattr(parent, name), r, alpha))
    return model


def merge_lora(model):
    """Fold B A into W0 so the model is a plain GPT again (for saving / inference)."""
    for blk in model.blocks:
        for parent in (blk.attn, blk.ffn):
            for name in TARGETS:
                m = getattr(parent, name, None)
                if isinstance(m, LoRALinear):
                    with torch.no_grad():
                        m.base.weight.copy_(m.merged_weight())
                    m.base.weight.requires_grad_(True)
                    setattr(parent, name, m.base)
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
