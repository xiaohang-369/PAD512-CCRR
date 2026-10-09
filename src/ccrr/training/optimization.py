"""Optimizer construction, learning-rate schedule and guarded updates."""

import math
import torch
from ccrr.errors import require


def optimizer(modules, lrs):
    groups = []
    seen = set()
    for module, lr in zip(modules, lrs):
        decay = []
        no_decay = []
        for name, p in module.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            (
                no_decay
                if p.ndim < 2
                or name.endswith(".B")
                or name == "B"
                or module.__class__.__name__ == "EntropyModel"
                else decay
            ).append(p)
        if decay:
            groups.append(
                {"params": decay, "lr": lr, "lr_max": lr, "weight_decay": 1e-4}
            )
        if no_decay:
            groups.append(
                {"params": no_decay, "lr": lr, "lr_max": lr, "weight_decay": 0}
            )
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)


def schedule(opt, step, total):
    w = min(500, math.ceil(0.05 * total))
    t = step + 1
    factor = (
        0.01 + 0.99 * (t - 1) / max(1, w - 1)
        if t <= w
        else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (t - w) / max(1, total - w)))
    )
    for group in opt.param_groups:
        group["lr"] = group["lr_max"] * factor


def update(opt):
    params = [
        p for group in opt.param_groups for p in group["params"] if p.grad is not None
    ]
    require(bool(params), "No gradients to update")
    for p in params:
        if not torch.isfinite(p.grad).all():
            raise FloatingPointError("Nonfinite gradient; optimizer not stepped")
    norm = torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True)
    opt.step()
    return float(norm)
