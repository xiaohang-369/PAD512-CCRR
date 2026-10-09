from pathlib import Path
import json
import math
import os
import platform
import random
import importlib.metadata
import numpy as np
import torch
from .errors import require


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def environment():
    packages = {}
    for name in (
        "torch",
        "torchvision",
        "numpy",
        "Pillow",
        "cryptography",
        "timm",
        "scipy",
        "scikit-learn",
        "scikit-image",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda": torch.version.cuda,
        "devices": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
    }


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return (
            "+inf"
            if value == float("inf")
            else "-inf"
            if value == float("-inf")
            else None
        )
    return value


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(json_safe(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    os.replace(tmp, path)


def log_json(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(json_safe(obj), ensure_ascii=False, allow_nan=False) + "\n")


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


def save_checkpoint(path, modules, opt, step, stage, extra):
    # No private keys, plaintext, nonce store or per-image payload in a checkpoint.
    obj = {
        "format": "CCRR-CHECKPOINT-1",
        "models": {k: m.state_dict() for k, m in modules.items()},
        "optimizer": opt.state_dict() if opt else None,
        "step": step,
        "stage": stage,
        "extra": json_safe(extra),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, modules, optimizer_instance=None):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    require(obj["format"] == "CCRR-CHECKPOINT-1", "Checkpoint format")
    for k, m in modules.items():
        m.load_state_dict(obj["models"][k], strict=True)
    if optimizer_instance is not None:
        optimizer_instance.load_state_dict(obj["optimizer"])
        torch.set_rng_state(obj["torch_rng"])
        if torch.cuda.is_available() and obj["cuda_rng"]:
            torch.cuda.set_rng_state_all(obj["cuda_rng"])
    return obj
