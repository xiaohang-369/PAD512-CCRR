from pathlib import Path
import json
import math
import os
import platform
import random
import importlib.metadata
import numpy as np
import torch


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
