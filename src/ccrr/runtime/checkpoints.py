"""Versioned checkpoint serialization; stage resumption is owned by the runner."""

from pathlib import Path
import os
import random
import numpy as np
import torch
from ccrr.errors import require
from ccrr.runtime.support import json_safe


def save_checkpoint(path, modules, opt, step, stage, extra, *, training_state=None):
    # No private keys, plaintext, nonce store or per-image payload in a checkpoint.
    obj = {
        "format": "CCRR-CHECKPOINT-2",
        "models": {k: m.state_dict() for k, m in modules.items()},
        "optimizer": opt.state_dict() if opt else None,
        "step": step,
        "stage": stage,
        "extra": json_safe(extra),
        "python_rng": random.getstate(),
        "numpy_rng": (
            np.random.get_state()[0],
            np.random.get_state()[1].tolist(),
            *np.random.get_state()[2:],
        ),
        "training_state": training_state,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        torch.save(obj, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def load_checkpoint(path, modules, optimizer_instance=None):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    require(
        obj["format"] in ("CCRR-CHECKPOINT-1", "CCRR-CHECKPOINT-2"), "Checkpoint format"
    )
    for k, m in modules.items():
        m.load_state_dict(obj["models"][k], strict=True)
    if optimizer_instance is not None:
        optimizer_instance.load_state_dict(obj["optimizer"])
        torch.set_rng_state(obj["torch_rng"])
        if "python_rng" in obj:
            random.setstate(obj["python_rng"])
            n = obj["numpy_rng"]
            np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
        if torch.cuda.is_available() and obj["cuda_rng"]:
            torch.cuda.set_rng_state_all(obj["cuda_rng"])
    return obj
