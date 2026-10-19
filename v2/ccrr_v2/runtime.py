"""New-run isolation and immutable bindings. No implicit resume or download."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
PUBLIC_HASHES = {
    "panderm": "be1e0fb108b3bc58721cb5195f136c948160799438f222acf1fd142230ac1ff1",
    "resnet50": "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca",
}
PANDERM_SOURCE = "63fbb03283bb5d61ebd24e4790b66f3495996cc37f000f7003eec0a277bd1ab1"
DATA_BINDING = "cb43f4fb6f180a58ccf372de990e7af7c6bb07a19f80cbb0d560a1364a3e9737"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def validate_resources(resources, hashes=True):
    if set(resources) != {
        "schema",
        "data_root",
        "workspace_root",
        "panderm",
        "resnet50",
    }:
        raise ValueError(
            "Only explicit V2 public base resources allowed; no old run/checkpoint config"
        )
    if resources["schema"] != "ccrr-v2-resources-1":
        raise ValueError("Unknown resource schema")
    for name, expected in PUBLIC_HASHES.items():
        entry = resources[name]
        if not Path(entry["path"]).is_absolute():
            raise ValueError(f"{name}: absolute weight path required")
        if entry.get("sha256") != expected:
            raise ValueError(f"{name}: not the bound public pretrained weight")
        if hashes and digest(entry["path"]) != expected:
            raise ValueError(f"{name}: weight bytes mismatch")
    p = resources["panderm"]
    if not Path(p["source"]).is_absolute():
        raise ValueError("PanDerm source requires an absolute path")
    if p.get("source_sha256") != PANDERM_SOURCE:
        raise ValueError("PanDerm source binding mismatch")
    if hashes and digest(p["source"]) != PANDERM_SOURCE:
        raise ValueError("PanDerm source bytes mismatch")
    data, workspace = (
        Path(resources[k]).resolve() for k in ("data_root", "workspace_root")
    )
    if not all(
        Path(resources[k]).is_absolute() for k in ("data_root", "workspace_root")
    ):
        raise ValueError("Absolute resource paths required")
    if workspace == data or data in workspace.parents or workspace in data.parents:
        raise ValueError("V2 workspace and read-only dataset must be separate")
    if workspace == REPO or REPO in workspace.parents:
        raise ValueError("Run workspace must be outside source checkout")
    manifest = REPO / "src/ccrr/data/frozen_evidence.json"
    if digest(manifest) != DATA_BINDING:
        raise ValueError("Inherited data binding changed")
    if hashes:
        for relative, expected in read(manifest)["files"].items():
            if digest(data / relative) != expected:
                raise ValueError(f"Data evidence mismatch: {relative}")


def new_run(resources, output, stage, config, parents=None):
    output = Path(output).resolve()
    runs = Path(resources["workspace_root"]).resolve() / "runs"
    if runs not in output.parents:
        raise ValueError("Outputs must be below the dedicated V2 workspace/runs")
    if output == REPO or REPO in output.parents:
        raise ValueError("Cannot write experiments inside source")
    # Require a committed checkout, including inherited read-only helpers.
    status = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain"], text=True
    )
    if status.strip():
        raise ValueError("Commit the source before a real experiment")
    commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    os.umask(0o077)
    output.mkdir(parents=True, exist_ok=False)
    write(
        output / "binding.json",
        {
            "schema": "ccrr-v2-run-1",
            "stage": stage,
            "source_commit": commit,
            "resources": resources,
            "config": config,
            "parents": parents or {},
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torchvision", "Pillow", "scipy", "scikit-learn", "timm")
            },
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "data_binding_sha256": DATA_BINDING,
            "old_trained_checkpoints_loaded": False,
            "test_used": False,
        },
    )
    write(
        output / "status.json",
        {"state": "running", "scientific_acceptance": "not_evaluated"},
    )
    return output


def save_model(path, model, step, extra=None):
    obj = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "step": step,
        "extra": extra or {},
    }
    temp = Path(path).with_suffix(".tmp")
    torch.save(obj, temp)
    temp.replace(path)


def completed(output, **extra):
    write(
        Path(output) / "status.json",
        {"state": "completed", "scientific_acceptance": "not_evaluated", **extra},
    )


def device_for(name):
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable; no silent CPU fallback")
    return torch.device(name)
