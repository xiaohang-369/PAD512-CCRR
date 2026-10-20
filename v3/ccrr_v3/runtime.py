"""V3-only outputs, exact parent bindings, explicit development registration."""

import json
import importlib.metadata
import math
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

import torch

from ccrr_v2.runtime import digest, read, write, seed_all, device_for  # noqa: F401
from ccrr_v2.runtime import validate_resources as validate_base

REPO = Path(__file__).resolve().parents[2]
ROLES = ("fit", "val_tune", "val_select")


def validate_config(cfg, registered=False):
    template = read(REPO / "v3/configs/development.example.json")
    if set(cfg) != set(template) or cfg["schema"] != template["schema"]:
        raise ValueError("Unknown/missing V3 configuration keys")
    if cfg["scope"] != "development_only" or cfg["registration_status"] not in (
        "draft",
        "registered",
    ):
        raise ValueError("V3 supports registered development, not formal acceptance")
    if not isinstance(cfg["registration_note"], str):
        raise ValueError("Registration note must be text")
    if registered and (
        cfg["registration_status"] != "registered"
        or not cfg["registration_note"].strip()
    ):
        raise ValueError(
            "Register finite budgets and assumptions explicitly before execution"
        )
    for k in (
        "seed",
        "warmup_steps",
        "joint_steps",
        "probe_steps",
        "attack_steps",
        "attack_warmup_steps",
        "reid_steps",
        "batch_size",
        "evaluate_every",
        "width",
        "code_channels",
        "bootstrap_repetitions",
    ):
        if type(cfg[k]) is not int or cfg[k] < (0 if k == "seed" else 1):
            raise ValueError(f"Invalid integer: {k}")
    if cfg["batch_size"] != 16:
        raise ValueError("First V3 development protocol uses batch=16")
    for k in (
        "lr",
        "probe_lr",
        "attack_lr",
        "reid_lr",
        "temperature_start",
        "temperature_end",
        "min_scale",
        "min_probability",
        "gradient_clip",
    ):
        if type(cfg[k]) not in (float, int) or not math.isfinite(cfg[k]) or cfg[k] <= 0:
            raise ValueError(f"Invalid positive coefficient: {k}")
    if cfg["min_probability"] >= 1 or cfg["temperature_end"] > cfg["temperature_start"]:
        raise ValueError("Invalid entropy bound or temperature schedule")
    if set(cfg["weights"]) != {"all", "link", "rate", "medical", "confusion"}:
        raise ValueError("Unknown loss weights")
    for k, v in cfg["weights"].items():
        if type(v) not in (float, int) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"Positive declared loss weight required: {k}")
    if cfg["head_mode"] not in ("frozen", "joint"):
        raise ValueError("Head-mode comparison must be explicitly registered")


def validate_resources(resources, hashes=True):
    if resources.get("schema") != "ccrr-v3-resources-1":
        raise ValueError("Require an explicit V3 resource config")
    base = {**resources, "schema": "ccrr-v2-resources-1"}
    validate_base(base, hashes=hashes)
    root = Path(resources["workspace_root"]).resolve()
    if root.name != "PAD512-CCRR-V3":
        raise ValueError("Use a separate PAD512-CCRR-V3 workspace; never V2 outputs")


def log(output, event, **values):
    record = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **values,
    }
    with (Path(output) / "events.jsonl").open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)


def new_run(resources, output, stage, config, parents, options):
    target = Path(output)
    if not target.is_absolute():
        raise ValueError("Absolute output path required")
    target = target.resolve()
    runs = Path(resources["workspace_root"]).resolve() / "runs"
    if runs not in target.parents or REPO in target.parents:
        raise ValueError("Output must be under V3 workspace/runs, outside source")
    if subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("Commit all source before a real development run")
    commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    os.umask(0o077)
    target.mkdir(parents=True, exist_ok=False)
    write(
        target / "binding.json",
        dict(
            schema="ccrr-v3-run-1",
            source_commit=commit,
            stage=stage,
            resources=resources,
            config=config,
            parents=parents,
            options=options,
            torch=str(torch.__version__),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            packages={
                k: importlib.metadata.version(k)
                for k in (
                    "numpy",
                    "torchvision",
                    "timm",
                    "scikit-learn",
                    "scipy",
                    "Pillow",
                )
            },
            cuda_runtime=torch.version.cuda,
            cudnn_version=torch.backends.cudnn.version(),
            test_images_used=False,
            scientific_acceptance="not_evaluated",
        ),
    )
    status(target, "running")
    log(target, "started", stage=stage, commit=commit, scope="development_only")
    return target


def status(output, state, **extra):
    write(
        Path(output) / "status.json",
        dict(state=state, scientific_acceptance="not_evaluated", **extra),
    )


def save(path, obj):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(obj, temp)
    temp.replace(path)


def finish(output, kind, payload_names, **extra):
    output = Path(output)
    write(
        output / "artifact.json",
        dict(
            schema="ccrr-v3-artifact-1",
            kind=kind,
            binding_sha256=digest(output / "binding.json"),
            files={name: digest(output / name) for name in payload_names},
            **extra,
        ),
    )
    status(output, "completed", artifact="artifact.json", scope="development_only")
    log(output, "completed", kind=kind, scientific_acceptance="not_evaluated")


def artifact(path, kind=None, resources=None, feature_sha=None):
    path = Path(path).resolve()
    manifest = read(path / "artifact.json")
    binding = read(path / "binding.json")
    if (
        manifest["schema"] != "ccrr-v3-artifact-1"
        or binding["schema"] != "ccrr-v3-run-1"
    ):
        raise ValueError("Not a bound V3 artifact")
    if read(path / "status.json")["state"] != "completed":
        raise ValueError("Parent did not complete")
    if manifest["binding_sha256"] != digest(path / "binding.json"):
        raise ValueError("Parent binding changed")
    if kind is not None and manifest["kind"] != kind:
        raise ValueError("Wrong parent artifact kind")
    if resources is not None and binding["resources"] != resources:
        raise ValueError("Parent resource mismatch")
    if (
        feature_sha is not None
        and binding["parents"].get("features_sha256") != feature_sha
    ):
        raise ValueError("Parent used different PanDerm features")
    for name, expected in manifest["files"].items():
        p = (path / name).resolve()
        if p.parent != path or digest(p) != expected:
            raise ValueError("Parent path escape or content mismatch")
    return manifest, binding
