"""Verified epoch-boundary continuation; never overwrite a parent run."""

from pathlib import Path
import torch

from .phase1.optim import ContinuousBatches
from .verify import read, digest


def load_resume(cfg, ctx):
    spec = cfg.get("resume")
    if spec is None:
        return None
    root = Path(spec["run"]).resolve()
    paths = [
        root / name
        for name in (
            "status.json",
            "report.json",
            "binding.json",
            "config.json",
            "preflight-binding.json",
        )
    ]
    status, report, binding, original, preflight_binding = map(read, paths)
    if (
        status["state"] != "completed"
        or not report["training_completed"]
        or not report["frozen_models_unchanged"]
        or not report["parents_unchanged"]
    ):
        raise ValueError("Resume requires a completed, unchanged parent run")
    if cfg["update_modules"] != ["E1", "D1"]:
        raise ValueError("This continuation is for the joint E1/D1 experiment")
    expected = {k: v for k, v in cfg.items() if k not in ("resume", "epochs")}
    old = {k: v for k, v in original.items() if k != "epochs"}
    if expected != old or binding["config"] != original or report["binding"] != binding:
        raise ValueError("Resume may only extend epochs; keep original training config")
    if binding["parents"] != ctx["parents"]:
        raise ValueError("Resume data/resources differ from the parent run")
    final = report["final_checkpoint"]
    if Path(final["file"]).name != final["file"]:
        raise ValueError("Invalid checkpoint filename")
    checkpoint_path = root / final["file"]
    if digest(checkpoint_path) != spec["sha256"] or final["sha256"] != spec["sha256"]:
        raise ValueError("Resume checkpoint checksum mismatch")
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    steps_per_epoch = (len(ctx["arrays"]["fit"]) + 15) // 16
    if (
        ck["schema"] != "ccrr-v4-g1-checkpoint-1"
        or ck["binding"] != binding
        or ck["epoch"] != final["epoch"]
        or ck["epoch"] != original["epochs"]
        or ck["step"] != final["step"]
        or ck["step"] != ck["epoch"] * steps_per_epoch
        or not ck["epoch"] < cfg["epochs"]
    ):
        raise ValueError("Resume checkpoint position/binding mismatch")
    groups = ck["optimizer"]["param_groups"]
    if (
        len(groups) != 1
        or groups[0]["updates"] != ck["step"]
        or groups[0]["lr"] != cfg["learning_rate"]
        or tuple(groups[0]["betas"]) != (0.9, 0.999)
        or groups[0]["eps"] != 1e-8
    ):
        raise ValueError("Resume optimizer does not match the original TF1 Adam")
    if set(ck["optimizer"]["state"]) != set(groups[0]["params"]):
        raise ValueError("Resume optimizer moments are incomplete")
    for state in ck["optimizer"]["state"].values():
        if set(state) != {"m", "v"} or not all(
            torch.isfinite(t).all() for t in state.values()
        ):
            raise ValueError("Missing/nonfinite Adam moments")
    if not all(torch.isfinite(t).all() for t in ck["model"].values()):
        raise ValueError("Nonfinite resume model")
    # Retain the ORIGINAL G0 medical targets and RGB bank. A new bank built
    # from epoch-30 G1 without these targets would change the objective.
    preflight = Path(preflight_binding["path"])
    if digest(preflight) != preflight_binding["sha256"]:
        raise ValueError("Original preflight report changed")
    initial = read(preflight)
    if not initial["passed"] or initial["binding"] != binding:
        raise ValueError("Original fit bank belongs to another experiment")
    if Path(initial["bank"]["file"]).name != initial["bank"]["file"]:
        raise ValueError("Invalid fit bank filename")
    bank_path = preflight.parent / initial["bank"]["file"]
    if digest(bank_path) != initial["bank"]["sha256"]:
        raise ValueError("Original fit bank checksum mismatch")
    paths += [checkpoint_path, preflight, bank_path]
    for path in paths:
        ctx["parents"][str(path)] = digest(path)
    ctx["model"].load_state_dict(ck["model"], strict=True)
    return dict(
        checkpoint=ck,
        bank_path=bank_path,
        evidence=dict(
            parent_run=str(root),
            checkpoint=str(checkpoint_path),
            sha256=spec["sha256"],
            start_epoch=ck["epoch"],
            start_step=ck["step"],
            stream_position=ck["step"] * 16,
            rng_state_saved_in_parent=False,
            bitwise_uninterrupted_equivalence_claimed=False,
        ),
    )


def restore_training_state(optimizer, cfg, count, resume):
    start_epoch, step = 0, 0
    if resume is not None:
        ck = resume["checkpoint"]
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch, step = ck["epoch"], ck["step"]
    return (
        start_epoch,
        step,
        ContinuousBatches(count, 16, cfg["seed"], position=step * 16),
    )
