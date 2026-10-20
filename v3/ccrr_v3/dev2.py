"""Explicit DEV2 registration, one training run or one val-tune audit per invocation."""

import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import warnings

from . import cli
from .runtime import REPO, artifact, digest, read, validate_config, write

ROOT = Path("/data/zwc/zyh/PAD512-CCRR-V3")
OLD = ROOT / "plans/separation-dev1-20261020"
PLAN = ROOT / "plans/separation-dev2-20261020"
FEATURES = Path("/data/zwc/zyh/PAD512-CCRR-V2/runs/A0-features/A0-features.txSIuAiL")
HEADS = ROOT / "runs/probe/probe-dev1.20261020-104540.2929847"
TARGETS = ROOT / "runs/targets/p2-public.20261020-123545.2940957"
STEPS = list(range(250, 3001, 250)) + list(range(3500, 10001, 500))


def configs(old_train, old_audit):
    train = {
        **old_train, "joint_steps": 9500, "temperature_decay_steps": 2000,
        "checkpoint_steps": STEPS,
        "registration_note": (
            "DEV2 fresh initialization, seed20261020, batch16 with replacement; "
            "same sampled positions in RGB/gray. Preserve DEV1 architecture/loss/lr/"
            "500 warmup and 2000-step temperature decay, then hold tau=.5. "
            "Maximum total10000, snapshots250 through3000 then500. "
            "No automatic best step, no resume, no automatic extension. "
            "Operator stop only at saved checkpoints; paired control must match budget. "
            "Both compression and separation have the same protocol. "
            "All old DEV1 artifacts remain unchanged."
        ),
    }
    audit = {
        **old_audit,
        "registration_note": (
            "DEV2 checkpoint screening on val_tune only. Fresh independent Linear/MLP "
            "each2000 updates, global/spatial attackers each3000, same seed/lr/batch16/"
            "selection rules as DEV1. All 44 metrics enter observed Pareto screening, "
            "no medical threshold or winner; at most2 diversity representatives per "
            "variant displayed while retaining the full frontier. "
            "No automatic val_select evaluation; a finite shortlist must be fixed first. "
            "val_select already used for DEV1 development; test remains sealed."
        ),
    }
    for cfg in (train, audit):
        validate_config(cfg, registered=True)
    return train, audit


def current_commit():
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise ValueError("Commit the release before registering or running DEV2")
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def check_hashes(mapping):
    for name, value in mapping.items():
        if digest(name) != value:
            raise ValueError(f"Registered input changed: {name}")


def register():
    prior = read(OLD / "registration.json")
    if prior["state"] != "registered":
        raise ValueError("DEV1 registration is required")
    check_hashes({str(OLD / k): v for k, v in prior["files"].items()})
    check_hashes(prior["parents"])
    train, audit = configs(read(OLD / "train.json"), read(OLD / "audit.json"))
    resources = read(ROOT / "private/resources.json")
    feature_sha = digest(FEATURES / "features.json")
    for path, kind in ((HEADS, "clinical_heads"), (TARGETS, "link_targets")):
        artifact(path, kind, resources, feature_sha)
    cli.validate_reference_review(OLD / "reference-review.json", TARGETS)
    commit = current_commit()
    PLAN.mkdir(parents=True, exist_ok=False)
    for name, obj in (("train.json", train), ("audit.json", audit),
                      ("reference-review.json", read(OLD / "reference-review.json"))):
        write(PLAN / name, obj)
    write(PLAN / "registration.json", dict(
        state="registered", source_commit=commit,
        files={name: digest(PLAN / name) for name in ("train.json", "audit.json", "reference-review.json")},
        parents={**prior["parents"], str(OLD / "registration.json"): digest(OLD / "registration.json")},
        maximum_steps=10000, checkpoint_steps=STEPS, checkpoints_per_variant=len(STEPS),
        selection_role="val_tune", final_winner=None, clinical_minimum=None,
        training_started=False, scientific_acceptance="not_evaluated",
    ))
    print("DEV2_REGISTERED:", PLAN, "training_started=False")


def run(args):
    registration = read(PLAN / "registration.json")
    if current_commit() != registration["source_commit"]:
        raise ValueError("Use the registered DEV2 release")
    check_hashes({str(PLAN / k): v for k, v in registration["files"].items()})
    check_hashes(registration["parents"])
    label = f"dev2-{args.variant}" if args.action == "train" else f"dev2-{args.variant}-step{args.step}-tune"
    stage = "train" if args.action == "train" else "audit"
    output = ROOT / "runs" / stage / f"{label}.{datetime.now():%Y%m%d-%H%M%S}.{os.getpid()}"
    command = [
        stage, "--resources", str(ROOT / "private/resources.json"),
        "--config", str(PLAN / ("train.json" if stage == "train" else "audit.json")),
        "--features", str(FEATURES), "--output", str(output), "--device", "cuda:0",
    ]
    if stage == "train":
        command += ["--heads", str(HEADS), "--variant", args.variant]
        if args.variant == "separation":
            command += ["--targets", str(TARGETS), "--reference-review", str(PLAN / "reference-review.json")]
    else:
        if args.step not in STEPS:
            raise ValueError("Step is outside the registered save schedule")
        manifest, binding = artifact(args.separator, "separator")
        if (manifest["variant"] != args.variant
                or binding["source_commit"] != registration["source_commit"]
                or binding["config"] != read(PLAN / "train.json")):
            raise ValueError("Not the registered DEV2 variant/training configuration")
        command += ["--separator", args.separator, "--separator-step", str(args.step),
                    "--view", "full" if args.variant == "compression" else "semantic",
                    "--evaluation-role", "val_tune"]
    print("RUN:", output, flush=True)
    # Existing display-only policy, not a change to bootstrap or numerical results.
    from sklearn.exceptions import UndefinedMetricWarning
    text = "No positive samples in y_true, true positive value should be meaningless"
    original, count = warnings.showwarning, 0

    def counted(message, category, filename, lineno, file=None, line=None):
        nonlocal count
        if (issubclass(category, UndefinedMetricWarning) and str(message) == text
                and str(filename).replace("\\", "/").endswith("/sklearn/metrics/_ranking.py")):
            count += 1
            if count > 1:
                return
        original(message, category, filename, lineno, file=file, line=line)

    with warnings.catch_warnings():
        warnings.filterwarnings("always", message="^"+text+"$", category=UndefinedMetricWarning,
                                module=r"^sklearn\.metrics\._ranking$")
        warnings.showwarning = counted
        try:
            cli.main(command)
        finally:
            warnings.showwarning = original
            print("Known missing-positive warnings:", count, "shown:", min(count, 1))
    check_hashes(registration["parents"])
    print("DEV2_COMPLETED:", output, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("register")
    train = sub.add_parser("train")
    train.add_argument("variant", choices=("compression", "separation"))
    audit = sub.add_parser("audit")
    audit.add_argument("variant", choices=("compression", "separation"))
    audit.add_argument("--separator", required=True)
    audit.add_argument("--step", required=True, type=int)
    args = p.parse_args(argv)
    if sys.prefix != "/home/zwc/anaconda3/envs/ccrr_v2":
        raise ValueError("Use the registered ccrr_v2 interpreter")
    register() if args.action == "register" else run(args)


if __name__ == "__main__":
    main()
