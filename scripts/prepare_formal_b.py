"""Prepare one new formal run from completed micro16 evidence; no training."""

import argparse
import json
import math
from pathlib import Path

import yaml

from ccrr.config import load_config, sha256
from ccrr.errors import require
from ccrr.runtime.support import write_json
from ccrr.training.b_initialization import CANDIDATE
from ccrr.training.b_resources import execution_options
from check_frozen_teacher_evidence import validate_metrics
from ccrr.training.b_resources import verify_teachers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--profile-run", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out).resolve()
    require(not (out / "startup-input.yaml").exists(), "Preparation already exists")
    root = Path(args.profile_run) / "result/purifier16"
    report = json.loads((root / "report.json").read_text())
    row = report["cases"]["16"]
    require(
        report["effective_image_batch"] == report["effective_pair_batch"] == 16
        and row["microbatch"] == 16
        and row["accumulation"] == 1
        and row["sampling_equal"] is True
        and row["one_step_numerical_screen"] is True
        and math.isfinite(row["seconds_per_step"])
        and row["seconds_per_step"] > 0,
        "Completed micro16 numerical/timing evidence missing",
    )
    repeats = row["measurement"]["repeats"]
    require(
        len(repeats) == 3
        and all(r["restore_and_continuation_exact"] is True for r in repeats),
        "Micro16 recovery checks incomplete",
    )
    profile = root / "micro-16/profile.json"
    segments = json.loads(profile.read_text())
    require(segments["activity"]["cuda_kernels_recorded"], "CUDA profile missing")
    write_json(
        out / "micro16-evidence.json",
        {
            "source": str(root),
            "source_report_sha256": sha256(root / "report.json"),
            "profile_sha256": sha256(profile),
            "completed_micro16": {k: v for k, v in row.items() if k != "measurement"},
            "other_microbatches_required": False,
            "phases": segments["phases"],
        },
    )
    cfg = load_config(args.config)
    cfg.update(
        output=str(out / "future-output"),
        resume_a=False,
        resume_b=False,
        b_initialization=CANDIDATE,
        microbatch=16,
        batch_teacher=16,
        batch_pairs=16,
        b_diagnostic=False,
        device="cuda",
        b_cache_device="cuda",
    )
    execution_options(cfg)  # Refuse a changed original 10000-step budget.
    require(cfg["seed"] == 20261008, "Original registered seed changed")
    aroot, frozen = verify_teachers(cfg)
    evidence = validate_metrics(frozen["acceptance"])
    evidence.update(
        freeze_sha256=sha256(aroot / "teacher_freeze_record.json"),
        stage_a_output=str(aroot),
    )
    write_json(out / "teacher-evidence.json", evidence)
    print(
        "FROZEN_TEACHER_EVIDENCE:", json.dumps(evidence, ensure_ascii=False), flush=True
    )
    # Existing host checker takes an A-output config and creates a separate B config.
    cfg["output"] = str(aroot)
    cfg.pop("b_execution_report", None)
    (out / "startup-input.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("MICRO16_ACCEPTED seconds_per_step=", row["seconds_per_step"], flush=True)
    print("INITIALIZATION:", CANDIDATE, "fresh 10000; no R3 warm start", flush=True)


if __name__ == "__main__":
    main()
