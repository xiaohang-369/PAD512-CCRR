"""Read-only CPU inventory for V4 phase 2; standard library, no model loading.

Only the explicitly requested new report is written. Existing models, source,
configs, data and run artifacts are never modified. No patient rows are emitted.
"""

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked(path, expected, evidence):
    path = Path(path)
    actual = digest(path)
    require(actual == expected, f"SHA256 mismatch: {path}")
    evidence[str(path)] = actual
    return actual


def artifact(root, kind, evidence):
    root = Path(root)
    manifest, binding = read(root / "artifact.json"), read(root / "binding.json")
    require(read(root / "status.json")["state"] == "completed", f"Incomplete: {root}")
    require(manifest["kind"] == kind, f"Wrong artifact kind: {root}")
    checked(root / "binding.json", manifest["binding_sha256"], evidence)
    evidence[str(root / "artifact.json")] = digest(root / "artifact.json")
    return manifest, binding


def inspect_baseline(cfg, evidence):
    run = Path(cfg["baseline_run"])
    status, report, binding = (
        read(run / n) for n in ("status.json", "report.json", "binding.json")
    )
    require(
        status["state"] == "completed" and status["mode"] == "train",
        "Baseline not completed",
    )
    require(
        report["training_completed"]
        and report["final_epoch"] == 120
        and report["final_step"] == 10800,
        "Baseline endpoint differs",
    )
    require(report["test_used_for_performance"] is False, "Baseline used test")
    config = binding["config"]
    require(
        config["image_size"] == 512 and config["global_batch_size"] == 16,
        "Baseline shape/batch differs",
    )
    require(
        [s["epochs"] for s in config["stages"]] == [90, 30], "Baseline schedule differs"
    )
    cp = report["baseline_checkpoint"]
    require(
        cp["file"] == "epoch-120.pt"
        and cp["sha256"] == cfg["baseline_checkpoint_sha256"],
        "Unregistered baseline checkpoint",
    )
    checked(run / cp["file"], cfg["baseline_checkpoint_sha256"], evidence)
    rows = [
        json.loads(s)
        for s in (run / "metrics.jsonl").read_text().splitlines()
        if s.strip()
    ]
    require(
        [r["epoch"] for r in rows] == list(range(1, 121)), "Missing/duplicate epochs"
    )
    require(
        all(
            r["step"] == r["epoch"] * 90
            and r["author_stage"] == (1 if r["epoch"] <= 90 else 2)
            for r in rows
        ),
        "Step/stage mismatch",
    )
    require(
        rows[-1]["validation"] == report["final_validation"],
        "Final metrics/report mismatch",
    )
    for name in ("status.json", "report.json", "binding.json", "metrics.jsonl"):
        evidence[str(run / name)] = digest(run / name)
    return dict(
        checkpoint=str(run / cp["file"]),
        sha256=cp["sha256"],
        epochs=120,
        steps=10800,
        source_commit=binding["code_commit"],
        final_validation=report["final_validation"],
    )


def inspect_supervision(cfg, evidence):
    parent = read(cfg["parent_resources"])
    require(parent["schema"] == "ccrr-v3-resources-1", "Wrong parent resource schema")
    evidence[cfg["parent_resources"]] = digest(cfg["parent_resources"])
    teacher = parent["panderm"]
    checked(teacher["path"], teacher["sha256"], evidence)
    checked(teacher["source"], teacher["source_sha256"], evidence)
    run, features = Path(cfg["separator"]), Path(cfg["features"])
    manifest, binding = artifact(run, "separator", evidence)
    require(binding["resources"] == parent, "Supervisor resource binding differs")
    checked(features / "features.json", binding["parents"]["features_sha256"], evidence)
    fm = read(features / "features.json")
    checked(features / fm["statistics"]["file"], fm["statistics"]["sha256"], evidence)
    index = read(run / "checkpoints.json")["checkpoints"]
    steps = []
    for step in cfg["supervisor_steps"]:
        name = f"separator-step-{step:06d}.pt"
        matches = [x for x in index if x["step"] == step and x["file"] == name]
        require(len(matches) == 1, f"Unregistered supervisor step: {step}")
        checked(run / name, manifest["files"][name], evidence)
        steps.append(
            dict(step=step, checkpoint=str(run / name), sha256=manifest["files"][name])
        )
    return dict(
        steps=steps,
        primary_step=8500,
        comparison_step=7000,
        data_root=parent["data_root"],
        panderm=teacher,
        pixel_identity_suppression_implemented=False,
        existing_pixel_interface="clinical_logits_and_semantic_features",
    )


def inspect_identity_reference(cfg, evidence):
    root = Path(cfg["targets"])
    manifest, binding = artifact(root, "link_targets", evidence)
    for name in ("capability.json", "targets.pt"):
        checked(root / name, manifest["files"][name], evidence)
    review = read(cfg["reference_review"])
    require(
        review["targets_artifact_sha256"] == digest(root / "artifact.json"),
        "Reference review/targets mismatch",
    )
    require(
        review["decision"] == "allow_development_reference",
        "Reference not admitted for development",
    )
    require(
        set(review["directions"]) == {"rgb_rgb", "gray_rgb", "gray_gray"},
        "Reference directions differ",
    )
    evidence[cfg["reference_review"]] = digest(cfg["reference_review"])
    capability = read(root / "capability.json")
    parent = read(cfg["parent_resources"])
    weight = parent["resnet50"]
    checked(weight["path"], weight["sha256"], evidence)
    return dict(
        source=capability["source"],
        teacher_qualified=capability.get("teacher_qualified"),
        p3_validated=capability.get("p3_validated"),
        review_decision=review["decision"],
        source_commit=binding.get("source_commit"),
        reference=binding.get("parents", {}).get("reference"),
        scope="existing_P2_same_lesion_linkage_reference_not_general_patient_identification",
    )


def inspect(cfg):
    require(cfg["schema"] == "ccrr-v4-phase2-inspection-1", "Wrong inspection config")
    require(
        cfg["supervisor_steps"] == [8500, 7000], "Preserve selected supervisor steps"
    )
    evidence, results, errors = {}, {}, {}
    for name, fn in (
        ("baseline", inspect_baseline),
        ("supervision", inspect_supervision),
        ("identity_reference", inspect_identity_reference),
    ):
        print(f"INSPECT {name}", flush=True)
        try:
            results[name] = fn(cfg, evidence)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
    return dict(
        schema="ccrr-v4-phase2-inspection-report-1",
        resource_checks_passed=not errors,
        results=results,
        errors=errors,
        checked_files=evidence,
        training_started=False,
        model_loaded=False,
        gpu_used=False,
        phase2_identity_or_medical_effect_evaluated=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(
        not args.output.exists(),
        "Use a new report path; existing reports are preserved",
    )
    report = inspect(read(args.config))
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"PHASE2_INSPECTION_REPORT={args.output}", flush=True)
    raise SystemExit(0 if report["resource_checks_passed"] else 2)


if __name__ == "__main__":
    main()
