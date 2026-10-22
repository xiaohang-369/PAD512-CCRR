"""Exercise offline binding checks with synthetic file fixtures, not model results."""

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/inspect-phase2-resources.py"
spec = importlib.util.spec_from_file_location("phase2_inspection", SCRIPT)
inspection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspection)


def fixture(tmp):
    def put(name, value):
        path = tmp / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def file(name):
        p = put(name, {"synthetic": name})
        return {"path": str(p), "sha256": inspection.digest(p)}

    cp = file("baseline/epoch-120.pt")
    val = {"png_psnr": 1.0, "images": 169}
    put("baseline/status.json", dict(state="completed", mode="train"))
    put(
        "baseline/report.json",
        dict(
            training_completed=True,
            final_epoch=120,
            final_step=10800,
            test_used_for_performance=False,
            baseline_checkpoint={"file": "epoch-120.pt", "sha256": cp["sha256"]},
            final_validation=val,
        ),
    )
    put(
        "baseline/binding.json",
        dict(
            code_commit="fixture",
            config=dict(
                image_size=512,
                global_batch_size=16,
                stages=[dict(epochs=90), dict(epochs=30)],
            ),
        ),
    )
    (tmp / "baseline/metrics.jsonl").write_text(
        "".join(
            json.dumps(
                dict(
                    epoch=e,
                    step=e * 90,
                    author_stage=1 if e <= 90 else 2,
                    validation=val,
                )
            )
            + "\n"
            for e in range(1, 121)
        )
    )
    teacher, source, reid = (
        file("weights/panderm"),
        file("weights/source.py"),
        file("weights/resnet"),
    )
    parent = dict(
        schema="ccrr-v3-resources-1",
        data_root="fixture-data",
        panderm={
            **teacher,
            "source": source["path"],
            "source_sha256": source["sha256"],
        },
        resnet50=reid,
    )
    parent_path = put("parent.json", parent)
    stats = file("features/statistics.pt")
    features = put(
        "features/features.json",
        dict(statistics={"file": "statistics.pt", "sha256": stats["sha256"]}),
    )
    for name, kind, bind in (
        (
            "separator",
            "separator",
            dict(
                resources=parent,
                parents=dict(features_sha256=inspection.digest(features)),
            ),
        ),
        ("targets", "link_targets", dict(source_commit="fixture", parents={})),
    ):
        binding = put(name + "/binding.json", bind)
        put(name + "/status.json", dict(state="completed"))
        files = {}
        if name == "separator":
            checkpoints = []
            for step in (8500, 7000):
                base = f"separator-step-{step:06d}.pt"
                files[base] = file(name + "/" + base)["sha256"]
                checkpoints.append(dict(step=step, file=base))
            put(name + "/checkpoints.json", dict(checkpoints=checkpoints))
        else:
            files["targets.pt"] = file(name + "/targets.pt")["sha256"]
            cap = put(
                name + "/capability.json",
                dict(
                    source="public_ResNet50_initialization",
                    teacher_qualified=False,
                    p3_validated=False,
                ),
            )
            files["capability.json"] = inspection.digest(cap)
        put(
            name + "/artifact.json",
            dict(kind=kind, files=files, binding_sha256=inspection.digest(binding)),
        )
    review = put(
        "review.json",
        dict(
            targets_artifact_sha256=inspection.digest(tmp / "targets/artifact.json"),
            decision="allow_development_reference",
            directions=["rgb_rgb", "gray_rgb", "gray_gray"],
        ),
    )
    return dict(
        schema="ccrr-v4-phase2-inspection-1",
        baseline_run=str(tmp / "baseline"),
        baseline_checkpoint_sha256=cp["sha256"],
        parent_resources=str(parent_path),
        separator=str(tmp / "separator"),
        features=str(tmp / "features"),
        targets=str(tmp / "targets"),
        reference_review=str(review),
        supervisor_steps=[8500, 7000],
    )


def test_bindings_pass_without_modifying_any_parent(tmp_path):
    cfg = fixture(tmp_path)
    before = {str(p): inspection.digest(p) for p in tmp_path.rglob("*") if p.is_file()}
    report = inspection.inspect(cfg)
    assert report["resource_checks_passed"]
    assert report["results"]["identity_reference"]["teacher_qualified"] is False
    assert not report["training_started"] and not report["model_loaded"]
    assert before == {
        str(p): inspection.digest(p) for p in tmp_path.rglob("*") if p.is_file()
    }


def test_wrong_baseline_hash_fails_but_collects_other_checks(tmp_path):
    cfg = fixture(tmp_path)
    Path(cfg["baseline_run"], "epoch-120.pt").write_bytes(b"changed")
    report = inspection.inspect(cfg)
    assert not report["resource_checks_passed"]
    assert "SHA256 mismatch" in report["errors"]["baseline"]
    assert "supervision" in report["results"]


def test_wrong_reference_binding_and_missing_teacher_are_reported(tmp_path):
    cfg = fixture(tmp_path)
    (tmp_path / "weights/panderm").unlink()
    review = inspection.read(cfg["reference_review"])
    review["targets_artifact_sha256"] = "0" * 64
    Path(cfg["reference_review"]).write_text(json.dumps(review))
    report = inspection.inspect(cfg)
    assert set(report["errors"]) == {"supervision", "identity_reference"}
    assert "baseline" in report["results"]
