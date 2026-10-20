from copy import deepcopy

import pytest
import torch

from ccrr_v3 import cli, data, runtime
from conftest import cohort


@pytest.fixture
def a0(tmp_path):
    root = tmp_path / "data"
    role_dir = root / "protocol_audit_20261009/roles"
    role_dir.mkdir(parents=True)
    cache = tmp_path / "A0"
    cache.mkdir()
    resources = dict(
        schema="ccrr-v3-resources-1",
        data_root=str(root),
        workspace_root=str(tmp_path / "PAD512-CCRR-V3"),
        panderm={"path": "synthetic-no-weight"},
        resnet50={"path": "synthetic-no-weight"},
    )
    runtime.write(
        cache / "binding.json",
        dict(
            stage="features", resources={**resources, "schema": "ccrr-v2-resources-1"}
        ),
    )
    runtime.write(cache / "status.json", {"state": "completed"})
    manifest = dict(
        schema="ccrr-v2-features-1",
        binding=runtime.digest(cache / "binding.json"),
        roles={},
    )
    for role in (*runtime.ROLES, "calibration", "test"):
        obj = cohort(role, 768)
        runtime.write(role_dir / f"{role}.json", obj["rows"])
        if role not in runtime.ROLES:
            continue
        obj["features"] = torch.randn(len(obj["rows"]), 2, 768, 14, 14)
        obj["coverage"] = torch.ones(len(obj["rows"]), 1, 14, 14)
        # Deliberately no P1 pool/protocol.
        runtime.save(cache / f"{role}.pt", obj)
        manifest["roles"][role] = dict(
            file=f"{role}.pt", sha256=runtime.digest(cache / f"{role}.pt")
        )
    runtime.save(cache / "stats.pt", dict(mean=torch.zeros(768), std=torch.ones(768)))
    manifest["statistics"] = dict(
        file="stats.pt", sha256=runtime.digest(cache / "stats.pt")
    )
    runtime.write(cache / "features.json", manifest)
    return cache, resources


def test_load_only_requested_roles_and_never_test_pixels(a0, monkeypatch):
    path, resources = a0
    observed = []
    original = torch.load

    def wrapped(p, **kwargs):
        observed.append(str(p))
        return original(p, **kwargs)

    monkeypatch.setattr(torch, "load", wrapped)
    loaded, _ = data.load_features(path, resources, ("fit", "val_tune"))
    assert set(loaded) == {"fit", "val_tune"}
    assert all("val_select.pt" not in p and "test.pt" not in p for p in observed)
    with pytest.raises(ValueError, match="test/calibration"):
        data.load_features(path, resources, ("test",))


def test_cache_tamper_and_patient_overlap_fail(a0):
    path, resources = a0
    fit = path / "fit.pt"
    original = fit.read_bytes()
    fit.write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes"):
        data.load_features(path, resources, ("fit",))
    fit.write_bytes(original)
    role_dir = (
        __import__("pathlib").Path(resources["data_root"])
        / "protocol_audit_20261009/roles"
    )
    rows = runtime.read(role_dir / "test.json")
    rows[0]["patient_id"] = "fit-p0"
    runtime.write(role_dir / "test.json", rows)
    with pytest.raises(ValueError, match="overlap"):
        data.load_features(path, resources)


def test_cli_probe_then_independent_compression_artifact(
    cfg, a0, tmp_path, monkeypatch
):
    path, resources = a0
    # Only external base weight/admission checks and git cleanliness are mocked.
    # Real A0 file hashing, tensor validation, loops, artifacts and parent checks run.
    monkeypatch.setattr(cli, "validate_resources", lambda _: None)
    monkeypatch.setattr(
        runtime.subprocess,
        "check_output",
        lambda args, **kw: "" if "status" in args else "a" * 40,
    )
    r, c = tmp_path / "resources.json", tmp_path / "config.json"
    runtime.write(r, resources)
    runtime.write(c, cfg)
    root = __import__("pathlib").Path(resources["workspace_root"]) / "runs"
    probe, train = root / "probe", root / "compression"
    base = [
        "--resources",
        str(r),
        "--config",
        str(c),
        "--features",
        str(path),
        "--device",
        "cpu",
    ]
    hashes = {p.name: runtime.digest(p) for p in path.iterdir()}
    cli.main(["probe", *base, "--output", str(probe)])
    runtime.artifact(
        probe, "clinical_heads", resources, runtime.digest(path / "features.json")
    )
    cli.main(
        [
            "train",
            *base,
            "--heads",
            str(probe),
            "--variant",
            "compression",
            "--output",
            str(train),
        ]
    )
    manifest, _ = runtime.artifact(train, "separator", resources)
    assert manifest["supervision_admitted"] is False
    assert (
        runtime.read(train / "status.json")["scientific_acceptance"] == "not_evaluated"
    )
    assert {p.name: runtime.digest(p) for p in path.iterdir()} == hashes
    with pytest.raises(FileExistsError):
        cli.main(
            [
                "train",
                *base,
                "--heads",
                str(probe),
                "--variant",
                "compression",
                "--output",
                str(train),
            ]
        )
    wrong = deepcopy(resources)
    wrong["panderm"] = {"path": "other"}
    with pytest.raises(ValueError, match="resource"):
        data.load_features(path, wrong)
    targets = root / "synthetic_targets"
    runtime.new_run(
        resources,
        targets,
        "targets",
        cfg,
        {"features_sha256": runtime.digest(path / "features.json")},
        {},
    )
    loaded, _ = data.load_features(path, resources)
    payload = {
        role: {
            "ids": [row["img_id"] for row in obj["rows"]],
            "embeddings": torch.randn(len(obj["rows"]), 2, 6),
        }
        for role, obj in loaded.items()
    }
    runtime.save(targets / "targets.pt", payload)
    runtime.write(targets / "capability.json", {"synthetic_only": True})
    runtime.finish(targets, "link_targets", ["targets.pt", "capability.json"])
    review = tmp_path / "review.json"
    runtime.write(
        review,
        dict(
            schema="ccrr-v3-reference-review-1",
            targets_artifact_sha256=runtime.digest(targets / "artifact.json"),
            decision="allow_development_reference",
            directions=["rgb_rgb", "gray_rgb", "gray_gray"],
            reason="Synthetic integration only, no real qualification",
        ),
    )
    sep = root / "separation"
    cli.main(
        [
            "train",
            *base,
            "--heads",
            str(probe),
            "--variant",
            "separation",
            "--targets",
            str(targets),
            "--reference-review",
            str(review),
            "--output",
            str(sep),
        ]
    )
    runtime.artifact(sep, "separator", resources)
    assert {p.name: runtime.digest(p) for p in path.iterdir()} == hashes
