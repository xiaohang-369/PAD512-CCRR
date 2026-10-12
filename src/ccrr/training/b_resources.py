"""Read-only, digest-bound Stage A dependencies for an independent B run."""

import json
from pathlib import Path

from ccrr.codec.metadata import CLASSES
from ccrr.config import sha256
from ccrr.errors import require


NAMES = ("clinical", "clinical_rgb", "clinical_gray", "mfm_probe", "reid")
# Implementation availability only; trained scientific acceptance is a separate gate.
SCIENTIFIC_GAPS = ()


def verify_execution_report(cfg):
    from ccrr.training.b_state import binding

    entry = cfg.get("b_execution_report", {})
    path = Path(entry.get("path", ""))
    require(
        path.is_file() and sha256(path) == entry.get("sha256"),
        "B execution report missing/digest mismatch",
    )
    r = json.loads(path.read_text())
    checker = Path(__file__).resolve().parents[3] / "scripts/check_stage_b_host.py"
    require(
        checker.is_file() and r.get("checker_sha256") == sha256(checker),
        "B checker source mismatch",
    )
    devices = r.get("environment", {}).get("devices", [])
    require(len(devices) == 1 and "H100" in devices[0], "B report requires one H100")
    require(
        r.get("version") == "CCRR-B-ENGINEERING-2"
        and r.get("engineering_passed") is True
        and r.get("formal_training_started") is False,
        "B execution report version/status mismatch",
    )
    require(
        r.get("stage_a_freeze_sha256") == cfg["stage_a_source"]["freeze_sha256"],
        "B report teacher mismatch",
    )
    rows = [v for v in r["variants"] if v["microbatch"] == cfg.get("microbatch", 1)]
    require(len(rows) == 1, "B report microbatch mismatch")
    v = rows[0]
    require(
        v.get("binding") == binding({**cfg, "b_diagnostic": True}),
        "B report source/config/environment mismatch",
    )
    require(
        v.get("complete_resume_state_match") is True
        and v.get("frozen_teachers_unchanged") is True
        and v.get("functional", {}).get("passed") is True
        and v.get("control_replay", {}).get("complete_resume_state_match") is True,
        "B report incomplete execution coverage",
    )
    require(
        v.get("audit_probe", {}).get("representations") == ["raw", "control", "sem"]
        and v["audit_probe"]
        .get("post_adaptation_probe", {})
        .get("original_attacker_unchanged")
        is True,
        "B audit/control/adaptation coverage missing",
    )
    return True


def verify_teachers(cfg):
    entry = cfg.get("stage_a_source", {})
    require(
        entry.get("output") and entry.get("freeze_sha256"),
        "B requires stage_a_source.output and freeze_sha256",
    )
    root = Path(entry["output"]).resolve()
    out = Path(cfg["output"]).resolve()
    require(
        root != out and root not in out.parents and out not in root.parents,
        "B output must be separate from Stage A (no parent/child overlap)",
    )
    record = root / "teacher_freeze_record.json"
    require(
        record.is_file() and sha256(record) == entry["freeze_sha256"],
        "Stage A freeze record digest mismatch",
    )
    r = json.loads(record.read_text())
    accepted = json.loads((root / "acceptance.json").read_text())
    require(
        r["version"] == "PAD512-CCRR-v1.1" and r["seed"] == cfg["seed"],
        "Stage A protocol/seed mismatch",
    )
    require(
        r["acceptance"] == accepted
        and all(
            accepted.get(n, {}).get("passed") is True
            for n in ("A-clinical", "A-mfm", "A-reid")
        ),
        "Stage A gates not frozen/passed",
    )
    require(list(r["class_order"]) == list(CLASSES), "Teacher class order mismatch")
    require(
        r["data_binding_sha256"]
        == sha256(Path(__file__).parents[1] / "data/frozen_evidence.json"),
        "Teacher data binding mismatch",
    )
    for name in ("efficientnet_b0", "resnet50", "panderm"):
        for key in (
            ("sha256", "source_sha256", "source_commit")
            if name == "panderm"
            else ("sha256",)
        ):
            require(
                r["base_resources"][name].get(key) == cfg["weights"][name].get(key),
                "Teacher base resource mismatch: " + name + "/" + key,
            )
    for name in NAMES:
        path = root / "checkpoints" / (name + ".pt")
        require(
            path.is_file() and sha256(path) == r["checkpoints"][name]["sha256"],
            "Frozen teacher checkpoint digest mismatch: " + name,
        )
        budget = 2000 if name == "mfm_probe" else 10000 if name == "reid" else 5000
        require(
            r["checkpoints"][name]["selection"]["budget_completed"] == budget,
            "Incomplete teacher budget: " + name,
        )
    return root, r


def execution_options(cfg):
    n = cfg.get("microbatch", 1)
    require(type(n) is int and n in (1, 2, 4, 8, 16), "B microbatch must be 1/2/4/8/16")
    require(
        cfg.get("batch_teacher", 16) == 16 and cfg.get("batch_pairs", 16) == 16,
        "B effective image/pair batches must remain 16",
    )
    require(cfg["budgets"]["purifier"] == 10000, "B budget must remain 10000")
    require(
        cfg.get("b_cache_device", "cpu") in ("cpu", "cuda"),
        "B cache device must be cpu/cuda",
    )
    require(
        type(cfg.get("b_cache_bytes", 4 * 1024**3)) is int
        and cfg.get("b_cache_bytes", 4 * 1024**3) > 0,
        "B cache budget must be positive",
    )
    require(
        type(cfg.get("b_audit_cache_bytes", 8 * 1024**3)) is int
        and cfg.get("b_audit_cache_bytes", 8 * 1024**3) > 0,
        "B audit cache budget must be positive",
    )
    require(
        type(cfg.get("resume_interval", 100)) is int
        and 1 <= cfg.get("resume_interval", 100) <= 1000,
        "Invalid B save interval",
    )
    return n
