"""Read-only, digest-bound Stage A dependencies for an independent B run."""

import json
from pathlib import Path

from ccrr.codec.metadata import CLASSES
from ccrr.config import sha256
from ccrr.errors import require


NAMES = ("clinical", "clinical_rgb", "clinical_gray", "mfm_probe", "reid")
SCIENTIFIC_GAPS = (
    "section 7.7 separately trained no-branch control and branch diagnostics",
    "section 26.3 post-adapted training attacker diagnostic",
    "local mask/feature functional validation",
)


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
