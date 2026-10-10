"""Private, content-bound freeze record after A gates (§6.8)."""

from pathlib import Path
import torch
from ccrr.codec.metadata import CLASSES
from ccrr.config import sha256
from ccrr.runtime.support import environment, write_json
from ccrr.runtime.preflight import COMMIT


def write_freeze_record(experiment):
    e = experiment
    artifacts = {}
    for name in ("clinical", "clinical_rgb", "clinical_gray", "mfm_probe", "reid"):
        path = e.output / "checkpoints" / (name + ".pt")
        obj = torch.load(path, map_location="cpu", weights_only=True)
        artifacts[name] = {
            "sha256": sha256(path),
            "selected_step": obj["step"],
            "selection": obj["extra"],
        }
    report = {
        "version": "PAD512-CCRR-v1.1",
        "seed": e.seed,
        "environment": environment(),
        "architecture": {
            "clinical": "torchvision EfficientNet-B0 IMAGENET1K_V1; 6-class head",
            "mfm": "PanDerm-Base patch16/224; norm after last block; CLS excluded; 14x14x768",
            "reid": "torchvision ResNet-50 IMAGENET1K_V2; 2048 embedding; fit-only identity head",
        },
        "input": "512 fixed preprocessing; valid-rectangle adapter per design section 6.5; FP32",
        "base_resources": e.cfg["weights"],
        "panderm_source_commit": COMMIT,
        "interface_evidence": e.cfg.get("interface_reports"),
        "checkpoints": artifacts,
        "class_order": CLASSES,
        "fit_patient_mapping": e.fit.pid,
        "data_binding_sha256": sha256(
            Path(__file__).parents[1] / "data/frozen_evidence.json"
        ),
        "acceptance": e.acceptance,
        "allowed_roles": [
            "Clinical supervision",
            "frozen MFM patch supervision",
            "P1 identity reference",
        ],
        "final_system_accepted": False,
        "three_seed_study_complete": False,
        "checkpoint_key_changes": "New downstream heads; official loading differences recorded in interface evidence",
    }
    write_json(e.output / "teacher_freeze_record.json", report)
    return report
