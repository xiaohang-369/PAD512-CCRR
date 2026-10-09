"""External-domain data contract. Patient identity is never inferred from file names."""

from pathlib import Path
import json
from ccrr.data.pad import preprocess
from ccrr.errors import require


class ExternalDataset:
    def __init__(self, manifest, size=512):
        self.path = Path(manifest)
        self.spec = json.loads(self.path.read_text())
        self.size = size
        require(
            self.spec["domain"] in ("dermatology", "retina", "endoscopy"),
            "Unregistered medical domain",
        )
        require(
            self.spec["transfer"] in ("zero_shot", "target_finetune", "target_retrain"),
            "Explicit transfer level required",
        )
        require(
            self.spec.get("license_audited") is True,
            "External data license audit missing",
        )
        require(
            self.spec.get("task_mapping_audited") is True,
            "External task-label mapping not verified",
        )
        self.rows = self.spec["images"]
        require(
            len({r["id"] for r in self.rows}) == len(self.rows), "Repeated external ID"
        )
        if self.spec["transfer"] == "zero_shot":
            require(
                not self.spec.get("target_tuning_used", False),
                "Target tuning invalidates zero-shot claim",
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x, v, rect, legal, digest = preprocess(self.path.parent / r["path"], self.size)
        return {
            "x": x,
            "v": v,
            "rect": rect,
            "legal": legal,
            "id": r["id"],
            "label": r.get("label"),
            "patient": r.get("patient_id"),
            "digest": digest,
        }

    def claim_scope(self):
        return {
            "domain": self.spec["domain"],
            "transfer": self.spec["transfer"],
            "patient_privacy_eligible": self.spec.get("patient_labels_verified", False),
            "metadata_profile": "PAD-M1 is not transferable; a new source schema/profile must be registered",
            "no_cross_specialty_claim_from_dermatology_only": True,
        }
