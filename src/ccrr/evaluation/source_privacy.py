"""P3 development check: fixed original-image gallery, no sender geometry input."""

from collections import defaultdict
import hashlib
import numpy as np
import torch
from ccrr.data.pad import collate
from ccrr.tensor_ops import gray
from ccrr.evaluation.attacks import ncc
from ccrr.training.losses import ssim_values


@torch.no_grad()
def evaluate_source_privacy(data, device, public_view, models):
    gallery = []
    queries = []
    for i in range(len(data)):
        b = collate([data[i]], device)
        y = gray(b["x"])
        public = public_view(b)
        rgbf = {name: model(b["x"])["embedding"] for name, model in models.items()}
        gallery.append(
            {"id": b["id"][0], "patient": b["patient"][0], "gray": y, "features": rgbf}
        )
        if public is not None:
            queries.append(
                {
                    "id": b["id"][0],
                    "patient": b["patient"][0],
                    "gray": y,
                    "public": public,
                    "gray_features": {
                        name: model(y)["embedding"] for name, model in models.items()
                    },
                    "public_features": {
                        name: model(public)["embedding"]
                        for name, model in models.items()
                    },
                }
            )
    reports = {}
    for kind in ("ncc", "ssim", *models):
        paired = []
        for q in queries:
            values = {}
            for rep in ("gray", "public"):
                scores = []
                for g in gallery:
                    if kind == "ncc":
                        s = float(ncc(q[rep], g["gray"]))
                    elif kind == "ssim":
                        s = float(
                            ssim_values(q[rep], g["gray"], torch.ones_like(q[rep]))[0]
                        )
                    else:
                        s = float(
                            (q[rep + "_features"][kind] * g["features"][kind]).sum()
                        )
                    scores.append(s)
                best = min(
                    range(len(gallery)),
                    key=lambda j: (
                        -scores[j],
                        hashlib.sha256(gallery[j]["id"].encode()).digest(),
                    ),
                )
                values[rep] = {
                    "source": float(gallery[best]["id"] == q["id"]),
                    "patient": float(gallery[best]["patient"] == q["patient"]),
                }
            values.update(
                patient_id=q["patient"],
                source_chance=1 / len(gallery),
                patient_chance=sum(g["patient"] == q["patient"] for g in gallery)
                / len(gallery),
            )
            paired.append(values)
        checks = {}
        for target in ("source", "patient"):
            groups = defaultdict(list)
            for q in paired:
                groups[q["patient_id"]].append(q)
            summary = {
                rep: float(
                    np.mean(
                        [
                            np.mean([q[rep][target] for q in group])
                            for group in groups.values()
                        ]
                    )
                )
                if groups
                else None
                for rep in ("gray", "public")
            }
            chance = (
                float(
                    np.mean(
                        [
                            np.mean([q[target + "_chance"] for q in group])
                            for group in groups.values()
                        ]
                    )
                )
                if groups
                else None
            )
            passed = bool(
                chance is not None
                and summary["gray"] - chance >= 0.1
                and summary["public"] <= chance + 0.1
                and summary["public"] - chance <= 0.5 * (summary["gray"] - chance)
            )
            checks[target] = {**summary, "chance": chance, "passed": passed}
        reports[kind] = checks
    return {
        "attacks": reports,
        "passed": bool(queries)
        and all(c["passed"] for r in reports.values() for c in r.values()),
        "gallery_size": len(gallery),
        "queries": len(queries),
    }
