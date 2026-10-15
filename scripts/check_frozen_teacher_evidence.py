"""Read frozen original metrics/intervals and checkpoint bindings; never train A."""

import argparse
import json
import math
from pathlib import Path

from ccrr.config import load_config, sha256
from ccrr.errors import require
from ccrr.evaluation.teachers import clinical_gate, reid_gate
from ccrr.runtime.support import write_json
from ccrr.training.b_resources import verify_teachers


def validate_metrics(accepted):
    clinical = accepted["A-clinical"]["report"]
    mfm = accepted["A-mfm"]["report"]
    reid = accepted["A-reid"]["report"]
    for reports in (clinical["joint"], clinical["controls"], mfm["domains"]):
        require(len(reports) == 2, "Missing RGB/gray evidence")
        for r in reports:
            require(
                len(r["per_class_auroc"]) == len(r["recall"]) == 6
                and r["macro_auroc"] is not None
                and math.isfinite(r["macro_auroc"]),
                "Incomplete/nonfinite clinical metrics",
            )
            require(
                all(
                    v is None or math.isfinite(v)
                    for v in r["per_class_auroc"] + r["recall"]
                ),
                "Nonfinite class metrics",
            )
    cg = clinical_gate(clinical["joint"], clinical["controls"])
    mg = clinical_gate(mfm["domains"], enforce_class_defects=False)
    pg = reid_gate(reid["p1"]["reports"])
    require(cg["passed"], "Clinical original metrics fail: " + str(cg))
    require(mg["passed"] and mfm["interface"]["passed"], "MFM evidence fails")
    require(pg["passed"], "P1 original AUROC/CI evidence fails: " + str(pg))
    p1 = []
    for direction, r in zip(
        ("RGB->RGB", "gray->RGB", "gray->gray"), reid["p1"]["reports"]
    ):
        c = r["cosine"]
        ci = c["auroc_ci"]
        require(
            r.get("threshold_role") == "val_tune"
            and math.isfinite(c["auroc"])
            and len(ci["interval"]) == 2
            and all(math.isfinite(x) for x in ci["interval"])
            and 0 <= ci["interval"][0] <= ci["interval"][1] <= 1,
            "Invalid P1 interval/threshold binding",
        )
        p1.append({"direction": direction, "auroc": c["auroc"], "auroc_ci": ci})
    return {
        "passed": True,
        "clinical": {
            "gate": cg,
            "macro_auroc_rgb_gray": [r["macro_auroc"] for r in clinical["joint"]],
        },
        "mfm": {
            "gate": mg,
            "macro_auroc_rgb_gray": [r["macro_auroc"] for r in mfm["domains"]],
            "interface": mfm["interface"],
        },
        "reid_p1": p1,
        "reid_p2_gate": reid_gate(reid["p2"]["reports"]),
        "scope": "Rechecked original frozen metrics and patient confidence intervals; no teacher retraining or new bootstrap.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    root, record = verify_teachers(load_config(args.config))
    result = validate_metrics(record["acceptance"])
    result.update(
        stage_a_output=str(root),
        freeze_sha256=sha256(root / "teacher_freeze_record.json"),
    )
    require(not Path(args.out).exists(), "Evidence output already exists")
    write_json(Path(args.out), result)
    print(
        "FROZEN_TEACHER_EVIDENCE:", json.dumps(result, ensure_ascii=False), flush=True
    )


if __name__ == "__main__":
    main()
