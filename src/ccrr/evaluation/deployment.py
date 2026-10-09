"""PNG-based deployment checkpoint selection and acceptance evaluation."""

import json
import numpy as np
from ccrr.codec.metadata import CLASSES
from ccrr.tensor_ops import gray
from ccrr.runtime.support import write_json
from ccrr.evaluation.runners import (
    evaluate_classifier,
    evaluate_identity,
    evaluate_system,
)
from ccrr.errors import require


def validate_deployment(experiment, stage, step):
    from dataclasses import replace

    experiment.profile = replace(
        experiment.profile, legal_levels=(1, 2, 3), quality_order=(1, 2, 3)
    )
    experiment.system.profile = experiment.profile
    experiment.bind_snapshot(f"{stage}-{step}-levels")
    reports = {
        r: _validate_deployment_once(experiment, f"{stage}-r{r}", step, (r,))
        for r in (1, 2, 3)
    }
    legal = tuple(r for r, report in reports.items() if report["passed"])
    if not legal:
        diagnostic = max(reports.values(), key=lambda r: (r["both_rate"], r["psnr"]))
        return {
            **diagnostic,
            "passed": False,
            "level_reports": reports,
            "reason": "No admissible deployment level",
        }
    actual = {}
    for r in legal:
        path = (
            experiment.output
            / "validation"
            / f"{stage}-r{r}-{step}-actual"
            / "system_results.json"
        )
        rows = json.loads(path.read_text())["rows"]
        actual[r] = {v["id"]: v for v in rows if v["c_success"]}
    common = set.intersection(*(set(v) for v in actual.values()))
    require(bool(common), "No common successful cohort for quality ordering")
    order = tuple(
        sorted(
            legal,
            key=lambda r: (
                -np.mean([actual[r][i]["quality"]["psnr"] for i in common]),
                -np.mean([actual[r][i]["quality"]["ssim"] for i in common]),
                np.mean([actual[r][i]["c_bytes"] for i in common]),
                r,
            ),
        )
    )
    experiment.profile = replace(
        experiment.profile, legal_levels=legal, quality_order=order
    )
    experiment.system.profile = experiment.profile
    experiment.bind_snapshot(f"{stage}-{step}-policy")
    report = _validate_deployment_once(experiment, stage, step, order)
    report["level_reports"] = reports
    report["legal_levels"] = list(legal)
    report["quality_order"] = list(order)
    report["ordering_common_n"] = len(common)
    return report


def _validate_deployment_once(experiment, stage, step, level_order=None):
    out = experiment.output / "validation" / f"{stage}-{step}-actual"
    result = evaluate_system(
        experiment.system,
        experiment.data["val_select"],
        experiment.device,
        experiment.allocator,
        out,
        lambda b: experiment.risks[b["id"][0]],
        lambda b: experiment.policies[b["id"][0]],
        level_order=level_order,
    )
    rows = result["rows"]
    pub = [r for r in rows if r["published"]]
    cs = [r for r in rows if r["c_success"]]
    both = sum(r["both_success"] for r in rows) / len(rows)
    coverage = len(pub) / len(rows)
    c_cond = len(cs) / len(pub) if pub else 0
    m_cond = sum(r["m_success"] for r in pub) / len(pub) if pub else 0
    psnr = float(np.mean([r["quality"]["psnr"] for r in cs])) if cs else 0
    ssim = float(np.mean([r["quality"]["ssim"] for r in cs])) if cs else 0
    # Generator selection uses only the development clinical teacher / attacks.
    from ccrr.runtime.system import load_public

    def public_view(b):
        p = out / b["id"][0]
        return (
            load_public(p, experiment.profile.size).to(experiment.device)
            if p.exists()
            else None
        )

    clin = (
        evaluate_classifier(
            experiment.clinical,
            experiment.data["val_select"],
            experiment.device,
            public_view,
            True,
        )
        if pub
        else None
    )
    identity = {}
    privacy_checks = {}
    from ccrr.evaluation.comparisons import paired_clinical, privacy_points

    public_ids = {r["id"] for r in pub}
    reference = evaluate_classifier(
        experiment.clinical,
        experiment.data["val_select"],
        experiment.device,
        None,
        False,
    )
    clinical_check = (
        paired_clinical(reference, clin) if clin else {"passed_point": False}
    )
    if pub:
        for scenario in ("p1", "p2"):
            for direction in ("original", "same"):
                q = public_view
                gallery = public_view if direction == "same" else None
                common_gallery = public_ids if direction == "same" else None
                current = evaluate_identity(
                    experiment.pub_attacker,
                    experiment.data["val_select"],
                    experiment.device,
                    scenario,
                    q,
                    gallery,
                    True,
                    public_ids,
                    common_gallery,
                )
                rgb = evaluate_identity(
                    experiment.pub_attacker,
                    experiment.data["val_select"],
                    experiment.device,
                    scenario,
                    None,
                    None,
                    True,
                    public_ids,
                    common_gallery,
                )
                gy = lambda b: gray(b["x"])
                standard = evaluate_identity(
                    experiment.pub_attacker,
                    experiment.data["val_select"],
                    experiment.device,
                    scenario,
                    gy,
                    gy if direction == "same" else None,
                    True,
                    public_ids,
                    common_gallery,
                )
                key = scenario + "-" + direction
                identity[key] = current
                privacy_checks[key] = privacy_points(rgb, standard, current)
    report = {
        "stage": stage,
        "step": step,
        "actual_results_path": str(out / "system_results.json"),
        "profile": experiment.profile.__dict__,
        "both_rate": both,
        "coverage": coverage,
        "c_conditional": c_cond,
        "m_conditional": m_cond,
        "psnr": psnr,
        "ssim": ssim,
        "mean_bpp": float(np.mean([r["protected_bpp"] for r in pub]))
        if pub
        else float("inf"),
        "clinical": clin,
        "clinical_check": clinical_check,
        "identity": identity,
        "privacy_checks": privacy_checks,
        "system": result["funnel"],
        "class_coverage": {
            label: {
                "publication": np.mean(
                    [r["published"] for r in rows if r["diagnostic"] == label]
                ),
                "both": np.mean(
                    [r["both_success"] for r in rows if r["diagnostic"] == label]
                ),
            }
            for label in CLASSES
        },
        "both_conditional": sum(r["both_success"] for r in pub) / len(pub)
        if pub
        else 0,
        "c_all": len(cs) / len(rows),
        "m_all": sum(r["m_success"] for r in rows) / len(rows),
        "fraction_psnr28_ssim90": np.mean(
            [r["quality"]["psnr"] >= 28 and r["quality"]["ssim"] >= 0.90 for r in cs]
        )
        if cs
        else 0,
    }
    from ccrr.evaluation.source_privacy import evaluate_source_privacy

    report["source_privacy"] = evaluate_source_privacy(
        experiment.data["val_select"],
        experiment.device,
        public_view,
        {"reid": experiment.reid, "adaptive": experiment.pub_attacker},
    )
    # Scientific utility/privacy acceptance is computed by evidence rules, not training loss.
    from ccrr.evaluation.gates import deployment_gate

    report.update(deployment_gate(report, experiment.acceptance))
    write_json(out / "selection.json", report)
    return report
