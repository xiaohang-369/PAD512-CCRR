"""Validation of recovery and semantic utility during C–E selection."""

from ccrr.codec.quantization import dequantize_latent


from ccrr.codec.quantization import quantize_latent

import numpy as np
import torch
from ccrr.data.pad import collate
from ccrr.tensor_ops import gray
from ccrr.runtime.support import write_json
from ccrr.evaluation.runners import (
    evaluate_classifier,
    evaluate_identity,
)
from ccrr.evaluation.metrics import quality
from ccrr.errors import require


def validate_recovery(experiment, stage, step, levels):
    results = []
    experiment.system.eval()
    with torch.no_grad():
        for i in range(len(experiment.data["val_select"])):
            b = collate([experiment.data["val_select"][i]], experiment.device)
            u, z, g = experiment.system.decompose(b["x"])
            continuous = experiment.system.inn.inverse(u, z)
            err = (continuous - b["x"]).abs()
            require(
                float(err.max()) <= 1e-4 and float(err.square().mean().sqrt()) <= 1e-5,
                "INN numeric gate",
            )
            for level in levels:
                n = quantize_latent(z, experiment.profile.step(level))
                rec = experiment.system.recover(
                    g, dequantize_latent(n, experiment.profile.step(level))
                )
                q = quality(b["x"], rec, b["v"])[0]
                q.update(id=b["id"][0], level=level)
                if stage == "E":
                    source = experiment.system.codec.encode(n, level)
                    decoded, zq = experiment.system.codec.decode(
                        source, level, experiment.device
                    )
                    require(
                        torch.equal(decoded.float(), n),
                        "Actual entropy integer mismatch",
                    )
                    q["source_bytes"] = len(source)
                    q["ideal_fits"] = (len(source) + 88 + 15) // 16 + 26 <= 4 * int(
                        b["legal"].sum()
                    )
                results.append(q)
    report = {"stage": stage, "step": step, "rows": results}
    report["mean_psnr"] = float(np.mean([r["psnr"] for r in results]))
    report["mean_ssim"] = float(
        np.mean([r["ssim"] for r in results if r["ssim"] is not None])
    )
    report["fraction_psnr28_ssim90"] = float(
        np.mean(
            [
                r["psnr"] >= 28 and r["ssim"] is not None and r["ssim"] >= 0.90
                for r in results
            ]
        )
    )
    report["passed_quality"] = (
        report["mean_psnr"] >= 30
        and report["mean_ssim"] >= 0.95
        and report["fraction_psnr28_ssim90"] >= 0.90
    )
    if stage == "E":
        good = {
            r["id"]
            for r in results
            if r["ideal_fits"]
            and r["psnr"] >= 30
            and r["ssim"] is not None
            and r["ssim"] >= 0.95
        }
        report["ideal_coverage"] = len(good) / len(experiment.data["val_select"])
        report["mean_protected_bpp"] = float(
            np.mean(
                [
                    8 * (r["source_bytes"] + 88 + 408) / experiment.profile.size**2
                    for r in results
                ]
            )
        )
    write_json(experiment.output / "validation" / f"{stage}-{step}.json", report)
    return report


def validate_representation_semantics(experiment):
    from ccrr.evaluation.comparisons import paired_clinical, privacy_points
    from ccrr.evaluation.source_privacy import evaluate_source_privacy

    ds = experiment.data["val_select"]

    def generated(b):
        with torch.no_grad():
            return experiment.system.decompose(b["x"])[2]

    standard = lambda b: gray(b["x"])
    rgb = evaluate_classifier(experiment.clinical, ds, experiment.device, None, False)
    generated_clinical = evaluate_classifier(
        experiment.clinical, ds, experiment.device, generated, False
    )
    clinical_check = paired_clinical(rgb, generated_clinical)
    checks = {}
    for scenario in ("p1", "p2"):
        for direction in ("original", "same"):
            rgb_report = evaluate_identity(
                experiment.pub_attacker,
                ds,
                experiment.device,
                scenario,
                None,
                None,
                True,
            )
            gray_report = evaluate_identity(
                experiment.pub_attacker,
                ds,
                experiment.device,
                scenario,
                standard,
                standard if direction == "same" else None,
                True,
            )
            public_report = evaluate_identity(
                experiment.pub_attacker,
                ds,
                experiment.device,
                scenario,
                generated,
                generated if direction == "same" else None,
                True,
            )
            checks[scenario + "-" + direction] = privacy_points(
                rgb_report, gray_report, public_report
            )
    source = evaluate_source_privacy(
        ds,
        experiment.device,
        generated,
        {"reid": experiment.reid, "adaptive": experiment.pub_attacker},
    )
    return {
        "clinical": clinical_check,
        "identity": checks,
        "source": source,
        "passed": clinical_check["passed_point"]
        and all(v["passed"] for v in checks.values())
        and source["passed"],
    }
