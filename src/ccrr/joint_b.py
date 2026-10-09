import json
from pathlib import Path
import numpy as np


def accepts(a, b):
    ra = json.loads(Path(a["actual_results_path"]).read_text())["rows"]
    rb = json.loads(Path(b["actual_results_path"]).read_text())["rows"]
    aa = {r["id"]: r for r in ra if r["c_success"]}
    bb = {r["id"]: r for r in rb if r["c_success"]}
    common = set(aa) & set(bb)
    if not common:
        return {
            "accepted": False,
            "reason": "No common authorized reconstruction cohort",
        }
    psnr = float(
        np.mean(
            [
                float(bb[k]["quality"]["psnr"]) - float(aa[k]["quality"]["psnr"])
                for k in common
            ]
        )
    )
    ssim = float(
        np.mean([bb[k]["quality"]["ssim"] - aa[k]["quality"]["ssim"] for k in common])
    )

    def strongest(r):
        return max(
            v["a_dagger"]
            for direction in r["identity"].values()
            for v in direction.values()
            if v.get("valid")
        )

    checks = {
        "coverage": b["both_rate"] >= a["both_rate"] - 0.01,
        "C_reliability": b["c_conditional"] >= 0.99,
        "M_reliability": b["m_conditional"] >= 0.99,
        "clinical": b["clinical"]["macro_auroc"] >= a["clinical"]["macro_auroc"] - 0.01,
        "identity": strongest(b) <= strongest(a) + 0.01,
        "rate": b["mean_bpp"] <= a["mean_bpp"] * 1.05,
        "psnr_improvement": psnr >= 0.5,
        "ssim_non_decreasing": ssim >= 0,
        "base_targets": b["passed"],
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "common_success_n": len(common),
        "psnr_gain": psnr,
        "ssim_difference": ssim,
    }
