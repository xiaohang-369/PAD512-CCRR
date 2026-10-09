"""Exploratory metrics never participate in checkpoint selection or clinical gates."""

from pathlib import Path
import importlib.util
import numpy as np
import torch
from torch.nn import functional as F
from ccrr.config import sha256
from ccrr.errors import require


class LPIPSMetric:
    def __init__(self, device="cpu"):
        import lpips

        self.model = (
            lpips.LPIPS(net="alex", version="0.1", lpips=True, spatial=False)
            .eval()
            .to(device)
        )
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, a, b, rects):
        result = []
        for i, (x, y, w, h) in enumerate(rects):
            aa = a[i : i + 1, :, y : y + h, x : x + w]
            bb = b[i : i + 1, :, y : y + h, x : x + w]
            if aa.shape[1] == 1:
                aa = aa.expand(-1, 3, -1, -1)
                bb = bb.expand(-1, 3, -1, -1)
            if min(w, h) < 64:
                size = (round(h * 64 / min(w, h)), round(w * 64 / min(w, h)))
                aa = F.interpolate(
                    aa, size=size, mode="bilinear", align_corners=False, antialias=True
                )
                bb = F.interpolate(
                    bb, size=size, mode="bilinear", align_corners=False, antialias=True
                )
            result.append(float(self.model(2 * aa - 1, 2 * bb - 1)))
        return result


@torch.no_grad()
def fid_directories(reference, recovered, device="cpu"):
    from pytorch_fid.fid_score import calculate_fid_given_paths

    a = sorted(p.name for p in Path(reference).glob("*.png"))
    b = sorted(p.name for p in Path(recovered).glob("*.png"))
    require(a == b and len(a) > 1, "FID requires identical paired image cohorts")
    try:
        value = float(
            calculate_fid_given_paths(
                [str(reference), str(recovered)],
                batch_size=min(32, len(a)),
                device=device,
                dims=2048,
                num_workers=0,
            )
        )
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        value = None
    return {
        "fid": value,
        "n": len(a),
        "exploratory": True,
        "covariance_rank_deficient": len(a) <= 2048,
        "selection_metric": False,
    }


class ClipIQA:
    """Official implementation adapter, with pinned source/weights and default prompts.

    Factory must return the official quality-only model. No silent PIQ reimplementation
    or alternate prompts are substituted. The source factory is part of the run manifest.
    """

    def __init__(
        self, factory_file, factory_sha256, weights, weights_sha256, device="cpu"
    ):
        require(
            sha256(factory_file) == factory_sha256
            and sha256(weights) == weights_sha256,
            "CLIP-IQA resource hash mismatch",
        )
        spec = importlib.util.spec_from_file_location(
            "ccrr_clipiqa_official_adapter", factory_file
        )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        self.model = m.create_default_quality_model(weights, device).eval()

    @torch.no_grad()
    def __call__(self, rgb):
        return {
            "scores": self.model(rgb).detach().cpu().tolist(),
            "exploratory": True,
            "selection_metric": False,
        }
