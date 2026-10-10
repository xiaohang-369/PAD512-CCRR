"""Frozen MFM spatial-interface checks on actual fit images; no capability surrogate."""

import hashlib
import torch
from ccrr.data.pad import collate, adapt
from ccrr.tensor_ops import gray, pooled
from ccrr.errors import require


def mfm_interface(model, data, device):
    model.eval()

    def digest():
        h = hashlib.sha256()
        for v in model.state_dict().values():
            h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    before = digest()
    results = []
    indices = [0]
    # Include an actual non-square image when available, so origin relocation
    # does not reduce to the full-canvas identity transform.
    for i, row in enumerate(data.rows):
        inv = getattr(data, "inventory", {}).get(row.get("img_id"), {})
        rect = inv.get("content_rect_xywh")
        if rect and rect != [0, 0, 512, 512] and i not in indices:
            indices.append(i)
            break
    for i in indices:
        b = collate([data[i]], device)
        for domain, value in [("rgb", b["x"]), ("gray", gray(b["x"]))]:
            x = value.detach().requires_grad_(True)
            f, v, transforms = model(x, b["rect"])
            require(
                f.shape == (1, 768, 14, 14) and torch.isfinite(f).all(),
                "MFM patch interface",
            )
            require(
                bool((v > 0).any()) and float(f.detach().var()) > 0,
                "MFM degenerate features/tokens",
            )
            grad = torch.autograd.grad(f[:, :16].square().mean(), x)[0]
            require(
                torch.isfinite(grad).all() and bool(grad.abs().sum() > 0),
                "MFM input gradient",
            )
            # Known source-coordinate relocation: crop away external padding; adapt
            # must yield the identical content and geometry in the 224 teacher canvas.
            left, top, w, h = b["rect"][0]
            crop = x.detach()[:, :, top : top + h, left : left + w]
            a, coverage, t = adapt(x.detach(), b["rect"], 224, panderm=True)
            moved, moved_v, mt = adapt(crop, [(0, 0, w, h)], 224, panderm=True)
            require(
                torch.equal(a, moved)
                and torch.equal(coverage, moved_v)
                and t[0][4:] == mt[0][4:],
                "MFM known-coordinate relocation failed",
            )
            spatial_variance = (
                pooled(f.detach().square(), v) - pooled(f.detach(), v).square()
            ).clamp_min(0)
            require(
                not bool(
                    grad.masked_select(~b["v"].bool().expand_as(grad)).abs().sum()
                ),
                "MFM padding must not receive input gradients",
            )
            results.append(
                {
                    "domain": domain,
                    "feature_variance": float(f.detach().var()),
                    "spatial_variance_mean": float(spatial_variance.mean()),
                    "spatial_zero_variance_channels": int(
                        (spatial_variance == 0).sum()
                    ),
                    "valid_tokens": int((v > 0).sum()),
                    "coverage_sum": float(v.sum()),
                    "gradient_l1": float(grad.abs().sum()),
                    "transform": transforms,
                    "coordinate_relocation_exact": True,
                }
            )
    require(
        digest() == before
        and all(p.grad is None and not p.requires_grad for p in model.parameters()),
        "Frozen MFM changed during interface check",
    )
    return {
        "passed": True,
        "cases": results,
        "local_medical_alignment_validated": False,
    }
