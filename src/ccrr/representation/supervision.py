"""Frozen differentiable teachers and exact rectangle-overlap local alignment."""

import torch
from torch.nn import functional as F
from ccrr.training.losses import clinical_loss
from ccrr.teachers.adapters import freeze


def geometry_weights(transform, native, canvas=512, grid=14, device="cpu"):
    x, y, w, h, l, t, nw, nh, size = transform
    gh, gw = native
    rx = torch.arange(grid + 1, device=device, dtype=torch.float32) * canvas / grid
    ry = rx
    bx = (torch.arange(gw + 1, device=device) * size / gw - l) * w / nw + x
    by = (torch.arange(gh + 1, device=device) * size / gh - t) * h / nh + y
    wx = (
        torch.minimum(
            torch.minimum(rx[1:, None], bx[None, 1:]),
            torch.tensor(x + w, device=device),
        )
        - torch.maximum(
            torch.maximum(rx[:-1, None], bx[None, :-1]), torch.tensor(x, device=device)
        )
    ).clamp_min(0)
    wy = (
        torch.minimum(
            torch.minimum(ry[1:, None], by[None, 1:]),
            torch.tensor(y + h, device=device),
        )
        - torch.maximum(
            torch.maximum(ry[:-1, None], by[None, :-1]), torch.tensor(y, device=device)
        )
    ).clamp_min(0)
    area = torch.einsum("ik,jl->ijkl", wy, wx).reshape(grid * grid, gh * gw)
    return area


def align(features, transforms, canvas):
    result = []
    coverage = []
    for f, t in zip(features, transforms):
        a = geometry_weights(t, f.shape[-2:], canvas=canvas, device=f.device)
        result.append((f.flatten(1) @ a.T) / a.sum(1).clamp_min(1e-12)[None])
        coverage.append(a.sum(1))
    return torch.stack(result), torch.stack(coverage)


def heat_weights(scores, area):
    pi = area / area.sum(1, keepdim=True).clamp_min(1e-12)
    weighted = scores * area
    total = weighted.sum(1, keepdim=True)
    mixed = 0.2 * pi + 0.8 * weighted / total.clamp_min(1e-12)
    return torch.where(total > 1e-12, mixed, pi)


class TeacherSupervision:
    def __init__(
        self,
        clinical,
        mfm,
        reid,
        purifier,
        class_weights,
        tau_id=None,
        local_enabled=True,
    ):
        self.clinical = freeze(clinical)
        self.mfm = freeze(mfm)
        self.reid = freeze(reid)
        self.purifier = freeze(purifier)
        self.class_weights = class_weights
        self.tau_id = tau_id
        self.local_enabled = local_enabled

    def targets(self, x, rects, labels):
        # CAM requires a graph despite frozen teacher parameters.
        with torch.enable_grad():
            inp = x.detach().requires_grad_(True)
            logits, f, _, trans = self.clinical(inp, rects, spatial=True)
            grad = torch.autograd.grad(logits.gather(1, labels[:, None]).sum(), f)[0]
            cam = (f * grad.mean((2, 3), keepdim=True)).sum(1, keepdim=True).relu()
            heat, area = align(cam.detach(), trans, x.shape[-1])
            heat = heat[:, 0]
            usable = (logits.argmax(1) == labels) & torch.isfinite(heat).all(1)
            heat = torch.where(usable[:, None], heat, torch.zeros_like(heat))
        with torch.no_grad():
            mf, mv, mt = self.mfm(x, rects)
            p = self.purifier(mf, mv)
            sem, _ = align(p["semantic"], mt, x.shape[-1])
            ids = self.reid(x, rects)
            idf, _ = align(ids["spatial"], ids["transform"], x.shape[-1])
            ih = F.cosine_similarity(
                ids["spatial"], ids["embedding"][:, :, None, None], dim=1, eps=1e-6
            ).relu()[:, None]
            ih, ia = align(ih, ids["transform"], x.shape[-1])
        return {
            "logits": logits.detach(),
            "mask": p["hard_mask"].detach(),
            "semantic": sem,
            "clinical_weights": heat_weights(heat, area).detach(),
            "id": idf,
            "id_weights": heat_weights(ih[:, 0], ia).detach(),
        }

    def loss(self, g, rects, labels, targets):
        logits = self.clinical(g, rects)
        clin = clinical_loss(
            logits, targets["logits"], labels, self.class_weights.to(g)
        )
        local = g.sum() * 0
        identity = g.sum() * 0
        if self.local_enabled:
            mf, mv, mt = self.mfm(g, rects)
            p = self.purifier(mf, mv, source_mask=targets["mask"])
            sem, _ = align(p["semantic"], mt, g.shape[-1])
            local = (
                (
                    (sem - targets["semantic"]).square().mean(1)
                    * targets["clinical_weights"]
                )
                .sum(1)
                .mean()
            )
            clin = clin + 0.1 * local
        if self.tau_id is not None:
            ids = self.reid(g, rects)
            f, _ = align(ids["spatial"], ids["transform"], g.shape[-1])
            similarity = F.cosine_similarity(f, targets["id"], dim=1, eps=1e-6)
            identity = (
                (F.relu(similarity - self.tau_id) * targets["id_weights"]).sum(1).mean()
            )
        return (
            clin,
            identity,
            {"clinical_local": local.detach(), "identity_local": identity.detach()},
        )
