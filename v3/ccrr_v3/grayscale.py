"""Design2.0 P/E_C/D, exact hard-forward quantization and pixel supervision.

Learned conditional RGB reconstruction, not an analytical invertible network.
No Writer, encryption, P2/P3 privacy training, or clinical acceptance is implied.
"""

import torch
from torch import nn
from torch.nn import functional as F

from ccrr.tensor_ops import gray, q8, masked_mean, ssim_values
from .models import Residual, FrozenSupervisor, freeze


def conv(cin, cout, kernel=3, stride=1):
    return nn.Conv2d(cin, cout, kernel, stride, kernel // 2)


def block(cin, cout, count, stride=1):
    return nn.Sequential(
        conv(cin, cout, stride=stride),
        nn.ReLU(),
        *[Residual(cout) for _ in range(count)],
    )


class Up(nn.Module):
    def __init__(self, cin, cout, skip):
        super().__init__()
        self.project = nn.Sequential(conv(cin, cout), nn.ReLU())
        self.fuse = block(cout + skip, cout, 2)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat((self.project(x), skip), 1))


class PublicEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.p0, self.p1 = block(3, 32, 2), block(32, 64, 2, 2)
        self.p2 = block(64, 128, 4, 2)
        self.p3, self.p4 = Up(128, 64, 64), Up(64, 32, 32)
        self.residual = conv(32, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, x):
        p0 = self.p0(x)
        p1 = self.p1(p0)
        raw = gray(x) + self.residual(self.p4(self.p3(self.p2(p1), p1), p0))
        return raw, q8(raw)


class PrivateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            conv(4, 64, 5, 2),
            nn.ReLU(),
            conv(64, 96, 5, 2),
            nn.ReLU(),
            conv(96, 128, 5, 2),
            nn.ReLU(),
            conv(128, 192, 5, 2),
            nn.ReLU(),
            *[Residual(192) for _ in range(4)],
            conv(192, 8),
            nn.Sigmoid(),
        )

    def forward(self, x, g):
        return self.net(torch.cat((x, g), 1))


def quantize_private(z, bits):
    if bits not in (4, 8):
        raise ValueError("Only registered 4/8-bit profiles")
    levels = 2**bits - 1
    hard = torch.floor(z * levels + 0.5)
    return hard.detach().to(torch.uint8), z + (hard / levels - z).detach()


class RGBDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gray = nn.ModuleList(
            [
                block(1, 32, 1),
                block(32, 64, 1, 2),
                block(64, 96, 1, 2),
                block(96, 128, 1, 2),
                block(128, 192, 1, 2),
            ]
        )
        self.private = nn.Sequential(conv(8, 128), nn.ReLU())
        self.precision = nn.Linear(2, 16)
        self.fuse = nn.Sequential(
            nn.Conv2d(336, 192, 1), nn.ReLU(), *[Residual(192) for _ in range(4)]
        )
        self.up = nn.ModuleList(
            [Up(192, 128, 128), Up(128, 96, 96), Up(96, 64, 64), Up(64, 32, 32)]
        )
        self.output = conv(32, 3)

    def forward(self, g, z, bits):
        if bits not in (4, 8):
            raise ValueError("Only registered 4/8-bit profiles")
        skips, t = [], g
        for layer in self.gray:
            t = layer(t)
            skips.append(t)
        if z.shape != (g.shape[0], 8, *t.shape[-2:]):
            raise ValueError("Private-code shape does not match received gray")
        onehot = g.new_tensor([1, 0] if bits == 4 else [0, 1]).expand(len(g), -1)
        b = self.precision(onehot)[:, :, None, None].expand(-1, -1, *t.shape[-2:])
        t = self.fuse(torch.cat((t, self.private(z), b), 1))
        for layer, skip in zip(self.up, reversed(skips[:-1])):
            t = layer(t, skip)
        raw = g.expand(-1, 3, -1, -1) + self.output(t)
        return raw, raw.clamp(0, 1)


class GrayscaleSystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.public, self.private, self.decoder = (
            PublicEncoder(),
            PrivateEncoder(),
            RGBDecoder(),
        )

    def reconstruct(self, x, g, bits=8):
        z = self.private(x, g)
        integers, decoded = quantize_private(z, bits)
        raw, rgb = self.decoder(g, decoded, bits)
        return dict(z=z, q=integers, decoded=decoded, raw=raw, rgb=rgb)

    def forward(self, x, bits=8):
        raw_gray, g = self.public(x)
        return dict(gray_raw=raw_gray, gray=g, **self.reconstruct(x, g, bits))


class PixelSupervisor(nn.Module):
    """Bound checkpoint's original heads; audit readout heads are not substituted."""

    def __init__(self, panderm, separator, heads, statistics):
        super().__init__()
        self.panderm = freeze(panderm)
        self.feature = FrozenSupervisor(
            separator, heads, statistics["mean"], statistics["std"]
        )
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, image, rects, domain):
        if domain not in (0, 1) or image.shape[1] != (3 if domain == 0 else 1):
            raise ValueError("Use RGB head for RGB and gray head for one-channel gray")
        raw, coverage, _ = self.panderm(image, rects)
        logits, semantic = self.feature(raw, coverage, domain)
        return logits, semantic


def clinical_loss(logits, labels, target, weights, kd=0.5):
    return F.cross_entropy(logits, labels, weight=weights) + kd * F.kl_div(
        F.log_softmax(logits, 1), F.softmax(target.detach(), 1), reduction="batchmean"
    )


def valid_average(values, support):
    counts = support.sum((1, 2, 3))
    present = counts > 0
    if not present.any():
        return values.sum() * 0, 0
    per_image = (values * support).sum((1, 2, 3)) / counts.clamp_min(1)
    return per_image[present].mean(), int(present.sum())


def local_variation(x):
    # Twelve edges within every 3x3 window, as specified by Design2.0.
    windows = F.unfold(x, kernel_size=3).reshape(len(x), 3, 3, -1)
    values = (
        (windows[:, 1:] - windows[:, :-1]).abs().sum((1, 2))
        + (windows[:, :, 1:] - windows[:, :, :-1]).abs().sum((1, 2))
    ) / 12
    return values.reshape(len(x), 1, x.shape[-2] - 2, x.shape[-1] - 2)


def appearance(g, y, v):
    light = masked_mean((g.sub(y).abs() - 8 / 255).clamp_min(0), v)
    pg, py = F.avg_pool2d(g, 8), F.avg_pool2d(y, 8)
    pv = (F.avg_pool2d(v, 8) >= 1 - 1e-7).to(v)
    dx = ((pg[..., 1:] - pg[..., :-1]) - (py[..., 1:] - py[..., :-1])).abs()
    dy = ((pg[..., 1:, :] - pg[..., :-1, :]) - (py[..., 1:, :] - py[..., :-1, :])).abs()
    cx, nx = valid_average(dx, pv[..., 1:] * pv[..., :-1])
    cy, ny = valid_average(dy, pv[..., 1:, :] * pv[..., :-1, :])
    lv = (F.avg_pool2d(v, 3, 1) >= 1 - 1e-7).to(v)
    local, nl = valid_average((local_variation(g) - local_variation(y)).abs(), lv)
    return light + 0.25 * (cx + cy) + 0.5 * local, dict(
        light=light.detach(),
        contrast=((cx + cy) / 2).detach(),
        local=local.detach(),
        contrast_x_images=nx,
        contrast_y_images=ny,
        local_images=nl,
    )


def reconstruction_loss(out, x, v, supervisor, rects, labels, target, weights):
    ssim, legal = ssim_values(out["rgb"], x, v)
    structural = (1 - ssim[legal]).mean() if legal.any() else out["rgb"].sum() * 0
    logits, _ = supervisor(out["rgb"], rects, 0)
    med = clinical_loss(logits, labels, target, weights)
    l1 = masked_mean((out["raw"] - x).abs(), v)
    return l1 + 0.1 * structural + 0.1 * med, dict(
        l1=l1.detach(),
        ssim=(ssim[legal].mean().detach() if legal.any() else None),
        ssim_images=int(legal.sum()),
        rgb_clinical=med.detach(),
    )


def parameter_mode(system, part):
    for name, module in (
        ("p", system.public),
        ("e", system.private),
        ("d", system.decoder),
    ):
        module.requires_grad_(name in part)
        for p in module.parameters():
            p.grad = None


def canonical_gray(g, valid):
    """Sender-only support rule; the receiver is never given the source mask."""
    if valid.shape != g.shape or not torch.all((valid == 0) | (valid == 1)):
        raise ValueError("Binary support must match the one-channel public image")
    return torch.where(valid.bool(), g, g.new_tensor(128 / 255))


def public_forward(system, x, valid, bits=8):
    raw, g = system.public(x)
    g = canonical_gray(g, valid)
    return dict(gray_raw=raw, gray=g, **system.reconstruct(x, g, bits))


def update(
    system,
    supervisor,
    b,
    targets,
    weights,
    optimizer,
    part,
    inspect_gradients=False,
    *,
    canonical_padding=False,
    step_optimizer=True,
):
    """One B/ED/P substep. Frozen E/D keep BOTH input paths differentiable."""
    parameter_mode(system, part)
    x, v, y = b["x"], b["v"], gray(b["x"])
    if part == "ed":
        with torch.no_grad():
            _, g = system.public(x)
            if canonical_padding:
                g = canonical_gray(g, v)
        out = system.reconstruct(x, g.detach(), 8)
    elif part == "b":
        parameter_mode(system, "ed")
        g = y.detach()
        if canonical_padding:
            g = canonical_gray(g, v)
        out = system.reconstruct(x, g, 8)
    elif part == "p":
        out = public_forward(system, x, v) if canonical_padding else system(x, 8)
        g = out["gray"]
    else:
        raise ValueError("Unknown update substep")
    loss, terms = reconstruction_loss(
        out, x, v, supervisor, b["rect"], b["label"], targets[0], weights
    )
    objectives = {"reconstruction": loss}
    if part == "p":
        logits, _ = supervisor(g, b["rect"], 1)
        medical = clinical_loss(logits, b["label"], targets[1], weights)
        app, app_terms = appearance(g, y, v)
        outside = masked_mean(F.relu(-out["gray_raw"]) + F.relu(out["gray_raw"] - 1), v)
        loss = loss + app + 0.1 * medical + 0.001 * outside
        objectives.update(appearance=app, medical=0.1 * medical, range=0.001 * outside)
        terms.update(
            public_clinical=medical.detach(),
            appearance=app.detach(),
            range=outside.detach(),
            **app_terms,
        )
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite generator loss")
    if part == "p" and inspect_gradients:
        gradients = {}
        params = tuple(system.public.parameters())
        for name, value in objectives.items():
            grads = torch.autograd.grad(value, params, retain_graph=True)
            gradients[name] = torch.cat([g.detach().flatten() for g in grads])
        terms["weighted_public_gradient_norms"] = {
            k: float(g.norm()) for k, g in gradients.items()
        }
        terms["weighted_public_gradient_cosines"] = {
            f"{a}:{b}": (
                float(torch.dot(ga, gb) / (ga.norm() * gb.norm()))
                if ga.norm() > 0 and gb.norm() > 0
                else None
            )
            for a, ga in gradients.items()
            for b, gb in gradients.items()
            if a < b
        }
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norms = {}
    for name, module in (
        ("p", system.public),
        ("e", system.private),
        ("d", system.decoder),
    ):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        if any(not torch.isfinite(g).all() for g in grads):
            raise RuntimeError(f"Nonfinite {name} gradient")
        norms[name] = float(
            torch.sqrt(
                sum(
                    (g.detach().square().sum() for g in grads),
                    torch.zeros((), device=x.device),
                )
            )
        )
    if any(p.grad is not None for p in supervisor.parameters()):
        raise RuntimeError("Frozen pixel supervisor accumulated a parameter gradient")
    if step_optimizer:
        optimizer.step()
    terms.update(loss=loss.detach(), **{f"grad_{k}": n for k, n in norms.items()})
    terms.update(
        private_min=out["z"].detach().min(),
        private_max=out["z"].detach().max(),
        private_saturation_fraction=(
            (out["z"].detach() < 0.01) | (out["z"].detach() > 0.99)
        )
        .float()
        .mean(),
        private_channel_symbol_mean=out["q"].float().mean((0, 2, 3)).cpu().tolist(),
    )
    if part == "p":
        terms["gray_clip_fraction"] = masked_mean(
            ((out["gray_raw"].detach() < 0) | (out["gray_raw"].detach() > 1)).float(), v
        )
    return {
        k: (float(v) if isinstance(v, torch.Tensor) else v) for k, v in terms.items()
    }
