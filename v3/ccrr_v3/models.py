"""ICCV-inspired feature separation. These codes are NOT transmitted q_C."""

import torch
from torch import nn
from torch.nn import functional as F


def pool(x, coverage):
    return (x * coverage).sum((2, 3)) / coverage.sum((2, 3)).clamp_min(1e-8)


def weighted_mean(x, coverage):
    """Per-image, then batch mean; include zero symbols, never divide by M.sum()."""
    denom = coverage.sum((1, 2, 3)) * x.shape[1]
    if (
        not torch.isfinite(coverage).all()
        or (coverage < 0).any()
        or (coverage > 1).any()
    ):
        raise ValueError("Invalid fixed coverage")
    if (denom <= 0).any():
        raise ValueError("Empty valid support")
    return ((x * coverage).sum((1, 2, 3)) / denom).mean()


def quantize(x):
    # Defined tie rule: floor(x+0.5), including negative half-integers.
    hard = torch.floor(x + 0.5)
    return x + (hard - x).detach()


class Residual(nn.Module):
    """No batch statistics or stochastic dropout; explicit V3 adaptation choice."""

    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class GaussianMixture(nn.Module):
    def __init__(self, channels, min_scale=0.05, min_probability=1e-9):
        super().__init__()
        if not 0 < min_scale or not 0 < min_probability < 1:
            raise ValueError("Invalid entropy numerical bounds")
        self.min_scale = min_scale
        self.min_probability = min_probability
        self.logits = nn.Parameter(torch.zeros(1, channels, 3, 1, 1))
        self.means = nn.Parameter(torch.zeros(1, channels, 3, 1, 1))
        self.raw_scales = nn.Parameter(
            torch.tensor([-1.0, 0.0, 1.0])[None, None, :, None, None].repeat(
                1, channels, 1, 1, 1
            )
        )

    def forward(self, q):
        scale = F.softplus(self.raw_scales) + self.min_scale
        centered = q.unsqueeze(2) - self.means
        lo, hi = (centered - 0.5) / scale, (centered + 0.5) / scale
        # Survival-function difference on positive tails avoids 1 - 1 cancellation.
        mass = torch.where(
            lo >= 0,
            torch.special.ndtr(-lo) - torch.special.ndtr(-hi),
            torch.special.ndtr(hi) - torch.special.ndtr(lo),
        )
        probability = (self.logits.softmax(2) * mass).sum(2)
        clipped = probability < self.min_probability
        bits = -torch.log2(probability.clamp_min(self.min_probability))
        return bits, clipped


class Separator(nn.Module):
    def __init__(
        self,
        input_channels=768,
        width=256,
        code_channels=32,
        embedding_dim=2048,
        min_scale=0.05,
        min_probability=1e-9,
        variant="separation",
    ):
        super().__init__()
        if variant not in ("compression", "separation", "adversarial"):
            raise ValueError("Unknown independently trained variant")
        self.variant = variant
        self.spec = dict(
            input_channels=input_channels,
            width=width,
            code_channels=code_channels,
            embedding_dim=embedding_dim,
            min_scale=min_scale,
            min_probability=min_probability,
            variant=variant,
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, width, 1),
            *[Residual(width) for _ in range(3)],
            nn.Conv2d(width, code_channels, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(code_channels, width, 3, padding=1),
            *[Residual(width) for _ in range(3)],
            nn.Conv2d(width, input_channels, 1),
        )
        self.entropy = GaussianMixture(code_channels, min_scale, min_probability)
        if variant != "compression":
            self.mask_net = nn.Conv2d(code_channels, 2 * code_channels, 3, padding=1)
            self.link_decoder = nn.Sequential(
                nn.Conv2d(code_channels, width, 1),
                *[Residual(width) for _ in range(3)],
            )
            self.link_head = nn.Linear(width, embedding_dim)

    def forward(self, features, coverage, temperature=1.0, stochastic=None):
        if temperature <= 0:
            raise ValueError("Mask temperature must be positive")
        stochastic = self.training if stochastic is None else stochastic
        if not self.training and stochastic:
            raise ValueError("Evaluation must use deterministic hard masks")
        q = quantize(self.encoder(features))
        fhat = self.decoder(q)
        if self.variant == "compression":
            return dict(q=q, q_sem=q, semantic=fhat, full=fhat)
        n, c, h, w = q.shape
        logits = self.mask_net(q).reshape(n, c, 2, h, w)
        if stochastic:
            mask = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=2)[:, :, 1]
        else:
            # Equal logits choose index 0 (semantic), deterministically.
            soft = (logits / temperature).softmax(2)[:, :, 1]
            hard = (logits.argmax(2) == 1).to(q.dtype)
            mask = soft + (hard - soft).detach()
        q_link, q_sem = q * mask, q * (1 - mask)
        return dict(
            q=q,
            mask=mask,
            q_link=q_link,
            q_sem=q_sem,
            full=fhat,
            semantic=self.decoder(q_sem),
            embedding=self.link_head(pool(self.link_decoder(q_link), coverage)),
        )

    def rate(self, output, coverage):
        bits, clipped = self.entropy(output["q"])
        total = weighted_mean(bits, coverage)
        stats = {
            "q_bits_per_valid_symbol": total.detach(),
            "q_probability_clamped": weighted_mean(clipped.float(), coverage).detach(),
        }
        if "q_link" in output:
            link_bits, link_clipped = self.entropy(output["q_link"])
            link_rate = weighted_mean(link_bits, coverage)
            total = total + link_rate
            stats.update(
                link_bits_per_valid_symbol=link_rate.detach(),
                link_probability_clamped=weighted_mean(
                    link_clipped.float(), coverage
                ).detach(),
                mask_occupancy=weighted_mean(output["mask"], coverage).detach(),
                mask_zeroed_nonzero=weighted_mean(
                    ((output["q"] != 0) & (output["mask"] == 0)).float(), coverage
                ).detach(),
            )
        stats["q_native_zero_fraction"] = weighted_mean(
            (output["q"] == 0).float(), coverage
        ).detach()
        return total, stats


class ClinicalHeads(nn.Module):
    def __init__(self, channels=768):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(channels, 6) for _ in range(2)])

    def forward(self, features, coverage, domain):
        return self.heads[domain](pool(features, coverage))


def freeze(module):
    module.eval()
    module.requires_grad_(False)
    return module


class FrozenSupervisor(nn.Module):
    """For later image pipelines: never wrap generated-image evaluation in no_grad.

    PanDerm image adaptation is outside this feature-level interface. Use raw
    PanDerm features here; training loaders apply the same bound fit statistics.
    """

    def __init__(self, separator, heads, mean, std):
        super().__init__()
        self.separator, self.heads = freeze(separator), freeze(heads)
        self.register_buffer("mean", mean.reshape(1, -1, 1, 1))
        self.register_buffer("std", std.reshape(1, -1, 1, 1))
        if not torch.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("Invalid fit statistics")
        self.train(False)

    def train(self, mode=True):
        # A parent generator.train() must not enable stochastic teacher masks.
        super().train(False)
        return self

    def forward(self, raw_features, coverage, domain):
        out = self.separator(
            (raw_features - self.mean) / self.std, coverage, stochastic=False
        )
        return self.heads(out["semantic"], coverage, domain), out["semantic"]
