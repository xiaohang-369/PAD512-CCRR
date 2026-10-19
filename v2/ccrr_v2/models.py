"""Continuous S0 and freshly initialized readouts; no INN or old Purifier."""

import torch
from torch import nn


def pool(x, coverage):
    return (x * coverage).sum((2, 3)) / coverage.sum((2, 3)).clamp_min(1e-6)


class SemanticAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(768, 256, 1), nn.ReLU(), nn.Conv2d(256, 128, 1)
        )

    def forward(self, x):
        return self.net(x)


class ClinicalProbe(nn.Module):
    def __init__(self, channels, mlp=False):
        super().__init__()
        self.head = (
            nn.Sequential(nn.Linear(channels, 256), nn.GELU(), nn.Linear(256, 6))
            if mlp
            else nn.Linear(channels, 6)
        )

    def forward(self, x, coverage):
        return self.head(pool(x, coverage))


class PairAttacker(nn.Module):
    def __init__(self, channels, spatial=False):
        super().__init__()
        self.spatial = spatial
        self.encoder = (
            nn.Sequential(
                nn.Conv2d(channels, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(),
            )
            if spatial
            else nn.Identity()
        )
        dim = 128 if spatial else channels
        self.head = nn.Sequential(nn.Linear(2 * dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, a, av, b, bv):
        a, b = pool(self.encoder(a), av), pool(self.encoder(b), bv)
        return self.head(torch.cat(((a - b).abs(), a * b), 1)).squeeze(1)


def confusion_loss(logits):
    return (
        nn.functional.binary_cross_entropy_with_logits(
            logits, torch.full_like(logits, 0.5)
        )
        - 0.6931471805599453
    )
