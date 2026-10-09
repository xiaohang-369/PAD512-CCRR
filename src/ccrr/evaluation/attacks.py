from ccrr.codec.quantization import dequantize_latent
from ccrr.tensor_ops import pooled
from ccrr.teachers.adapters import IdentityModel
import torch
from torch import nn
from torch.nn import functional as F
from ccrr.tensor_ops import unpack


class DoubleConv(nn.Sequential):
    def __init__(self, a, b):
        super().__init__(
            nn.Conv2d(a, b, 3, padding=1),
            nn.GroupNorm(8, b),
            nn.SiLU(),
            nn.Conv2d(b, b, 3, padding=1),
            nn.GroupNorm(8, b),
            nn.SiLU(),
        )


class ReconstructionAttack(nn.Module):
    def __init__(self, latent=False, metadata_conditioned=False):
        super().__init__()
        self.latent = latent
        self.metadata_conditioned = metadata_conditioned
        widths = [32, 64, 128, 256, 256]
        self.first = DoubleConv(1, 32)
        self.down = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(a, b, 3, stride=2, padding=1), DoubleConv(b, b))
                for a, b in zip(widths[:-1], widths[1:])
            ]
        )
        self.up = nn.ModuleList(
            [DoubleConv(a + b, b) for a, b in zip(widths[:0:-1], widths[-2::-1])]
        )
        self.out = nn.Conv2d(32, 2 if latent else 3, 1)
        if metadata_conditioned:
            self.bytes = nn.Embedding(257, 16, padding_idx=256)
            self.meta = nn.Sequential(nn.Linear(416, 256), nn.GELU())
            self.fuse = nn.Conv2d(512, 256, 1)

    def forward(self, g, metadata_bytes=None):
        levels = [self.first(g)]
        for down in self.down:
            levels.append(down(levels[-1]))
        h = levels[-1]
        if self.metadata_conditioned:
            if metadata_bytes is None:
                raise ValueError("Authorized metadata required by this attack")
            mask = (metadata_bytes != 256)[..., None]
            emb = self.bytes(metadata_bytes)
            means = (emb * mask).sum(2) / mask.sum(2).clamp_min(1)
            m = self.meta(means.flatten(1))[:, :, None, None].expand(
                -1, -1, *h.shape[-2:]
            )
            h = self.fuse(torch.cat((h, m), 1))
        for up, skip in zip(self.up, reversed(levels[:-1])):
            h = up(
                torch.cat(
                    (
                        F.interpolate(
                            h,
                            size=skip.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ),
                        skip,
                    ),
                    1,
                )
            )
        out = self.out(h)
        return out if self.latent else out.sigmoid()


class SideChannelAttack(nn.Module):
    def __init__(self, outputs, image_dim=0):
        super().__init__()
        self.side = nn.Sequential(
            nn.Linear(8, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU()
        )
        self.head = nn.Linear(32 + image_dim, outputs)

    def forward(self, side, image_embedding=None):
        h = self.side(side)
        if image_embedding is not None:
            h = torch.cat((h, image_embedding), 1)
        return self.head(h)


def constant_latent_attack(system, g, level, map_symbol=False):
    p = system.profile
    if not map_symbol:
        z = g.new_zeros(len(g), 2, p.size, p.size)
    else:
        import numpy as np

        freq = np.diff(system.codec.cdfs[level - 1], axis=1)[:, :511]
        best = torch.from_numpy(freq.argmax(1) - 255).to(g)
        z = dequantize_latent(
            unpack(
                best[None, :, None, None].expand(len(g), 128, p.size // 8, p.size // 8)
            ),
            p.step(level),
        )
    return system.recover(g, z)


def ncc(a, b):
    a = a.flatten(1)
    b = b.flatten(1)
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    den = a.norm(dim=1) * b.norm(dim=1)
    return torch.where(
        den >= 1e-12, (a * b).sum(1) / den.clamp_min(1e-12), torch.zeros_like(den)
    )


def metadata_tensor(rows, fields, device):
    out = torch.full((len(rows), 26, 64), 256, dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        for j, key in enumerate(fields):
            raw = list(row[key].encode("utf-8"))
            out[i, j, : len(raw)] = torch.tensor(raw, device=device)
    return out


class PooledAttacker(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1536, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, a, va, b, vb):
        a = pooled(a, va)
        b = pooled(b, vb)
        return self.net(torch.cat(((a - b).abs(), a * b), 1)).squeeze(1)


class AttributeModel(nn.Module):
    def __init__(self, kind, outputs):
        super().__init__()
        self.kind = kind
        if kind != "side":
            self.image = IdentityModel(1)
        if kind == "image":
            self.head = nn.Linear(2048, outputs)
        else:
            self.head = SideChannelAttack(outputs, 2048 if kind == "combined" else 0)

    def forward(self, x, side):
        emb = self.image(x)["embedding"] if self.kind != "side" else None
        return self.head(emb) if self.kind == "image" else self.head(side, emb)
