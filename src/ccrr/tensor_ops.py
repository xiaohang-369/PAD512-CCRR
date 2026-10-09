import torch
from torch import nn
from torch.nn import functional as F


def ste_round(x):
    y = torch.sign(x) * torch.floor(x.abs() + 0.5)
    return x + (y - x).detach()


def q8(x):
    x = x.clamp(0, 1)
    y = torch.floor(x * 255 + 0.5) / 255
    return x + (y - x).detach()


def public_adapter(g, eps=1 / 510):
    return torch.logit(g.clamp(eps, 1 - eps))


def gray(x):
    return q8(
        (x * x.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]).sum(
            1, keepdim=True
        )
    )


def squeeze(x):
    return F.pixel_unshuffle(x, 2)


def unsqueeze(x):
    return F.pixel_shuffle(x, 2)


def pack(x):
    for _ in range(3):
        x = squeeze(x)
    return x


def unpack(x):
    for _ in range(3):
        x = unsqueeze(x)
    return x


def pooled(x, v):
    return (x * v).sum((-2, -1)) / v.sum((-2, -1)).clamp_min(1e-6)


def finite(x, name):
    if not torch.isfinite(x).all():
        raise FloatingPointError(name)


class LNch(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.LayerNorm(c, eps=1e-5)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = LNch(c)
        self.n2 = LNch(c)
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x, mask=None):
        m = 1 if mask is None else (mask > 0).to(x)
        y = self.c1(F.gelu(self.n1(x))) * m
        return (x + self.c2(F.gelu(self.n2(y))) * m) * m


def pos1(n, d, device=None):
    p = torch.arange(n, device=device).float()[:, None]
    angle = p / 10000 ** (torch.arange(0, d, 2, device=device).float() / d)
    return torch.stack((angle.sin(), angle.cos()), -1).flatten(1)


def pos2(h, w, d, device=None):
    return torch.cat(
        (
            pos1(h, d // 2, device)[:, None, :].expand(h, w, -1),
            pos1(w, d // 2, device)[None, :, :].expand(h, w, -1),
        ),
        -1,
    ).reshape(h * w, d)


def haar(x):
    a, b, c, d = (
        x[..., ::2, ::2],
        x[..., ::2, 1::2],
        x[..., 1::2, ::2],
        x[..., 1::2, 1::2],
    )
    return torch.cat(
        (
            (a + b + c + d) / 2,
            (a - b + c - d) / 2,
            (a + b - c - d) / 2,
            (a - b - c + d) / 2,
        ),
        1,
    )


def ihaar(x):
    l, hx, hy, hd = x.chunk(4, 1)
    z = torch.stack(
        (
            (l + hx + hy + hd) / 2,
            (l - hx + hy - hd) / 2,
            (l + hx - hy - hd) / 2,
            (l - hx - hy + hd) / 2,
        ),
        2,
    )
    return F.pixel_shuffle(z.flatten(1, 2), 2)


def init_comm(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class SymmetricHead(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim * 2, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, a, b):
        return self.net(torch.cat(((a - b).abs(), a * b), 1)).squeeze(1)


def masked_mean(x, v):
    return (
        (x * v).sum((1, 2, 3)) / (v.sum((1, 2, 3)) * x.shape[1]).clamp_min(1)
    ).mean()


def ssim_values(x, y, v):
    c = x.shape[1]
    n = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    w = torch.exp(-n * n / (2 * 1.5**2))
    w = w / w.sum()
    kernel = (w[:, None] * w[None, :])[None, None].expand(c, 1, 11, 11)
    conv = lambda z: F.conv2d(z, kernel, groups=c)
    mx, my = conv(x), conv(y)
    vx = (conv(x * x) - mx * mx).clamp_min(0)
    vy = (conv(y * y) - my * my).clamp_min(0)
    cov = conv(x * y) - mx * my
    s = (
        (2 * mx * my + 0.01**2)
        * (2 * cov + 0.03**2)
        / ((mx * mx + my * my + 0.01**2) * (vx + vy + 0.03**2))
    )
    valid = F.avg_pool2d(v, 11, stride=1) >= 1 - 1e-7
    count = valid.sum((1, 2, 3)) * c
    value = (s * valid).sum((1, 2, 3)) / count.clamp_min(1)
    return value, count > 0
