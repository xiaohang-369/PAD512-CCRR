import math
import torch
from torch.nn import functional as F
from ccrr.tensor_ops import gray


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


def reconstruction(x, rec, v):
    score, valid = ssim_values(x, rec.clamp(0, 1), v)
    return masked_mean((rec - x).abs(), v) + 0.1 * ((1 - score) * valid).mean()


def gray_loss(g, x, v):
    score, valid = ssim_values(g, gray(x), v)
    return ((1 - score) * valid).mean()


def saturation(u, v):
    return masked_mean(F.relu(u.abs() - math.log(509)).square(), v)


def balanced_mean(values, labels):
    parts = [values[labels == k].mean() for k in (0, 1) if (labels == k).any()]
    return torch.stack(parts).mean() if parts else values.sum() * 0


def pair_bce(logits, labels):
    return balanced_mean(
        F.binary_cross_entropy_with_logits(logits, labels, reduction="none"), labels
    )


def contrastive(ea, eb, labels):
    d = torch.linalg.vector_norm(ea - eb, dim=1)
    return balanced_mean(
        labels * d.square() + (1 - labels) * F.relu(1 - d).square(), labels
    )


def message_loss(logits, bits, occ):
    x = logits.reshape(-1, 177)
    t = bits.reshape(-1, 171)
    o = occ.reshape(-1)
    out = {
        "occupancy": balanced_mean(
            F.binary_cross_entropy_with_logits(x[:, 0], o, reduction="none"), o
        )
    }
    active = o > 0
    alpha = t[:, 2].long()

    def type_mean(v):
        terms = [
            v[active & (alpha == k)].mean()
            for k in (0, 1)
            if (active & (alpha == k)).any()
        ]
        return torch.stack(terms).mean() if terms else x.sum() * 0

    out["version"] = type_mean(
        F.cross_entropy(x[:, 1:5], (2 * t[:, 0] + t[:, 1]).long(), reduction="none")
    )
    out["type"] = type_mean(F.cross_entropy(x[:, 5:7], alpha, reduction="none"))
    out["level"] = type_mean(
        F.cross_entropy(x[:, 7:11], (2 * t[:, 3] + t[:, 4]).long(), reduction="none")
    )
    for name, a, b, c, d in [
        ("sequence", 11, 21, 5, 15),
        ("count", 21, 31, 15, 25),
        ("length", 31, 49, 25, 43),
        ("data", 49, 177, 43, 171),
    ]:
        out[name] = type_mean(
            F.binary_cross_entropy_with_logits(
                x[:, a:b], t[:, c:d], reduction="none"
            ).mean(1)
        )
    return sum(out.values()), out


def clinical_loss(logits, target_logits, labels, weights):
    task = F.cross_entropy(logits, labels, weight=weights)
    kd = F.kl_div(
        F.log_softmax(logits, 1),
        target_logits.detach().softmax(1),
        reduction="batchmean",
    )
    return task + 0.5 * kd


def js_divergence(a, b):
    pa, pb = a.softmax(-1), b.softmax(-1)
    m = (pa + pb) / 2
    return 0.5 * (
        (pa * (pa.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(-1)
        + (pb * (pb.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(-1)
    )
