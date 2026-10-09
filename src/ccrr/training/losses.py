from ccrr.tensor_ops import masked_mean, ssim_values
import math
import torch
from torch.nn import functional as F
from ccrr.tensor_ops import gray


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


def weighted_cross_entropy(logits, labels, weights, *, weight_sum=None):
    """§26.2: summed weighted CE / weight sum of the entire effective batch.

    With an explicit denominator this returns a microbatch *contribution*;
    sum contributions without another division. Omitting it means full batch.
    """
    if weight_sum is None:
        weight_sum = weights[labels].sum()
    return F.cross_entropy(logits, labels, weight=weights, reduction="sum") / weight_sum


def clinical_loss(
    logits, target_logits, labels, weights, *, effective_weight_mean=None
):
    """Microbatch mean, accumulated with microbatch_size / effective_batch_size.

    Pass the effective batch's mean class weight before any forward/backward.
    CE then has one global denominator; KD remains an unweighted sample mean.
    The default is appropriate only when this call contains the full batch.
    """
    weight_sum = (
        None if effective_weight_mean is None else len(labels) * effective_weight_mean
    )
    task = weighted_cross_entropy(logits, labels, weights, weight_sum=weight_sum)
    kd = F.kl_div(
        F.log_softmax(logits, 1),
        target_logits.detach().softmax(1),
        reduction="batchmean",
    )
    return task + 0.5 * kd


def js_divergence(a, b):
    """§8.5 normalized JS of logits, in bits (range [0, 1]).

    Log-space mixture avoids probability floors. Zero-probability terms are
    exactly zero, including when callers represent them by -inf logits.
    """
    la, lb = a.log_softmax(-1), b.log_softmax(-1)
    za, zb = la.isneginf(), lb.isneginf()
    absent = za & zb
    lm = torch.logaddexp(
        la.masked_fill(absent, 0), lb.masked_fill(absent, 0)
    ) - math.log(2)
    value = (
        (la.exp() * (la.masked_fill(za, 0) - lm)).sum(-1)
        + (lb.exp() * (lb.masked_fill(zb, 0) - lm)).sum(-1)
    ) / (2 * math.log(2))
    # Only roundoff can leave the mathematical range; thresholds are unchanged.
    return value.clamp(0, 1)
