"""True Stage A microbatches with unchanged effective-cohort loss definitions.

Batching changes floating-point reduction and stochastic-layer RNG ordering.
It is a separate execution configuration, not a bitwise legacy replacement.
"""

import torch
from torch.nn import functional as F

from ccrr.codec.metadata import CLASSES
from ccrr.data.pad import collate
from ccrr.errors import require
from ccrr.tensor_ops import gray, pooled
from ccrr.training.losses import weighted_cross_entropy


def microbatch(config):
    n = config.get("microbatch", 1)
    require(
        type(n) is int and n in (1, 2, 4, 8, 16),
        "A microbatch must be 1, 2, 4, 8 or 16",
    )
    return n


def chunks(values, n):
    for start in range(0, len(values), n):
        yield values[start : start + n]


def flipped(x, rects, flags):
    out, boxes = [], []
    for image, rect, flag in zip(x, rects, flags):
        left, top, width, height = rect
        out.append(image.flip(-1) if flag else image)
        boxes.append(
            (image.shape[-1] - left - width, top, width, height) if flag else rect
        )
    return torch.stack(out), boxes


def denominator(e, selected, domains):
    labels = torch.tensor(
        [CLASSES.index(e.fit.rows[int(i)]["diagnostic"]) for i in selected],
        device=e.weights.device,
    )
    return e.weights[labels].sum() * domains


def clinical_backward(e, model, selected, rng, views):
    denom = denominator(e, selected, len(views))
    total = torch.zeros((), device=e.device)
    for indices in chunks(selected, microbatch(e.cfg)):
        b = collate([e.fit[int(i)] for i in indices], e.device)
        flags = [rng.random() < 0.5 for _ in indices]
        for view in views:
            x = view(b)
            if x is None:
                continue
            x, rects = flipped(x, b["rect"], flags)
            loss = weighted_cross_entropy(
                model(x, rects), b["label"], e.weights, weight_sum=denom
            )
            loss.backward()
            total += loss.detach()
    return total


def mfm_backward(e, head, selected):
    denom = denominator(e, selected, 2)
    for indices in chunks(selected, microbatch(e.cfg)):
        b = collate([e.fit[int(i)] for i in indices], e.device)
        for x in (b["x"], gray(b["x"])):
            with torch.no_grad():
                f, v, _ = e.mfm(x, b["rect"])
                h = pooled(f, v)
            weighted_cross_entropy(
                head(h), b["label"], e.weights, weight_sum=denom
            ).backward()


def identity_loss(fa, fb, ya, yb, labels, effective_pairs=16):
    # The legacy loop evaluates contrastive() one pair at a time, then /16.
    # Do NOT balance positives/negatives inside each microbatch: their counts
    # can differ and that would change the frozen objective.
    distance = torch.linalg.vector_norm(fa["embedding"] - fb["embedding"], dim=1)
    pair = labels * distance.square() + (1 - labels) * F.relu(1 - distance).square()
    ce = 0.5 * (
        F.cross_entropy(fa["logits"], ya, reduction="sum")
        + F.cross_entropy(fb["logits"], yb, reduction="sum")
    )
    return (ce + 0.5 * pair.sum()) / effective_pairs


def reid_backward(e, model, pairs, rng):
    sampled = []
    for group in chunks(pairs, microbatch(e.cfg)):
        ba = collate([e.fit[a] for a, _, _, _ in group], e.device)
        bb = collate([e.fit[b] for _, b, _, _ in group], e.device)
        xa, xb, ra, rb = [], [], [], []
        for j, (_, _, label, scenario) in enumerate(group):
            domain = int(rng.integers(3))
            flip_a, flip_b = rng.random() < 0.5, rng.random() < 0.5
            a, b = ba["x"][j : j + 1], bb["x"][j : j + 1]
            a = gray(a) if domain in (1, 2) else a
            b = gray(b) if domain == 2 else b
            # Mixed-domain cohorts need 3 channels; this is the same replication
            # already performed for gray inputs by the teacher adapter.
            a, ar = flipped(a.expand(-1, 3, -1, -1), [ba["rect"][j]], [flip_a])
            b, br = flipped(b.expand(-1, 3, -1, -1), [bb["rect"][j]], [flip_b])
            xa.append(a)
            xb.append(b)
            ra.extend(ar)
            rb.extend(br)
            sampled.append(
                {
                    "a": ba["id"][j],
                    "b": bb["id"][j],
                    "label": label,
                    "scenario": scenario,
                    "domain": domain,
                }
            )
        fa, fb = model(torch.cat(xa), ra), model(torch.cat(xb), rb)
        labels = ba["x"].new_tensor([p[2] for p in group])
        identity_loss(
            fa, fb, ba["patient_index"], bb["patient_index"], labels
        ).backward()
    return sampled
