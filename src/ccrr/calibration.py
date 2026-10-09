"""Snapshot-bound active probes; unknown regions are never labelled as passed."""

import hashlib
import math
import numpy as np
import torch
from .crypto import seal
from . import metadata
from .records import split_stream, logits_to_records
from .models.common import ihaar, q8
from .losses import js_divergence
from .errors import require


def wilson(successes, total, upper=False):
    require(total > 0, "Wilson needs observations")
    p = successes / total
    z = 1.6448536269514722
    sign = 1 if upper else -1
    return (
        p
        + z * z / (2 * total)
        + sign * z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    ) / (1 + z * z / total)


def empirical_p95(values):
    require(len(values) > 0, "Empty percentile")
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def labels(probes, final=False):
    """probes[k]: fixed-count success/rmse/js/maximum arrays for k=0..4."""
    passed = []
    fractions = []
    details = []
    expected = 256 if final else 64
    for k in range(5):
        p = probes[k]
        s = sum(p["success"])
        n = len(p["success"])
        require(n == expected, "Probe count must be fixed")
        frac = s / n
        low = wilson(s, n)
        fractions.append(frac)
        ok = (
            frac >= 0.99
            and low >= (0.98 if final else 0.95)
            and max(p["maximum"]) <= 4
            and empirical_p95(p["rmse"]) <= 2 / 255
            and len(p["js"]) == (64 if final else 32)
            and np.mean(p["js"]) <= 0.001
            and empirical_p95(p["js"]) <= 0.005
        )
        passed.append(bool(ok))
        details.append({"k": k, "p_hat": frac, "wilson_lower": low, "passed": bool(ok)})
    kstar = 0
    if passed[0]:
        for k in range(1, 5):
            if not passed[k]:
                break
            kstar = k
    return {
        "kstar": kstar,
        "empty_passed": passed[0],
        "reliability": [min(fractions[: k + 1]) for k in range(5)],
        "details": details,
    }


def select_regions(image_id, legal, snapshot_index):
    n = int(math.sqrt(len(legal)))
    out = []
    for qr in range(2):
        for qc in range(2):
            ids = [
                i
                for i, ok in enumerate(legal)
                if ok
                and (i // n >= n // 2) == bool(qr)
                and (i % n >= n // 2) == bool(qc)
            ]
            if ids:
                out.append(
                    min(
                        ids,
                        key=lambda i: hashlib.sha256(
                            f"CAL-REG-v1|{snapshot_index}|{image_id}|{i}".encode()
                        ).digest(),
                    )
                )
    return out


def weighted_quantile(values, weights, q=0.95):
    v = np.asarray(values)
    w = np.asarray(weights)
    require(
        len(v) > 0 and np.isfinite(v).all() and (w >= 0).all() and w.sum() > 0,
        "Invalid weighted quantile",
    )
    order = np.argsort(v, kind="stable")
    i = np.searchsorted(np.cumsum(w[order]), q * w.sum(), side="left")
    return float(v[order[min(i, len(v) - 1)]])


class ProbeGenerator:
    def __init__(self, profile, digest, allocator):
        self.profile = profile
        self.digest = digest
        self.allocator = allocator

    def streams(self, row, rng):
        kc, km = self.allocator.new_object(self.digest)
        m = seal(metadata.serialize(row), km, 0, self.allocator)
        level = int(rng.integers(1, 4))
        length = int(rng.integers(2, 3959)) * 4
        # Explicitly a stress object. It is never sent to SourceCodec.decode.
        plain = (
            b"ZC01"
            + bytes([1, level])
            + b"\0\0"
            + self.profile.symbols.to_bytes(4, "little")
            + length.to_bytes(4, "little")
            + self.digest
            + rng.bytes(length)
        )
        c = seal(plain, kc, level, self.allocator)
        return split_stream(c, 0, level), split_stream(m, 1, 0)

    def table(self, row, loads, rng):
        c, m = self.streams(row, rng)
        bits = np.zeros((len(loads), 4, 171), np.float32)
        occ = np.zeros((len(loads), 4), np.float32)
        for i, k in enumerate(loads):
            k = int(k)
            if k == 0:
                continue
            mode = int(rng.integers(2 if k == 1 else 3))
            nm = 0 if mode == 0 else k if mode == 1 else int(rng.integers(1, k))
            nc = k - nm
            selected = []
            if nc:
                if rng.random() < 0.25:
                    ids = [len(c) - 1] + rng.choice(
                        len(c) - 1, nc - 1, replace=False
                    ).tolist()
                else:
                    ids = rng.choice(len(c), nc, replace=False).tolist()
                selected.extend(c[j] for j in ids)
            if nm:
                selected.extend(m[j] for j in rng.choice(len(m), nm, replace=False))
            selected.sort(key=lambda r: (r.alpha, r.index))
            for j, r in enumerate(selected):
                bits[i, j] = r.bits()
                occ[i, j] = 1
        return bits, occ

    def background(self, legal, rng, deployment=None):
        mode = int(rng.choice(3, p=[0.25, 0.25, 0.5]))
        n = len(legal)
        if mode == 0:
            loads = rng.choice(5, n, p=[0.75, 0.0625, 0.0625, 0.0625, 0.0625])
        elif mode == 1:
            loads = rng.choice(5, n, p=[0.05, 0.2375, 0.2375, 0.2375, 0.2375])
        else:
            loads = np.array(
                deployment if deployment is not None else np.asarray(legal) * 4,
                dtype=int,
            )
        loads = loads * np.asarray(legal, dtype=int)
        return loads, ("sparse", "crowded", "deployment")[mode]


@torch.no_grad()
def clinical_risk(g, legal, image_id, clinical, rects):
    size = g.shape[-1]
    side = size // 32
    out = np.ones(side * side, np.float64)
    base = clinical(g, rects)
    for i in np.flatnonzero(np.asarray(legal)):
        row, col = divmod(int(i), side)
        scores = []
        for band in range(4):
            seed = int.from_bytes(
                hashlib.sha256(f"20261011|{image_id}|{i}|{band}".encode()).digest()[:8],
                "big",
            )
            rng = np.random.Generator(np.random.PCG64(seed))
            coeff = g.new_zeros(1, 4, 16, 16)
            coeff[:, band] = torch.from_numpy(rng.choice([-1.0, 1.0], (16, 16))).to(g)
            residual = ihaar(coeff)
            residual = residual / residual.abs().max()
            for sign in (-1, 1):
                pert = g.clone()
                pert[:, :, row * 32 : (row + 1) * 32, col * 32 : (col + 1) * 32] += (
                    sign * 4 / 255 * residual
                )
                scores.append(float(js_divergence(clinical(q8(pert), rects), base)))
        out[i] = np.mean(scores)
    return out


@torch.no_grad()
def probe_region(
    comm,
    g,
    legal,
    row,
    rects,
    clinical,
    generator,
    target,
    k,
    trials,
    rng,
    threshold,
    deployment=None,
):
    require(trials in (32, 64, 256), "Unregistered probe count")
    result = {
        "success": [],
        "maximum": [],
        "rmse": [],
        "js": [],
        "conditions": [],
        "logits": [],
        "truth": [],
        "occupancy": [],
    }
    n = g.shape[-1] // 32
    rr, cc = divmod(target, n)
    sl = (
        slice(None),
        slice(None),
        slice(rr * 32, (rr + 1) * 32),
        slice(cc * 32, (cc + 1) * 32),
    )
    for t in range(trials):
        loads, condition = generator.background(legal, rng, deployment)
        loads[target] = k
        bits, occ = generator.table(row, loads, rng)
        bt = torch.from_numpy(bits)[None].to(g)
        ot = torch.from_numpy(occ)[None].to(g)
        public, _ = comm.write(g, bt, ot, torch.as_tensor(legal, device=g.device)[None])
        logits = comm.read(public)[0, target]
        predicted, errors = logits_to_records(logits[None], threshold)
        from .records import Record

        expected = [Record.from_bits(bits[target, j]) for j in range(k)]
        result["success"].append(
            not errors[0] and not errors[1] and predicted == expected
        )
        diff = (public - g)[sl]
        result["maximum"].append(float((diff * 255).abs().max().round()))
        result["rmse"].append(float(diff.square().mean().sqrt()))
        if t < (64 if trials == 256 else 32):
            background = public.clone()
            background[sl] = g[sl]
            result["js"].append(
                float(
                    js_divergence(clinical(public, rects), clinical(background, rects))
                )
            )
        result["conditions"].append(condition)
        if trials == 32:
            result["logits"].append(logits.cpu())
            result["truth"].append(bits[target])
            result["occupancy"].append(occ[target])
    return result


def choose_threshold(samples):
    rows = []
    for threshold in (0.5, 0.7, 0.9, 0.95, 0.99):
        empty = []
        full = []
        for logits, bits, occ in samples:
            from .records import Record

            got, errors = logits_to_records(logits[None], threshold)
            truth = [Record.from_bits(bits[j]) for j in range(4) if occ[j]]
            if truth:
                full.append(not errors[0] and not errors[1] and got == truth)
            else:
                empty.append(bool(got) or bool(errors[0]) or bool(errors[1]))
        require(empty and full, "Threshold selection needs empty and occupied probes")
        rows.append(
            {
                "threshold": threshold,
                "false_empty": float(np.mean(empty)),
                "full_success": float(np.mean(full)),
            }
        )
    accepted = [r for r in rows if r["false_empty"] <= 0.01]
    chosen = (
        max(accepted, key=lambda r: (r["full_success"], r["threshold"]))
        if accepted
        else min(
            rows, key=lambda r: (r["false_empty"], -r["full_success"], -r["threshold"])
        )
    )
    return chosen, bool(accepted), rows
