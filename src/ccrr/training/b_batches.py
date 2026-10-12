"""B's immutable teacher bank and true batches. No trainable features cached."""

import time
import torch
from torch.nn import functional as F
from ccrr.codec.metadata import CLASSES
from ccrr.data.pad import collate, rng_for
from ccrr.errors import require
from ccrr.tensor_ops import gray, pooled, masked_mean
from ccrr.training.losses import weighted_cross_entropy
from ccrr.training.teacher_batches import chunks
from ccrr.training.optimization import update


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class TeacherBank:
    """Rebuilt in canonical dataset order after reload, independent of sampling.

    Both input domains use fixed FP32 teacher batches. Cache contents never
    include a Purifier output, source mask, augmentation or an autograd graph.
    """

    def __init__(self, e, role="fit", identity=True):
        self.e, self.role = e, role
        ds = e.data[role]
        self.device = torch.device(e.cfg.get("b_cache_device", "cpu"))
        budget = e.cfg.get("b_cache_bytes", 4 * 1024**3)
        # Registered teacher shapes; allocate only after checking the bound.
        estimate = len(ds) * (2 * 768 * 14 * 14 + 196 + (4096 if identity else 0)) * 4
        require(estimate <= budget, "B teacher bank exceeds b_cache_bytes")
        for model in (e.mfm, e.reid):
            require(
                not model.training
                and not any(p.requires_grad for p in model.parameters()),
                "Teacher bank requires frozen eval teachers",
            )
        storage = {}
        start = time.perf_counter()
        n = e.cfg.get("microbatch", 1)
        with torch.no_grad():
            for ids in chunks(list(range(len(ds))), n):
                b = collate([ds[i] for i in ids], e.device)
                result = {}
                for domain, x in enumerate((b["x"], gray(b["x"]))):
                    f, v, _ = e.mfm(x, b["rect"])
                    result["f" + str(domain)] = f
                    if identity:
                        result["e" + str(domain)] = e.reid(x, b["rect"])["embedding"]
                    if domain == 0:
                        result["v"] = v
                    else:
                        require(
                            torch.equal(result["v"], v),
                            "Teacher domain geometry mismatch",
                        )
                for key, value in result.items():
                    require(
                        value.dtype == torch.float32 and torch.isfinite(value).all(),
                        "Nonfinite/non-FP32 frozen teacher bank",
                    )
                    if key not in storage:
                        storage[key] = torch.empty(
                            (len(ds), *value.shape[1:]),
                            dtype=value.dtype,
                            device=self.device,
                        )
                    storage[key][ids[0] : ids[-1] + 1].copy_(
                        value.detach().to(self.device)
                    )
                if (
                    ids[0] == 0
                    or ids[-1] + 1 == len(ds)
                    or (ids[-1] + 1) % (n * 20) == 0
                ):
                    print(
                        f"B_TEACHER_CACHE: {role} {ids[-1] + 1}/{len(ds)}", flush=True
                    )
        self.storage = storage
        self.bytes = sum(v.numel() * v.element_size() for v in storage.values())
        require(self.bytes <= budget, "B teacher bank actual size exceeds budget")
        sync(e.device)
        self.seconds = time.perf_counter() - start

    def get(self, indices):
        index = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        return {
            k: v.index_select(0, index).to(self.e.device)
            for k, v in self.storage.items()
        }


def semantic_batch(e, targets, domains):
    # Group by the already sampled view. RGB computation is shared only within
    # this graph; source-mask gradients are never cached or detached.
    rgb_ids = [i for i, domain in enumerate(domains) if domain in (0, 2)]
    px = None
    if rgb_ids:
        px = e.purifier(targets["f0"][rgb_ids], targets["v"][rgb_ids])
    parts, order = [], []
    for domain in (0, 1, 2):
        ids = [i for i, d in enumerate(domains) if d == domain]
        if not ids:
            continue
        if domain == 0:
            feature = px["semantic"][[rgb_ids.index(i) for i in ids]]
        else:
            mask = px["mask"][[rgb_ids.index(i) for i in ids]] if domain == 2 else None
            feature = e.purifier(
                targets["f1"][ids], targets["v"][ids], source_mask=mask
            )["semantic"]
        parts.append(feature)
        order.extend(ids)
    inverse = [order.index(i) for i in range(len(domains))]
    return torch.cat(parts)[inverse], targets["v"]


def pair_backward(e, bank, pairs, rng, coefficient, detach):
    n = e.cfg.get("microbatch", 1)
    total = torch.zeros((), device=e.device)
    for group in chunks(pairs, n):
        domains = [(int(rng.integers(3)), int(rng.integers(3))) for _ in group]
        ta = bank.get([a for a, _, _, _ in group])
        tb = bank.get([b for _, b, _, _ in group])
        with torch.set_grad_enabled(not detach):
            fa, va = semantic_batch(e, ta, [d[0] for d in domains])
            fb, vb = semantic_batch(e, tb, [d[1] for d in domains])
        logit = e.sem_attacker(fa, va, fb, vb)
        loss = (
            coefficient
            * F.binary_cross_entropy_with_logits(
                logit, logit.new_tensor([s for _, _, s, _ in group]), reduction="sum"
            )
            / 16
        )
        loss.backward()
        total += loss.detach()
    return total


def image_backward(e, bank, ids):
    labels = [CLASSES.index(e.fit.rows[int(i)]["diagnostic"]) for i in ids]
    denom = e.weights[torch.tensor(labels, device=e.device)].sum() * 3
    total = torch.zeros((), device=e.device)
    n = e.cfg.get("microbatch", 1)
    for start in range(0, len(ids), n):
        group = ids[start : start + n]
        t = bank.get(group)
        v = t["v"]
        px, py = e.purifier(t["f0"], v), e.purifier(t["f1"], v)
        cross = e.purifier(t["f1"], v, source_mask=px["mask"])
        loss = torch.zeros((), device=e.device)
        for out, target in ((px, t["e0"]), (py, t["e1"])):
            all_loss = masked_mean((out["all"] - out["standardized"]).square(), v)
            id_loss = (out["identity"] - target).square().sum(1).mean()
            occupancy = pooled(out["mask"], v).mean(1)
            mask_loss = (
                F.relu(0.1 - occupancy).square() + F.relu(occupancy - 0.5).square()
            ).mean()
            loss = loss + (all_loss + 0.1 * id_loss + 10 * mask_loss) * len(group) / 32
        y = torch.tensor(labels[start : start + n], device=e.device)
        for out in (px, py, cross):
            loss = loss + weighted_cross_entropy(
                out["logits"], y, e.weights, weight_sum=denom
            )
        loss.backward()
        total += loss.detach()
    return total


def train_step(e, bank, step, opt, advopt, scheduler, advscheduler):
    sync(e.device)
    start = time.perf_counter()
    e.purifier.train().requires_grad_(True)
    e.sem_attacker.train().requires_grad_(True)
    attacks = []
    for sub in range(2):
        rng = rng_for(e.seed, "B", "attacker" + str(sub), step)
        pairs = e.sampler.sample(rng)
        advopt.zero_grad(set_to_none=True)
        attacks.append(pair_backward(e, bank, pairs, rng, 1.0, True))
        advscheduler.apply(2 * step + sub)
        update(advopt)
        advscheduler.advance()
    e.sem_attacker.eval().requires_grad_(False)
    advopt.zero_grad(set_to_none=True)
    sync(e.device)
    attack_seconds = time.perf_counter() - start
    opt.zero_grad(set_to_none=True)
    rng = rng_for(e.seed, "B", "images", step)
    ids = rng.integers(len(e.fit), size=16)
    loss = image_backward(e, bank, ids)
    ramp = max(0, min(1, (step - 999) / 1000))
    if ramp:
        pairs = e.sampler.sample(rng_for(e.seed, "B", "generator_pairs", step))
        loss += pair_backward(e, bank, pairs, rng, -0.1 * ramp, False)
    scheduler.apply(step)
    norm = update(opt)
    scheduler.advance()
    sync(e.device)
    return {
        "loss": float(loss),
        "attacker_loss": float(torch.stack(attacks).mean()),
        "gradient_norm": norm,
        "adversarial_ramp": ramp,
        "seconds_per_step": time.perf_counter() - start,
        "attacker_seconds": attack_seconds,
        "generator_seconds": time.perf_counter() - start - attack_seconds,
        "purifier_updates": scheduler.next_step,
        "attacker_updates": advscheduler.next_step,
    }
