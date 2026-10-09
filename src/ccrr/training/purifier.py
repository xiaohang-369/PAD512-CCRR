"""Stage B: Purifier optimization and independent feature acceptance."""

import json
import torch
from torch.nn import functional as F
from ccrr.config import sha256
from ccrr.data.pad import collate, rng_for
from ccrr.codec.metadata import CLASSES
from ccrr.teachers.adapters import (
    freeze,
)
from ccrr.tensor_ops import gray, pooled
from ccrr.representation.supervision import TeacherSupervision, align
from ccrr.tensor_ops import masked_mean
from ccrr.runtime.support import write_json
from ccrr.training.optimization import optimizer, schedule, update
from ccrr.runtime.checkpoints import save_checkpoint, load_checkpoint
from ccrr.errors import require
from ccrr.calibration.probes import (
    weighted_quantile,
)


def semantic_view(experiment, b, domain):
    x = b["x"]
    y = gray(x)
    with torch.no_grad():
        fx, v, _ = experiment.mfm(x, b["rect"])
        fy, _, _ = experiment.mfm(y, b["rect"])
    px = experiment.purifier(fx, v)
    if domain == 0:
        return px, v
    return experiment.purifier(
        fy, v, source_mask=px["mask"] if domain == 2 else None
    ), v


def semantic_adversary_step(experiment, step, substep, opt):
    rng = rng_for(experiment.seed, "B", "attacker" + str(substep), step)
    pairs = experiment.sampler.sample(rng)
    opt.zero_grad(set_to_none=True)
    for a, b, s, _ in pairs:
        ba = collate([experiment.fit[a]], experiment.device)
        bb = collate([experiment.fit[b]], experiment.device)
        with torch.no_grad():
            pa, va = semantic_view(experiment, ba, int(rng.integers(3)))
            pb, vb = semantic_view(experiment, bb, int(rng.integers(3)))
        logit = experiment.sem_attacker(
            pa["semantic"].detach(), va, pb["semantic"].detach(), vb
        )
        (
            F.binary_cross_entropy_with_logits(logit, logit.new_tensor([s])) / 16
        ).backward()
    update(opt)


def stage_b(experiment):
    require(
        experiment.mfm is not None and bool(experiment.purifier.stats_ready),
        "Teacher stages required",
    )
    experiment.purifier.train()
    experiment.sem_attacker.train()
    opt = optimizer([experiment.purifier], [1e-4])
    advopt = optimizer([experiment.sem_attacker], [1e-4])
    total = experiment.cfg["budgets"]["purifier"]
    for step in range(total):
        for p in experiment.sem_attacker.parameters():
            p.requires_grad_(True)
        for sub in range(2):
            semantic_adversary_step(experiment, step, sub, advopt)
        freeze(experiment.sem_attacker)
        opt.zero_grad(set_to_none=True)
        rng = rng_for(experiment.seed, "B", "images", step)
        ids = rng.integers(len(experiment.fit), size=16)
        denom = (
            sum(
                float(
                    experiment.weights[
                        CLASSES.index(experiment.fit.rows[int(i)]["diagnostic"])
                    ]
                )
                for i in ids
            )
            * 3
        )
        for i in ids:
            b = collate([experiment.fit[int(i)]], experiment.device)
            x = b["x"]
            y = gray(x)
            with torch.no_grad():
                fx, v, _ = experiment.mfm(x, b["rect"])
                fy, _, _ = experiment.mfm(y, b["rect"])
                ex = experiment.reid(x, b["rect"])["embedding"]
                ey = experiment.reid(y, b["rect"])["embedding"]
            px = experiment.purifier(fx, v)
            py = experiment.purifier(fy, v)
            cross = experiment.purifier(fy, v, source_mask=px["mask"])
            loss = x.sum() * 0
            for out, e in ((px, ex), (py, ey)):
                all_loss = masked_mean((out["all"] - out["standardized"]).square(), v)
                id_loss = (out["identity"] - e).square().sum(1).mean()
                occupancy = pooled(out["mask"], v).mean(1)
                mask_loss = (
                    F.relu(0.1 - occupancy).square() + F.relu(occupancy - 0.5).square()
                ).mean()
                loss = loss + (all_loss + 0.1 * id_loss + 10 * mask_loss) / 32
            for out in (px, py, cross):
                loss = (
                    loss
                    + F.cross_entropy(
                        out["logits"],
                        b["label"],
                        weight=experiment.weights,
                        reduction="sum",
                    )
                    / denom
                )
            loss.backward()
        ramp = max(0, min(1, (step - 999) / 1000))
        if ramp:
            pairs = experiment.sampler.sample(
                rng_for(experiment.seed, "B", "generator_pairs", step)
            )
            for a, b, s, _ in pairs:
                ba = collate([experiment.fit[a]], experiment.device)
                bb = collate([experiment.fit[b]], experiment.device)
                pa, va = semantic_view(experiment, ba, int(rng.integers(3)))
                pb, vb = semantic_view(experiment, bb, int(rng.integers(3)))
                logit = experiment.sem_attacker(pa["semantic"], va, pb["semantic"], vb)
                (
                    -0.1
                    * ramp
                    * F.binary_cross_entropy_with_logits(logit, logit.new_tensor([s]))
                    / 16
                ).backward()
        schedule(opt, step, total)
        norm = update(opt)
        if (step + 1) % 100 == 0:
            experiment.log(
                "B", step + 1, {"gradient_norm": norm, "adversarial_ramp": ramp}
            )
        if (step + 1) % 1000 == 0 or step + 1 == total:
            save_checkpoint(
                experiment.output / "checkpoints" / f"purifier-{step + 1}.pt",
                {"purifier": experiment.purifier, "attacker": experiment.sem_attacker},
                opt,
                step + 1,
                "B",
                {},
            )
    # Select among all registered endpoints using independently trained readouts.
    best = None
    for path in sorted(
        (experiment.output / "checkpoints").glob("purifier-*.pt"),
        key=lambda p: int(p.stem.split("-")[-1]),
    ):
        checkpoint = load_checkpoint(path, {"purifier": experiment.purifier})
        freeze(experiment.purifier)
        experiment.feature_audit_tag = "step-" + str(checkpoint["step"])
        report = validate_purifier(experiment)
        if report["passed"]:
            clinical_min = min(
                r["macro_auroc"] for r in report["clinical"]["readouts"]["sem_linear"]
            )
            strongest = max(r["sem_strongest"] for r in report["identity"]["checks"])
            key = (clinical_min, -strongest, -checkpoint["step"])
            if best is None or key > best[0]:
                best = (key, report)
                save_checkpoint(
                    experiment.output / "checkpoints/purifier.pt",
                    {"purifier": experiment.purifier},
                    None,
                    checkpoint["step"],
                    "B",
                    report,
                )
    experiment.mark(
        "B",
        best is not None,
        best[1] if best else {"reason": "No endpoint passed frozen-feature acceptance"},
    )
    load_checkpoint(
        experiment.output / "checkpoints/purifier.pt", {"purifier": experiment.purifier}
    )
    freeze(experiment.purifier)
    tau = identity_threshold(experiment)
    experiment.supervision = TeacherSupervision(
        experiment.clinical,
        experiment.mfm,
        experiment.reid,
        experiment.purifier,
        experiment.weights,
        tau,
    )


def validate_purifier(experiment):
    from ccrr.training.feature_audit import FeatureAudit

    return FeatureAudit(experiment).run()


def identity_threshold(experiment):
    path = (
        experiment.fit.root
        / "protocol_audit_20261009/pair_protocol/fit_local_identity_negative_pairs.json"
    )
    obj = json.loads(path.read_text())
    pairs = obj.get("pairs", obj) if isinstance(obj, dict) else obj
    values = []
    weights = []
    with torch.no_grad():
        for pair in pairs:
            # Schema is checked explicitly; no guessed pairing or fabricated threshold.
            a = pair.get("rgb_img")
            b = pair.get("gray_img")
            require(
                a in experiment.fit.index and b in experiment.fit.index,
                "Local negative-pair schema requires rgb_img/gray_img",
            )
            ba = collate([experiment.fit[experiment.fit.index[a]]], experiment.device)
            bb = collate([experiment.fit[experiment.fit.index[b]]], experiment.device)
            fa = experiment.reid(ba["x"], ba["rect"])
            fb = experiment.reid(gray(bb["x"]), bb["rect"])
            aa, av = align(fa["spatial"], fa["transform"], experiment.profile.size)
            bf, bv = align(fb["spatial"], fb["transform"], experiment.profile.size)
            s = F.cosine_similarity(aa, bf, dim=1, eps=1e-6)[0]
            w = torch.minimum(av, bv)[0]
            w = w / w.sum()
            values.extend(s.cpu().tolist())
            weights.extend((w * float(pair["pair_weight"])).cpu().tolist())
    tau = weighted_quantile(values, weights)
    write_json(
        experiment.output / "identity_threshold.json",
        {"tau": tau, "pair_manifest": sha256(path)},
    )
    return tau
