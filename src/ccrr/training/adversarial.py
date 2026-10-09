"""Alternating public-image adversary updates shared by representation and joint stages."""

import torch
from torch.nn import functional as F
from ccrr.data.pad import collate, rng_for
from ccrr.teachers.adapters import (
    freeze,
    freeze_bn,
)
from ccrr.training.optimization import update
from ccrr.errors import require, CCRRError, Status
from collections import Counter


def pair_public(experiment, b, stage, rng):
    if stage in ("D", "E"):
        return experiment.system.decompose(b["x"])[2]
    require(b["id"][0] in experiment.policies, "Missing snapshot-bound sender policy")
    credentials = experiment.allocator.new_object(experiment.system.digest)
    sent = experiment.system.send(
        b["x"],
        b["metadata"][0],
        b["legal"],
        experiment.risks[b["id"][0]],
        credentials,
        experiment.allocator,
        experiment.policies[b["id"][0]],
        rng.permutation(experiment.profile.legal_levels).tolist(),
    )
    return sent["public"]


def pair_objective(experiment, stage, step, attacker_update=False, opt=None, sub=0):
    rng = rng_for(
        experiment.seed,
        stage,
        "attack" if attacker_update else "generator-pairs",
        step * 3 + sub,
    )
    pairs = experiment.sampler.sample(rng)
    feasible = []
    # Determine feasibility from complete source rate before forming the two means.
    for a, b, s, scenario in pairs:
        domain = int(rng.integers(2))
        ba = collate([experiment.fit[a]], experiment.device)
        bb = collate([experiment.fit[b]], experiment.device)
        try:
            with torch.no_grad():
                xa = pair_public(experiment, ba, stage, rng)
                if domain:
                    xb = pair_public(experiment, bb, stage, rng)
            feasible.append((a, b, s, domain, scenario))
        except CCRRError as exc:
            if exc.status != Status.CAPACITY:
                raise
    counts = Counter([s for _, _, s, _, _ in feasible])
    if not counts.get(0) or not counts.get(1):
        return False
    if attacker_update:
        opt.zero_grad(set_to_none=True)
    for a, b, s, domain, scenario in feasible:
        ba = collate([experiment.fit[a]], experiment.device)
        bb = collate([experiment.fit[b]], experiment.device)
        if attacker_update:
            with torch.no_grad():
                xa = pair_public(experiment, ba, stage, rng)
                xb = pair_public(experiment, bb, stage, rng) if domain else bb["x"]
        else:
            xa = pair_public(experiment, ba, stage, rng)
            xb = pair_public(experiment, bb, stage, rng) if domain else bb["x"]
        fa = experiment.pub_attacker(xa)
        fb = experiment.pub_attacker(xb)
        logit = experiment.pub_attacker.score(fa, fb)
        loss = F.binary_cross_entropy_with_logits(logit, logit.new_tensor([s])) / (
            2 * counts[s]
        )
        if not attacker_update:
            loss = (
                -experiment.cfg.get("lambda_privacy", 0.1)
                * (min(1, (step + 1) / 2000) if stage == "D" else 1)
                * loss
            )
        loss.backward()
    if attacker_update:
        update(opt)
    return True


def adversarial_updates(experiment, stage, step, opt):
    experiment.pub_attacker.train()
    freeze_bn(experiment.pub_attacker)
    for p in experiment.pub_attacker.parameters():
        p.requires_grad_(True)
    for sub in range(2):
        valid = False
        for attempt in range(100):
            if pair_objective(experiment, stage, step * 100 + attempt, True, opt, sub):
                valid = True
                break
        require(valid, "100 invalid attacker pair batches; stop stage")
    freeze(experiment.pub_attacker)
