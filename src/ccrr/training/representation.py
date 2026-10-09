"""Stages C–E: quantized recovery, representation and multirate INN optimization."""

import numpy as np
from ccrr.data.pad import rng_for
from ccrr.training.losses import (
    reconstruction,
    gray_loss,
    saturation,
)
from ccrr.training.optimization import optimizer, schedule, update
from ccrr.runtime.checkpoints import save_checkpoint, load_checkpoint
from ccrr.errors import require
import ccrr.training.adversarial as adversarial
import ccrr.evaluation.selection as validation


def train_representation(experiment, stage):
    require(stage in ("C", "D", "E"), "Representation stage")
    if stage != "C":
        require(
            experiment.supervision is not None,
            "Validated teacher/Purifier bundle required",
        )
    for p in experiment.system.inn.parameters():
        p.requires_grad_(True)
    for p in experiment.system.entropy.parameters():
        p.requires_grad_(stage != "C")
    name = {"C": "recovery", "D": "representation", "E": "multirate"}[stage]
    total = experiment.cfg["budgets"][name]
    opt = optimizer(
        [experiment.system.inn, experiment.system.entropy],
        [5e-5 if stage == "E" else 1e-4, 5e-4 if stage == "E" else 1e-3],
    )

    for p in experiment.pub_attacker.parameters():
        p.requires_grad_(True)
    advopt = optimizer([experiment.pub_attacker], [1e-4])
    best = None
    selected = None
    interval = 1000 if stage == "C" else 2000
    if stage == "D":
        for p in experiment.pub_attacker.parameters():
            p.requires_grad_(True)
        for j in range(1000):
            adversarial.pair_objective(experiment, "D", j, True, advopt)
    for step in range(total):
        experiment.system.train()
        rng = rng_for(experiment.seed, stage, "images", step)
        if stage != "C":
            adversarial.adversarial_updates(experiment, stage, step, advopt)
        opt.zero_grad(set_to_none=True)
        loss_values = []
        level = int(rng.integers(1, 4)) if stage == "E" else 1
        for _ in range(experiment.cfg.get("batch_images", 8)):
            b = experiment.batch(rng)
            u, z, g, n, zq = experiment.system.quantized(b["x"], level)
            rec = experiment.system.recover(g, zq)
            loss = (
                reconstruction(b["x"], rec, b["v"])
                + 0.1 * gray_loss(g, b["x"], b["v"])
                + 1e-3 * saturation(u, b["v"])
            )
            if stage != "C":
                target = experiment.supervision.targets(b["x"], b["rect"], b["label"])
                clin, ident, _ = experiment.supervision.loss(
                    g, b["rect"], b["label"], target
                )
                rate = experiment.system.entropy.rate(
                    z, level, experiment.profile.step(level)
                ).mean()
                ramp = min(1, (step + 1) / 2000) if stage == "D" else 1
                loss = loss + ramp * (
                    clin
                    + 0.1 * experiment.cfg["lambda_privacy"] * ident
                    + experiment.cfg["lambda_rate"] * rate
                )
            (loss / experiment.cfg.get("batch_images", 8)).backward()
            loss_values.append(float(loss.detach()))
        if stage != "C" and not experiment.system.fixed_gray:
            adversarial.pair_objective(experiment, stage, step)
        schedule(opt, step, total)
        norm = update(opt)
        if (step + 1) % 100 == 0:
            experiment.log(
                stage,
                step + 1,
                {
                    "loss": np.mean(loss_values),
                    "gradient_norm": norm,
                    "level": level,
                },
            )
        if (step + 1) % interval == 0 or step + 1 == total:
            if stage == "E":
                experiment.bind_snapshot(f"E-{step + 1}")
            report = validation.validate_recovery(
                experiment, stage, step + 1, (1, 2, 3) if stage == "E" else (1,)
            )
            if stage in ("D", "E") and not experiment.system.fixed_gray:
                report["semantic_screen"] = (
                    validation.validate_representation_semantics(experiment)
                )
                report["passed_quality"] = (
                    report["passed_quality"] and report["semantic_screen"]["passed"]
                )
            value = (
                (
                    -report.get("ideal_coverage", 0),
                    report.get("mean_protected_bpp", 0),
                    -report["mean_psnr"],
                )
                if stage == "E"
                else (-report["mean_psnr"],)
            )
            if report["passed_quality"] and (best is None or value < best):
                best = value
                selected = report
                save_checkpoint(
                    experiment.output / "checkpoints" / f"{stage}.pt",
                    {"system": experiment.system},
                    opt,
                    step + 1,
                    stage,
                    report,
                )
    require(selected is not None, f"{stage}: no quality-qualified checkpoint")
    load_checkpoint(
        experiment.output / "checkpoints" / f"{stage}.pt", {"system": experiment.system}
    )
    experiment.mark(
        stage,
        selected["passed_quality"]
        and (stage != "E" or selected["ideal_coverage"] >= 0.95),
        selected,
    )
    if stage == "E":
        experiment.bind_snapshot("E-selected")
