"""Stages H–I: joint system optimization and periodic calibration."""

import copy
import numpy as np
import torch
from ccrr.config import Profile
from ccrr.data.pad import collate, rng_for
from ccrr.training.losses import reconstruction, gray_loss, saturation, message_loss
from ccrr.tensor_ops import masked_mean
from ccrr.runtime.support import write_json
from ccrr.training.optimization import optimizer, schedule, update
from ccrr.runtime.checkpoints import save_checkpoint, load_checkpoint
from ccrr.errors import require, CCRRError, Status
import ccrr.training.adversarial as adversarial
import ccrr.calibration.workflow as calibration
import ccrr.evaluation.deployment as deployment


def joint(experiment, stage="H"):
    require(stage in ("H", "I"), "Joint stage")
    total = experiment.cfg["budgets"]["joint_a" if stage == "H" else "joint_b"]
    interval = 2000 if stage == "H" else 500
    if stage == "I":
        require(
            experiment.acceptance.get("H", {}).get("passed"),
            "Joint-B requires accepted Joint-A communication",
        )
    for p in experiment.system.inn.parameters():
        p.requires_grad_(stage == "I")
    for p in experiment.system.entropy.parameters():
        p.requires_grad_(stage == "I")
    modules = [
        experiment.system.comm.carrier,
        experiment.system.comm.writer,
        experiment.system.comm.decoder,
    ]
    rates = [5e-5 if stage == "H" else 1e-5] * 3
    if stage == "I":
        modules.extend([experiment.system.inn, experiment.system.entropy])
        rates.extend([1e-6, 1e-4])
    opt = optimizer(modules, rates)
    for p in experiment.pub_attacker.parameters():
        p.requires_grad_(True)
    advopt = optimizer([experiment.pub_attacker], [1e-4])
    for j in range(1000):
        adversarial.pair_objective(experiment, stage, j, True, advopt)
    step = 0
    attempts = 0
    empty_streak = 0
    best = None
    while step < total and attempts < 100000:
        attempts += 1
        rng = rng_for(experiment.seed, stage, "images", attempts)
        chosen = rng.integers(
            len(experiment.fit), size=experiment.cfg.get("batch_images", 8)
        )
        feasible = _prepare_joint_batch(experiment, chosen, rng)
        if not feasible and stage == "H":
            empty_streak += 1
            require(
                empty_streak < 100,
                "100 capacity-rejected batches; joint stage stopped",
            )
            continue
        empty_streak = 0
        adversarial.adversarial_updates(experiment, stage, step, advopt)
        opt.zero_grad(set_to_none=True)
        values = _accumulate_message_gradients(experiment, feasible)
        if stage == "I":
            _accumulate_representation_gradients(experiment, chosen, rng)
        adversarial.pair_objective(experiment, stage, step)
        schedule(opt, step, total)
        norm = update(opt)
        step += 1
        if step % 100 == 0:
            experiment.log(
                stage,
                step,
                {
                    "loss": np.mean(values) if values else None,
                    "feasible": len(feasible),
                    "sampled": len(chosen),
                    "gradient_norm": norm,
                    "attempts": attempts,
                },
            )
        if step % interval == 0 or step == total:
            experiment.bind_snapshot(f"{stage}-{step}")
            calibration.stage_g(experiment)
            report = deployment.validate_deployment(experiment, stage, step)
            score = (report["both_rate"], -report["mean_bpp"], report["psnr"])
            if report["passed"] and (best is None or score > best[0]):
                best = (score, report, copy.deepcopy(experiment.profile))
                save_checkpoint(
                    experiment.output / "checkpoints" / f"{stage}.pt",
                    {"system": experiment.system},
                    opt,
                    step,
                    stage,
                    report,
                )
    require(step == total, "Joint stage did not reach prescribed updates")
    return _finalize_joint(experiment, stage, best, step)


def _prepare_joint_batch(experiment, chosen, rng):
    feasible = []
    for i in chosen:
        b = collate([experiment.fit[int(i)]], experiment.device)
        try:
            with torch.no_grad():
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
            # Reuse this exact protected table for the gradient pass; no success filtering.
            feasible.append((int(i), sent))
        except CCRRError as exc:
            if exc.status != Status.CAPACITY:
                raise
    return feasible


def _accumulate_message_gradients(experiment, feasible):
    values = []
    for i, cached in feasible:
        b = collate([experiment.fit[i]], experiment.device)
        u, z, g, n, zq = experiment.system.quantized(b["x"], cached["level"])
        require(
            torch.equal(n.detach(), cached["n"]),
            "Latent changed within one optimizer update",
        )
        public, _ = experiment.system.comm.write(
            g, cached["bits"], cached["occupancy"], b["legal"]
        )
        logits = experiment.system.comm.read(public)
        dec, _ = message_loss(logits, cached["bits"], cached["occupancy"])
        rec = experiment.system.recover(public, zq)
        target = experiment.supervision.targets(b["x"], b["rect"], b["label"])
        clin, ident, _ = experiment.supervision.loss(
            public, b["rect"], b["label"], target
        )
        loss = (
            dec
            + masked_mean((public - g).abs() / (4 / 255), b["v"])
            + reconstruction(b["x"], rec, b["v"])
            + clin
            + 0.1 * experiment.cfg["lambda_privacy"] * ident
        )
        (loss / len(feasible)).backward()
        values.append(float(loss.detach()))
    return values


def _accumulate_representation_gradients(experiment, chosen, rng):
    for i in chosen:
        b = collate([experiment.fit[int(i)]], experiment.device)
        r = int(rng.integers(1, 4))
        u, z, g, n, zq = experiment.system.quantized(b["x"], r)
        target = experiment.supervision.targets(b["x"], b["rect"], b["label"])
        clin, ident, _ = experiment.supervision.loss(g, b["rect"], b["label"], target)
        loss = (
            reconstruction(b["x"], experiment.system.recover(g, zq), b["v"])
            + 0.1 * gray_loss(g, b["x"], b["v"])
            + 1e-3 * saturation(u, b["v"])
            + clin
            + 0.1 * experiment.cfg["lambda_privacy"] * ident
            + experiment.cfg["lambda_rate"]
            * experiment.system.entropy.rate(z, r, experiment.profile.step(r)).mean()
        )
        (loss / len(chosen)).backward()


def _finalize_joint(experiment, stage, best, step):
    if stage == "I":
        from ccrr.evaluation.joint_b import accepts

        baseline = experiment.acceptance["H"]["report"]
        decision = (
            accepts(baseline, best[1])
            if best
            else {
                "accepted": False,
                "reason": "No Joint-B candidate passed base targets",
            }
        )
        write_json(experiment.output / "joint_b_comparison.json", decision)
        if not decision["accepted"]:
            load_checkpoint(
                experiment.output / "checkpoints/H.pt", {"system": experiment.system}
            )
            experiment.profile = Profile.from_dict(baseline["profile"])
            experiment.system.profile = experiment.profile
            experiment.bind_snapshot("H-restored-after-I")
            experiment.acceptance["I"] = {
                "passed": False,
                "report": decision,
                "action": "Restored accepted Joint-A",
            }
            write_json(experiment.output / "acceptance.json", experiment.acceptance)
            return
    experiment.mark(
        stage,
        best is not None,
        best[1] if best else {"reason": "No admissible deployment checkpoint"},
    )
    load_checkpoint(
        experiment.output / "checkpoints" / f"{stage}.pt", {"system": experiment.system}
    )
    experiment.profile = best[2]
    experiment.system.profile = experiment.profile
    experiment.bind_snapshot(f"{stage}-selected")
    calibration.stage_g(experiment, final=True)
    final_report = deployment.validate_deployment(experiment, stage + "-final", step)
    if stage == "I":
        from ccrr.evaluation.joint_b import accepts

        decision = accepts(experiment.acceptance["H"]["report"], final_report)
        if not decision["accepted"]:
            baseline = experiment.acceptance["H"]["report"]
            load_checkpoint(
                experiment.output / "checkpoints/H.pt", {"system": experiment.system}
            )
            experiment.profile = Profile.from_dict(baseline["profile"])
            experiment.system.profile = experiment.profile
            experiment.bind_snapshot("H-restored-after-final-I")
            experiment.acceptance["I"] = {
                "passed": False,
                "report": decision,
                "action": "Restored Joint-A after final calibration",
            }
            write_json(experiment.output / "acceptance.json", experiment.acceptance)
            return
    experiment.mark(stage, final_report["passed"], final_report)
    save_checkpoint(
        experiment.output / "checkpoints" / f"{stage}.pt",
        {"system": experiment.system},
        None,
        step,
        stage,
        final_report,
    )
    experiment.bind_snapshot(f"{stage}-final-accepted")
