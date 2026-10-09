"""Stage F: Writer and blind Decoder warm-up."""

import numpy as np
import torch
from ccrr.data.pad import collate, rng_for
from ccrr.teachers.adapters import (
    freeze,
)
from ccrr.training.losses import message_loss
from ccrr.tensor_ops import masked_mean
from ccrr.runtime.support import write_json
from ccrr.training.optimization import optimizer, schedule, update
from ccrr.runtime.checkpoints import save_checkpoint, load_checkpoint
from ccrr.errors import require
from ccrr.calibration.probes import (
    ProbeGenerator,
    probe_region,
    select_regions,
)


def stage_f(experiment):
    freeze(experiment.system.inn)
    freeze(experiment.system.entropy)
    experiment.system.comm.train()
    total = experiment.cfg["budgets"]["communication"]
    opt = optimizer(
        [
            experiment.system.comm.carrier,
            experiment.system.comm.writer,
            experiment.system.comm.decoder,
        ],
        [1e-4] * 3,
    )
    generator = ProbeGenerator(
        experiment.profile, experiment.system.digest, experiment.allocator
    )
    best = None
    for step in range(total):
        rng = rng_for(experiment.seed, "F", "records", step)
        opt.zero_grad(set_to_none=True)
        values = []
        maxload = 1 if step < 5000 else 2 if step < 10000 else 4
        for _ in range(experiment.cfg.get("batch_images", 8)):
            b = experiment.batch(rng)
            with torch.no_grad():
                g = experiment.system.decompose(b["x"])[2]
            loads = (
                np.zeros(experiment.profile.regions, int)
                if rng.random() < 0.1
                else rng.integers(0, maxload + 1, experiment.profile.regions)
                * b["legal"][0].cpu().numpy().astype(int)
            )
            bits, occ = generator.table(b["metadata"][0], loads, rng)
            bt = torch.tensor(bits, device=experiment.device)[None]
            ot = torch.tensor(occ, device=experiment.device)[None]
            p, _ = experiment.system.comm.write(g, bt, ot, b["legal"])
            logits = experiment.system.comm.read(p)
            dec, _ = message_loss(logits, bt, ot)
            loss = dec + masked_mean((p - g).abs() / (4 / 255), b["v"])
            (loss / experiment.cfg.get("batch_images", 8)).backward()
            values.append(float(loss.detach()))
        schedule(opt, step, total)
        norm = update(opt)
        if (step + 1) % 100 == 0:
            experiment.log(
                "F",
                step + 1,
                {
                    "loss": np.mean(values),
                    "gradient_norm": norm,
                    "maximum_load": maxload,
                },
            )
        if (step + 1) % 2000 == 0 or step + 1 == total:
            save_checkpoint(
                experiment.output / "checkpoints" / f"F-{step + 1}.pt",
                {"system": experiment.system},
                opt,
                step + 1,
                "F",
                {},
            )
    candidates = sorted(
        (experiment.output / "checkpoints").glob("F-*.pt"),
        key=lambda p: int(p.stem.split("-")[-1]),
    )
    for path in candidates:
        checkpoint = load_checkpoint(path, {"system": experiment.system})
        if checkpoint["step"] < 10000:
            continue
        report = validate_communication_warmup(experiment, checkpoint["step"])
        score = (
            report["full_table_success"],
            -report["empty_false_rate"],
            -report["mean_modification"],
            -checkpoint["step"],
        )
        if best is None or score > best:
            best = score
            save_checkpoint(
                experiment.output / "checkpoints/F.pt",
                {"system": experiment.system},
                None,
                checkpoint["step"],
                "F",
                report,
            )
    require(best is not None, "No last-curriculum communication checkpoint")
    load_checkpoint(
        experiment.output / "checkpoints/F.pt", {"system": experiment.system}
    )
    experiment.bind_snapshot("F-selected")


def validate_communication_warmup(experiment, step):
    experiment.system.eval()
    generator = ProbeGenerator(
        experiment.profile, experiment.system.digest, experiment.allocator
    )
    successes = []
    empty_errors = []
    modifications = []
    for i in range(len(experiment.data["val_tune"])):
        b = collate([experiment.data["val_tune"][i]], experiment.device)
        legal = b["legal"][0].cpu().numpy()
        g = experiment.system.decompose(b["x"])[2]
        for region in select_regions(b["id"][0], legal, 0):
            for k in range(5):
                rng = rng_for(
                    experiment.seed, "F-validation", b["id"][0] + str(region), k
                )
                result = probe_region(
                    experiment.system.comm,
                    g,
                    legal,
                    b["metadata"][0],
                    b["rect"],
                    experiment.clinical,
                    generator,
                    region,
                    k,
                    32,
                    rng,
                    experiment.profile.occupancy_threshold,
                )
                if k:
                    successes.extend(result["success"])
                else:
                    empty_errors.extend(not v for v in result["success"])
                modifications.extend(result["rmse"])
    report = {
        "full_table_success": float(np.mean(successes)),
        "empty_false_rate": float(np.mean(empty_errors)),
        "mean_modification": float(np.mean(modifications)),
        "step": step,
    }
    write_json(experiment.output / "validation" / f"F-{step}.json", report)
    return report
