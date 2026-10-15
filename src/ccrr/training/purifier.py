"""Stage B: Purifier optimization and independent feature acceptance."""

import json
import torch
from torch.nn import functional as F
from ccrr.config import sha256
from ccrr.data.pad import collate, rng_for
from ccrr.teachers.adapters import (
    freeze,
)
from ccrr.tensor_ops import gray
from ccrr.representation.supervision import align
from ccrr.runtime.support import write_json
from ccrr.training.optimization import optimizer, update
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
    from ccrr.training.b_batches import TeacherBank, train_step
    from ccrr.training.b_state import BProgress
    from ccrr.training.optimization import TeacherSchedule
    import time

    e = experiment
    from ccrr.training.b_resources import verify_execution_report

    verify_execution_report(e.cfg)
    require(
        e.mfm is not None and bool(e.purifier.stats_ready), "Teacher stages required"
    )
    from ccrr.training.b_initialization import record_start

    record_start(e)
    e.purifier.train().requires_grad_(True)
    e.sem_attacker.train().requires_grad_(True)
    opt = optimizer([e.purifier], [1e-4])
    advopt = optimizer([e.sem_attacker], [1e-4])
    total = e.cfg["budgets"]["purifier"]
    scheduler, advscheduler = (
        TeacherSchedule(opt, total),
        TeacherSchedule(advopt, 2 * total),
    )
    # Build before restoring RNG; fixed eval teachers, canonical cohorts, no sampling.
    bank = TeacherBank(e)
    e.b_teacher_bank = bank
    progress = BProgress(e, opt, advopt, scheduler, advscheduler)
    progress.check_endpoints()
    if not progress.path.exists():
        progress.save()
    print(
        f"B_TRAIN_START: step={scheduler.next_step}, microbatch={e.cfg.get('microbatch', 1)}, cache_bytes={bank.bytes}",
        flush=True,
    )
    for step in range(scheduler.next_step, total):
        report = train_step(e, bank, step, opt, advopt, scheduler, advscheduler)
        if (step + 1) % 100 == 0 or step == 0:
            e.log("B", step + 1, report)
            print(
                f"B_STEP: {step + 1}/{total} seconds={report['seconds_per_step']:.3f}",
                flush=True,
            )
        endpoint = (step + 1) % 1000 == 0 or step + 1 == total
        if endpoint or (step + 1) % e.cfg.get("resume_interval", 100) == 0:
            progress.save(endpoint=endpoint)
    if progress.phase == "train":
        progress.phase = "audit"
        progress.save()
    from ccrr.training.b_control import prepare_control

    prepare_control(e, bank)
    # Audit transactions: a completed endpoint is never retrained on resume.
    # An interrupted current endpoint is replayed from the prior saved RNG boundary.
    for step in range(1000, total + 1, 1000):
        if step in progress.audit_done:
            continue
        load_checkpoint(
            e.output / "checkpoints" / f"purifier-{step}.pt",
            {"purifier": e.purifier, "attacker": e.sem_attacker},
        )
        freeze(e.purifier)
        e.feature_audit_tag = f"step-{step}"
        print(f"B_AUDIT_START: endpoint={step}", flush=True)
        start = time.perf_counter()
        report = validate_purifier(e)
        e.log(
            "B-audit",
            step,
            {"seconds": time.perf_counter() - start, "passed": report["passed"]},
        )
        if report["passed"]:
            clinical_min = min(
                r["macro_auroc"] for r in report["clinical"]["readouts"]["sem_linear"]
            )
            strongest = max(r["sem_strongest"] for r in report["identity"]["checks"])
            key = [clinical_min, -strongest, -report["l_all"], -step]
            if progress.best is None or tuple(key) > tuple(progress.best["key"]):
                progress.best = {"key": key, "step": step, "report": report}
        progress.audit_done.append(step)
        progress.save()
    require(progress.best is not None, "No endpoint passed frozen-feature acceptance")
    best = progress.best
    load_checkpoint(
        e.output / "checkpoints" / f"purifier-{best['step']}.pt",
        {"purifier": e.purifier},
    )
    freeze(e.purifier)
    save_checkpoint(
        e.output / "checkpoints/purifier.pt",
        {"purifier": e.purifier},
        None,
        best["step"],
        "B",
        best["report"],
    )
    write_json(
        e.output / "b_scientific_status.json",
        {
            "feature_acceptance_passed": True,
            "protocol_complete": True,
            "no_branch_control_sha256": e.no_branch_control_sha256,
            "local_heatmap_ranking_validated": False,
            "limitations": [
                "Feature purification acceptance only; section 8.6 heatmap weighting and downstream image privacy remain unvalidated"
            ],
        },
    )
    e.mark(
        "B",
        True,
        {
            **best["report"],
            "protocol_complete": True,
            "no_branch_control_sha256": e.no_branch_control_sha256,
            "local_heatmap_ranking_validated": False,
        },
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
