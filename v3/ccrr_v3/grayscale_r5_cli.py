"""R5 explicit register / verify / plan / train / resume / report. No auto-start."""

import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import math
import os
from pathlib import Path
import time

import torch

from ccrr.data.pad import PADData, collate
from . import grayscale_dev as old
from .grayscale import public_forward, update
from .grayscale_r5 import (
    CONFIG,
    configure_numerics,
    Engine,
    atomic_json,
    diagnose,
    pack_symbols,
    unpack_symbols,
    sync,
)
from .runtime import (
    artifact,
    digest,
    read,
    new_run,
    finish,
    status,
    log,
    validate_resources,
)
from .dev2 import ROOT, current_commit, check_hashes

PLAN = ROOT / "plans/grayscale-r5-direct16-steps-20261020"


def registration():
    reg = read(PLAN / "registration.json")
    if reg["state"] != "registered" or reg["source_commit"] != current_commit():
        raise ValueError("Use the registered R5 release")
    if (
        read(PLAN / "config.json") != CONFIG
        or digest(PLAN / "config.json") != reg["config_sha256"]
    ):
        raise ValueError("Registered configuration changed")
    check_hashes(reg["parents"])
    validate_resources(reg["resources"])
    return reg


def base_parents(reg):
    return {
        **reg["parents"],
        str(PLAN / "registration.json"): digest(PLAN / "registration.json"),
    }


def make_engine(resources, step, device, variant):
    teacher = old.load_teacher(resources, step, device)
    return Engine(
        teacher, PADData(resources["data_root"], "fit"), step, device, variant
    )


def verify(output, resources, step, device):
    # Reuses live PanDerm/A0 equality, each-domain gradient, PNG and frozen-state checks.
    configure_numerics()
    a0_binding = read(old.FEATURES / "binding.json")
    if a0_binding["config"]["batch_size"] != CONFIG["batch_size"]:
        raise ValueError("A0 and live verification batch must match the registered16")
    files = old.verify(
        output, resources, step, device, validation_batch_size=CONFIG["batch_size"]
    )
    engine = make_engine(resources, step, device, "learned")
    before = old.state_digest(engine.model)
    b = collate(
        [engine.fit[i % len(engine.fit)] for i in range(CONFIG["batch_size"])], device
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    started = time.perf_counter()
    targets = old.teacher_targets(engine.teacher, b)
    ed = update(
        engine.model,
        engine.teacher,
        b,
        targets,
        engine.weights,
        engine.ed,
        "ed",
        canonical_padding=True,
        step_optimizer=False,
    )
    p = update(
        engine.model,
        engine.teacher,
        b,
        targets,
        engine.weights,
        engine.p,
        "p",
        canonical_padding=True,
        step_optimizer=False,
        inspect_gradients=True,
    )
    sync(device)
    elapsed = time.perf_counter() - started
    assert ed["grad_e"] > 0 and ed["grad_d"] > 0 and p["grad_p"] > 0
    assert old.state_digest(engine.model) == before
    assert old.state_digest(engine.teacher) == engine.teacher_hash
    assert not engine.ed.state and not engine.p.state
    with torch.no_grad():
        out = public_forward(engine.model, b["x"][:1], b["v"][:1])
        public = old.png_roundtrip(out["gray"], output / "r5-gray.png")
        body = pack_symbols(out["q"], 8)
        (output / "private-body-8bit.bin").write_bytes(body)
        q = unpack_symbols((output / "private-body-8bit.bin").read_bytes(), 8, device)
        _, rec = engine.model.decoder(public, q.float() / 255, 8)
        assert torch.equal(q, out["q"]) and torch.allclose(
            rec, out["rgb"], atol=1e-6, rtol=1e-6
        )
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    files += diagnose(
        engine,
        PADData(resources["data_root"], "val_tune"),
        output,
        "initial-bound-head",
    )
    nparams = sum(p.numel() for p in engine.model.parameters())
    report = dict(
        supervisor_step=step,
        batch_size=16,
        accumulation_steps=1,
        numeric_policy=CONFIG["numeric_policy"],
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        verified=True,
        model_parameters_updated=False,
        training_started=False,
        full_batch_ed_p_backward_seconds=elapsed,
        timing_scope="forward_backward_no_optimizer_step_estimate_not_training_throughput",
        allocated_peak_bytes=peak,
        initial_diagnostic_seconds=engine.elapsed_seconds,
        parameter_count=nparams,
        estimated_snapshot_bytes=12 * nparams + 1048576,
        snapshot_estimate_scope="model_plus_two_adam_moments_plus_margin_not_measured_full_optimizer_file",
        original_bound_head_diagnostics="initial-bound-head-diagnostic.json",
        teacher_qualified_by_this_check=False,
        ed=ed,
        public=p,
        real_pixel_supervision_verified=True,
        private_body_roundtrip=True,
        scientific_acceptance="not_evaluated",
    )
    atomic_json(output / "r5-verification.json", report)
    return files + ["r5-gray.png", "private-body-8bit.bin", "r5-verification.json"]


def checked_verification(path, reg, step):
    _, binding = artifact(path, "r5_pixel_verification", reg["resources"])
    if binding["source_commit"] != reg["source_commit"] or binding[
        "parents"
    ] != base_parents(reg):
        raise ValueError("Verification source/parents mismatch")
    report = read(Path(path) / "r5-verification.json")
    if (
        report["supervisor_step"] != step
        or not report["verified"]
        or report["model_parameters_updated"]
    ):
        raise ValueError("Verification is not for this frozen supervisor")
    return report


def plan(args, reg):
    report = checked_verification(args.verification, reg, args.supervisor_step)
    if args.steps_only:
        if args.wall_hours is not None or args.checkpoint_gib is not None:
            raise ValueError("steps-only cannot be combined with resource ceilings")
    elif not all(
        x is not None and math.isfinite(x) and x > 0
        for x in (args.wall_hours, args.checkpoint_gib)
    ):
        raise ValueError("Use steps-only or finite positive resource ceilings")
    now = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = PLAN / f"budget-s{args.supervisor_step}-{args.variant}-{now}.json"
    learned = args.variant == "learned"
    budget = dict(
        schema="r5-budget-2",
        resource_policy="steps_only" if args.steps_only else "bounded",
        source_commit=reg["source_commit"],
        config=CONFIG,
        parents=base_parents(reg),
        supervisor_step=args.supervisor_step,
        variant=args.variant,
        verification=str(Path(args.verification).resolve()),
        verification_sha256=digest(Path(args.verification) / "artifact.json"),
        wall_seconds=None if args.steps_only else args.wall_hours * 3600,
        checkpoint_bytes=None if args.steps_only else int(args.checkpoint_gib * 2**30),
        max_optimizer_attempts=CONFIG["baseline_updates"]
        + CONFIG["c1_cycles"] * (2 if learned else 1),
        max_sampled_positions=(CONFIG["baseline_updates"] + CONFIG["c1_cycles"]) * 16,
        checkpoint_count=48,
        full_config=CONFIG,
        measured_verification=report,
        automatic_training=False,
        note="Step counts remain fixed. Null resource ceilings disable wall/storage stops; costs are still recorded.",
    )
    atomic_json(path, budget)
    print("R5_BUDGET:", path, "training_started=False", flush=True)
    return path


@contextmanager
def run_lock(run):
    with (Path(run) / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                "This exact run is active; no duplicate/resume from active run"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def wall_reached(elapsed, budget):
    limit = budget["wall_seconds"]
    return limit is not None and elapsed >= limit


def valid_resource_policy(budget):
    if budget.get("schema") != "r5-budget-2":
        return False
    wall, storage = budget.get("wall_seconds"), budget.get("checkpoint_bytes")
    if budget.get("resource_policy") == "steps_only":
        return wall is None and storage is None
    return (
        budget.get("resource_policy") == "bounded"
        and isinstance(wall, (float, int))
        and not isinstance(wall, bool)
        and math.isfinite(wall)
        and wall > 0
        and isinstance(storage, int)
        and not isinstance(storage, bool)
        and storage > 0
    )


class StorageCeiling(Exception):
    pass


def save_checkpoint(engine, output, payload, used_bytes, max_bytes=None):
    """Commit a complete cycle via tensor file then hash sidecar; never overwrite an endpoint."""
    phase, cycle, _ = engine.phase_at(engine.position)
    path = Path(output) / f"generator-{phase}-{cycle:06d}.pt"
    if path.exists() or path.with_suffix(".json").exists():
        raise ValueError("Refuse checkpoint overwrite")
    sync(engine.device)
    started = time.perf_counter()
    state = engine.state()
    temp = path.with_suffix(".pt.tmp")
    with temp.open("wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    if max_bytes is not None and used_bytes + temp.stat().st_size > max_bytes:
        temp.unlink()
        engine.elapsed_seconds += time.perf_counter() - started
        raise StorageCeiling(
            "Checkpoint would exceed the registered persistent storage budget"
        )
    os.replace(temp, path)
    engine.elapsed_seconds += time.perf_counter() - started
    manifest = dict(
        schema="r5-checkpoint-1",
        checkpoint=path.name,
        sha256=digest(path),
        binding_sha256=digest(Path(output) / "binding.json"),
        position=engine.position,
        elapsed_seconds=engine.elapsed_seconds,
        attempted_updates=engine.attempted_updates,
        checkpoint_bytes=used_bytes + path.stat().st_size,
        diagnostics={name: digest(Path(output) / name) for name in payload},
    )
    atomic_json(path.with_suffix(".json"), manifest)
    atomic_json(
        Path(output) / "latest.json",
        dict(checkpoint=path.name, manifest_sha256=digest(path.with_suffix(".json"))),
    )
    # Directory fsync makes committed file names durable on supported local filesystems.
    fd = os.open(output, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return path, manifest


def read_checkpoint(path, budget, reg):
    path = Path(path).resolve()
    with run_lock(path.parent):
        latest = read(path.parent / "latest.json")
        if latest["checkpoint"] != path.name or latest["manifest_sha256"] != digest(
            path.with_suffix(".json")
        ):
            raise ValueError("Resume only the latest committed full cycle")
        manifest = read(path.with_suffix(".json"))
        binding = read(path.parent / "binding.json")
        if manifest["sha256"] != digest(path) or manifest["binding_sha256"] != digest(
            path.parent / "binding.json"
        ):
            raise ValueError("Checkpoint or original binding changed")
        if (
            binding["config"] != CONFIG
            or binding["source_commit"] != reg["source_commit"]
        ):
            raise ValueError("Resume source or configuration mismatch")
        if binding["resources"] != reg["resources"] or binding["options"][
            "budget_sha256"
        ] != argsafe_budget_hash(budget):
            raise ValueError("Resume budget/resources mismatch")
        check_hashes(binding["parents"])
        check_hashes(
            {str(path.parent / k): v for k, v in manifest["diagnostics"].items()}
        )
        state = torch.load(path, map_location="cpu", weights_only=True)
        if (
            manifest["schema"] != "r5-checkpoint-1"
            or manifest["checkpoint"] != path.name
            or manifest["position"] != state["position"]
        ):
            raise ValueError("Checkpoint manifest/state mismatch")
        costs = read(path.parent / "cost.json")
        if costs.get("operation_in_progress", False):
            raise ValueError(
                "Unclean interruption: incomplete cost journal; reconcile costs before resume"
            )
        # Preserve costs of work after the saved boundary, even if its weights are discarded.
        state["attempted_updates"] = max(
            state["attempted_updates"], costs["attempted_updates"]
        )
        state["elapsed_seconds"] = max(
            state["elapsed_seconds"],
            costs["elapsed_seconds"],
            manifest["elapsed_seconds"],
        )
        costs["checkpoint_bytes"] = max(
            costs["checkpoint_bytes"], manifest["checkpoint_bytes"]
        )
        return state, costs


def argsafe_budget_hash(budget):
    # Hash content, independent of pretty printing; file hash is separately a parent.
    from .grayscale_r5 import json_hash

    return json_hash(budget)


def cost_record(engine, used_bytes, sampled_attempts, in_progress=False):
    return dict(
        operation_in_progress=in_progress,
        position=engine.position,
        attempted_updates=engine.attempted_updates,
        elapsed_seconds=engine.elapsed_seconds,
        checkpoint_bytes=used_bytes,
        sampled_positions_attempted=sampled_attempts,
        successful_ed_updates=engine.ed_updates,
        successful_p_updates=engine.p_updates,
    )


def run_training(
    output,
    resources,
    step,
    device,
    variant,
    budget,
    resume_state=None,
    previous_costs=None,
):
    started = time.perf_counter()
    engine = make_engine(resources, step, device, variant)
    tune = PADData(resources["data_root"], "val_tune")
    used_bytes, sampled_attempts = 0, 0
    if resume_state:
        engine.restore(resume_state)
        used_bytes = previous_costs["checkpoint_bytes"]
        sampled_attempts = previous_costs["sampled_positions_attempted"]
    prior_elapsed = engine.elapsed_seconds

    def account_wall():
        sync(device)
        engine.elapsed_seconds = prior_elapsed + time.perf_counter() - started

    payload, stop_reason = [], "budget_completed"
    account_wall()
    atomic_json(output / "cost.json", cost_record(engine, used_bytes, sampled_attempts))
    try:
        if not resume_state:
            atomic_json(
                output / "cost.json",
                cost_record(engine, used_bytes, sampled_attempts, True),
            )
            payload.extend(diagnose(engine, tune, output, "initial"))
            account_wall()
        while engine.position < engine.total:
            account_wall()
            phase, cycle, _ = engine.phase_at(engine.position + 1)
            next_updates = 2 if phase == "c1" and variant == "learned" else 1
            if wall_reached(engine.elapsed_seconds, budget):
                stop_reason = "wall_ceiling"
                break
            if (
                engine.attempted_updates + next_updates
                > budget["max_optimizer_attempts"]
            ):
                stop_reason = "optimizer_attempt_ceiling_including_discarded_work"
                break
            if (
                sampled_attempts + CONFIG["batch_size"]
                > budget["max_sampled_positions"]
            ):
                stop_reason = "sample_ceiling_including_discarded_work"
                break
            sampled_attempts += CONFIG["batch_size"]
            atomic_json(
                output / "cost.json",
                cost_record(engine, used_bytes, sampled_attempts, True),
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            info = engine.cycle(
                inspect_gradients=(
                    cycle == 1 or cycle % CONFIG["checkpoint_every"] == 0
                )
            )
            account_wall()
            info["allocated_peak_bytes"] = (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None
            )
            log(output, "r5_cycle", **info)
            atomic_json(
                output / "cost.json", cost_record(engine, used_bytes, sampled_attempts)
            )
            _, _, total = engine.phase_at(engine.position)
            checkpoint_due = cycle % CONFIG["checkpoint_every"] == 0 or cycle == total
            operator_stop = (
                output / "STOP_AFTER_CHECKPOINT"
            ).exists() and checkpoint_due
            if checkpoint_due or wall_reached(engine.elapsed_seconds, budget):
                atomic_json(
                    output / "cost.json",
                    cost_record(engine, used_bytes, sampled_attempts, True),
                )
                label = f"{phase}-{cycle:06d}"
                diagnostics = []
                if not wall_reached(engine.elapsed_seconds, budget):
                    diagnostics = diagnose(engine, tune, output, label)
                    payload.extend(diagnostics)
                    account_wall()
                # Serialize to know exact bytes. Never discard prior checkpoints to make room.
                try:
                    ckpt, manifest = save_checkpoint(
                        engine,
                        output,
                        diagnostics,
                        used_bytes,
                        budget["checkpoint_bytes"],
                    )
                except StorageCeiling:
                    stop_reason = "checkpoint_storage_ceiling_no_new_checkpoint"
                    break
                account_wall()
                used_bytes = manifest["checkpoint_bytes"]
                payload.extend([ckpt.name, ckpt.with_suffix(".json").name])
                atomic_json(
                    output / "cost.json",
                    cost_record(engine, used_bytes, sampled_attempts),
                )
                log(
                    output,
                    "r5_checkpoint",
                    file=ckpt.name,
                    position=engine.position,
                    selected=False,
                )
                if (
                    budget["checkpoint_bytes"] is not None
                    and used_bytes >= budget["checkpoint_bytes"]
                ):
                    stop_reason = "checkpoint_storage_ceiling"
                    break
                if operator_stop:
                    stop_reason = "operator_stop_at_checkpoint"
                    break
        assert old.state_digest(engine.teacher) == engine.teacher_hash
        assert all(p.grad is None for p in engine.teacher.parameters())
        account_wall()
        summary = {
            **cost_record(engine, used_bytes, sampled_attempts),
            "state": "completed"
            if engine.position == engine.total and stop_reason == "budget_completed"
            else "stopped",
            "stop_reason": stop_reason,
            "supervisor_step": step,
            "variant": variant,
            "selected_checkpoint": None,
            "private_body_bytes": 8192,
            "precision_bits_trained": 8,
            "batch_size": 16,
            "accumulation_steps": 1,
            "initial_model_hash": engine.initial_model_hash,
            "frozen_teacher_unchanged": True,
            "independent_medical_evaluation": False,
            "privacy_training": False,
            "direct_pixel_p2_p3_evaluated": False,
            "authorization_and_writer_implemented": False,
            "scientific_acceptance": "not_evaluated",
        }
        atomic_json(output / "training_summary.json", summary)
        return payload + ["training_summary.json", "cost.json"], summary["state"]
    finally:
        # Includes loading, diagnostics, I/O and failed attempts in the resumed lineage.
        account_wall()
        atomic_json(
            output / "cost.json", cost_record(engine, used_bytes, sampled_attempts)
        )


def report_run(path):
    path = Path(path)
    binding = read(path / "binding.json")
    if binding["config"] != CONFIG:
        raise ValueError("Not an R5 direct16 run")
    reports = []
    for p in sorted(path.glob("*-diagnostic.json")):
        d = read(p)
        reports.append(
            {
                "file": p.name,
                "position": d["position"],
                "medical": d["medical"],
                **{k: v for k, v in d.items() if k.startswith("mean_")},
            }
        )
    print(
        __import__("json").dumps(
            dict(
                run=str(path),
                status=read(path / "status.json"),
                diagnostics=reports,
                winner=None,
                privacy_evaluated=False,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("register", "verify", "plan", "train", "resume", "report")
    )
    parser.add_argument(
        "--supervisor-step", type=int, choices=(8500, 7000), default=8500
    )
    parser.add_argument(
        "--variant", choices=("learned", "fixed_gray"), default="learned"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verification")
    parser.add_argument("--wall-hours", type=float)
    parser.add_argument("--checkpoint-gib", type=float)
    parser.add_argument(
        "--steps-only",
        action="store_true",
        help="No wall-time or checkpoint-storage limit; stop at configured step counts",
    )
    parser.add_argument("--budget")
    parser.add_argument("--checkpoint")
    parser.add_argument("--run")
    args = parser.parse_args(argv)
    if args.action == "register":
        return old.register(PLAN, CONFIG)
    if args.action == "report":
        if not args.run:
            parser.error("report requires --run")
        return report_run(args.run)
    reg = registration()
    if args.action == "plan":
        if not args.verification or (
            not args.steps_only
            and (args.wall_hours is None or args.checkpoint_gib is None)
        ):
            parser.error(
                "plan requires verification and --steps-only or explicit wall/storage ceilings"
            )
        return plan(args, reg)
    parents = base_parents(reg)
    budget, resume_state, previous_costs = None, None, None
    options = dict(
        supervisor_step=args.supervisor_step, variant=args.variant, device=args.device
    )
    if args.action in ("train", "resume"):
        if not args.budget:
            parser.error("Training requires an explicitly registered --budget")
        budget = read(args.budget)
        if (
            budget["source_commit"] != reg["source_commit"]
            or budget["config"] != CONFIG
            or budget["parents"] != parents
            or budget["supervisor_step"] != args.supervisor_step
            or budget["variant"] != args.variant
        ):
            raise ValueError("Budget does not match this experiment")
        if (
            not valid_resource_policy(budget)
            or budget["max_optimizer_attempts"]
            != CONFIG["baseline_updates"]
            + CONFIG["c1_cycles"] * (2 if args.variant == "learned" else 1)
            or budget["max_sampled_positions"]
            != (CONFIG["baseline_updates"] + CONFIG["c1_cycles"]) * 16
        ):
            raise ValueError("Invalid registered resource ceilings")
        checked_verification(budget["verification"], reg, args.supervisor_step)
        if (
            digest(Path(budget["verification"]) / "artifact.json")
            != budget["verification_sha256"]
        ):
            raise ValueError("Verification artifact changed")
        parents[str(Path(args.budget).resolve())] = digest(args.budget)
        parents[str(Path(budget["verification"]) / "artifact.json")] = budget[
            "verification_sha256"
        ]
        options["budget_sha256"] = argsafe_budget_hash(budget)
        if args.action == "resume":
            if not args.checkpoint:
                parser.error("resume requires --checkpoint")
            resume_state, previous_costs = read_checkpoint(args.checkpoint, budget, reg)
            parents[str(Path(args.checkpoint).resolve())] = digest(args.checkpoint)
            parents[str(Path(args.checkpoint).with_suffix(".json").resolve())] = digest(
                Path(args.checkpoint).with_suffix(".json")
            )
        elif args.checkpoint:
            parser.error(
                "Use resume explicitly, never restart from a supplied checkpoint"
            )
    output = (
        ROOT
        / "runs/grayscale-r5"
        / f"{args.action}-{args.variant}-s{args.supervisor_step}.{datetime.now():%Y%m%d-%H%M%S-%f}.{os.getpid()}"
    )
    out = new_run(
        reg["resources"], output, "r5-" + args.action, CONFIG, parents, options
    )
    print("RUN:", out, flush=True)
    try:
        with run_lock(out):
            if args.action == "verify":
                files = verify(
                    out,
                    reg["resources"],
                    args.supervisor_step,
                    torch.device(args.device),
                )
                state = "completed"
            else:
                files, state = run_training(
                    out,
                    reg["resources"],
                    args.supervisor_step,
                    torch.device(args.device),
                    args.variant,
                    budget,
                    resume_state,
                    previous_costs,
                )
            check_hashes(parents)
            log(out, "parents_unchanged")
            finish(
                out,
                "r5_pixel_verification"
                if args.action == "verify"
                else "r5_grayscale_training",
                files,
            )
            if state != "completed":
                status(out, state, artifact="artifact.json")
    except BaseException as exc:
        status(out, "failed", error=str(exc))
        raise
    print("R5_FINISHED:", out, "state:", state, flush=True)
    return out


if __name__ == "__main__":
    main()
