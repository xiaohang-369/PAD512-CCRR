"""Fresh GPU R3: unchanged science, interval-100 recovery, no old-run migration."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from ccrr.config import load_config, sha256
from ccrr.errors import require
from ccrr.runtime.support import environment, write_json
from ccrr.training.experiment import Experiment
from ccrr.training.runner import restore_dependencies
from ccrr.training.b_resources import verify_teachers, execution_options
from ccrr.training.b_batches import TeacherBank
from ccrr.training.b_control import ReconstructionControl, control_update
from ccrr.training.b_functional import check_functional
from ccrr.training.b_diagnostics import isolated_rng
from ccrr.training.optimization import optimizer, TeacherSchedule
from ccrr.data.pad import rng_for
from b_r3_state import Budget, Paused, INIT
from b_r3_gpu_state import State, PROTOCOL, Timings
from b_r3_gpu_audit import Audit
from b_r3_diagnostics import (
    initialize,
    train_step,
    observation,
    full_scores,
    mask_diagnostic,
)

STEPS = 3000
POINTS = (0, 100, 500, 1000, 2000, 3000)
R2_SHA = "388e148167504254a3ffe13da17e484f6cdd6add52374b409663f8fa7e4e666a"


def tensor_digest(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(
            v.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    return h.hexdigest()


def validate_inputs(config, report, out, resume):
    source = Path(__file__).resolve().parents[1]
    cfg = load_config(config)
    require(
        cfg["seed"] == 20261008 and cfg["profile"]["size"] == 512, "R3 fixed seed/size"
    )
    execution_options(cfg)
    require(cfg["microbatch"] == 16, "Use admitted microbatch 16")
    require(sha256(report) == R2_SHA, "R2 report digest mismatch")
    prior = json.loads(report.read_text())
    require(
        prior["protocol"] == "B-R2-FULLFIT-v1"
        and prior["existing_checkpoints_unchanged"],
        "Wrong R2 evidence",
    )
    require(prior["config_sha256"] == sha256(config), "R2 config binding mismatch")
    sources = {
        str(p.relative_to(source)): sha256(p)
        for p in sorted((source / "src/ccrr").rglob("*.py"))
    }
    require(sources == prior["source_sha256"], "Production source changed since R2")
    aroot, _ = verify_teachers(cfg)
    roots = (
        source,
        aroot,
        Path(cfg["data_root"]).resolve(),
        Path(cfg["output"]).resolve(),
        report.parent.resolve(),
    )
    require(
        all(out != r and r not in out.parents and out not in r.parents for r in roots),
        "Output overlaps protected inputs",
    )
    require(
        out.is_dir() if resume else not out.exists(),
        "Use new output, or --resume existing R3",
    )
    paths = {
        "A_initial": aroot / "checkpoints/mfm_probe.pt",
        "B_1000": Path(cfg["output"]) / "checkpoints/purifier-1000.pt",
        "B_10000": Path(cfg["output"]) / "checkpoints/purifier-10000.pt",
        "control_selected": Path(cfg["output"]) / "no_branch_control/selected.pt",
    }
    require(
        {k: sha256(p) for k, p in paths.items()} == prior["checkpoint_digests"],
        "Prior weights changed",
    )
    protected = {str(p.resolve()): sha256(p) for p in paths.values()}
    for p in [
        config,
        report,
        aroot / "teacher_freeze_record.json",
        aroot / "acceptance.json",
    ] + [
        aroot / "checkpoints" / (n + ".pt")
        for n in ("clinical", "clinical_rgb", "clinical_gray", "mfm_probe", "reid")
    ]:
        protected[str(p.resolve())] = sha256(p)
    for entry in cfg["weights"].values():
        if isinstance(entry, dict) and entry.get("path"):
            p = Path(entry["path"])
            require(sha256(p) == entry["sha256"], "Base teacher weight digest mismatch")
            protected[str(p.resolve())] = entry["sha256"]
    data_root = Path(cfg["data_root"])
    for p in sorted((data_root / "protocol_audit_20261009").rglob("*.json")):
        protected[str(p.resolve())] = sha256(p)
    p = data_root / "raw/metadata.csv"
    protected[str(p.resolve())] = sha256(p)
    scripts = list((source / "scripts").glob("b_r3_*.py")) + [
        Path(__file__),
        source / "scripts/probe_stage_b_reconstruction.py",
        source / "scripts/probe_b_r3_gpu_speed.py",
        source / "scripts/run_stage_b_short_multi_gpu.sh",
        source / "tests/test_b_r3_gpu.py",
    ]
    for rel, digest in sources.items():
        protected[str((source / rel).resolve())] = digest
    for path in scripts:
        protected[str(path.resolve())] = sha256(path)
    return cfg, {
        "protocol": PROTOCOL,
        "init": INIT,
        "config_sha256": sha256(config),
        "source": sources,
        "scripts": {p.name: sha256(p) for p in scripts},
        "protected": protected,
    }


def verify_unchanged(e, bound, frozen):
    require(
        all(sha256(Path(p)) == h for p, h in bound["protected"].items()),
        "Protected evidence changed",
    )
    for k, h in frozen.items():
        model = getattr(e, k)
        require(
            not model.training and not any(p.requires_grad for p in model.parameters()),
            "Teacher unfrozen",
        )
        require(tensor_digest(model) == h, "Teacher tensor changed")
    verify_teachers(e.cfg)


def make_state(e, bound, resume):
    if resume and (e.output / "progress.pt").exists():
        # Resume only restores; NEVER apply the scale a second time.
        control = ReconstructionControl(e.purifier.mean, e.purifier.std).to(e.device)
    else:
        control = initialize(e)
    e.no_branch_control = control
    opts = {
        "purifier": optimizer([e.purifier], [1e-4]),
        "attacker": optimizer([e.sem_attacker], [1e-4]),
        "control": optimizer([control], [1e-4]),
    }
    schedules = {
        k: TeacherSchedule(o, 20000 if k == "attacker" else 10000)
        for k, o in opts.items()
    }
    state = State(
        e.output / "progress.pt",
        bound,
        {"purifier": e.purifier, "attacker": e.sem_attacker, "control": control},
        opts,
        schedules.items(),
        e.sampler,
    )
    if not state.loaded:
        state.data = {
            "next": 0,
            "phase": "purifier",
            "observations": {},
            "control_observations": {},
            "losses": [],
            "control_losses": [],
        }
        state.save()
    require(
        state.data["phase"] in ("purifier", "control", "audit", "complete"),
        "Bad R3 phase",
    )
    require(
        schedules["attacker"].next_step == 2 * schedules["purifier"].next_step,
        "Alternating counters mismatch",
    )
    require(
        all(
            s.next_step <= (6000 if k == "attacker" else STEPS)
            for k, s in schedules.items()
        ),
        "R3 budget exceeded",
    )
    return state


def training(e, banks, state, budget, frozen):
    sched = state.schedules
    opt = state.opts
    d = state.data
    while d["phase"] == "purifier":
        budget.check()
        n = sched["purifier"].next_step
        if n in POINTS and str(n) not in d["observations"]:
            print(f"R3_OBSERVATION: step={n}", flush=True)
            d["observations"][str(n)] = observation(e, banks, n)
            verify_unchanged(e, state.bound, frozen)
            state.save()
            write_json(e.output / f"trajectory-{n}.json", d["observations"][str(n)])
        if n in POINTS and not (e.output / "snapshots" / f"purifier-{n}.pt").exists():
            state.save(e.output / "snapshots" / f"purifier-{n}.pt")
        if n == STEPS:
            d["phase"] = "control"
            state.save()
            break
        metrics = Timings.call(
            "update.purifier",
            train_step,
            e,
            banks["fit"],
            n,
            opt["purifier"],
            opt["attacker"],
            sched["purifier"],
            sched["attacker"],
        )
        d["losses"].append(metrics)
        d["next"] = n + 1
        state.save()
        budget.record()
        if (n + 1) % 100 == 0:
            Timings.write()
            print(
                f"R3_PURIFIER: {n + 1}/{STEPS} loss={metrics['loss']:.6f}", flush=True
            )
    while d["phase"] == "control":
        budget.check()
        n = sched["control"].next_step
        if n in POINTS and str(n) not in d["control_observations"]:
            with isolated_rng():
                d["control_observations"][str(n)] = {
                    r: full_scores(e, e.no_branch_control, b, len(e.data[r]), True)
                    for r, b in banks.items()
                }
            state.save()
        if n in POINTS and not (e.output / "snapshots" / f"control-{n}.pt").exists():
            state.save(e.output / "snapshots" / f"control-{n}.pt")
        if n == STEPS:
            d["phase"] = "audit"
            state.save()
            break
        ids = rng_for(e.seed, "B", "images", n).integers(len(e.fit), size=16).tolist()
        loss = Timings.call(
            "update.control",
            control_update,
            e.no_branch_control,
            banks["fit"],
            ids,
            e.cfg["microbatch"],
            opt["control"],
            sched["control"],
        )
        require(np.isfinite(loss), "Nonfinite control loss")
        d["control_losses"].append(float(loss))
        state.save()
        budget.record()
        if (n + 1) % 100 == 0:
            Timings.write()
            print(f"R3_CONTROL: {n + 1}/{STEPS}", flush=True)


def audit_update_counts(a):
    counts = {"clinical_readouts": 0, "independent_attackers": 0, "post_adaptation": 0}
    jobs = [
        (f"clinical-{r}-{f}", 2000, "clinical_readouts")
        for r in ("raw", "control", "sem")
        for f in ("linear", "mlp")
    ]
    jobs += [
        (f"identity-{s}-{r}-{f}", 5000, "independent_attackers")
        for s in ("p1", "p2")
        for r in ("raw", "control", "sem")
        for f in ("spatial", "pooled")
    ]
    jobs += [("post-adapt", 1000, "post_adaptation")]
    for name, total, group in jobs:
        raw = torch.load(
            a.heads / (name + ".pt"), map_location="cpu", weights_only=True
        )
        t = raw["training_state"]
        require(
            raw["stage"] == PROTOCOL and t["binding"] == {**a.bound, "head": name},
            "Final head binding mismatch",
        )
        require(t["data"]["next"] == total, "Incomplete evaluation budget")
        require(
            "after" in t["data"] if name == "post-adapt" else t["data"]["complete"],
            "Incomplete evaluation selection",
        )
        counts[group] += total
    return counts


def audit_stage(e, banks, state, budget, frozen):
    e.purifier.eval().requires_grad_(False)
    e.no_branch_control.eval().requires_grad_(False)
    e.sem_attacker.eval().requires_grad_(False)
    e.feature_audit_tag = "fixed-3000"
    a = Audit(e, state.bound, budget)

    # Restore cached read-only evaluation artifacts without repeating head updates.
    def task(name, fn):
        path = a.root / (name + ".json")
        if path.exists():
            saved = json.loads(path.read_text())
            require(saved["binding"] == state.bound, "Audit artifact binding mismatch")
            return saved["result"]
        budget.check()
        result = Timings.call("audit." + name, fn)
        write_json(path, {"binding": state.bound, "result": result})
        budget.record()
        return result

    Timings.call("audit.cache", a.cache)
    functional = task("functional", lambda: check_functional(e, banks["fit"]))
    adapted = task("adapted", a.adapted)
    clinical = task("classifiers", a.classifiers)
    identity = task("attackers", a.attackers)
    for row in identity["checks"]:
        row["sem_minus_control_a_dagger"] = (
            row["sem_strongest"] - row["control_diagnostic"]["strongest_a_dagger"]
        )
    mask = task("mask", lambda: mask_diagnostic(e, banks["val_select"], a))
    occupancy = np.asarray(a.occupancy)
    fraction = ((occupancy >= 0.1) & (occupancy <= 0.5)).mean(0)
    numerical = bool(
        clinical["passed"] and identity["passed"] and (fraction >= 0.95).all()
    )
    conflicts = []
    for row in identity["checks"]:
        d = row["domain"]
        scenario = row["scenario"]
        ad = next(x for x in adapted["checks"][scenario] if x["domain"] == d)
        # A stronger adapted score is explicitly compared to the raw-relative point target.
        if ad["adapted_a_dagger"] - 0.5 > 0.75 * (row["raw_strongest"] - 0.5):
            conflicts.append(
                {
                    "scenario": scenario,
                    "domain": d,
                    "reason": "adapted attack exceeds raw-relative privacy point target",
                }
            )
    counts = audit_update_counts(a)
    verify_unchanged(e, state.bound, frozen)
    budget.check()
    trajectory_summary = []
    for step, row in state.data["observations"].items():
        roles = {}
        for role, values in row["roles"].items():
            roles[role] = {}
            for domain, probe in values["probe_diagnostics"].items():
                roles[role][domain] = {
                    k: probe[k]
                    for k in (
                        "saturation_fraction",
                        "tanh_derivative_mean",
                        "channel_entropy_bits_min_mean_max",
                        "between_image_normalized_code_variance_same_site_mean",
                    )
                }
                roles[role][domain]["all_image_mse"] = float(
                    np.mean([x["mse"] for x in values["all_images"][domain]])
                )
        trajectory_summary.append({"step": int(step), "roles": roles})
    summary = {
        "protocol": PROTOCOL,
        "initialization": INIT,
        "fixed_endpoint": STEPS,
        "execution_variant": "gpu-resident-checkpoint100",
        "maximum_hard_crash_replay_transactions": 100,
        "trajectory_summary": trajectory_summary,
        "evaluation_updates_verified_from_checkpoints": counts,
        "updates": {
            "purifier": 3000,
            "alternating_attacker": 6000,
            "control": 3000,
            "post_adaptation": 1000,
            "clinical_readouts": 12000,
            "independent_attackers": 60000,
            "total": 85000,
        },
        "clinical": clinical,
        "identity": identity,
        "occupancy_in_range": fraction.tolist(),
        "post_adaptation": adapted,
        "adaptation_conflicts": conflicts,
        "functional": functional,
        "mask_diagnostic_path": str(a.root / "mask.json"),
        "mask_rows": len(mask["rows"]),
        "listed_development_thresholds_met": numerical,
        "needs_attack_review": bool(
            conflicts or adapted["alternating_attack_insufficient"]
        ),
        "formal_training_started": False,
        "scientific_admission_complete": False,
        "stage_b_passed": False,
        "frozen_teachers_unchanged": True,
        "existing_checkpoints_unchanged": True,
        "limitations": [
            "Single seed short diagnostic; no formal B or INN admission",
            "Feature privacy is not public-image privacy",
            "Mask permutation is a frozen-model perturbation, not a retrained ablation",
        ],
    }
    write_json(
        e.output / "report.json",
        {"binding": state.bound, "training": state.data, "summary": summary},
    )
    summary["report_sha256"] = sha256(e.output / "report.json")
    write_json(e.output / "summary.json", summary)
    write_json(
        e.output / "status.json", {"status": "complete", "stage_b_passed": False}
    )
    state.data["phase"] = "complete"
    state.save()
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--r2-report", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    State.reset_registry()
    out = args.out.resolve()
    print(
        "R3_VERIFY_INPUTS: frozen A, R2 evidence, original code and checkpoints",
        flush=True,
    )
    cfg, bound = validate_inputs(
        args.config.resolve(), args.r2_report.resolve(), out, args.resume
    )
    bound["execution_variant"] = {
        "checkpoint_interval": 100,
        "audit_cache_device": "cuda",
        "scoring": "fixed-pairs-exact-batch-cache",
    }
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(
        gpu == "GPU-76f0daac-5625-a97a-38ec-efdb3ed44d54", "Use the registered GPU UUID"
    )
    busy = subprocess.check_output(
        ["nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    ).strip()
    require(not busy, "GPU occupied; stop")
    require(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
        "Set deterministic CUBLAS",
    )
    require(
        torch.cuda.is_available()
        and torch.cuda.device_count() == 1
        and "H100" in torch.cuda.get_device_name(0),
        "One H100 required",
    )
    bound["environment"] = environment()
    bound["execution"] = {
        "gpu": gpu,
        "threads": torch.get_num_threads(),
        "interop": torch.get_num_interop_threads(),
        "cublas": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }
    if args.resume:
        require(
            json.loads((out / "binding.json").read_text()) == bound,
            "Resume source/config/environment/evidence mismatch",
        )
    else:
        out.mkdir(parents=True, mode=0o700)
        write_json(out / "binding.json", bound)
    with (out / ".r3.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = Budget(out)
        budget.install()
        (out / "worker.pid").write_text(str(os.getpid()) + "\n")
        print(f"R3_WORKER_PID={os.getpid()}", flush=True)
        e = None
        Timings.start(out / f"performance-{time.time_ns()}.json")
        print(f"R3_PERFORMANCE_LOG={Timings.path}", flush=True)
        try:
            budget.check()
            cfg.update(
                output=str(out),
                resume_a=False,
                resume_b=args.resume,
                b_diagnostic=True,
                b_cache_device="cuda",
            )
            print("R3_RESTORE_FROZEN_A", flush=True)
            e = Experiment(cfg)
            restore_dependencies(e, "B")
            del e.system, e.pub_attacker  # not used or trained by this diagnostic
            require(e.purifier.stats_ready, "Frozen A statistics absent")
            # A checkpoint includes the pre-B Es/Ds/mask/identity parameters; do not reset them.
            frozen = {
                k: tensor_digest(getattr(e, k)) for k in ("mfm", "clinical", "reid")
            }
            fit = set(r["patient_id"] for r in e.fit.rows)
            val = set(r["patient_id"] for r in e.data["val_select"].rows)
            require(
                not fit & val
                and len(e.fit) == 1433
                and len(e.data["val_select"]) == 170
                and len(fit) == 863
                and len(val) == 103,
                "Cohort mismatch",
            )
            budget.check()
            print("R3_BUILD_TEACHER_CACHES", flush=True)
            banks = {r: TeacherBank(e, r) for r in ("fit", "val_select")}
            e.b_teacher_bank = banks["fit"]
            e.b_teacher_banks = banks
            # Rebuild all immutable caches BEFORE restoring checkpoint RNGs.
            state = make_state(e, bound, args.resume)
            training(e, banks, state, budget, frozen)
            if state.data["phase"] != "complete":
                audit_stage(e, banks, state, budget, frozen)
            verify_unchanged(e, bound, frozen)
            print("R3_SUMMARY:", out / "summary.json", flush=True)
            print("FORMAL_TRAINING_STARTED: false", flush=True)
        except Paused as exc:
            State.flush_pending()
            write_json(
                out / "status.json",
                {
                    "status": "paused",
                    "reason": str(exc),
                    "resume_available": True,
                    "formal_training_started": False,
                },
            )
            print("R3_PAUSED:", str(exc), flush=True)
            return 75
        except Exception as exc:
            write_json(
                out / "status.json",
                {
                    "status": "error",
                    "type": type(exc).__name__,
                    "reason": str(exc),
                    "formal_training_started": False,
                    "stage_b_passed": False,
                },
            )
            raise
        finally:
            if e is not None:
                e.allocator.close()
            budget.close()
            Timings.write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
