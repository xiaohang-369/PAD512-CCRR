"""Isolated real-data H100 measurements. Never dispatches or resumes Stage B.

The selected engine is imported from its archived source. The profiler lives
outside that archive; no production module or existing run is edited.
"""

import argparse
from contextlib import contextmanager, ExitStack
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import torch

from b_r3_perf_common import cpu_copy, digest, exact, trace_summary, compare_reports

GPU = "GPU-76f0daac-5625-a97a-38ec-efdb3ed44d54"
OLD_REPORT_SHA = "cb64142fc464479919f8b26dc1e9e4d9150050f60946c2a9bcb1d502ea5b142a"


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def gpu_pids():
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            GPU,
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=15,
    ).strip()
    return {int(v.strip()) for v in raw.splitlines() if v.strip()}


class Guard:
    """Sample this GPU only; refuse measurements if another process appears."""

    def __init__(self, out, seconds):
        self.out, self.deadline = out, time.monotonic() + seconds
        self.done = threading.Event()
        self.error = None
        self.phase = "setup"
        self.rows = []
        self.thread = threading.Thread(target=self.monitor, daemon=True)

    def monitor(self):
        while not self.done.is_set():
            try:
                foreign = gpu_pids() - {os.getpid()}
                if foreign:
                    self.error = f"Other GPU processes appeared: {sorted(foreign)}; do not disturb them."
                row = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        GPU,
                        "--query-gpu=timestamp,uuid,utilization.gpu,memory.used,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=15,
                ).strip()
                self.rows.append(
                    {
                        "monotonic": time.monotonic(),
                        "phase": self.phase,
                        "nvidia_smi": row,
                    }
                )
            except Exception as exc:
                self.error = f"GPU monitoring failed: {exc}"
            self.done.wait(0.5)

    def check(self):
        if self.error:
            raise RuntimeError(self.error)
        if time.monotonic() >= self.deadline:
            raise TimeoutError(
                "Isolated benchmark time limit reached; no full training started"
            )

    def close(self):
        self.done.set()
        self.thread.join(timeout=35)
        write(self.out / "gpu-samples.json", self.rows)


def timed(fn):
    torch.cuda.synchronize()
    begin, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start = time.perf_counter()
    begin.record()
    value = fn()
    end.record()
    end.synchronize()
    return value, {
        "wall_seconds": time.perf_counter() - start,
        "cuda_event_elapsed_seconds": begin.elapsed_time(end) / 1000,
    }


@contextmanager
def sampling_trace(e, banks, audit):
    events = []

    def wrap(fn, name, result=False):
        def call(*args, **kwargs):
            value = fn(*args, **kwargs)
            # Only indices/domains/pair identities, never feature bytes or patient images.
            events.append((name, value if result else (args, kwargs)))
            return value

        return call

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(e.sampler, "sample", wrap(e.sampler.sample, "pairs", True))
        )
        for role, bank in banks.items():
            stack.enter_context(
                patch.object(bank, "get", wrap(bank.get, "teacher_indices." + role))
            )
        if audit is not None:
            stack.enter_context(
                patch.object(
                    audit,
                    "feature_batch",
                    wrap(audit.feature_batch, "audit_indices_domains"),
                )
            )
            if hasattr(audit, "pooled_batch"):
                stack.enter_context(
                    patch.object(
                        audit,
                        "pooled_batch",
                        wrap(audit.pooled_batch, "audit_indices_domains"),
                    )
                )
        yield events


@contextmanager
def instrumentation(work):
    """Labels and events only in separate profiler pass, never throughput pass."""
    events = []

    def wrap(fn, label, cuda=True):
        def call(*args, **kwargs):
            with torch.profiler.record_function(label):
                begin = torch.cuda.Event(enable_timing=True) if cuda else None
                end = torch.cuda.Event(enable_timing=True) if cuda else None
                if begin:
                    begin.record()
                start = time.perf_counter()
                try:
                    return fn(*args, **kwargs)
                finally:
                    if end:
                        end.record()
                    shapes = [
                        list(a.shape) for a in args if isinstance(a, torch.Tensor)
                    ]
                    events.append(
                        (label, time.perf_counter() - start, begin, end, shapes)
                    )

        return call

    with ExitStack() as stack:
        for name, model in work.modules.items():
            stack.enter_context(
                patch.object(
                    model, "forward", wrap(model.forward, "phase.forward." + name)
                )
            )
        stack.enter_context(
            patch.object(
                torch.Tensor, "backward", wrap(torch.Tensor.backward, "phase.backward")
            )
        )
        stack.enter_context(
            patch.object(
                torch.nn.utils,
                "clip_grad_norm_",
                wrap(torch.nn.utils.clip_grad_norm_, "phase.gradient_validation_clip"),
            )
        )
        stack.enter_context(
            patch.object(work, "score", wrap(work.score, "phase.nine_direction_score"))
        )
        stack.enter_context(
            patch.object(work, "reset", wrap(work.reset, "phase.full_restore"))
        )
        for name, opt in work.opts.items():
            stack.enter_context(
                patch.object(opt, "step", wrap(opt.step, "phase.optimizer." + name))
            )
        stack.enter_context(
            patch.object(
                work.e.sampler,
                "sample",
                wrap(work.e.sampler.sample, "phase.sampling", False),
            )
        )
        for name, bank in work.banks.items():
            stack.enter_context(
                patch.object(
                    bank,
                    "get",
                    wrap(bank.get, "phase.feature_gather_or_transfer." + name),
                )
            )
        if work.audit is not None:
            stack.enter_context(
                patch.object(
                    work.audit,
                    "feature_batch",
                    wrap(work.audit.feature_batch, "phase.audit_feature_gather"),
                )
            )
        stack.enter_context(
            patch.object(torch, "save", wrap(torch.save, "phase.checkpoint_serialize"))
        )
        stack.enter_context(
            patch.object(torch, "load", wrap(torch.load, "phase.checkpoint_load"))
        )
        yield events


class Workload:
    def __init__(self, e, banks, audit, engine, root, kind, start_snapshot):
        self.e, self.banks, self.audit, self.engine = e, banks, audit, engine
        self.root, self.kind = root, kind
        root.mkdir()
        from b_r3_state import Budget

        self.budget = Budget(root)
        e.output = root
        State = engine["state"].State
        State.reset_registry()
        self.bound = {"performance_only": True, "case": kind}
        self.seed_file = root / "initial.pt"
        self.schedules = {}
        if kind in ("purifier", "control"):
            from ccrr.runtime.checkpoints import load_checkpoint

            # The preceding Purifier transaction freezes this alternating head
            # during the generator update. requires_grad is not in state_dict;
            # restore its optimizer membership before constructing the next case.
            # Frozen Stage A/PanDerm teachers are separate and remain untouched.
            e.sem_attacker.requires_grad_(True)
            self.state = engine["runner"].make_state(e, self.bound, False)
            raw = load_checkpoint(
                start_snapshot, self.state.modules, self.state.opts["purifier"]
            )
            for k, opt in self.state.opts.items():
                opt.load_state_dict(raw["training_state"]["optimizers"][k])
            for k, schedule in self.state.schedules.items():
                schedule.load_state_dict(raw["training_state"]["schedulers"][k])
            self.state.data = copy.deepcopy(raw["training_state"]["data"])
            e.sampler.fallbacks = raw["training_state"]["sampler_fallbacks"]
            self.modules, self.opts, self.schedules = (
                self.state.modules,
                self.state.opts,
                self.state.schedules,
            )
        else:
            from ccrr.data.pad import rng_for
            from ccrr.runtime.support import seed_everything
            from ccrr.training.optimization import optimizer
            from ccrr.representation.purifier import SemanticAttacker
            from ccrr.evaluation.attacks import PooledAttacker

            _, scenario, family = kind.split("-")
            name = f"identity-{scenario}-sem-{family}"
            seed_everything(int(rng_for(e.seed, "B-readout", name, 0).integers(2**31)))
            head = (SemanticAttacker() if family == "spatial" else PooledAttacker()).to(
                e.device
            )
            self.modules, self.opts = (
                {"head": head},
                {"head": optimizer([head], [1e-4])},
            )
            e.sampler.fallbacks = 0
            self.state = State(
                root / "progress.pt",
                self.bound,
                self.modules,
                self.opts,
                sampler=e.sampler,
            )
            self.state.data = {
                "next": 0,
                "evaluated": 0,
                "best_score": -1.0,
                "best": None,
                "best_weights": None,
                "complete": False,
            }
        self.state.save(self.seed_file)
        self.reset()

    def reset(self, checkpoint=None):
        State = self.engine["state"].State
        State.reset_registry()
        self.state = State(
            checkpoint or self.seed_file,
            self.bound,
            self.modules,
            self.opts,
            self.schedules.items(),
            self.e.sampler,
        )
        self.state.path = self.root / "progress.pt"
        self.schedules = self.state.schedules
        for model in self.modules.values():
            for p in model.parameters():
                p.grad = None

    def update(self):
        from ccrr.data.pad import rng_for

        if self.kind == "purifier":
            n = self.schedules["purifier"].next_step
            metrics = self.engine["runner"].train_step(
                self.e,
                self.banks["fit"],
                n,
                self.opts["purifier"],
                self.opts["attacker"],
                self.schedules["purifier"],
                self.schedules["attacker"],
            )
            self.state.data["losses"].append(metrics)
            self.state.data["next"] = n + 1
        elif self.kind == "control":
            n = self.schedules["control"].next_step
            ids = (
                rng_for(self.e.seed, "B", "images", n)
                .integers(len(self.e.fit), size=16)
                .tolist()
            )
            value = self.engine["runner"].control_update(
                self.e.no_branch_control,
                self.banks["fit"],
                ids,
                16,
                self.opts["control"],
                self.schedules["control"],
            )
            self.state.data["control_losses"].append(float(value))
        else:
            _, scenario, family = self.kind.split("-")
            n = self.state.data["next"]
            self.audit.attacker_update(
                self.modules["head"], self.opts["head"], "sem", scenario, family, n
            )
            self.state.data["next"] = n + 1

    def score(self):
        if self.kind in ("purifier", "control"):
            return None
        _, scenario, _ = self.kind.split("-")
        return self.audit.attack_scores(self.modules["head"], "sem", scenario)

    def remember_scores(self, result):
        if result is None:
            return
        self.state.data["evaluated"] = self.state.data["next"]
        self.state.data["best"] = result
        self.state.data["best_weights"] = cpu_copy(self.modules["head"].state_dict())
        # This is a timing checkpoint, not a scientific best-head selection.

    def capture(self, gradients=True):
        import random
        import numpy as np

        result = {
            "models": {k: m.state_dict() for k, m in self.modules.items()},
            "optimizers": {k: o.state_dict() for k, o in self.opts.items()},
            "schedulers": {k: s.state_dict() for k, s in self.schedules.items()},
            "progress": copy.deepcopy(self.state.data),
            "sampler_fallbacks": self.e.sampler.fallbacks,
            "rng": {
                "python": random.getstate(),
                "numpy": (
                    np.random.get_state()[0],
                    np.random.get_state()[1].tolist(),
                    *np.random.get_state()[2:],
                ),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
            },
        }
        if gradients:
            result["gradients_after_clip"] = {
                k: {n: p.grad for n, p in m.named_parameters()}
                for k, m in self.modules.items()
            }
        return cpu_copy(result)


def profile(work, args, guard):
    work.reset()
    for _ in range(args.warmup):
        guard.check()
        work.update()
    torch.cuda.synchronize()
    trace = work.root / "trace.json"
    with (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof,
        instrumentation(work) as events,
    ):
        with torch.profiler.record_function("R3_PROFILE_WINDOW"):
            for _ in range(args.profile_steps):
                guard.check()
                work.budget.check()
                work.update()
                work.state.save()
                work.budget.record()
            work.remember_scores(work.score())
            work.state.save(work.root / "profile-checkpoint.pt")
            work.reset(work.root / "profile-checkpoint.pt")
            torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace))
    (work.root / "profiler-table.txt").write_text(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total", row_limit=50
        )
    )
    phases = {}
    for label, host, begin, end, shapes in events:
        row = phases.setdefault(
            label,
            {
                "calls": 0,
                "host_seconds": 0.0,
                "cuda_event_elapsed_seconds": 0.0,
                "input_shape_counts": {},
            },
        )
        row["calls"] += 1
        row["host_seconds"] += host
        if shapes:
            key = json.dumps(shapes)
            row["input_shape_counts"][key] = row["input_shape_counts"].get(key, 0) + 1
        if begin:
            row["cuda_event_elapsed_seconds"] += begin.elapsed_time(end) / 1000
    result = {
        "trace": trace.name,
        "phases": phases,
        "activity": trace_summary(trace),
        "note": "Profile pass includes intrusive instrumentation. Nested phases overlap; CUDA Event elapsed time includes launch gaps, not just kernels.",
    }
    write(work.root / "profile.json", result)
    return result


def benchmark(work, args, guard):
    work.reset()
    initial_hash = digest(work.capture())
    hashes = []
    # Separate equivalence pass: state copies/hashes are OUTSIDE throughput timing.
    preclip = []
    original_clip = torch.nn.utils.clip_grad_norm_

    def capture_clip(parameters, *args, **kwargs):
        parameters = list(parameters)
        preclip.append(digest([p.grad for p in parameters]))
        return original_clip(parameters, *args, **kwargs)

    with (
        sampling_trace(work.e, work.banks, work.audit) as samples,
        patch.object(torch.nn.utils, "clip_grad_norm_", capture_clip),
    ):
        for _ in range(args.equivalence_steps):
            guard.check()
            preclip.clear()
            work.update()
            hashes.append(
                {
                    "unclipped_gradient_hashes": list(preclip),
                    "end_state_sha256": digest(work.capture()),
                }
            )
        sampling_hash = digest(samples)
    repeats = []
    reference = None
    for rep in range(args.repeats):
        guard.phase = work.kind + f"/throughput/{rep}"
        work.reset()
        for _ in range(args.warmup):
            guard.check()
            work.update()
            work.state.save()
        # Warm immutable pair-index tables and scoring kernels separately.
        # Scoring never updates parameters or changes the sampled update sequence.
        work.score()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        def loop():
            for _ in range(args.steps):
                guard.check()
                work.budget.check()
                # The reference runner's timing barriers and save-request cadence
                # remain part of the measured transaction.
                work.engine["state"].Timings.call("update." + work.kind, work.update)
                work.state.save()
                work.budget.record()

        _, updates = timed(loop)
        scores, scoring = timed(work.score)
        work.remember_scores(scores)
        final = work.capture()
        final["scores"] = scores
        if reference is None:
            reference = final
        else:
            check(
                exact(reference, final),
                "Repeated workload is not exactly deterministic",
            )
        _, saving = timed(lambda: work.state.save(work.root / "full-checkpoint.pt"))
        before_restore = work.capture(gradients=False)
        _, restoring = timed(lambda: work.reset(work.root / "full-checkpoint.pt"))
        check(
            exact(before_restore, work.capture(gradients=False)),
            "Full restore mismatch",
        )
        # Verify continuation, including RNG, sampler, optimizer and gradients.
        work.update()
        continuation = work.capture()
        work.reset(work.root / "full-checkpoint.pt")
        work.update()
        check(exact(continuation, work.capture()), "Post-restore update mismatch")
        repeats.append(
            {
                "repeat": rep,
                "updates": updates,
                "score": scoring,
                "full_checkpoint": saving,
                "full_restore": restoring,
                "restore_and_continuation_exact": True,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "steps": args.steps,
                "effective_batch": 16,
                "seconds_per_step": updates["wall_seconds"] / args.steps,
                "effective_items_per_second": args.steps * 16 / updates["wall_seconds"],
                "cycle_seconds": sum(
                    x["wall_seconds"] for x in (updates, scoring, saving, restoring)
                ),
            }
        )
        print("H100_REPEAT", work.kind, json.dumps(repeats[-1]), flush=True)
    torch.save(reference, work.root / "final-state.pt")
    # Preserve completed measurements even if trace export/parsing later fails.
    write(
        work.root / "timing-results.json",
        {
            "initial_state_sha256": initial_hash,
            "sampling_sha256": sampling_hash,
            "equivalence_step_hashes": hashes,
            "repeats": repeats,
            "profiler_pending": True,
            "valid_complete_measurement": False,
        },
    )
    guard.phase = work.kind + "/profiler"
    profiling = profile(work, args, guard)
    return {
        "initial_state_sha256": initial_hash,
        "sampling_sha256": sampling_hash,
        "equivalence_step_hashes": hashes,
        "state_file": str((work.root / "final-state.pt").relative_to(args.out)),
        "repeats": repeats,
        "median_step_seconds": statistics.median(
            r["seconds_per_step"] for r in repeats
        ),
        "median_score_seconds": statistics.median(
            r["score"]["wall_seconds"] for r in repeats
        ),
        "median_cycle_seconds": statistics.median(r["cycle_seconds"] for r in repeats),
        "profile": profiling,
        "note": "Purifier step is 2 attacker updates + 1 generator update. Head step is 16 pairs. Cycle includes update transactions, one score, one full save and one full restore. Head selection/bootstrap not benchmarked here.",
    }


def main():
    global GPU
    process_start = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gpu-uuid",
        default=GPU,
        help="Explicit authorized GPU UUID; must match CUDA_VISIBLE_DEVICES",
    )
    p.add_argument("--engine-source", type=Path, required=True)
    p.add_argument(
        "--run", type=Path, required=True, help="Completed v0.2 R3 run, read only"
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--out", type=Path, required=True, help="New private benchmark directory"
    )
    p.add_argument("--compare-to", type=Path)
    p.add_argument(
        "--suite", choices=("baseline", "scaling", "all", "ab"), default="baseline"
    )
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--profile-steps", type=int, default=3)
    p.add_argument("--equivalence-steps", type=int, default=3)
    p.add_argument("--max-seconds", type=int, default=1800)
    args = p.parse_args()
    import re

    check(
        re.fullmatch(
            r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", args.gpu_uuid
        ),
        "Expected one full GPU UUID",
    )
    GPU = args.gpu_uuid
    check(
        not args.compare_to or args.suite == "baseline",
        "Engine comparison uses baseline suite only",
    )
    check(
        1 <= args.steps <= 200 and 1 <= args.warmup <= 20 and 2 <= args.repeats <= 5,
        "Use bounded benchmark counts",
    )
    check(
        1 <= args.profile_steps <= 5 and 1 <= args.equivalence_steps <= 10,
        "Use bounded probe counts",
    )
    check(60 <= args.max_seconds <= 3600, "Benchmark time cap must be 60..3600 seconds")
    args.out = args.out.resolve()
    check(not args.out.exists(), "Output must be new; never overwrite an experiment")
    old = args.run.resolve() / "result"
    check(
        args.out != old and old not in args.out.parents and args.out not in old.parents,
        "Output overlaps old run",
    )
    check(
        os.environ.get("CUDA_VISIBLE_DEVICES") == GPU,
        "Expose only the authorized GPU UUID",
    )
    check(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
        "Deterministic CUBLAS setting required",
    )
    check(
        os.environ.get("PYTHONHASHSEED") == "20261008",
        "Fixed Python hash seed required",
    )
    check(not gpu_pids(), "Registered GPU is occupied; no work started")
    sys.path.insert(0, str(args.engine_source.resolve() / "src"))
    sys.path.insert(0, str(args.engine_source.resolve() / "scripts"))
    import b_r3_gpu_state as state
    import check_stage_b_short_multi_gpu as runner
    from b_r3_gpu_features import GPUFeatureAudit
    from ccrr.training.experiment import Experiment
    from ccrr.training.runner import restore_dependencies
    from ccrr.training.b_batches import TeacherBank
    from ccrr.config import load_config
    from ccrr.runtime.support import environment

    check(sha(old / "report.json") == OLD_REPORT_SHA, "Wrong completed R3 report")
    completed = json.loads((old / "report.json").read_text())
    check(
        json.loads((old / "status.json").read_text())["status"] == "complete",
        "Original run is not complete",
    )
    cfg = load_config(args.config)
    check(
        sha(args.config) == completed["binding"]["config_sha256"],
        "Configuration changed",
    )
    check(
        cfg["microbatch"] == 16 and cfg["seed"] == 20261008,
        "Fixed microbatch/seed required",
    )
    for protected_root in (
        Path(cfg["data_root"]),
        args.engine_source,
        Path(cfg["output"]),
        Path(cfg["stage_a_source"]["output"]),
    ):
        r = protected_root.resolve()
        check(
            args.out != r and r not in args.out.parents and args.out not in r.parents,
            "Output overlaps protected inputs",
        )
    # Validate historical protected files before any CUDA workload.
    protected = dict(completed["binding"]["protected"])
    for path in (
        old / "report.json",
        old / "progress.pt",
        old / "snapshots/purifier-2000.pt",
    ):
        protected[str(path)] = sha(path)
    for path, expected in protected.items():
        check(sha(path) == expected, "Protected input digest mismatch: " + path)
    engine_sources = {
        str(f.relative_to(args.engine_source.resolve())): sha(f)
        for f in sorted((args.engine_source.resolve() / "src/ccrr").rglob("*.py"))
    }
    check(
        engine_sources == completed["binding"]["source"],
        "Scientific production source changed; probe refused",
    )
    check(not gpu_pids(), "GPU became occupied during input verification")
    check(
        torch.cuda.is_available()
        and torch.cuda.device_count() == 1
        and "H100" in torch.cuda.get_device_name(),
        "One H100 required",
    )
    args.out.mkdir(mode=0o700, parents=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    engine = {"runner": runner, "state": state}
    guard = Guard(args.out, args.max_seconds)
    report = {
        "valid_measurement": False,
        "suite": args.suite,
        "scientific_admission_complete": False,
        "full_training_started": False,
        "workloads": {},
        "engine_source": str(args.engine_source.resolve()),
        "engine_script_sha256": {
            f.name: sha(f) for f in (args.engine_source / "scripts").glob("*.py")
        },
    }
    report["input_verification_seconds"] = time.perf_counter() - process_start
    write(args.out / "report.json", report)
    e = None
    # Shared read lock rejects an old-run writer. No original file is modified.
    with (old / ".r3.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        guard.thread.start()
        try:
            setup_start = time.perf_counter()
            cfg.update(
                output=str(args.out / "resources"),
                resume_a=False,
                resume_b=False,
                b_diagnostic=True,
                b_cache_device="cuda",
            )
            print(
                "H100_REAL_INPUTS_VERIFIED; building original frozen teacher banks",
                flush=True,
            )
            e = Experiment(cfg)
            restore_dependencies(e, "B")
            del e.system, e.pub_attacker
            banks = {r: TeacherBank(e, r) for r in ("fit", "val_select")}
            e.b_teacher_bank, e.b_teacher_banks = banks["fit"], banks
            check(
                len(e.fit) == 1433 and len(e.data["val_select"]) == 170,
                "Unexpected cohort",
            )
            check(
                not (
                    {r["patient_id"] for r in e.fit.rows}
                    & {r["patient_id"] for r in e.data["val_select"].rows}
                ),
                "Patient split overlap",
            )
            # Real independent-audit features from the completed Purifier, never a toy fixture.
            full = torch.load(
                old / "progress.pt", map_location="cpu", weights_only=True
            )
            e.no_branch_control = runner.ReconstructionControl(
                e.purifier.mean, e.purifier.std
            ).to(e.device)
            for name, model in (
                ("purifier", e.purifier),
                ("attacker", e.sem_attacker),
                ("control", e.no_branch_control),
            ):
                model.load_state_dict(full["models"][name])
                model.eval()
            e.feature_audit_tag = "performance-only"
            audit = GPUFeatureAudit(e)
            audit.cache()
            torch.cuda.synchronize()
            report["resource_and_real_feature_cache_seconds"] = (
                time.perf_counter() - setup_start
            )
            report["comparison_contract"] = {
                "probe_source_sha256": {
                    Path(__file__).name: sha(__file__),
                    "b_r3_perf_common.py": sha(
                        Path(__file__).with_name("b_r3_perf_common.py")
                    ),
                    "b_r3_h100_scaling.py": sha(
                        Path(__file__).with_name("b_r3_h100_scaling.py")
                    ),
                    "b_r3_cached_execution.py": sha(
                        Path(__file__).with_name("b_r3_cached_execution.py")
                    ),
                    "b_r3_h100_ab.py": sha(Path(__file__).with_name("b_r3_h100_ab.py")),
                },
                "protected_sha256": protected,
                "seed": e.seed,
                "microbatch": 16,
                "gpu_uuid": GPU,
                "device_name": torch.cuda.get_device_name(),
                "device_total_memory_bytes": torch.cuda.get_device_properties(
                    0
                ).total_memory,
                "driver": subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        GPU,
                        "--query-gpu=driver_version",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=15,
                ).strip(),
                "runtime": environment(),
                "steps": args.steps,
                "warmup": args.warmup,
                "threads": torch.get_num_threads(),
                "interop_threads": torch.get_num_interop_threads(),
                "cublas_workspace": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                "python_hash_seed": os.environ["PYTHONHASHSEED"],
                "repeats": args.repeats,
                "profile_steps": args.profile_steps,
                "equivalence_steps": args.equivalence_steps,
                "precision": "original FP32, no AMP or compilation",
                "workloads": [
                    "purifier",
                    "control",
                    "identity-p1-spatial",
                    "identity-p1-pooled",
                    "identity-p2-spatial",
                    "identity-p2-pooled",
                ],
            }
            for kind in (
                report["comparison_contract"]["workloads"]
                if args.suite in ("baseline", "all")
                else []
            ):
                guard.check()
                state.Timings.path = None
                state.Timings.values = {}
                state.Timings.events = []
                work = Workload(
                    e,
                    banks,
                    audit,
                    engine,
                    args.out / kind,
                    kind,
                    old / "snapshots/purifier-2000.pt",
                )
                print("H100_CASE_START", kind, flush=True)
                report["workloads"][kind] = benchmark(work, args, guard)
                write(args.out / "report.json", report)
                print("H100_CASE_COMPLETE", kind, flush=True)
                del work
            if args.suite in ("scaling", "all"):
                # Import our probe helper, not anything from the archived engine.
                import importlib.util

                scaling_spec = importlib.util.spec_from_file_location(
                    "r3_capacity_probe",
                    Path(__file__).with_name("b_r3_h100_scaling.py"),
                )
                scaling_module = importlib.util.module_from_spec(scaling_spec)
                scaling_spec.loader.exec_module(scaling_module)
                report["scaling"] = scaling_module.run_scaling(audit, args, guard)
            if args.suite == "ab":
                from b_r3_h100_ab import run_ab

                report["ab"] = run_ab(audit, args, guard)
            guard.check()
            for path, expected in protected.items():
                check(
                    sha(path) == expected,
                    "Protected input changed during probe: " + path,
                )
            check(
                all(
                    w["profile"]["activity"]["cuda_kernels_recorded"]
                    for w in report["workloads"].values()
                ),
                "CUDA profiler kernel activity absent; measurement incomplete",
            )
            report["protected_inputs_unchanged"] = True
            report["valid_measurement"] = True
        except BaseException as exc:
            report["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            guard.close()
            if guard.error:
                report["valid_measurement"] = False
                report["gpu_monitor_error"] = guard.error
            if e is not None:
                e.allocator.close()
            report["whole_probe_seconds_including_profiler_and_verification"] = (
                time.perf_counter() - process_start
            )
            report["gpu_samples_file"] = "gpu-samples.json"
            utilization = {}
            for sample in guard.rows:
                fields = [v.strip() for v in sample["nvidia_smi"].split(",")]
                if len(fields) == 5:
                    try:
                        utilization.setdefault(sample["phase"], []).append(
                            (float(fields[2]), float(fields[3]))
                        )
                    except ValueError:
                        pass
            report["gpu_samples_by_phase"] = {
                key: {
                    "samples": len(rows),
                    "sampled_mean_gpu_util_percent": statistics.mean(
                        v[0] for v in rows
                    ),
                    "max_gpu_memory_used_MiB": max(v[1] for v in rows),
                }
                for key, rows in utilization.items()
            }
            write(args.out / "report.json", report)
            write(
                args.out / "summary.json",
                {
                    "valid_measurement": report["valid_measurement"],
                    "error": report.get("error"),
                    "gpu_monitor_error": report.get("gpu_monitor_error"),
                    "protected_inputs_unchanged": report.get(
                        "protected_inputs_unchanged", False
                    ),
                    "resource_and_real_feature_cache_seconds": report.get(
                        "resource_and_real_feature_cache_seconds"
                    ),
                    "whole_probe_seconds_including_profiler_and_verification": report[
                        "whole_probe_seconds_including_profiler_and_verification"
                    ],
                    "workloads": {
                        name: {
                            "median_step_seconds": row["median_step_seconds"],
                            "median_score_seconds": row["median_score_seconds"],
                            "median_cycle_seconds": row["median_cycle_seconds"],
                            "median_full_checkpoint_seconds": statistics.median(
                                r["full_checkpoint"]["wall_seconds"]
                                for r in row["repeats"]
                            ),
                            "median_full_restore_seconds": statistics.median(
                                r["full_restore"]["wall_seconds"]
                                for r in row["repeats"]
                            ),
                            "peak_allocated_GiB": max(
                                r["peak_allocated_bytes"] for r in row["repeats"]
                            )
                            / 1024**3,
                            "peak_reserved_GiB": max(
                                r["peak_reserved_bytes"] for r in row["repeats"]
                            )
                            / 1024**3,
                            "profile": row["profile"],
                        }
                        for name, row in report["workloads"].items()
                    },
                    "gpu_samples_by_phase": report["gpu_samples_by_phase"],
                    "scaling": report.get("scaling", {}).get("brief"),
                    "ab": report.get("ab", {}).get("brief"),
                    "no_optimization_speedup_claimed": args.compare_to is None
                    and not (
                        report["valid_measurement"]
                        and report.get("ab", {}).get("all_exact", False)
                        and report.get("ab", {}).get("cycle_speedup", 0) > 1
                    ),
                    "scientific_admission_complete": False,
                },
            )
    check(
        report["valid_measurement"], "Probe invalidated; inspect report and GPU monitor"
    )
    if args.compare_to:
        write(
            args.out / "comparison.json",
            compare_reports(args.compare_to, args.out / "report.json"),
        )
    print("H100_PROFILE_COMPLETE", args.out / "report.json", flush=True)
    print("No formal training dispatched; no existing checkpoint changed.", flush=True)


if __name__ == "__main__":
    main()
