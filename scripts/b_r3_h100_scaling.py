"""Bounded real-feature capacity probes; never a production training entry point.

Batch >16 changes the experiment. Stream concurrency keeps each head's original
update code, named stateless sampling RNG, optimizer and batch=16. All modes do
identical total work. Scoring and checkpointing use coordinator barriers; no MPS,
subprocess CUDA contexts, environment installs or scientific budget changes.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
import copy
import itertools
import json
from pathlib import Path
import random
import statistics

import numpy as np
import torch
from torch.nn import functional as F

from b_r3_perf_common import cpu_copy, digest, exact, trace_summary


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rng_state():
    n = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (n[0], n[1].tolist(), *n[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(value):
    random.setstate(value["python"])
    n = value["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(value["torch"])
    if value["cuda"]:
        torch.cuda.set_rng_state_all(value["cuda"])


class Head:
    def __init__(self, source_audit, spec):
        from ccrr.data.pad import rng_for
        from ccrr.runtime.support import seed_everything
        from ccrr.training.b_diagnostics import isolated_rng
        from ccrr.training.optimization import optimizer
        from ccrr.representation.purifier import SemanticAttacker
        from ccrr.evaluation.attacks import PooledAttacker

        self.scenario, self.rep, self.family = spec
        self.name = "identity-" + "-".join(spec)
        # Immutable feature storage/patient metadata may be shared. Mutable sampler
        # accounting, configuration, score caches, models and optimizers may not.
        self.audit = copy.copy(source_audit)
        self.audit.e = copy.copy(source_audit.e)
        self.audit.e.cfg = dict(source_audit.e.cfg, microbatch=16)
        self.audit.e.sampler = copy.copy(source_audit.e.sampler)
        self.audit.e.sampler.fallbacks = 0
        self.audit.pair_tables = dict(getattr(source_audit, "pair_tables", {}))
        self.e = self.audit.e
        self.seed = int(rng_for(self.e.seed, "B-readout", self.name, 0).integers(2**31))
        # Only main-thread initialization consumes global RNG, then restores it.
        with isolated_rng():
            seed_everything(self.seed)
            self.model = (
                SemanticAttacker() if self.family == "spatial" else PooledAttacker()
            ).to(self.e.device)
            self.opt = optimizer([self.model], [1e-4])
        self.next = 0
        self.evaluated = 0
        self.best_score = -1.0
        self.best = self.best_weights = None
        self.initial = self.capture()

    def update(self):
        self.audit.attacker_update(
            self.model, self.opt, self.rep, self.scenario, self.family, self.next
        )
        self.next += 1

    def batch_update(self, batch):
        """Capacity experiment: actual one-forward B pairs, loss normalized by B.

        B=16 is checked against the engine. Larger B is explicitly NON-equivalent.
        Each block preserves the sampler's 8 positive / 8 negative composition.
        """
        from ccrr.data.pad import rng_for
        from ccrr.training.optimization import schedule, update

        require(batch in (16, 32, 64, 128), "Unsupported capacity batch")
        self.model.train()
        self.opt.zero_grad()
        rng = rng_for(
            self.e.seed,
            "feature-attack",
            self.scenario + self.rep + self.family,
            self.next,
        )
        pairs, domains = [], []
        for _ in range(batch // 16):
            block = self.e.sampler.sample(rng, self.scenario)
            require(len(block) == 16, "Original pair sampler must return 16")
            pairs.extend(block)
            domains.extend((int(rng.integers(3)), int(rng.integers(3))) for _ in block)
        fa, va = self.audit.feature_batch(
            "fit", [p[0] for p in pairs], self.rep, [d[0] for d in domains]
        )
        fb, vb = self.audit.feature_batch(
            "fit", [p[1] for p in pairs], self.rep, [d[1] for d in domains]
        )
        logit = self.model(fa, va, fb, vb)
        (
            F.binary_cross_entropy_with_logits(
                logit, logit.new_tensor([p[2] for p in pairs]), reduction="sum"
            )
            / batch
        ).backward()
        schedule(self.opt, self.next, 5000)
        update(self.opt)
        self.next += 1

    def score(self):
        from sklearn.metrics import roc_auc_score

        scores = self.audit.attack_scores(self.model, self.rep, self.scenario)
        auc = [
            roc_auc_score(r["labels"], r["scores"], sample_weight=r["weights"])
            for r in scores
        ]
        key = float(np.mean([max(a, 1 - a) for a in auc]))
        if key > self.best_score:
            self.best_score, self.best = key, scores
            self.best_weights = cpu_copy(self.model.state_dict())
        self.evaluated = self.next
        return scores

    def capture(self, gradients=False):
        value = {
            "name": self.name,
            "initial_seed": self.seed,
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "next": self.next,
            "sampler_fallbacks": self.e.sampler.fallbacks,
            "evaluated": self.evaluated,
            "best_score": self.best_score,
            "best": self.best,
            "best_weights": self.best_weights,
            "sampling_rng": {
                "seed": self.e.seed,
                "stage": "feature-attack",
                "role": self.scenario + self.rep + self.family,
                "next": self.next,
            },
        }
        if gradients:
            value["gradients"] = {n: p.grad for n, p in self.model.named_parameters()}
        return cpu_copy(value)

    def restore(self, saved=None):
        value = self.initial if saved is None else saved
        require(
            value["name"] == self.name and value["initial_seed"] == self.seed,
            "Head identity mismatch",
        )
        self.model.load_state_dict(value["model"])
        # Optimizer can alias CPU tensors from load_state_dict on CPU fixtures.
        self.opt.load_state_dict(copy.deepcopy(value["optimizer"]))
        for p in self.model.parameters():
            p.grad = None
        self.next = value["next"]
        self.e.sampler.fallbacks = value["sampler_fallbacks"]
        self.evaluated, self.best_score = value["evaluated"], value["best_score"]
        self.best, self.best_weights = (
            copy.deepcopy(value["best"]),
            cpu_copy(value["best_weights"]),
        )
        require(
            value["sampling_rng"]
            == {
                "seed": self.e.seed,
                "stage": "feature-attack",
                "role": self.scenario + self.rep + self.family,
                "next": self.next,
            },
            "Sampling stream mismatch",
        )


@contextmanager
def capture_unclipped(head):
    """Used only in the separate equality pass; one backward at batch=16."""
    grads, handles = {}, []
    for name, param in head.model.named_parameters():

        def hook(grad, name=name):
            require(name not in grads, "Unexpected repeated backward in a head update")
            grads[name] = grad.detach().clone()

        handles.append(param.register_hook(hook))
    try:
        yield grads
    finally:
        for handle in handles:
            handle.remove()


def execute(heads, lanes, steps, guard, *, verify=False):
    """Single CUDA context, one stream per task, no shared mutable head state."""
    from profile_b_r3_h100 import sampling_trace

    require(lanes in (1, 2, 4), "Unsupported concurrency")
    cuda = str(heads[0].e.device).startswith("cuda")
    before = rng_state()
    if cuda:
        torch.cuda.synchronize()
    streams = {h.name: torch.cuda.Stream() if cuda else None for h in heads}

    def run(head):
        records = []
        stream = streams[head.name]
        with torch.cuda.stream(stream) if cuda else nullcontext():
            for _ in range(steps):
                guard.check()
                if verify:
                    with (
                        sampling_trace(head.e, {}, head.audit) as samples,
                        capture_unclipped(head) as grads,
                    ):
                        head.update()
                    records.append(
                        {
                            "samples": digest(samples),
                            "unclipped_gradients": digest(grads),
                            "state": digest(head.capture(gradients=True)),
                        }
                    )
                else:
                    head.update()
            if stream is not None:
                stream.synchronize()
        return records

    if lanes == 1:
        result = {h.name: run(h) for h in heads}
    else:
        with ThreadPoolExecutor(max_workers=lanes) as pool:
            futures = {h.name: pool.submit(run, h) for h in heads}
            result = {name: f.result() for name, f in futures.items()}
    require(
        exact(before, rng_state()),
        "Concurrent update consumed shared global RNG; reject candidate",
    )
    return result


def snapshot(heads):
    return {
        "format": "R3-CAPACITY-PROBE-1",
        "heads": {h.name: h.capture() for h in heads},
        "rng": cpu_copy(rng_state()),
        "formal_training": False,
    }


def restore(heads, value):
    require(value["format"] == "R3-CAPACITY-PROBE-1", "Wrong probe checkpoint")
    require(set(value["heads"]) == {h.name for h in heads}, "Wrong head set")
    for head in heads:
        head.restore(value["heads"][head.name])
    restore_rng(value["rng"])


def states(heads):
    return {h.name: h.capture(gradients=True) for h in heads}


def memory():
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def run_scaling(audit, args, guard):
    from profile_b_r3_h100 import timed
    from b_r3_gpu_state import atomic_save

    root = args.out / "scaling"
    root.mkdir()
    report = {
        "complete": False,
        "batch_sweep": {},
        "parallel": {},
        "formal_training_started": False,
        "limits": [
            "Capacity tests use independent identity attackers, not the complete Purifier objective.",
            "Batch >16 changes gradient statistics and is not equivalent acceleration.",
            "100-update blocks plus explicit end scoring/checkpoint are probe workloads, not the full 5000-update scientific audit.",
            "Scoring/checkpoints remain serial at coordinator barriers. No scientific bootstrap benchmark.",
            "Concurrency speedup compares this probe's identical update/scoring/save/restore blocks, not total production R3 wall time. Original per-step save-request/heartbeat bookkeeping is not included in these blocks.",
            "Per-head named sampling RNG is stateless; global RNG must remain unchanged during parallel updates.",
        ],
    }

    def persist():
        write(root / "summary.json", report)

    # Prebuild immutable endpoint tables before concurrency. Scoring is serial.
    for scenario in ("p1", "p2"):
        audit.pair_table(scenario)
    specs = list(
        itertools.product(
            ("p1", "p2"), ("raw", "control", "sem"), ("spatial", "pooled")
        )
    )
    heads = [Head(audit, spec) for spec in specs]
    originals = {h.name: digest(h.initial) for h in heads}
    report["initial_state_sha256"] = originals
    persist()
    reference_verify = None
    reference_final = None
    # Repeat order alternates to reduce always-first warmup/thermal bias.
    order = [
        (rep, lanes)
        for rep in range(args.repeats)
        for lanes in ((1, 2, 4) if rep % 2 == 0 else (4, 2, 1))
    ]
    for rep, lanes in order:
        guard.phase = f"parallel/{lanes}/equality/{rep}"
        for h in heads:
            h.restore()
        verification = execute(heads, lanes, args.equivalence_steps, guard, verify=True)
        write(root / f"verification-{lanes}-{rep}.json", verification)
        if reference_verify is None:
            reference_verify = verification
        equivalent = exact(reference_verify, verification)
        row = report["parallel"].setdefault(str(lanes), {"repeats": []})
        row["stepwise_parameters_gradients_optimizer_sampling_exact"] = equivalent
        if not equivalent:
            row["rejected"] = (
                "Stepwise equality mismatch; no speedup claim or promotion"
            )
            persist()
            continue
        for h in heads:
            h.restore()
        execute(heads, lanes, args.warmup, guard)
        torch.cuda.reset_peak_memory_stats()
        guard.phase = f"parallel/{lanes}/updates/{rep}"
        _, timing = timed(lambda: execute(heads, lanes, args.steps, guard))
        guard.phase = f"parallel/{lanes}/scoring/{rep}"
        _, scoring = timed(lambda: {h.name: h.score() for h in heads})
        final = states(heads)
        if reference_final is None:
            reference_final = final
            torch.save(reference_final, root / "serial-reference.pt")
        final_equal = exact(reference_final, final)
        differences = {
            name: [
                key
                for key in value
                if not exact(reference_final[name][key], value[key])
            ]
            for name, value in final.items()
            if not exact(reference_final[name], value)
        }
        checkpoint = root / f"parallel-{lanes}-{rep}.pt"
        guard.phase = f"parallel/{lanes}/checkpoint/{rep}"
        # Include GPU-to-CPU copies, full state and durable file IO in save timing.
        _, saving = timed(lambda: atomic_save(checkpoint, snapshot(heads)))
        expected = snapshot(heads)
        _, loading = timed(
            lambda: restore(
                heads, torch.load(checkpoint, map_location="cpu", weights_only=True)
            )
        )
        recovered = exact(expected, snapshot(heads))
        execute(heads, lanes, 1, guard)
        continuation = states(heads)
        restore(heads, torch.load(checkpoint, map_location="cpu", weights_only=True))
        execute(heads, 1, 1, guard)
        recovered = recovered and exact(continuation, states(heads))
        require(recovered, "Full bundle restore or serial continuation mismatch")
        sample = {
            "repeat": rep,
            "heads": len(heads),
            "updates_per_head": args.steps,
            "total_pair_updates": len(heads) * args.steps * 16,
            "updates": timing,
            "score_all_heads_nine_directions": scoring,
            "full_checkpoint": saving,
            "full_restore": loading,
            "cycle_seconds": sum(
                t["wall_seconds"] for t in (timing, scoring, saving, loading)
            ),
            "pairs_per_second": len(heads) * args.steps * 16 / timing["wall_seconds"],
            "final_models_gradients_optimizer_progress_scores_exact": final_equal,
            "different_heads_and_fields": differences,
            "restore_and_continuation_exact": recovered,
            **memory(),
        }
        row["repeats"].append(sample)
        print("H100_PARALLEL_REPEAT", lanes, json.dumps(sample), flush=True)
        persist()
    for lanes, row in report["parallel"].items():
        good = (
            len(row["repeats"]) == args.repeats
            and all(
                r["final_models_gradients_optimizer_progress_scores_exact"]
                for r in row["repeats"]
            )
            and not row.get("rejected")
        )
        row["eligible_for_performance_comparison"] = good
        if good:
            row["median_cycle_seconds"] = statistics.median(
                r["cycle_seconds"] for r in row["repeats"]
            )
            row["median_update_seconds"] = statistics.median(
                r["updates"]["wall_seconds"] for r in row["repeats"]
            )
    serial = report["parallel"]["1"]
    for lanes, row in report["parallel"].items():
        if (
            row["eligible_for_performance_comparison"]
            and serial["eligible_for_performance_comparison"]
        ):
            row["cycle_speedup_vs_same_serial_work"] = (
                serial["median_cycle_seconds"] / row["median_cycle_seconds"]
            )
    persist()

    # Single-model batch capacity scans, same starting weights for every B.
    for family in ("spatial", "pooled"):
        h = next(
            h for h in heads if (h.scenario, h.rep, h.family) == ("p1", "sem", family)
        )
        h.restore()
        for _ in range(args.equivalence_steps):
            h.update()
        reference = h.capture(gradients=True)
        h.restore()
        for _ in range(args.equivalence_steps):
            h.batch_update(16)
        require(
            exact(reference, h.capture(gradients=True)),
            "Batch sweep B=16 does not match original update",
        )
        for batch in (16, 32, 64, 128):
            key = f"{family}/batch-{batch}"
            rows, equality = [], None
            report["batch_sweep"][key] = {
                "equivalent_to_original_experiment": batch == 16,
                "repeats": rows,
            }
            for rep in range(args.repeats):
                guard.phase = f"batch/{key}/{rep}"
                h.restore()
                for _ in range(args.warmup):
                    guard.check()
                    h.batch_update(batch)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

                def loop():
                    for _ in range(args.steps):
                        guard.check()
                        h.batch_update(batch)

                _, timing = timed(loop)
                final = h.capture(gradients=True)
                if equality is None:
                    equality = final
                require(exact(equality, final), "Batch-scan repeat not deterministic")
                rows.append(
                    {
                        "repeat": rep,
                        "batch": batch,
                        "steps": args.steps,
                        "seconds_per_step": timing["wall_seconds"] / args.steps,
                        "pairs_per_second": batch * args.steps / timing["wall_seconds"],
                        **timing,
                        **memory(),
                    }
                )
                print("H100_BATCH_REPEAT", key, json.dumps(rows[-1]), flush=True)
                persist()
            report["batch_sweep"][key]["median_pairs_per_second"] = statistics.median(
                r["pairs_per_second"] for r in rows
            )
    # Profiler is separated from throughput; CUPTI must actually capture kernels.
    for lanes in (1, 2, 4):
        guard.phase = f"parallel/{lanes}/profiler"
        for h in heads:
            h.restore()
        execute(heads, lanes, args.warmup, guard)
        trace = root / f"parallel-{lanes}-trace.json"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            with torch.profiler.record_function("R3_PROFILE_WINDOW"):
                execute(heads, lanes, args.profile_steps, guard)
                torch.cuda.synchronize()
        prof.export_chrome_trace(str(trace))
        activity = trace_summary(trace)
        report["parallel"][str(lanes)]["profile_activity"] = activity
        require(
            activity["cuda_kernels_recorded"],
            "No CUDA kernel activity in parallel trace; profile invalid",
        )
        persist()

    report["complete"] = True
    report["brief"] = {
        "complete": True,
        "batch_scope": "Independent spatial/pooled identity attackers only; larger batch is non-equivalent",
        "batch_sweep": {
            name: {
                "median_pairs_per_second": row["median_pairs_per_second"],
                "median_seconds_per_step": statistics.median(
                    r["seconds_per_step"] for r in row["repeats"]
                ),
                "peak_allocated_GiB": max(
                    r["peak_allocated_bytes"] for r in row["repeats"]
                )
                / 1024**3,
            }
            for name, row in report["batch_sweep"].items()
        },
        "parallel": {
            lanes: {
                **{k: v for k, v in row.items() if k != "repeats"},
                "median_score_seconds": statistics.median(
                    r["score_all_heads_nine_directions"]["wall_seconds"]
                    for r in row["repeats"]
                )
                if row["repeats"]
                else None,
                "median_save_seconds": statistics.median(
                    r["full_checkpoint"]["wall_seconds"] for r in row["repeats"]
                )
                if row["repeats"]
                else None,
                "median_restore_seconds": statistics.median(
                    r["full_restore"]["wall_seconds"] for r in row["repeats"]
                )
                if row["repeats"]
                else None,
            }
            for lanes, row in report["parallel"].items()
        },
        "full_results": "scaling/summary.json",
        "formal_training_started": False,
    }
    persist()
    return report
