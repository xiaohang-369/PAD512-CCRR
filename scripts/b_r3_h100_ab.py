"""Same-session, batch-16 reference/candidate benchmark, no full training."""

import copy
import itertools
import statistics
import time

import torch

from b_r3_cached_execution import CachedAudit
from b_r3_h100_scaling import (
    Head,
    execute,
    memory,
    require,
    restore,
    snapshot,
    states,
    write,
)
from b_r3_perf_common import digest, exact, trace_summary


def verify_sampling(reference, candidate, specs, count, guard):
    from ccrr.data.pad import rng_for

    checks = {}
    for scenario, rep, family in specs:
        left, right = copy.copy(reference.e.sampler), copy.copy(candidate.e.sampler)
        left.fallbacks = right.fallbacks = 0
        history = []
        for step in range(count):
            guard.check()
            a = rng_for(
                reference.e.seed, "feature-attack", scenario + rep + family, step
            )
            b = rng_for(
                candidate.e.seed, "feature-attack", scenario + rep + family, step
            )
            x, y = left.sample(a, scenario), right.sample(b, scenario)
            da = [(int(a.integers(3)), int(a.integers(3))) for _ in x]
            db = [(int(b.integers(3)), int(b.integers(3))) for _ in y]
            require(
                x == y
                and da == db
                and exact(a.bit_generator.state, b.bit_generator.state),
                "Candidate sampling/RNG mismatch",
            )
            history.append((x, da))
        require(left.fallbacks == right.fallbacks, "Candidate fallback count mismatch")
        checks["-".join((scenario, rep, family))] = {
            "steps": count,
            "sampling_sha256": digest(history),
            "fallbacks": left.fallbacks,
        }
    return checks


def run_ab(audit, args, guard):
    from profile_b_r3_h100 import timed
    from b_r3_gpu_state import atomic_save

    root = args.out / "ab"
    root.mkdir()
    print("H100_AB_START: 12 heads, batch=16, lanes=2, reference/candidate", flush=True)
    report = {
        "complete": False,
        "all_exact": False,
        "lanes": 2,
        "batch": 16,
        "variants": {},
        "limits": [
            "Same 12 independent heads, identical 100-update blocks; not the complete scientific 5000-update audit.",
            "Cycle includes updates, nine-direction scoring for every head, full save and full restore. Production per-step heartbeat and save-request overhead and scientific bootstrap are excluded.",
            "Candidate cache setup is separately measured and reported; performance ratios alone do not authorize full training.",
            "Memory peaks include both resident head sets and shared feature caches in this process; they are not isolated per-variant memory costs.",
        ],
    }

    def persist():
        write(root / "summary.json", report)

    persist()
    for scenario in ("p1", "p2"):
        audit.pair_table(scenario)
    guard.phase = "ab/candidate-cache"
    started = time.perf_counter()
    candidate = CachedAudit(audit)
    report["candidate_cache_setup_seconds"] = time.perf_counter() - started
    report["pooled_cache_bytes"] = sum(
        v.numel() * v.element_size()
        for role in candidate.pooled_store.values()
        for v in role.values()
    )
    print(
        "H100_AB_CACHE_READY: seconds=",
        report["candidate_cache_setup_seconds"],
        flush=True,
    )
    specs = list(
        itertools.product(
            ("p1", "p2"), ("raw", "control", "sem"), ("spatial", "pooled")
        )
    )
    guard.phase = "ab/sampling-equivalence"
    report["sampling_contract"] = verify_sampling(
        audit, candidate, specs, args.warmup + args.steps + 1, guard
    )
    variants = {
        name: [Head(source, spec) for spec in specs]
        for name, source in [("reference", audit), ("candidate", candidate)]
    }
    require(
        exact(
            {h.name: h.initial for h in variants["reference"]},
            {h.name: h.initial for h in variants["candidate"]},
        ),
        "Initial weights/optimizer mismatch",
    )
    reference_verify = reference_scores = None
    for name, heads in variants.items():
        guard.phase = f"ab/{name}/equality"
        evidence = execute(heads, 2, args.equivalence_steps, guard, verify=True)
        write(root / f"{name}-stepwise.json", evidence)
        scores = {h.name: h.score() for h in heads}
        if name == "reference":
            reference_verify, reference_scores = evidence, scores
        else:
            report["stepwise_sampling_gradients_optimizer_exact"] = exact(
                reference_verify, evidence
            )
            report["nine_direction_scores_exact"] = exact(reference_scores, scores)
            persist()
            require(
                report["stepwise_sampling_gradients_optimizer_exact"]
                and report["nine_direction_scores_exact"],
                "Candidate rejected: update or score differs; inspect ab/summary.json",
            )
        report["variants"][name] = {"repeats": []}
    print(
        "H100_AB_EQUIVALENCE_PASSED: sampling, gradients, optimizer, scores", flush=True
    )
    reference_final = None
    for rep in range(args.repeats):
        for name in (
            ("reference", "candidate") if rep % 2 == 0 else ("candidate", "reference")
        ):
            heads = variants[name]
            print(f"H100_AB_CASE_START: {name} repeat={rep}", flush=True)
            guard.phase = f"ab/{name}/warmup/{rep}"
            for h in heads:
                h.restore()
            execute(heads, 2, args.warmup, guard)
            torch.cuda.reset_peak_memory_stats()
            guard.phase = f"ab/{name}/updates/{rep}"
            _, timing = timed(lambda: execute(heads, 2, args.steps, guard))
            guard.phase = f"ab/{name}/scoring/{rep}"
            _, scoring = timed(lambda: [h.score() for h in heads])
            final = states(heads)
            if reference_final is None:
                reference_final = final
                torch.save(final, root / "reference-final.pt")
            same = exact(reference_final, final)
            row = {
                "repeat": rep,
                "updates": timing,
                "score_all_heads_nine_directions": scoring,
                "final_exact": same,
            }
            report["variants"][name]["repeats"].append(row)
            persist()
            require(
                same,
                "Candidate or repeat final model/gradient/optimizer/score mismatch",
            )
            guard.phase = f"ab/{name}/checkpoint/{rep}"
            checkpoint = root / f"{name}-{rep}.pt"
            _, saving = timed(lambda: atomic_save(checkpoint, snapshot(heads)))
            expected = snapshot(heads)
            _, loading = timed(
                lambda: restore(
                    heads, torch.load(checkpoint, weights_only=True, map_location="cpu")
                )
            )
            require(exact(expected, snapshot(heads)), "Full restore mismatch")
            execute(heads, 2, 1, guard)
            next_state = states(heads)
            restore(
                heads, torch.load(checkpoint, weights_only=True, map_location="cpu")
            )
            execute(heads, 2, 1, guard)
            require(
                exact(next_state, states(heads)), "Continuation mismatch after restore"
            )
            row.update(
                full_checkpoint=saving,
                full_restore=loading,
                restore_and_continuation_exact=True,
                cycle_seconds=sum(
                    t["wall_seconds"] for t in (timing, scoring, saving, loading)
                ),
                pairs_per_second=len(heads) * args.steps * 16 / timing["wall_seconds"],
                head_updates=len(heads) * args.steps,
                amortized_seconds_per_head_update=timing["wall_seconds"]
                / (len(heads) * args.steps),
                **memory(),
            )
            print("H100_AB_REPEAT", name, row, flush=True)
            persist()
    # Separate instrumented update windows; timings above contain no profiler.
    for name, heads in variants.items():
        guard.phase = f"ab/{name}/profiler"
        for h in heads:
            h.restore()
        execute(heads, 2, args.warmup, guard)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            with torch.profiler.record_function("R3_PROFILE_WINDOW"):
                execute(heads, 2, args.profile_steps, guard)
                torch.cuda.synchronize()
        trace = root / f"{name}-trace.json"
        prof.export_chrome_trace(str(trace))
        activity = trace_summary(trace)
        report["variants"][name]["profile_activity"] = activity
        persist()
        require(
            activity["cuda_kernels_recorded"],
            "CUDA kernels absent; incomplete measurement",
        )
    for name, row in report["variants"].items():
        for key, field in [
            ("update", "updates"),
            ("score", "score_all_heads_nine_directions"),
            ("save", "full_checkpoint"),
            ("restore", "full_restore"),
        ]:
            row[f"median_{key}_seconds"] = statistics.median(
                r[field]["wall_seconds"] for r in row["repeats"]
            )
        row["median_cycle_seconds"] = statistics.median(
            r["cycle_seconds"] for r in row["repeats"]
        )
    a, b = report["variants"]["reference"], report["variants"]["candidate"]
    report.update(
        complete=True,
        all_exact=True,
        update_speedup=a["median_update_seconds"] / b["median_update_seconds"],
        score_speedup=a["median_score_seconds"] / b["median_score_seconds"],
        cycle_speedup=a["median_cycle_seconds"] / b["median_cycle_seconds"],
        first_cycle_seconds_including_cache=b["median_cycle_seconds"]
        + report["candidate_cache_setup_seconds"],
        first_cycle_speedup_including_cache=a["median_cycle_seconds"]
        / (b["median_cycle_seconds"] + report["candidate_cache_setup_seconds"]),
    )
    report["brief"] = {
        k: v for k, v in report.items() if k not in ("variants", "sampling_contract")
    }
    report["brief"]["variants"] = {
        name: {k: v for k, v in row.items() if k != "repeats"}
        for name, row in report["variants"].items()
    }
    persist()
    print("H100_AB_COMPLETE: inspect summary.json for measured ratios", flush=True)
    return report
