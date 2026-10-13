"""Paired synthetic H100 throughput gate. No medical data, teachers or B dispatch."""

import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_stage_b_execution import context
from test_stage_a_training_loops import compare
from b_r3_state import State as OldState
from b_r3_gpu_state import State, Timings
from b_r3_gpu_features import GPUFeatureAudit
from ccrr.errors import require
from ccrr.runtime.support import environment, seed_everything, write_json
from ccrr.training.feature_audit import FeatureAudit
from ccrr.training.optimization import optimizer
from ccrr.representation.purifier import SemanticAttacker
from ccrr.evaluation.attacks import PooledAttacker


def fixture(path):
    e = context(path, micro=16)
    ds = copy.copy(e.data["val_select"])
    ds.rows = [{**r, "patient_id": str(i // 2)} for i, r in enumerate(ds.rows)]
    ds.index = {str(i): i for i in range(6)}
    ds.root = path
    e.data["val_select"] = ds
    pair_root = path / "protocol_audit_20261009/pair_protocol"
    pair_root.mkdir(parents=True)
    queries = [
        {
            "query": str(i),
            "patient_id": str(i // 2),
            "positive_gallery": [str(i ^ 1)],
            "negative_gallery": [str((i + 2) % 6), str((i + 3) % 6)],
        }
        for _ in range(64)
        for i in range(6)
    ]
    (pair_root / "val_select_p1.json").write_text(json.dumps({"queries": queries}))
    seed_everything(144)
    storage = {"sem": torch.randn(6, 3, 768, 14, 14), "v": torch.ones(6, 1, 14, 14)}
    return e, storage


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_state(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_state(v) for v in value)
    return copy.deepcopy(value)


def arm(path, fast, family, steps):
    e, storage = fixture(path)
    seed_everything(93)
    model = (SemanticAttacker() if family == "spatial" else PooledAttacker()).to(
        e.device
    )
    opt = optimizer([model], [1e-4])
    audit = (GPUFeatureAudit if fast else FeatureAudit)(e)
    State.reset_registry()
    Timings.start(path / "timings.json")
    # Include the resident-cache copy in the measured workflow.
    Timings.sync()
    start = time.perf_counter()
    audit.store = {
        role: {k: v.to(e.device) if fast else v for k, v in storage.items()}
        for role in ("fit", "val_select")
    }
    state = (State if fast else OldState)(
        path / "head.pt", {"synthetic": True}, {"head": model}, {"head": opt}
    )
    state.data = {"next": 0, "evaluated": 0, "complete": False}
    state.save()
    for step in range(steps):
        Timings.call(
            "update", audit.attacker_update, model, opt, "sem", "p1", family, step
        )
        state.data["next"] = step + 1
        Timings.call("save_request", state.save)
    if fast:
        State.flush_pending()
    scores = Timings.call(
        "nine_direction_score", audit.attack_scores, model, "sem", "p1"
    )
    Timings.sync()
    elapsed = time.perf_counter() - start
    Timings.write()
    return {
        "seconds": elapsed,
        "timings": copy.deepcopy(Timings.values),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if torch.cuda.is_available()
        else 0,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved()
        if torch.cuda.is_available()
        else 0,
    }, {
        "model": cpu_state(model.state_dict()),
        "optimizer": cpu_state(opt.state_dict()),
        "scores": scores,
        "rng": torch.get_rng_state().clone(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--cpu-smoke", action="store_true")
    args = p.parse_args()
    require(not args.out.exists(), "Benchmark output must be new")
    if not args.cpu_smoke:
        require(
            torch.cuda.is_available() and "H100" in torch.cuda.get_device_name(),
            "H100 required",
        )
        import os

        os.environ["CCRR_TEST_DEVICE"] = "cuda"
    args.out.mkdir(parents=True)
    rows = []
    repetitions, steps = (1, 4) if args.cpu_smoke else (3, 100)
    for family in ("spatial", "pooled"):
        for rep in range(repetitions):
            # Alternating order avoids always giving the second arm warmed kernels.
            results = {}
            for fast in (False, True) if rep % 2 == 0 else (True, False):
                label = "candidate" if fast else "baseline"
                results[label] = arm(
                    args.out / f"{family}-{rep}-{label}", fast, family, steps
                )
            compare(results["baseline"][1], results["candidate"][1])
            row = {
                "family": family,
                "repeat": rep,
                "steps": steps,
                "baseline": results["baseline"][0],
                "candidate": results["candidate"][0],
                "exact_state_and_scores": True,
            }
            rows.append(row)
            print("R3_GPU_BENCH:", json.dumps(row), flush=True)
    ratios = {
        f: statistics.median(
            [
                r["baseline"]["seconds"] / r["candidate"]["seconds"]
                for r in rows
                if r["family"] == f
            ]
        )
        for f in ("spatial", "pooled")
    }
    faster = all(v > 1 for v in ratios.values())
    write_json(
        args.out / "report.json",
        {
            "environment": environment(),
            "rows": rows,
            "median_paired_speedup": ratios,
            "engineering_passed": not args.cpu_smoke and faster,
            "synthetic_only": True,
            "medical_speedup_verified": False,
            "full_evaluation_completed": False,
            "scientific_admission_complete": False,
            "limitations": "Synthetic repeated pairs favor cache reuse. Not a real-data total-time forecast.",
        },
    )
    if not args.cpu_smoke:
        require(faster, "No measured speedup for both families; do not start new R3")


if __name__ == "__main__":
    main()
