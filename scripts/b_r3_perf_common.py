"""Read-only comparison and trace accounting for isolated real-data R3 probes."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


def cpu_copy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().clone()
    if isinstance(x, dict):
        return {k: cpu_copy(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(cpu_copy(v) for v in x)
    return x


def digest(x):
    h = hashlib.sha256()

    def visit(v):
        h.update(type(v).__name__.encode() + b":")
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu().contiguous()
            h.update(str((t.dtype, tuple(t.shape))).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                visit(k)
                visit(v[k])
        elif isinstance(v, (list, tuple)):
            for item in v:
                visit(item)
        elif isinstance(v, np.ndarray):
            h.update(str((v.dtype, v.shape)).encode())
            h.update(v.tobytes())
        else:
            h.update(repr(v).encode())

    visit(x)
    return h.hexdigest()


def exact(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def interval_union(intervals):
    total, end = 0.0, -math.inf
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def trace_summary(path):
    events = json.loads(Path(path).read_text())["traceEvents"]
    markers = [
        e for e in events if e.get("name") == "R3_PROFILE_WINDOW" and e.get("ph") == "X"
    ]
    # Kineto can emit both CPU and GPU annotations for one record_function.
    # The GPU annotation spans device work, not the bounded host workload.
    windows = [e for e in markers if e.get("cat") == "user_annotation"]
    if len(windows) != 1:
        metadata = [
            {k: e.get(k) for k in ("cat", "pid", "tid", "ts", "dur")} for e in markers
        ]
        raise ValueError(
            f"Expected exactly one CPU user_annotation profiler window; markers={metadata}"
        )
    lo, hi = windows[0]["ts"], windows[0]["ts"] + windows[0]["dur"]
    kernels, copies, runtime_sync = [], [], []
    for e in events:
        if e.get("ph") != "X" or not e.get("dur"):
            continue
        a, b = max(lo, e["ts"]), min(hi, e["ts"] + e["dur"])
        if b <= a:
            continue
        category = str(e.get("cat", "")).lower()
        if category == "kernel":
            kernels.append((a, b))
        elif "gpu_memcpy" in category or "gpu_memset" in category:
            copies.append((a, b))
        if "runtime" in category and "synchronize" in e.get("name", "").lower():
            runtime_sync.append((a, b))
    active = interval_union(kernels + copies)
    return {
        "window_category": windows[0]["cat"],
        "window_marker_count": len(markers),
        "window_seconds": (hi - lo) / 1e6,
        "kernel_union_seconds": interval_union(kernels) / 1e6,
        "copy_or_memset_union_seconds": interval_union(copies) / 1e6,
        "gpu_active_union_seconds": active / 1e6,
        "window_without_recorded_gpu_activity_seconds": max(0, hi - lo - active) / 1e6,
        "host_sync_api_union_seconds": interval_union(runtime_sync) / 1e6,
        "cuda_kernels_recorded": bool(kernels),
        "note": "Profiler-instrumented window only. Empty GPU intervals are not proof of a specific CPU bottleneck; categories overlap. Not SM occupancy.",
    }


def compare_reports(baseline, candidate):
    a, b = (json.loads(Path(p).read_text()) for p in (baseline, candidate))
    if a["comparison_contract"] != b["comparison_contract"]:
        raise ValueError(
            "Different data/weights/seed/workload/runtime/device/settings; comparison refused"
        )
    if not a["valid_measurement"] or not b["valid_measurement"]:
        raise ValueError(
            "Incomplete or GPU-contaminated measurement; comparison refused"
        )
    results = {}
    for name, old in a["workloads"].items():
        new = b["workloads"][name]
        same = old["equivalence_step_hashes"] == new["equivalence_step_hashes"]
        old_state = torch.load(
            Path(baseline).parent / old["state_file"], weights_only=True
        )
        new_state = torch.load(
            Path(candidate).parent / new["state_file"], weights_only=True
        )
        same = same and exact(old_state, new_state)
        results[name] = {
            "exact_parameters_gradients_optimizer_rng_progress_scores": same,
            "initial_state_equal": old["initial_state_sha256"]
            == new["initial_state_sha256"],
            "sampling_equal": old["sampling_sha256"] == new["sampling_sha256"],
            "baseline_step_seconds": old["median_step_seconds"],
            "candidate_step_seconds": new["median_step_seconds"],
            "update_speedup": old["median_step_seconds"] / new["median_step_seconds"],
            "baseline_score_seconds": old["median_score_seconds"],
            "candidate_score_seconds": new["median_score_seconds"],
            "baseline_cycle_seconds": old["median_cycle_seconds"],
            "candidate_cycle_seconds": new["median_cycle_seconds"],
            "cycle_speedup": old["median_cycle_seconds"] / new["median_cycle_seconds"],
        }
    return {
        "workloads": results,
        "all_exact": all(
            r["exact_parameters_gradients_optimizer_rng_progress_scores"]
            and r["initial_state_equal"]
            and r["sampling_equal"]
            for r in results.values()
        ),
        "scientific_admission_complete": False,
        "full_training_authorized_by_this_report": False,
    }
