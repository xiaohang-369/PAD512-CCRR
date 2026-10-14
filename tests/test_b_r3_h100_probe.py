"""Probe contract/recovery tests. These are NOT H100 performance evidence."""

import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import b_r3_gpu_state as state
import check_stage_b_short_multi_gpu as runner
from b_r3_perf_common import (
    exact,
    digest,
    interval_union,
    trace_summary,
    compare_reports,
)
from profile_b_r3_h100 import Workload, sampling_trace, benchmark
from b_r3_gpu_features import GPUFeatureAudit
from ccrr.training.b_batches import TeacherBank
from test_stage_b_execution import context
from test_b_r3_gpu import score_context


def test_trace_union_clips_to_bounded_window_and_does_not_double_count(tmp_path):
    assert interval_union([(0, 10), (5, 20), (30, 35)]) == 25
    path = tmp_path / "trace.json"
    path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "name": "R3_PROFILE_WINDOW",
                        "cat": "user_annotation",
                        "ph": "X",
                        "ts": 100,
                        "dur": 100,
                    },
                    {
                        "name": "R3_PROFILE_WINDOW",
                        "cat": "gpu_user_annotation",
                        "ph": "X",
                        "ts": 105,
                        "dur": 40,
                    },
                    {"cat": "kernel", "ph": "X", "ts": 90, "dur": 30},
                    {"cat": "kernel", "ph": "X", "ts": 110, "dur": 30},
                    {"cat": "gpu_memcpy", "ph": "X", "ts": 130, "dur": 20},
                    {
                        "cat": "cuda_runtime",
                        "name": "cudaDeviceSynchronize",
                        "ph": "X",
                        "ts": 140,
                        "dur": 20,
                    },
                ]
            }
        )
    )
    r = trace_summary(path)
    assert r["window_category"] == "user_annotation"
    assert r["window_marker_count"] == 2
    assert r["kernel_union_seconds"] == 40 / 1e6
    assert r["gpu_active_union_seconds"] == 50 / 1e6
    assert r["window_without_recorded_gpu_activity_seconds"] == 50 / 1e6
    assert r["host_sync_api_union_seconds"] == 20 / 1e6


@pytest.mark.parametrize(
    "categories", [[], ["gpu_user_annotation"], ["user_annotation", "user_annotation"]]
)
def test_trace_refuses_missing_or_ambiguous_cpu_window(tmp_path, categories):
    path = tmp_path / "trace.json"
    path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "name": "R3_PROFILE_WINDOW",
                        "cat": cat,
                        "ph": "X",
                        "ts": 100,
                        "dur": 100,
                    }
                    for cat in categories
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="CPU user_annotation"):
        trace_summary(path)


def test_comparator_refuses_contaminated_or_different_contracts(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(
        json.dumps({"comparison_contract": {"seed": 1}, "valid_measurement": True})
    )
    b.write_text(
        json.dumps({"comparison_contract": {"seed": 2}, "valid_measurement": True})
    )
    with pytest.raises(ValueError, match="Different"):
        compare_reports(a, b)
    b.write_text(
        json.dumps({"comparison_contract": {"seed": 1}, "valid_measurement": False})
    )
    with pytest.raises(ValueError, match="contaminated"):
        compare_reports(a, b)


def test_exact_digest_covers_gradients_optimizer_and_rng():
    a = {"gradient": torch.tensor([1.0]), "optimizer": {"step": 2}, "rng": [3]}
    b = {"gradient": torch.tensor([1.0]), "optimizer": {"step": 2}, "rng": [3]}
    assert exact(a, b) and digest(a) == digest(b)
    b["gradient"].add_(torch.finfo(torch.float32).eps)
    assert not exact(a, b) and digest(a) != digest(b)


def test_comparator_rejects_empty_scaling_only_reports(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {"comparison_contract": {}, "valid_measurement": True, "workloads": {}}
        )
    )
    with pytest.raises(ValueError, match="workload sets"):
        compare_reports(path, path)


@pytest.mark.parametrize("kind", ["purifier", "control"])
def test_real_update_probe_resets_without_mutating_source_snapshot(tmp_path, kind):
    e = context(tmp_path / "resources", micro=16)
    banks = {"fit": TeacherBank(e)}
    st = runner.make_state(e, {"source": "synthetic"}, False)
    st.schedules["purifier"].next_step = 2000
    st.schedules["attacker"].next_step = 4000
    st.data["next"] = 2000
    source = tmp_path / "source.pt"
    st.save(source)
    original = source.read_bytes()
    work = Workload(
        e,
        banks,
        None,
        {"state": state, "runner": runner},
        tmp_path / kind,
        kind,
        source,
    )
    start = work.capture()
    with sampling_trace(e, banks, None) as samples:
        work.update()
    first, sequence = work.capture(), digest(samples)
    work.reset()
    assert exact(start, work.capture())
    with sampling_trace(e, banks, None) as samples:
        work.update()
    assert exact(first, work.capture()) and sequence == digest(samples)
    work.state.save(tmp_path / "restorable.pt")
    before = work.capture(gradients=False)
    work.reset(tmp_path / "restorable.pt")
    assert exact(before, work.capture(gradients=False))
    assert source.read_bytes() == original


@pytest.mark.parametrize("family", ["spatial", "pooled"])
def test_head_probe_reproduces_parameters_gradients_scores_and_sampling(
    tmp_path, family
):
    e = score_context(tmp_path / "resources")
    e.cfg["microbatch"] = 16
    audit = GPUFeatureAudit(e)
    audit.store = {
        r: {"v": torch.ones(6, 1, 14, 14), "sem": torch.randn(6, 3, 768, 14, 14)}
        for r in ("fit", "val_select")
    }
    work = Workload(
        e,
        {},
        audit,
        {"state": state, "runner": runner},
        tmp_path / family,
        "identity-p1-" + family,
        None,
    )
    snapshots, sequences = [], []
    for _ in range(2):
        work.reset()
        with sampling_trace(e, {}, audit) as samples:
            work.update()
        sequences.append(digest(samples))
        scores = work.score()
        work.remember_scores(scores)
        snapshots.append({"state": work.capture(), "scores": scores})
    assert exact(*snapshots) and sequences[0] == sequences[1]


def test_full_probe_orchestration_with_cpu_profiler_and_fake_cuda_clock(
    tmp_path, monkeypatch
):
    # Instrumentation/checkpoint/report plumbing only, NOT a GPU timing test.
    class FakeEvent:
        def __init__(self, **kwargs):
            self.t = 0

        def record(self):
            self.t = time.perf_counter()

        def synchronize(self):
            pass

        def elapsed_time(self, end):
            return (end.t - self.t) * 1000

    actual_profile = torch.profiler.profile

    def cpu_profile(**kwargs):
        kwargs["activities"] = [torch.profiler.ProfilerActivity.CPU]
        return actual_profile(**kwargs)

    monkeypatch.setattr(torch.profiler, "profile", cpu_profile)
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    e = score_context(tmp_path / "resources")
    e.cfg["microbatch"] = 16
    audit = GPUFeatureAudit(e)
    audit.store = {
        r: {"v": torch.ones(6, 1, 14, 14), "sem": torch.randn(6, 3, 768, 14, 14)}
        for r in ("fit", "val_select")
    }
    work = Workload(
        e,
        {},
        audit,
        {"state": state, "runner": runner},
        tmp_path / "head",
        "identity-p1-pooled",
        None,
    )
    args = SimpleNamespace(
        warmup=1, steps=2, repeats=2, profile_steps=1, equivalence_steps=1, out=tmp_path
    )
    guard = SimpleNamespace(check=lambda: None, phase="test")
    result = benchmark(work, args, guard)
    assert all(r["restore_and_continuation_exact"] for r in result["repeats"])
    assert not result["profile"]["activity"]["cuda_kernels_recorded"]
    assert result["profile"]["phases"]["phase.forward.head"]["input_shape_counts"]
    assert (work.root / "trace.json").is_file()
