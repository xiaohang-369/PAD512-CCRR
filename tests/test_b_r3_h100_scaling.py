"""CPU correctness fixtures, never evidence of H100 throughput/concurrency."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from b_r3_gpu_features import GPUFeatureAudit
from b_r3_h100_scaling import Head, execute, snapshot, restore, states, rng_state
from b_r3_perf_common import exact, digest
from test_b_r3_gpu import score_context


@pytest.fixture
def audit(tmp_path):
    e = score_context(tmp_path / "resources")
    e.cfg["microbatch"] = 16
    audit = GPUFeatureAudit(e)
    audit.store = {
        r: {
            "v": torch.ones(6, 1, 14, 14),
            "raw": torch.randn(6, 2, 768, 14, 14),
            "control": torch.randn(6, 2, 768, 14, 14),
            "sem": torch.randn(6, 3, 768, 14, 14),
        }
        for r in ("fit", "val_select")
    }
    return audit


@pytest.mark.parametrize("family", ["spatial", "pooled"])
def test_batch16_matches_engine_and_larger_batches_are_actual_forward_sizes(
    audit, family
):
    h = Head(audit, ("p1", "sem", family))
    initial = digest(h.initial)
    for _ in range(2):
        h.update()
    expected = h.capture(gradients=True)
    h.restore()
    for _ in range(2):
        h.batch_update(16)
    assert exact(expected, h.capture(gradients=True))
    for size in (32, 64, 128, 256):
        h.restore()
        sizes = []
        hook = h.model.register_forward_pre_hook(
            lambda model, args: sizes.append(args[0].shape[0])
        )
        h.batch_update(size)
        hook.remove()
        assert sizes == [size] and h.next == 1
    assert initial == digest(h.initial)


def test_threaded_heads_keep_individual_sampling_gradients_optimizer_and_rng(audit):
    specs = [
        ("p1", "raw", "spatial"),
        ("p1", "sem", "pooled"),
        ("p2", "control", "spatial"),
        ("p2", "sem", "pooled"),
    ]
    before = rng_state()
    heads = [Head(audit, s) for s in specs]
    assert exact(before, rng_state())
    assert len({id(h.e.sampler) for h in heads}) == len(heads)
    assert len({id(h.audit.pair_tables) for h in heads}) == len(heads)
    assert all(h.audit.store is audit.store for h in heads)
    guard = SimpleNamespace(check=lambda: None)
    reference = execute(heads, 1, 2, guard, verify=True)
    expected = states(heads)
    original_store = digest(audit.store)
    for lanes in (2, 4):
        for h in heads:
            h.restore()
        assert execute(heads, lanes, 2, guard, verify=True) == reference
        assert exact(states(heads), expected)
    assert original_store == digest(audit.store)
    assert exact(before, rng_state())


def test_full_group_checkpoint_restores_best_scores_progress_and_continuation(
    audit, tmp_path
):
    heads = [
        Head(audit, s) for s in [("p1", "sem", "spatial"), ("p2", "raw", "pooled")]
    ]
    guard = SimpleNamespace(check=lambda: None)
    execute(heads, 2, 2, guard)
    for h in heads:
        h.score()
    saved = snapshot(heads)
    path = tmp_path / "probe.pt"
    torch.save(saved, path)
    execute(heads, 2, 1, guard)
    expected = states(heads)
    restore(heads, torch.load(path, weights_only=True))
    assert exact(saved, snapshot(heads))
    execute(heads, 1, 1, guard)
    assert exact(expected, states(heads))
    wrong = copy.deepcopy(saved)
    wrong["heads"].pop(heads[0].name)
    with pytest.raises(RuntimeError, match="Wrong head set"):
        restore(heads, wrong)


def test_global_rng_use_is_rejected(audit):
    h = Head(audit, ("p1", "sem", "pooled"))
    original = h.update

    def impure():
        torch.rand(1)
        original()

    h.update = impure
    with pytest.raises(RuntimeError, match="shared global RNG"):
        execute([h], 1, 1, SimpleNamespace(check=lambda: None))


def test_bounded_suite_records_all_cases_before_rejecting_absent_cuda_trace(
    audit, tmp_path, monkeypatch
):
    # Real CPU updates/profiler; fake CUDA clocks only for exercising orchestration.
    # This MUST reject the missing CUDA trace and cannot count as H100 evidence.
    import time
    from b_r3_h100_scaling import run_scaling

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
    args = SimpleNamespace(
        out=tmp_path, steps=1, warmup=1, repeats=2, equivalence_steps=1, profile_steps=1
    )
    guard = SimpleNamespace(check=lambda: None, phase="test")
    with pytest.raises(RuntimeError, match="No CUDA kernel activity"):
        run_scaling(audit, args, guard)
    import json

    saved = json.loads((tmp_path / "scaling/summary.json").read_text())
    assert not saved["complete"]
    assert len(saved["batch_sweep"]) == 8
    assert all(len(row["repeats"]) == 2 for row in saved["batch_sweep"].values())
    for lanes in ("1", "2", "4"):
        row = saved["parallel"][lanes]
        assert row["eligible_for_performance_comparison"]
        assert all(
            r["heads"] == 12 and r["total_pair_updates"] == 192 for r in row["repeats"]
        )
        assert all(r["restore_and_continuation_exact"] for r in row["repeats"])
