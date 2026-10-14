"""Candidate correctness fixtures only; never H100 performance evidence."""

import itertools
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_b_r3_h100_scaling import audit as base_audit  # noqa: F401
from b_r3_cached_execution import CachedAudit, CachedPairSampler
from b_r3_h100_scaling import Head, execute, states
from b_r3_perf_common import digest, exact
from ccrr.data.pad import PairSampler


@pytest.fixture
def audit(base_audit):  # noqa: F811
    # P1 patient has separate lesions, P2 patient has repeat observations.
    # One metadata group has matches; another exercises fallback negatives.
    for ds in base_audit.e.data.values():
        for i, row in enumerate(ds.rows):
            row.update(
                lesion_id=str(i if i < 2 else i // 2),
                region="unique" if i < 2 else "shared",
            )
    base_audit.e.sampler = PairSampler(base_audit.e.fit)
    for role in base_audit.store.values():
        role["v"] = torch.rand_like(role["v"])
        role["v"][0, :, :2] = 0
    return base_audit


def test_cached_sampler_preserves_order_rng_and_fallback_with_ineligible_rows():
    rows = []
    # Interleave lesions so image order differs from row order.
    for p, lesion, eligible in [
        ("z", "b", "1"),
        ("a", "x", "1"),
        ("z", "a", "1"),
        ("a", "x", "1"),
        ("z", "b", "1"),
        ("ignored", "x", "0"),
    ]:
        rows.append(
            dict(
                patient_id=p,
                lesion_id=lesion,
                reid_observation_eligible=eligible,
                diagnostic="d",
                region="unique" if p == "z" else "shared",
            )
        )
    source = PairSampler(SimpleNamespace(rows=rows))
    cached = CachedPairSampler(source)
    for scenario, seed in itertools.product(("p1", "p2", "mixed"), range(30)):
        a, b = np.random.default_rng(seed), np.random.default_rng(seed)
        assert source.sample(a, scenario) == cached.sample(b, scenario)
        assert exact(a.bit_generator.state, b.bit_generator.state)
        assert source.fallbacks == cached.fallbacks
    assert source.fallbacks > 0
    assert "ignored" not in cached.images


def test_candidate_all_heads_sampling_gradients_scores_and_no_stale_encodings(audit):
    candidate = CachedAudit(audit)
    protected = digest(audit.store)
    specs = list(
        itertools.product(
            ("p1", "p2"), ("raw", "control", "sem"), ("spatial", "pooled")
        )
    )
    guard = SimpleNamespace(check=lambda: None)
    for spec in specs:
        old, new = Head(audit, spec), Head(candidate, spec)
        assert exact(old.initial, new.initial)
        for _ in range(2):
            assert execute([old], 1, 1, guard, verify=True) == execute(
                [new], 1, 1, guard, verify=True
            )
            assert exact(old.score(), new.score())
            assert exact(states([old]), states([new]))
    assert protected == digest(audit.store)


def test_ab_runs_both_variants_and_recovery_but_rejects_cpu_only_trace(
    audit, tmp_path, monkeypatch
):
    import time
    from b_r3_h100_ab import run_ab

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
    for name in (
        "synchronize",
        "reset_peak_memory_stats",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    args = SimpleNamespace(
        out=tmp_path, steps=1, warmup=1, repeats=2, equivalence_steps=1, profile_steps=1
    )
    with pytest.raises(RuntimeError, match="CUDA kernels absent"):
        run_ab(audit, args, SimpleNamespace(check=lambda: None, phase="test"))
    saved = json.loads((tmp_path / "ab/summary.json").read_text())
    assert not saved["complete"] and not saved["all_exact"]
    assert saved["stepwise_sampling_gradients_optimizer_exact"]
    assert saved["nine_direction_scores_exact"]
    for variant in saved["variants"].values():
        assert len(variant["repeats"]) == 2
        assert all(
            r["final_exact"] and r["restore_and_continuation_exact"]
            for r in variant["repeats"]
        )
