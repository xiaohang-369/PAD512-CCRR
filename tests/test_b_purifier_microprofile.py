"""Real networks, synthetic teachers: engineering checks, no H100 claims."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import b_purifier_microprofile as microprobe
import b_r3_gpu_state as state
import check_stage_b_short_multi_gpu as runner
from b_r3_perf_common import digest, exact
from ccrr.training.b_batches import TeacherBank
from profile_b_r3_h100 import Workload
from test_stage_b_execution import context


@pytest.fixture
def work(tmp_path):
    e = context(tmp_path / "resources", micro=16)
    banks = {"fit": TeacherBank(e)}
    st = runner.make_state(e, {"test": "original"}, False)
    st.schedules["purifier"].next_step = 2000
    st.schedules["attacker"].next_step = 4000
    st.data["next"] = 2000
    snapshot = tmp_path / "source.pt"
    st.save(snapshot)
    yield Workload(
        e,
        banks,
        None,
        {"state": state, "runner": runner},
        tmp_path / "work",
        "purifier",
        snapshot,
    )
    state.State.reset_registry()


def test_all_microbatches_keep_effective_batch_full_adversarial_path_and_sampling(work):
    snapshot_hash = digest(work.seed_file.read_bytes())
    teachers = digest({k: getattr(work.e, k).state_dict() for k in ("mfm", "reid")})
    reference = microprobe.one_step_evidence(work)
    # Baseline wrapper itself has no numerical or optimizer effect.
    work.reset()
    work.update()
    actual = work.capture()
    for key in ("models", "optimizers", "schedulers", "rng"):
        assert exact(reference[key], actual[key])
    for micro in (1, 2, 4, 8, 16):
        work.e.cfg["microbatch"] = micro
        got = microprobe.one_step_evidence(work)
        assert got["sampling_sha256"] == reference["sampling_sha256"]
        assert got["image_draws"] == 16
        assert got["pair_draws_per_backward_group"] == [16, 16, 16]
        assert len(got["unclipped_gradients"]) == 3
        assert exact(got["schedulers"], reference["schedulers"])
        assert exact(got["rng"], reference["rng"])
        for key in ("unclipped_gradients", "models", "optimizers"):
            diff = microprobe.differences(reference[key], got[key])
            assert not diff["nonfinite"]
        # No false equality assertion: STE/FP32 chunking may change gradients.
    assert digest(work.seed_file.read_bytes()) == snapshot_hash
    assert (
        digest({k: getattr(work.e, k).state_dict() for k in ("mfm", "reid")})
        == teachers
    )


def test_numerical_screen_detects_nonfinite_and_different_states():
    a = {"x": torch.tensor([1.0, 2.0])}
    assert microprobe.differences(a, a)["within_tolerance"]
    assert not microprobe.differences(a, {"x": torch.tensor([1.0, 3.0])})[
        "within_tolerance"
    ]
    assert microprobe.differences(a, {"x": torch.tensor([1.0, float("nan")])})[
        "nonfinite"
    ]
    with pytest.raises(ValueError, match="schema"):
        microprobe.differences(a, {"x": torch.ones(3)})


@pytest.mark.parametrize("micro", [4, 16])
def test_r3_step_matches_original_stage_b_production_update(work, micro):
    from ccrr.training.b_batches import train_step

    work.e.cfg["microbatch"] = micro
    work.reset()
    work.update()
    diagnostic = work.capture()
    work.reset()
    train_step(
        work.e,
        work.banks["fit"],
        2000,
        work.opts["purifier"],
        work.opts["attacker"],
        work.schedules["purifier"],
        work.schedules["attacker"],
    )
    production = work.capture()
    for key in ("models", "optimizers", "schedulers", "rng", "gradients_after_clip"):
        assert exact(diagnostic[key], production[key]), key


def test_failed_probe_restores_microbatch_and_keeps_partial_report(
    work, tmp_path, monkeypatch
):
    import profile_b_r3_h100 as probe

    def fail(*args):
        raise TimeoutError("test interruption")

    monkeypatch.setattr(probe, "benchmark", fail)
    args = SimpleNamespace(out=tmp_path / "output")
    args.out.mkdir()
    guard = SimpleNamespace(check=lambda: None, rows=[])
    with pytest.raises(TimeoutError):
        microprobe.run_microprofile(
            work.e, work.banks, work.engine, work.seed_file, args, guard
        )
    assert work.e.cfg["microbatch"] == 16
    import json

    report = json.loads((args.out / "purifier16/report.json").read_text())
    assert not report["complete"]
    assert not report["formal_training_started"]
