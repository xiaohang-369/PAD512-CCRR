import copy
import importlib.util
from pathlib import Path

import pytest
import torch
from ccrr.errors import CCRRError

from ccrr.tensor_ops import ste_round, masked_mean
from ccrr.training.b_control import ReconstructionControl

spec = importlib.util.spec_from_file_location(
    "reconstruction_probe",
    Path(__file__).parents[1] / "scripts/probe_stage_b_reconstruction.py",
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def small_model():
    torch.manual_seed(3)
    return ReconstructionControl(torch.zeros(1, 768, 1, 1), torch.ones(1, 768, 1, 1))


def test_normalized_quantizer_retains_tanh_derivative():
    z = torch.tensor([-9.0, -2.0, -0.25, 0.0, 0.25, 2.0, 9.0], requires_grad=True)
    q = ste_round(7 * z.tanh())
    (q / 7).sum().backward()
    torch.testing.assert_close(z.grad, 1 - z.detach().tanh().square())
    assert torch.equal(q.detach(), q.detach().round())
    assert z.grad[0] < 1e-6 and z.grad[-1] < 1e-6


def test_initialization_factorial_only_changes_intended_terminal_weights():
    base = small_model()
    saved = copy.deepcopy(base.state_dict())
    for name, (es, ds) in probe.ARMS.items():
        model = probe.make_arm(base, name)
        for key, value in model.state_dict().items():
            scale = (
                es
                if key == "encoder.end.weight"
                else ds
                if key == "decoder.end.weight"
                else 1
            )
            torch.testing.assert_close(value, saved[key] * scale)
    for key, value in base.state_dict().items():
        assert torch.equal(value, saved[key])


def test_measure_matches_production_and_does_not_update_or_accumulate_gradients():
    model = small_model()
    saved = copy.deepcopy(model.state_dict())
    f = torch.randn(2, 768, 4, 4)
    v = torch.ones(2, 1, 4, 4)
    v[..., 0, :] = 0
    v[..., 1, :] = 0.25
    actual = model(f, v)
    expected = float(
        masked_mean((actual["all"] - actual["standardized"]).square(), v).detach()
    )
    report = probe.measure(model, f, v)
    assert report["reconstruction_mse"] == pytest.approx(expected)
    assert sum(report["code_histogram_levels_minus7_to7"]) == pytest.approx(1, abs=1e-6)
    assert report["encoder_terminal_grad_l2"] > 0
    assert report["decoder_terminal_grad_l2"] > 0
    assert all(p.grad is None for p in model.parameters())
    for key, value in model.state_dict().items():
        assert torch.equal(value, saved[key])
    # Padding cannot influence the reported entropy, loss, or tanh occupancy.
    f[..., 0, :] = 1e6
    again = probe.measure(model, f, v)
    assert again == report


def test_fit_cohort_patient_disjoint_and_row_order_independent():
    rows = [
        {"patient_id": str(p), "img_id": f"{p}-{i}"}
        for p in range(90)
        for i in range(2)
    ]
    train, holdout = probe.cohort(rows)
    assert len(train) == 64 and len(holdout) == 16
    assert not {rows[i]["patient_id"] for i in train} & {
        rows[i]["patient_id"] for i in holdout
    }
    other = list(reversed(rows))
    t2, h2 = probe.cohort(other)
    assert [rows[i]["img_id"] for i in train + holdout] == [
        other[i]["img_id"] for i in t2 + h2
    ]
    with pytest.raises(CCRRError):
        probe.cohort(rows[:10])


def test_pilot_same_batches_and_original_schedule_for_all_arms(monkeypatch):
    batches, totals, rows = [], [], []

    def update(model, bank, ids, micro, opt, schedule):
        batches.append(ids)
        totals.append(schedule.total)
        schedule.advance()

    monkeypatch.setattr(probe, "control_update", update)
    monkeypatch.setattr(probe, "evaluate", lambda *args: {})
    base = small_model()
    saved = copy.deepcopy(base.state_dict())
    probe.pilot(
        base, None, list(range(64)), list(range(64, 80)), 1000, 20261008, rows.append
    )
    assert len(batches) == 4000 and set(totals) == {10000}
    assert batches[:1000] == batches[1000:2000] == batches[2000:3000] == batches[3000:]
    assert len(rows) == 16
    for key, value in base.state_dict().items():
        assert torch.equal(value, saved[key])
