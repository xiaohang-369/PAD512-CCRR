"""Synthetic optimizer/restart tests, not real teacher training or capability evidence."""

import random
import os
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from ccrr.errors import CCRRError
from ccrr.runtime.support import seed_everything
from ccrr.training.optimization import optimizer, TeacherSchedule, update
from ccrr.training.a_state import AProgress


def cfg():
    return {
        "seed": 20261008,
        "device": os.environ.get("CCRR_TEST_DEVICE", "cpu"),
        "profile": {"size": 512},
        "budgets": {"clinical": 800},
        "weights": {},
    }


def create(root, resume=False, classifier=False):
    model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.3), nn.Linear(4, 2)).to(
        cfg()["device"]
    )
    opt = optimizer([model], [1e-3])
    for g in opt.param_groups:
        g["head"] = not classifier
    scheduler = TeacherSchedule(opt, 800, classifier)
    e = SimpleNamespace(output=root, cfg={**cfg(), "resume_a": resume})
    return (
        model,
        opt,
        scheduler,
        AProgress(e, "synthetic", {"model": model}, opt, scheduler),
    )


def steps(m, opt, scheduler, progress, stop):
    trace = []
    for step in range(progress.start, stop):
        m.train()
        opt.zero_grad(set_to_none=True)
        x = (
            torch.rand(3, 4, device=next(m.parameters()).device)
            + random.random()
            + float(np.random.random())
        )
        loss = m(x).square().mean()
        loss.backward()
        scheduler.apply(step)
        trace.append((float(loss.detach()), [g["lr"] for g in opt.param_groups]))
        update(opt)
        scheduler.advance()
        # Equal scores retain earlier step; selection is not simply the last state.
        progress.select(float(step // 11), {"step": step})
        progress.audit["fallbacks"] += step % 2
        progress.save(endpoint=step in (499, 749))
    return trace


def equal_nested(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal_nested(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_nested(x, y)
    else:
        assert a == b


@pytest.mark.parametrize(
    "start,split,end", [(0, 5, 13), (498, 500, 503), (747, 750, 753)]
)
def test_continuous_matches_interrupt_restore_bitwise(tmp_path, start, split, end):
    seed_everything(20261008)
    m, o, s, p = create(tmp_path / "continuous", classifier=True)
    s.next_step = start
    whole = steps(m, o, s, p, end)
    expected = torch.load(p.path, weights_only=True)
    seed_everything(20261008)
    m, o, s, p = create(tmp_path / "resumed", classifier=True)
    s.next_step = start
    before = steps(m, o, s, p, split)
    # Fresh process construction consumes all RNG streams before restoration.
    seed_everything(99)
    m, o, s, p = create(tmp_path / "resumed", resume=True, classifier=True)
    after = steps(m, o, s, p, end)
    actual = torch.load(p.path, weights_only=True)
    assert whole == before + after
    equal_nested(expected, actual)
    p.finish(tmp_path / "selected.pt")
    selected = torch.load(tmp_path / "selected.pt", weights_only=True)
    assert selected["step"] == p.selection["step"]
    assert selected["optimizer"] is None
    assert selected["extra"]["budget_completed"] == end
    equal_nested(selected["models"], p.selection["models"])


def test_resume_rejects_stale_output_binding_and_step(tmp_path):
    m, o, s, p = create(tmp_path)
    steps(m, o, s, p, 1)
    with pytest.raises(CCRRError, match="--resume"):
        create(tmp_path)
    raw = torch.load(p.path, weights_only=True)
    raw["training_state"]["binding"] = "different"
    torch.save(raw, p.path)
    with pytest.raises(CCRRError, match="mismatch"):
        create(tmp_path, True)


def test_no_evaluable_selection_fails_and_tie_keeps_earlier(tmp_path):
    m, o, s, p = create(tmp_path)
    with pytest.raises(CCRRError, match="no evaluable"):
        p.finish(tmp_path / "best.pt")
    s.next_step = 1
    p.select(0.8, {"name": "first"})
    s.next_step = 2
    p.select(0.8, {"name": "second"})
    assert p.selection["step"] == 1 and p.selection["reports"]["name"] == "first"
    p.save()
    _, _, _, p = create(tmp_path, True)
    assert p.selection["step"] == 1 and p.start == 2


def test_scheduler_rejects_different_budget_and_wrong_step(tmp_path):
    _, _, s, _ = create(tmp_path)
    with pytest.raises(CCRRError):
        s.apply(1)
    state = s.state_dict()
    state["total"] = 12
    with pytest.raises(CCRRError):
        s.load_state_dict(state)


def test_output_guard_refuses_legacy_or_accidental_restart(tmp_path):
    from ccrr.training.a_state import validate_a_output

    c = {**cfg(), "output": str(tmp_path)}
    validate_a_output(c)
    with pytest.raises(CCRRError, match="No durable"):
        validate_a_output({**c, "resume_a": True})
    (tmp_path / "checkpoints").mkdir()
    torch.save({"legacy": True}, tmp_path / "checkpoints/clinical.pt")
    with pytest.raises(CCRRError, match="Existing A"):
        validate_a_output(c)
    with pytest.raises(CCRRError, match="No durable"):
        validate_a_output({**c, "resume_a": True})


def test_output_guard_rejects_missing_preceding_substage(tmp_path):
    from ccrr.training.a_state import validate_a_output

    c = {**cfg(), "output": str(tmp_path), "resume_a": True}
    (tmp_path / "checkpoints").mkdir()
    torch.save({}, tmp_path / "checkpoints/clinical_rgb.last.pt")
    with pytest.raises(CCRRError, match="preceding"):
        validate_a_output(c)


def test_endpoint_is_saved_before_committing_progress(tmp_path, monkeypatch):
    import ccrr.training.a_state as module

    m, o, s, p = create(tmp_path)
    steps(m, o, s, p, 1)
    previous = p.path.read_bytes()
    real = module.save_checkpoint

    def fail_latest(path, *args, **kwargs):
        if path == p.path:
            raise OSError("injected failure before latest commit")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(module, "save_checkpoint", fail_latest)
    s.next_step = 2
    with pytest.raises(OSError):
        p.save(endpoint=True)
    assert p.path.read_bytes() == previous
    endpoint = torch.load(p.path.parent / "endpoints/synthetic-2.pt", weights_only=True)
    assert endpoint["step"] == 2
    _, _, _, restored = create(tmp_path, True)
    assert restored.start == 1
