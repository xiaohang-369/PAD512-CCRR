"""Registered initialization and restart checks, not medical capability evidence."""

import copy
import json

import pytest
import torch

from ccrr.errors import CCRRError
from ccrr.training.b_initialization import (
    CANDIDATE,
    after_frozen_a_restore,
    record_start,
    initialize_control,
    image_rng_stage,
)
from ccrr.training.b_control import ReconstructionControl
from ccrr.training.b_state import BProgress, binding
from ccrr.training.b_batches import TeacherBank, train_step
from test_stage_b_execution import context, setup


def test_scaling_only_two_weights_and_matched_control(tmp_path):
    e = context(tmp_path)
    e.cfg["b_initialization"] = CANDIDATE
    before = copy.deepcopy(e.purifier.state_dict())
    after_frozen_a_restore(e)
    for key, value in e.purifier.state_dict().items():
        expected = (
            before[key] * 0.1
            if key in ("encoder.end.weight", "decoder.end.weight")
            else before[key]
        )
        assert torch.equal(value, expected), key
    initial = copy.deepcopy(e.purifier.state_dict())
    record_start(e)
    with torch.no_grad():
        e.purifier.encoder.end.weight.add_(10)
    control = ReconstructionControl(e.purifier.mean, e.purifier.std)
    initialize_control(e, control)
    for part in ("encoder", "decoder"):
        for key, value in getattr(control, part).state_dict().items():
            assert torch.equal(value, initial[part + "." + key])
    with pytest.raises(CCRRError, match="repeated"):
        after_frozen_a_restore(e)
    assert image_rng_stage(e.cfg) == "B"
    assert image_rng_stage({}) == "B-control"


def test_resume_restores_optimizer_and_does_not_rescale(tmp_path):
    e = context(tmp_path, micro=16)
    e.cfg["b_initialization"] = CANDIDATE
    after_frozen_a_restore(e)
    record_start(e)
    bank = TeacherBank(e)
    opt, adv, sched, advsched = setup(e)
    progress = BProgress(e, opt, adv, sched, advsched)
    train_step(e, bank, 0, opt, adv, sched, advsched)
    progress.save()
    expected = copy.deepcopy(e.purifier.state_dict())
    initial_file = (tmp_path / "b_initialization.pt").read_bytes()
    re = context(tmp_path, micro=16, resume=True)
    re.cfg["b_initialization"] = CANDIDATE
    a_weights = copy.deepcopy(re.purifier.state_dict())
    after_frozen_a_restore(re)
    assert all(
        torch.equal(v, a_weights[k]) for k, v in re.purifier.state_dict().items()
    )
    record_start(re)
    o2, a2, s2, as2 = setup(re)
    BProgress(re, o2, a2, s2, as2)
    assert all(torch.equal(v, expected[k]) for k, v in re.purifier.state_dict().items())
    assert s2.next_step == 1 and as2.next_step == 2
    assert o2.state_dict()["state"]
    control = ReconstructionControl(re.purifier.mean, re.purifier.std)
    initialize_control(re, control)
    saved = torch.load(tmp_path / "b_initialization.pt", weights_only=True)["models"][
        "purifier"
    ]
    assert torch.equal(control.encoder.end.weight, saved["encoder.end.weight"])
    assert (tmp_path / "b_initialization.pt").read_bytes() == initial_file
    record = json.loads((tmp_path / "b_initialization.json").read_text())
    record["sha256"] = "bad"
    (tmp_path / "b_initialization.json").write_text(json.dumps(record))
    with pytest.raises(CCRRError, match="changed"):
        record_start(re)


def test_legacy_opt_in_and_binding(tmp_path):
    e = context(tmp_path)
    initial = copy.deepcopy(e.purifier.state_dict())
    old_binding = binding(e.cfg)
    after_frozen_a_restore(e)
    record_start(e)
    assert all(torch.equal(v, initial[k]) for k, v in e.purifier.state_dict().items())
    assert not (tmp_path / "b_initialization.pt").exists()
    e.cfg["b_initialization"] = CANDIDATE
    assert binding(e.cfg) != old_binding
    with pytest.raises(CCRRError, match="Unknown"):
        after_frozen_a_restore(type("E", (), {"cfg": {"b_initialization": "typo"}})())
