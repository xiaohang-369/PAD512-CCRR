"""Performance probes must retain exact updates and leave progress untouched."""

import importlib.util
import os
from pathlib import Path
import shutil

import pytest
import torch

from ccrr.config import sha256
from ccrr.runtime.support import seed_everything
from ccrr.training import teachers
from ccrr.training.optimization import optimizer, update
from test_stage_a_training_loops import Data, TinyClinical, context


spec = importlib.util.spec_from_file_location(
    "speed_probe", Path(__file__).parents[1] / "scripts/probe_stage_a_speed.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
DEVICE = os.environ.get("CCRR_TEST_DEVICE", "cpu")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_merged_check_rejects_nonfinite_without_updating(bad):
    model = torch.nn.Linear(3, 2).to(DEVICE)
    opt = optimizer([model], [1e-3])
    before = probe.cpu_copy(model.state_dict())
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    model.bias.grad[0] = bad
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        probe.merged_gradient_update(opt)
    assert probe.exact_equal(before, probe.cpu_copy(model.state_dict()))
    assert not opt.state


def test_merged_check_preserves_exact_adamw_updates():
    seed_everything(42)
    a = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Linear(8, 2)).to(DEVICE)
    b = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Linear(8, 2)).to(DEVICE)
    b.load_state_dict(a.state_dict())
    oa, ob = optimizer([a], [1e-3]), optimizer([b], [1e-3])
    for _ in range(3):
        for pa, pb in zip(a.parameters(), b.parameters()):
            pa.grad = torch.randn_like(pa)
            pb.grad = pa.grad.clone()
        assert update(oa) == probe.merged_gradient_update(ob)
        assert probe.exact_equal(a.state_dict(), b.state_dict())
        assert probe.exact_equal(oa.state_dict(), ob.state_dict())


@pytest.mark.parametrize("name", ["clinical", "clinical_rgb", "clinical_gray"])
def test_probe_replays_saved_state_without_writes(tmp_path, monkeypatch, name):
    class ProbeData(Data):
        def class_weights(self):
            return torch.arange(1, 7).float()

    e = context(tmp_path / "formal")
    e.fit = ProbeData()
    e.cfg["budgets"]["clinical"] = 8
    monkeypatch.setattr(teachers, "initialize_base", lambda *args: None)
    original_update = teachers.update
    count = 0

    def stop_after_two(opt):
        nonlocal count
        count += 1
        if count == 3:
            raise probe.ProbeFinished()
        return original_update(opt)

    seed_everything(e.seed)
    views = (
        [lambda b: b["x"], lambda b: teachers.gray(b["x"])]
        if name == "clinical"
        else [lambda b: teachers.gray(b["x"])]
        if name == "clinical_gray"
        else [lambda b: b["x"]]
    )
    monkeypatch.setattr(teachers, "update", stop_after_two)
    with pytest.raises(probe.ProbeFinished):
        teachers.train_classifier(e, TinyClinical().to(DEVICE), name, 8, views)
    monkeypatch.setattr(teachers, "update", original_update)
    source = e.output / "checkpoints" / (name + ".last.pt")
    before = {
        p.relative_to(e.output): sha256(p) for p in e.output.rglob("*") if p.is_file()
    }
    states = []
    prior_threads = torch.get_num_threads()
    try:
        for label, threads, cache, merged in (
            ("base", prior_threads, False, False),
            ("threads", 2, False, False),
            ("merged", 2, False, True),
            ("cache", 2, True, True),
        ):
            folder = tmp_path / label
            (folder / "checkpoints").mkdir(parents=True)
            dest = folder / "checkpoints" / source.name
            shutil.copyfile(source, dest)
            digest = sha256(dest)
            result, state = probe.replay(
                e.cfg,
                folder,
                e.fit,
                name,
                2,
                3,
                threads,
                cached=cache,
                merged=merged,
                device=DEVICE,
                model_factory=TinyClinical,
            )
            assert len(result["steps"]) == 3 and result["steps"][-1]["step"] == 5
            assert sha256(dest) == digest
            assert list(folder.iterdir()) == [folder / "checkpoints"]
            states.append(state)
    finally:
        torch.set_num_threads(prior_threads)
    assert all(probe.exact_equal(states[0], s) for s in states[1:])
    assert before == {
        p.relative_to(e.output): sha256(p) for p in e.output.rglob("*") if p.is_file()
    }
