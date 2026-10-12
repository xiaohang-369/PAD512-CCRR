"""The batch benchmark exercises full-backbone production loops, not live progress."""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from ccrr.training import teachers
from test_stage_a_training_loops import Data, TinyClinical, context

scripts = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location(
        "batch_speed", scripts / "check_stage_a_batch_speed.py"
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
finally:
    sys.path.remove(str(scripts))


@pytest.mark.parametrize("n", [1, 4, 8])
def test_full_backbone_benchmark_is_repeatable(tmp_path, monkeypatch, n):
    class Fit(Data):
        def class_weights(self):
            return torch.arange(1, 7).float()

    monkeypatch.setattr(teachers, "initialize_base", lambda *a: None)
    e = context(tmp_path / "formal")
    e.cfg.update(microbatch=n)
    e.cfg["budgets"]["clinical"] = 5000
    first, second = tmp_path / "first", tmp_path / "second"
    r1, a = probe.replay(e.cfg, Fit(), first, 3, model_factory=TinyClinical)
    r2, b = probe.replay(e.cfg, Fit(), second, 3, model_factory=TinyClinical)
    assert len(r1["step_seconds"]) == len(r2["step_seconds"]) == 3
    assert a["step"] == 753 and probe.exact_equal(a, b)
    assert not list(e.output.iterdir())
    assert len(a["optimizer"]["state"]) == 4
