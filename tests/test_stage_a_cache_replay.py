"""Complete optimizer-boundary replay with a cold cache after reload."""

import importlib.util
import os
from pathlib import Path
import sys

import pytest
import torch

from ccrr.data.cache import PixelCache
from ccrr.errors import CCRRError
from ccrr.runtime.support import seed_everything
from ccrr.training import teachers
from ccrr.training.a_state import binding
from test_pixel_cache import dataset
from test_stage_a_training_loops import TinyClinical, context

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location(
        "cache_probe", SCRIPTS / "check_stage_a_cache.py"
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
finally:
    sys.path.remove(str(SCRIPTS))

DEVICE = os.environ.get("CCRR_TEST_DEVICE", "cpu")


@pytest.mark.parametrize(
    "name,start",
    [("clinical", 2), ("clinical_rgb", 2), ("clinical_gray", 2), ("clinical", 498)],
)
def test_cache_complete_checkpoint_reload_and_selection(
    tmp_path, monkeypatch, name, start
):
    data = dataset(tmp_path / "data", size=32)
    e = context(tmp_path / "formal")
    e.fit = data
    e.data["val_select"] = data
    e.cfg["budgets"]["clinical"] = start + 6
    monkeypatch.setattr(teachers, "initialize_base", lambda *a: None)
    original_update = teachers.update
    calls = 0

    def interrupt(opt):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise probe.BoundaryReached()
        return original_update(opt)

    seed_everything(e.seed)
    views = (
        [lambda b: b["x"], lambda b: teachers.gray(b["x"])]
        if name == "clinical"
        else [lambda b: teachers.gray(b["x"])]
        if name == "clinical_gray"
        else [lambda b: b["x"]]
    )
    with monkeypatch.context() as m:
        m.setattr(teachers, "update", interrupt)
        with pytest.raises(probe.BoundaryReached):
            teachers.train_classifier(
                e, TinyClinical().to(DEVICE), name, start + 6, views
            )
    path = e.output / "checkpoints" / (name + ".last.pt")
    digest = probe.sha256(path)
    raw = torch.load(path, weights_only=True, map_location="cpu")
    # Synthetic transition fixture: same initialized AdamW history in each path,
    # placed just before the real backbone-unfreeze/validation boundary.
    raw["step"] = raw["training_state"]["scheduler"]["next_step"] = start
    expected = None
    for label, cached, restart in (
        ("baseline", False, False),
        ("cache", True, False),
        ("reload", True, True),
    ):
        cfg = {**e.cfg, "pixel_cache_bytes": 1024**2 if cached else 0}
        folder = tmp_path / label
        (folder / "checkpoints").mkdir(parents=True)
        rebound = probe.private_snapshot(raw, binding(e.cfg), binding(cfg))
        torch.save(rebound, folder / "checkpoints" / path.name)
        ds = PixelCache(cfg["pixel_cache_bytes"]).wrap(data)
        if cached:
            probe.warm(ds, cfg, name, start, 6)
        mid = start + 3 if restart else start + 6
        _, actual = probe.segment(
            cfg, folder, ds, data, name, mid, model_factory=TinyClinical
        )
        if restart:
            del ds
            ds = PixelCache(cfg["pixel_cache_bytes"]).wrap(data)
            _, tail = probe.segment(
                cfg, folder, ds, data, name, start + 6, model_factory=TinyClinical
            )
            tail["losses"] = torch.cat([actual["losses"], tail["losses"]])
            tail["gradient_norms"] = actual["gradient_norms"] + tail["gradient_norms"]
            actual = tail
        actual = probe.comparable(actual, binding(cfg))
        assert actual["checkpoint"]["training_state"]["selection"]["models"] is not None
        assert (
            actual["checkpoint"]["training_state"]["scheduler"]["next_step"]
            == start + 6
        )
        if expected is None:
            expected = actual
        else:
            assert probe.exact_equal(expected, actual)
    assert probe.sha256(path) == digest
    assert raw["training_state"]["binding"] == binding(e.cfg)
    with pytest.raises(CCRRError, match="binding mismatch"):
        probe.private_snapshot(raw, "unverified", binding(e.cfg))


def test_source_comparison_rejects_training_loop_changes(tmp_path):
    import shutil

    source = Path(teachers.__file__).parents[1]
    legacy = tmp_path / "old"
    shutil.copytree(
        source, legacy / "src/ccrr", ignore=shutil.ignore_patterns("__pycache__")
    )
    cfg = context(tmp_path / "config").cfg
    digest, changes = probe.legacy_binding(legacy, cfg)
    assert digest == binding(cfg) and changes == []
    p = legacy / "src/ccrr/training/teachers.py"
    p.write_text(p.read_text() + "\n# changed\n")
    with pytest.raises(CCRRError, match="Unexpected production source differences"):
        probe.legacy_binding(legacy, cfg)
