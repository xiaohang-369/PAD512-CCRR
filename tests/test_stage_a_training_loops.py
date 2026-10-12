"""Execute the production A loops on tiny synthetic tensors and injected interruption.

Only the pretrained-model constructor is replaced with a small test network;
these tests are explicitly NOT real weight or medical capability evidence.
"""

from types import SimpleNamespace
import os
import pytest
import torch
from torch import nn
from ccrr.codec.metadata import CLASSES
from ccrr.runtime.support import seed_everything
from ccrr.training import teachers
from ccrr.tensor_ops import gray


class Data:
    rows = [{"diagnostic": c} for c in CLASSES]

    def __len__(self):
        return 6

    def __getitem__(self, i):
        x = torch.arange(3 * 8 * 8).reshape(3, 8, 8).float() / 192 + i / 12
        return {
            "x": x,
            "v": torch.ones(1, 8, 8),
            "legal": torch.ones(1),
            "rect": (0, 0, 8, 8),
            "label": i,
            "patient_index": i,
            "id": str(i),
            "patient": str(i),
            "metadata": {},
        }


class TinyClinical(nn.Module):
    architecture = "synthetic"

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.backbone = nn.Linear(1, 8)
        self.model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(8, 6))

    def forward(self, x, rects):
        return self.model.classifier(self.model.backbone(x.mean((1, 2, 3))[:, None]))


class TinyMFM(nn.Module):
    def forward(self, x, rects):
        # This synthetic fixture uses pooling, unlike the real PanDerm backbone.
        # Keep its pooling on CPU for deterministic backward, retaining autograd
        # through the device copies and running the probe/optimizer on x.device.
        f = (
            torch.nn.functional.adaptive_avg_pool2d(
                x.mean(1, keepdim=True).cpu(), (14, 14)
            )
            .to(x.device)
            .expand(-1, 768, -1, -1)
        )
        return f, torch.ones_like(f[:, :1]), None


class Stats(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, 768, 1, 1))
        self.register_buffer("std", torch.ones(1, 768, 1, 1))
        self.register_buffer("stats_ready", torch.tensor(False))


def context(path, resume=False):
    path.mkdir(parents=True, exist_ok=True)
    e = SimpleNamespace(
        output=path,
        device=os.environ.get("CCRR_TEST_DEVICE", "cpu"),
        seed=20261008,
        cfg={
            "seed": 20261008,
            "device": os.environ.get("CCRR_TEST_DEVICE", "cpu"),
            "profile": {"size": 512},
            "weights": {},
            "budgets": {"clinical": 4, "mfm_probe": 4, "reid": 4},
            "batch_teacher": 16,
            "resume_interval": 1,
            "resume_a": resume,
        },
        fit=Data(),
        data={"val_select": Data()},
        weights=torch.arange(1, 7).float(),
        mfm=TinyMFM(),
        purifier=Stats(),
        log=lambda *a: None,
        mark=lambda *a: None,
    )
    e.weights = e.weights.to(e.device)
    e.mfm = e.mfm.to(e.device)
    e.purifier = e.purifier.to(e.device)
    return e


def compare(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            compare(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            compare(x, y)
    else:
        assert a == b


def cache_fixture(data, root):
    """Synthetic tensors with file-stat sentinels; real pixel tests are separate."""
    from ccrr.data.cache import PixelCache

    data.root = root
    data.rows = [
        dict(r, source_relpath=f"synthetic-source-{i}") for i, r in enumerate(data.rows)
    ]
    for r in data.rows:
        (root / r["source_relpath"]).write_bytes(b"synthetic stat sentinel")
    return PixelCache(1024**2).wrap(data)


@pytest.mark.parametrize("kind", ["clinical", "mfm_probe"])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("micro", [1, 4, 8])
def test_actual_a_loop_resume_matches_uninterrupted(
    tmp_path, monkeypatch, kind, cached, micro
):
    monkeypatch.setattr(teachers, "initialize_base", lambda *a: None)
    # Statistics have dedicated real numerical tests; avoid repeating 2000 draws
    # in this optimizer replay test. No fabricated passing capability is returned.
    monkeypatch.setattr(teachers, "clinical_intervals", lambda r: r)
    monkeypatch.setattr(
        teachers, "paired_gray_change", lambda r: {"not_evaluated_in_loop_test": True}
    )
    original_update = teachers.update

    def run(path, resume=False):
        e = context(path, resume)
        e.cfg["microbatch"] = micro
        e.cfg["pixel_cache_bytes"] = 1024**2 if cached else 0
        if cached and path.name != "continuous":
            e.fit = cache_fixture(e.fit, path)
            e.data["val_select"] = e.fit
        if kind == "clinical":
            return teachers.train_classifier(
                e,
                TinyClinical().to(e.device),
                "clinical",
                4,
                [lambda b: b["x"], lambda b: gray(b["x"])],
            )
        return teachers.train_mfm_probe(e)

    seed_everything(20261008)
    run(tmp_path / "continuous")
    expected = torch.load(
        tmp_path / "continuous/checkpoints" / f"{kind}.last.pt", weights_only=True
    )
    calls = 0

    def interrupt(opt):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("synthetic power loss before optimizer update")
        return original_update(opt)

    seed_everything(20261008)
    monkeypatch.setattr(teachers, "update", interrupt)
    with pytest.raises(RuntimeError, match="power loss"):
        run(tmp_path / "resumed")
    partial = torch.load(
        tmp_path / "resumed/checkpoints" / f"{kind}.last.pt", weights_only=True
    )
    assert partial["step"] == 2
    monkeypatch.setattr(teachers, "update", original_update)
    seed_everything(99)
    run(tmp_path / "resumed", True)
    actual = torch.load(
        tmp_path / "resumed/checkpoints" / f"{kind}.last.pt", weights_only=True
    )
    compare(expected, actual)
    best = torch.load(
        tmp_path / "resumed/checkpoints" / f"{kind}.pt", weights_only=True
    )
    assert best["step"] == 4 and best["extra"]["budget_completed"] == 4
    # Re-running completed A substage with --resume performs no optimizer update.
    monkeypatch.setattr(
        teachers, "update", lambda opt: pytest.fail("completed budget replayed")
    )
    run(tmp_path / "resumed", True)


class TinyReID(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(1, 8), nn.Dropout(0.2))
        self.classifier = nn.Linear(8, 6)

    def forward(self, x, rects=None):
        h = self.model(x.mean((1, 2, 3))[:, None])
        return {
            "embedding": torch.nn.functional.normalize(h, dim=1),
            "h": h,
            "logits": self.classifier(h),
        }


class IdentityData(Data):
    rows = [
        {
            "img_id": str(i),
            "patient_id": str(i // 4),
            "lesion_id": str(i % 4 // 2),
            "diagnostic": CLASSES[i // 4],
            "region": "ARM",
            "reid_observation_eligible": "1",
        }
        for i in range(24)
    ]
    index = {str(i): i for i in range(24)}

    def __len__(self):
        return 24

    def __getitem__(self, i):
        b = super().__getitem__(i // 4)
        b.update(id=str(i), patient=str(i // 4))
        b["x"] = b["x"] + i % 4 / 100
        return b


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("micro", [1, 4, 8])
def test_reid_actual_loop_restores_sampling_optimizer_and_selection(
    tmp_path, monkeypatch, cached, micro
):
    import json
    from ccrr.data.pad import PairSampler

    original = teachers.update
    # Final capability has independent numerical tests; do not claim success from
    # this small optimizer-only experiment.
    monkeypatch.setattr(
        teachers,
        "reid_capability",
        lambda *a: {"p1": {"passed": False}, "not_a_capability_test": True},
    )

    def run(path, resume=False):
        e = context(path, resume)
        e.cfg["microbatch"] = micro
        ds = IdentityData()
        ds.root = path
        ds.role = "val_select"
        e.cfg["pixel_cache_bytes"] = 1024**2 if cached else 0
        if cached and path.name != "continuous":
            ds = cache_fixture(ds, path)
        e.fit = ds
        e.data = {"val_select": ds}
        e.sampler = PairSampler(ds)
        e.reid = TinyReID().to(e.device)
        folder = path / "protocol_audit_20261009/pair_protocol"
        folder.mkdir(parents=True, exist_ok=True)
        p = {"gallery": list(ds.index), "queries": []}
        for r in ds.rows:
            p["queries"].append(
                {
                    "query": r["img_id"],
                    "patient_id": r["patient_id"],
                    "positive_gallery": [
                        s["img_id"]
                        for s in ds.rows
                        if s["patient_id"] == r["patient_id"]
                        and s["lesion_id"] != r["lesion_id"]
                    ],
                    "negative_gallery": [
                        s["img_id"]
                        for s in ds.rows
                        if s["patient_id"] != r["patient_id"]
                    ],
                }
            )
        (folder / "val_select_p1.json").write_text(json.dumps(p))
        teachers.train_reid_teacher(e)

    seed_everything(20261008)
    run(tmp_path / "continuous")
    expected = torch.load(
        tmp_path / "continuous/checkpoints/reid.last.pt", weights_only=True
    )
    count = 0

    def interrupted(opt):
        nonlocal count
        count += 1
        if count == 3:
            raise RuntimeError("power loss")
        return original(opt)

    monkeypatch.setattr(teachers, "update", interrupted)
    seed_everything(20261008)
    with pytest.raises(RuntimeError):
        run(tmp_path / "resumed")
    monkeypatch.setattr(teachers, "update", original)
    seed_everything(7)
    run(tmp_path / "resumed", True)
    actual = torch.load(
        tmp_path / "resumed/checkpoints/reid.last.pt", weights_only=True
    )
    compare(expected, actual)
    for step in range(1, 5):
        assert (tmp_path / f"continuous/sampling/reid-{step}.json").read_bytes() == (
            tmp_path / f"resumed/sampling/reid-{step}.json"
        ).read_bytes()
