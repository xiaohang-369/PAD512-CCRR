"""Real spawn workers and image preprocessing; synthetic data, never patients."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ccrr.data.pad import PADData, PairSampler, collate, preprocess
from ccrr_v2.loading import ReIDBatchIndices, cpu_batches, to_device


def image_dataset(root):
    # Exercise the real PADData.__getitem__, including preprocessing SHA checks,
    # without needing any private cohort or rewriting the frozen role protocol.
    data = object.__new__(PADData)
    data.root, data.size, data.verify = Path(root), 512, True
    data.rows, data.inventory, data.metadata, data.pid = [], {}, {}, {}
    for patient in range(4):
        pid = f"p{patient}"
        data.pid[pid] = patient
        for lesion in range(2):
            for view in range(2):
                name = f"{patient}-{lesion}-{view}.png"
                pixels = np.random.default_rng(len(data.rows)).integers(
                    0, 256, (24 + patient, 31 + view, 3), dtype=np.uint8
                )
                Image.fromarray(pixels).save(data.root / name)
                data.rows.append(
                    {
                        "img_id": name,
                        "patient_id": pid,
                        "lesion_id": f"{pid}-{lesion}",
                        "source_relpath": name,
                        "diagnostic": "ACK",
                        "region": "arm",
                        "reid_observation_eligible": "1",
                    }
                )
                data.inventory[name] = {
                    "rgb512_sha256": preprocess(data.root / name)[-1]
                }
                data.metadata[name] = {"synthetic": True}
    data.patients = sorted(data.pid)
    return data


def legacy_batches(data, seed, steps):
    sampler, rng = PairSampler(data), np.random.default_rng(seed)
    for _ in range(steps):
        pairs = sampler.sample(rng)[:8]
        indices = [i for a, b, _, _ in pairs for i in (a, b)]
        yield collate([data[i] for i in indices], "cpu")


def assert_batch_equal(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], torch.Tensor):
            assert torch.equal(a[key], b[key]), key
        else:
            assert a[key] == b[key], key


def update(model, batch):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    logits = model(batch["x"].mean((2, 3)))
    loss = torch.nn.functional.cross_entropy(logits, batch["patient_index"])
    loss.backward()
    optimizer.step()
    return loss.detach()


@pytest.mark.parametrize("workers", [0, 2])
def test_order_pixels_rng_and_optimizer_updates(tmp_path, workers):
    torch.set_num_threads(1)
    data = image_dataset(tmp_path)
    steps = 6  # Beyond the initial 2 workers x 2 batches prefetch window.
    indices = ReIDBatchIndices(PairSampler(data), 20261019, steps)
    torch.manual_seed(123)
    a = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(3, 4))
    b = deepcopy(a)
    actual = cpu_batches(data, indices, workers=workers, prefetch=2, seed=31)
    for expected in legacy_batches(data, 20261019, steps):
        before = torch.get_rng_state().clone()
        batch = next(actual)
        assert torch.equal(torch.get_rng_state(), before)
        assert_batch_equal(expected, batch)
        loss_a = update(a, expected)
        after_update = torch.get_rng_state().clone()
        torch.set_rng_state(before)
        loss_b = update(b, to_device(batch, "cpu", non_blocking=workers > 0))
        assert torch.equal(loss_a, loss_b)
        assert torch.equal(torch.get_rng_state(), after_update)
        assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
    with pytest.raises(StopIteration):
        next(actual)


def test_sampling_preserves_full_rng_and_fallbacks(tmp_path):
    data = image_dataset(tmp_path)
    # Force diagnosis/region-matched negatives to use the fallback path too.
    for row in data.rows:
        row["region"] = row["patient_id"]
    expected_sampler, actual_sampler = PairSampler(data), PairSampler(data)
    rng = np.random.default_rng(42)
    expected = []
    for _ in range(12):
        pairs = expected_sampler.sample(rng)[:8]
        expected.append([i for a, b, _, _ in pairs for i in (a, b)])
    assert list(ReIDBatchIndices(actual_sampler, 42, 12)) == expected
    assert actual_sampler.fallbacks == expected_sampler.fallbacks == 48


def test_worker_propagates_pixel_integrity_failure(tmp_path):
    data = image_dataset(tmp_path)
    data.inventory[data.rows[0]["img_id"]]["rgb512_sha256"] = "invalid"
    batches = cpu_batches(data, [[0]], workers=1, prefetch=1)
    with pytest.raises(Exception, match="Preprocessing hash mismatch"):
        next(batches)
    batches.close()


def test_loader_cli_rejects_unsupported_stage_before_resources():
    from ccrr_v2.cli import main

    with pytest.raises(ValueError, match="only to ReID"):
        main(["probe", "--resources", "/nonexistent", "--loader-workers", "2"])
    with pytest.raises(ValueError, match="Invalid loader"):
        main(["reid", "--resources", "/nonexistent", "--prefetch-batches", "0"])


def test_reid_training_integration_same_checkpoint(tmp_path, monkeypatch):
    from ccrr_v2 import reid, runtime

    images = tmp_path / "images"
    images.mkdir()
    data = image_dataset(images)

    class TinyReID(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(3, 8)
            self.classifier = torch.nn.Linear(8, 4)

        def forward(self, x, rects):
            if x.shape[1] == 1:
                x = x.expand(-1, 3, -1, -1)
            z = self.backbone(x.mean((2, 3)))
            return {
                "embedding": torch.nn.functional.normalize(z, dim=1),
                "logits": self.classifier(z),
            }

    monkeypatch.setattr(reid, "PADData", lambda *args: data)
    monkeypatch.setattr(reid, "initialize", lambda *args: TinyReID())
    monkeypatch.setattr(
        reid,
        "evaluate",
        lambda *args, **kw: {
            p: {"reports": [{"cosine": {"auroc": 0.8}}] * 3, "passed": True}
            for p in ("p1", "p2")
        },
    )
    cfg = {"seed": 42, "reid_lr": 0.0001, "reid_steps": 3}
    checkpoints = []
    for workers in (0, 2):
        out = tmp_path / str(workers)
        out.mkdir()
        reid.train({"data_root": str(images)}, cfg, out, "cpu", loader_workers=workers)
        checkpoints.append(torch.load(out / "reid.pt", weights_only=True))
        assert runtime.read(out / "status.json")["state"] == "completed"
        assert (out / "training.jsonl").is_file()
        assert (out / "validation.jsonl").is_file()
    assert checkpoints[0]["step"] == checkpoints[1]["step"] == 3
    assert all(
        torch.equal(value, checkpoints[1]["model"][key])
        for key, value in checkpoints[0]["model"].items()
    )
