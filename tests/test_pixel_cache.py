"""Cache exact frozen preprocessing, independent storage, bounds and RNG behavior."""

import random
import numpy as np
import pytest
import torch
from PIL import Image

from ccrr.codec.metadata import CLASSES
from ccrr.data.cache import PixelCache, cache_budget
from ccrr.data.pad import PADData, preprocess, collate
from ccrr.errors import CCRRError
from ccrr.training.a_state import binding
from test_stage_a_training_loops import compare, context


def dataset(root, size=512):
    root.mkdir(parents=True, exist_ok=True)
    d = PADData.__new__(PADData)
    d.root, d.role, d.size, d.verify = root, "fit", size, True
    d.rows, d.metadata, d.inventory = [], {}, {}
    d.patients = [str(i) for i in range(6)]
    d.pid = {p: i for i, p in enumerate(d.patients)}
    for i, cls in enumerate(CLASSES):
        path = root / f"{i}.png"
        # Varied shapes and RGB pixels, fully synthetic and random-stream-free.
        a = (np.arange((33 + i) * 47 * 3).reshape(33 + i, 47, 3) % 256).astype("uint8")
        Image.fromarray(a).save(path)
        d.rows.append(
            {
                "source_relpath": path.name,
                "img_id": str(i),
                "diagnostic": cls,
                "patient_id": str(i),
            }
        )
        d.metadata[str(i)] = {"synthetic": str(i)}
        d.inventory[str(i)] = {"rgb512_sha256": preprocess(path, size)[-1]}
    return d


def test_cache_preserves_pixels_layout_rng_and_independent_storage(tmp_path):
    d = dataset(tmp_path)
    pool = PixelCache(32 * 1024**2)
    c = pool.wrap(d)
    rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    for i in range(len(d)):
        raw, miss, hit = d[i], c[i], c[i]
        compare(raw, miss)
        compare(raw, hit)
        for k in ("x", "v", "legal"):
            assert raw[k].stride() == hit[k].stride()
            hit[k].zero_()
            miss[k].fill_(7)
        compare(raw, c[i])
        compare(collate([raw]), collate([c[i]]))
    assert rng[0] == random.getstate()
    np.testing.assert_equal(rng[1], np.random.get_state())
    assert torch.equal(rng[2], torch.get_rng_state())
    assert pool.hits == 18 and pool.misses == 6
    compare(c[-1], d[5])
    with pytest.raises(IndexError):
        c[-7]


def test_cache_is_bounded_shared_and_evicts(tmp_path):
    d = dataset(tmp_path)
    item = d[0]
    size = sum(
        v.numel() * v.element_size()
        for v in item.values()
        if isinstance(v, torch.Tensor)
    )
    pool = PixelCache(size)
    a, b = pool.wrap(d), pool.wrap(d)
    a[0]
    b[0]
    assert pool.tensor_bytes == size and pool.evictions == 1
    a[0]
    assert pool.misses == 3 and pool.evictions == 2
    small = PixelCache(size - 1)
    compare(small.wrap(d)[0], item)
    assert not small.entries
    assert PixelCache(0).wrap(d) is d


def test_cache_never_hides_changed_source_or_bad_digest(tmp_path):
    d = dataset(tmp_path)
    c = PixelCache(32 * 1024**2).wrap(d)
    c[0]
    path = d.root / d.rows[0]["source_relpath"]
    replacement = path.with_suffix(".new")
    replacement.write_bytes(path.read_bytes())
    replacement.replace(path)
    with pytest.raises(CCRRError, match="Cached source changed"):
        c[0]
    d.inventory["1"]["rgb512_sha256"] = "0" * 64
    with pytest.raises(CCRRError, match="Preprocessing hash mismatch"):
        c[1]
    assert len(c.cache.entries) == 1


@pytest.mark.parametrize("bad", [-1, True, "8192", 1.5, None])
def test_invalid_budget_rejected(bad):
    with pytest.raises(CCRRError, match="nonnegative integer"):
        cache_budget({"pixel_cache_bytes": bad})


def test_cache_setting_is_bound_to_checkpoint(tmp_path):
    cfg = context(tmp_path).cfg
    assert binding(cfg) != binding({**cfg, "pixel_cache_bytes": 8 * 1024**3})
