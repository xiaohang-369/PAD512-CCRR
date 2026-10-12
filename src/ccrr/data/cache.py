"""Bounded, process-local cache of verified PAD pixels; no disk artifacts or RNG."""

from collections import OrderedDict
import operator

import torch

from ccrr.errors import require


def cache_budget(config):
    value = config.get("pixel_cache_bytes", 0)
    require(
        type(value) is int and value >= 0,
        "pixel_cache_bytes must be a nonnegative integer",
    )
    return value


def signature(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def fresh(item):
    # Preserve tensor dtype, layout and independent storage on every retrieval.
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in item.items()}


class PixelCache:
    """One LRU tensor budget shared by the experiment's dataset roles.

    Budget covers retained tensor bytes, not Python objects or returned batches.
    Lazy population uses PADData's original preprocessing and digest checks.
    Source-stat changes fail closed; immutable admitted source files are required.
    This cache is for the current single-process loader, not worker sharing.
    """

    def __init__(self, max_bytes):
        self.max_bytes = cache_budget({"pixel_cache_bytes": max_bytes})
        self.entries = OrderedDict()
        self.tensor_bytes = 0
        self.hits = self.misses = self.evictions = 0

    def wrap(self, data):
        return CachedPADData(data, self) if self.max_bytes else data

    def stats(self):
        return {
            "max_tensor_bytes": self.max_bytes,
            "tensor_bytes": self.tensor_bytes,
            "entries": len(self.entries),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }


class CachedPADData:
    def __init__(self, data, cache):
        self.data, self.cache = data, cache
        # Separate dataset instances/roles, even when they use the same image.
        self._namespace = object()

    def __getattr__(self, name):
        return getattr(self.data, name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = self.data.rows[index]
        path = self.data.root / row["source_relpath"]
        before = signature(path)
        key = self._namespace, index
        c = self.cache
        if key in c.entries:
            previous, item, _ = c.entries[key]
            require(
                before == previous, "Cached source changed; stop and verify the dataset"
            )
            c.hits += 1
            c.entries.move_to_end(key)
            return fresh(item)
        c.misses += 1
        item = self.data[index]
        require(signature(path) == before, "Source changed during preprocessing")
        tensors = [v for v in item.values() if isinstance(v, torch.Tensor)]
        require(
            all(v.device.type == "cpu" and not v.requires_grad for v in tensors),
            "Pixel cache requires CPU input tensors",
        )
        size = sum(v.numel() * v.element_size() for v in tensors)
        if size <= c.max_bytes:
            while c.tensor_bytes + size > c.max_bytes:
                _, (_, _, removed) = c.entries.popitem(last=False)
                c.tensor_bytes -= removed
                c.evictions += 1
            c.entries[key] = before, item, size
            c.tensor_bytes += size
            return fresh(item)
        return item
