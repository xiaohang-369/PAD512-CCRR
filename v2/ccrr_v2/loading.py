"""Ordered CPU prefetch for ReID; sampling stays in the parent process."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from ccrr.data.pad import collate


class ReIDBatchIndices:
    """Keep the legacy sampler's full RNG consumption, including unused pairs."""

    def __init__(self, sampler, seed, steps):
        self.sampler = sampler
        self.seed = seed
        self.steps = steps

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.steps):
            positives = self.sampler.sample(rng)[:8]
            yield [i for a, b, _, _ in positives for i in (a, b)]


def cpu_collate(items):
    # Reuse the exact image, label, rectangle and metadata contract.
    return collate(items, "cpu")


def init_worker(_worker_id):
    # Decode/resize in separate processes without each spawning a CPU thread pool.
    torch.set_num_threads(1)


def cpu_batches(data, indices, *, workers=0, prefetch=2, seed=0, pin_memory=False):
    if workers < 0 or prefetch < 1:
        raise ValueError("workers must be nonnegative and prefetch must be positive")
    if workers == 0:
        # No DataLoader RNG draws or multiprocessing for the legacy control.
        for batch_indices in indices:
            yield cpu_collate([data[i] for i in batch_indices])
        return
    loader = DataLoader(
        data,
        batch_sampler=indices,
        collate_fn=cpu_collate,
        num_workers=workers,
        prefetch_factor=prefetch,
        pin_memory=pin_memory,
        multiprocessing_context="spawn",
        worker_init_fn=init_worker,
        # DataLoader worker seeds must not advance the model's torch RNG.
        generator=torch.Generator().manual_seed(seed),
    )
    # Default FIFO delivery preserves optimizer step/domain alignment even when
    # workers finish out of order. Workers never touch CUDA or generate pairs.
    yield from loader


def to_device(batch, device, *, non_blocking=False):
    return {
        key: value.to(device, non_blocking=non_blocking)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }
