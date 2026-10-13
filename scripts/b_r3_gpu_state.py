"""GPU diagnostic recovery: ordinary complete files, interval 100, no object store."""

import copy
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from ccrr.errors import require
from ccrr.runtime.checkpoints import load_checkpoint
from ccrr.runtime.support import write_json, json_safe

PROTOCOL = "B-R3-MULTI-GPU-v0.2"
INIT = "PUR-INIT-v1.2-rc1"


class Timings:
    path = None
    values = {}
    start_time = None
    events = []

    @classmethod
    def start(cls, path):
        cls.path, cls.values, cls.start_time = Path(path), {}, time.perf_counter()
        cls.events = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @classmethod
    def record(cls, key, seconds):
        v = cls.values.setdefault(key, {"calls": 0, "seconds": 0.0, "max_seconds": 0.0})
        v["calls"] += 1
        v["seconds"] += seconds
        v["max_seconds"] = max(v["max_seconds"], seconds)

    @staticmethod
    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @classmethod
    def call(cls, key, fn, *args, **kwargs):
        cls.sync()
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            cls.sync()
            cls.record(key, time.perf_counter() - start)

    @classmethod
    def gpu_call(cls, key, fn, *args):
        # No per-kernel synchronize: defer CUDA event reads until the direction ends.
        if not torch.cuda.is_available():
            return cls.call(key, fn, *args)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        result = fn(*args)
        end.record()
        cls.events.append((key, begin, end))
        return result

    @classmethod
    def collect_events(cls):
        if cls.events:
            cls.events[-1][2].synchronize()
            for key, begin, end in cls.events:
                cls.record(key + ".cuda", begin.elapsed_time(end) / 1000)
            cls.events = []

    @classmethod
    def write(cls):
        cls.collect_events()
        if cls.path is not None:
            write_json(
                cls.path,
                {
                    "protocol": PROTOCOL,
                    "timings": cls.values,
                    "session_wall_seconds": time.perf_counter() - cls.start_time,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated()
                    if torch.cuda.is_available()
                    else 0,
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved()
                    if torch.cuda.is_available()
                    else 0,
                    "note": "Nested wall intervals overlap; do not sum all categories. Current process session only.",
                    "scientific_admission_complete": False,
                },
            )


def snapshot(state):
    start = time.perf_counter()
    np_state = np.random.get_state()
    obj = {
        "format": "CCRR-CHECKPOINT-2",
        "models": {k: m.state_dict() for k, m in state.modules.items()},
        "optimizer": next(iter(state.opts.values())).state_dict(),
        "step": state.data.get("next", 0),
        "stage": PROTOCOL,
        "extra": {"diagnostic_only": True, "initialization": INIT},
        "python_rng": random.getstate(),
        "numpy_rng": (np_state[0], np_state[1].tolist(), *np_state[2:]),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "training_state": {
            "binding": state.bound,
            "optimizers": {k: o.state_dict() for k, o in state.opts.items()},
            "schedulers": {k: s.state_dict() for k, s in state.schedules.items()},
            "sampler_fallbacks": getattr(state.sampler, "fallbacks", 0),
            "data": state.data,
        },
    }
    Timings.record("checkpoint.prepare", time.perf_counter() - start)
    return obj


def atomic_save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    # Separate outstanding GPU wait from torch.save's D2H/serialization/write.
    start = time.perf_counter()
    Timings.sync()
    Timings.record("checkpoint.gpu_wait", time.perf_counter() - start)
    start = time.perf_counter()
    with tmp.open("wb") as stream:
        Timings.call("checkpoint.serialize_write", torch.save, obj, stream)
        t = time.perf_counter()
        stream.flush()
        Timings.record("checkpoint.flush", time.perf_counter() - t)
        t = time.perf_counter()
        os.fsync(stream.fileno())
        Timings.record("checkpoint.fsync", time.perf_counter() - t)
    t = time.perf_counter()
    os.replace(tmp, path)
    Timings.record("checkpoint.rename", time.perf_counter() - t)
    t = time.perf_counter()
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    Timings.record("checkpoint.directory_fsync", time.perf_counter() - t)
    Timings.record("checkpoint.total", time.perf_counter() - start)


class State:
    """An optimizer-boundary transaction, with strict protocol/source/input binding.

    Full recovery is saved every 100 transactions and at selection/phase boundaries.
    A hard crash may replay up to 100 transactions. Old R3 checkpoints are rejected.
    """

    def __init__(self, path, bound, modules, opts, schedules=(), sampler=None):
        self.path, self.bound = Path(path), bound
        self.modules, self.opts = modules, opts
        self.schedules = dict(schedules)
        self.sampler = sampler
        self.data = {}
        self.pending = None
        self.dirty = False
        self.signature = None
        self.registry[str(self.path)] = self
        self.loaded = self.path.exists()
        if self.loaded:
            raw = torch.load(self.path, map_location="cpu", weights_only=True)
            s = raw.get("training_state") or {}
            require(
                raw.get("stage") == PROTOCOL and s.get("binding") == bound,
                "R3 checkpoint protocol/binding mismatch",
            )
            require(
                set(s["optimizers"]) == set(opts)
                and set(s["schedulers"]) == set(self.schedules),
                "R3 optimizer/scheduler set mismatch",
            )
            require(
                len(raw["cuda_rng"])
                == (torch.cuda.device_count() if torch.cuda.is_available() else 0),
                "R3 CUDA topology mismatch",
            )
            first = next(iter(opts.values()))
            load_checkpoint(self.path, modules, first)  # also restores all RNGs
            for k, opt in opts.items():
                opt.load_state_dict(s["optimizers"][k])
            for k, sched in self.schedules.items():
                sched.load_state_dict(s["schedulers"][k])
            if sampler is not None:
                sampler.fallbacks = s["sampler_fallbacks"]
            self.data = s["data"]
            self.signature = self.marker()

    registry = {}
    interval = 100

    @classmethod
    def reset_registry(cls):
        cls.registry = {}

    @classmethod
    def flush_pending(cls):
        for state in list(cls.registry.values()):
            if state.dirty:
                state._commit()

    def marker(self):
        d = self.data
        counts = tuple(
            (k, s.next_step // self.interval) for k, s in self.schedules.items()
        )
        # Attacker moves twice per Purifier transaction: don't accidentally save every 50.
        if "purifier" in self.schedules:
            counts = tuple(
                (k, s.next_step // self.interval)
                for k, s in self.schedules.items()
                if k != "attacker"
            )
        return (
            d.get("phase"),
            d.get("next", 0) // self.interval,
            counts,
            tuple(d.get("observations", {})),
            tuple(d.get("control_observations", {})),
            d.get("evaluated"),
            d.get("complete"),
            "before" in d,
            "after" in d,
        )

    def save(self, destination=None):
        # Capture RNG NOW, including inside isolated readout scopes. Flushing after
        # stack unwinding must not capture the restored caller RNG instead.
        self.pending = snapshot(self)
        self.dirty = True
        marker = self.marker()
        if destination is not None:
            atomic_save(destination, self.pending)
        elif marker != self.signature:
            self._commit()
        if (
            self.data.get("complete")
            or "after" in self.data
            or self.data.get("phase") == "complete"
        ):
            self.registry.pop(str(self.path), None)

    def _commit(self):
        atomic_save(self.path, self.pending)
        self.signature = self.marker()
        self.dirty = False


def run_job(state, total, update, evaluate, interval, budget):
    """Resume an independent head including pending scoring and its best result."""
    d = state.data
    if not state.loaded:
        d.update(
            next=0,
            evaluated=0,
            best_score=-1.0,
            best=None,
            best_weights=None,
            complete=False,
        )
        state.save()
    require(
        0 <= d["next"] <= total and d["evaluated"] <= d["next"], "Head progress corrupt"
    )
    while not d["complete"]:
        budget.check()
        # Save the update BEFORE scoring. Resume can repeat scoring, never a committed update.
        if d["next"] and d["next"] % interval == 0 and d["evaluated"] != d["next"]:
            if state.dirty:
                state._commit()
            score, result = Timings.call("score." + state.path.stem, evaluate)
            if score > d["best_score"]:
                d["best_score"], d["best"] = float(score), json_safe(result)
                d["best_weights"] = {
                    k: v.detach().cpu().clone()
                    for k, v in next(iter(state.modules.values())).state_dict().items()
                }
            d["evaluated"] = d["next"]
            state.save()
            budget.check()
        if d["next"] == total:
            require(d["best"] is not None, "No evaluable head endpoint")
            d["complete"] = True
            state.save()
            break
        Timings.call("update." + state.path.stem, update, d["next"])
        d["next"] += 1
        state.save()
        budget.record()
        if d["next"] % 100 == 0:
            Timings.write()
            print(f"R3_HEAD_STEP: {state.path.stem} {d['next']}/{total}", flush=True)
    return copy.deepcopy(d["best"])
