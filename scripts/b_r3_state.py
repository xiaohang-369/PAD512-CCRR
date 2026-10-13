"""Diagnostic-only atomic recovery and cumulative wall budget. No production imports."""

import copy
import time
import signal
import threading
from pathlib import Path

import torch

from ccrr.errors import require
from ccrr.runtime.checkpoints import save_checkpoint, load_checkpoint
from ccrr.runtime.support import write_json, json_safe

PROTOCOL = "B-R3-MULTI-v0.1"
INIT = "PUR-INIT-v1.2-rc1"


class Paused(Exception):
    pass


class Budget:
    """Elapsed execution, including cache/IO/evaluation. Downtime does not add credit."""

    def __init__(self, root, limit=14400, clock=time.monotonic):
        self.path = Path(root) / "wall_budget.json"
        self.clock, self.start, self.stop = clock, clock(), None
        self.limit = limit
        if self.path.exists():
            import json

            r = json.loads(self.path.read_text())
            require(r["limit_seconds"] == limit, "Budget changed")
            self.previous = r["used_seconds"]
        else:
            self.previous = 0.0
        self.handlers = {}
        self.lock = threading.Lock()
        self.finished = threading.Event()
        self.heartbeat = None

    def install(self):
        def pulse():
            while not self.finished.wait(1):
                if self.record() >= self.limit:
                    self.stop = "four-hour cumulative budget exhausted"

        self.heartbeat = threading.Thread(target=pulse, daemon=True)
        self.heartbeat.start()
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.handlers[sig] = signal.getsignal(sig)
            signal.signal(
                sig, lambda signum, frame: setattr(self, "stop", f"signal-{signum}")
            )

    def close(self):
        self.finished.set()
        if self.heartbeat is not None:
            self.heartbeat.join()
        for sig, handler in self.handlers.items():
            signal.signal(sig, handler)
        self.record()

    def record(self):
        with self.lock:
            used = self.previous + self.clock() - self.start
            write_json(self.path, {"limit_seconds": self.limit, "used_seconds": used})
            return used

    def check(self):
        if self.record() >= self.limit or self.stop:
            raise Paused(self.stop or "four-hour cumulative budget exhausted")


class State:
    """An optimizer-boundary transaction, with strict protocol/source/input binding.

    Each committed update is saved (not just weights). A hard crash before atomic
    rename can replay the single uncommitted transaction; never skip its updates.
    """

    def __init__(self, path, bound, modules, opts, schedules=(), sampler=None):
        self.path, self.bound = Path(path), bound
        self.modules, self.opts = modules, opts
        self.schedules = dict(schedules)
        self.sampler = sampler
        self.data = {}
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

    def save(self, destination=None):
        save_checkpoint(
            destination or self.path,
            self.modules,
            next(iter(self.opts.values())),
            self.data.get("next", 0),
            PROTOCOL,
            {"diagnostic_only": True, "initialization": INIT},
            training_state={
                "binding": self.bound,
                "optimizers": {k: o.state_dict() for k, o in self.opts.items()},
                "schedulers": {k: s.state_dict() for k, s in self.schedules.items()},
                "sampler_fallbacks": getattr(self.sampler, "fallbacks", 0),
                "data": self.data,
            },
        )


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
            score, result = evaluate()
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
        update(d["next"])
        d["next"] += 1
        state.save()
        budget.record()
        if d["next"] % 100 == 0:
            print(f"R3_HEAD_STEP: {state.path.stem} {d['next']}/{total}", flush=True)
    return copy.deepcopy(d["best"])
