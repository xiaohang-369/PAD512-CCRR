"""Isolated clinical-step replay: benchmark candidates without changing a live run.

Only this script contains candidate optimizations. The production source and its
resume binding remain untouched. Private checkpoint copies and reports belong
outside the repository. Probe optimizer updates are discarded, not formal steps.
"""

import argparse
import copy
import json
import random
import shutil
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from ccrr.config import load_config, sha256
from ccrr.data.pad import PADData, rng_for
from ccrr.errors import require
from ccrr.runtime.support import environment, seed_everything, write_json
from ccrr.teachers.adapters import ClinicalModel
from ccrr.training import teachers
from ccrr.training.a_state import binding


def merged_gradient_update(opt):
    """Same finite predicate, clipping and optimizer; fewer host synchronizations."""
    params = [p for g in opt.param_groups for p in g["params"] if p.grad is not None]
    require(bool(params), "No gradients to update")
    # Candidate is intentionally restricted to a single device and dense grads.
    require(len({p.grad.device for p in params}) == 1, "Single-device probe only")
    require(all(p.grad.layout == torch.strided for p in params), "Dense gradients only")
    if not torch.stack([torch.isfinite(p.grad).all() for p in params]).all():
        raise FloatingPointError("Nonfinite gradient; optimizer not stepped")
    norm = torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True)
    opt.step()
    return float(norm)


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def exact_equal(a, b):
    if isinstance(a, torch.Tensor):
        return (
            isinstance(b, torch.Tensor)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and torch.equal(a.cpu(), b.cpu())
        )
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(exact_equal(x, y) for x, y in zip(a, b))
    return a == b


class TimedData:
    def __init__(self, data, cache=None):
        self.data, self.cache, self.seconds = data, cache, 0.0

    def __getattr__(self, name):
        return getattr(self.data, name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        start = time.perf_counter()
        value = self.data[index] if self.cache is None else self.cache[int(index)]
        self.seconds += time.perf_counter() - start
        return value


class ProbeFinished(Exception):
    pass


def replay(
    cfg,
    folder,
    data,
    name,
    start,
    steps,
    threads,
    *,
    cached=False,
    merged=False,
    device="cuda",
    model_factory=ClinicalModel,
):
    """Run the actual classifier loop, stopping before its writes/evaluation."""
    torch.set_num_threads(threads)
    seed_everything(cfg["seed"])
    views = (
        [lambda b: b["x"], lambda b: teachers.gray(b["x"])]
        if name == "clinical"
        else [lambda b: teachers.gray(b["x"])]
        if name == "clinical_gray"
        else [lambda b: b["x"]]
    )
    cache, cache_seconds = None, 0.0
    if cached:
        # Warm-cache bound only: preload precisely the probe's sampled images.
        # Uses per-step local RNG, never consumes the training RNG streams.
        ids = set()
        for step in range(start, start + steps):
            rng = rng_for(cfg["seed"], name, "images", step)
            ids.update(map(int, rng.integers(len(data), size=cfg["batch_teacher"])))
        begin = time.perf_counter()
        cache = {i: data[i] for i in sorted(ids)}
        cache_seconds = time.perf_counter() - begin
    timed = TimedData(data, cache)
    local_cfg = {**cfg, "resume_a": True, "output": str(folder)}
    e = SimpleNamespace(
        cfg=local_cfg,
        output=folder,
        seed=cfg["seed"],
        device=torch.device(device),
        fit=timed,
        weights=data.class_weights().to(device),
        data={},
        log=lambda *args: None,
    )
    model = model_factory().to(device)
    original_rng, original_loss, original_update = (
        teachers.rng_for,
        teachers.weighted_cross_entropy,
        teachers.update,
    )
    updater = merged_gradient_update if merged else original_update
    rows, losses = [], []
    started, data_before, state = None, None, None

    def sync():
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()

    def begin_step(*args, **kwargs):
        nonlocal started, data_before
        sync()
        started, data_before = time.perf_counter(), timed.seconds
        return original_rng(*args, **kwargs)

    def record_loss(*args, **kwargs):
        result = original_loss(*args, **kwargs)
        losses.append(result.detach())
        return result

    def finish_step(opt):
        nonlocal state
        sync()
        before_update = time.perf_counter()
        norm = updater(opt)
        sync()
        now = time.perf_counter()
        rows.append(
            {
                "step": start + len(rows) + 1,
                "seconds": now - started,
                "data_read_seconds": timed.seconds - data_before,
                "guard_clip_optimizer_seconds": now - before_update,
                "gradient_norm": norm,
            }
        )
        if len(rows) == steps:
            # Snapshot only after the timed region; no live model is involved.
            np_state = np.random.get_state()
            state = cpu_copy(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "gradients": {n: p.grad for n, p in model.named_parameters()},
                    "gradient_norms": [r["gradient_norm"] for r in rows],
                    "losses": torch.stack(losses),
                    "python_rng": random.getstate(),
                    "numpy_rng": (np_state[0], np_state[1].tolist(), *np_state[2:]),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else [],
                }
            )
            raise ProbeFinished()
        return norm

    with (
        patch.object(teachers, "rng_for", begin_step),
        patch.object(teachers, "weighted_cross_entropy", record_loss),
        patch.object(teachers, "update", finish_step),
        patch.object(teachers.AProgress, "save", lambda *a, **k: None),
    ):
        try:
            teachers.train_classifier(e, model, name, cfg["budgets"]["clinical"], views)
        except ProbeFinished:
            pass
    require(
        state is not None and len(rows) == steps, "Probe did not reach stop boundary"
    )
    measured = rows[1:] if len(rows) > 1 else rows
    tensor_bytes = (
        sum(
            v.numel() * v.element_size()
            for item in cache.values()
            for v in item.values()
            if isinstance(v, torch.Tensor)
        )
        if cache is not None
        else 0
    )
    return {
        "threads": threads,
        "cached": cached,
        "merged_gradient_checks": merged,
        "steps": rows,
        "first_step_excluded": len(rows) > 1,
        "median_step_seconds": statistics.median(r["seconds"] for r in measured),
        "median_data_read_seconds": statistics.median(
            r["data_read_seconds"] for r in measured
        ),
        "median_update_seconds": statistics.median(
            r["guard_clip_optimizer_seconds"] for r in measured
        ),
        "cache_warmup_seconds": cache_seconds,
        "cache_tensor_bytes": tensor_bytes,
    }, state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument(
        "--out",
        required=True,
        help="New private directory outside source and formal output",
    )
    p.add_argument("--steps", type=int, default=6)
    args = p.parse_args()
    require(2 <= args.steps <= 10, "Probe limited to 2-10 steps")
    require(
        torch.cuda.is_available()
        and torch.cuda.device_count() == 1
        and "H100" in torch.cuda.get_device_name(0),
        "Expose one idle H100 only",
    )
    cfg = load_config(args.config)
    require(
        cfg["device"] == "cuda"
        and cfg["batch_teacher"] == 16
        and cfg["microbatch"] == 1,
        "Expected admitted A configuration",
    )
    formal = Path(cfg["output"]).resolve()
    out = Path(args.out).resolve()
    repo = Path(teachers.__file__).resolve().parents[3]
    require(
        not out.exists()
        and not out.is_relative_to(formal)
        and not out.is_relative_to(repo),
        "Use a new external probe directory",
    )
    out.mkdir(parents=True, mode=0o700)
    # AProgress is atomically replaced. Opening/copying it pins one inode even
    # while the live trainer commits a later checkpoint; never write its source.
    candidates = [
        formal / "checkpoints" / (n + ".last.pt")
        for n in ("clinical", "clinical_rgb", "clinical_gray")
    ]
    raw, name = None, None
    for path in reversed(candidates):
        if not path.is_file():
            continue
        dest = out / "snapshots" / path.name
        dest.parent.mkdir(exist_ok=True)
        with path.open("rb") as source, dest.open("xb") as target:
            shutil.copyfileobj(source, target)
        candidate = torch.load(dest, map_location="cpu", weights_only=True)
        if candidate["step"] < cfg["budgets"]["clinical"]:
            raw, name, snapshot = candidate, path.name.removesuffix(".last.pt"), dest
            break
    require(
        raw is not None, "No unfinished clinical checkpoint; do not stop the live run"
    )
    require(
        raw["format"] == "CCRR-CHECKPOINT-2"
        and raw["stage"] == name
        and raw["training_state"]["binding"] == binding(cfg),
        "Source/config/environment differs from saved progress",
    )
    require(
        raw["step"] == raw["training_state"]["scheduler"]["next_step"],
        "Checkpoint step/scheduler mismatch",
    )
    start = raw["step"]
    steps = min(args.steps, cfg["budgets"]["clinical"] - start, 100 - start % 100)
    default_threads = torch.get_num_threads()
    data = PADData(cfg["data_root"], "fit", 512)
    report = {
        "environment": environment(),
        "script_sha256": sha256(__file__),
        "config_sha256": sha256(args.config),
        "checkpoint_sha256": sha256(snapshot),
        "substage": name,
        "saved_step": start,
        "probe_steps_per_variant": steps,
        "formal_output_modified": False,
        "probe_only": True,
        "variants": {},
        "limitations": [
            "Short isolated replay; concurrent CPU and disk load affects timing",
            "Warm-cache candidate is an upper bound, not a deployed cache",
            "Timing excludes loading, checkpoint writes, validation and bootstrap",
            "All probe optimizer updates are discarded; live training continues",
        ],
    }
    reference = None
    for label, threads, cached, merged in (
        ("baseline", default_threads, False, False),
        ("threads2", 2, False, False),
        ("threads2_merged", 2, False, True),
        ("threads2_merged_cached", 2, True, True),
    ):
        folder = out / label
        (folder / "checkpoints").mkdir(parents=True)
        shutil.copyfile(snapshot, folder / "checkpoints" / (name + ".last.pt"))
        print(f"PROBE_START: {label}, {name}, saved_step={start}", flush=True)
        result, state = replay(
            cfg, folder, data, name, start, steps, threads, cached=cached, merged=merged
        )
        if reference is None:
            reference = state
        result["exact_state_match"] = exact_equal(reference, state)
        report["variants"][label] = result
        write_json(out / "report.json", report)
        print(
            json.dumps(
                {
                    "variant": label,
                    "seconds_per_step": result["median_step_seconds"],
                    "data_seconds": result["median_data_read_seconds"],
                    "update_seconds": result["median_update_seconds"],
                    "exact_state_match": result["exact_state_match"],
                }
            ),
            flush=True,
        )
        require(
            result["exact_state_match"], "Candidate changed values/state; do not deploy"
        )
        del state
    report["passed"] = True
    write_json(out / "report.json", report)
    print("PROBE_REPORT:", out / "report.json", flush=True)


if __name__ == "__main__":
    main()
