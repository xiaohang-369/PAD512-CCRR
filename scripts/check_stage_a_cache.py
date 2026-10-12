"""Isolated cache-only replay with complete checkpoint and save/reload comparison.

Run from a separate candidate source export. The live checkout and formal run
are read-only. Any source-binding translation happens ONLY in disposable copies;
passing this check does not authorize or implement formal checkpoint migration.
"""

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ccrr.config import load_config, sha256
from ccrr.data.cache import PixelCache
from ccrr.data.pad import PADData, rng_for
from ccrr.errors import require
from ccrr.runtime.support import seed_everything, environment, write_json
from ccrr.teachers.adapters import ClinicalModel
from ccrr.training import teachers
from ccrr.training.a_state import binding
from probe_stage_a_speed import cpu_copy, exact_equal, TimedData


class BoundaryReached(Exception):
    pass


def legacy_binding(repo, cfg):
    old = Path(repo).resolve() / "src/ccrr"
    new = Path(teachers.__file__).resolve().parents[1]
    require(old != new, "Use a separate candidate source export")
    files = set(p.relative_to(old) for p in old.rglob("*.py")) | set(
        p.relative_to(new) for p in new.rglob("*.py")
    )
    changed = {
        str(p)
        for p in files
        if not (old / p).exists()
        or not (new / p).exists()
        or sha256(old / p) != sha256(new / p)
    }
    require(
        changed
        <= {
            "data/cache.py",
            "training/experiment.py",
            "training/a_state.py",
            "runtime/preflight.py",
        },
        "Unexpected production source differences; stop cache validation",
    )
    spec = importlib.util.spec_from_file_location(
        "legacy_a_state", old / "training/a_state.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.binding(cfg), sorted(changed)


def private_snapshot(raw, old_binding, new_binding):
    require(
        raw["format"] == "CCRR-CHECKPOINT-2"
        and raw["training_state"]["binding"] == old_binding,
        "Snapshot source/config/environment binding mismatch",
    )
    result = copy.deepcopy(raw)
    result["training_state"]["binding"] = new_binding
    return result


def segment(cfg, folder, data, validation, name, stop, *, model_factory=ClinicalModel):
    """Exit only AFTER scheduler.advance and an actual durable progress save."""
    seed_everything(cfg["seed"])
    device = cfg["device"]
    timed = TimedData(data)
    e = SimpleNamespace(
        cfg={**cfg, "output": str(folder), "resume_a": True, "resume_interval": 1},
        output=folder,
        seed=cfg["seed"],
        device=torch.device(device),
        fit=timed,
        weights=data.class_weights().to(device),
        data={"val_select": validation},
        log=lambda *a: None,
    )
    model = model_factory().to(device)
    views = (
        [lambda b: b["x"], lambda b: teachers.gray(b["x"])]
        if name == "clinical"
        else [lambda b: teachers.gray(b["x"])]
        if name == "clinical_gray"
        else [lambda b: b["x"]]
    )
    original_rng, original_update, original_save, original_loss = (
        teachers.rng_for,
        teachers.update,
        teachers.AProgress.save,
        teachers.weighted_cross_entropy,
    )
    rows, losses, gradients = [], [], None
    started = data_before = None

    def sync():
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()

    def begin(*args, **kwargs):
        nonlocal started, data_before
        sync()
        started, data_before = time.perf_counter(), timed.seconds
        return original_rng(*args, **kwargs)

    def loss(*args, **kwargs):
        value = original_loss(*args, **kwargs)
        losses.append(value.detach())
        return value

    def update(opt):
        norm = original_update(opt)
        sync()
        rows.append(
            {
                "seconds": time.perf_counter() - started,
                "data_seconds": timed.seconds - data_before,
                "gradient_norm": norm,
            }
        )
        return norm

    def save(progress, endpoint=False):
        nonlocal gradients
        original_save(progress, endpoint)
        if progress.start == stop:
            gradients = cpu_copy({n: p.grad for n, p in model.named_parameters()})
            raise BoundaryReached()

    with (
        patch.object(teachers, "rng_for", begin),
        patch.object(teachers, "update", update),
        patch.object(teachers, "weighted_cross_entropy", loss),
        patch.object(teachers.AProgress, "save", save),
    ):
        try:
            teachers.train_classifier(e, model, name, cfg["budgets"]["clinical"], views)
        except BoundaryReached:
            pass
    require(
        gradients is not None, "Replay did not reach the complete optimizer boundary"
    )
    state = torch.load(
        folder / "checkpoints" / (name + ".last.pt"),
        weights_only=True,
        map_location="cpu",
    )
    require(
        state["step"] == stop == state["training_state"]["scheduler"]["next_step"],
        "Saved scheduler mismatch",
    )
    return rows, {
        "checkpoint": state,
        "gradients": gradients,
        "losses": cpu_copy(torch.stack(losses)),
        "gradient_norms": [r["gradient_norm"] for r in rows],
    }


def comparable(state, expected_binding):
    state = copy.deepcopy(state)
    require(
        state["checkpoint"]["training_state"]["binding"] == expected_binding,
        "Unexpected candidate binding",
    )
    # The sole intentional difference is recorded separately in the report.
    state["checkpoint"]["training_state"]["binding"] = "explicit-probe-binding"
    return state


def warm(data, cfg, name, start, steps):
    ids = set()
    for step in range(start, start + steps):
        ids.update(
            map(
                int,
                rng_for(cfg["seed"], name, "images", step).integers(
                    len(data), size=cfg["batch_teacher"]
                ),
            )
        )
    then = time.perf_counter()
    for i in sorted(ids):
        data[i]
    return time.perf_counter() - then


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--legacy-repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()
    require(4 <= args.steps <= 10, "Use 4-10 isolated steps")
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
        "Expected admitted A config",
    )
    old_binding, changes = legacy_binding(args.legacy_repo, cfg)
    formal, out = Path(cfg["output"]).resolve(), Path(args.out).resolve()
    roots = (
        formal,
        Path(args.legacy_repo).resolve(),
        Path(__file__).resolve().parents[1],
    )
    require(
        not out.exists() and all(not out.is_relative_to(r) for r in roots),
        "Use a new directory outside both source trees and formal output",
    )
    out.mkdir(parents=True, mode=0o700)
    print("COPYING_SNAPSHOT", flush=True)
    raw = name = snapshot = None
    for n in ("clinical_gray", "clinical_rgb", "clinical"):
        source = formal / "checkpoints" / (n + ".last.pt")
        if not source.is_file():
            continue
        dest = out / source.name
        with source.open("rb") as a, dest.open("xb") as b:
            shutil.copyfileobj(a, b)
        candidate = torch.load(dest, weights_only=True, map_location="cpu")
        if candidate["step"] + args.steps <= cfg["budgets"]["clinical"]:
            raw, name, snapshot = candidate, n, dest
            break
    require(
        raw is not None,
        "No unfinished clinical checkpoint with enough steps; leave formal run running",
    )
    start = raw["step"]
    require(
        raw["stage"] == name
        and raw["training_state"]["scheduler"]["next_step"] == start,
        "Invalid checkpoint stage/step",
    )
    plain = PADData(cfg["data_root"], "fit", 512)
    validation = PADData(cfg["data_root"], "val_select", 512)
    report = {
        "passed": False,
        "probe_only": True,
        "formal_output_modified": False,
        "formal_resume_compatible": False,
        "environment": environment(),
        "threads": torch.get_num_threads(),
        "script_sha256": sha256(__file__),
        "config_sha256": sha256(args.config),
        "snapshot_sha256": sha256(snapshot),
        "saved_step": start,
        "substage": name,
        "source_changes": changes,
        "old_binding": old_binding,
        "variants": {},
        "limitations": [
            "Short clinical replay only; not full Stage A or medical capability validation",
            "Timing excludes checkpoint I/O, validation, initial loading and warmup",
            "Private snapshot binding translation is not a formal resume migration",
            "Baseline and candidate share unchanged training loop source and environment",
        ],
    }
    reference = None
    for label, cached, restart in (
        ("baseline", False, False),
        ("cache_only", True, False),
        ("cache_reloaded", True, True),
    ):
        local_cfg = {**cfg, "pixel_cache_bytes": 8 * 1024**3 if cached else 0}
        expected = binding(local_cfg)
        folder = out / label
        (folder / "checkpoints").mkdir(parents=True)
        torch.save(
            private_snapshot(raw, old_binding, expected),
            folder / "checkpoints" / (name + ".last.pt"),
        )
        pool = PixelCache(local_cfg["pixel_cache_bytes"])
        data = pool.wrap(plain)
        print(f"CACHE_CHECK_START: {label}, {name}, saved_step={start}", flush=True)
        warmup = warm(data, cfg, name, start, args.steps) if cached else 0
        mid = start + args.steps // 2 if restart else start + args.steps
        rows, state = segment(local_cfg, folder, data, validation, name, mid)
        if restart:
            # Discard the cache as a real process restart would; the next segment
            # constructs a fresh model/optimizer/scheduler and reloads all RNG.
            del data, pool
            pool = PixelCache(local_cfg["pixel_cache_bytes"])
            data = pool.wrap(plain)
            more, tail = segment(
                local_cfg, folder, data, validation, name, start + args.steps
            )
            tail["losses"] = torch.cat([state["losses"], tail["losses"]])
            tail["gradient_norms"] = state["gradient_norms"] + tail["gradient_norms"]
            rows += more
            state = tail
        normalized = comparable(state, expected)
        if reference is None:
            reference = normalized
        match = exact_equal(reference, normalized)
        result = {
            "complete_state_match": match,
            "binding": expected,
            "steps": len(rows),
            "seconds_per_step": statistics.median(r["seconds"] for r in rows[1:]),
            "data_seconds": statistics.median(r["data_seconds"] for r in rows[1:]),
            "warmup_seconds": warmup,
            "cache": pool.stats(),
            "cold_after_reload": restart,
        }
        report["variants"][label] = result
        write_json(out / "report.json", report)
        print(json.dumps({"variant": label, **result}), flush=True)
        require(match, "Cache changed complete replay state; do not deploy")
        del state, normalized, data, pool
    report["passed"] = True
    write_json(out / "report.json", report)
    print("CACHE_CHECK_REPORT:", out / "report.json", flush=True)


if __name__ == "__main__":
    main()
