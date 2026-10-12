"""Compare actual clinical loops at microbatch 1/4/8; never read a live checkpoint.

Official initialization, admitted fit pixels and full-backbone backward are used.
The schedule starts at 750 ONLY inside this disposable benchmark, to avoid a
misleading head-only speed measurement. All benchmark updates are discarded.
"""

import argparse
import json
from pathlib import Path
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
from probe_stage_a_speed import exact_equal


class Finished(Exception):
    pass


def replay(cfg, data, folder, steps, start=750, model_factory=ClinicalModel):
    folder.mkdir(parents=True)
    seed_everything(cfg["seed"])
    device = torch.device(cfg["device"])
    e = SimpleNamespace(
        cfg={**cfg, "output": str(folder), "resume_interval": 1},
        output=folder,
        seed=cfg["seed"],
        device=device,
        fit=data,
        data={},
        weights=data.class_weights().to(device),
        log=lambda *a: None,
    )
    times = []
    began = None
    init, save, sample, update = (
        teachers.AProgress.__init__,
        teachers.AProgress.save,
        teachers.rng_for,
        teachers.update,
    )

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def initialize(progress, *args, **kwargs):
        init(progress, *args, **kwargs)
        require(not progress.path.exists(), "Benchmark output already exists")
        progress.scheduler.next_step = start

    def begin(*args, **kwargs):
        nonlocal began
        sync()
        began = time.perf_counter()
        return sample(*args, **kwargs)

    def finish_update(opt):
        norm = update(opt)
        sync()
        times.append(time.perf_counter() - began)
        return norm

    def stop(progress, endpoint=False):
        if progress.start == start + steps:
            save(progress, endpoint)
            raise Finished()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model = model_factory().to(device)
    with (
        patch.object(teachers.AProgress, "__init__", initialize),
        patch.object(teachers.AProgress, "save", stop),
        patch.object(teachers, "rng_for", begin),
        patch.object(teachers, "update", finish_update),
    ):
        try:
            teachers.train_classifier(
                e,
                model,
                "clinical",
                cfg["budgets"]["clinical"],
                [lambda b: b["x"], lambda b: teachers.gray(b["x"])],
            )
        except Finished:
            pass
    require(len(times) == steps, "Benchmark did not finish requested steps")
    state = torch.load(
        folder / "checkpoints/clinical.last.pt", map_location="cpu", weights_only=True
    )
    require(
        state["step"]
        == state["training_state"]["scheduler"]["next_step"]
        == start + steps,
        "Incomplete benchmark boundary",
    )
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
    return {
        "step_seconds": times,
        "seconds_per_step": statistics.median(times[1:]),
        "peak_allocated_bytes": peak,
    }, state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    require(
        torch.cuda.is_available()
        and torch.cuda.device_count() == 1
        and "H100" in torch.cuda.get_device_name(0),
        "Expose one idle H100",
    )
    require(
        cfg["device"] == "cuda"
        and cfg["batch_teacher"] == 16
        and cfg["batch_pairs"] == 16,
        "Keep effective batches at 16",
    )
    out = Path(args.out).resolve()
    require(
        not out.exists()
        and not out.is_relative_to(Path(cfg["output"]).resolve())
        and not out.is_relative_to(Path(__file__).resolve().parents[1]),
        "Use a new external benchmark directory",
    )
    out.mkdir(parents=True, mode=0o700)
    cfg = {**cfg, "pixel_cache_bytes": 8 * 1024**3, "resume_a": False}
    pool = PixelCache(cfg["pixel_cache_bytes"])
    data = pool.wrap(PADData(cfg["data_root"], "fit", 512))
    ids = set()
    for step in range(750, 756):
        ids.update(
            map(
                int,
                rng_for(cfg["seed"], "clinical", "images", step).integers(
                    len(data), size=16
                ),
            )
        )
    print(f"WARMING_CACHE: {len(ids)} images", flush=True)
    for i in sorted(ids):
        data[i]
    report = {
        "passed": False,
        "formal_training_started": False,
        "old_run_modified": False,
        "environment": environment(),
        "config_sha256": sha256(args.config),
        "script_sha256": sha256(__file__),
        "synthetic_schedule_start": 750,
        "steps_per_repeat": 6,
        "threads": torch.get_num_threads(),
        "cache": pool.stats(),
        "variants": {},
        "limitations": [
            "Clinical full-backbone short benchmark; validation, saving and cold cache excluded",
            "Exact repeatability checked within each microbatch; no legacy trajectory equivalence claim",
            "Results do not establish MFM/ReID throughput or medical capability",
        ],
    }
    for n in (1, 4, 8):
        results, reference = [], None
        for repeat in range(2):
            print(
                f"BATCH_CHECK_START: microbatch={n}, repeat={repeat + 1}/2", flush=True
            )
            r, state = replay(
                {**cfg, "microbatch": n}, data, out / f"mb{n}-repeat{repeat}", 6
            )
            if reference is None:
                reference = state
            require(
                exact_equal(reference, state),
                "Within-mode deterministic replay failed; do not start training",
            )
            results.append(r)
        seconds = statistics.median([s for r in results for s in r["step_seconds"][1:]])
        result = {
            "seconds_per_step": seconds,
            "peak_allocated_gib": max(r["peak_allocated_bytes"] for r in results)
            / 1024**3,
            "deterministic_repeat_match": True,
            "repeats": results,
        }
        report["variants"][str(n)] = result
        write_json(out / "report.json", report)
        print(
            json.dumps(
                {"microbatch": n, **{k: v for k, v in result.items() if k != "repeats"}}
            ),
            flush=True,
        )
        del state, reference
    fastest = min((4, 8), key=lambda n: report["variants"][str(n)]["seconds_per_step"])
    speedup = (
        report["variants"]["1"]["seconds_per_step"]
        / report["variants"][str(fastest)]["seconds_per_step"]
    )
    report.update(
        passed=True,
        fastest_batched_microbatch=fastest,
        speedup_vs_cached_microbatch1=speedup,
    )
    write_json(out / "report.json", report)
    print(f"BATCH_RESULT: microbatch={fastest}, speedup={speedup:.3f}x", flush=True)
    print("REPORT:", out / "report.json", flush=True)


if __name__ == "__main__":
    main()
