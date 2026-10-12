"""Disposable B replay on frozen A teachers. Does not launch a formal B run.

Rebuild the canonical teacher bank after reload; compare both optimizers,
schedulers, models and RNG at the same microbatch. Include the full adversarial
branch. Timing excludes initial teacher extraction, checkpoint IO and full
endpoint readout/bootstrap studies (those costs are reported separately).
"""

import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import torch
import yaml

from ccrr.config import load_config, sha256
from ccrr.errors import require
from ccrr.runtime.support import seed_everything, write_json, environment
from ccrr.training.experiment import Experiment
from ccrr.training.runner import restore_dependencies
from ccrr.training.b_resources import (
    verify_teachers,
    execution_options,
    SCIENTIFIC_GAPS,
)
from ccrr.training.b_batches import TeacherBank, train_step
from ccrr.training.b_state import BProgress, binding
from ccrr.training.optimization import optimizer, TeacherSchedule
from ccrr.training.feature_audit import FeatureAudit
from ccrr.representation.purifier import SemanticAttacker
from probe_stage_a_speed import exact_equal


def replay(e, template, folder, steps, resume=False):
    if hasattr(e, "b_probe_optimizers"):
        del e.b_probe_optimizers
    folder.mkdir(parents=True, exist_ok=resume)
    e.output = folder
    e.cfg = {**e.cfg, "output": str(folder), "resume_b": resume}
    seed_everything(e.seed)
    e.purifier.load_state_dict(template["purifier"])
    e.sem_attacker.load_state_dict(template["attacker"])
    e.purifier.requires_grad_(True)
    e.sem_attacker.requires_grad_(True)
    opt = optimizer([e.purifier], [1e-4])
    adv = optimizer([e.sem_attacker], [1e-4])
    sched, advsched = TeacherSchedule(opt, 10000), TeacherSchedule(adv, 20000)
    bank = TeacherBank(e)  # always cold-rebuilt; never rely on a saved tensor cache
    progress = BProgress(e, opt, adv, sched, advsched)
    if not resume:
        sched.next_step, advsched.next_step = 1999, 3998
    metrics = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(sched.next_step, sched.next_step + steps):
        metrics.append(train_step(e, bank, step, opt, adv, sched, advsched))
    start = time.perf_counter()
    progress.save()
    checkpoint_seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 1024**3
    raw = torch.load(progress.path, weights_only=True, map_location="cpu")
    e.b_teacher_bank = bank
    e.b_probe_optimizers = (opt, adv)  # Match production audit's resident AdamW states.
    return raw, {
        "steps": metrics,
        "peak_allocated_gib": peak,
        "cache_build_seconds": bank.seconds,
        "cache_tensor_bytes": bank.bytes,
        "checkpoint_seconds": checkpoint_seconds,
    }


def audit_probe(e, count):
    # Full feature extraction, but only disposable attack updates/scoring.
    # Explicitly NOT a 5000-step independent capability evaluation.
    e.feature_audit_tag = "engineering-only"
    e.purifier.eval().requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    a = FeatureAudit(e)
    start = time.perf_counter()
    a.cache()
    cache_seconds = time.perf_counter() - start
    model = SemanticAttacker().to(e.device)
    opt = optimizer([model], [1e-4])
    times = []
    for step in range(count):
        torch.cuda.synchronize()
        start = time.perf_counter()
        a.attacker_update(model, opt, "sem", "p1", "spatial", step)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    start = time.perf_counter()
    scores = a.attack_scores(model, "sem", "p1")
    torch.cuda.synchronize()
    return {
        "feature_cache_seconds": cache_seconds,
        "cache_tensor_bytes": a.cache_bytes,
        "attacker_seconds_per_step": statistics.median(times[1:]),
        "one_p1_nine_direction_scoring_seconds": time.perf_counter() - start,
        "scoring_rows": sum(len(r["scores"]) for r in scores),
        "full_audit_completed": False,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Completed Stage A server YAML")
    p.add_argument("--out", required=True, help="New private diagnostic directory")
    p.add_argument("--microbatches", nargs="+", type=int, default=[8, 16])
    p.add_argument("--steps", type=int, default=6)
    args = p.parse_args()
    require(3 <= args.steps <= 10, "Use 3..10 disposable steps")
    require(
        args.microbatches
        and len(set(args.microbatches)) == len(args.microbatches)
        and all(n in (1, 2, 4, 8, 16) for n in args.microbatches),
        "Invalid microbatches",
    )
    # Read-only occupancy check before this process creates a CUDA context.
    import subprocess
    import os

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(gpu.startswith("GPU-") and "," not in gpu, "Select the idle GPU by UUID")
    busy = subprocess.check_output(
        ["nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    ).strip()
    require(not busy, "Selected GPU has a compute process; stop and inspect nvidia-smi")
    require(
        torch.cuda.is_available()
        and torch.cuda.device_count() == 1
        and "H100" in torch.cuda.get_device_name(0),
        "Expose one idle H100 only",
    )
    cfg = load_config(args.config)
    require(
        cfg.get("device", "").startswith("cuda"), "B host probe requires a CUDA config"
    )
    require(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"),
        "Set CUBLAS_WORKSPACE_CONFIG before starting Python",
    )
    aroot = Path(cfg["output"]).resolve()
    out = Path(args.out).resolve()
    repo = Path(__file__).resolve().parents[1]
    require(
        not out.exists()
        and all(
            root != out and root not in out.parents and out not in root.parents
            for root in (aroot, repo)
        ),
        "Diagnostic output must be new and separate from A/source",
    )
    cfg.update(
        output=str(out / "initialization"),
        resume_a=False,
        resume_b=False,
        stage_a_source={
            "output": str(aroot),
            "freeze_sha256": sha256(aroot / "teacher_freeze_record.json"),
        },
        b_cache_device="cuda",
        b_cache_bytes=4 * 1024**3,
        b_audit_cache_bytes=8 * 1024**3,
        b_diagnostic=True,
    )
    verify_teachers(cfg)
    out.mkdir(mode=0o700, parents=True)
    results = []
    for n in args.microbatches:
        print(f"B_REPLAY_START: microbatch={n}", flush=True)
        cfg["microbatch"] = n
        cfg["output"] = str(out / f"initialization-{n}")
        execution_options(cfg)
        e = Experiment(cfg.copy())
        try:
            restore_dependencies(e, "B")
            frozen_before = {
                name: {
                    k: t.detach().cpu().clone() for k, t in model.state_dict().items()
                }
                for name, model in (
                    ("mfm", e.mfm),
                    ("reid", e.reid),
                    ("clinical", e.clinical),
                )
            }
            template = {
                name: {
                    k: t.detach().cpu().clone() for k, t in model.state_dict().items()
                }
                for name, model in (
                    ("purifier", e.purifier),
                    ("attacker", e.sem_attacker),
                )
            }
            full, metric = replay(e, template, out / f"mb{n}-full", args.steps)
            del e.b_teacher_bank
            replay(e, template, out / f"mb{n}-split", 1)
            del e.b_teacher_bank
            split, _ = replay(e, template, out / f"mb{n}-split", args.steps - 1, True)
            require(
                exact_equal(full, split), "B uninterrupted/save/reload state mismatch"
            )
            del full, split
            row = {
                "microbatch": n,
                "complete_resume_state_match": True,
                "seconds_per_step": statistics.median(
                    m["seconds_per_step"] for m in metric["steps"][1:]
                ),
                **metric,
            }
            print("B_AUDIT_SPEED_START", flush=True)
            row["audit_probe"] = audit_probe(e, args.steps)
            row["frozen_teachers_unchanged"] = all(
                exact_equal(
                    frozen_before[name],
                    {k: t.detach().cpu() for k, t in model.state_dict().items()},
                )
                for name, model in (
                    ("mfm", e.mfm),
                    ("reid", e.reid),
                    ("clinical", e.clinical),
                )
            )
            require(row["frozen_teachers_unchanged"], "Frozen teachers changed")
            row["binding"] = binding(e.cfg)
            results.append(row)
            print(
                json.dumps({k: v for k, v in row.items() if k != "steps"}), flush=True
            )
        finally:
            e.allocator.close()
        del e, template, frozen_before
        gc.collect()
        torch.cuda.empty_cache()
    fastest = min(results, key=lambda row: row["seconds_per_step"])["microbatch"]
    report = {
        "engineering_passed": True,
        "formal_training_started": False,
        "scientific_admission_complete": False,
        "scientific_gaps": list(SCIENTIFIC_GAPS),
        "environment": environment(),
        "checker_sha256": sha256(__file__),
        "stage_a_config_sha256": sha256(args.config),
        "stage_a_freeze_sha256": cfg["stage_a_source"]["freeze_sha256"],
        "variants": results,
        "fastest_microbatch": fastest,
        "limitations": [
            "Short warm-cache replay is not complete B throughput",
            "Full 10-endpoint independent readouts and bootstrap not executed",
        ],
    }
    write_json(out / "report.json", report)
    cfg.update(
        output=str(out / "candidate-output"), microbatch=fastest, b_diagnostic=False
    )
    (out / "candidate-config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("B_ENGINEERING_REPORT:", out / "report.json", flush=True)
    print("FORMAL_TRAINING_STARTED: false", flush=True)


if __name__ == "__main__":
    main()
