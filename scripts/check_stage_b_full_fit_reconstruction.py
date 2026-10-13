"""B-R2-FULLFIT-v1: paired 10k reconstruction controls, never formal B.

All fit images update parameters; val_tune is a development evaluation role.
The original 16 fit holdout patients are now TRAINING patients, not holdout.
Final step 10000 is the fixed comparison; no best-endpoint selection/resume.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from probe_stage_b_reconstruction import make_arm, measure, cohort
from ccrr.config import load_config, sha256
from ccrr.data.pad import PADData, collate, rng_for
from ccrr.errors import require
from ccrr.runtime.support import environment, seed_everything, write_json, log_json
from ccrr.teachers.adapters import PanDerm
from ccrr.tensor_ops import gray
from ccrr.training.b_control import (
    ReconstructionControl,
    control_update,
    CONTROL_PROTOCOL,
)
from ccrr.training.b_resources import verify_teachers
from ccrr.training.optimization import optimizer, TeacherSchedule

PROTOCOL = "B-R2-FULLFIT-v1"
SEED = 20261008
STEPS = 10000
POINTS = (0, 100, 500, 1000, 2000, 5000, 10000)
ARMS = ("original", "both_x01")


def per_image_mse(prediction, target, v):
    return ((prediction - target).square() * v).sum((1, 2, 3)) / (
        target.shape[1] * v.sum((1, 2, 3))
    )


def standardized(features, v, mean, std):
    return (features - mean) / std.clamp_min(1e-3) * (v > 0)


class Cache:
    def __init__(self, storage):
        self.storage = storage
        self.count = len(storage["v"])

    def get(self, ids):
        return {k: v[ids] for k, v in self.storage.items()}


@torch.no_grad()
def cache_features(ds, teacher, device, role):
    require(len(ds) * (2 * 768 * 196 + 196) * 4 <= 4 * 1024**3, "Cache budget")
    storage = {}
    for start in range(0, len(ds), 16):
        ids = list(range(start, min(start + 16, len(ds))))
        batch = collate([ds[i] for i in ids], device)
        for domain, x in enumerate((batch["x"], gray(batch["x"]))):
            f, v, _ = teacher(x, batch["rect"])
            require(f.shape[1:] == (768, 14, 14), "Feature geometry changed")
            require(
                f.dtype == torch.float32 and torch.isfinite(f).all(), "Invalid features"
            )
            require(
                torch.isfinite(v).all() and (v.sum((1, 2, 3)) > 0).all(),
                "Invalid coverage",
            )
            for key, value in ((f"f{domain}", f), ("v", v)):
                if key == "v" and domain == 1:
                    require(
                        torch.equal(storage[key][ids], value),
                        "Domain coverage mismatch",
                    )
                    continue
                if key not in storage:
                    storage[key] = value.new_empty((len(ds), *value.shape[1:]))
                storage[key][ids] = value
        if start == 0 or (start + len(ids)) % 320 == 0 or ids[-1] == len(ds) - 1:
            print(f"FULLFIT_CACHE: {role} {ids[-1] + 1}/{len(ds)}", flush=True)
    return Cache(storage)


@torch.no_grad()
def fit_templates(bank, mean, std):
    """Per-domain best fixed spatial predictor under the native equal-image MSE.

    Fit only. FP64 weighted accumulation, zero at never-supported fit sites.
    More generous than a single template shared by RGB and gray.
    """
    numerator = {}
    denominator = None
    for start in range(0, bank.count, 16):
        t = bank.get(list(range(start, min(start + 16, bank.count))))
        w = t["v"].double() / t["v"].double().sum((2, 3), keepdim=True)
        denominator = w.sum(0) if denominator is None else denominator + w.sum(0)
        for d in ("f0", "f1"):
            y = standardized(t[d], t["v"], mean, std).double()
            part = (y * w).sum(0)
            numerator[d] = part if d not in numerator else numerator[d] + part
    return {
        d: (n / denominator.clamp_min(1e-30)).float()[None]
        for d, n in numerator.items()
    }


@torch.no_grad()
def score(model, bank, templates):
    model.eval()
    errors = {d: {k: [] for k in ("model", "zero", "template")} for d in ("f0", "f1")}
    for start in range(0, bank.count, 16):
        t = bank.get(list(range(start, min(start + 16, bank.count))))
        for d in errors:
            p = model(t[d], t["v"])
            target = p["standardized"]
            for key, prediction in (
                ("model", p["all"]),
                ("zero", torch.zeros_like(target)),
                ("template", templates[d]),
            ):
                e = per_image_mse(prediction, target, t["v"])
                require(torch.isfinite(e).all(), "Nonfinite evaluation; stop")
                errors[d][key].extend(e.cpu().double().tolist())
    summary = {}
    for d, values in errors.items():
        means = {k: float(np.mean(v)) for k, v in values.items()}
        require(means["zero"] > 0 and means["template"] > 0, "Degenerate baseline")
        summary[d] = {
            **means,
            "mse_over_zero": means["model"] / means["zero"],
            "mse_over_template": means["model"] / means["template"],
        }
    return summary, errors


def bootstrap_difference(candidate, reference, patients):
    """Paired patient-cluster bootstrap of image-mean MSE difference, 2000 draws."""
    c, r = np.asarray(candidate, dtype=float), np.asarray(reference, dtype=float)
    require(len(c) == len(r) == len(patients) and len(c) > 0, "Paired errors mismatch")
    require(np.isfinite(c).all() and np.isfinite(r).all(), "Nonfinite paired errors")
    _, groups = np.unique(np.asarray(patients, dtype=str), return_inverse=True)
    n = int(groups.max()) + 1
    sums = np.bincount(groups, weights=c - r)
    counts = np.bincount(groups)
    draws = np.random.default_rng(SEED).integers(n, size=(2000, n))
    differences = sums[draws].sum(1) / counts[draws].sum(1)
    return {
        "mean_mse_difference": float((c - r).mean()),
        "patient_bootstrap_95_percentile_interval": np.quantile(
            differences, [0.025, 0.975]
        ).tolist(),
        "resamples": 2000,
        "patients": n,
        "simultaneous_interval": False,
    }


def verify_evidence(config, baseline_path, replicate_root):
    source = Path(__file__).resolve().parents[1]
    paths = [baseline_path] + [
        replicate_root / f"seed-{s}/report.json" for s in (20261009, 20261010)
    ]
    reports = [json.loads(p.read_text()) for p in paths]
    base = reports[0]
    require(base["protocol"] == "B-R1-DIAG-v1", "Wrong prior protocol")
    require(
        base["arms"]
        == {
            "original": [1.0, 1.0],
            "encoder_x01": [0.1, 1.0],
            "decoder_x01": [1.0, 0.1],
            "both_x01": [0.1, 0.1],
        },
        "Prior initialization factors changed",
    )
    require(
        base["cohort"]["role"] == "fit"
        and base["cohort"]["train_patients"] == 64
        and base["cohort"]["holdout_patients"] == 16
        and base["cohort"]["one_image_per_patient"] is True,
        "Prior cohort changed",
    )
    require(base["config_sha256"] == sha256(config), "Config changed")
    require(
        base["checker_sha256"]
        == sha256(source / "scripts/probe_stage_b_reconstruction.py"),
        "Probe changed",
    )
    actual = {
        str(p.relative_to(source)): sha256(p)
        for p in sorted((source / "src/ccrr").rglob("*.py"))
    }
    require(base["source_sha256"] == actual, "Production source changed")
    for seed, report in zip((20261008, 20261009, 20261010), reports):
        require(
            report["seed"] == seed
            and report["steps_per_arm"] == 1000
            and report["disposable_optimizer_updates"] == 4000,
            "Prior budget mismatch",
        )
        require(
            report["existing_checkpoints_unchanged"] is True
            and report["formal_training_started"] is False
            and report["scientific_admission_complete"] is False,
            "Prior status mismatch",
        )
        for key in (
            "protocol",
            "cohort",
            "checkpoint_digests",
            "config_sha256",
            "checker_sha256",
            "source_sha256",
            "arms",
        ):
            require(report[key] == base[key], "Prior binding mismatch: " + key)
        require(
            set(report["trajectories"])
            == {"original", "encoder_x01", "decoder_x01", "both_x01"},
            "Prior arms missing",
        )
        for arm, rows in report["trajectories"].items():
            require(
                [r["step"] for r in rows] == [0, 100, 500, 1000]
                and all(r["arm"] == arm for r in rows),
                "Prior trajectory mismatch",
            )
    return base, {str(p): sha256(p) for p in paths}


def run_arm(base, name, banks, probes, templates, out):
    seed_everything(SEED)
    model = make_arm(base, name)
    opt = optimizer([model], [1e-4])
    schedule = TeacherSchedule(opt, STEPS)
    trajectory, final = [], None
    for done in range(STEPS + 1):
        if done in POINTS:
            row = {"arm": name, "step": done, "scores": {}, "probe_diagnostics": {}}
            errors = {}
            for role, bank in banks.items():
                row["scores"][role], errors[role] = score(model, bank, templates)
                t = bank.get(probes[role])
                row["probe_diagnostics"][role] = {
                    d: measure(model, t[d], t["v"]) for d in ("f0", "f1")
                }
            write_json(out / f"errors-{name}-{done}.json", errors)
            torch.save(
                {
                    "stage": PROTOCOL,
                    "arm": name,
                    "step": done,
                    "diagnostic_only": True,
                    "state_dict": model.state_dict(),
                },
                out / f"diagnostic-{name}-{done}.pt",
            )
            trajectory.append(row)
            log_json(out / "trajectory.jsonl", row)
            print(
                json.dumps({"arm": name, "step": done, "scores": row["scores"]}),
                flush=True,
            )
            final = errors
        if done == STEPS:
            break
        ids = (
            rng_for(SEED, PROTOCOL, "batch", done)
            .integers(banks["fit"].count, size=16)
            .tolist()
        )
        loss = control_update(model, banks["fit"], ids, 16, opt, schedule)
        require(np.isfinite(loss), "Nonfinite training loss; stop")
        if (done + 1) % 100 == 0:
            print(
                f"FULLFIT_STEP: {name} {done + 1}/{STEPS} loss={loss:.6f}", flush=True
            )
    return trajectory, final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--baseline-report", required=True, type=Path)
    parser.add_argument("--replicate-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    prior, evidence = verify_evidence(
        args.config, args.baseline_report, args.replicate_root
    )
    cfg = load_config(args.config)
    out, source = args.out.resolve(), Path(__file__).resolve().parents[1]
    require(not out.exists(), "Use a new output directory")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(gpu.startswith("GPU-") and "," not in gpu, "Expose one GPU UUID")
    busy = subprocess.check_output(
        ["nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    ).strip()
    require(not busy, "GPU occupied; stop")
    require(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"),
        "Set deterministic CUBLAS",
    )
    require(
        torch.cuda.is_available()
        and torch.cuda.device_count() == 1
        and "H100" in torch.cuda.get_device_name(0),
        "One H100 required",
    )
    print("FULLFIT_VERIFYING_FROZEN_A", flush=True)
    aroot, _ = verify_teachers(cfg)
    for root in (
        source,
        Path(cfg["output"]).resolve(),
        Path(cfg["data_root"]).resolve(),
        aroot.resolve(),
        args.baseline_report.resolve().parent,
        args.replicate_root.resolve(),
    ):
        require(
            out != root and root not in out.parents and out not in root.parents,
            "Output overlaps evidence/source/data",
        )
    files = {
        "A_initial": aroot / "checkpoints/mfm_probe.pt",
        "B_1000": Path(cfg["output"]) / "checkpoints/purifier-1000.pt",
        "B_10000": Path(cfg["output"]) / "checkpoints/purifier-10000.pt",
        "control_selected": Path(cfg["output"]) / "no_branch_control/selected.pt",
    }
    require(
        {k: sha256(p) for k, p in files.items()} == prior["checkpoint_digests"],
        "Old weights changed",
    )
    seed_everything(SEED)
    data = {role: PADData(cfg["data_root"], role, 512) for role in ("fit", "val_tune")}
    patients = {
        role: [str(r["patient_id"]) for r in ds.rows] for role, ds in data.items()
    }
    require(not set(patients["fit"]) & set(patients["val_tune"]), "Patient overlap")
    probes = {role: cohort(ds.rows)[0] for role, ds in data.items()}
    out.mkdir(mode=0o700, parents=True)
    initial = torch.load(files["A_initial"], map_location="cpu", weights_only=True)[
        "models"
    ]["purifier"]
    device = torch.device("cuda:0")
    item = cfg["weights"]["panderm"]
    teacher = (
        PanDerm(item["source"], item["source_sha256"], item["path"], item["sha256"])
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    banks = {
        role: cache_features(ds, teacher, device, role) for role, ds in data.items()
    }
    del teacher
    seed_everything(int(rng_for(SEED, "B", CONTROL_PROTOCOL, 0).integers(2**31)))
    base = ReconstructionControl(initial["mean"], initial["std"]).to(device)
    templates = fit_templates(banks["fit"], base.mean, base.std)
    torch.save(
        {"protocol": PROTOCOL, "fit_only": True, "templates": templates},
        out / "fit_templates.pt",
    )
    run_record = {
        "protocol": PROTOCOL,
        "seed": SEED,
        "steps_per_arm": STEPS,
        "arms": list(ARMS),
        "fixed_comparison_step": STEPS,
        "source_sha256": prior["source_sha256"],
        "checker_sha256": sha256(__file__),
        "config_sha256": sha256(args.config),
        "evidence_sha256": evidence,
        "checkpoint_digests": prior["checkpoint_digests"],
        "environment": environment(),
        "cohorts": {
            role: {
                "images": len(ds),
                "patients": len(set(patients[role])),
                "ordered_rows_sha256": hashlib.sha256(
                    json.dumps(
                        [[r["patient_id"], r["img_id"]] for r in ds.rows],
                        ensure_ascii=False,
                    ).encode()
                ).hexdigest(),
                "probe_images": len(probes[role]),
                "probe_ordered_indices_sha256": hashlib.sha256(
                    json.dumps(probes[role]).encode()
                ).hexdigest(),
            }
            for role, ds in data.items()
        },
        "limitations": [
            "One seed, fixed candidate; no scale or checkpoint search",
            "val_tune is development validation, not independent scientific admission",
            "Full scores are image-weighted; gradient/entropy/code variance diagnostics use fixed 64-patient probes",
            "The former fit holdout is now used for training",
            "Diagnostic weights cannot resume production B",
        ],
    }
    write_json(out / "run_record.json", run_record)
    trajectories, finals = {}, {}
    for name in ARMS:
        trajectories[name], finals[name] = run_arm(
            base, name, banks, probes, templates, out
        )
    comparisons = {}
    for role in data:
        comparisons[role] = {}
        for d in ("f0", "f1"):
            c = finals["both_x01"][role][d]
            references = {
                "original": finals["original"][role][d]["model"],
                "zero": c["zero"],
                "fit_template": c["template"],
            }
            comparisons[role][d] = {
                k: bootstrap_difference(c["model"], v, patients[role])
                for k, v in references.items()
            }
    require(
        {k: sha256(p) for k, p in files.items()} == prior["checkpoint_digests"],
        "Old weights changed during experiment",
    )
    require(
        all(sha256(Path(p)) == h for p, h in evidence.items()),
        "Evidence reports changed",
    )
    require(
        sha256(args.config) == run_record["config_sha256"],
        "Config changed during experiment",
    )
    report = {
        **run_record,
        "trajectories": trajectories,
        "final_paired_comparisons": comparisons,
        "existing_checkpoints_unchanged": True,
        "formal_training_started": False,
        "scientific_admission_complete": False,
        "disposable_optimizer_updates": 2 * STEPS,
    }
    write_json(out / "report.json", report)
    print("FULLFIT_RECONSTRUCTION_REPORT:", out / "report.json", flush=True)
    print("FORMAL_TRAINING_STARTED: false", flush=True)


if __name__ == "__main__":
    main()
