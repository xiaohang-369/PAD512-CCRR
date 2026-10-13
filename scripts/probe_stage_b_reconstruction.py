"""B-R1-DIAG-v1: isolated, fit-only reconstruction investigation.

Default: measurements only. --steps 1000 explicitly runs four disposable
initialization arms, NOT formal B. No production checkpoint is written.
The 1000 updates retain the first 1000 positions of the original 10000-step
schedule. Results cannot certify clinical utility, privacy, or a revised B.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch

from ccrr.config import load_config, sha256
from ccrr.data.pad import PADData, collate, rng_for
from ccrr.errors import require
from ccrr.representation.purifier import Purifier
from ccrr.runtime.support import environment, seed_everything, write_json, log_json
from ccrr.teachers.adapters import PanDerm
from ccrr.tensor_ops import gray, masked_mean, ste_round
from ccrr.training.b_control import (
    ReconstructionControl,
    control_update,
    CONTROL_PROTOCOL,
)
from ccrr.training.b_resources import verify_teachers
from ccrr.training.optimization import optimizer, TeacherSchedule

PROTOCOL = "B-R1-DIAG-v1"
# Factor levels fixed before the new experiment, not searched using val/test.
ARMS = {
    "original": (1.0, 1.0),
    "encoder_x01": (0.1, 1.0),
    "decoder_x01": (1.0, 0.1),
    "both_x01": (0.1, 0.1),
}


def cohort(rows, train_n=64, holdout_n=16):
    """One real image per fit patient; diagnostic holdout is also inside fit."""
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(str(row["patient_id"]), []).append(i)

    def key(s):
        return hashlib.sha256((PROTOCOL + ":" + s).encode()).hexdigest()

    patients = sorted(groups, key=key)
    require(len(patients) >= train_n + holdout_n, "Insufficient fit patients")
    chosen = [
        min(groups[p], key=lambda i: key(str(rows[i]["img_id"])))
        for p in patients[: train_n + holdout_n]
    ]
    return chosen[:train_n], chosen[train_n:]


def make_arm(base, name):
    """Scale terminal weights only, in a private copy. Not a new forward op."""
    model = copy.deepcopy(base)
    es, ds = ARMS[name]
    with torch.no_grad():
        model.encoder.end.weight.mul_(es)
        model.decoder.end.weight.mul_(ds)
    return model


class Bank:
    def __init__(self, storage):
        self.storage = storage

    def get(self, ids):
        return {k: v[ids] for k, v in self.storage.items()}


def measure(model, features, coverage):
    """Coverage/image-weighted diagnostics; gradient queries never step Adam."""
    model.eval()
    target = (
        (features - model.mean) / model.std.clamp_min(1e-3) * (coverage > 0)
    ).detach()
    z = model.encoder(target, coverage)
    # Preserve the production round STE AND the tanh derivative.
    q = ste_round(7 * z.tanh()) * (coverage > 0)
    reconstruction = model.decoder(q / 7, coverage)
    loss = masked_mean((reconstruction - target).square(), coverage)
    gz, ge, gd = torch.autograd.grad(
        loss, (z, model.encoder.end.weight, model.decoder.end.weight)
    )
    with torch.no_grad():
        zero = masked_mean(target.square(), coverage)
        derivative = 1 - z.tanh().square()
        weights = coverage / coverage.sum((2, 3), keepdim=True)
        hist = torch.stack(
            [
                ((q == k).float() * weights).sum((0, 2, 3)) / q.shape[0]
                for k in range(-7, 8)
            ],
            dim=1,
        )
        entropy = -(hist * hist.clamp_min(1e-30).log2()).sum(1)
        site_weight = coverage.sum(0).clamp_min(1e-12)
        site_mean = (q / 7 * coverage).sum(0) / site_weight
        site_variance = ((q / 7 - site_mean).square() * coverage).sum(0) / site_weight
        supported = (coverage.sum(0) > 0).expand_as(site_variance)
        valid = (coverage > 0).expand_as(z)
        return {
            "reconstruction_mse": float(loss),
            "zero_output_mse": float(zero),
            "mse_over_zero": float(loss / zero.clamp_min(1e-12)),
            "saturation_fraction": float(masked_mean((q.abs() == 7).float(), coverage)),
            "zero_code_fraction": float(masked_mean((q == 0).float(), coverage)),
            "abs_z_quantiles_unweighted_valid": torch.quantile(
                z[valid].abs(), z.new_tensor([0.5, 0.95, 0.99])
            )
            .cpu()
            .tolist(),
            "tanh_derivative_mean": float(masked_mean(derivative, coverage)),
            "derivative_below_1e_minus3_fraction": float(
                masked_mean((derivative < 1e-3).float(), coverage)
            ),
            "code_histogram_levels_minus7_to7": hist.mean(0).cpu().tolist(),
            "channel_entropy_bits_min_mean_max": [
                float(entropy.min()),
                float(entropy.mean()),
                float(entropy.max()),
            ],
            "between_image_normalized_code_variance_same_site_mean": float(
                site_variance[supported].mean()
            ),
            "encoder_terminal_grad_l2": float(ge.norm()),
            "decoder_terminal_grad_l2": float(gd.norm()),
            "dL_d_log_z_scale": float((gz * z).sum()),
            "gradient_note": "local reconstruction gradient, not an Adam update; positive radial derivative favors shrinking z under a direct infinitesimal descent",
        }


def evaluate(model, bank, ids):
    # Maximum 64 images here; only the diagnostic bank, no teacher graph.
    t = bank.get(ids)
    return {key: measure(model, t[key], t["v"]) for key in ("f0", "f1")}


def pilot(base, bank, train_ids, holdout_ids, steps, seed, emit):
    require(steps in (0, 1000), "Only read-only or preregistered 1000 updates")
    results = {}
    for name in ARMS:
        seed_everything(seed)
        model = make_arm(base, name)
        opt = optimizer([model], [1e-4])
        schedule = TeacherSchedule(opt, 10000)
        snapshots = []
        for done in range(steps + 1):
            if done in (0, 100, 500, 1000):
                row = {
                    "arm": name,
                    "step": done,
                    "train": evaluate(model, bank, train_ids),
                    "fit_holdout": evaluate(model, bank, holdout_ids),
                }
                snapshots.append(row)
                emit(row)
            if done == steps:
                break
            rng = rng_for(seed, PROTOCOL, "batch", done)
            ids = [train_ids[i] for i in rng.integers(len(train_ids), size=16)]
            control_update(model, bank, ids, 16, opt, schedule)
        results[name] = snapshots
        del model, opt, schedule
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Existing B candidate config")
    parser.add_argument(
        "--out", required=True, help="New isolated diagnostic directory"
    )
    parser.add_argument("--steps", type=int, choices=(0, 1000), default=0)
    parser.add_argument(
        "--seed", type=int, choices=(20261008, 20261009, 20261010), default=20261008
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    print("VERIFYING_FROZEN_A: read-only checkpoint checks", flush=True)
    aroot, _ = verify_teachers(cfg)
    out = Path(args.out).resolve()
    roots = (
        aroot,
        Path(cfg["output"]).resolve(),
        Path(cfg["data_root"]).resolve(),
        Path(__file__).resolve().parents[1],
    )
    require(
        not out.exists()
        and all(
            out != r and out not in r.parents and r not in out.parents for r in roots
        ),
        "Use a new isolated output",
    )
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(gpu.startswith("GPU-") and "," not in gpu, "Expose one idle GPU UUID")
    busy = subprocess.check_output(
        ["nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    ).strip()
    require(not busy, "GPU occupied; stop")
    require(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"),
        "Set deterministic CUBLAS configuration",
    )
    require(
        torch.cuda.is_available()
        and torch.cuda.device_count() == 1
        and "H100" in torch.cuda.get_device_name(0),
        "One H100 required",
    )
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    out.mkdir(mode=0o700, parents=True)
    print(f"DIAGNOSTIC_START: {out}; steps_per_arm={args.steps}", flush=True)
    files = {"A_initial": aroot / "checkpoints/mfm_probe.pt"}
    broot = Path(cfg["output"])
    for step in (1000, 10000):
        files[f"B_{step}"] = broot / "checkpoints" / f"purifier-{step}.pt"
    files["control_selected"] = broot / "no_branch_control/selected.pt"
    digests = {k: sha256(p) for k, p in files.items()}
    initial = torch.load(files["A_initial"], map_location="cpu", weights_only=True)[
        "models"
    ]["purifier"]
    ds = PADData(cfg["data_root"], "fit", 512)
    train, holdout = cohort(ds.rows)
    chosen = train + holdout
    # Only a private digest is exported, not patient/image identities.
    selection_digest = hashlib.sha256(
        json.dumps(
            [[ds.rows[i]["patient_id"], ds.rows[i]["img_id"]] for i in chosen],
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    item = cfg["weights"]["panderm"]
    teacher = (
        PanDerm(item["source"], item["source_sha256"], item["path"], item["sha256"])
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    parts = {"f0": [], "f1": [], "v": []}
    with torch.no_grad():
        for start in range(0, len(chosen), 16):
            b = collate([ds[i] for i in chosen[start : start + 16]], device)
            for d, x in enumerate((b["x"], gray(b["x"]))):
                f, v, _ = teacher(x, b["rect"])
                require(
                    f.shape[1:] == (768, 14, 14) and torch.isfinite(f).all(),
                    "Invalid frozen MFM features",
                )
                parts[f"f{d}"].append(f.detach())
                if d == 0:
                    parts["v"].append(v.detach())
                else:
                    require(torch.equal(parts["v"][-1], v), "Domain geometry mismatch")
            print(
                f"DIAGNOSTIC_CACHE: {min(start + 16, len(chosen))}/{len(chosen)}",
                flush=True,
            )
    del teacher
    bank = Bank({k: torch.cat(v) for k, v in parts.items()})
    del parts
    train_ids, holdout_ids = list(range(64)), list(range(64, 80))
    observed = {}
    for name, path in files.items():
        control = name == "control_selected"
        raw = torch.load(path, map_location="cpu", weights_only=True)
        model = (
            ReconstructionControl(initial["mean"], initial["std"])
            if control
            else Purifier()
        )
        model.load_state_dict(
            raw["models"]["control" if control else "purifier"], strict=True
        )
        model = model.to(device).eval()
        observed[name] = {
            "train": evaluate(model, bank, train_ids),
            "fit_holdout": evaluate(model, bank, holdout_ids),
        }
        write_json(out / (name + ".json"), observed[name])
        print(f"CHECKPOINT_MEASURED: {name}", flush=True)
        del model, raw
    control_seed = int(rng_for(args.seed, "B", CONTROL_PROTOCOL, 0).integers(2**31))
    seed_everything(control_seed)
    base = ReconstructionControl(initial["mean"], initial["std"]).to(device)

    def emit(row):
        log_json(out / "trajectory.jsonl", row)
        compact = {
            "arm": row["arm"],
            "step": row["step"],
            "holdout": {
                k: {
                    n: v[n]
                    for n in (
                        "mse_over_zero",
                        "saturation_fraction",
                        "tanh_derivative_mean",
                    )
                }
                for k, v in row["fit_holdout"].items()
            },
        }
        print(json.dumps(compact), flush=True)

    arms = pilot(base, bank, train_ids, holdout_ids, args.steps, args.seed, emit)
    require(
        all(sha256(files[k]) == v for k, v in digests.items()),
        "Checkpoint changed during audit",
    )
    report = {
        "protocol": PROTOCOL,
        "seed": args.seed,
        "arms": ARMS,
        "steps_per_arm": args.steps,
        "formal_training_started": False,
        "disposable_optimizer_updates": args.steps * len(ARMS),
        "scientific_admission_complete": False,
        "checkpoint_digests": digests,
        "existing_checkpoints_unchanged": True,
        "checker_sha256": sha256(__file__),
        "config_sha256": sha256(args.config),
        "environment": environment(),
        "source_sha256": {
            str(p.relative_to(Path(__file__).parents[1])): sha256(p)
            for p in sorted((Path(__file__).parents[1] / "src/ccrr").rglob("*.py"))
        },
        "cohort": {
            "role": "fit",
            "train_patients": 64,
            "holdout_patients": 16,
            "one_image_per_patient": True,
            "sha256": selection_digest,
        },
        "observed": observed,
        "trajectories": arms,
        "limitations": [
            "Diagnostic fit subset; not independent scientific validation",
            "Initialization scale 0.1 is a preregistered hypothesis, not a verified fix",
            "No clinical/identity admission, no automatic arm promotion",
            "Zero-output baseline is frozen-fit mean in standardized coordinates",
            "Entropy is marginal per channel, not mutual information or privacy",
        ],
    }
    write_json(out / "report.json", report)
    print(f"RECONSTRUCTION_DIAGNOSTIC_REPORT: {out / 'report.json'}", flush=True)
    print("FORMAL_TRAINING_STARTED: false", flush=True)


if __name__ == "__main__":
    main()
