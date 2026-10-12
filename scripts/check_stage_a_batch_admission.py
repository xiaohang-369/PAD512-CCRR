"""Check three real Stage A models at one microbatch; no formal training.

Interface inputs are synthetic. Disposable production-loop updates use admitted
fit images and an artificial schedule offset of 750, from official base weights.
Compare two uninterrupted updates against one update + checkpoint reload + one.
This does not measure learned capability, full-run throughput, validation or MFM
statistics initialization. Old run outputs and checkpoints are never opened.
"""

import argparse
import copy
import gc
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml

from check_teacher_interface import check_inputs, fingerprints, load_real_model
from probe_stage_a_speed import exact_equal
from ccrr.config import load_config, sha256
from ccrr.data.cache import PixelCache
from ccrr.data.pad import PADData, PairSampler
from ccrr.data.resize import POLICY, teacher_input_sources
from ccrr.errors import require
from ccrr.representation.purifier import Purifier
from ccrr.runtime.preflight import BASE, PINNED, batch_checker_sources
from ccrr.runtime.support import seed_everything, environment, write_json
from ccrr.training import teachers
from ccrr.training.a_state import binding


class Finished(Exception):
    pass


def replay(cfg, data, template, kind, folder, steps, resume=False):
    """Run unchanged production loops, stopping only after a durable save."""
    folder.mkdir(parents=True, exist_ok=resume)
    seed_everything(cfg["seed"])
    device = torch.device(cfg["device"])
    model = copy.deepcopy(template).to(device)
    if kind != "panderm":
        for p in model.parameters():
            p.requires_grad_(True)
    e = SimpleNamespace(
        cfg={**cfg, "output": str(folder), "resume_interval": 1, "resume_a": resume},
        output=folder,
        seed=cfg["seed"],
        device=device,
        fit=data,
        data={},
        weights=data.class_weights().to(device),
        log=lambda *a: None,
        sampler=PairSampler(data) if kind == "resnet50" else None,
        reid=model,
        mfm=model if kind == "panderm" else None,
        purifier=Purifier().to(device) if kind == "panderm" else None,
    )
    name = {"efficientnet_b0": "clinical", "resnet50": "reid", "panderm": "mfm_probe"}[
        kind
    ]
    init, save, update = (
        teachers.AProgress.__init__,
        teachers.AProgress.save,
        teachers.update,
    )
    updates, start, target = 0, None, None
    before_buffers, before_parameters = {}, {}
    mfm_before = fingerprints(model) if kind == "panderm" else None

    def initialize(progress, *args, **kwargs):
        nonlocal start, target, before_buffers, before_parameters
        init(progress, *args, **kwargs)
        if not resume:
            require(not progress.path.exists(), "Refuse existing diagnostic progress")
            progress.scheduler.next_step = 750
        start = progress.start
        target = start + steps
        before_buffers = {k: v.detach().cpu().clone() for k, v in model.named_buffers()}
        before_parameters = {
            f"{key}/{k}": v.detach().cpu().clone()
            for key, module in progress.modules.items()
            for k, v in module.named_parameters()
        }

    def checked_update(opt):
        nonlocal updates
        grads = [
            p.grad for g in opt.param_groups for p in g["params"] if p.grad is not None
        ]
        require(
            grads and any(bool(g.abs().sum() > 0) for g in grads),
            "All gradients absent/zero",
        )
        require(
            all(
                p.dtype == torch.float32 for g in opt.param_groups for p in g["params"]
            ),
            "Expected FP32 parameters",
        )
        value = update(opt)  # Original finite checks, clipping and AdamW update.
        updates += 1
        return value

    def stop(progress, endpoint=False):
        save(progress, endpoint)
        if progress.start == target:
            raise Finished()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with (
        patch.object(teachers.AProgress, "__init__", initialize),
        patch.object(teachers.AProgress, "save", stop),
        patch.object(teachers, "update", checked_update),
    ):
        try:
            if kind == "efficientnet_b0":
                teachers.train_classifier(
                    e,
                    model,
                    name,
                    cfg["budgets"]["clinical"],
                    [lambda b: b["x"], lambda b: teachers.gray(b["x"])],
                )
            elif kind == "resnet50":
                teachers.initialize_base(e, model, "resnet50")
                teachers.train_reid_teacher(e)
            else:
                teachers.train_mfm_probe(e)
        except Finished:
            pass
    require(updates == steps and target is not None, "Incomplete diagnostic loop")
    state = torch.load(
        folder / "checkpoints" / f"{name}.last.pt",
        map_location="cpu",
        weights_only=True,
    )
    require(
        state["step"] == state["training_state"]["scheduler"]["next_step"] == target,
        "Incomplete save boundary",
    )
    unchanged = all(
        torch.equal(before_buffers[k], v.detach().cpu())
        for k, v in model.named_buffers()
    )
    changed = any(
        not torch.equal(
            value, state["models"][key.split("/", 1)[0]][key.split("/", 1)[1]]
        )
        for key, value in before_parameters.items()
    )
    require(
        unchanged and changed and bool(state["optimizer"]["state"]),
        "Unexpected buffers/parameters/optimizer state",
    )
    frozen = mfm_before is None or fingerprints(model) == mfm_before
    require(frozen, "Frozen PanDerm backbone changed")
    return {
        "finite_nonzero_gradients": True,
        "optimizer_state_allocated": True,
        "buffers_unchanged": True,
        "trainable_parameters_updated": True,
        "frozen_backbone_unchanged": frozen,
        "fp32": True,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if device.type == "cuda"
        else None,
    }, state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        required=True,
        help="Existing admitted batch-1 configuration; read only",
    )
    p.add_argument("--out", required=True, help="New private diagnostic directory")
    p.add_argument(
        "--training-output",
        required=True,
        help="New future run directory; must not exist",
    )
    p.add_argument("--microbatch", type=int, choices=(2, 4, 8, 16), required=True)
    args = p.parse_args()
    base = load_config(args.config)
    out, future = Path(args.out).resolve(), Path(args.training_output).resolve()
    source = Path(__file__).resolve().parents[1]
    old = Path(base["output"]).resolve()
    require(
        not out.exists() and not future.exists(),
        "Use new diagnostic and training directories",
    )
    require(
        not any(
            a.is_relative_to(b) or b.is_relative_to(a)
            for a, b in (
                (out, old),
                (future, old),
                (out, future),
                (out, source),
                (future, source),
            )
        ),
        "Directories must be separate",
    )
    require(
        base["device"] == "cuda" and base["batch_teacher"] == base["batch_pairs"] == 16,
        "Keep CUDA/effective batches 16",
    )
    require(
        torch.cuda.is_available()
        and torch.cuda.device_count() == 1
        and "H100" in torch.cuda.get_device_name(0),
        "Expose one idle H100",
    )
    cfg = {
        **base,
        "microbatch": args.microbatch,
        "pixel_cache_bytes": 8 * 1024**3,
        "output": str(future),
        "resume_a": False,
    }
    cfg.pop("batch_execution_report", None)
    out.mkdir(parents=True, mode=0o700)
    seed_everything(cfg["seed"])
    pool = PixelCache(cfg["pixel_cache_bytes"])
    data = pool.wrap(PADData(cfg["data_root"], "fit", 512))
    report = {
        "passed": False,
        "formal_training_started": False,
        "old_run_modified": False,
        "microbatch": args.microbatch,
        "configuration_binding": binding(cfg),
        "checker_sources": batch_checker_sources(),
        "environment": environment(),
        "H100_validated": True,
        "models": {},
        "execution_flags": {
            "fp32": torch.get_default_dtype() == torch.float32,
            "strict_determinism": torch.are_deterministic_algorithms_enabled()
            and not torch.is_deterministic_algorithms_warn_only_enabled(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
        "limitations": [
            "Two disposable updates from official initialization at synthetic schedule index 750",
            "Does not measure full-run throughput, validation, MFM statistics initialization or learned capability",
            "Same-configuration save/reload equality; no equivalence to old microbatch=1 trajectory",
        ],
    }
    try:
        for name in BASE:
            print(
                f"BATCH_ADMISSION_START: {name}, microbatch={args.microbatch}",
                flush=True,
            )
            entry = cfg["weights"][name]
            require(entry["sha256"] == PINNED[name], "Frozen weights required")
            interface = {
                "model": name,
                "sha256": entry["sha256"],
                "device": "cuda",
                "torch": torch.__version__,
                "batch_size": args.microbatch,
                "adapter_source_sha256": sha256(
                    source / "src/ccrr/teachers/adapters.py"
                ),
                "teacher_input_sources": teacher_input_sources(),
                "teacher_resize_policy": POLICY,
            }
            model = load_real_model(
                SimpleNamespace(
                    model=name,
                    weights=entry["path"],
                    sha256=entry["sha256"],
                    source=entry.get("source"),
                ),
                interface,
            ).cuda()
            before = fingerprints(model)
            torch.cuda.reset_peak_memory_stats()
            interface["forward_backward"] = check_inputs(model, name, args.microbatch)
            require(fingerprints(model) == before, "Frozen interface mutated model")
            interface.update(
                passed=True,
                parameters_and_buffers_unchanged=True,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            )
            model.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            folder = out / name
            print(f"BATCH_REPLAY: {name}, uninterrupted 2 updates", flush=True)
            full, expected = replay(cfg, data, model, name, folder / "full", 2)
            print(f"BATCH_REPLAY: {name}, 1 update + reload + 1 update", flush=True)
            first, _ = replay(cfg, data, model, name, folder / "split", 1)
            last, actual = replay(
                cfg, data, model, name, folder / "split", 1, resume=True
            )
            require(
                exact_equal(expected, actual), "Full saved state differs after reload"
            )
            training = {
                **last,
                "steps": 2,
                "resume_state_match": True,
                "peak_allocated_bytes": max(
                    r["peak_allocated_bytes"] for r in (full, first, last)
                ),
            }
            report["models"][name] = {"interface": interface, "training": training}
            write_json(out / "report.json", report)
            print(
                f"BATCH_ADMISSION_PASS: {name}, resume_state_match=True, training_peak_gib={training['peak_allocated_bytes'] / 1024**3:.3f}",
                flush=True,
            )
            del model, expected, actual, _
            gc.collect()
            torch.cuda.empty_cache()
        report["passed"] = True
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        write_json(out / "report.json", report)
    cfg["batch_execution_report"] = {
        "path": str(out / "report.json"),
        "sha256": sha256(out / "report.json"),
    }
    with (out / "config.yaml").open("x") as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False, allow_unicode=True)
    print("CONFIG_READY:", out / "config.yaml", flush=True)
    print("FORMAL_TRAINING_STARTED: false", flush=True)


if __name__ == "__main__":
    main()
