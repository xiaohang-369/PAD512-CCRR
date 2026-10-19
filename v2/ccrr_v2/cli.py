"""Explicit one-stage commands, never automatic A->B->generator progression."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import torch

from .runtime import (
    REPO,
    completed,
    device_for,
    digest,
    new_run,
    read,
    seed_all,
    validate_resources,
    write,
)


def validate_config(cfg):
    expected = read(REPO / "v2/configs/first_validation.json")
    if set(cfg) != set(expected) or cfg["schema"] != expected["schema"]:
        raise ValueError("Unknown or missing configuration fields")
    if cfg["batch_size"] != 16:
        raise ValueError("First V2 protocol uses 16 images / 16 patient pairs")
    for key, value in cfg.items():
        if key.endswith("steps") or key in ("evaluate_every", "bootstrap_repetitions"):
            if type(value) is not int or value < 1:
                raise ValueError(f"Invalid positive integer: {key}")
        if key.endswith("lr") or key == "identity_weight":
            if not isinstance(value, (int, float)) or not 0 < value < 1:
                raise ValueError(f"Invalid coefficient: {key}")
    if type(cfg["seed"]) is not int:
        raise ValueError("Integer seed required")
    if cfg["scope"] != expected["scope"]:
        raise ValueError("This executable supports development validation only")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=("preflight", "features", "probe", "reid", "s0", "audit")
    )
    p.add_argument("--resources", required=True, help="Private V2 resources JSON")
    p.add_argument("--config", default=str(REPO / "v2/configs/first_validation.json"))
    p.add_argument("--output", help="New absolute directory below V2 workspace/runs")
    p.add_argument("--features", help="Completed V2 feature cache run")
    p.add_argument("--s0", help="Completed V2 S0 run; omit for raw-feature audit")
    p.add_argument("--variant", choices=("medical", "joint"))
    p.add_argument("--device", default="cuda:0")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    resources, cfg = read(args.resources), read(args.config)
    validate_config(cfg)
    validate_resources(resources)
    if args.stage == "preflight":
        from ccrr.data.admission import verify_dataset

        result = verify_dataset(resources["data_root"], pixels=False)
        base = Path(resources["workspace_root"])
        while not base.exists():
            base = base.parent
        print(
            json.dumps(
                {
                    "resource_hashes_verified": True,
                    "data": result,
                    "free_bytes": shutil.disk_usage(base).free,
                    "torch": torch.__version__,
                    "python": sys.version,
                    "training_started": False,
                    "pixels_reverified": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if not args.output or not Path(args.output).is_absolute():
        raise ValueError("An absolute, fresh --output is required")
    if args.stage in ("probe", "s0", "audit") and not args.features:
        raise ValueError("This stage requires a completed V2 --features run")
    if args.stage == "s0" and args.variant is None:
        raise ValueError("Choose the explicit matched S0 variant")
    if args.s0 and args.stage != "audit":
        raise ValueError("--s0 is only valid for independent audit")
    if args.variant and args.stage != "s0":
        raise ValueError("--variant is only valid for S0 training")
    device = device_for(args.device)
    parents = {}
    if args.features:
        parents = {
            "features": str(Path(args.features).resolve()),
            "features_sha256": digest(Path(args.features) / "features.json"),
        }
    if args.s0:
        parents.update(
            s0=str(Path(args.s0).resolve()), s0_sha256=digest(Path(args.s0) / "s0.pt")
        )
    output = new_run(
        resources,
        args.output,
        args.stage,
        {**cfg, "variant": args.variant, "device": args.device},
        parents,
    )
    try:
        seed_all(cfg["seed"])
        from .features import extract, load_cache

        if args.stage == "features":
            extract(resources, cfg, output, device)
        elif args.stage == "reid":
            from .reid import train

            train(resources, cfg, output, device)
        else:
            cache = load_cache(args.features, resources, device)
            from .learning import apply_s0, fit_probe, train_s0

            if args.s0:
                apply_s0(cache, args.s0, resources, parents["features_sha256"])
            if args.stage == "probe":
                fit_probe(cache, cfg, output)
                completed(output, scope="fresh_disease_probe")
            elif args.stage == "s0":
                train_s0(cache, cfg, output, args.variant)
            elif args.stage == "audit":
                from .audit import audit

                audit(cache, cfg, output)
    except BaseException as exc:
        write(
            output / "status.json",
            {
                "state": "interrupted"
                if isinstance(exc, KeyboardInterrupt)
                else "failed",
                "error_type": type(exc).__name__,
                "scientific_acceptance": "not_evaluated",
            },
        )
        raise


if __name__ == "__main__":
    main()
