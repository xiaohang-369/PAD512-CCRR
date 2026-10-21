#!/usr/bin/env python3
"""Launch with config-selected GPUs. No fixed GPU index/count or batch scaling."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

repo = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("command", choices=["prepare", "preflight", "train"])
p.add_argument("--config", required=True)
p.add_argument("--resources")
p.add_argument("--root")
args, extra = p.parse_known_args()
if subprocess.check_output(
    ["git", "-C", str(repo), "status", "--porcelain"], text=True
).strip():
    raise SystemExit(
        "Run a clean fixed release; edit a COPY of config outside the release"
    )
if args.resources:
    if not args.root:
        raise SystemExit("--resources requires --root")
    root = Path(args.root).resolve()
    resource = json.loads(Path(args.resources).read_text())
    if resource["schema"] != "ccrr-v4-phase1-resources-1":
        raise SystemExit("Wrong phase-1 resources schema")
    paths = {
        k: (root / resource[k]).resolve()
        for k in ("vgg_file", "prepared_dir", "runs_dir")
    }
    if any(not v.is_relative_to(root) for v in paths.values()):
        raise SystemExit("V4 resources/cache/output must remain inside this V4 root")
    if args.command == "prepare":
        defaults = {
            "--data-root": resource["data_root"],
            "--vgg": str(paths["vgg_file"]),
            "--vgg-sha256": resource["vgg_sha256"],
            "--output": str(paths["prepared_dir"]),
        }
    else:
        name = (
            args.command
            + "."
            + datetime.now().strftime("%Y%m%d-%H%M%S")
            + "."
            + str(os.getpid())
        )
        defaults = {
            "--prepared": str(paths["prepared_dir"] / "prepared.json"),
            "--output": str(paths["runs_dir"] / name),
        }
    for key, value in defaults.items():
        if key not in extra and not any(a.startswith(key + "=") for a in extra):
            extra.extend([key, value])
config = Path(args.config).resolve()
cfg = json.loads(config.read_text())
env = os.environ.copy()
env["PYTHONDONTWRITEBYTECODE"] = "1"
env["PYTHONPATH"] = os.pathsep.join([str(repo / "v4"), str(repo / "src")])
env["OMP_NUM_THREADS"] = str(cfg["runtime"]["cpu_threads"])
env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
command = [sys.executable, "-B"]
if args.command == "prepare" or cfg["runtime"]["device"] == "cpu":
    env["CUDA_VISIBLE_DEVICES"] = ""
else:
    ids = cfg["runtime"]["gpu_ids"]
    if (
        not ids
        or len(set(map(str, ids))) != len(ids)
        or cfg["global_batch_size"] % len(ids)
    ):
        raise SystemExit(
            "Set unique physical gpu_ids in config; count must divide global batch 8"
        )
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, ids))
    if len(ids) > 1:
        command += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={len(ids)}",
        ]
command += ["-m", "ccrr_v4.phase1.cli", args.command, "--config", str(config), *extra]
raise SystemExit(subprocess.call(command, env=env))
