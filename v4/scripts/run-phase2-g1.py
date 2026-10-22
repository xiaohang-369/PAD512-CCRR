"""Launch a new G1 preflight or supervised training run; never reuse an output."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / "v4"))
from ccrr_v4.phase2_objectives import validate_config  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("mode", choices=("preflight", "train"))
p.add_argument("--config", required=True, type=Path)
p.add_argument("--root", required=True, type=Path)
p.add_argument("--preflight", type=Path)
a = p.parse_args()
if a.mode == "train" and a.preflight is None:
    p.error("train requires --preflight /absolute/path/report.json")
if subprocess.check_output(
    ["git", "-C", str(repo), "status", "--porcelain"], text=True
).strip():
    raise SystemExit("Use a clean fixed release; edit the external config")
cfg = json.loads(a.config.read_text())
validate_config(cfg)
output = (
    a.root.resolve()
    / "runs"
    / (
        "g1-"
        + a.mode
        + "."
        + datetime.now().strftime("%Y%m%d-%H%M%S")
        + f".{os.getpid()}"
    )
)
output.mkdir(parents=True, exist_ok=False)
config = output / "config.json"
config.write_text(json.dumps(cfg, indent=2) + "\n")
env = os.environ.copy()
env.update(
    PYTHONDONTWRITEBYTECODE="1",
    PYTHONUNBUFFERED="1",
    PYTHONPATH=os.pathsep.join(str(repo / d) for d in ("v4", "v3", "v2", "src")),
    CUDA_DEVICE_ORDER="PCI_BUS_ID",
    CUDA_VISIBLE_DEVICES=",".join(map(str, cfg["gpu_ids"])),
    OMP_NUM_THREADS=str(cfg["cpu_threads"]),
)
command = [sys.executable, "-B"]
if len(cfg["gpu_ids"]) > 1:
    command += [
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={len(cfg['gpu_ids'])}",
    ]
command += [
    "-m",
    "ccrr_v4.phase2_g1",
    a.mode,
    "--config",
    str(config),
    "--output",
    str(output),
]
if a.preflight:
    command += ["--preflight", str(a.preflight.resolve())]
status = dict(state="running", mode=a.mode, training_requested=a.mode == "train")
status_path = output / "status.json"
status_path.write_text(json.dumps(status, indent=2) + "\n")
print(f"G1_RUN={output}", flush=True)
print(
    f"GPUS={cfg['gpu_ids']} GLOBAL_BATCH=16 IMAGE=512 EPOCHS={cfg['epochs']}",
    flush=True,
)
code = subprocess.call(command, env=env)
complete = code == 0 and (output / "report.json").is_file()
status.update(state="completed" if complete else "failed", exit_code=code)
status_path.write_text(json.dumps(status, indent=2) + "\n")
raise SystemExit(0 if complete else code or 1)
