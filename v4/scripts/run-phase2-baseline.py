"""Launch an isolated, frozen baseline comparison on config-selected GPUs."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / "v4"))
from ccrr_v4.phase2_baseline import validate_config  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--config", required=True, type=Path)
p.add_argument("--root", required=True, type=Path)
a = p.parse_args()
if subprocess.check_output(
    ["git", "-C", str(repo), "status", "--porcelain"], text=True
).strip():
    raise SystemExit("Use a clean fixed release; edit external config only")
cfg = json.loads(a.config.read_text())
validate_config(cfg)
root = a.root.resolve()
output = (
    root
    / "runs"
    / (
        "phase2-baseline."
        + datetime.now().strftime("%Y%m%d-%H%M%S")
        + f".{os.getpid()}"
    )
)
output.mkdir(parents=True, exist_ok=False)
config = output / "config.json"
config.write_text(json.dumps(cfg, indent=2) + "\n")
env = os.environ.copy()
env["PYTHONDONTWRITEBYTECODE"] = "1"
env["PYTHONUNBUFFERED"] = "1"
env["PYTHONPATH"] = os.pathsep.join(str(repo / d) for d in ("v4", "v3", "v2", "src"))
env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg["gpu_ids"]))
env["OMP_NUM_THREADS"] = str(cfg["cpu_threads"])
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
    "ccrr_v4.phase2_baseline",
    "--config",
    str(config),
    "--output",
    str(output),
]
status = dict(state="running", mode="evaluation", training_started=False)
(output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
print(f"PHASE2_BASELINE_RUN={output}", flush=True)
code = subprocess.call(command, env=env)
complete = code == 0 and (output / "report.json").is_file()
status.update(state="completed" if complete else "failed", exit_code=code)
(output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
raise SystemExit(0 if complete else code or 1)
