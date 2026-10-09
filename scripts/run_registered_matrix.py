"""Create the four registered development configs; execute only on explicit --run.

Selection never reads test results. Additional seeds use the selected configuration
and are explicit invocations, avoiding automatic choice of a lucky seed.
"""

from pathlib import Path
import argparse
import copy
import subprocess
import sys
import yaml

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--out", required=True)
p.add_argument("--run", action="store_true")
args = p.parse_args()
base = yaml.safe_load(Path(args.config).read_text())
root = Path(args.out)
root.mkdir(parents=True, exist_ok=True)
for rate, privacy in ((0.01, 0.1), (0.03, 0.1), (0.01, 0.3), (0.03, 0.3)):
    cfg = copy.deepcopy(base)
    cfg.update(
        seed=20261008,
        lambda_rate=rate,
        lambda_privacy=privacy,
        output=str(root / f"r{rate}-p{privacy}"),
    )
    file = root / f"r{rate}-p{privacy}.yaml"
    file.write_text(yaml.safe_dump(cfg, sort_keys=False))
    if args.run:
        subprocess.run(
            [sys.executable, "-m", "ccrr.cli", "train", "--config", str(file)],
            check=True,
        )
