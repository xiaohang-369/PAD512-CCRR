"""The convenience launcher must never train after failed/wrong preflight."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest


@pytest.mark.parametrize("outcome", ["failed", "joint", "e1only"])
def test_start_script_gates_background_training_on_e1only_report(tmp_path, outcome):
    package = tmp_path / "package with spaces"
    package.mkdir()
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "scripts/start-g1-e1only.sh",
        package / "start-training.sh",
    )
    report = dict(
        passed=True,
        mode="preflight",
        optimizer_updates=0,
        update_modules=["E1"] if outcome == "e1only" else ["E1", "D1"],
        reconstruction_loss_enabled=True,
        binding={
            "config": {"schema": "ccrr-v4-phase2-g1-e1only-1"},
            "decoder_parameters_frozen": outcome == "e1only",
        },
    )
    (package / "report.json").write_text(json.dumps(report))
    (package / "phase2.sh").write_text("""#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == preflight ]]; then
    if [[ "$OUTCOME" == failed ]]; then exit 7; fi
    printf 'G1_REPORT=%s/report.json\n' "$MOCK_PACKAGE"
else
    test "$1" = train
    test "$2" = --preflight
    test "$3" = "$MOCK_PACKAGE/report.json"
    printf 'launched\n' > "$MOCK_PACKAGE/started"
fi
""")
    result = subprocess.run(
        ["bash", str(package / "start-training.sh")],
        env={
            **os.environ,
            "V4_ROOT": str(tmp_path / "isolated root"),
            "V4_PYTHON": sys.executable,
            "MOCK_PACKAGE": str(package),
            "OUTCOME": outcome,
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    if outcome == "e1only":
        assert result.returncode == 0, result.stderr
        deadline = time.monotonic() + 5
        while not (package / "started").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (package / "started").read_text() == "launched\n"
    else:
        assert result.returncode != 0
        assert not (package / "started").exists()
