"""Reproducible CPU engineering verification; no downloads or formal training."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="verification/refactor")
    parser.add_argument("--reference-ans")
    parser.add_argument("--reference-sha256")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = (root / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    reference = None
    if args.reference_ans:
        path = Path(args.reference_ans).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if not args.reference_sha256 or digest != args.reference_sha256:
            raise SystemExit("Explicit matching SHA-256 required for native reference")
        env["CCRR_REFERENCE_ANS"] = str(path)
        reference = {"path": str(path), "sha256": digest}
    checks = []

    def check(name, command, expected=0):
        result = subprocess.run(
            command, cwd=root, env=env, text=True, capture_output=True
        )
        (out / (name + ".log")).write_text(result.stdout + result.stderr)
        checks.append(
            {
                "name": name,
                "command": command,
                "exit_code": result.returncode,
                "expected_exit_code": expected,
                "matched": result.returncode == expected,
            }
        )
        print(name, result.returncode, flush=True)
        return result

    py = sys.executable
    check("dependency_consistency", [py, "-m", "pip", "check"])
    check(
        "static_names",
        [py, "-m", "ruff", "check", "src", "tests", "scripts", "--select", "F"],
    )
    check("syntax", [py, "-m", "compileall", "-q", "src", "scripts"])
    check(
        "tests", [py, "-m", "pytest", "-q", "-rx", f"--junitxml={out / 'pytest.xml'}"]
    )
    entry = Path(py).with_name("ccrr")
    check("installed_cli", [str(entry), "--help"])
    check("module_cli", [py, "-m", "ccrr.cli", "--help"])
    check(
        "inspect", [py, "-m", "ccrr.cli", "inspect", "--config", "configs/pad512.yaml"]
    )
    with tempfile.TemporaryDirectory(prefix="ccrr-refactor-") as temp:
        folder = Path(temp)
        cfg = yaml.safe_load((root / "configs/pad512.yaml").read_text())
        cfg["output"] = str(folder / "preflight")
        config = folder / "config.yaml"
        config.write_text(yaml.safe_dump(cfg))
        check(
            "preflight_missing_resources",
            [py, "-m", "ccrr.cli", "preflight", "--config", str(config)],
            2,
        )
        check(
            "smoke512",
            [
                py,
                "-m",
                "ccrr.cli",
                "smoke",
                "--size",
                "512",
                "--out",
                str(folder / "smoke"),
            ],
        )
        smoke_path = folder / "smoke/smoke.json"
        smoke_report = (
            json.loads(smoke_path.read_text()) if smoke_path.exists() else None
        )
        if smoke_report is not None:
            (out / "smoke512.json").write_text(
                json.dumps(smoke_report, indent=2) + "\n"
            )
    check(
        "architecture",
        [py, "scripts/audit_architecture.py", "--out", str(out / "architecture.json")],
    )
    suites = ET.parse(out / "pytest.xml").getroot().findall("testsuite")
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "design": "PAD512-CCRR-v1.1",
        "checks": checks,
        "native_reference": reference,
        "pytest": {
            key: sum(int(s.attrib[key]) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")
        },
        "engineering_checks_matched_expected": all(c["matched"] for c in checks),
        "expected_design_failures": [],
        "repaired_design_regressions": ["D01", "D03", "D04", "D11", "D12"],
        "complete_training_passed": False,
        "scientific_targets_validated": False,
        "real_teacher_weights_validated": False,
        "H100_validated": False,
        "note": "D01/D03/D04 are required regressions, not xfails. Analytical stage fixtures and dispatch checks are not teacher or training acceptance evidence.",
    }
    (out / "verification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    if not report["engineering_checks_matched_expected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
