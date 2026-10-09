"""Reproduce engineering checks; never starts training or writes patient data."""

from pathlib import Path
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

root = Path(__file__).resolve().parents[1]
out = root / "verification"
out.mkdir(exist_ok=True)
checks = []
for name, command in (
    ("dependency_consistency", [sys.executable, "-m", "pip", "check"]),
    (
        "static_names",
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "src",
            "tests",
            "scripts",
            "--select",
            "F",
        ],
    ),
    ("syntax", [sys.executable, "-m", "compileall", "-q", "src", "scripts"]),
    (
        "tests",
        [
            sys.executable,
            "-m",
            "pytest",
            "tests",
            "-q",
            f"--junitxml={out / 'pytest.xml'}",
        ],
    ),
):
    p = subprocess.run(command, cwd=root, text=True, capture_output=True)
    checks.append(
        {"name": name, "exit_code": p.returncode, "output": p.stdout + p.stderr}
    )
    print(name, p.returncode)
    if p.returncode:
        raise SystemExit(p.stdout + p.stderr)
files = {
    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(root.rglob("*"))
    if p.is_file()
    and p.suffix in (".py", ".toml", ".yaml", ".md", ".lock", ".json")
    and "verification" not in p.parts
    and "__pycache__" not in p.parts
    and ".egg-info" not in str(p)
}
xml = ET.parse(out / "pytest.xml").getroot()
suites = xml.findall("testsuite")
report = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "baseline": "PAD512-CCRR-v1.1",
    "scope": "First research implementation engineering verification",
    "source_files": files,
    "checks": checks,
    "tests": sum(int(s.attrib["tests"]) for s in suites),
    "failures": sum(
        int(s.attrib["failures"]) + int(s.attrib["errors"]) for s in suites
    ),
    "skipped": sum(int(s.attrib["skipped"]) for s in suites),
    "complete_medical_training_executed": False,
    "teacher_capability_validated": False,
    "H100_validated": False,
    "implementation_scope": [
        "INN and fixed-gray baseline",
        "teacher wrappers and Purifier",
        "independent AEAD and persistent nonces",
        "bounded rANS and source quantization",
        "neural carrier encoder, writer, decoder and heads",
        "training and calibration orchestration",
        "adaptive attacks, clinical evaluation and denominators",
        "send/receive/export CLI",
    ],
    "limitations": [
        "Full training orchestration has not been exercised with real trained teachers",
        "PanDerm checkpoint bytes and capability not available",
        "H100 memory, throughput and cross-host numerical compatibility not checked",
        "Exploratory metrics and cross-specialty data require separately audited resources",
    ],
    "external_evidence": {},
}
external = Path("/Users/zhuyanghang/Desktop/CCRR-experiments")
for key, p in [
    ("data_adapter", external / "data-adapter-audit.json"),
    ("smoke512", external / "engineering-smoke-512/smoke.json"),
    ("codec_reference", external / "codec-reference/reference.json"),
]:
    if p.exists():
        report["external_evidence"][key] = {
            "path": str(p),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "result": json.loads(p.read_text()),
        }
(out / "implementation_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n"
)
print(
    json.dumps(
        {
            k: report[k]
            for k in (
                "tests",
                "failures",
                "skipped",
                "complete_medical_training_executed",
            )
        },
        indent=2,
    )
)
