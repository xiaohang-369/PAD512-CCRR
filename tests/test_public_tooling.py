"""Public architecture checks must work without internal files or Git history."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def checkout(tmp_path):
    root = Path(__file__).resolve().parents[1]
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(root / "scripts/audit_architecture.py", scripts)
    package = tmp_path / "src/ccrr"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "example.py").write_text("def value():\n    return 1\n")
    return tmp_path


def audit(checkout, *args):
    return subprocess.run(
        [
            sys.executable,
            "scripts/audit_architecture.py",
            "--out",
            "result.json",
            *args,
        ],
        cwd=checkout,
        text=True,
        capture_output=True,
    )


def test_architecture_without_git_or_internal_evidence(checkout):
    result = audit(checkout)
    assert result.returncode == 0, result.stderr
    report = json.loads((checkout / "result.json").read_text())
    assert report["baseline_commit"] is None
    assert report["comparison_performed"] is False
    assert "before" not in report
    assert report["after"]["python_files"] == 2
    assert report["after"]["import_cycles_including_lazy"] == []


def test_explicit_missing_baseline_is_failure(checkout):
    result = audit(checkout, "--baseline", "missing-baseline")
    assert result.returncode != 0
    assert not (checkout / "result.json").exists()


def test_explicit_baseline_compares_real_git_source(checkout):
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(["git", "add", "src"], cwd=checkout, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "Fixture baseline",
        ],
        cwd=checkout,
        check=True,
    )
    before = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    (checkout / "src/ccrr/example.py").write_text("def value():\n    return 2\n")
    result = audit(checkout, "--baseline", before)
    assert result.returncode == 0, result.stderr
    report = json.loads((checkout / "result.json").read_text())
    assert report["comparison_performed"] is True
    assert report["baseline_commit"] == before
    path = "src/ccrr/example.py"
    assert (
        report["before"]["files"][path]["sha256"]
        != report["after"]["files"][path]["sha256"]
    )
