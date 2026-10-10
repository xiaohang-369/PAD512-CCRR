"""Dynamic-source import mechanics only; no substitute for real MFM weights."""

import sys
import pytest
from ccrr.config import sha256
from ccrr.teachers.adapters import load_panderm_source
from ccrr.errors import CCRRError


def test_source_is_registered_before_execution(tmp_path):
    source = tmp_path / "registration.py"
    source.write_text(
        "import sys\n"
        "registered = sys.modules[__name__]\n"
        "assert registered.__dict__ is globals()\n"
    )
    module = load_panderm_source(source, sha256(source))
    assert module.registered is module
    assert sys.modules[module.__name__] is module


def test_import_failure_removes_partial_module(tmp_path):
    source = tmp_path / "broken.py"
    source.write_text("raise RuntimeError('source failed')\n")
    digest = sha256(source)
    name = "ccrr_official_panderm_" + digest
    before = dict(sys.modules)
    with pytest.raises(RuntimeError, match="source failed"):
        load_panderm_source(source, digest)
    assert name not in sys.modules
    assert all(sys.modules[k] is v for k, v in before.items())


def test_source_digest_is_checked_before_executing(tmp_path):
    source = tmp_path / "untrusted.py"
    marker = tmp_path / "executed"
    source.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    with pytest.raises(CCRRError, match="source hash"):
        load_panderm_source(source, "0" * 64)
    assert not marker.exists()
