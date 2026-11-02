import hashlib
import importlib.util
from pathlib import Path


def test_asset_gate_rejects_missing_and_changed_files(tmp_path):
    path=Path(__file__).resolve().parents[1]/'scripts/check_assets.py'
    spec=importlib.util.spec_from_file_location('asset_check',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    p=tmp_path/'asset';p.write_bytes(b'original')
    bindings={str(p):hashlib.sha256(b'original').hexdigest()}
    assert mod.check(bindings)['passed']
    p.write_bytes(b'changed');assert not mod.check(bindings)['passed']
    p.unlink();assert mod.check(bindings)['errors'][0]['error']=='missing'
