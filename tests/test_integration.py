import importlib.util
import importlib.metadata
import dataclasses
from pathlib import Path
import os
import numpy as np
import pytest
import torch
from ccrr.config import Profile, export_manifest, load_manifest, sha256
from ccrr.runtime.system import CCRR, save_public, load_public
from ccrr.codec.crypto import NonceAllocator, seal
from ccrr.communication.records import split_stream
from ccrr.codec.metadata import serialize, FIELDS
from ccrr.errors import Status, CCRRError
from ccrr.codec.entropy import EntropyModel
from ccrr.codec import rans
from ccrr.evaluation.metrics import failure_funnel, quality
from ccrr.calibration.probes import labels, wilson
from ccrr.cli import main


def fixture_system(tmp_path):
    p = Profile(size=256)
    s = CCRR(p)
    digest = os.urandom(32)
    cdf = np.tile(np.arange(513, dtype=np.int32), (3, 128, 1))
    cdf[:, :, 256:] += 65024
    s.bind(digest, cdf)
    a = NonceAllocator(tmp_path / "nonce.sqlite")
    kc, km = a.new_object(digest)
    n = torch.zeros(1, 2, 256, 256)
    source = s.codec.encode(n, 1)
    c = seal(source, kc, 1, a)
    row = dict.fromkeys(FIELDS, "")
    row.update(
        patient_id="SYNTHETIC", lesion_id="1", img_id="test.png", diagnostic="ACK"
    )
    m = seal(serialize(row), km, 0, a)
    records = split_stream(c, 0, 1) + split_stream(m, 1, 0)
    return s, a, kc, km, records, row


def test_full_digital_capability_matrix_and_malformed_isolation(tmp_path):
    s, a, kc, km, records, row = fixture_system(tmp_path)
    g = torch.full((1, 1, 256, 256), 128 / 255)
    for creds, expected in [
        ({}, (Status.NO_CREDENTIAL, Status.NO_CREDENTIAL)),
        ({0: kc}, (Status.SUCCESS, Status.NO_CREDENTIAL)),
        ({1: km}, (Status.NO_CREDENTIAL, Status.SUCCESS)),
        ({0: kc, 1: km}, (Status.SUCCESS, Status.SUCCESS)),
    ]:
        with torch.no_grad():
            r = s.receive_records(g, records, {0: [], 1: []}, creds)
        assert (r["C"].status, r["M"].status) == expected
        if r["M"].status == Status.SUCCESS:
            assert r["M"].value == row
    wrong = dataclasses.replace(kc, key=os.urandom(32))
    r = s.receive_records(g, records, {0: [], 1: []}, {0: wrong, 1: km})
    assert r["C"].status == Status.AUTH and r["M"].status == Status.SUCCESS
    r = s.receive_records(g, records, {0: ["invalid header"], 1: []}, {0: kc, 1: km})
    assert r["C"].status == Status.EXTRACTION and r["M"].status == Status.SUCCESS
    # Identical duplicate records must fail; M remains independently decodable.
    r = s.receive_records(g, records + [records[0]], {0: [], 1: []}, {0: kc, 1: km})
    assert r["C"].status == Status.EXTRACTION and r["M"].status == Status.SUCCESS
    a.close()


def test_source_profile_and_tail_rejection(tmp_path):
    s, a, *_ = fixture_system(tmp_path)
    n = torch.zeros(1, 2, 256, 256)
    n[0, 0, 0, :7] = torch.tensor([-1048575, -256, -255, 0, 255, 256, 1048575])
    blob = s.codec.encode(n, 2)
    decoded, z = s.codec.decode(blob, 2)
    assert torch.equal(decoded.float(), n) and torch.equal(z, n * s.profile.step(2))
    for malformed in (
        blob[:-4],
        blob + bytes(4),
        blob[:16] + os.urandom(32) + blob[48:],
    ):
        with pytest.raises(CCRRError):
            s.codec.decode(malformed, 2)
    a.close()


def test_native_compressai_byte_compatibility():
    configured = os.environ.get("CCRR_REFERENCE_ANS")
    if configured:
        libraries = [Path(configured)]
    else:
        try:
            d = importlib.metadata.distribution("compressai")
        except importlib.metadata.PackageNotFoundError:
            pytest.skip("Optional isolated official codec reference not configured")
        root = Path(d.locate_file("compressai"))
        libraries = list(root.glob("ans*.so"))
    if not libraries:
        pytest.skip("Native reference unavailable on this platform")
    spec = importlib.util.spec_from_file_location("ans", libraries[0])
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    cdfs = EntropyModel().export()[0]
    rng = np.random.default_rng(42)
    for symbols in (
        [0] * 4096,
        [-1048575, -256, -255, 0, 255, 256, 1048575],
        rng.integers(-1000, 1000, 4096).tolist(),
    ):
        ix = rng.integers(0, 128, len(symbols)).tolist()
        enc = native.BufferedRansEncoder()
        enc.encode_with_indexes(symbols, ix, cdfs.tolist(), [513] * 128, [-255] * 128)
        official = enc.flush()
        ours = rans.encode(symbols, ix, cdfs)
        assert ours == official
        assert rans.decode(official, ix, cdfs).tolist() == symbols
        assert (
            native.RansDecoder().decode_with_indexes(
                ours, ix, cdfs.tolist(), [513] * 128, [-255] * 128
            )
            == symbols
        )


def test_entropy_rate_gradient_not_zero():
    m = EntropyModel()
    z = torch.randn(1, 2, 16, 16, requires_grad=True) * 0.01
    z.retain_grad()
    rate = m.rate(z, 1, 1 / 32).mean()
    rate.backward()
    assert torch.isfinite(rate) and z.grad.abs().sum() > 0 and m.b.grad.abs().sum() > 0


def test_public_png_has_no_private_side_data(tmp_path):
    g = torch.randint(0, 256, (1, 1, 256, 256)).float() / 255
    save_public(g, tmp_path / "g.png")
    assert torch.equal(load_public(tmp_path / "g.png", 256), g)
    from PIL import Image, PngImagePlugin

    im = Image.open(tmp_path / "g.png")
    info = PngImagePlugin.PngInfo()
    info.add_text("latent", "not allowed")
    im.save(tmp_path / "bad.png", pnginfo=info)
    with pytest.raises(CCRRError):
        load_public(tmp_path / "bad.png", 256)


def test_funnel_never_drops_auth_failure_from_public_metrics():
    rows = [
        {"id": "capacity", "published": False, "failure": "capacity"},
        {"id": "auth", "published": True, "c_success": False, "m_success": False},
        {
            "id": "success",
            "published": True,
            "c_success": True,
            "m_success": True,
            "both_success": True,
        },
    ]
    f = failure_funnel(rows)
    assert f["public_utility_privacy_ids"] == ["auth", "success"]
    assert f["authorized_quality_ids"] == ["success"]
    assert f["all_input_rates"]["both_success"] == 1 / 3


def test_exact_quality_and_padding_exclusion():
    x = torch.rand(1, 3, 32, 32)
    v = torch.ones(1, 1, 32, 32)
    v[:, :, :, 0:12] = 0
    y = x.clone()
    y[:, :, :, 0:12] = 0
    q = quality(x, y, v)[0]
    assert q["psnr"] == float("inf") and q["ssim"] == pytest.approx(1)


def test_calibration_must_pass_every_lower_load_and_empty_table():
    def perfect():
        return {
            "success": [True] * 64,
            "maximum": [0] * 64,
            "rmse": [0] * 64,
            "js": [0] * 32,
        }

    p = {k: perfect() for k in range(5)}
    assert labels(p)["kstar"] == 4
    p[2]["success"][0] = False
    assert labels(p)["kstar"] == 1
    p[0]["success"][0] = False
    assert labels(p)["kstar"] == 0
    assert 0.98 < wilson(256, 256) < 0.99


def test_profile_binding_rejects_resource_modification(tmp_path):
    resource = tmp_path / "system.pt"
    resource.write_bytes(b"fixture")
    export_manifest(
        tmp_path / "manifest.json",
        Profile(size=256),
        {"system": {"path": "system.pt", "sha256": sha256(resource)}},
        {"engineering_fixture": True},
    )
    profile, digest, obj = load_manifest(tmp_path / "manifest.json")
    assert profile.size == 256 and len(digest) == 32
    resource.write_bytes(b"changed")
    with pytest.raises(CCRRError):
        load_manifest(tmp_path / "manifest.json")


def test_all_modules_import_and_cli_inspection():
    import importlib

    for name in (
        "training.legacy",
        "training.final_study",
        "training.feature_audit",
        "training.attribute_study",
        "evaluation.optional_metrics",
        "evaluation.robustness",
        "data.external",
        "calibration.routing",
        "evaluation.structural_attacks",
    ):
        assert importlib.import_module("ccrr." + name)
    assert main(["inspect"]) is None


def test_official_frozen_golden_vectors_without_native_dependency():
    import json

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/compressai_1_2_8_vectors.json").read_text()
    )
    cdf = np.tile(np.array(fixture["cdf_row"], dtype=np.int32), (128, 1))
    for case in fixture["cases"]:
        stream = bytes.fromhex(case["encoded_hex"])
        assert rans.encode(case["symbols"], case["indexes"], cdf) == stream
        assert rans.decode(stream, case["indexes"], cdf).tolist() == case["symbols"]
