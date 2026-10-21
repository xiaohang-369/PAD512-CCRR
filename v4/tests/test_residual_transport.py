from io import BytesIO

import numpy as np
import pytest
from PIL import Image
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ccrr_v4.residual_transport import (
    CapacityError,
    HEADER,
    extract,
    inflate,
    recompose,
    recover_color,
    recover_records,
    residual,
    send,
)


def pair():
    y, x = np.indices((128, 128))
    public = ((x // 8 + y // 8) % 48 * 4 + 16).astype(np.uint8)
    original = (public.astype(np.int16) + (x % 3) - 1).astype(np.uint8)
    return original, public


def keys():
    return AESGCM.generate_key(bit_length=256), AESGCM.generate_key(bit_length=256)


def png(image):
    stream = BytesIO()
    Image.fromarray(image).save(stream, format="PNG")
    stream.seek(0)
    return np.asarray(Image.open(stream)).copy()


def test_all_65536_pixel_pairs_without_wrapping():
    a, b = np.indices((256, 256)).astype(np.uint8)
    delta = residual(a, b)
    assert delta.min() == -255 and delta.max() == 255
    assert np.array_equal(recompose(b, delta), a)
    with pytest.raises(ValueError, match="clip or wrap"):
        recompose(np.zeros((1, 1), np.uint8), np.array([[-1]], np.int16))


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_real_png_independent_key_paths_and_original_base(bits):
    original, public = pair()
    kc, km = keys()
    records = "synthetic record, no patient information".encode()
    transmitted, report = send(original, public, records, kc, km, bits)
    # Receiver receives no sender residual, original base, or original image.
    restored, cc, cm = extract(png(transmitted), bits)
    assert np.array_equal(restored, public)
    assert np.array_equal(recover_color(restored, cc, kc, bits), original)
    assert recover_records(restored, cm, km, bits) == records
    assert report["frame_bytes"] == (
        HEADER.size + report["base_recovery_bytes"] + len(cc) + len(cm)
    )
    assert abs(transmitted.astype(int) - public.astype(int)).max() <= (1 << bits) - 1
    with pytest.raises(InvalidTag):
        recover_color(restored, cc, km, bits)
    with pytest.raises(InvalidTag):
        recover_records(restored, cm, kc, bits)


def test_authentication_binds_base_route_and_ciphertext():
    original, public = pair()
    kc, km = keys()
    transmitted, _ = send(original, public, b"record", kc, km)
    restored, cc, cm = extract(transmitted)
    changed = restored.copy()
    changed[0, 0] ^= 4
    with pytest.raises(InvalidTag):
        recover_color(changed, cc, kc)
    with pytest.raises(InvalidTag):
        recover_records(restored, cc, kc)  # right key, wrong route
    damaged = bytearray(cc)
    damaged[-1] ^= 1
    with pytest.raises(InvalidTag):
        recover_color(restored, bytes(damaged), kc)
    assert recover_records(restored, cm, km) == b"record"


def test_capacity_failure_keeps_input_unchanged():
    rng = np.random.default_rng(11)
    public = rng.integers(0, 256, (128, 128), dtype=np.uint8)
    original = rng.integers(0, 256, public.shape, dtype=np.uint8)
    before = public.copy()
    with pytest.raises(CapacityError) as caught:
        send(original, public, b"x", *keys())
    assert caught.value.report["spare_bits"] < 0
    assert np.array_equal(before, public)


def test_bad_profile_header_and_upper_pixels_are_rejected():
    original, public = pair()
    transmitted, _ = send(original, public, b"x", *keys())
    with pytest.raises(ValueError):
        extract(transmitted, 1)
    bad = transmitted.copy()
    bad[0, 0] ^= 1
    with pytest.raises(ValueError):
        extract(bad)
    bad = transmitted.copy()
    bad[-1, -1] ^= 128
    with pytest.raises(ValueError, match="checksum"):
        extract(bad)


def test_no_truncation_or_decompression_expansion():
    import zlib

    packed = zlib.compress(b"a" * 1000)
    with pytest.raises(ValueError):
        inflate(packed, 5)
    with pytest.raises(ValueError):
        inflate(packed[:-1], 1000)
    with pytest.raises(ValueError):
        inflate(packed + b"trailing", 1000)
    assert inflate(packed, 1000) == b"a" * 1000


def test_nonce_freshness_and_key_separation():
    original, public = pair()
    kc, km = keys()
    a, _ = send(original, public, b"x", kc, km)
    b, _ = send(original, public, b"x", kc, km)
    assert extract(a)[1][:12] != extract(b)[1][:12]
    with pytest.raises(ValueError, match="independent"):
        send(original, public, b"x", kc, kc)
