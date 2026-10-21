"""V4 integer residual transport experiment; no learned generator or Writer.

The diagnostic carrier replaces b low bitplanes and carries their compressed
original values IN the same image. This is NOT the proposed semantic low-bit
split: the protected object is the full signed difference of two encoded grays.
Incompressible bases/payloads are explicitly rejected; capacity is not assumed.
Only lossless grayscale PNG reception is supported. Keys never enter the frame.
"""

import hashlib
import os
import struct
import zlib

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


MAGIC = b"V4RDI001"
HEADER = struct.Struct(">8sIIBIII32s")
MAX_PIXELS = 4096 * 4096


class CapacityError(ValueError):
    def __init__(self, report):
        super().__init__("Complete authenticated payload and base recovery do not fit")
        self.report = report


def gray(image):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 2 or not image.size:
        raise ValueError("Require nonempty HxW uint8 grayscale")
    if image.size > MAX_PIXELS:
        raise ValueError("Diagnostic image exceeds pixel bound")
    return np.ascontiguousarray(image)


def residual(original, public):
    original, public = gray(original), gray(public)
    if original.shape != public.shape:
        raise ValueError("Original and public grayscale shapes differ")
    return original.astype(np.int16) - public.astype(np.int16)


def recompose(public, delta):
    public = gray(public)
    if delta.shape != public.shape or delta.dtype.kind != "i":
        raise ValueError("Require matching signed integer residual")
    if np.any((delta < -255) | (delta > 255)):
        raise ValueError("Residual out of range")
    value = public.astype(np.int32) + delta.astype(np.int32)
    if np.any((value < 0) | (value > 255)):
        raise ValueError("Invalid reconstruction; do not clip or wrap")
    return value.astype(np.uint8)


def inflate(data, expected):
    decoder = zlib.decompressobj()
    result = decoder.decompress(data, expected + 1)
    if (
        len(result) != expected
        or not decoder.eof
        or decoder.unused_data
        or decoder.unconsumed_tail
    ):
        raise ValueError("Invalid bounded compressed stream")
    return result


def profile(image, bits):
    image = gray(image)
    if type(bits) is not int or not 1 <= bits <= 4:
        raise ValueError("Diagnostic bits must be an integer in [1,4]")
    return image


def associated(public, bits):
    return (
        MAGIC
        + struct.pack(">IIB", *public.shape, bits)
        + hashlib.sha256(public.tobytes()).digest()
    )


def seal(plaintext, key, aad):
    if not isinstance(key, bytes) or len(key) != 32:
        raise ValueError("Require a private 256-bit key")
    nonce = os.urandom(12)
    return nonce + AESGCM(key).encrypt(nonce, plaintext, aad)


def open_sealed(ciphertext, key, aad):
    if not isinstance(key, bytes) or len(key) != 32 or len(ciphertext) < 28:
        raise ValueError("Invalid key or sealed message length")
    return AESGCM(key).decrypt(ciphertext[:12], ciphertext[12:], aad)


def send(original, public, records, kc, km, bits=2):
    """Return the only transmitted image and a non-sensitive length report.

    Ephemeral random nonces are generated per call. The caller manages keys;
    this diagnostic is not a persistent production nonce/key-management system.
    """
    public = profile(public, bits)
    delta = residual(original, public)
    if not isinstance(records, bytes):
        raise ValueError("Records must already be serialized bytes")
    if len(records) > MAX_PIXELS:
        raise ValueError("Diagnostic records exceed length bound")
    if kc == km:
        raise ValueError("Use independent color and record keys")
    packed_delta = zlib.compress(delta.astype("<i2").tobytes(), level=9)
    aad = associated(public, bits)
    cc = seal(packed_delta, kc, aad + b"/color")
    cm = seal(records, km, aad + b"/records")
    mask = (1 << bits) - 1
    low = public & mask
    auxiliary = zlib.compress(low.tobytes(), level=9)
    header = HEADER.pack(
        MAGIC,
        *public.shape,
        bits,
        len(auxiliary),
        len(cc),
        len(cm),
        hashlib.sha256(public.tobytes()).digest(),
    )
    frame = header + auxiliary + cc + cm
    report = dict(
        pixels=public.size,
        bitplanes=bits,
        raw_residual_bytes=delta.nbytes,
        compressed_residual_bytes=len(packed_delta),
        color_ciphertext_bytes=len(cc),
        record_ciphertext_bytes=len(cm),
        base_recovery_bytes=len(auxiliary),
        header_bytes=len(header),
        frame_bytes=len(frame),
        gross_carrier_bits=public.size * bits,
        spare_bits=public.size * bits - len(frame) * 8,
        max_allowed_pixel_change=mask,
        carrier="diagnostic_reversible_bitplanes_not_learned_writer",
    )
    if report["spare_bits"] < 0:
        raise CapacityError(report)
    # Serialize each pixel's low planes MSB first; unused slots retain originals.
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint8)
    stream = ((low.reshape(-1, 1) >> shifts) & 1).reshape(-1)
    frame_bits = np.unpackbits(np.frombuffer(frame, dtype=np.uint8))
    stream[: len(frame_bits)] = frame_bits
    replacement = (stream.reshape(-1, bits) << shifts).sum(1).astype(np.uint8)
    transmitted = (public & (255 ^ mask)) | replacement.reshape(public.shape)
    return transmitted, report


def extract(transmitted, bits=2):
    """Recover public base and ciphertexts using only the received image/profile."""
    transmitted = profile(transmitted, bits)
    mask = (1 << bits) - 1
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint8)
    stream = ((transmitted.reshape(-1, 1) >> shifts) & 1).reshape(-1)
    data = np.packbits(stream[: len(stream) // 8 * 8]).tobytes()
    if len(data) < HEADER.size:
        raise ValueError("Carrier too small for header")
    magic, h, w, stored_bits, na, nc, nm, expected_hash = HEADER.unpack_from(data)
    if magic != MAGIC or (h, w) != transmitted.shape or stored_bits != bits:
        raise ValueError("Invalid carrier header/profile")
    end = HEADER.size + na + nc + nm
    if na == 0 or nc < 28 or nm < 28 or end > len(data):
        raise ValueError("Invalid frame lengths")
    raw_low = inflate(data[HEADER.size : HEADER.size + na], transmitted.size)
    low = np.frombuffer(raw_low, dtype=np.uint8).reshape(transmitted.shape)
    if np.any(low > mask):
        raise ValueError("Invalid restored low planes")
    public = (transmitted & (255 ^ mask)) | low
    if hashlib.sha256(public.tobytes()).digest() != expected_hash:
        raise ValueError("Restored base checksum mismatch")
    start = HEADER.size + na
    return public, data[start : start + nc], data[start + nc : end]


def recover_color(public, ciphertext, kc, bits=2):
    public = profile(public, bits)
    compressed = open_sealed(ciphertext, kc, associated(public, bits) + b"/color")
    raw = inflate(compressed, public.size * 2)
    delta = np.frombuffer(raw, dtype="<i2").reshape(public.shape)
    return recompose(public, delta)


def recover_records(public, ciphertext, km, bits=2):
    public = profile(public, bits)
    return open_sealed(ciphertext, km, associated(public, bits) + b"/records")
