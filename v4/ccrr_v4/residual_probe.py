"""CPU-only, zero-update residual protocol probe; never trains a generator."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .residual_transport import (
    CapacityError,
    extract,
    recover_color,
    recover_records,
    send,
)


def read_png(path):
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "L":
            raise ValueError(
                "Require an actual single-channel 8-bit PNG, no conversion"
            )
        return np.asarray(image).copy()


def case(original, public, cfg, output, name):
    kc = AESGCM.generate_key(bit_length=256)
    km = AESGCM.generate_key(bit_length=256)
    records = bytes(
        np.random.default_rng(cfg["seed"]).integers(
            0,
            256,
            cfg["synthetic_record_bytes"],
            dtype=np.uint8,
        )
    )
    bits = cfg["carrier_bits"]
    try:
        transmitted, report = send(original, public, records, kc, km, bits)
    except CapacityError as exc:
        return dict(state="capacity_rejected", **exc.report)
    path = output / f"{name}.png"
    Image.fromarray(transmitted).save(path, format="PNG")
    received = read_png(path)
    base, cc, cm = extract(received, bits)
    recovered = recover_color(base, cc, kc, bits)
    record = recover_records(base, cm, km, bits)
    checks = dict(
        received_png_exact=bool(np.array_equal(received, transmitted)),
        public_base_exact=bool(np.array_equal(base, public)),
        original_encoded_gray_exact=bool(np.array_equal(recovered, original)),
        records_exact=record == records,
        wrong_color_key_rejected=False,
        wrong_record_key_rejected=False,
    )
    try:
        recover_color(base, cc, km, bits)
    except InvalidTag:
        checks["wrong_color_key_rejected"] = True
    try:
        recover_records(base, cm, kc, bits)
    except InvalidTag:
        checks["wrong_record_key_rejected"] = True
    if not all(checks.values()):
        raise RuntimeError(f"Roundtrip failed: {checks}")
    return dict(
        state="engineering_pass",
        **report,
        **checks,
        image=str(path),
        keys="ephemeral_discarded_not_exported",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-png", type=Path)
    parser.add_argument("--public-png", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    required = {
        "schema",
        "seed",
        "image_size",
        "carrier_bits",
        "synthetic_record_bytes",
    }
    if set(cfg) != required or cfg["schema"] != "ccrr-v4-residual-probe-1":
        raise ValueError("Unknown probe configuration")
    if type(cfg["seed"]) is not int or cfg["seed"] < 0:
        raise ValueError("Invalid seed")
    if type(cfg["carrier_bits"]) is not int or not 1 <= cfg["carrier_bits"] <= 4:
        raise ValueError("Invalid carrier_bits")
    if (
        type(cfg["synthetic_record_bytes"]) is not int
        or not 0 <= cfg["synthetic_record_bytes"] <= 1048576
    ):
        raise ValueError("Invalid synthetic_record_bytes")
    size = cfg["image_size"]
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(type(n) is not int or not 32 <= n <= 4096 for n in size)
    ):
        raise ValueError("Invalid synthetic shape")
    if bool(args.original_png) != bool(args.public_png):
        raise ValueError("Provide both original and public PNG, or neither")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(
        schema="ccrr-v4-residual-probe-result-1",
        config=cfg,
        optimizer_updates=0,
        scientific_acceptance="not_evaluated",
        learned_writer=False,
        identity_and_medical_effect="not_evaluated",
        decoder_weights_used=False,
        cases={},
    )
    if args.original_png:
        result["input_kind"] = "user_supplied_pair_not_proof_of_training_provenance"
        result["cases"]["provided"] = case(
            read_png(args.original_png),
            read_png(args.public_png),
            cfg,
            args.output,
            "provided",
        )
    else:
        result["input_kind"] = "synthetic_protocol_fixtures_not_learned_gray"
        y, x = np.indices(size)
        public = ((x // 8 + y // 8) % 48 * 4 + 16).astype(np.uint8)
        original = (public.astype(np.int16) + (x % 3) - 1).astype(np.uint8)
        result["cases"]["compressible"] = case(
            original, public, cfg, args.output, "compressible"
        )
        rng = np.random.default_rng(cfg["seed"])
        public = rng.integers(0, 256, size, dtype=np.uint8)
        original = rng.integers(0, 256, size, dtype=np.uint8)
        result["cases"]["incompressible"] = case(
            original, public, cfg, args.output, "incompressible"
        )
    result["state"] = (
        "completed_with_capacity_rejection"
        if any(c["state"] == "capacity_rejected" for c in result["cases"].values())
        else "completed"
    )
    path = args.output / "report.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"V4_RESIDUAL_REPORT={path}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
