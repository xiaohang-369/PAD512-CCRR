#!/usr/bin/env python3
"""Compare mirrored tensorflow-vgg conv tensors to the original Oxford Caffe file.

Protobuf field numbers are from BVLC/caffe src/caffe/proto/caffe.proto:
NetParameter.layers=2, V1LayerParameter.name=4/blobs=6, BlobProto.data=5.
Read-only, no Caffe install, no executable model deserialization. Emits a safe,
feature-only NPZ containing the exact nine conv layers used by the author.
"""

import argparse
import hashlib
import json
import mmap
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccrr_v4.phase1.vgg import read_author_npy, LAYERS  # noqa: E402
from ccrr_v4.verify import digest  # noqa: E402


def varint(buf, pos):
    value = 0
    for shift in range(0, 70, 7):
        byte = buf[pos]
        pos += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, pos
    raise ValueError("Invalid protobuf varint")


def fields(buf):
    pos = 0
    while pos < len(buf):
        key, pos = varint(buf, pos)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = varint(buf, pos)
        elif wire in (1, 2, 5):
            if wire == 2:
                length, pos = varint(buf, pos)
            else:
                length = 8 if wire == 1 else 4
            if pos + length > len(buf):
                raise ValueError("Truncated protobuf")
            value = buf[pos : pos + length]
            pos += length
        else:
            raise ValueError(f"Unsupported protobuf wire {wire}")
        yield number, value


def compare(npy, caffe, output):
    output.mkdir(parents=True, exist_ok=False)
    data = read_author_npy(npy, digest(npy))
    records, arrays = {}, {}
    with caffe.open("rb") as f:
        mapped = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        for number, layer in fields(mapped):
            if number != 2:
                continue
            parts = list(fields(layer))
            name = bytes(next(v for k, v in parts if k == 4)).decode()
            if name not in {n for n, _, _ in LAYERS}:
                continue
            blobs = [dict(fields(v)) for k, v in parts if k == 6]
            w, b = data[name]
            cw = np.frombuffer(blobs[0][5], dtype="<f4").reshape(
                tuple(blobs[0][k] for k in (1, 2, 3, 4))
            )
            cb = np.frombuffer(blobs[1][5], dtype="<f4")
            if not np.array_equal(cw.transpose(2, 3, 1, 0), w) or not np.array_equal(
                cb, b
            ):
                raise ValueError(f"Caffe/NPY parameter mismatch: {name}")
            arrays[name + ".weight"], arrays[name + ".bias"] = w, b
            records[name] = dict(
                weight_shape=list(w.shape),
                bias_shape=list(b.shape),
                exact_equal=True,
                weight_sha256=hashlib.sha256(w.tobytes()).hexdigest(),
                bias_sha256=hashlib.sha256(b.tobytes()).hexdigest(),
            )
        mapped.close()
    if len(records) != len(LAYERS):
        raise ValueError("Incomplete official comparison")
    np.savez(output / "vgg19-conv4_1.npz", **arrays)
    report = dict(
        schema="ccrr-v4-vgg-origin-1",
        npy_sha256=digest(npy),
        npy_source="https://www.dropbox.com/s/691wtp4oq5ip38p/vgg19.npy?dl=1",
        mirror_reference="https://github.com/rikardocorp/tensorflow_vgg_train",
        original_mega_file_byte_identity="unavailable_original_link_returned_minus16",
        caffe_source="https://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel",
        caffe_reference="https://gist.github.com/ksimonyan/3785162f95cd2d5fee77",
        caffe_sha256=digest(caffe),
        exact_conv_layers_compared=len(records),
        layers=records,
        feature_file="vgg19-conv4_1.npz",
        feature_sha256=digest(output / "vgg19-conv4_1.npz"),
        feature_bytes=(output / "vgg19-conv4_1.npz").stat().st_size,
    )
    (output / "vgg-provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "layers"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npy", type=Path, required=True)
    parser.add_argument("--caffe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    compare(a.npy, a.caffe, a.output)
