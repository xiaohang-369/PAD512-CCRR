from dataclasses import dataclass
import math
import numpy as np
import torch
from .errors import require, Status


def int_bits(value, width):
    require(0 <= int(value) < 2**width, "Bit field overflow", Status.EXTRACTION)
    return [(int(value) >> i) & 1 for i in reversed(range(width))]


def bits_int(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


@dataclass(frozen=True)
class Record:
    alpha: int
    level: int
    index: int
    count: int
    length: int
    data: bytes
    version: int = 1

    def validate(self):
        require(
            self.version == 1 and self.alpha in (0, 1),
            "Record version/type",
            Status.EXTRACTION,
        )
        require(
            (self.alpha == 0 and self.level in (1, 2, 3))
            or (self.alpha == 1 and self.level == 0),
            "Record level",
            Status.EXTRACTION,
        )
        require(
            1 <= self.index <= self.count <= 1024
            and 0 < self.length <= 131072
            and self.length % 8 == 0,
            "Record range",
            Status.EXTRACTION,
        )
        require(
            self.count == math.ceil(self.length / 128) and len(self.data) == 16,
            "Record count/data",
            Status.EXTRACTION,
        )

    def bits(self):
        self.validate()
        return np.array(
            int_bits(self.version, 2)
            + int_bits(self.alpha, 1)
            + int_bits(self.level, 2)
            + int_bits(self.index - 1, 10)
            + int_bits(self.count - 1, 10)
            + int_bits(self.length, 18)
            + np.unpackbits(np.frombuffer(self.data, dtype=np.uint8)).tolist(),
            dtype=np.float32,
        )

    @classmethod
    def from_bits(cls, bits):
        require(len(bits) == 171, "Record bit count", Status.EXTRACTION)
        b = np.asarray(bits)
        require(np.all((b == 0) | (b == 1)), "Nonbinary record", Status.EXTRACTION)
        r = cls(
            bits_int(b[2:3]),
            bits_int(b[3:5]),
            bits_int(b[5:15]) + 1,
            bits_int(b[15:25]) + 1,
            bits_int(b[25:43]),
            np.packbits(b[43:].astype(np.uint8)).tobytes(),
            bits_int(b[:2]),
        )
        r.validate()
        return r


def split_stream(stream, alpha, level):
    require(0 < len(stream) <= 16384, "Stream size", Status.CAPACITY)
    n = math.ceil(len(stream) / 16)
    return [
        Record(
            alpha,
            level,
            j + 1,
            n,
            len(stream) * 8,
            stream[16 * j : 16 * (j + 1)].ljust(16, b"\x00"),
        )
        for j in range(n)
    ]


def reassemble(records, alpha):
    rows = [r for r in records if r.alpha == alpha]
    require(bool(rows), "Missing stream", Status.EXTRACTION)
    ref = rows[0]
    for r in rows:
        r.validate()
        require(
            (r.version, r.level, r.count, r.length)
            == (ref.version, ref.level, ref.count, ref.length),
            "Conflicting stream headers",
            Status.EXTRACTION,
        )
    require(
        len(rows) == ref.count
        and {r.index for r in rows} == set(range(1, ref.count + 1)),
        "Duplicate/missing record",
        Status.EXTRACTION,
    )
    raw = b"".join(r.data for r in sorted(rows, key=lambda r: r.index))
    size = ref.length // 8
    require(not any(raw[size:]), "Nonzero stream padding", Status.EXTRACTION)
    return raw[:size], ref.level


def route(c_records, m_records, budgets, reliability, sorted_policy=False):
    k = np.asarray(budgets)
    u = np.asarray(reliability)
    require(
        k.ndim == 1 and k.shape == u.shape and np.isfinite(u).all(), "Routing shape"
    )
    require(np.all((k >= 0) & (k <= 4) & (k == k.astype(int))), "Invalid budget")
    require(
        len(c_records) + len(m_records) <= int(k.sum()),
        "Complete payload does not fit",
        Status.CAPACITY,
    )
    order = (
        sorted(range(len(k)), key=lambda i: (-float(u[i]), i))
        if sorted_policy
        else range(len(k))
    )
    queue = list(m_records) + list(c_records)
    table = np.zeros((len(k), 4, 171), np.float32)
    occ = np.zeros((len(k), 4), np.float32)
    offset = 0
    for i in order:
        rows = queue[offset : offset + int(k[i])]
        offset += len(rows)
        rows.sort(key=lambda r: (r.alpha, r.index))
        for j, r in enumerate(rows):
            table[i, j] = r.bits()
            occ[i, j] = 1
    require(offset == len(queue), "Internal routing failure")
    return table, occ


def logits_to_records(logits, threshold):
    """No sender table, geometry mask, budgets or keys are accepted here.

    Errors are attributed per decoded type, preserving the other capability.
    """
    x = logits.detach().cpu()
    require(
        x.ndim == 3
        and x.shape[-2:] == (4, 177)
        and x.shape[0] <= 256
        and torch.isfinite(x).all(),
        "Decoder tensor",
        Status.EXTRACTION,
    )
    result = []
    errors = {0: [], 1: []}
    for region in x:
        for row in region:
            if row[0].sigmoid().item() < threshold:
                continue
            v, a, r = (
                row[1:5].argmax().item(),
                row[5:7].argmax().item(),
                row[7:11].argmax().item(),
            )
            b = (
                int_bits(v, 2)
                + int_bits(a, 1)
                + int_bits(r, 2)
                + (row[11:] > 0).int().tolist()
            )
            try:
                result.append(Record.from_bits(b))
            except Exception as exc:
                errors[a].append(str(exc))
    return result, errors
