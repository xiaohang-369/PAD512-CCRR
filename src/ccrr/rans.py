"""Bounded Python rANS64, byte-compatible with CompressAI 1.2.8 integer API.

Independent implementation of the specified 16-bit CDF / 4-bit bypass format.
Every read is checked before use; authenticated data must still pass these checks.
The native decoder's unchecked pointer reads are deliberately not used.
"""

from bisect import bisect_right
import struct
import numpy as np
from .errors import require, Status

LOW = 1 << 31
BOUND = (1 << 20) - 1


def validate_cdfs(cdfs):
    a = np.asarray(cdfs)
    require(
        a.ndim == 2 and a.shape[1] == 513 and a.dtype.kind in "iu",
        "CDF dimensions",
        Status.DECODE,
    )
    require(
        np.all(a[:, 0] == 0)
        and np.all(a[:, -1] == 65536)
        and np.all(np.diff(a.astype(np.int64), axis=1) > 0),
        "Invalid CDF",
        Status.DECODE,
    )
    return a.tolist()


def encode(symbols, indexes, cdfs):
    require(
        len(symbols) == len(indexes) and 0 < len(symbols) <= 524288,
        "Symbol count",
        Status.DECODE,
    )
    tables = validate_cdfs(cdfs)
    state = LOW
    emitted = []

    def put(start, freq, bypass=False):
        nonlocal state
        f = 4096 if bypass else freq
        if state >= ((LOW >> 16) << 32) * f:
            emitted.append(state & 0xFFFFFFFF)
            state >>= 32
        state = (
            ((state << 4) | start)
            if bypass
            else ((state // freq) << 16) + (state % freq) + start
        )

    for n, idx in zip(reversed(symbols), reversed(indexes)):
        n, idx = int(n), int(idx)
        require(
            abs(n) <= BOUND and 0 <= idx < len(tables),
            "Symbol/index range",
            Status.DECODE,
        )
        if -255 <= n <= 255:
            s = n + 255
        else:
            s = 511
            u = 2 * (n - 256) if n >= 256 else 2 * (-n - 256) + 1
            groups = (u.bit_length() + 3) // 4
            for j in reversed(range(groups)):
                put((u >> (4 * j)) & 15, 0, True)
            put(groups, 0, True)  # bounded symbols require at most six groups
        cdf = tables[idx]
        put(cdf[s], cdf[s + 1] - cdf[s])
    words = [state & 0xFFFFFFFF, state >> 32] + list(reversed(emitted))
    return struct.pack("<" + "I" * len(words), *words)


def decode(data, indexes, cdfs):
    require(
        8 <= len(data) <= 8 * 524288 and len(data) % 4 == 0,
        "rANS byte bounds",
        Status.DECODE,
    )
    require(0 < len(indexes) <= 524288, "Decode count", Status.DECODE)
    tables = validate_cdfs(cdfs)
    words = struct.unpack("<" + "I" * (len(data) // 4), data)
    state = words[0] | (words[1] << 32)
    pos = 2
    require(LOW <= state < (1 << 63), "rANS initial state", Status.DECODE)

    def normalize():
        nonlocal state, pos
        if state < LOW:
            require(pos < len(words), "Truncated rANS body", Status.DECODE)
            state = (state << 32) | words[pos]
            pos += 1
            require(state >= LOW, "Invalid rANS renormalization", Status.DECODE)

    def get_bits():
        nonlocal state
        value = state & 15
        state >>= 4
        normalize()
        return value

    out = []
    for idx in indexes:
        require(0 <= int(idx) < len(tables), "CDF index", Status.DECODE)
        cdf = tables[int(idx)]
        cum = state & 65535
        s = bisect_right(cdf, cum) - 1
        state = (cdf[s + 1] - cdf[s]) * (state >> 16) + cum - cdf[s]
        normalize()
        if s == 511:
            groups = get_bits()
            require(groups <= 6, "Unbounded bypass count", Status.DECODE)
            u = sum(get_bits() << (4 * j) for j in range(groups))
            n = -(u >> 1) - 256 if u & 1 else (u >> 1) + 256
        else:
            n = s - 255
        require(abs(n) <= BOUND, "Decoded symbol exceeds bound", Status.DECODE)
        out.append(n)
    require(
        pos == len(words) and state == LOW,
        "Noncanonical/trailing rANS state",
        Status.DECODE,
    )
    require(
        encode(out, indexes, cdfs) == data, "Noncanonical rANS bytes", Status.DECODE
    )
    return np.array(out, dtype=np.int32)
