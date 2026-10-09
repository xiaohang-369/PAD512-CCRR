import dataclasses
import os
import concurrent.futures
import numpy as np
import pytest
from ccrr.errors import CCRRError, Status
from ccrr.metadata import FIELDS, serialize, deserialize
from ccrr.crypto import NonceAllocator, seal, open_stream, aad
from ccrr.records import Record, split_stream, reassemble, route
from ccrr.entropy import EntropyModel
from ccrr import rans


def row():
    r = dict.fromkeys(FIELDS, "")
    r.update(
        patient_id="PAT_test",
        lesion_id="1",
        img_id="synthetic.png",
        diagnostic="ACK",
        age="1.0",
    )
    return r


def test_metadata_canonical_and_errors():
    m = row()
    m["region"] = 'a\n"\\中文'
    b = serialize(m)
    assert len(b) == 320 and deserialize(b) == m
    bad = bytearray(b)
    bad[-1] = 1
    with pytest.raises(CCRRError):
        deserialize(bytes(bad))
    with pytest.raises(CCRRError):
        serialize({**m, "age": 1})
    with pytest.raises(CCRRError):
        serialize({**m, "age": "x" * 65})


def test_independent_capabilities_and_nonce_restart(tmp_path):
    path = tmp_path / "state.sqlite"
    alloc = NonceAllocator(path)
    digest = os.urandom(32)
    kc, km = alloc.new_object(digest)
    m = seal(serialize(row()), km, 0, alloc)
    cplain = b"ZC01" + bytes(52)
    c = seal(cplain, kc, 1, alloc)
    assert len(m) == 408 and len(aad(m[:72], len(m))) == 96
    assert open_stream(m, km, 1, 0, digest)[0] == serialize(row())
    assert open_stream(c, kc, 0, 1, digest)[0] == cplain
    for cred in (kc, dataclasses.replace(km, key=os.urandom(32))):
        with pytest.raises(CCRRError) as e:
            open_stream(m, cred, 1, 0, digest)
        assert e.value.status == Status.AUTH
    with pytest.raises(CCRRError) as e:
        open_stream(m, None, 1, 0, digest)
    assert e.value.status == Status.NO_CREDENTIAL
    bad = bytearray(m)
    bad[-1] ^= 1
    with pytest.raises(CCRRError):
        open_stream(bytes(bad), km, 1, 0, digest)
    alloc.close()
    alloc = NonceAllocator(path)
    assert int.from_bytes(alloc.reserve(km), "big") == 1
    other = NonceAllocator(tmp_path / "lost.sqlite")
    with pytest.raises(CCRRError):
        other.reserve(km)
    alloc.close()
    other.close()


def test_concurrent_nonce_reservations(tmp_path):
    path = tmp_path / "n.sqlite"
    a = NonceAllocator(path)
    _, c = a.new_object(os.urandom(32))
    a.close()

    def reserve(_):
        b = NonceAllocator(path)
        try:
            return b.reserve(c)
        finally:
            b.close()

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        nonces = list(pool.map(reserve, range(20)))
    assert len(set(nonces)) == 20


def test_record_routing_rejects_duplicates_and_padding():
    c = split_stream(os.urandom(88), 0, 1)
    m = split_stream(os.urandom(408), 1, 0)
    bits, occ = route(c, m, [4] * 10, np.ones(10), True)
    records = [Record.from_bits(bits[i, j]) for i, j in zip(*np.nonzero(occ))]
    assert reassemble(records, 0)[0] == b"".join(x.data for x in c)[:88]
    assert occ.sum() == len(c) + len(m)
    assert all(Record.from_bits(bits[0, j]).alpha == 1 for j in range(4))
    with pytest.raises(CCRRError):
        reassemble(records + [c[0]], 0)
    with pytest.raises(CCRRError):
        route(c, m, [1] * 3, np.ones(3))
    bad = dataclasses.replace(c[-1], data=c[-1].data[:-1] + b"\x01")
    with pytest.raises(CCRRError):
        reassemble(c[:-1] + [bad], 0)


def test_rans_boundary_fuzz_and_truncation():
    cdf = EntropyModel().export()[0]
    rng = np.random.default_rng(3)
    ns = np.concatenate(
        (
            [-1048575, -256, -255, 0, 255, 256, 1048575],
            rng.integers(-10000, 10000, 1000),
        )
    )
    ix = rng.integers(0, 128, len(ns))
    body = rans.encode(ns, ix, cdf)
    assert np.array_equal(rans.decode(body, ix, cdf), ns)
    for bad in (body[:4], body[:-4], body + bytes(4), bytes(len(body))):
        with pytest.raises(CCRRError):
            rans.decode(bad, ix, cdf)
