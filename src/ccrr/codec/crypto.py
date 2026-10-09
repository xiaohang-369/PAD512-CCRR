"""Independent object capabilities. Never derive real credentials from training RNG."""

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import os
import secrets
import sqlite3
import struct
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.exceptions import InvalidTag
from ccrr.errors import CCRRError, Status, require

HEADER = struct.Struct(">4sBBBB16s12sI32s")


@dataclass(frozen=True)
class Credential:
    object_id: bytes
    alpha: int
    profile_digest: bytes
    key: bytes = field(repr=False)

    def __post_init__(self):
        require(
            len(self.object_id) == 16
            and len(self.profile_digest) == 32
            and len(self.key) == 32
            and self.alpha in (0, 1),
            "Invalid credential",
        )

    def save(self, path):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(
                self.object_id + bytes([self.alpha]) + self.profile_digest + self.key
            )
            f.flush()
            os.fsync(f.fileno())

    @classmethod
    def load(cls, path):
        raw = Path(path).read_bytes()
        require(len(raw) == 81, "Credential file size")
        return cls(raw[:16], raw[16], raw[17:49], raw[49:])


class NonceAllocator:
    """Local single-authority counter. Entire-store rollback is outside the model.

    Existing credentials cannot register in a missing/restored store. Registration
    is exposed only by new_object(), which always creates new OS-random keys.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(path, timeout=30, isolation_level=None)
        os.chmod(path, 0o600)
        mode = self.db.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        self.db.execute("PRAGMA synchronous=FULL")
        require(
            mode == "wal" and self.db.execute("PRAGMA synchronous").fetchone()[0] == 2,
            "Nonce durability settings",
            Status.NONCE,
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS counters (object BLOB, alpha INTEGER, fingerprint BLOB, next INTEGER NOT NULL, PRIMARY KEY(object,alpha,fingerprint))"
        )

    def new_object(self, digest):
        object_id = secrets.token_bytes(16)
        kc, km = secrets.token_bytes(32), secrets.token_bytes(32)
        while km == kc:
            km = secrets.token_bytes(32)
        creds = tuple(
            Credential(object_id, a, digest, k) for a, k in enumerate((kc, km))
        )
        try:
            self.db.execute("BEGIN IMMEDIATE")
            for c in creds:
                self.db.execute(
                    "INSERT INTO counters VALUES (?,?,?,0)",
                    (c.object_id, c.alpha, hashlib.sha256(c.key).digest()),
                )
            self.db.execute("COMMIT")
        except sqlite3.Error as exc:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise CCRRError(Status.NONCE, "Object reservation failed") from exc
        return creds

    def reserve(self, c):
        args = (c.object_id, c.alpha, hashlib.sha256(c.key).digest())
        try:
            self.db.execute("BEGIN IMMEDIATE")
            result = self.db.execute(
                "SELECT next FROM counters WHERE object=? AND alpha=? AND fingerprint=?",
                args,
            ).fetchone()
            require(
                result is not None,
                "Unknown key; rotate object after lost nonce state",
                Status.NONCE,
            )
            t = result[0]
            require(0 <= t < 2**32, "Nonce counter exhausted", Status.NONCE)
            self.db.execute(
                "UPDATE counters SET next=? WHERE object=? AND alpha=? AND fingerprint=?",
                (t + 1, *args),
            )
            self.db.execute("COMMIT")
            return t.to_bytes(12, "big")
        except (sqlite3.Error, CCRRError) as exc:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise CCRRError(
                Status.NONCE, "Nonce reservation failed; no encryption performed"
            ) from exc

    def close(self):
        self.db.close()


def aad(header, size):
    n = (size + 15) // 16
    return b"MCR-AEAD-v1" + header + struct.pack(">BHHHHI", 1, 128, 43, 4, n, size * 8)


def seal(plaintext, credential, level, allocator):
    c = credential
    require(
        (c.alpha == 1 and level == 0 and len(plaintext) == 320)
        or (
            c.alpha == 0
            and level in (1, 2, 3)
            and len(plaintext) >= 56
            and (len(plaintext) - 48) % 4 == 0
        ),
        "Invalid plaintext contract",
    )
    require(
        len(plaintext) + 88 <= 16384,
        "Encrypted stream exceeds carrier bound",
        Status.CAPACITY,
    )
    nonce = allocator.reserve(c)
    header = HEADER.pack(
        b"MCR1",
        1,
        1,
        c.alpha,
        level,
        c.object_id,
        nonce,
        len(plaintext),
        c.profile_digest,
    )
    return header + ChaCha20Poly1305(c.key).encrypt(
        nonce, plaintext, aad(header, len(plaintext) + 88)
    )


def open_stream(stream, credential, alpha, level, digest, legal_levels=(1, 2, 3)):
    require(88 <= len(stream) <= 16384, "Envelope size", Status.EXTRACTION)
    magic, v, suite, a, r, obj, nonce, p, dg = HEADER.unpack(stream[:72])
    require(
        (magic, v, suite, a, r, dg) == (b"MCR1", 1, 1, alpha, level, digest),
        "Envelope context",
        Status.EXTRACTION,
    )
    require(
        len(stream) == p + 88 and int.from_bytes(nonce, "big") < 2**32,
        "Envelope length/nonce",
        Status.EXTRACTION,
    )
    require(
        (a == 1 and r == 0 and p == 320)
        or (a == 0 and r in legal_levels and p >= 56 and (p - 48) % 4 == 0),
        "Envelope format",
        Status.EXTRACTION,
    )
    if credential is None:
        raise CCRRError(Status.NO_CREDENTIAL, "No credential supplied")
    try:
        plain = ChaCha20Poly1305(credential.key).decrypt(
            nonce, stream[72:], aad(stream[:72], len(stream))
        )
    except InvalidTag as exc:
        raise CCRRError(Status.AUTH, "Authentication rejected") from exc
    require(
        (obj, a, dg)
        == (credential.object_id, credential.alpha, credential.profile_digest),
        "Credential context",
        Status.AUTH,
    )
    return plain, obj
