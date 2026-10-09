from dataclasses import dataclass
import numpy as np
from PIL import Image
import torch
from torch import nn
from ccrr.config import Profile
from ccrr.errors import CCRRError, Status, require
from ccrr.representation.inn import INN
from ccrr.communication.network import Communication
from ccrr.representation.inn import FixedGrayDecomposition
from ccrr.tensor_ops import q8, public_adapter, ste_round
from ccrr.codec.entropy import EntropyModel, SourceCodec
from ccrr.codec import metadata
from ccrr.codec.crypto import seal, open_stream
from ccrr.communication.records import (
    split_stream,
    route,
    reassemble,
    logits_to_records,
)


@dataclass
class CapabilityResult:
    status: Status
    value: object = None
    object_id: bytes | None = None
    detail: str = ""
    image_status: str | None = None


class CCRR(nn.Module):
    def __init__(self, profile=None, fixed_gray=False, patch_batch=16):
        super().__init__()
        self.profile = profile or Profile()
        self.fixed_gray = fixed_gray
        self.inn = FixedGrayDecomposition() if fixed_gray else INN()
        self.entropy = EntropyModel()
        self.comm = Communication(patch_batch)
        self.codec = None
        self.digest = None

    def bind(self, digest, cdfs):
        require(len(digest) == 32, "Digest size")
        self.digest = digest
        self.codec = SourceCodec(self.profile, digest, cdfs)

    def decompose(self, x):
        u, z = self.inn(x)
        g = q8(u if self.fixed_gray else u.sigmoid())
        return u, z, g

    def recover(self, g, z):
        return self.inn.inverse(
            g if self.fixed_gray else public_adapter(g, self.profile.logit_eps), z
        )

    def quantized(self, x, level):
        u, z, g = self.decompose(x)
        n = ste_round(z / self.profile.step(level))
        return u, z, g, n, n * self.profile.step(level)

    def payload(self, z, row, credentials, allocator, budgets, level_order=None):
        require(self.codec is not None, "Bind a snapshot before encoding")
        kc, km = credentials
        require(
            kc.alpha == 0
            and km.alpha == 1
            and kc.object_id == km.object_id
            and kc.profile_digest == km.profile_digest == self.digest,
            "Object capability mismatch",
        )
        m = seal(metadata.serialize(row), km, 0, allocator)
        mr = split_stream(m, 1, 0)
        failures = []
        for level in level_order or self.profile.quality_order:
            n = self.codec.quantize(z, level)
            plain = self.codec.encode(n, level)
            required = (len(plain) + 88 + 15) // 16 + len(mr)
            if len(plain) + 88 > 16384 or required > int(np.asarray(budgets).sum()):
                failures.append(
                    {"level": level, "source_bytes": len(plain), "records": required}
                )
                continue
            c = seal(plain, kc, level, allocator)
            return {
                "c": c,
                "m": m,
                "c_records": split_stream(c, 0, level),
                "m_records": mr,
                "level": level,
                "n": n,
                "zq": n * self.profile.step(level),
                "rejected": failures,
            }
        raise CCRRError(
            Status.CAPACITY, f"No complete visual stream fits; candidates={failures}"
        )

    def send(
        self, x, row, legal, risk, credentials, allocator, policy=None, level_order=None
    ):
        require(x.shape == (1, 3, self.profile.size, self.profile.size), "Send shape")
        u, z, g = self.decompose(x)
        if policy is None:
            with torch.no_grad():
                tokens = self.comm.carrier(g)
                cap, _ = self.comm.heads(tokens, risk)
                k = (cap.argmax(-1) - self.profile.conservative_budget).clamp_min(
                    0
                ) * legal.long()
                _, rel = self.comm.heads(tokens, risk, k)
            budgets = k[0].cpu().numpy()
            reliability = rel[0].cpu().numpy()
        else:
            budgets, reliability = policy
        require(
            np.all(np.asarray(budgets)[legal[0].detach().cpu().numpy() == 0] == 0),
            "Policy writes padding",
        )
        payload = self.payload(z, row, credentials, allocator, budgets, level_order)
        table, occ = route(
            payload["c_records"],
            payload["m_records"],
            budgets,
            reliability,
            self.profile.reliability_sort,
        )
        bits = torch.from_numpy(table)[None].to(x)
        occupancy = torch.from_numpy(occ)[None].to(x)
        public, gates = self.comm.write(g, bits, occupancy, legal, self.profile.delta)
        payload.update(
            {
                "public": public,
                "g0": g,
                "u": u,
                "z": z,
                "bits": bits,
                "occupancy": occupancy,
                "gates": gates,
                "budgets": budgets,
            }
        )
        return payload

    def receive_records(self, g, records, errors, credentials):
        require(self.codec is not None, "Missing deployed profile")
        results = {}
        for alpha in (0, 1):
            try:
                require(
                    not errors[alpha], "Malformed decoded record", Status.EXTRACTION
                )
                stream, level = reassemble(records, alpha)
                plain, obj = open_stream(
                    stream,
                    credentials.get(alpha),
                    alpha,
                    level,
                    self.digest,
                    self.profile.legal_levels,
                )
                if alpha == 1:
                    value = metadata.deserialize(plain)
                    image_status = None
                else:
                    n, z = self.codec.decode(plain, level, g.device)
                    try:
                        rgb = self.recover(g, z)
                    except FloatingPointError as exc:
                        raise CCRRError(Status.NUMERICAL, "Nonfinite inverse") from exc
                    require(
                        torch.isfinite(rgb).all(), "Nonfinite inverse", Status.NUMERICAL
                    )
                    value = {
                        "rgb_unclipped": rgb,
                        "rgb": rgb.clamp(0, 1),
                        "integers": n,
                        "level": level,
                    }
                    image_status = "Numerically valid; quality not assessed"
                results[alpha] = CapabilityResult(
                    Status.SUCCESS, value, obj, image_status=image_status
                )
            except CCRRError as exc:
                results[alpha] = CapabilityResult(exc.status, detail=exc.detail)
        both = all(results[a].status == Status.SUCCESS for a in (0, 1))
        return {
            "C": results[0],
            "M": results[1],
            "association_status": (
                "Same object"
                if results[0].object_id == results[1].object_id
                else "Object mismatch"
            )
            if both
            else "Not established",
        }

    @torch.no_grad()
    def receive(self, g, credentials=None):
        require(
            g.shape == (1, 1, self.profile.size, self.profile.size), "Receive shape"
        )
        require(
            torch.isfinite(g).all() and ((g >= 0) & (g <= 1)).all(), "Receive pixels"
        )
        records, errors = logits_to_records(
            self.comm.read(g)[0], self.profile.occupancy_threshold
        )
        return self.receive_records(g, records, errors, credentials or {})


def save_public(g, path):
    a = torch.floor(g.detach().cpu().squeeze() * 255 + 0.5).to(torch.uint8).numpy()
    require(a.ndim == 2, "Public PNG must be grayscale")
    Image.fromarray(a).save(path, format="PNG")
    loaded = load_public(path, size=a.shape[0])
    require(
        np.array_equal(a, (loaded.squeeze().numpy() * 255).round().astype(np.uint8)),
        "PNG roundtrip mismatch",
    )
    return loaded


def load_public(path, size=512):
    with Image.open(path) as im:
        require(
            im.format == "PNG" and im.mode == "L" and im.size == (size, size),
            "Expected single-channel PNG",
        )
        require(not im.info, "Public PNG must not carry ancillary metadata")
        a = np.asarray(im).copy()
    return torch.from_numpy(a)[None, None].float() / 255
