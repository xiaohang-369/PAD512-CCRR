from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import yaml
from ccrr.errors import require


@dataclass(frozen=True)
class Profile:
    name: str = "PAD512-CCRR-v1.1"
    size: int = 512
    patch: int = 32
    kmax: int = 4
    delta: float = 4 / 255
    latent_step: float = 1 / 32
    multipliers: tuple = (1, 2, 4)
    legal_levels: tuple = (1, 2, 3)
    quality_order: tuple = (1, 2, 3)
    occupancy_threshold: float = 0.5
    conservative_budget: int = 0
    reliability_sort: bool = False
    q95: float = 1.0
    symbol_bound: int = 2**20 - 1
    logit_eps: float = 1 / 510
    research: bool = True

    def __post_init__(self):
        require(
            self.size in (256, 512),
            "Only 256 engineering and 512 research geometries supported",
        )
        require(self.patch == 32 and self.kmax == 4, "Record architecture is fixed")
        require(self.size == 512 or self.research, "256 cannot be a deployment profile")
        require(self.multipliers == (1, 2, 4), "Quantization profile mismatch")
        require(
            set(self.legal_levels) <= {1, 2, 3} and len(self.legal_levels) > 0,
            "Invalid levels",
        )
        require(
            set(self.quality_order) == set(self.legal_levels),
            "Quality order must cover legal levels",
        )
        require(len(self.quality_order) == len(self.legal_levels), "Repeated level")
        require(
            0 < self.occupancy_threshold < 1 and self.conservative_budget in (0, 1, 2),
            "Invalid policy",
        )
        require(self.delta == 4 / 255 and self.q95 > 0, "Invalid amplitude/risk scale")

    @property
    def regions(self):
        return (self.size // self.patch) ** 2

    @property
    def symbols(self):
        return 2 * self.size**2

    @property
    def packed_shape(self):
        return (128, self.size // 8, self.size // 8)

    def step(self, level):
        require(level in self.legal_levels, "Unsupported quantization level")
        return self.latent_step * self.multipliers[level - 1]

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        for k in ("multipliers", "legal_levels", "quality_order"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(obj):
    return json.dumps(
        obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False
    ).encode()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_manifest(path, verify=True):
    path = Path(path)
    raw = path.read_bytes()
    obj = json.loads(raw)
    profile = Profile.from_dict(obj["profile"])
    if verify:
        for resource in obj["resources"].values():
            p = path.parent / resource["path"]
            require(
                sha256(p) == resource["sha256"], f"Resource hash mismatch: {p.name}"
            )
    return profile, hashlib.sha256(raw).digest(), obj


def export_manifest(path, profile, resources, evidence):
    path = Path(path)
    require(not path.exists(), "Manifest must not overwrite an existing deployment")
    if not profile.research:
        require(
            evidence.get("deployment_accepted") is True, "Missing deployment acceptance"
        )
    obj = {
        "format": "CCRR-PROFILE-1",
        "profile": asdict(profile),
        "resources": resources,
        "evidence": evidence,
        "codec": "bounded-python-rans64-v1",
        "bit_order": "MSB-first",
        "record_bits": 171,
        "source_header_bytes": 48,
        "aead_header_bytes": 72,
        "schema": "PAD-M1",
        "quantization": "ties-away-from-zero",
        "squeeze": "4c+2i+j",
        "codec_reference": "CompressAI-1.2.8/79e51871e972c517bc528b689ccf15137fcc7785",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(obj) + b"\n")
    return hashlib.sha256(path.read_bytes()).digest()
