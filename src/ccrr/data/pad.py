from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import math
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from ccrr.errors import require
from ccrr.codec.metadata import FIELDS, CLASSES
from ccrr.data.resize import teacher_resize


def preprocess(path, size=512):
    with Image.open(path) as src:
        im = ImageOps.exif_transpose(src)
        require(im.mode in ("RGB", "RGBA"), "Unsupported source color mode")
        if im.mode == "RGBA":
            require(
                np.all(np.asarray(im)[:, :, 3] == 255),
                "Nonopaque source requires a new profile",
            )
        a = np.asarray(im.convert("RGB"))
        h, w = a.shape[:2]
    nw = max(1, math.floor(size * w / max(w, h) + 0.5))
    nh = max(1, math.floor(size * h / max(w, h) + 0.5))
    channels = [
        np.asarray(
            Image.fromarray(a[:, :, c].astype(np.float32)).resize(
                (nw, nh), Image.Resampling.BILINEAR
            )
        )
        for c in range(3)
    ]
    rgb = np.clip(np.floor(np.stack(channels, -1) + 0.5), 0, 255).astype(np.uint8)
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), 128, np.uint8)
    out[y0 : y0 + nh, x0 : x0 + nw] = rgb
    v = np.zeros((size, size), np.float32)
    v[y0 : y0 + nh, x0 : x0 + nw] = 1
    legal = v.reshape(size // 32, 32, size // 32, 32).min((1, 3)).reshape(-1)
    return (
        torch.from_numpy(out.copy()).permute(2, 0, 1).float() / 255,
        torch.from_numpy(v[None]),
        (x0, y0, nw, nh),
        torch.from_numpy(legal),
        hashlib.sha256(out.tobytes()).hexdigest(),
    )


def adapt(x, rects, size, normalize=True, panderm=False):
    if x.shape[1] == 1:
        x = x.expand(-1, 3, -1, -1)
    out = []
    masks = []
    transforms = []
    for img, rect in zip(x, rects):
        x0, y0, w, h = map(int, rect)
        nw = max(1, math.floor(size * w / max(w, h) + 0.5))
        nh = max(1, math.floor(size * h / max(w, h) + 0.5))
        resized = teacher_resize(
            img[None, :, y0 : y0 + h, x0 : x0 + w],
            (nh, nw),
        )
        l, t = (size - nw) // 2, (size - nh) // 2
        out.append(
            F.pad(resized, (l, size - nw - l, t, size - nh - t), value=128 / 255)
        )
        masks.append(
            F.pad(torch.ones_like(resized[:, :1]), (l, size - nw - l, t, size - nh - t))
        )
        transforms.append((x0, y0, w, h, l, t, nw, nh, size))
    out = torch.cat(out)
    if normalize:
        mean = out.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = out.new_tensor([0.228 if panderm else 0.229, 0.224, 0.225])[
            None, :, None, None
        ]
        out = (out - mean) / std
    return out, torch.cat(masks), transforms


class PADData:
    def __init__(self, root, role, size=512, verify=True):
        require(
            role in ("fit", "calibration", "val_tune", "val_select", "test"),
            "Unknown data role",
        )
        self.root = Path(root)
        self.role = role
        self.size = size
        audit = self.root / "protocol_audit_20261009"
        self.rows = json.loads((audit / "roles" / f"{role}.json").read_text())
        with (self.root / "raw/metadata.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        require(
            len({r["img_id"] for r in rows}) == len(rows),
            "Duplicate source metadata ID",
        )
        self.metadata = {r["img_id"]: {k: r[k] for k in FIELDS} for r in rows}
        self.inventory = {
            r["img_id"]: r
            for r in json.loads((audit / "preprocessing_inventory.json").read_text())
        }
        self.verify = verify
        self.index = {r["img_id"]: i for i, r in enumerate(self.rows)}
        self.patients = sorted({r["patient_id"] for r in self.rows})
        self.pid = {p: i for i, p in enumerate(self.patients)}
        if verify:
            groups = [
                {
                    r["patient_id"]
                    for r in json.loads((audit / "roles" / f"{v}.json").read_text())
                }
                for v in ("fit", "calibration", "val_tune", "val_select", "test")
            ]
            require(
                sum(map(len, groups)) == len(set.union(*groups)), "Patient role overlap"
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        x, v, rect, legal, digest = preprocess(
            self.root / row["source_relpath"], self.size
        )
        if self.verify and self.size == 512:
            require(
                digest == self.inventory[row["img_id"]]["rgb512_sha256"],
                "Preprocessing hash mismatch",
            )
        return {
            "x": x,
            "v": v,
            "rect": rect,
            "legal": legal,
            "label": CLASSES.index(row["diagnostic"]),
            "patient": row["patient_id"],
            "patient_index": self.pid[row["patient_id"]],
            "id": row["img_id"],
            "metadata": self.metadata[row["img_id"]],
        }

    def class_weights(self):
        count = np.bincount(
            [CLASSES.index(r["diagnostic"]) for r in self.rows], minlength=6
        )
        require(np.all(count > 0), "Fit missing a diagnosis")
        w = 1 / np.sqrt(count)
        return torch.tensor(w / w.mean(), dtype=torch.float32)


def collate(items, device="cpu"):
    out = {
        k: torch.stack([v[k] for v in items]).to(device) for k in ("x", "v", "legal")
    }
    out.update(
        {
            k: torch.tensor([v[k] for v in items], device=device)
            for k in ("label", "patient_index")
        }
    )
    out.update(
        {k: [v[k] for v in items] for k in ("rect", "patient", "id", "metadata")}
    )
    return out


def rng_for(seed, stage, role, step):
    raw = hashlib.sha256(f"{seed}|{stage}|{role}|{step}".encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(raw[:16], "big")))


class PairSampler:
    def __init__(self, data):
        self.data = data
        self.patients = defaultdict(lambda: defaultdict(list))
        self.fallbacks = 0
        for i, row in enumerate(data.rows):
            if row.get("reid_observation_eligible", "1") == "1":
                self.patients[row["patient_id"]][row["lesion_id"]].append(i)
        self.p1 = sorted(p for p, l in self.patients.items() if len(l) >= 2)
        self.p2 = sorted(
            p for p, l in self.patients.items() if any(len(v) >= 2 for v in l.values())
        )
        self.all = sorted(self.patients)

    def image(self, p, rng):
        return int(rng.choice([i for v in self.patients[p].values() for i in v]))

    def sample(self, rng, scenario="mixed"):
        require(bool(self.p1) and bool(self.p2), "Missing real P1/P2 positive pools")
        result = []
        for j in range(8):
            s = ("p1" if j < 4 else "p2") if scenario == "mixed" else scenario
            p = str(rng.choice(self.p1 if s == "p1" else self.p2))
            lesions = self.patients[p]
            if s == "p1":
                la, lb = rng.choice(sorted(lesions), 2, replace=False)
                a = int(rng.choice(lesions[la]))
                b = int(rng.choice(lesions[lb]))
            else:
                l = str(
                    rng.choice(sorted(k for k, v in lesions.items() if len(v) >= 2))
                )
                a, b = map(int, rng.choice(lesions[l], 2, replace=False))
            result.append((a, b, 1, s))
        for j in range(8):
            if j < 4:
                pa, pb = rng.choice(self.all, 2, replace=False)
                a = self.image(pa, rng)
                b = self.image(pb, rng)
            else:
                a = result[j - 4][int(rng.integers(2))]
                row = self.data.rows[a]
                candidates = defaultdict(list)
                for p, ls in self.patients.items():
                    if p == row["patient_id"]:
                        continue
                    for ids in ls.values():
                        for i in ids:
                            r = self.data.rows[i]
                            if (r["diagnostic"], r["region"]) == (
                                row["diagnostic"],
                                row["region"],
                            ):
                                candidates[p].append(i)
                if candidates:
                    p = str(rng.choice(sorted(candidates)))
                    b = int(rng.choice(candidates[p]))
                else:
                    self.fallbacks += 1
                    p = str(rng.choice([p for p in self.all if p != row["patient_id"]]))
                    b = self.image(p, rng)
            result.append((a, b, 0, "negative"))
        return result
