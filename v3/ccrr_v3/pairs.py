"""Explicit P2 only. No dependence on P1-positive eligibility."""

from collections import defaultdict


def validate_rows(rows):
    ids = [r["img_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate image IDs")
    for r in rows:
        for key in ("img_id", "patient_id", "lesion_id", "diagnostic"):
            if not isinstance(r.get(key), str) or not r[key]:
                raise ValueError(f"Missing row field {key}")


class P2Sampler:
    def __init__(self, rows, pairs=16):
        validate_rows(rows)
        if pairs < 2 or pairs % 2:
            raise ValueError("Even balanced pair count required")
        self.rows, self.count = rows, pairs
        groups = defaultdict(lambda: defaultdict(list))
        for i, r in enumerate(rows):
            if str(r.get("reid_observation_eligible", "1")) == "1":
                groups[r["patient_id"]][r["lesion_id"]].append(i)
        self.groups = dict(groups)
        self.eligible = sorted(
            p
            for p, lesions in groups.items()
            if any(len(v) >= 2 for v in lesions.values())
        )
        if not self.eligible or len(groups) < 2:
            raise ValueError("Need real P2 positives and cross-patient negatives")
        self.patients = sorted(groups)
        self.images = {p: [i for v in groups[p].values() for i in v] for p in groups}
        self.fallbacks = 0

    def sample(self, rng):
        result = []
        for _ in range(self.count // 2):
            p = str(rng.choice(self.eligible))
            lesion = str(
                rng.choice(sorted(k for k, v in self.groups[p].items() if len(v) >= 2))
            )
            a, b = map(int, rng.choice(self.groups[p][lesion], 2, replace=False))
            result.append((a, b, 1, "p2"))
        for j in range(self.count // 2):
            a = result[j][0]
            row = self.rows[a]
            candidates = {}
            for p in self.patients:
                if p == row["patient_id"]:
                    continue
                ids = self.images[p]
                if j % 2:
                    ids = [
                        i
                        for i in ids
                        if self.rows[i]["diagnostic"] == row["diagnostic"]
                        and self.rows[i].get("region") == row.get("region")
                    ]
                if ids:
                    candidates[p] = ids
            if not candidates:
                self.fallbacks += 1
                candidates = {
                    p: self.images[p] for p in self.patients if p != row["patient_id"]
                }
            p = str(rng.choice(sorted(candidates)))
            result.append((a, int(rng.choice(candidates[p])), 0, "negative"))
        return result


def validate_p2_protocol(protocol, rows):
    validate_rows(rows)
    index = {r["img_id"]: r for r in rows}
    seen = set()
    if not protocol.get("queries"):
        raise ValueError("No P2 queries")
    for query in protocol["queries"]:
        name = query["query"]
        if name not in index or name in seen:
            raise ValueError("Missing or duplicate P2 query")
        seen.add(name)
        a = index[name]
        if query["patient_id"] != a["patient_id"]:
            raise ValueError("Query patient mismatch")
        pos, neg = query["positive_gallery"], query["negative_gallery"]
        if not pos or not neg or len(set(pos + neg)) != len(pos + neg):
            raise ValueError("Invalid P2 gallery groups")
        for b in pos:
            if (
                b == name
                or b not in index
                or (index[b]["patient_id"], index[b]["lesion_id"])
                != (a["patient_id"], a["lesion_id"])
            ):
                raise ValueError(
                    "P2 positive must be a different real image of same lesion"
                )
        for b in neg:
            if b not in index or index[b]["patient_id"] == a["patient_id"]:
                raise ValueError("P2 negatives must be cross-patient")
    return protocol
