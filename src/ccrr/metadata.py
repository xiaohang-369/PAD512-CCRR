import json
from .errors import require, Status, CCRRError

FIELDS = tuple(
    "patient_id lesion_id smoke drink background_father background_mother age pesticide gender skin_cancer_history cancer_history has_piped_water has_sewage_system fitspatrick region diameter_1 diameter_2 diagnostic itch grew hurt changed bleed elevation img_id biopsed".split()
)
CLASSES = ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK")


def serialize(row):
    require(set(row) == set(FIELDS), "PAD-M1 requires exactly 26 fields")
    values = [row[k] for k in FIELDS]
    for value in values:
        require(type(value) is str, "PAD-M1 values must be strings")
        try:
            size = len(value.encode("utf-8", "strict"))
        except UnicodeError:
            raise ValueError("Invalid Unicode") from None
        require(size <= 64, "PAD-M1 field exceeds 64 UTF-8 bytes")
    require(
        all(row[k] for k in ("patient_id", "lesion_id", "img_id", "diagnostic")),
        "Empty core field",
    )
    require(row["diagnostic"] in CLASSES, "Unknown diagnosis")
    raw = json.dumps(
        {"schema": "PAD-M1", "values": values},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    require(len(raw) <= 318, "Metadata Too Large")
    return len(raw).to_bytes(2, "big") + raw + bytes(318 - len(raw))


def deserialize(data):
    require(len(data) == 320, "Metadata length", Status.DECODE)
    size = int.from_bytes(data[:2], "big")
    require(
        0 < size <= 318 and not any(data[2 + size :]),
        "Metadata length/padding",
        Status.DECODE,
    )

    def unique(pairs):
        d = {}
        for k, v in pairs:
            require(k not in d, "Duplicate JSON key", Status.DECODE)
            d[k] = v
        return d

    try:
        obj = json.loads(
            data[2 : 2 + size].decode("utf-8", "strict"), object_pairs_hook=unique
        )
        require(
            type(obj) is dict and set(obj) == {"schema", "values"},
            "Schema keys",
            Status.DECODE,
        )
        require(
            obj["schema"] == "PAD-M1"
            and type(obj["values"]) is list
            and len(obj["values"]) == 26,
            "Schema",
            Status.DECODE,
        )
        row = dict(zip(FIELDS, obj["values"]))
        require(serialize(row) == data, "Noncanonical metadata", Status.DECODE)
        return row
    except (ValueError, TypeError, KeyError, UnicodeError, CCRRError) as exc:
        raise CCRRError(Status.DECODE, "Invalid metadata") from exc
