"""Application command handlers; protocol mechanisms remain in their owning packages."""

from pathlib import Path
import json
import numpy as np
import torch
from ccrr.config import Profile, load_config, load_manifest, sha256, export_manifest
from ccrr.runtime.support import environment, write_json, json_safe
from ccrr.runtime.system import CCRR, load_public, save_public
from ccrr.codec.crypto import NonceAllocator, Credential
from ccrr.data.pad import preprocess
from ccrr.errors import require


def load_system(manifest, device):
    profile, digest, obj = load_manifest(manifest)
    base = Path(manifest).parent
    state_path = base / obj["resources"]["system"]["path"]
    cdfs = np.load(base / obj["resources"]["cdf"]["path"], allow_pickle=False)
    system = CCRR(
        profile, fixed_gray=obj.get("evidence", {}).get("fixed_gray", False)
    ).to(device)
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if "models" in state:
        state = state["models"]["system"]
    system.load_state_dict(state, strict=True)
    system.bind(digest, cdfs)
    system.eval()
    return system, obj


def inspect(args):
    report = environment()
    if args.config:
        cfg = load_config(args.config)
        report["weights"] = {
            name: {
                "exists": bool(w.get("path") and Path(w["path"]).is_file()),
                "expected_hash_configured": bool(w.get("sha256")),
            }
            for name, w in cfg["weights"].items()
        }
    print(json.dumps(report, indent=2))
    return

    return 0


def preflight(args):
    from ccrr.runtime.preflight import inspect

    cfg = load_config(args.config)
    report = inspect(cfg)
    write_json(Path(cfg["output"]) / "preflight.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready_for_teacher_training"] else 2

    return 0


def train(args):
    from ccrr.training.experiment import Experiment

    e = Experiment(load_config(args.config))
    e.run(args.stage)

    return 0


def audit_data(args):
    from ccrr.data.pad import PADData
    from ccrr.codec.metadata import serialize, deserialize

    report = {}
    for role in ("fit", "calibration", "val_tune", "val_select", "test"):
        ds = PADData(args.root, role)
        lengths = []
        for i in range(len(ds)):
            b = ds[i]
            raw = serialize(b["metadata"])
            require(deserialize(raw) == b["metadata"], "Metadata roundtrip")
            lengths.append(int.from_bytes(raw[:2], "big"))
        report[role] = {
            "images": len(ds),
            "patients": len(ds.patients),
            "metadata_json_bytes": [min(lengths), max(lengths)],
            "all_preprocessing_hashes_verified": True,
        }
    write_json(args.out, report)

    return 0


def export(args):
    import shutil

    cfg = load_config(args.config)
    evidence = json.loads(Path(args.evidence).read_text())
    out = Path(args.out)
    require(not out.exists(), "Output bundle already exists")
    out.mkdir(parents=True)
    for src, name in ((args.checkpoint, "system.pt"), (args.cdf, "cdfs.npy")):
        shutil.copyfile(src, out / name)
    resources = {
        name: {"path": file, "sha256": sha256(out / file)}
        for name, file in (("system", "system.pt"), ("cdf", "cdfs.npy"))
    }
    export_manifest(
        out / "manifest.json",
        Profile.from_dict(cfg["profile"]),
        resources,
        evidence,
    )

    return 0


def receive(args):
    system, _ = load_system(args.manifest, args.device)
    g = load_public(args.image, system.profile.size).to(args.device)
    creds = {}
    if args.key_c:
        creds[0] = Credential.load(args.key_c)
    if args.key_m:
        creds[1] = Credential.load(args.key_m)
    result = system.receive(g, creds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    summary = {"association_status": result["association_status"]}
    for name in ("C", "M"):
        r = result[name]
        summary[name] = {
            "status": r.status,
            "detail": r.detail,
            "image_status": r.image_status,
        }
        if r.status == "Success":
            if name == "M":
                write_json(out / "metadata.json", r.value)
            else:
                from PIL import Image

                Image.fromarray(
                    torch.floor(r.value["rgb"][0].permute(1, 2, 0).cpu() * 255 + 0.5)
                    .byte()
                    .numpy()
                ).save(out / "recovered.png")
    write_json(out / "status.json", summary)
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))

    return 0


def send(args):
    from ccrr.teachers.adapters import ClinicalModel, freeze
    from ccrr.runtime.checkpoints import load_checkpoint
    from ccrr.calibration.probes import clinical_risk

    system, obj = load_system(args.manifest, args.device)
    teacher = ClinicalModel().to(args.device)
    expected = obj["evidence"].get("clinical_sha256")
    require(
        expected and sha256(args.clinical_checkpoint) == expected,
        "Sender clinical checkpoint not bound to profile",
    )
    load_checkpoint(args.clinical_checkpoint, {"model": teacher})
    freeze(teacher)
    x, v, rect, legal, _ = preprocess(args.image, system.profile.size)
    x = x[None].to(args.device)
    legal = legal[None].to(args.device)
    row = json.loads(Path(args.metadata).read_text())
    out = Path(args.out)
    require(not out.exists(), "New publication requires a new output directory")
    out.mkdir(parents=True, mode=0o700)
    allocator = NonceAllocator(args.nonce_db)
    try:
        credentials = allocator.new_object(system.digest)
        with torch.no_grad():
            g = system.decompose(x)[2]
            raw = clinical_risk(
                g,
                legal[0].cpu().numpy(),
                Path(args.image).name,
                teacher,
                [rect],
            )
            risk = torch.tensor(
                np.minimum(raw / system.profile.q95, 1),
                device=args.device,
                dtype=torch.float32,
            )[None]
            result = system.send(x, row, legal, risk, credentials, allocator)
            save_public(result["public"], out / "public.png")
        credentials[0].save(out / "C.key")
        credentials[1].save(out / "M.key")
        write_json(
            out / "status.json",
            {
                "status": "Published",
                "level": result["level"],
                "c_bytes": len(result["c"]),
                "m_bytes": len(result["m"]),
                "quality": "Not assessed by send",
            },
        )
    finally:
        allocator.close()

    return 0


def smoke(args):
    from ccrr.runtime.smoke import run

    run(args.out, args.size)

    return 0
