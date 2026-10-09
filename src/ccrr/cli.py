from pathlib import Path
import argparse
import json
import sys
import numpy as np
import torch
from .config import Profile, load_config, load_manifest, sha256, export_manifest
from .runtime import environment, write_json, json_safe
from .system import CCRR, load_public, save_public
from .crypto import NonceAllocator, Credential
from .data import preprocess
from .errors import CCRRError, require


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


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="ccrr",
        description="Complete capability-conditioned medical representation research system",
    )
    sub = p.add_subparsers(dest="command", required=True)
    x = sub.add_parser("inspect")
    x.add_argument("--config")
    x = sub.add_parser("train")
    x.add_argument("--config", required=True)
    x.add_argument(
        "--stage",
        default="all",
        choices=["all", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    )
    x = sub.add_parser("preflight")
    x.add_argument("--config", required=True)
    x = sub.add_parser("audit-data")
    x.add_argument("--root", required=True)
    x.add_argument("--out", required=True)
    x = sub.add_parser("export")
    x.add_argument("--checkpoint", required=True)
    x.add_argument("--cdf", required=True)
    x.add_argument("--config", required=True)
    x.add_argument("--evidence", required=True)
    x.add_argument("--out", required=True)
    x = sub.add_parser("receive")
    x.add_argument("--manifest", required=True)
    x.add_argument("--image", required=True)
    x.add_argument("--key-c")
    x.add_argument("--key-m")
    x.add_argument("--out", required=True)
    x.add_argument("--device", default="cpu")
    x = sub.add_parser("send")
    x.add_argument("--manifest", required=True)
    x.add_argument("--image", required=True)
    x.add_argument("--metadata", required=True)
    x.add_argument("--clinical-checkpoint", required=True)
    x.add_argument("--nonce-db", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--device", default="cpu")
    x = sub.add_parser("smoke")
    x.add_argument("--out", required=True)
    x.add_argument("--size", type=int, choices=(256, 512), default=512)
    args = p.parse_args(argv)
    try:
        if args.command == "inspect":
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
        if args.command == "preflight":
            from .preflight import inspect

            cfg = load_config(args.config)
            report = inspect(cfg)
            write_json(Path(cfg["output"]) / "preflight.json", report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["ready_for_teacher_training"] else 2
        elif args.command == "train":
            from .training import Experiment

            e = Experiment(load_config(args.config))
            e.run(args.stage)
        elif args.command == "audit-data":
            from .data import PADData
            from .metadata import serialize, deserialize

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
        elif args.command == "export":
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
        elif args.command == "receive":
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
                            torch.floor(
                                r.value["rgb"][0].permute(1, 2, 0).cpu() * 255 + 0.5
                            )
                            .byte()
                            .numpy()
                        ).save(out / "recovered.png")
            write_json(out / "status.json", summary)
            print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
        elif args.command == "send":
            from .models.teachers import ClinicalModel, freeze
            from .runtime import load_checkpoint
            from .calibration import clinical_risk

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
        elif args.command == "smoke":
            from .smoke import run

            run(args.out, args.size)
    except (CCRRError, FileNotFoundError, ValueError) as exc:
        print(
            json.dumps(
                {"status": getattr(exc, "status", "Error"), "detail": str(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
