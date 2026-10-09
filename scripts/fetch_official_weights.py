"""Download only the audited official ImageNet weights; write hashes to a new config.

PanDerm's Google Drive artifact is supplied explicitly, then SHA-256 recorded and
strictly checked at load. This tool does not guess Drive access or a downstream model.
"""

from pathlib import Path
import argparse
import hashlib
import json
import urllib.request
import yaml

URLS = {
    "efficientnet_b0": "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
    "convnext_tiny": "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
    "swin_t": "https://download.pytorch.org/models/swin_t-704ceda3.pth",
}
p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--directory", required=True)
p.add_argument("--out-config", required=True)
p.add_argument("--panderm-weights")
p.add_argument("--panderm-source")
args = p.parse_args()
root = Path(args.directory)
root.mkdir(parents=True, exist_ok=True)
cfg = yaml.safe_load(Path(args.config).read_text())
report = {}
for name, url in URLS.items():
    path = root / url.rsplit("/", 1)[1]
    if not path.exists():
        tmp = path.with_suffix(".download")
        with urllib.request.urlopen(url, timeout=60) as source, tmp.open("wb") as dst:
            while block := source.read(1024 * 1024):
                dst.write(block)
        tmp.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    prefix = path.stem.rsplit("-", 1)[1]
    if not digest.startswith(prefix):
        raise ValueError(f"Official filename checksum mismatch: {name}")
    cfg["weights"][name] = {"path": str(path.resolve()), "sha256": digest}
    report[name] = {"url": url, "sha256": digest}
for arg, key, hashkey in (
    (args.panderm_weights, "path", "sha256"),
    (args.panderm_source, "source", "source_sha256"),
):
    if arg:
        path = Path(arg)
        cfg["weights"]["panderm"][key] = str(path.resolve())
        cfg["weights"]["panderm"][hashkey] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
Path(args.out_config).write_text(yaml.safe_dump(cfg, sort_keys=False))
(root / "sources.json").write_text(json.dumps(report, indent=2) + "\n")
