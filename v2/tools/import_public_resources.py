"""Copy only public base resource references from an old YAML, never learned weights."""

import argparse
import json
import os
from pathlib import Path
import yaml

from ccrr_v2.runtime import validate_resources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-config", required=True)
    p.add_argument("--workspace-root", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    old = yaml.safe_load(Path(args.old_config).read_text())
    fields = {
        "panderm": ("path", "sha256", "source", "source_sha256"),
        "resnet50": ("path", "sha256"),
    }
    resources = {
        "schema": "ccrr-v2-resources-1",
        "data_root": old["data_root"],
        "workspace_root": args.workspace_root,
    }
    for name, keys in fields.items():
        resources[name] = {k: old["weights"][name][k] for k in keys}
    validate_resources(resources)
    target = Path(args.output).resolve()
    expected = Path(args.workspace_root).resolve() / "private"
    if expected not in target.parents:
        raise ValueError("Resource config belongs in the V2 private directory")
    os.umask(0o077)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as f:
        json.dump(resources, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(
        "PUBLIC_BASE_REFERENCES_VERIFIED",
        str(target),
        "old_trained_checkpoints_loaded=False",
    )


if __name__ == "__main__":
    main()
