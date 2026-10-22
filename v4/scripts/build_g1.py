"""Package committed G1 source; reuse existing environment/resources without installs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--destination", required=True, type=Path)
a = p.parse_args()
repo = Path(__file__).resolve().parents[2]
if subprocess.check_output(
    ["git", "-C", str(repo), "status", "--porcelain"], text=True
).strip():
    raise SystemExit("Commit release before packaging")
if (
    subprocess.check_output(
        ["git", "-C", str(repo), "branch", "--show-current"], text=True
    ).strip()
    != "V4"
):
    raise SystemExit("Expected V4 branch")
commit = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
).strip()
short = commit[:7]
dest = a.destination / f"PAD512-CCRR-V4-g1-{short}"
dest.mkdir(parents=True, exist_ok=False)
subprocess.run(
    [
        "git",
        "-C",
        str(repo),
        "bundle",
        "create",
        str(dest / "code.bundle"),
        "refs/heads/V4",
    ],
    check=True,
)
subprocess.run(
    ["git", "-C", str(repo), "bundle", "verify", str(dest / "code.bundle")], check=True
)
shutil.copyfile(repo / "v4/configs/phase2-g1.json", dest / "phase2-g1.json")
shutil.copyfile(repo / "v4/PHASE2_G1.md", dest / "README.md")
(dest / "manifest.json").write_text(
    json.dumps(
        dict(
            commit=commit,
            scope="G1_supervised_continuation_only",
            server_execution="pending",
            original_phase1_unchanged=True,
            new_network=False,
            training_started=False,
            medical_acceptance_threshold=None,
        ),
        indent=2,
    )
    + "\n"
)
(dest / "verify-package.py").write_text("""import hashlib,pathlib,sys
root=pathlib.Path(sys.argv[1])
for line in (root/'SHA256SUMS').read_text().splitlines():
    expected,name=line.split('  ',1)
    if pathlib.Path(name).name!=name or hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
        raise SystemExit('Package checksum failed: '+name)
print('PACKAGE_SHA256_OK')
""")
install = r"""#!/usr/bin/env bash
set -euo pipefail
umask 077
V4_PACKAGE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
V4_ROOT=${V4_ROOT:-/data/zwc/zyh/PAD512-CCRR-V4}
V4_PYTHON=${V4_PYTHON:-/home/zwc/anaconda3/envs/ccrr_v2/bin/python}
V4_COMMIT=__COMMIT__
"$V4_PYTHON" -B "$V4_PACKAGE/verify-package.py" "$V4_PACKAGE"
mkdir -p "$V4_ROOT/releases" "$V4_ROOT/configs" "$V4_ROOT/runs"
V4_SOURCE="$V4_ROOT/source-offline-__SHORT__"
V4_RELEASE="$V4_ROOT/releases/$V4_COMMIT"
if [[ ! -e "$V4_SOURCE" ]]; then
    git clone --branch V4 "$V4_PACKAGE/code.bundle" "$V4_SOURCE"
fi
test "$(git -C "$V4_SOURCE" rev-parse HEAD)" = "$V4_COMMIT"
test -z "$(git -C "$V4_SOURCE" status --porcelain)"
if [[ ! -e "$V4_RELEASE" ]]; then
    git -C "$V4_SOURCE" worktree add --detach "$V4_RELEASE" "$V4_COMMIT"
fi
test "$(git -C "$V4_RELEASE" rev-parse HEAD)" = "$V4_COMMIT"
test -z "$(git -C "$V4_RELEASE" status --porcelain)"
"$V4_PYTHON" -B - "$V4_PACKAGE/phase2-g1.json" "$V4_ROOT/configs/phase2-g1-__SHORT__.json" <<'PYIN'
from pathlib import Path
import sys
source,dest=map(Path,sys.argv[1:])
if not dest.exists():
    with dest.open('xb') as f:f.write(source.read_bytes())
print('G1_CONFIG='+str(dest))
PYIN
export PYTHONPATH="$V4_RELEASE/v4:$V4_RELEASE/v3:$V4_RELEASE/v2:$V4_RELEASE/src"
"$V4_PYTHON" -B - <<'PYIN'
import torch
from ccrr_v4.phase2_g1 import run
from ccrr_v4.verify import source_check
from ccrr_v3.learning import load_separator
from ccrr_v2.reid import initialize
print('TORCH',torch.__version__)
print(source_check())
print('END_G1_INSTALL training_started=False environment_modified=False')
PYIN
"""
runner = r"""#!/usr/bin/env bash
set -euo pipefail
umask 077
V4_ROOT=${V4_ROOT:-/data/zwc/zyh/PAD512-CCRR-V4}
V4_PYTHON=${V4_PYTHON:-/home/zwc/anaconda3/envs/ccrr_v2/bin/python}
V4_RELEASE="$V4_ROOT/releases/__COMMIT__"
V4_G1_CONFIG=${V4_G1_CONFIG:-$V4_ROOT/configs/phase2-g1-__SHORT__.json}
test "$(git -C "$V4_RELEASE" rev-parse HEAD)" = '__COMMIT__'
exec "$V4_PYTHON" -B "$V4_RELEASE/v4/scripts/run-phase2-g1.py" --config "$V4_G1_CONFIG" --root "$V4_ROOT" "$@"
"""
for name, source in (("install-code.sh", install), ("phase2.sh", runner)):
    path = dest / name
    path.write_text(source.replace("__COMMIT__", commit).replace("__SHORT__", short))
    path.chmod(0o755)
(dest / "SHA256SUMS").write_text(
    "".join(
        f"{sha(path)}  {path.name}\n"
        for path in sorted(dest.iterdir())
        if path.is_file()
    )
)
archive = dest.with_suffix(".tar.gz")
with tarfile.open(archive, "w:gz") as tf:
    tf.add(dest, arcname=dest.name)
checksum = sha(archive)
Path(str(archive) + ".sha256").write_text(f"{checksum}  {archive.name}\n")
print(
    json.dumps(
        dict(package=str(archive), sha256=checksum, bytes=archive.stat().st_size)
    )
)
