"""Package a committed V4 source baseline and exact verified VGG feature weights."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--destination", type=Path, required=True)
p.add_argument("--vgg", type=Path, required=True)
a = p.parse_args()
repo = Path(__file__).resolve().parents[2]
if subprocess.check_output(
    ["git", "-C", str(repo), "status", "--porcelain"], text=True
).strip():
    raise SystemExit("Commit release files before packaging")
if (
    subprocess.check_output(
        ["git", "-C", str(repo), "branch", "--show-current"], text=True
    ).strip()
    != "V4"
):
    raise SystemExit("Explicit V4 branch required")
commit = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
).strip()
reg = json.loads((repo / "v4/upstream/vgg-provenance.json").read_text())
if sha(a.vgg) != reg["feature_sha256"]:
    raise SystemExit("Verified VGG feature weights required")
dest = a.destination / f"PAD512-CCRR-V4-phase1-{commit[:7]}"
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
for name in ("phase1-train.json", "phase1-resources.json"):
    shutil.copyfile(repo / "v4/configs" / name, dest / name)
shutil.copyfile(a.vgg, dest / "vgg19-conv4_1.npz")
shutil.copyfile(repo / "v4/upstream/vgg-provenance.json", dest / "vgg-provenance.json")
shutil.copyfile(repo / "v4/PHASE1_SERVER.md", dest / "README.md")
(dest / "manifest.json").write_text(
    json.dumps(
        dict(
            source_commit=commit,
            branch="V4",
            scope="phase1_original_E_G_D_only",
            author_commit="5dc2a0f7c6b692ca7914941745bd29b227ff4dbb",
            formal_training_started=False,
            server_execution="pending_user",
            vgg_sha256=reg["feature_sha256"],
        ),
        indent=2,
    )
    + "\n"
)
(dest / "verify-package.py").write_text("""import pathlib,hashlib,sys
p=pathlib.Path(sys.argv[1])
for line in (p/'SHA256SUMS').read_text().splitlines():
    expected,name=line.split('  ',1)
    f=p/name
    if f.parent!=p or hashlib.sha256(f.read_bytes()).hexdigest()!=expected:
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
test -x "$V4_PYTHON"
"$V4_PYTHON" -B "$V4_PACKAGE/verify-package.py" "$V4_PACKAGE"
mkdir -p "$V4_ROOT/releases" "$V4_ROOT/resources" "$V4_ROOT/configs" "$V4_ROOT/runs/phase1" "$V4_ROOT/data"
V4_SOURCE="$V4_ROOT/source-offline-${V4_COMMIT:0:7}"
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
"$V4_PYTHON" -B - "$V4_PACKAGE" "$V4_ROOT" "$V4_COMMIT" <<'PYIN'
import hashlib,json,pathlib,shutil,sys
package,root=map(pathlib.Path,sys.argv[1:3]); commit=sys.argv[3]
sha=json.loads((package/'vgg-provenance.json').read_text())['feature_sha256']
weight=root/'resources/vgg19-conv4_1.npz'
if weight.exists():
    if hashlib.sha256(weight.read_bytes()).hexdigest()!=sha:
        raise SystemExit('Existing V4 weight differs; no overwrite')
else:
    with weight.open('xb') as out, (package/weight.name).open('rb') as src:
        shutil.copyfileobj(src,out)
for name in ('phase1-train','phase1-resources'):
    dest=root/'configs'/f'{name}-{commit[:7]}.json'
    if not dest.exists():
        with dest.open('xb') as f: f.write((package/(name+'.json')).read_bytes())
print('PHASE1_CONFIG='+str(root/'configs'/f'phase1-train-{commit[:7]}.json'))
PYIN
export PYTHONPATH="$V4_RELEASE/v4:$V4_RELEASE/src"
export PYTHONDONTWRITEBYTECODE=1
"$V4_PYTHON" -B -c 'import torch,numpy,PIL; from ccrr_v4.verify import source_check; from ccrr_v4.phase1.vgg import AuthorVGG; print("TORCH",torch.__version__); print(source_check())'
echo "V4_RELEASE=$V4_RELEASE"
echo 'END_PHASE1_INSTALL training_started=False environment_modified=False'
""".replace("__COMMIT__", commit)
(dest / "install-code.sh").write_text(install)
launch = r"""#!/usr/bin/env bash
set -euo pipefail
umask 077
V4_ROOT=${V4_ROOT:-/data/zwc/zyh/PAD512-CCRR-V4}
V4_PYTHON=${V4_PYTHON:-/home/zwc/anaconda3/envs/ccrr_v2/bin/python}
V4_RELEASE="$V4_ROOT/releases/__COMMIT__"
test "$(git -C "$V4_RELEASE" rev-parse HEAD)" = '__COMMIT__'
V4_CONFIG=${V4_CONFIG:-$V4_ROOT/configs/phase1-train-__SHORT__.json}
V4_RESOURCES=${V4_RESOURCES:-$V4_ROOT/configs/phase1-resources-__SHORT__.json}
exec "$V4_PYTHON" -B "$V4_RELEASE/v4/scripts/run-phase1.py" --config "$V4_CONFIG" --resources "$V4_RESOURCES" --root "$V4_ROOT" "$@"
""".replace("__COMMIT__", commit).replace("__SHORT__", commit[:7])
(dest / "phase1.sh").write_text(launch)
for name in ("install-code.sh", "phase1.sh"):
    (dest / name).chmod(0o755)
    subprocess.run(["bash", "-n", str(dest / name)], check=True)
hashes = {f.name: sha(f) for f in sorted(dest.iterdir()) if f.is_file()}
(dest / "SHA256SUMS").write_text(
    "".join(f"{value}  {name}\n" for name, value in hashes.items())
)
archive = dest.with_suffix(".tar.gz")
if archive.exists():
    raise FileExistsError(archive)
with tarfile.open(archive, "w:gz") as tar:
    tar.add(dest, arcname=dest.name)
with tarfile.open(archive) as tar:
    for name, expected in hashes.items():
        assert (
            hashlib.sha256(tar.extractfile(dest.name + "/" + name).read()).hexdigest()
            == expected
        )
report = dict(
    package=str(archive.resolve()),
    source_commit=commit,
    sha256=sha(archive),
    bytes=archive.stat().st_size,
)
archive.with_suffix(".tar.gz.sha256").write_text(
    report["sha256"] + "  " + archive.name + "\n"
)
print(json.dumps(report, indent=2))
