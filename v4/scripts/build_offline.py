"""Create a new fixed-commit offline bundle; does not contact or change a server."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


repo = Path(__file__).resolve().parents[2]
dest_root = Path(sys.argv[1]).resolve()
if subprocess.check_output(
    ["git", "-C", str(repo), "status", "--porcelain"], text=True
).strip():
    raise SystemExit("Commit all release files before packaging")
commit = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
).strip()
branch = subprocess.check_output(
    ["git", "-C", str(repo), "branch", "--show-current"], text=True
).strip()
if branch != "V4":
    raise SystemExit("Build only the explicit V4 branch")
dest = dest_root / f"PAD512-CCRR-V4-{commit[:7]}-offline"
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
(dest / "manifest.json").write_text(
    json.dumps(
        dict(
            source_commit=commit,
            branch=branch,
            author_commit="5dc2a0f7c6b692ca7914941745bd29b227ff4dbb",
            scope="source_port_and_residual_protocol_verification",
            training_enabled=False,
            server_execution="pending_user",
            scientific_acceptance="not_evaluated",
        ),
        indent=2,
    )
    + "\n"
)
(dest / "verify-package.py").write_text("""import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1])
for line in (p/'SHA256SUMS').read_text().splitlines():
    expected,name=line.split('  ',1)
    f=p/name
    if f.parent != p or hashlib.sha256(f.read_bytes()).hexdigest()!=expected:
        raise SystemExit('Package checksum failed: '+name)
print('PACKAGE_SHA256_OK')
""")
install = r"""#!/usr/bin/env bash
set -euo pipefail
V4_PACKAGE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
V4_ROOT=${V4_ROOT:-/data/zwc/zyh/PAD512-CCRR-V4}
V4_PYTHON=${V4_PYTHON:-/home/zwc/anaconda3/envs/ccrr_v2/bin/python}
V4_COMMIT=__COMMIT__
test -x "$V4_PYTHON"
"$V4_PYTHON" -B "$V4_PACKAGE/verify-package.py" "$V4_PACKAGE"
mkdir -p "$V4_ROOT/releases" "$V4_ROOT/logs" "$V4_ROOT/runs" "$V4_ROOT/configs"
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
export PYTHONPATH="$V4_RELEASE/v4:$V4_RELEASE/v3:$V4_RELEASE/v2:$V4_RELEASE/src"
export PYTHONDONTWRITEBYTECODE=1
"$V4_PYTHON" -B -c 'import sys,torch,numpy,PIL,cryptography; from ccrr_v4.verify import source_check; from ccrr_v4 import residual_transport; print("PYTHON:",sys.executable); print("TORCH:",torch.__version__); print(source_check())'
echo "V4_RELEASE: $V4_RELEASE"
echo 'END_V4_INSTALL environment_modified=False training_started=False'
""".replace("__COMMIT__", commit)
(dest / "install-code.sh").write_text(install)
run = r"""#!/usr/bin/env bash
set -euo pipefail
V4_ROOT=${V4_ROOT:-/data/zwc/zyh/PAD512-CCRR-V4}
V4_RELEASE="$V4_ROOT/releases/__COMMIT__"
test "$(git -C "$V4_RELEASE" rev-parse HEAD)" = '__COMMIT__'
exec bash "$V4_RELEASE/v4/scripts/run-verify.sh" --root "$V4_ROOT" "$@"
""".replace("__COMMIT__", commit)
(dest / "run-verify.sh").write_text(run)
residual_run = r"""#!/usr/bin/env bash
set -euo pipefail
V4_ROOT=${V4_ROOT:-/data/zwc/zyh/PAD512-CCRR-V4}
V4_PYTHON=${V4_PYTHON:-/home/zwc/anaconda3/envs/ccrr_v2/bin/python}
V4_RELEASE="$V4_ROOT/releases/__COMMIT__"
test "$(git -C "$V4_RELEASE" rev-parse HEAD)" = '__COMMIT__'
test -z "$(git -C "$V4_RELEASE" status --porcelain)"
export PYTHONPATH="$V4_RELEASE/v4:$V4_RELEASE/v3:$V4_RELEASE/v2:$V4_RELEASE/src"
export PYTHONDONTWRITEBYTECODE=1
V4_OUTPUT="$V4_ROOT/runs/residual.$(date +%Y%m%d-%H%M%S).$$.$RANDOM"
exec "$V4_PYTHON" -B -m ccrr_v4.residual_probe --config "$V4_RELEASE/v4/configs/probe-residual.json" --output "$V4_OUTPUT" "$@"
""".replace("__COMMIT__", commit)
(dest / "run-residual-probe.sh").write_text(residual_run)
for name in ("install-code.sh", "run-verify.sh", "run-residual-probe.sh"):
    (dest / name).chmod(0o755)
    subprocess.run(["bash", "-n", str(dest / name)], check=True)
for name in (
    "verify-cpu.json",
    "verify-server.json",
    "resources.example.json",
    "train-author-reference.json",
    "train-pad-design.json",
    "probe-residual.json",
):
    (dest / name).write_bytes((repo / "v4/configs" / name).read_bytes())
(dest / "README.txt").write_text(f"""V4原论文基础与差值收发验证版，提交 {commit}
本包没有训练入口。来源完整性、工程检查通过不等于医学/隐私或容量通过。

1. 上传本压缩包到服务器自己的目录，解压并进入本目录。
2. bash install-code.sh
   默认使用现有 /home/zwc/anaconda3/envs/ccrr_v2/bin/python；不安装、升级依赖。
   独立写入 /data/zwc/zyh/PAD512-CCRR-V4；旧V2/V3不修改。
   可用 V4_ROOT 与 V4_PYTHON 环境变量指定自己的根目录和已有解释器。
3. bash run-verify.sh --config "$PWD/verify-cpu.json"
   先回传CPU报告。下面GPU命令分开执行，不自动串联或训练。
   本次新增的差值收发试验使用CPU即可：
   bash run-residual-probe.sh --config "$PWD/probe-residual.json"
   合成可压缩样本应重组成功，随机样本应明确容量拒绝；两者都不是医学结果。
   如已有同一输入的实际编码图，可追加 --original-png /path/ref.png --public-png /path/public.png。
   仅接受单通道8bit PNG。试验密钥仅驻内存、结束丢弃，此命令不是生产交付加密入口。
4. 选择实际可用卡后：
   bash run-verify.sh --config "$PWD/verify-server.json" --gpus 0,2
   0,2只是示例，必须改成用户选定物理卡；也可只写一张，如2。
   本版多卡逐卡验证，不是DDP训练。不检查其他用户的GPU进程或修改环境。
5. 在实际卡上检查已有监督器（重新读原始fit16及A0，零参数更新）：
   bash run-verify.sh --config "$PWD/verify-server.json" --gpus 2 --resources "$PWD/resources.example.json"
   示例资源路径来自已有记录，启动时检查SHA绑定；有迁移请复制JSON修改路径。
   输出会明确是否源图/A0一致、教师冻结、E/D输入梯度连通；不评价test。

配置都在本包或release/v4/configs。可复制到自己的configs目录修改后--config指定，勿改固定release源码。
train-author-reference.json是作者源码参考，train-pad-design.json是尚未登记的设计，两者都不能启动训练。
未来训练全局batch和每卡batch分开；卡数不写死，不因增加卡自动倍增全局batch/学习率。
已选择原基线与学习公共灰度的整数差值；数字收发诊断使用确定的可逆载体，尚非自适应神经Writer。
当前未验证原VGG权重、未训练同域基线或监督改进，未完成最终自适应嵌入授权系统。
请回传V4_REPORT指向的report.json及终端失败信息（如有）；无需回传患者图像、预测明细或模型权重。
""")
hashes = {p.name: sha(p) for p in sorted(dest.iterdir()) if p.is_file()}
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
            hashlib.sha256(tar.extractfile(f"{dest.name}/{name}").read()).hexdigest()
            == expected
        )
result = dict(
    package=str(archive),
    source_commit=commit,
    sha256=sha(archive),
    bytes=archive.stat().st_size,
)
archive.with_suffix(archive.suffix + ".sha256").write_text(
    result["sha256"] + "  " + archive.name + "\n"
)
print(json.dumps(result, indent=2))
