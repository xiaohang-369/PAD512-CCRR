#!/usr/bin/env bash
# Finite diagnostics only. Existing runs and environment are read-only inputs.
set -euo pipefail
umask 077

V2_ROOT=${1:?Pass the dedicated V2 workspace root}
V2_PYTHON=${2:?Pass the existing V2 Python executable}
V2_PARENT=${3:?Pass the completed V2 ReID run}
V2_RELEASE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
test -x "$V2_PYTHON"
test -d "$V2_ROOT/logs"
test -f "$V2_ROOT/private/resources.json"
test -n "${CUDA_VISIBLE_DEVICES:?Set the permitted GPU UUID explicitly}"

export PYTHONPATH="$V2_RELEASE/v2:$V2_RELEASE/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

V2_LOG=$(mktemp "$V2_ROOT/logs/reid-diagnostics.XXXXXXXX.log")
V2_RUN_ID=$(basename "$V2_LOG" .log)
V2_OUTPUT="$V2_ROOT/runs/reid-diagnostics/$V2_RUN_ID"
exec > >(tee "$V2_LOG") 2>&1
trap 'V2_EXIT=$?; date -Is; echo "DIAGNOSTIC_EXIT=$V2_EXIT"; echo "输出：$V2_OUTPUT"; echo "日志：$V2_LOG"; echo END_V2_REID_DIAGNOSTICS' EXIT

echo BEGIN_V2_REID_DIAGNOSTICS
date -Is
echo "代码：$(git -C "$V2_RELEASE" rev-parse HEAD)"
echo "输入：$V2_PARENT"
echo "输出：$V2_OUTPUT"
echo "日志：$V2_LOG"
echo "GPU：$CUDA_VISIBLE_DEVICES（允许共享，不停止其他进程）"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES"
"$V2_PYTHON" -u -B -m ccrr_v2.reid_diagnostics \
  --resources "$V2_ROOT/private/resources.json" \
  --training-run "$V2_PARENT" --output "$V2_OUTPUT" --device cuda:0

echo '[采样覆盖汇总]'
cat "$V2_OUTPUT/coverage_summary.json"
echo '[固定模型比较：val_tune，点估计]'
cat "$V2_OUTPUT/comparison_table.tsv"
echo '[诊断状态]'
cat "$V2_OUTPUT/status.json"
