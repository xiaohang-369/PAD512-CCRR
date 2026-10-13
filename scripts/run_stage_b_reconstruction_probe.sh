#!/usr/bin/env bash
# B-R1-DIAG-v1. Default is read-only measurement; 1000 means 4 disposable arms.
set -euo pipefail

CONFIG=${1:?Usage: bash scripts/run_stage_b_reconstruction_probe.sh CONFIG [0|1000]}
STEPS=${2:-0}
case "$STEPS" in 0|1000) ;; *) echo 'Only 0 or 1000 steps are registered.' >&2; exit 2;; esac
SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=/home/zwc/anaconda3/envs/ccrr_pad512/bin/python
GPU=GPU-76f0daac-5625-a97a-38ec-efdb3ed44d54
test -f "$CONFIG"
test -x "$PY"
BUSY=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$BUSY" ]]; then
    printf 'GPU occupied; stop:\n%s\n' "$BUSY" >&2
    exit 1
fi
mkdir -p /data/zwc/zyh/ccrr-checks/stage-b/reconstruction-diagnostics
WORK=$(mktemp -d /data/zwc/zyh/ccrr-checks/stage-b/reconstruction-diagnostics/run-XXXXXXXX)
export PYTHONPATH="$SOURCE/src"
export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=20261008
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
cd "$SOURCE"
nohup bash -c '
    "$@"
    rc=$?
    printf "\nDIAGNOSTIC_EXIT_CODE=%s\n" "$rc"
    exit "$rc"
' bash "$PY" -u scripts/probe_stage_b_reconstruction.py \
    --config "$CONFIG" --out "$WORK/result" --steps "$STEPS" \
    > "$WORK/console.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$WORK/launcher.pid"
printf 'DIAGNOSTIC_LAUNCHER_PID=%s\nSTEPS_PER_ARM=%s\nLOG=%s/console.log\n' "$PID" "$STEPS" "$WORK"
printf 'View log: tail -n 50 -F %q\n' "$WORK/console.log"
printf 'Four arms x %s disposable updates; no formal B and no production checkpoint output.\n' "$STEPS"
