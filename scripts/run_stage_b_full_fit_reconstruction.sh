#!/usr/bin/env bash
# Fixed B-R2-FULLFIT-v1: two reconstruction arms, 10000 updates each.
set -euo pipefail
SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=${1:?Usage: bash scripts/run_stage_b_full_fit_reconstruction.sh ABS_CONFIG ABS_BASE_REPORT ABS_REPLICATE_ROOT}
BASE_REPORT=${2:?Missing first report}
REPLICATE_ROOT=${3:?Missing completed replicate directory}
[[ "$CONFIG" == /* && "$BASE_REPORT" == /* && "$REPLICATE_ROOT" == /* ]]
PY=/home/zwc/anaconda3/envs/ccrr_pad512/bin/python
GPU=GPU-76f0daac-5625-a97a-38ec-efdb3ed44d54
test -x "$PY"
test -f "$CONFIG"
test -f "$BASE_REPORT"
test -f "$REPLICATE_ROOT/seed-20261009/report.json"
test -f "$REPLICATE_ROOT/seed-20261010/report.json"
BUSY=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$BUSY" ]]; then
    printf 'GPU occupied; stop:\n%s\n' "$BUSY" >&2
    exit 1
fi
ROOT=/data/zwc/zyh/ccrr-checks/stage-b/reconstruction-diagnostics
WORK=$(mktemp -d "$ROOT/fullfit-XXXXXXXX")
export PYTHONPATH="$SOURCE/src" CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=20261008 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
nohup bash -c '
    "$@"
    rc=$?
    printf "\nFULLFIT_EXIT_CODE=%s\n" "$rc"
    exit "$rc"
' bash "$PY" -u "$SOURCE/scripts/check_stage_b_full_fit_reconstruction.py" \
    --config "$CONFIG" --baseline-report "$BASE_REPORT" \
    --replicate-root "$REPLICATE_ROOT" --out "$WORK/result" \
    > "$WORK/console.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$WORK/launcher.pid"
printf 'FULLFIT_LAUNCHER_PID=%s\nLOG=%s/console.log\n' "$PID" "$WORK"
printf 'View log: tail -n 50 -F %q\n' "$WORK/console.log"
printf 'Two sequential reconstruction arms x 10000 updates; val_tune development evaluation; no formal B.\n'
