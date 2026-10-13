#!/usr/bin/env bash
# Fresh GPU B-R3: preserved science, complete recovery every 100 transactions.
set -euo pipefail
SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=${1:?Usage: bash run_stage_b_short_multi_gpu.sh ABS_CONFIG ABS_R2_REPORT [EXISTING_RUN_DIRECTORY]}
REPORT=${2:?Missing completed R2 report}
PY=/home/zwc/anaconda3/envs/ccrr_pad512/bin/python
GPU=GPU-76f0daac-5625-a97a-38ec-efdb3ed44d54
ROOT=/data/zwc/zyh/ccrr-checks/stage-b/reconstruction-diagnostics
[[ "$CONFIG" == /* && "$REPORT" == /* ]]
test -x "$PY"
test -f "$CONFIG"
test -f "$REPORT"
BUSY=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$BUSY" ]]; then
    printf 'GPU occupied; stop:\n%s\n' "$BUSY" >&2
    exit 1
fi
export PYTHONPATH="$SOURCE/src"
export CUDA_VISIBLE_DEVICES="$GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=20261008
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
ARGS=()
if [[ $# -eq 3 ]]; then
    WORK=$3
    [[ "$WORK" == /* ]]
    test -f "$WORK/result/binding.json"
    test -f "$WORK/engineering-passed"
    "$PY" - "$WORK/result/binding.json" <<'VERIFY'
import json, sys
assert json.load(open(sys.argv[1]))["protocol"] == "B-R3-MULTI-GPU-v0.2", "Old R3 cannot migrate"
VERIFY
    ARGS+=(--resume)
else
    [[ $# -eq 2 ]]
    WORK=$(mktemp -d "$ROOT/r3-gpu-XXXXXXXX")
    printf 'R3_WORK=%s\nR3_ENGINEERING_CHECKS_START\n' "$WORK"
    cd "$SOURCE"
    CCRR_TEST_DEVICE=cuda "$PY" -m pytest tests/test_b_r3_short_multi.py tests/test_b_r3_gpu.py -q \
      2>&1 | tee "$WORK/engineering.log"
    "$PY" "$SOURCE/scripts/probe_b_r3_gpu_speed.py" --out "$WORK/speed-probe" \
      2>&1 | tee "$WORK/speed.log"
    "$PY" - "$WORK/speed-probe/report.json" <<'VERIFY'
import json, sys
r = json.load(open(sys.argv[1]))
assert r["engineering_passed"] is True
assert r["synthetic_only"] is True
VERIFY
    printf 'GPU equivalence and synthetic speed gate passed; not scientific acceptance.\n' > "$WORK/engineering-passed"
fi
nohup bash -c '
    "$@"
    rc=$?
    printf "\nR3_EXIT_CODE=%s\n" "$rc"
    exit "$rc"
' bash "$PY" -u "$SOURCE/scripts/check_stage_b_short_multi_gpu.py" \
    --config "$CONFIG" --r2-report "$REPORT" --out "$WORK/result" "${ARGS[@]}" \
    >> "$WORK/console.log" 2>&1 < /dev/null &
PID=$!
nohup bash -c '
    while kill -0 "$1" 2>/dev/null; do
        nvidia-smi -i "$2" --query-gpu=timestamp,uuid,utilization.gpu,memory.used,power.draw --format=csv,noheader
        sleep 1
    done
' bash "$PID" "$GPU" > "$WORK/gpu-utilization.csv" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$WORK/monitor.pid"
printf '%s\n' "$PID" > "$WORK/launcher.pid"
printf 'R3_LAUNCHER_PID=%s\nR3_WORK=%s\nLOG=%s/console.log\n' "$PID" "$WORK" "$WORK"
printf 'View log: tail -n 50 -F %q\n' "$WORK/console.log"
printf 'Resume (remaining cumulative budget only): bash %q %q %q %q\n' \
  "$SOURCE/scripts/run_stage_b_short_multi_gpu.sh" "$CONFIG" "$REPORT" "$WORK"
printf '3000 Purifier + 3000 matched control; independent heads; no formal B/INN.\n'
