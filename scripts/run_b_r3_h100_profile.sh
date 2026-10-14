#!/usr/bin/env bash
# Launch bounded profiling only. No environment changes or full-training dispatch.
set -euo pipefail
SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
GPU_SELECTOR=GPU-76f0daac-5625-a97a-38ec-efdb3ed44d54
if [[ ${1:-} == --gpu ]]; then
    GPU_SELECTOR=${2:?Missing GPU index or UUID}
    shift 2
fi
CONFIG=${1:?Usage: bash run_b_r3_h100_profile.sh [--gpu INDEX_OR_UUID] ABS_CONFIG ABS_COMPLETED_RUN ABS_ENGINE_SOURCE [ABS_BASELINE_REPORT|--scaling|--all|--ab|--capacity]}
OLD=${2:?Missing completed run}
ENGINE=${3:?Missing archived execution engine}
PY=/home/zwc/anaconda3/envs/ccrr_pad512/bin/python
GPU=$(nvidia-smi -i "$GPU_SELECTOR" --query-gpu=uuid --format=csv,noheader,nounits)
[[ "$GPU" =~ ^GPU-[[:xdigit:]]{8}-[[:xdigit:]]{4}-[[:xdigit:]]{4}-[[:xdigit:]]{4}-[[:xdigit:]]{12}$ ]] || { printf 'Expected exactly one GPU UUID; no probe started.\n' >&2; exit 1; }
printf 'Selected GPU: %s -> %s\n' "$GPU_SELECTOR" "$GPU"
ROOT=/data/zwc/zyh/ccrr-checks/stage-b/reconstruction-diagnostics
[[ $# -eq 3 || $# -eq 4 ]]
[[ "$CONFIG" == /* && "$OLD" == /* && "$ENGINE" == /* ]]
test -x "$PY"
test -f "$CONFIG"
test -f "$OLD/result/status.json"
test -f "$ENGINE/scripts/b_r3_gpu_state.py"
exec 9>"$ROOT/.r3-h100-profile-${GPU}.lock"
flock -n 9 || { printf 'Another private H100 probe holds the lock; stop.\n' >&2; exit 1; }
BUSY=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits)
if [[ -n "$BUSY" ]]; then
    printf 'Registered GPU is occupied; no process changed, no probe started:\n%s\n' "$BUSY" >&2
    exit 1
fi
WORK=$(mktemp -d "$ROOT/r3-h100-profile-XXXXXXXX")
export CUDA_VISIBLE_DEVICES="$GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=20261008
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$ENGINE/src"
# Prevent Python from creating __pycache__ in either archived source tree.
export PYTHONDONTWRITEBYTECODE=1
ARGS=()
STEPS=100
WARMUP=5
REPEATS=3
if [[ $# -eq 4 ]]; then
    case "$4" in
        --scaling) ARGS+=(--suite scaling) ;;
        --all) ARGS+=(--suite all) ;;
        --ab) ARGS+=(--suite ab) ;;
        --capacity) ARGS+=(--suite capacity); STEPS=20; WARMUP=3; REPEATS=2 ;;
        *)
            [[ "$4" == /* ]]
            test -f "$4"
            ARGS+=(--compare-to "$4")
            ;;
    esac
fi
nohup bash -c '
    "$@"
    rc=$?
    printf "\nH100_PROFILE_EXIT_CODE=%s\n" "$rc"
    exit "$rc"
' bash "$PY" -B -u "$SOURCE/scripts/profile_b_r3_h100.py" \
    --gpu-uuid "$GPU" \
    --engine-source "$ENGINE" --config "$CONFIG" --run "$OLD" --out "$WORK/result" \
    --steps "$STEPS" --warmup "$WARMUP" --repeats "$REPEATS" --profile-steps 3 --equivalence-steps 3 \
    --max-seconds 1800 "${ARGS[@]}" > "$WORK/console.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$WORK/launcher.pid"
printf 'H100_PROFILE_WORK=%s\nH100_PROFILE_LAUNCHER_PID=%s\n' "$WORK" "$!"
printf 'Watch: tail -n 60 -F %q\n' "$WORK/console.log"
printf 'Background probe submitted, NOT completed. Follow Watch above; final status is H100_PROFILE_EXIT_CODE.\n'
printf 'This is a bounded real-data performance probe; no full Stage B or R3 restart.\n'
