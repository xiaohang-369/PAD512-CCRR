#!/usr/bin/env bash
set -euo pipefail
V4_SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
V4_PYTHON=${V4_PYTHON:-/home/zwc/anaconda3/envs/ccrr_v2/bin/python}
V4_ROOT=${V4_ROOT:-/data/zwc/zyh/PAD512-CCRR-V4}
V4_CONFIG="$V4_SOURCE/v4/configs/verify-cpu.json"
V4_GPUS=""
V4_RESOURCES=""
while (($#)); do
  case "$1" in
    --config) V4_CONFIG=$2; shift 2 ;;
    --gpus) V4_GPUS=$2; shift 2 ;;
    --resources) V4_RESOURCES=$2; shift 2 ;;
    --root) V4_ROOT=$2; shift 2 ;;
    --python) V4_PYTHON=$2; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
test -x "$V4_PYTHON"
test -f "$V4_CONFIG"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$V4_SOURCE/v4:$V4_SOURCE/v3:$V4_SOURCE/v2:$V4_SOURCE/src"
V4_ARGS=()
if [[ -n "$V4_GPUS" ]]; then
  [[ "$V4_GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Use --gpus 0 or --gpus 0,2 (examples only)' >&2; exit 2; }
  export CUDA_VISIBLE_DEVICES="$V4_GPUS"
  V4_ARGS+=(--devices all-cuda)
else
  "$V4_PYTHON" -B -c 'import json,sys; c=json.load(open(sys.argv[1])); assert all(d=="cpu" for d in c["devices"]), "CUDA verification requires explicit --gpus; no physical card is assumed"' "$V4_CONFIG"
  export CUDA_VISIBLE_DEVICES=""
fi
if [[ -n "$V4_RESOURCES" ]]; then
  test -f "$V4_RESOURCES"
  V4_ARGS+=(--resources "$V4_RESOURCES")
fi
mkdir -p "$V4_ROOT/runs" "$V4_ROOT/logs"
V4_RUN="$V4_ROOT/runs/verify.$(date -u +%Y%m%d-%H%M%S).$$.$RANDOM"
V4_LOG=$(mktemp "$V4_ROOT/logs/verify.XXXXXXXX.log")
echo "BEGIN_V4_VERIFY source=$V4_SOURCE config=$V4_CONFIG selected_gpus=${V4_GPUS:-cpu}"
echo "LOG: $V4_LOG"
"$V4_PYTHON" -B -m ccrr_v4.verify --config "$V4_CONFIG" --output "$V4_RUN" "${V4_ARGS[@]}" 2>&1 | tee "$V4_LOG"
echo 'END_V4_VERIFY training_started=False optimizer_updates=0'
