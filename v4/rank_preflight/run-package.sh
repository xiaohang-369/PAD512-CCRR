#!/usr/bin/env bash
set -euo pipefail
RANK_PACKAGE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
RANK_PYTHON=${LAB_PYTHON:-/home/zwc/anaconda3/envs/ccrr_v2/bin/python}
RANK_ACTION=${1:-verify}
export PYTHONPATH="$RANK_PACKAGE_DIR/runtime/src:$RANK_PACKAGE_DIR/runtime/v4:$RANK_PACKAGE_DIR/runtime/v3:$RANK_PACKAGE_DIR/runtime/v2"
export PYTHONDONTWRITEBYTECODE=1
"$RANK_PYTHON" "$RANK_PACKAGE_DIR/package_tools.py" verify "$RANK_PACKAGE_DIR"
case "$RANK_ACTION" in
  verify)
    "$RANK_PYTHON" -m pytest -p no:cacheprovider "$RANK_PACKAGE_DIR/runtime/v4/rank_preflight/tests" -q
    "$RANK_PYTHON" -m pytest -p no:cacheprovider "$RANK_PACKAGE_DIR/runtime/v4/tests/test_phase2_g1.py" -q -k 'sequential_vjp or ste_has or retrieval_excludes or two_rank_gradients'
    ;;
  assets)
    "$RANK_PYTHON" "$RANK_PACKAGE_DIR/runtime/v4/rank_preflight/scripts/check_assets.py" --bindings "$RANK_PACKAGE_DIR/configs/server-assets.json"
    ;;
  zero)
    RANK_GPU=${2:-}
    if [[ ! "$RANK_GPU" =~ ^[0-9]+$ ]]; then
      echo 'Usage: bash e1d1-rank-zero-run.sh zero PHYSICAL_GPU_ID' >&2
      exit 2
    fi
    RANK_BATCH=${RANK_DIAG_BATCH:-2}
    RANK_QUERIES=${RANK_DIAG_QUERIES:-16}
    RANK_RUN_BASE=${RANK_RUN_BASE:-/data/zwc/zyh/PAD512-CCRR-V4/runs}
    mkdir -p "$RANK_RUN_BASE"
    RANK_RUN=$(mktemp -d "$RANK_RUN_BASE/E1D1-rank-zero.XXXXXXXX")
    echo "RUN_DIRECTORY=$RANK_RUN"
    trap 'RANK_EXIT=$?; trap - EXIT; "$RANK_PYTHON" "$RANK_PACKAGE_DIR/package_tools.py" export "$RANK_RUN"; exit "$RANK_EXIT"' EXIT
    "$RANK_PYTHON" "$RANK_PACKAGE_DIR/runtime/v4/rank_preflight/scripts/check_assets.py" --bindings "$RANK_PACKAGE_DIR/configs/server-assets.json" --output "$RANK_RUN/assets.json" 2>&1 | tee "$RANK_RUN/assets.log"
    "$RANK_PYTHON" "$RANK_PACKAGE_DIR/runtime/v4/rank_preflight/scripts/audit_fit.py" --root /data/zwc/zyh/data/PAD-UFES-20 --output "$RANK_RUN/fit-audit" 2>&1 | tee "$RANK_RUN/fit-audit.log"
    CUDA_VISIBLE_DEVICES="$RANK_GPU" "$RANK_PYTHON" "$RANK_PACKAGE_DIR/runtime/v4/rank_preflight/scripts/full_zero.py" --config "$RANK_PACKAGE_DIR/configs/phase2-fresh-zero.json" --device cuda:0 --output "$RANK_RUN/probe" --tau 0.1 --margin 0 --batch-size "$RANK_BATCH" --fit-queries "$RANK_QUERIES" 2>&1 | tee "$RANK_RUN/probe.log"
    ;;
  *)
    echo 'Supported actions: verify | assets | zero PHYSICAL_GPU_ID. No train action.' >&2
    exit 2
    ;;
esac
