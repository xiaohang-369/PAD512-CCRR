#!/usr/bin/env bash
# Worker called from the private, revision-pinned server launcher.
set -euo pipefail
umask 077
B_BASE_CONFIG=$1
B_PROFILE_RUN=$2
B_RUN=$3
B_SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
B_PY=/home/zwc/anaconda3/envs/ccrr_pad512/bin/python
export PYTHONPATH="$B_SOURCE/src:$B_SOURCE/scripts"
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=20261008
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
trap 'rc=$?; printf "%s\n" "$rc" > "$B_RUN/exit-code.txt"; printf "STAGE_B_WORKER_EXIT=%s\n" "$rc"' EXIT
printf 'STEP 1/3: verify existing micro16 evidence and frozen teacher metrics\n'
"$B_PY" "$B_SOURCE/scripts/check_stage_b_idle.py"
"$B_PY" "$B_SOURCE/scripts/prepare_formal_b.py" \
  --config "$B_BASE_CONFIG" --profile-run "$B_PROFILE_RUN" --out "$B_RUN"
printf 'STEP 2/3: one micro16 initialization/recovery check, three disposable updates per replay\n'
"$B_PY" "$B_SOURCE/scripts/check_stage_b_host.py" \
  --config "$B_RUN/startup-input.yaml" --out "$B_RUN/check" --microbatches 16 --steps 3
printf 'STEP 3/3: conditions passed; preflight then NEW formal Stage B (10000 Purifier updates)\n'
"$B_PY" "$B_SOURCE/scripts/check_stage_b_idle.py"
"$B_PY" -m ccrr.cli train --stage B --config "$B_RUN/check/candidate-config.yaml"
printf 'STAGE_B_COMMAND_FINISHED; inspect acceptance.json for scientific outcome\n'
