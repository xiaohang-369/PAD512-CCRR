#!/usr/bin/env bash
# Confirmatory seeds for the unchanged B-R1-DIAG-v1 four-arm experiment.
set -euo pipefail
SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=/home/zwc/anaconda3/envs/ccrr_pad512/bin/python
GPU=GPU-76f0daac-5625-a97a-38ec-efdb3ed44d54

verify_report() {
    "$PY" - "$SOURCE" "$CONFIG" "$BASE_REPORT" "$@" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

source, config, baseline = map(Path, sys.argv[1:4])
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
base = json.loads(baseline.read_text())
assert base['protocol'] == 'B-R1-DIAG-v1'
assert base['seed'] == 20261008
assert base['steps_per_arm'] == 1000
assert base['disposable_optimizer_updates'] == 4000
assert base['existing_checkpoints_unchanged'] is True
assert base['formal_training_started'] is False
assert base['scientific_admission_complete'] is False
assert base['cohort']['role'] == 'fit'
assert base['cohort']['train_patients'] == 64
assert base['cohort']['holdout_patients'] == 16
assert base['cohort']['one_image_per_patient'] is True
assert digest(config) == base['config_sha256'], 'Config changed'
assert digest(source / 'scripts/probe_stage_b_reconstruction.py') == base['checker_sha256'], 'Probe changed'
actual = {str(p.relative_to(source)): digest(p) for p in sorted((source / 'src/ccrr').rglob('*.py'))}
assert actual == base['source_sha256'], 'Production source changed'
assert base['arms'] == {'original': [1.0, 1.0], 'encoder_x01': [0.1, 1.0], 'decoder_x01': [1.0, 0.1], 'both_x01': [0.1, 0.1]}
for arm, rows in base['trajectories'].items():
    assert [r['step'] for r in rows] == [0, 100, 500, 1000]
    assert all(r['arm'] == arm for r in rows)
if len(sys.argv) > 4:
    report, seed = Path(sys.argv[4]), int(sys.argv[5])
    new = json.loads(report.read_text())
    assert new['seed'] == seed
    for key in ('protocol', 'arms', 'steps_per_arm', 'disposable_optimizer_updates',
                'formal_training_started', 'scientific_admission_complete',
                'existing_checkpoints_unchanged', 'cohort', 'checkpoint_digests',
                'checker_sha256', 'config_sha256', 'source_sha256'):
        assert new[key] == base[key], 'Replication mismatch: ' + key
    for arm, rows in new['trajectories'].items():
        assert [r['step'] for r in rows] == [0, 100, 500, 1000]
        assert all(r['arm'] == arm for r in rows)
    assert set(new['trajectories']) == set(base['arms'])
    print('REPLICATE_BINDING_VERIFIED:', seed, flush=True)
else:
    print('BASELINE_SOURCE_CONFIG_VERIFIED', flush=True)
PY
}

if [[ "${1:-}" == --worker ]]; then
    CONFIG=${2:?}; BASE_REPORT=${3:?}; WORK=${4:?}
    trap 'rc=$?; printf "REPLICATION_EXIT_CODE=%s\n" "$rc"' EXIT
    verify_report
    for SEED in 20261009 20261010; do
        printf 'REPLICATE_START: seed=%s\n' "$SEED"
        "$PY" -u "$SOURCE/scripts/probe_stage_b_reconstruction.py" \
            --config "$CONFIG" --out "$WORK/seed-$SEED" \
            --steps 1000 --seed "$SEED"
        verify_report "$WORK/seed-$SEED/report.json" "$SEED"
    done
    printf 'REPLICATION_COMPLETE: %s\n' "$WORK"
    exit 0
fi

CONFIG=${1:?Usage: bash scripts/run_stage_b_reconstruction_replicates.sh ABS_CONFIG ABS_BASE_REPORT}
BASE_REPORT=${2:?Pass the completed seed-20261008 report.json}
[[ "$CONFIG" == /* && "$BASE_REPORT" == /* ]]
test -x "$PY"
verify_report
BUSY=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$BUSY" ]]; then
    printf 'GPU occupied; stop:\n%s\n' "$BUSY" >&2
    exit 1
fi
WORK=$(mktemp -d /data/zwc/zyh/ccrr-checks/stage-b/reconstruction-diagnostics/replicate-XXXXXXXX)
export PYTHONPATH="$SOURCE/src" CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUBLAS_WORKSPACE_CONFIG=:4096:8
# Retain the first run's hash seed; --seed explicitly controls model and batches.
export PYTHONHASHSEED=20261008 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
nohup bash "$SOURCE/scripts/run_stage_b_reconstruction_replicates.sh" \
    --worker "$CONFIG" "$BASE_REPORT" "$WORK" \
    > "$WORK/console.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$WORK/launcher.pid"
printf 'REPLICATION_LAUNCHER_PID=%s\nLOG=%s/console.log\n' "$PID" "$WORK"
printf 'View log: tail -n 50 -F %q\n' "$WORK/console.log"
printf 'Two sequential seeds x four arms x 1000 disposable updates; no formal B.\n'
