#!/usr/bin/env bash
# New preflight, then continue the completed joint run in a new output directory.
set -euo pipefail
umask 077
V4_PACKAGE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export V4_ROOT=${V4_ROOT:-/data/zwc/zyh/PAD512-CCRR-V4}
export V4_PYTHON=${V4_PYTHON:-/home/zwc/anaconda3/envs/ccrr_v2/bin/python}
mkdir -p "$V4_ROOT/runs"
# Retain this file and lock for the lifetime of the background launcher, so
# running this entry twice cannot dispatch two copies of the same continuation.
exec 9>"$V4_ROOT/runs/g1-joint-resume60.lock"
flock -n 9 || { printf '此续训入口已经在运行，请先查看原日志。\n' >&2; exit 1; }
V4_PREFLIGHT_LOG=$(mktemp "$V4_ROOT/runs/g1-resume-preflight-console.XXXXXX")
printf '预检日志=%s\n' "$V4_PREFLIGHT_LOG"
bash "$V4_PACKAGE/phase2.sh" preflight 2>&1 | tee "$V4_PREFLIGHT_LOG"
V4_REPORT=$("$V4_PYTHON" -B - "$V4_PREFLIGHT_LOG" <<'PY'
import json
from pathlib import Path
import sys
reports=[s.partition('=')[2] for s in Path(sys.argv[1]).read_text().splitlines() if s.startswith('G1_REPORT=')]
if len(reports)!=1: raise SystemExit('Expected exactly one new preflight report')
p=Path(reports[0]); r=json.loads(p.read_text())
assert r['passed'] and r['mode']=='preflight' and r['optimizer_updates']==0
assert r['update_modules']==['E1','D1'] and r['reconstruction_loss_enabled']
assert r['start_epoch']==30 and r['start_step']==2700 and r['optimizer_state_updates']==2700
assert r['initial_G1_equals_G0'] is False and r['binding']['config']['epochs']==60
assert json.loads((p.parent/'status.json').read_text())['state']=='completed'
print(p)
PY
)
V4_TRAIN_LOG=$(mktemp "$V4_ROOT/runs/g1-resume-train-console.XXXXXX")
nohup env PYTHONUNBUFFERED=1 bash "$V4_PACKAGE/phase2.sh" train \
    --preflight "$V4_REPORT" > "$V4_TRAIN_LOG" 2>&1 < /dev/null &
V4_PID=$!
printf '\n续训进程 PID=%s\n训练日志=%s\n预检报告=%s\n' "$V4_PID" "$V4_TRAIN_LOG" "$V4_REPORT"
printf '查看进度：tail -n 50 -f "%s"\n' "$V4_TRAIN_LOG"
