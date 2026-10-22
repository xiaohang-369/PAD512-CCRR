#!/usr/bin/env bash
# Package entry: independent preflight, then background E1-only training.
set -euo pipefail
umask 077
V4_PACKAGE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export V4_ROOT=${V4_ROOT:-/data/zwc/zyh/PAD512-CCRR-V4}
export V4_PYTHON=${V4_PYTHON:-/home/zwc/anaconda3/envs/ccrr_v2/bin/python}
mkdir -p "$V4_ROOT/runs"
V4_E1_PREFLIGHT_LOG=$(mktemp "$V4_ROOT/runs/g1-e1only-preflight-console.XXXXXX")
printf '预检日志=%s\n' "$V4_E1_PREFLIGHT_LOG"
if bash "$V4_PACKAGE/phase2.sh" preflight 2>&1 | tee "$V4_E1_PREFLIGHT_LOG"; then
    V4_E1_REPORT=$("$V4_PYTHON" -B - "$V4_E1_PREFLIGHT_LOG" <<'PY'
import json
from pathlib import Path
import sys
lines=Path(sys.argv[1]).read_text().splitlines()
reports=[line.partition('=')[2] for line in lines if line.startswith('G1_REPORT=')]
if len(reports)!=1:
    raise SystemExit('Expected exactly one report from this preflight')
p=Path(reports[0])
r=json.loads(p.read_text())
assert r['passed'] and r['mode']=='preflight' and r['optimizer_updates']==0
assert r['update_modules']==['E1'] and r['reconstruction_loss_enabled'] is False
assert r['binding']['config']['schema']=='ccrr-v4-phase2-g1-e1only-1'
print(p)
PY
)
    V4_E1_TRAIN_LOG=$(mktemp "$V4_ROOT/runs/g1-e1only-train-console.XXXXXX")
    nohup env PYTHONUNBUFFERED=1 bash "$V4_PACKAGE/phase2.sh" train \
        --preflight "$V4_E1_REPORT" > "$V4_E1_TRAIN_LOG" 2>&1 < /dev/null &
    V4_E1_PID=$!
    printf '\n训练进程 PID=%s\n训练日志=%s\n预检报告=%s\n' "$V4_E1_PID" "$V4_E1_TRAIN_LOG" "$V4_E1_REPORT"
    printf '查看进度：tail -n 50 -f "%s"\n' "$V4_E1_TRAIN_LOG"
else
    printf '\n预检失败，未启动此对照训练。日志：%s\n' "$V4_E1_PREFLIGHT_LOG" >&2
    exit 1
fi
