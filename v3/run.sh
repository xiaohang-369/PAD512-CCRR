#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$ROOT/v3:$ROOT/v2:$ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
: "${V3_PYTHON:?Set V3_PYTHON explicitly; do not modify the old environment}"
exec "$V3_PYTHON" -m ccrr_v3.cli "$@"
