#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$ROOT/v2:$ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
: "${V2_PYTHON:?Set V2_PYTHON to the dedicated V2 environment/bin/python}"
exec "$V2_PYTHON" -m ccrr_v2.cli "$@"
