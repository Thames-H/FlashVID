#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LMMS_EVAL_ROOT="${PROJECT_ROOT}/lmms-eval"
export PYTHONPATH="${PROJECT_ROOT}:${LMMS_EVAL_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

echo "analyze sink-analysis artifacts"
python -m sink_analysis.cli --repo-root "${PROJECT_ROOT}" analyze
