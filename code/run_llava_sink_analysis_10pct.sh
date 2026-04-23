#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LIMIT="${LIMIT:-64}"
export RATIOS_CSV="${RATIOS_CSV:-10%}"
export TARGET_LAYER="${TARGET_LAYER:-15}"

bash "${SCRIPT_DIR}/run_llava_sink_analysis.sh"
