#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

TARGET_LAYER="${1:-${TARGET_LAYER:-18}}"

if [[ ! "$TARGET_LAYER" =~ ^-?[0-9]+$ ]]; then
    echo "Error: TARGET_LAYER must be an integer, got '$TARGET_LAYER'." >&2
    echo "Usage: bash scripts/ours_v3/run_internvl3_5_8b_question_video_safe.sh [target_layer]" >&2
    exit 2
fi

export TARGET_LAYER
export CACHE_REQUESTS="${CACHE_REQUESTS:-false}"
export LOG_SAMPLES="${LOG_SAMPLES:-false}"
export WATCHDOG_RESPAWN="${WATCHDOG_RESPAWN:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export MAX_SCORING_VISUAL_TOKENS="${MAX_SCORING_VISUAL_TOKENS:-4096}"
export MAX_SCORE_TEXT_TOKENS="${MAX_SCORE_TEXT_TOKENS:-8}"
export MAX_SCORE_HEADS="${MAX_SCORE_HEADS:-4}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running InternVL3.5 safe question video watchdog."
echo "  TARGET_LAYER=${TARGET_LAYER}"
echo "  MAX_SCORING_VISUAL_TOKENS=${MAX_SCORING_VISUAL_TOKENS}"
echo "  MAX_SCORE_TEXT_TOKENS=${MAX_SCORE_TEXT_TOKENS}"
echo "  MAX_SCORE_HEADS=${MAX_SCORE_HEADS}"
echo "  CACHE_REQUESTS=${CACHE_REQUESTS}"
echo "  LOG_SAMPLES=${LOG_SAMPLES}"
echo "  WATCHDOG_RESPAWN=${WATCHDOG_RESPAWN}"

exec bash "${SCRIPT_DIR}/run_internvl3_5_8b_question_video_watchdog.sh" "${TARGET_LAYER}"
