#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

TARGET_LAYER="${1:-${TARGET_LAYER:-18}}"

if [[ ! "$TARGET_LAYER" =~ ^-?[0-9]+$ ]]; then
    echo "Error: TARGET_LAYER must be an integer, got '$TARGET_LAYER'." >&2
    echo "Usage: bash scripts/ours_v3/run_qwen3_vl_8b_question_video_watchdog.sh [target_layer]" >&2
    exit 2
fi

export TARGET_LAYER
export CACHE_REQUESTS="${CACHE_REQUESTS:-true}"
export LMMS_EVAL_HOME="${LMMS_EVAL_HOME:-${PROJECT_ROOT}/.cache/lmms-eval}"

DONE_DIR="${WATCHDOG_DONE_DIR:-logs/watchdog_done/qwen3_vl_8b_question_video_layer${TARGET_LAYER}}"
mkdir -p "$DONE_DIR"

QUEUE_FILE="$(mktemp)"
trap 'rm -f "$QUEUE_FILE"' EXIT

append_job() {
    local job_name="$1"
    local script_path="$2"
    local done_marker="${DONE_DIR}/${job_name}.done"
    printf "%s\t%s\tTARGET_LAYER=%s CACHE_REQUESTS=true LMMS_EVAL_HOME=%q bash %q\n" \
        "$job_name" \
        "$done_marker" \
        "$TARGET_LAYER" \
        "$LMMS_EVAL_HOME" \
        "$script_path" \
        >> "$QUEUE_FILE"
}

append_job "longvideobench_question_only" "scripts/ours_v3/qwen3_vl_8b_longvideobench_question_only.sh"
append_job "longvideobench_question_tokens" "scripts/ours_v3/qwen3_vl_8b_longvideobench_question_tokens.sh"
append_job "videomme_question_only" "scripts/ours_v3/qwen3_vl_8b_videomme_question_only.sh"
append_job "videomme_question_tokens" "scripts/ours_v3/qwen3_vl_8b_videomme_question_tokens.sh"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running Qwen3-VL question video watchdog."
echo "  TARGET_LAYER=${TARGET_LAYER}"
echo "  LMMS_EVAL_HOME=${LMMS_EVAL_HOME}"
echo "  DONE_DIR=${DONE_DIR}"
echo "  CACHE_REQUESTS=${CACHE_REQUESTS}"

bash "${SCRIPT_DIR}/run_cached_watchdog.sh" "$QUEUE_FILE"
