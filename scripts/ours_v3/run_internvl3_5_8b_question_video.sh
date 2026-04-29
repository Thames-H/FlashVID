#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

WAIT_FOR_GPU="${WAIT_FOR_GPU:-false}"
TARGET_LAYER="${1:-${TARGET_LAYER:-18}}"

if [[ ! "$TARGET_LAYER" =~ ^-?[0-9]+$ ]]; then
    echo "Error: TARGET_LAYER must be an integer, got '$TARGET_LAYER'." >&2
    echo "Usage: bash scripts/ours_v3/run_internvl3_5_8b_question_video.sh [target_layer]" >&2
    exit 2
fi
export TARGET_LAYER
export CACHE_REQUESTS="${CACHE_REQUESTS:-false}"
export LOG_SAMPLES="${LOG_SAMPLES:-false}"

JOBS=(
    "scripts/ours_v3/internvl3_5_8b_longvideobench_question_only.sh"
    "scripts/ours_v3/internvl3_5_8b_longvideobench_question_tokens.sh"
    "scripts/ours_v3/internvl3_5_8b_videomme_question_only.sh"
    "scripts/ours_v3/internvl3_5_8b_videomme_question_tokens.sh"
)

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

if [[ "$WAIT_FOR_GPU" == "true" ]]; then
    queue_file="$(mktemp)"
    trap 'rm -f "$queue_file"' EXIT

    for job in "${JOBS[@]}"; do
        printf "TARGET_LAYER=%q CACHE_REQUESTS=%q LOG_SAMPLES=%q bash %q\n" \
            "$TARGET_LAYER" \
            "$CACHE_REQUESTS" \
            "$LOG_SAMPLES" \
            "$job" \
            >> "$queue_file"
    done

    echo "[$(timestamp)] Running InternVL3.5 question video jobs via GPU-idle queue with TARGET_LAYER=${TARGET_LAYER}, CACHE_REQUESTS=${CACHE_REQUESTS}, LOG_SAMPLES=${LOG_SAMPLES}."
    bash "${SCRIPT_DIR}/run_when_gpu_free.sh" "$queue_file"
else
    echo "[$(timestamp)] Running InternVL3.5 question video jobs sequentially with TARGET_LAYER=${TARGET_LAYER}, CACHE_REQUESTS=${CACHE_REQUESTS}, LOG_SAMPLES=${LOG_SAMPLES}."
    for job in "${JOBS[@]}"; do
        echo "[$(timestamp)] Starting: TARGET_LAYER=${TARGET_LAYER} CACHE_REQUESTS=${CACHE_REQUESTS} LOG_SAMPLES=${LOG_SAMPLES} bash $job"
        TARGET_LAYER="$TARGET_LAYER" CACHE_REQUESTS="$CACHE_REQUESTS" LOG_SAMPLES="$LOG_SAMPLES" bash "$job"
        echo "[$(timestamp)] Finished: TARGET_LAYER=${TARGET_LAYER} CACHE_REQUESTS=${CACHE_REQUESTS} LOG_SAMPLES=${LOG_SAMPLES} bash $job"
    done
fi

echo "[$(timestamp)] All InternVL3.5 question video jobs are done."
