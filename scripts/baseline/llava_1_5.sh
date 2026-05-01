#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LMMS_EVAL_ROOT="${LMMS_EVAL_ROOT:-${PROJECT_ROOT}/lmms-eval}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${HOME}/autodl-tmp/hf_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PRETRAINED="${PRETRAINED:-${HOME}/autodl-tmp/llavav-1.5-7b}"
AUTO_DOWNLOAD="${AUTO_DOWNLOAD:-false}"

NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12346}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOG_SAMPLES_SUFFIX="${LOG_SAMPLES_SUFFIX:-llava_1_5_baseline}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/logs/llava1_5}"

if [[ -n "${TASKS_CSV:-}" ]]; then
    IFS=',' read -r -a TASKS <<< "$TASKS_CSV"
else
    TASKS=("mme")
fi

if [[ ! -f "${PRETRAINED}/config.json" ]]; then
    if [[ "$AUTO_DOWNLOAD" == "true" ]]; then
        MODEL_DIR="$PRETRAINED" bash "${PROJECT_ROOT}/scripts/download_llava_1_5.sh"
    else
        echo "Missing local LLaVA-1.5 model at ${PRETRAINED}" >&2
        echo "Run: MODEL_DIR=${PRETRAINED} bash ${PROJECT_ROOT}/scripts/download_llava_1_5.sh" >&2
        exit 1
    fi
fi

cd "$LMMS_EVAL_ROOT"

for task in "${TASKS[@]}"; do
    echo "========== Evaluating task: ${task} =========="
    accelerate launch \
        --main_process_port "$MAIN_PROCESS_PORT" \
        --num_processes "$NUM_PROCESSES" \
        -m lmms_eval \
        --model llava \
        --model_args "pretrained=${PRETRAINED},device_map=auto" \
        --tasks "$task" \
        --batch_size "$BATCH_SIZE" \
        --log_samples \
        --log_samples_suffix "$LOG_SAMPLES_SUFFIX" \
        --output_path "$OUTPUT_PATH" \
        --verbosity=DEBUG
done
