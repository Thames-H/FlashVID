#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LMMS_EVAL_ROOT="${PROJECT_ROOT}/lmms-eval"
cd "${LMMS_EVAL_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-18904}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOG_SAMPLES_SUFFIX="${LOG_SAMPLES_SUFFIX:-llava_onevision1_5_video}"
OUTPUT_PATH="${OUTPUT_PATH:-${LMMS_EVAL_ROOT}/logs/llava_onevision1_5_video}"

if [[ -n "${TASKS_CSV:-}" ]]; then
    IFS=',' read -r -a TASKS <<< "${TASKS_CSV}"
else
    TASKS=("videomme" "longvideobench_val_v")
fi

AUTODL_MODEL_PATH="${HOME}/autodl-tmp/LLaVA-OneVision-1.5-8B-Instruct"
DEFAULT_PRETRAINED="lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
if [[ -d "$AUTODL_MODEL_PATH" ]]; then
    DEFAULT_PRETRAINED="$AUTODL_MODEL_PATH"
fi
PRETRAINED="${PRETRAINED:-$DEFAULT_PRETRAINED}"

MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-32}"
MAX_PIXELS="${MAX_PIXELS:-1605632}"
MIN_PIXELS="${MIN_PIXELS:-200704}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
BASE_MODEL_ARGS="pretrained=$PRETRAINED,max_num_frames=$MAX_NUM_FRAMES,max_pixels=$MAX_PIXELS,min_pixels=$MIN_PIXELS,attn_implementation=$ATTN_IMPLEMENTATION"

for task in "${TASKS[@]}"; do
    echo "Evaluating task: $task"
    accelerate launch \
        --main_process_port "$MAIN_PROCESS_PORT" \
        --num_processes "$NUM_PROCESSES" \
        -m lmms_eval \
        --model llava_onevision1_5 \
        --model_args "$BASE_MODEL_ARGS" \
        --tasks "$task" \
        --batch_size "$BATCH_SIZE" \
        --log_samples \
        --log_samples_suffix "$LOG_SAMPLES_SUFFIX" \
        --output_path "$OUTPUT_PATH"
done
