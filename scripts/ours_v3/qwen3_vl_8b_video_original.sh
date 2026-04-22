#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-18896}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOG_SAMPLES_SUFFIX="${LOG_SAMPLES_SUFFIX:-qwen3_vl_original_8b_video}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/qwen3_vl_original_8b_video}"

DEVICE_MAP_DEFAULT=""
if [[ "$NUM_PROCESSES" == "1" ]]; then
    DEVICE_MAP_DEFAULT="auto"
fi
DEVICE_MAP="${DEVICE_MAP:-$DEVICE_MAP_DEFAULT}"
if [[ "$NUM_PROCESSES" != "1" && "$DEVICE_MAP" == "auto" ]]; then
    echo "Error: device_map=auto requires NUM_PROCESSES=1."
    exit 1
fi

if [[ -n "${TASKS_CSV:-}" ]]; then
    IFS=',' read -r -a TASKS <<< "${TASKS_CSV}"
else
    TASKS=("videomme" "longvideobench_val_v")
fi

AUTODL_MODEL_PATH="${HOME}/autodl-tmp/Qwen3-VL-8B-Instruct"
DEFAULT_PRETRAINED="Qwen/Qwen3-VL-8B-Instruct"
if [[ -d "$AUTODL_MODEL_PATH" ]]; then
    DEFAULT_PRETRAINED="$AUTODL_MODEL_PATH"
fi
PRETRAINED="${PRETRAINED:-$DEFAULT_PRETRAINED}"

MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-32}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
BASE_MODEL_ARGS="pretrained=$PRETRAINED,max_num_frames=$MAX_NUM_FRAMES,attn_implementation=$ATTN_IMPLEMENTATION"
if [[ -n "${MIN_PIXELS:-}" ]]; then
    BASE_MODEL_ARGS="$BASE_MODEL_ARGS,min_pixels=$MIN_PIXELS"
fi
if [[ -n "${MAX_PIXELS:-}" ]]; then
    BASE_MODEL_ARGS="$BASE_MODEL_ARGS,max_pixels=$MAX_PIXELS"
fi
if [[ -n "$DEVICE_MAP" ]]; then
    BASE_MODEL_ARGS="$BASE_MODEL_ARGS,device_map=$DEVICE_MAP"
fi

for task in "${TASKS[@]}"; do
    echo "Running Qwen3-VL-8B-Instruct original video benchmark"
    echo "Evaluating task: $task"
    accelerate launch \
        --main_process_port "$MAIN_PROCESS_PORT" \
        --num_processes "$NUM_PROCESSES" \
        -m lmms_eval \
        --model qwen3_vl_original \
        --model_args "$BASE_MODEL_ARGS" \
        --tasks "$task" \
        --batch_size "$BATCH_SIZE" \
        --log_samples \
        --log_samples_suffix "$LOG_SAMPLES_SUFFIX" \
        --output_path "$OUTPUT_PATH"
done
