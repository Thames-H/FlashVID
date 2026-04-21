#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-18896}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOG_SAMPLES_SUFFIX="${LOG_SAMPLES_SUFFIX:-llava_onevision_original_7b_img}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/original_llava_onevision_7b_img}"

if [[ -n "${TASKS_CSV:-}" ]]; then
    IFS=',' read -r -a TASKS <<< "${TASKS_CSV}"
else
    TASKS=("gqa" "scienceqa_img" "mmbench_en" "mme" "pope" "ocrbench")
fi

AUTODL_MODEL_PATH="${HOME}/autodl-tmp/llava-onevision-qwen2-7b-ov-hf"
DEFAULT_PRETRAINED="llava-hf/llava-onevision-qwen2-7b-ov-hf"
if [[ -d "$AUTODL_MODEL_PATH" ]]; then
    DEFAULT_PRETRAINED="$AUTODL_MODEL_PATH"
fi
PRETRAINED="${PRETRAINED:-$DEFAULT_PRETRAINED}"

MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-8}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
DTYPE="${DTYPE:-float16}"
BASE_MODEL_ARGS="pretrained=$PRETRAINED,max_frames_num=$MAX_FRAMES_NUM,attn_implementation=$ATTN_IMPLEMENTATION,dtype=$DTYPE"

for task in "${TASKS[@]}"; do
    echo "Evaluating task: $task"
    accelerate launch \
    --main_process_port "$MAIN_PROCESS_PORT" \
    --num_processes "$NUM_PROCESSES" \
    -m lmms_eval \
    --model llava_onevision_original \
    --model_args "$BASE_MODEL_ARGS" \
    --tasks "$task" \
    --batch_size "$BATCH_SIZE" \
    --log_samples \
    --log_samples_suffix "$LOG_SAMPLES_SUFFIX" \
    --output_path "$OUTPUT_PATH"
done
