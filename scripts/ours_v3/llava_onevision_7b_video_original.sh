#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Editable configuration. Change values here instead of exporting env vars.
CUDA_VISIBLE_DEVICES="0,1,2,3"
LMMS_EVAL_USE_CACHE="True"
LMMS_EVAL_HOME="$PROJECT_ROOT/.cache/lmms-eval"
NUM_PROCESSES=4
MAIN_PROCESS_PORT=18902
BATCH_SIZE=1
LOG_SAMPLES_SUFFIX="llava_onevision_original_7b_video"
OUTPUT_PATH="./logs/original_llava_onevision_7b_video"
TASKS=("videomme" "longvideobench_val_v")

AUTODL_MODEL_PATH="$HOME/autodl-tmp/llava-onevision-qwen2-7b-ov-hf"
DEFAULT_PRETRAINED="llava-hf/llava-onevision-qwen2-7b-ov-hf"
PRETRAINED="$DEFAULT_PRETRAINED"

MAX_FRAMES_NUM=32
ATTN_IMPLEMENTATION="flash_attention_2"
DTYPE="float16"

if [[ -d "$AUTODL_MODEL_PATH" ]]; then
    PRETRAINED="$AUTODL_MODEL_PATH"
fi

export CUDA_VISIBLE_DEVICES
export LMMS_EVAL_USE_CACHE
export LMMS_EVAL_HOME

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
