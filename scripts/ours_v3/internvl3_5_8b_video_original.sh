#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export LMMS_EVAL_USE_CACHE="${LMMS_EVAL_USE_CACHE:-True}"
export LMMS_EVAL_HOME="${LMMS_EVAL_HOME:-$PROJECT_ROOT/.cache/lmms-eval}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-18897}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOG_SAMPLES_SUFFIX="${LOG_SAMPLES_SUFFIX:-internvl3_5_original_8b_video}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/original_internvl3_5_8b_video}"

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

AUTODL_MODEL_PATH="${HOME}/autodl-tmp/InternVL3_5-8B"
DEFAULT_PRETRAINED="OpenGVLab/InternVL3_5-8B"
if [[ -d "$AUTODL_MODEL_PATH" ]]; then
    DEFAULT_PRETRAINED="$AUTODL_MODEL_PATH"
fi
PRETRAINED="${PRETRAINED:-$DEFAULT_PRETRAINED}"

NUM_FRAME="${NUM_FRAME:-8}"
MAX_NUM="${MAX_NUM:-1}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-true}"
MODEL_ARGS="pretrained=$PRETRAINED,modality=video,num_frame=$NUM_FRAME,max_num=$MAX_NUM,use_flash_attn=$USE_FLASH_ATTN"
if [[ -n "$DEVICE_MAP" ]]; then
    MODEL_ARGS="$MODEL_ARGS,device_map=$DEVICE_MAP"
fi

for task in "${TASKS[@]}"; do
    echo "Running InternVL3.5-8B original video benchmark"
    echo "Evaluating task: $task"
    accelerate launch \
        --main_process_port "$MAIN_PROCESS_PORT" \
        --num_processes "$NUM_PROCESSES" \
        -m lmms_eval \
        --model internvl3_5_original \
        --model_args "$MODEL_ARGS" \
        --tasks "$task" \
        --batch_size "$BATCH_SIZE" \
        --log_samples \
        --log_samples_suffix "$LOG_SAMPLES_SUFFIX" \
        --output_path "$OUTPUT_PATH"
done
