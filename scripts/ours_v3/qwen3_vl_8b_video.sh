#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-18894}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOG_SAMPLES_SUFFIX="${LOG_SAMPLES_SUFFIX:-qwen3_vl_ours_v3_8b_video}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/ours_v3_qwen3_vl_8b_video}"

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

if [[ -n "${RETENTION_RATIOS_CSV:-}" ]]; then
    IFS=',' read -r -a RETENTION_RATIOS <<< "${RETENTION_RATIOS_CSV}"
else
    RETENTION_RATIOS=(0.05 0.10 0.20)
fi
SCORING_METHOD="${SCORING_METHOD:-full}"
SHALLOW_LAYERS="${SHALLOW_LAYERS:-4}"
TARGET_LAYER="${TARGET_LAYER:-20}"
USE_ALPHA="${USE_ALPHA:-true}"
USE_DEVIATION="${USE_DEVIATION:-true}"
TWO_STAGE="${TWO_STAGE:-false}"
TEXT_CHUNK_SIZE="${TEXT_CHUNK_SIZE:-32}"
STATS_OUTPUT_PATH="${STATS_OUTPUT_PATH:-}"

MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-32}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
BASE_MODEL_ARGS="pretrained=$PRETRAINED,max_num_frames=$MAX_NUM_FRAMES,attn_implementation=$ATTN_IMPLEMENTATION,scoring_method=$SCORING_METHOD,shallow_layers=$SHALLOW_LAYERS,target_layer=$TARGET_LAYER,use_alpha=$USE_ALPHA,use_deviation=$USE_DEVIATION,two_stage=$TWO_STAGE,text_chunk_size=$TEXT_CHUNK_SIZE"
if [[ -n "$DEVICE_MAP" ]]; then
    BASE_MODEL_ARGS="$BASE_MODEL_ARGS,device_map=$DEVICE_MAP"
fi
if [[ -n "$STATS_OUTPUT_PATH" ]]; then
    BASE_MODEL_ARGS="$BASE_MODEL_ARGS,stats_output_path=$STATS_OUTPUT_PATH"
fi

for retention_ratio in "${RETENTION_RATIOS[@]}"; do
    echo "Running Qwen3-VL-8B-Instruct FETP-v3 video benchmarks with retention_ratio=${retention_ratio}"
    MODEL_ARGS="$BASE_MODEL_ARGS,retention_ratio=${retention_ratio}"
    for task in "${TASKS[@]}"; do
        echo "Evaluating task: $task"
        accelerate launch \
        --main_process_port "$MAIN_PROCESS_PORT" \
        --num_processes "$NUM_PROCESSES" \
        -m lmms_eval \
        --model qwen3_vl_ours_v3 \
        --model_args "$MODEL_ARGS" \
        --tasks "$task" \
        --batch_size "$BATCH_SIZE" \
        --log_samples \
        --log_samples_suffix "$LOG_SAMPLES_SUFFIX" \
        --output_path "$OUTPUT_PATH"
    done
    echo "Finished running with retention_ratio=${retention_ratio}"
done
