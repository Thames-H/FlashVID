#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Editable configuration. Change values here instead of exporting env vars.
CUDA_VISIBLE_DEVICES="0,1,2,3"
NUM_PROCESSES=4
MAIN_PROCESS_PORT=18894
BATCH_SIZE=1
LOG_SAMPLES_SUFFIX="qwen3_vl_ours_v3_8b_video"
OUTPUT_PATH="./logs/ours_v3_qwen3_vl_8b_video"
TASKS=("videomme" "longvideobench_val_v")

AUTODL_MODEL_PATH="$HOME/autodl-tmp/Qwen3-VL-8B-Instruct"
DEFAULT_PRETRAINED="Qwen/Qwen3-VL-8B-Instruct"
PRETRAINED="$DEFAULT_PRETRAINED"

RETENTION_RATIOS=(0.05 0.10 0.20)
SCORING_METHOD="full"
SHALLOW_LAYERS=4
TARGET_LAYER=20
USE_ALPHA="true"
USE_DEVIATION="true"
TWO_STAGE="false"
TEXT_CHUNK_SIZE=32
STATS_OUTPUT_PATH=""

MAX_NUM_FRAMES=32
ATTN_IMPLEMENTATION="flash_attention_2"

if [[ -d "$AUTODL_MODEL_PATH" ]]; then
    PRETRAINED="$AUTODL_MODEL_PATH"
fi

export CUDA_VISIBLE_DEVICES

BASE_MODEL_ARGS="pretrained=$PRETRAINED,max_num_frames=$MAX_NUM_FRAMES,attn_implementation=$ATTN_IMPLEMENTATION,scoring_method=$SCORING_METHOD,shallow_layers=$SHALLOW_LAYERS,target_layer=$TARGET_LAYER,use_alpha=$USE_ALPHA,use_deviation=$USE_DEVIATION,two_stage=$TWO_STAGE,text_chunk_size=$TEXT_CHUNK_SIZE"
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
