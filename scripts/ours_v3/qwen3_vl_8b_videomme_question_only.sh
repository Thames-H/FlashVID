#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Editable configuration. Change values here instead of exporting env vars.
CUDA_VISIBLE_DEVICES="0"
LMMS_EVAL_USE_CACHE="True"
LMMS_EVAL_HOME="$PROJECT_ROOT/.cache/lmms-eval"
NUM_PROCESSES=1
MAIN_PROCESS_PORT=18902
BATCH_SIZE=1
LOG_SAMPLES_SUFFIX="qwen3_vl_ours_v3_8b_videomme_question_only"
OUTPUT_PATH="./logs/ours_v3_qwen3_vl_8b_videomme_question_only"
CACHE_REQUESTS="true"
TASKS=("videomme")

AUTODL_MODEL_PATH="$HOME/autodl-tmp/Qwen3-VL-8B-Instruct"
DEFAULT_PRETRAINED="Qwen/Qwen3-VL-8B-Instruct"
PRETRAINED="$DEFAULT_PRETRAINED"

RETENTION_RATIOS=(0.10 0.15)
SCORING_METHOD="full"
SHALLOW_LAYERS=4
TARGET_LAYER="${TARGET_LAYER:-18}"
USE_ALPHA="true"
USE_DEVIATION="true"
TWO_STAGE="false"
TEXT_CHUNK_SIZE=32
SCORING_TEXT_MODE="benchmark_question_only"
STATS_OUTPUT_PATH=""

MAX_NUM_FRAMES=32
DEVICE_MAP="auto"
MIN_PIXELS=$((64 * 28 * 28))
MAX_PIXELS=$((256 * 28 * 28))
ATTN_IMPLEMENTATION="flash_attention_2"
OPENCV_LOG_LEVEL="ERROR"
OPENCV_FFMPEG_LOGLEVEL="8"
AV_LOG_FORCE_NOCOLOR="1"
FLASHVID_SUPPRESS_DECODER_STDERR="1"

if [[ -d "$AUTODL_MODEL_PATH" ]]; then
    PRETRAINED="$AUTODL_MODEL_PATH"
fi

export CUDA_VISIBLE_DEVICES
export LMMS_EVAL_USE_CACHE
export LMMS_EVAL_HOME
export OPENCV_LOG_LEVEL
export OPENCV_FFMPEG_LOGLEVEL
export AV_LOG_FORCE_NOCOLOR
export FLASHVID_SUPPRESS_DECODER_STDERR

REQUEST_CACHE_ARGS=()
if [[ -n "$CACHE_REQUESTS" ]]; then
    REQUEST_CACHE_ARGS=(--cache_requests "$CACHE_REQUESTS")
fi

BASE_MODEL_ARGS="pretrained=$PRETRAINED,device_map=$DEVICE_MAP,max_num_frames=$MAX_NUM_FRAMES,max_pixels=$MAX_PIXELS,min_pixels=$MIN_PIXELS,attn_implementation=$ATTN_IMPLEMENTATION,scoring_method=$SCORING_METHOD,shallow_layers=$SHALLOW_LAYERS,target_layer=$TARGET_LAYER,use_alpha=$USE_ALPHA,use_deviation=$USE_DEVIATION,two_stage=$TWO_STAGE,text_chunk_size=$TEXT_CHUNK_SIZE,scoring_text_mode=$SCORING_TEXT_MODE"
if [[ -n "$STATS_OUTPUT_PATH" ]]; then
    BASE_MODEL_ARGS="$BASE_MODEL_ARGS,stats_output_path=$STATS_OUTPUT_PATH"
fi

for retention_ratio in "${RETENTION_RATIOS[@]}"; do
    echo "Running Qwen3-VL-8B-Instruct FETP-v3 VideoMME with question-only scoring, retention_ratio=${retention_ratio}"
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
        "${REQUEST_CACHE_ARGS[@]}" \
        --log_samples \
        --log_samples_suffix "$LOG_SAMPLES_SUFFIX" \
        --output_path "$OUTPUT_PATH"
    done
    echo "Finished running VideoMME with retention_ratio=${retention_ratio}"
done
