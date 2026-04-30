#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Default inference configuration. Override with env vars or edit values here.
CUDA_VISIBLE_DEVICES="0"
LMMS_EVAL_USE_CACHE="True"
LMMS_EVAL_HOME="${LMMS_EVAL_HOME:-$PROJECT_ROOT/.cache/lmms-eval}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
NUM_PROCESSES=1
MAIN_PROCESS_PORT=18911
BATCH_SIZE=1
LOW_CPU_MEM_USAGE="${LOW_CPU_MEM_USAGE:-true}"
LOG_SAMPLES_SUFFIX="internvl3_5_ours_v3_8b_videomme_question_tokens"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/ours_v3_internvl3_5_8b_videomme_question_tokens}"
CACHE_REQUESTS="${CACHE_REQUESTS:-false}"
LOG_SAMPLES="${LOG_SAMPLES:-false}"
TASKS=("videomme")

AUTODL_MODEL_PATH="$HOME/autodl-tmp/InternVL3_5-8B-HF"
LEGACY_AUTODL_MODEL_PATH="$HOME/autodl-tmp/InternVL3_5-8B"
DEFAULT_PRETRAINED="OpenGVLab/InternVL3_5-8B-HF"
PRETRAINED="$DEFAULT_PRETRAINED"

RETENTION_RATIOS=(0.10 0.15)
SCORING_METHOD="full"
SHALLOW_LAYERS=4
TARGET_LAYER="${TARGET_LAYER:-18}"
USE_ALPHA="true"
USE_DEVIATION="true"
TWO_STAGE="false"
CANDIDATE_RATIO=1.0
MAX_SCORING_VISUAL_TOKENS="${MAX_SCORING_VISUAL_TOKENS:-16384}"
MAX_SCORE_TEXT_TOKENS="${MAX_SCORE_TEXT_TOKENS:-16}"
MAX_SCORE_HEADS="${MAX_SCORE_HEADS:-8}"
TEXT_CHUNK_SIZE=32
SCORING_TEXT_MODE="benchmark_question"

DEVICE_MAP="auto"
MIN_PATCHES=1
MAX_PATCHES=1
NUM_FRAMES=32
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
export PYTORCH_CUDA_ALLOC_CONF
export OPENCV_LOG_LEVEL
export OPENCV_FFMPEG_LOGLEVEL
export AV_LOG_FORCE_NOCOLOR
export FLASHVID_SUPPRESS_DECODER_STDERR

if [[ "$PRETRAINED" == "$LEGACY_AUTODL_MODEL_PATH" ]]; then
    echo "Error: '$PRETRAINED' is the original InternVL chat-format checkpoint."
    echo "internvl3_5_ours_v3 expects the HF-format checkpoint:"
    echo "  - OpenGVLab/InternVL3_5-8B-HF"
    echo "  - or a local directory such as $HOME/autodl-tmp/InternVL3_5-8B-HF"
    exit 1
fi

REQUEST_CACHE_ARGS=()
if [[ -n "$CACHE_REQUESTS" && "$CACHE_REQUESTS" != "false" ]]; then
    REQUEST_CACHE_ARGS=(--cache_requests "$CACHE_REQUESTS")
fi

LOG_SAMPLE_ARGS=()
if [[ "$LOG_SAMPLES" == "true" ]]; then
    LOG_SAMPLE_ARGS=(--log_samples --log_samples_suffix "$LOG_SAMPLES_SUFFIX")
fi

BASE_MODEL_ARGS="pretrained=$PRETRAINED,device_map=$DEVICE_MAP,min_patches=$MIN_PATCHES,max_patches=$MAX_PATCHES,num_frames=$NUM_FRAMES,attn_implementation=$ATTN_IMPLEMENTATION,low_cpu_mem_usage=$LOW_CPU_MEM_USAGE,scoring_method=$SCORING_METHOD,shallow_layers=$SHALLOW_LAYERS,target_layer=$TARGET_LAYER,use_alpha=$USE_ALPHA,use_deviation=$USE_DEVIATION,two_stage=$TWO_STAGE,candidate_ratio=$CANDIDATE_RATIO,max_scoring_visual_tokens=$MAX_SCORING_VISUAL_TOKENS,text_chunk_size=$TEXT_CHUNK_SIZE,scoring_text_mode=$SCORING_TEXT_MODE"
if [[ "$MAX_SCORE_TEXT_TOKENS" != "0" ]]; then
    BASE_MODEL_ARGS="$BASE_MODEL_ARGS,max_score_text_tokens=$MAX_SCORE_TEXT_TOKENS"
fi
if [[ "$MAX_SCORE_HEADS" != "0" ]]; then
    BASE_MODEL_ARGS="$BASE_MODEL_ARGS,max_score_heads=$MAX_SCORE_HEADS"
fi

for retention_ratio in "${RETENTION_RATIOS[@]}"; do
    echo "Running InternVL3.5-8B FETP VideoMME with question-token scoring, retention_ratio=${retention_ratio}"
    MODEL_ARGS="$BASE_MODEL_ARGS,retention_ratio=${retention_ratio}"
    for task in "${TASKS[@]}"; do
        echo "Evaluating task: $task"
        accelerate launch \
            --main_process_port "$MAIN_PROCESS_PORT" \
            --num_processes "$NUM_PROCESSES" \
            -m lmms_eval \
            --model internvl3_5_ours_v3 \
            --model_args "$MODEL_ARGS" \
            --tasks "$task" \
            --batch_size "$BATCH_SIZE" \
            "${REQUEST_CACHE_ARGS[@]}" \
            "${LOG_SAMPLE_ARGS[@]}" \
            --output_path "$OUTPUT_PATH"
    done
    echo "Finished running VideoMME with retention_ratio=${retention_ratio}"
done
