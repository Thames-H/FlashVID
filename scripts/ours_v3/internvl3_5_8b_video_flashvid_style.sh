#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# FlashVID-style evaluation harness. InternVL was not one of the original
# FlashVID scripts, so this mirrors the same lmms-eval loop and task suite.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-18888}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CACHE_REQUESTS="${CACHE_REQUESTS:-true}"
LMMS_EVAL_USE_CACHE="${LMMS_EVAL_USE_CACHE:-True}"
LMMS_EVAL_HOME="${LMMS_EVAL_HOME:-${PROJECT_ROOT}/.cache/lmms-eval}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/ours_v3_flashvid_style/internvl3_5_8b}"
LOG_SAMPLES_SUFFIX="${LOG_SAMPLES_SUFFIX:-internvl3_5_ours_v3_8b_flashvid_style}"

TASKS_CSV="${TASKS_CSV:-videomme,egoschema,mvbench,longvideobench_val_v,mlvu_test}"
RETENTION_RATIOS_CSV="${RETENTION_RATIOS_CSV:-0.10,0.15,0.20,0.25}"
IFS=',' read -r -a TASKS <<< "$TASKS_CSV"
IFS=',' read -r -a RETENTION_RATIOS <<< "$RETENTION_RATIOS_CSV"

AUTODL_MODEL_PATH="$HOME/autodl-tmp/InternVL3_5-8B-HF"
LEGACY_AUTODL_MODEL_PATH="$HOME/autodl-tmp/InternVL3_5-8B"
PRETRAINED="${PRETRAINED:-OpenGVLab/InternVL3_5-8B-HF}"
if [[ -d "$AUTODL_MODEL_PATH" && "$PRETRAINED" == "OpenGVLab/InternVL3_5-8B-HF" ]]; then
    PRETRAINED="$AUTODL_MODEL_PATH"
fi

DEVICE_MAP="${DEVICE_MAP:-auto}"
MIN_PATCHES="${MIN_PATCHES:-1}"
MAX_PATCHES="${MAX_PATCHES:-12}"
NUM_FRAMES="${NUM_FRAMES:-32}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
LOW_CPU_MEM_USAGE="${LOW_CPU_MEM_USAGE:-true}"

SCORING_METHOD="${SCORING_METHOD:-full}"
SHALLOW_LAYERS="${SHALLOW_LAYERS:-4}"
TARGET_LAYER="${1:-${TARGET_LAYER:-18}}"
USE_ALPHA="${USE_ALPHA:-true}"
USE_DEVIATION="${USE_DEVIATION:-true}"
TWO_STAGE="${TWO_STAGE:-false}"
CANDIDATE_RATIO="${CANDIDATE_RATIO:-1.0}"
MAX_SCORING_VISUAL_TOKENS="${MAX_SCORING_VISUAL_TOKENS:-16384}"
MAX_SCORE_TEXT_TOKENS="${MAX_SCORE_TEXT_TOKENS:-16}"
MAX_SCORE_HEADS="${MAX_SCORE_HEADS:-8}"
TEXT_CHUNK_SIZE="${TEXT_CHUNK_SIZE:-32}"
SCORING_TEXT_MODE="${SCORING_TEXT_MODE:-benchmark_question}"

OPENCV_LOG_LEVEL="${OPENCV_LOG_LEVEL:-ERROR}"
OPENCV_FFMPEG_LOGLEVEL="${OPENCV_FFMPEG_LOGLEVEL:-8}"
AV_LOG_FORCE_NOCOLOR="${AV_LOG_FORCE_NOCOLOR:-1}"
FLASHVID_SUPPRESS_DECODER_STDERR="${FLASHVID_SUPPRESS_DECODER_STDERR:-1}"

if [[ ! "$TARGET_LAYER" =~ ^-?[0-9]+$ ]]; then
    echo "Error: TARGET_LAYER must be an integer, got '$TARGET_LAYER'." >&2
    echo "Usage: bash scripts/ours_v3/internvl3_5_8b_video_flashvid_style.sh [target_layer]" >&2
    exit 2
fi

if [[ "$PRETRAINED" == "$LEGACY_AUTODL_MODEL_PATH" ]]; then
    echo "Error: '$PRETRAINED' is the original InternVL chat-format checkpoint."
    echo "internvl3_5_ours_v3 expects the HF-format checkpoint:"
    echo "  - OpenGVLab/InternVL3_5-8B-HF"
    echo "  - or a local directory such as $HOME/autodl-tmp/InternVL3_5-8B-HF"
    exit 1
fi

export CUDA_VISIBLE_DEVICES
export LMMS_EVAL_USE_CACHE
export LMMS_EVAL_HOME
export PYTORCH_CUDA_ALLOC_CONF
export OPENCV_LOG_LEVEL
export OPENCV_FFMPEG_LOGLEVEL
export AV_LOG_FORCE_NOCOLOR
export FLASHVID_SUPPRESS_DECODER_STDERR

REQUEST_CACHE_ARGS=()
if [[ -n "$CACHE_REQUESTS" ]]; then
    REQUEST_CACHE_ARGS=(--cache_requests "$CACHE_REQUESTS")
fi

BASE_MODEL_ARGS="pretrained=$PRETRAINED,device_map=$DEVICE_MAP,min_patches=$MIN_PATCHES,max_patches=$MAX_PATCHES,num_frames=$NUM_FRAMES,attn_implementation=$ATTN_IMPLEMENTATION,low_cpu_mem_usage=$LOW_CPU_MEM_USAGE,scoring_method=$SCORING_METHOD,shallow_layers=$SHALLOW_LAYERS,target_layer=$TARGET_LAYER,use_alpha=$USE_ALPHA,use_deviation=$USE_DEVIATION,two_stage=$TWO_STAGE,candidate_ratio=$CANDIDATE_RATIO,max_scoring_visual_tokens=$MAX_SCORING_VISUAL_TOKENS,text_chunk_size=$TEXT_CHUNK_SIZE,scoring_text_mode=$SCORING_TEXT_MODE"
if [[ "$MAX_SCORE_TEXT_TOKENS" != "0" ]]; then
    BASE_MODEL_ARGS="$BASE_MODEL_ARGS,max_score_text_tokens=$MAX_SCORE_TEXT_TOKENS"
fi
if [[ "$MAX_SCORE_HEADS" != "0" ]]; then
    BASE_MODEL_ARGS="$BASE_MODEL_ARGS,max_score_heads=$MAX_SCORE_HEADS"
fi

for retention_ratio in "${RETENTION_RATIOS[@]}"; do
    echo "Running InternVL3.5-8B ours_v3 FlashVID-style eval with retention_ratio=${retention_ratio}"
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
            --log_samples \
            --log_samples_suffix "$LOG_SAMPLES_SUFFIX" \
            --output_path "$OUTPUT_PATH"
    done
    echo "Finished InternVL3.5 retention_ratio=${retention_ratio}"
done
