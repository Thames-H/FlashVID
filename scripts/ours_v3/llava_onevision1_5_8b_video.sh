#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LMMS_EVAL_ROOT="${PROJECT_ROOT}/lmms-eval"
cd "${LMMS_EVAL_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export LMMS_EVAL_USE_CACHE="${LMMS_EVAL_USE_CACHE:-True}"
export LMMS_EVAL_HOME="${LMMS_EVAL_HOME:-${PROJECT_ROOT}/.cache/lmms-eval}"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-18906}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOG_SAMPLES_SUFFIX="${LOG_SAMPLES_SUFFIX:-llava_onevision1_5_ours_v3_8b_video}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/logs/ours_v3_llava_onevision1_5_8b_video}"
CACHE_REQUESTS="${CACHE_REQUESTS:-true}"

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

RETENTION_RATIOS_CSV="${RETENTION_RATIOS_CSV:-0.05,0.10,0.20}"
IFS=',' read -r -a RETENTION_RATIOS <<< "${RETENTION_RATIOS_CSV}"

SCORING_METHOD="${SCORING_METHOD:-full}"
SHALLOW_LAYERS="${SHALLOW_LAYERS:-4}"
TARGET_LAYER="${TARGET_LAYER:-20}"
USE_ALPHA="${USE_ALPHA:-true}"
USE_DEVIATION="${USE_DEVIATION:-true}"
TWO_STAGE="${TWO_STAGE:-false}"
TEXT_CHUNK_SIZE="${TEXT_CHUNK_SIZE:-32}"

MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-32}"
MAX_PIXELS="${MAX_PIXELS:-1605632}"
MIN_PIXELS="${MIN_PIXELS:-200704}"
FPS="${FPS:-}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
DEVICE="${DEVICE:-}"
DEVICE_MAP="${DEVICE_MAP:-}"

REQUEST_CACHE_ARGS=()
if [[ -n "$CACHE_REQUESTS" ]]; then
    REQUEST_CACHE_ARGS=(--cache_requests "$CACHE_REQUESTS")
fi

BASE_MODEL_ARGS="pretrained=$PRETRAINED,max_num_frames=$MAX_NUM_FRAMES,max_pixels=$MAX_PIXELS,min_pixels=$MIN_PIXELS,attn_implementation=$ATTN_IMPLEMENTATION,scoring_method=$SCORING_METHOD,shallow_layers=$SHALLOW_LAYERS,target_layer=$TARGET_LAYER,use_alpha=$USE_ALPHA,use_deviation=$USE_DEVIATION,two_stage=$TWO_STAGE,text_chunk_size=$TEXT_CHUNK_SIZE"
if [[ -n "$FPS" ]]; then
    BASE_MODEL_ARGS="$BASE_MODEL_ARGS,fps=$FPS"
fi
if [[ -n "$DEVICE" ]]; then
    BASE_MODEL_ARGS="$BASE_MODEL_ARGS,device=$DEVICE"
fi
if [[ -n "$DEVICE_MAP" ]]; then
    BASE_MODEL_ARGS="$BASE_MODEL_ARGS,device_map=$DEVICE_MAP"
fi

for retention_ratio in "${RETENTION_RATIOS[@]}"; do
    echo "Running LLaVA-OneVision-1.5-8B FETP-v3 video benchmarks with retention_ratio=${retention_ratio}"
    MODEL_ARGS="$BASE_MODEL_ARGS,retention_ratio=${retention_ratio}"
    for task in "${TASKS[@]}"; do
        echo "Evaluating task: $task"
        accelerate launch \
            --main_process_port "$MAIN_PROCESS_PORT" \
            --num_processes "$NUM_PROCESSES" \
            -m lmms_eval \
            --model llava_onevision1_5_ours_v3 \
            --model_args "$MODEL_ARGS" \
            --tasks "$task" \
            --batch_size "$BATCH_SIZE" \
            "${REQUEST_CACHE_ARGS[@]}" \
            --log_samples \
            --log_samples_suffix "$LOG_SAMPLES_SUFFIX" \
            --output_path "$OUTPUT_PATH"
    done
    echo "Finished running with retention_ratio=${retention_ratio}"
done
