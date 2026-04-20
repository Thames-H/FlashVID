#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-18891}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOG_SAMPLES_SUFFIX="${LOG_SAMPLES_SUFFIX:-internvl3_5_ours_v3_8b_img}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/ours_v3_internvl3_5_8b_img}"

if [[ -n "${TASKS_CSV:-}" ]]; then
    IFS=',' read -r -a TASKS <<< "${TASKS_CSV}"
else
    TASKS=("gqa" "scienceqa_img" "mmbench_en" "mme" "pope" "ocrbench")
fi

AUTODL_MODEL_PATH="${HOME}/autodl-tmp/InternVL3_5-8B"
DEFAULT_PRETRAINED="OpenGVLab/InternVL3_5-8B-HF"
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
TARGET_LAYER="${TARGET_LAYER:--15}"
USE_ALPHA="${USE_ALPHA:-true}"
USE_DEVIATION="${USE_DEVIATION:-true}"
CANDIDATE_RATIO="${CANDIDATE_RATIO:-1.0}"
MAX_SCORE_TEXT_TOKENS="${MAX_SCORE_TEXT_TOKENS:-0}"
MAX_SCORE_HEADS="${MAX_SCORE_HEADS:-0}"

MAX_PATCHES="${MAX_PATCHES:-12}"
NUM_FRAMES="${NUM_FRAMES:-8}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
BASE_MODEL_ARGS="pretrained=$PRETRAINED,max_patches=$MAX_PATCHES,num_frames=$NUM_FRAMES,attn_implementation=$ATTN_IMPLEMENTATION,scoring_method=$SCORING_METHOD,shallow_layers=$SHALLOW_LAYERS,target_layer=$TARGET_LAYER,use_alpha=$USE_ALPHA,use_deviation=$USE_DEVIATION,candidate_ratio=$CANDIDATE_RATIO"

if [[ "$MAX_SCORE_TEXT_TOKENS" != "0" ]]; then
    BASE_MODEL_ARGS="$BASE_MODEL_ARGS,max_score_text_tokens=$MAX_SCORE_TEXT_TOKENS"
fi
if [[ "$MAX_SCORE_HEADS" != "0" ]]; then
    BASE_MODEL_ARGS="$BASE_MODEL_ARGS,max_score_heads=$MAX_SCORE_HEADS"
fi

for retention_ratio in "${RETENTION_RATIOS[@]}"; do
    echo "Running InternVL3.5-8B FETP image benchmarks with retention_ratio=${retention_ratio}"
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
        --log_samples \
        --log_samples_suffix "$LOG_SAMPLES_SUFFIX" \
        --output_path "$OUTPUT_PATH"
    done
    echo "Finished running with retention_ratio=${retention_ratio}"
done
