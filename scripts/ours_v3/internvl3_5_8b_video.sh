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
MAIN_PROCESS_PORT=18893
BATCH_SIZE=1
LOG_SAMPLES_SUFFIX="internvl3_5_ours_v3_8b_video"
OUTPUT_PATH="./logs/ours_v3_internvl3_5_8b_video"
TASKS=("videomme" "longvideobench_val_v")

AUTODL_MODEL_PATH="$HOME/autodl-tmp/InternVL3_5-8B-HF"
LEGACY_AUTODL_MODEL_PATH="$HOME/autodl-tmp/InternVL3_5-8B"
DEFAULT_PRETRAINED="OpenGVLab/InternVL3_5-8B-HF"
PRETRAINED="$DEFAULT_PRETRAINED"

RETENTION_RATIOS=(0.05 0.10 0.20)
SCORING_METHOD="full"
SHALLOW_LAYERS=4
TARGET_LAYER=-15
USE_ALPHA="true"
USE_DEVIATION="true"
TWO_STAGE="false"
CANDIDATE_RATIO=1.0
MAX_SCORE_TEXT_TOKENS=0
MAX_SCORE_HEADS=0
TEXT_CHUNK_SIZE=32

MAX_PATCHES=12
NUM_FRAMES=32
ATTN_IMPLEMENTATION="flash_attention_2"

if [[ -d "$AUTODL_MODEL_PATH" ]]; then
    PRETRAINED="$AUTODL_MODEL_PATH"
fi

export CUDA_VISIBLE_DEVICES
export LMMS_EVAL_USE_CACHE
export LMMS_EVAL_HOME

if [[ "$PRETRAINED" == "$LEGACY_AUTODL_MODEL_PATH" ]]; then
    echo "Error: '$PRETRAINED' is the original InternVL chat-format checkpoint."
    echo "internvl3_5_ours_v3 expects the HF-format checkpoint:"
    echo "  - OpenGVLab/InternVL3_5-8B-HF"
    echo "  - or a local directory such as $HOME/autodl-tmp/InternVL3_5-8B-HF"
    exit 1
fi

BASE_MODEL_ARGS="pretrained=$PRETRAINED,max_patches=$MAX_PATCHES,num_frames=$NUM_FRAMES,attn_implementation=$ATTN_IMPLEMENTATION,scoring_method=$SCORING_METHOD,shallow_layers=$SHALLOW_LAYERS,target_layer=$TARGET_LAYER,use_alpha=$USE_ALPHA,use_deviation=$USE_DEVIATION,two_stage=$TWO_STAGE,candidate_ratio=$CANDIDATE_RATIO,text_chunk_size=$TEXT_CHUNK_SIZE"

if [[ "$MAX_SCORE_TEXT_TOKENS" != "0" ]]; then
    BASE_MODEL_ARGS="$BASE_MODEL_ARGS,max_score_text_tokens=$MAX_SCORE_TEXT_TOKENS"
fi
if [[ "$MAX_SCORE_HEADS" != "0" ]]; then
    BASE_MODEL_ARGS="$BASE_MODEL_ARGS,max_score_heads=$MAX_SCORE_HEADS"
fi

for retention_ratio in "${RETENTION_RATIOS[@]}"; do
    echo "Running InternVL3.5-8B FETP video benchmarks with retention_ratio=${retention_ratio}"
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
done
