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
MAIN_PROCESS_PORT=18895
BATCH_SIZE=1
LOG_SAMPLES_SUFFIX="internvl3_5_original_8b_img"
OUTPUT_PATH="./logs/original_internvl3_5_8b_img"
CACHE_REQUESTS="true"
TASKS=("gqa" "scienceqa_img" "mmbench_en" "mme" "pope" "ocrbench")

AUTODL_MODEL_PATH="$HOME/autodl-tmp/InternVL3_5-8B-HF"
LEGACY_AUTODL_MODEL_PATH="$HOME/autodl-tmp/InternVL3_5-8B"
DEFAULT_PRETRAINED="OpenGVLab/InternVL3_5-8B-HF"
PRETRAINED="$DEFAULT_PRETRAINED"

MAX_PATCHES=12
NUM_FRAMES=8
ATTN_IMPLEMENTATION="flash_attention_2"

if [[ -d "$AUTODL_MODEL_PATH" ]]; then
    PRETRAINED="$AUTODL_MODEL_PATH"
fi

export CUDA_VISIBLE_DEVICES
export LMMS_EVAL_USE_CACHE
export LMMS_EVAL_HOME

if [[ "$PRETRAINED" == "$LEGACY_AUTODL_MODEL_PATH" ]]; then
    echo "Error: '$PRETRAINED' is the original InternVL chat-format checkpoint."
    echo "internvl3_5_original expects the HF-format checkpoint:"
    echo "  - OpenGVLab/InternVL3_5-8B-HF"
    echo "  - or a local directory such as $HOME/autodl-tmp/InternVL3_5-8B-HF"
    exit 1
fi

BASE_MODEL_ARGS="pretrained=$PRETRAINED,max_patches=$MAX_PATCHES,num_frames=$NUM_FRAMES,attn_implementation=$ATTN_IMPLEMENTATION"

REQUEST_CACHE_ARGS=()
if [[ -n "$CACHE_REQUESTS" ]]; then
    REQUEST_CACHE_ARGS=(--cache_requests "$CACHE_REQUESTS")
fi

for task in "${TASKS[@]}"; do
    echo "Evaluating task: $task"
    accelerate launch \
    --main_process_port "$MAIN_PROCESS_PORT" \
    --num_processes "$NUM_PROCESSES" \
    -m lmms_eval \
    --model internvl3_5_original \
    --model_args "$BASE_MODEL_ARGS" \
    --tasks "$task" \
    --batch_size "$BATCH_SIZE" \
    "${REQUEST_CACHE_ARGS[@]}" \
    --log_samples \
    --log_samples_suffix "$LOG_SAMPLES_SUFFIX" \
    --output_path "$OUTPUT_PATH"
done
