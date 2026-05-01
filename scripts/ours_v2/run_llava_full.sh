#!/usr/bin/env bash
set -euo pipefail

# FETP for LLaVA-1.5 (HF format)
# Mode: full (approach 3)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LMMS_EVAL_ROOT="${LMMS_EVAL_ROOT:-${PROJECT_ROOT}/lmms-eval}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${HOME}/autodl-tmp/hf_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,6,7}"

cd "$LMMS_EVAL_ROOT"

# Evaluation benchmarks.
TASKS=("mme")

# This custom adapter expects the Transformers-converted LLaVA format.
PRETRAINED="${PRETRAINED:-llava-hf/llava-1.5-7b-hf}"

# FETP arguments.
RETENTION_RATIOS=(64 128 192)
SCORING_METHOD=full
TARGET_LAYER=15   # LLaMA-7B has 32 layers, middle = 15

# Model arguments.
ATTN_IMPLEMENTATION=flash_attention_2

BASE_MODEL_ARGS="pretrained=$PRETRAINED,attn_implementation=$ATTN_IMPLEMENTATION"
BASE_MODEL_ARGS="$BASE_MODEL_ARGS,scoring_method=$SCORING_METHOD,target_layer=$TARGET_LAYER"

for retention_ratio in "${RETENTION_RATIOS[@]}"; do
    echo "Running FETP-full on LLaVA-1.5 with retention_ratio=${retention_ratio}"
    MODEL_ARGS="$BASE_MODEL_ARGS,retention_ratio=${retention_ratio}"
    for task in "${TASKS[@]}"; do
        echo "Evaluating task: $task"
        accelerate launch \
        --main_process_port 18899 \
        --num_processes 4 \
        -m lmms_eval \
        --model llava_hf_ours_v2 \
        --model_args "$MODEL_ARGS" \
        --tasks "$task" \
        --batch_size 1 \
        --output_path "${PROJECT_ROOT}/logs/llava_hf_ours_v2" \
        --verbosity=DEBUG
    done
    echo "Finished running with retention_ratio=${retention_ratio}"
done
