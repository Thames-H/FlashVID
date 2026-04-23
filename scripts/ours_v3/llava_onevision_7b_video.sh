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
MAIN_PROCESS_PORT=18895
BATCH_SIZE=1
LOG_SAMPLES_SUFFIX="llava_onevision_ours_v3_7b_video"
OUTPUT_PATH="./logs/ours_v3_llava_onevision_7b_video"
TASKS=("videomme" "longvideobench_val_v")

AUTODL_MODEL_PATH="$HOME/autodl-tmp/llava-onevision-qwen2-7b-ov-hf"
DEFAULT_PRETRAINED="llava-hf/llava-onevision-qwen2-7b-ov-hf"
PRETRAINED="$DEFAULT_PRETRAINED"

RETENTION_RATIOS=(0.05 0.10 0.20)
SCORING_METHOD="full"
SHALLOW_LAYERS=4
TARGET_LAYER=15
USE_ALPHA="true"
USE_DEVIATION="true"
TWO_STAGE="true"
TEXT_CHUNK_SIZE=32

MAX_FRAMES_NUM=16
ATTN_IMPLEMENTATION="flash_attention_2"
DTYPE="float16"

if [[ -d "$AUTODL_MODEL_PATH" ]]; then
    PRETRAINED="$AUTODL_MODEL_PATH"
fi

export CUDA_VISIBLE_DEVICES
export LMMS_EVAL_USE_CACHE
export LMMS_EVAL_HOME

BASE_MODEL_ARGS="pretrained=$PRETRAINED,max_frames_num=$MAX_FRAMES_NUM,attn_implementation=$ATTN_IMPLEMENTATION,dtype=$DTYPE,scoring_method=$SCORING_METHOD,shallow_layers=$SHALLOW_LAYERS,target_layer=$TARGET_LAYER,use_alpha=$USE_ALPHA,use_deviation=$USE_DEVIATION,two_stage=$TWO_STAGE,text_chunk_size=$TEXT_CHUNK_SIZE"

for retention_ratio in "${RETENTION_RATIOS[@]}"; do
    echo "Running LLaVA-OneVision-7B FETP-v3 video benchmarks with retention_ratio=${retention_ratio}"
    MODEL_ARGS="$BASE_MODEL_ARGS,retention_ratio=${retention_ratio}"
    for task in "${TASKS[@]}"; do
        echo "Evaluating task: $task"
        accelerate launch \
        --main_process_port "$MAIN_PROCESS_PORT" \
        --num_processes "$NUM_PROCESSES" \
        -m lmms_eval \
        --model llava_onevision_ours_v3 \
        --model_args "$MODEL_ARGS" \
        --tasks "$task" \
        --batch_size "$BATCH_SIZE" \
        --log_samples \
        --log_samples_suffix "$LOG_SAMPLES_SUFFIX" \
        --output_path "$OUTPUT_PATH"
    done
    echo "Finished running with retention_ratio=${retention_ratio}"
done
