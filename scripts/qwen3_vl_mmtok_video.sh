#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LMMS_EVAL_ROOT="${PROJECT_ROOT}/lmms-eval"
cd "${LMMS_EVAL_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-18897}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOG_SAMPLES_SUFFIX="${LOG_SAMPLES_SUFFIX:-qwen3_vl_mmtok_video}"
OUTPUT_PATH="${OUTPUT_PATH:-${LMMS_EVAL_ROOT}/logs/qwen3_vl_mmtok_video}"

if [[ -n "${TASKS_CSV:-}" ]]; then
    IFS=',' read -r -a TASKS <<< "${TASKS_CSV}"
else
    TASKS=("videomme" "longvideobench_val_v")
fi

AUTODL_MODEL_PATH="${HOME}/autodl-tmp/Qwen3-VL-8B-Instruct"
DEFAULT_PRETRAINED="Qwen/Qwen3-VL-8B-Instruct"
if [[ -d "$AUTODL_MODEL_PATH" ]]; then
    DEFAULT_PRETRAINED="$AUTODL_MODEL_PATH"
fi
PRETRAINED="${PRETRAINED:-$DEFAULT_PRETRAINED}"

if [[ -n "${RETENTION_RATIOS_CSV:-}" ]]; then
    IFS=',' read -r -a RETENTION_RATIOS <<< "${RETENTION_RATIOS_CSV}"
else
    RETENTION_RATIOS=(0.20)
fi

MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-32}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
BASE_MODEL_ARGS="pretrained=$PRETRAINED,max_num_frames=$MAX_NUM_FRAMES,attn_implementation=$ATTN_IMPLEMENTATION"

for retention_ratio in "${RETENTION_RATIOS[@]}"; do
    echo "Running Qwen3-VL MMTok video benchmarks with retention_ratio=${retention_ratio}"
    MODEL_ARGS="$BASE_MODEL_ARGS,retain_ratio=${retention_ratio}"
    for task in "${TASKS[@]}"; do
        echo "Evaluating task: $task"
        accelerate launch \
            --main_process_port "$MAIN_PROCESS_PORT" \
            --num_processes "$NUM_PROCESSES" \
            -m lmms_eval \
            --model qwen3_vl_mmtok \
            --model_args "$MODEL_ARGS" \
            --tasks "$task" \
            --batch_size "$BATCH_SIZE" \
            --log_samples \
            --log_samples_suffix "$LOG_SAMPLES_SUFFIX" \
            --output_path "$OUTPUT_PATH"
    done
    echo "Finished running with retention_ratio=${retention_ratio}"
done
