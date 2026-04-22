#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LMMS_EVAL_ROOT="${PROJECT_ROOT}/lmms-eval"
cd "${LMMS_EVAL_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-18893}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LIMIT_PER_TASK="${LIMIT_PER_TASK:-4}"
OUTPUT_PATH="${OUTPUT_PATH:-${LMMS_EVAL_ROOT}/logs/qwen3_vl_pca_compare_img}"

if [[ -n "${TASKS_CSV:-}" ]]; then
    IFS=',' read -r -a TASKS <<< "${TASKS_CSV}"
else
    TASKS=("gqa" "mme" "pope")
fi

AUTODL_MODEL_PATH="${HOME}/autodl-tmp/Qwen3-VL-8B-Instruct"
DEFAULT_PRETRAINED="Qwen/Qwen3-VL-8B-Instruct"
if [[ -d "${AUTODL_MODEL_PATH}" ]]; then
    DEFAULT_PRETRAINED="${AUTODL_MODEL_PATH}"
fi
PRETRAINED="${PRETRAINED:-${DEFAULT_PRETRAINED}}"

RETENTION_RATIO="${RETENTION_RATIO:-0.20}"
TARGET_LAYER="${TARGET_LAYER:-20}"
TEXT_CHUNK_SIZE="${TEXT_CHUNK_SIZE:-32}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
LAYER_OUTPUT_PATH="${OUTPUT_PATH}/layer_${TARGET_LAYER}"

COMMON_ARGS="pretrained=${PRETRAINED},max_num_frames=8,attn_implementation=${ATTN_IMPLEMENTATION}"
FETP_ARGS="${COMMON_ARGS},retention_ratio=${RETENTION_RATIO},scoring_method=full,target_layer=${TARGET_LAYER},use_alpha=True,use_deviation=True,two_stage=False,text_chunk_size=${TEXT_CHUNK_SIZE},stats_output_path=${LAYER_OUTPUT_PATH}"
MMTOK_ARGS="${COMMON_ARGS},retain_ratio=${RETENTION_RATIO},stats_output_path=${LAYER_OUTPUT_PATH}"

mkdir -p "${LAYER_OUTPUT_PATH}"

for task in "${TASKS[@]}"; do
    echo "[1/3] FETP artifact export for task=${task}"
    accelerate launch \
        --main_process_port "${MAIN_PROCESS_PORT}" \
        --num_processes "${NUM_PROCESSES}" \
        -m lmms_eval \
        --model qwen3_vl_ours_v3 \
        --model_args "${FETP_ARGS}" \
        --tasks "${task}" \
        --batch_size "${BATCH_SIZE}" \
        --limit "${LIMIT_PER_TASK}" \
        --log_samples \
        --predict_only \
        --log_samples_suffix "qwen3_vl_fetp_pca_compare" \
        --output_path "${LAYER_OUTPUT_PATH}"

    echo "[2/3] MMTok artifact export for task=${task}"
    accelerate launch \
        --main_process_port "$((MAIN_PROCESS_PORT + 1))" \
        --num_processes "${NUM_PROCESSES}" \
        -m lmms_eval \
        --model qwen3_vl_mmtok \
        --model_args "${MMTOK_ARGS}" \
        --tasks "${task}" \
        --batch_size "${BATCH_SIZE}" \
        --limit "${LIMIT_PER_TASK}" \
        --log_samples \
        --predict_only \
        --log_samples_suffix "qwen3_vl_mmtok_pca_compare" \
        --output_path "${LAYER_OUTPUT_PATH}"
done

echo "[3/3] Building PCA plots and markdown report"
python "${PROJECT_ROOT}/tools/qwen3_vl_token_pruning_pca_compare.py" \
    --artifact-root "${LAYER_OUTPUT_PATH}" \
    --output-dir "${LAYER_OUTPUT_PATH}/pca_compare"
