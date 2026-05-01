#!/usr/bin/env bash
set -euo pipefail

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate llava15-eval
fi

cd /root/autodl-tmp/FlashVID_llava15_eval/lmms-eval

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/root/autodl-tmp/hf_cache/datasets}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
unset HF_HUB_ENABLE_HF_TRANSFER

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PRETRAINED="${PRETRAINED:-/root/autodl-tmp/llava-v1.5-7b}"
OUTPUT_PATH="${OUTPUT_PATH:-/root/autodl-tmp/FlashVID_llava15_eval/logs/llava15_mme}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12348}"
BATCH_SIZE="${BATCH_SIZE:-1}"

test -f "${PRETRAINED}/config.json" || { echo "Missing model: ${PRETRAINED}" >&2; exit 1; }
mkdir -p "${OUTPUT_PATH}"

accelerate launch \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    -m lmms_eval \
    --model llava \
    --model_args "pretrained=${PRETRAINED},device_map=auto" \
    --tasks mme \
    --batch_size "${BATCH_SIZE}" \
    --log_samples \
    --log_samples_suffix llava15_mme \
    --output_path "${OUTPUT_PATH}" \
    --verbosity=DEBUG
