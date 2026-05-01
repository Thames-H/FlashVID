#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LMMS_EVAL_ROOT="${LMMS_EVAL_ROOT:-${PROJECT_ROOT}/lmms-eval}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${HOME}/autodl-tmp/hf_cache}"
PRETRAINED="${PRETRAINED:-${HOME}/autodl-tmp/llavav-1.5-7b}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12346}"

if [[ ! -f "${PRETRAINED}/config.json" ]]; then
    echo "Missing local LLaVA-1.5 model at ${PRETRAINED}" >&2
    echo "Run: MODEL_DIR=${PRETRAINED} bash ${PROJECT_ROOT}/scripts/download_llava_1_5.sh" >&2
    exit 1
fi

cd "$LMMS_EVAL_ROOT"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,6,7}" accelerate launch \
    --num_processes="$NUM_PROCESSES" \
    --main_process_port="$MAIN_PROCESS_PORT" \
    -m lmms_eval \
    --model llava_mmtok \
    --model_args "pretrained=${PRETRAINED},attn_implementation=${ATTN_IMPLEMENTATION}" \
    --tasks ocrbench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "llava_motk_192" \
    --output_path "${PROJECT_ROOT}/logs/mmtok_ocr" \
    --verbosity=DEBUG
