#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LMMS_EVAL_ROOT="${PROJECT_ROOT}/lmms-eval"
export PYTHONPATH="${PROJECT_ROOT}:${LMMS_EVAL_ROOT}:${PYTHONPATH:-}"
cd "${LMMS_EVAL_ROOT}"

METHOD=""
KEEP_RATIO="full"
TASKS_CSV="${TASKS_CSV:-gqa}"
LIMIT="${LIMIT:-8}"
RESUME=false
OVERRIDE_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)
            METHOD="$2"
            shift 2
            ;;
        --keep-ratio)
            KEEP_RATIO="$2"
            shift 2
            ;;
        --tasks)
            TASKS_CSV="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --override-file)
            OVERRIDE_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${METHOD}" ]]; then
    echo "--method is required" >&2
    exit 1
fi

KEEP_RATIO_SLUG="${KEEP_RATIO//%/pct}"
RETENTION_RATIO="$(python -c "label='${KEEP_RATIO}'.strip(); print(1.0 if label.lower() == 'full' else (float(label.rstrip('%'))/100.0 if label.endswith('%') else float(label)))" )"

PRETRAINED="${PRETRAINED:-Qwen/Qwen3-VL-8B-Instruct}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-8}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-18891}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SCORING_METHOD="${SCORING_METHOD:-full}"
SHALLOW_LAYERS="${SHALLOW_LAYERS:-4}"
TARGET_LAYER="${TARGET_LAYER:-20}"
TEXT_CHUNK_SIZE="${TEXT_CHUNK_SIZE:-32}"
PARTIAL_ROOT="${PROJECT_ROOT}/sink_analysis/artifacts_partial"
RUN_ROOT="${PROJECT_ROOT}/sink_analysis/lmms_eval_outputs/qwen3-vl/${METHOD}/${KEEP_RATIO_SLUG}"
LOG_SAMPLES_SUFFIX="sink_analysis_qwen3_${METHOD}_${KEEP_RATIO_SLUG}"

if [[ "${RESUME}" == "true" ]] && find "${RUN_ROOT}" -name '*_results.json' -print -quit | grep -q .; then
    echo "Skipping qwen3-vl ${METHOD} ${KEEP_RATIO}: existing results found under ${RUN_ROOT}"
    exit 0
fi

case "${METHOD}" in
    full)
        MODEL_NAME="qwen3_vl_chat"
        MODEL_ARGS="pretrained=${PRETRAINED},max_num_frames=${MAX_NUM_FRAMES},attn_implementation=${ATTN_IMPLEMENTATION},sink_analysis_output_root=${PARTIAL_ROOT},sink_analysis_method_name=full,sink_analysis_keep_ratio=full"
        ;;
    fetp)
        MODEL_NAME="qwen3_vl_ours_v3"
        MODEL_ARGS="pretrained=${PRETRAINED},max_num_frames=${MAX_NUM_FRAMES},attn_implementation=${ATTN_IMPLEMENTATION},retention_ratio=${RETENTION_RATIO},scoring_method=${SCORING_METHOD},shallow_layers=${SHALLOW_LAYERS},target_layer=${TARGET_LAYER},use_alpha=true,use_deviation=true,two_stage=false,text_chunk_size=${TEXT_CHUNK_SIZE},sink_analysis_output_root=${PARTIAL_ROOT},sink_analysis_method_name=fetp,sink_analysis_keep_ratio=${KEEP_RATIO}"
        ;;
    attention)
        MODEL_NAME="qwen3_vl_ours_v3"
        MODEL_ARGS="pretrained=${PRETRAINED},max_num_frames=${MAX_NUM_FRAMES},attn_implementation=${ATTN_IMPLEMENTATION},retention_ratio=${RETENTION_RATIO},scoring_method=${SCORING_METHOD},shallow_layers=${SHALLOW_LAYERS},target_layer=${TARGET_LAYER},use_alpha=true,use_deviation=false,two_stage=false,text_chunk_size=${TEXT_CHUNK_SIZE},sink_analysis_output_root=${PARTIAL_ROOT},sink_analysis_method_name=attention,sink_analysis_keep_ratio=${KEEP_RATIO}"
        ;;
    mmtok)
        MODEL_NAME="qwen3_vl_mmtok"
        MODEL_ARGS="pretrained=${PRETRAINED},max_num_frames=${MAX_NUM_FRAMES},attn_implementation=${ATTN_IMPLEMENTATION},retain_ratio=${RETENTION_RATIO},sink_analysis_output_root=${PARTIAL_ROOT},sink_analysis_method_name=mmtok,sink_analysis_keep_ratio=${KEEP_RATIO}"
        ;;
    ablation_b)
        MODEL_NAME="qwen3_vl_ours_v3"
        MODEL_ARGS="pretrained=${PRETRAINED},max_num_frames=${MAX_NUM_FRAMES},attn_implementation=${ATTN_IMPLEMENTATION},retention_ratio=${RETENTION_RATIO},scoring_method=${SCORING_METHOD},shallow_layers=${SHALLOW_LAYERS},target_layer=${TARGET_LAYER},use_alpha=true,use_deviation=false,two_stage=false,text_chunk_size=${TEXT_CHUNK_SIZE},sink_analysis_output_root=${PARTIAL_ROOT},sink_analysis_method_name=ablation_b,sink_analysis_keep_ratio=${KEEP_RATIO},sink_analysis_override_path=${OVERRIDE_FILE}"
        ;;
    ablation_d)
        MODEL_NAME="qwen3_vl_ours_v3"
        MODEL_ARGS="pretrained=${PRETRAINED},max_num_frames=${MAX_NUM_FRAMES},attn_implementation=${ATTN_IMPLEMENTATION},retention_ratio=${RETENTION_RATIO},scoring_method=${SCORING_METHOD},shallow_layers=${SHALLOW_LAYERS},target_layer=${TARGET_LAYER},use_alpha=true,use_deviation=true,two_stage=false,text_chunk_size=${TEXT_CHUNK_SIZE},sink_analysis_output_root=${PARTIAL_ROOT},sink_analysis_method_name=ablation_d,sink_analysis_keep_ratio=${KEEP_RATIO},sink_analysis_override_path=${OVERRIDE_FILE}"
        ;;
    *)
        echo "Unsupported method: ${METHOD}" >&2
        exit 1
        ;;
esac

CMD=(
    accelerate launch
    --main_process_port "${MAIN_PROCESS_PORT}"
    --num_processes "${NUM_PROCESSES}"
    -m lmms_eval
    --model "${MODEL_NAME}"
    --model_args "${MODEL_ARGS}"
    --tasks "${TASKS_CSV}"
    --batch_size "${BATCH_SIZE}"
    --output_path "${RUN_ROOT}"
    --log_samples
    --log_samples_suffix "${LOG_SAMPLES_SUFFIX}"
)

if [[ -n "${LIMIT}" ]]; then
    CMD+=(--limit "${LIMIT}")
fi

echo "collect qwen3 sink-analysis partials"
echo "Running: ${METHOD} keep_ratio=${KEEP_RATIO} tasks=${TASKS_CSV} limit=${LIMIT}"
"${CMD[@]}"
