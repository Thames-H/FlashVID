#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LMMS_EVAL_ROOT="${PROJECT_ROOT}/lmms-eval"
SINK_SCRIPT_DIR="${PROJECT_ROOT}/scripts/sink_analysis"
export PYTHONPATH="${PROJECT_ROOT}:${LMMS_EVAL_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_PROCESSES="${NUM_PROCESSES:-1}"
export PRETRAINED="${PRETRAINED:-/root/autodl-tmp/llava-onevision-qwen2-7b-ov-hf}"
export MODEL_DTYPE="${MODEL_DTYPE:-float16}"
export TASKS_CSV="${TASKS_CSV:-gqa}"
export LIMIT="${LIMIT:-8}"
export RATIOS_CSV="${RATIOS_CSV:-5%,10%,20%,25%,50%,75%}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
export MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-8}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-18892}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export SCORING_METHOD="${SCORING_METHOD:-full}"
export SHALLOW_LAYERS="${SHALLOW_LAYERS:-4}"
export TARGET_LAYER="${TARGET_LAYER:-15}"
export TEXT_CHUNK_SIZE="${TEXT_CHUNK_SIZE:-32}"

RESUME="${RESUME:-true}"
RESUME_ARGS=()
if [[ "${RESUME}" == "true" ]]; then
    RESUME_ARGS+=(--resume)
fi

IFS=',' read -r -a RATIOS <<< "${RATIOS_CSV}"

run_collect() {
    local method_name="$1"
    local keep_ratio="$2"
    shift 2

    bash "${SINK_SCRIPT_DIR}/collect_llava.sh" \
        --method "${method_name}" \
        --keep-ratio "${keep_ratio}" \
        --tasks "${TASKS_CSV}" \
        --limit "${LIMIT}" \
        "${RESUME_ARGS[@]}" \
        "$@"
}

echo "repo root: ${PROJECT_ROOT}"
echo "model path: ${PRETRAINED}"
echo "model dtype: ${MODEL_DTYPE}"
echo "tasks: ${TASKS_CSV}"
echo "limit: ${LIMIT}"
echo "ratios: ${RATIOS_CSV}"
echo "target layer: ${TARGET_LAYER}"
echo "resume: ${RESUME}"

echo "stage: full"
bash "${SINK_SCRIPT_DIR}/collect_llava.sh" \
    --method full \
    --tasks "${TASKS_CSV}" \
    --limit "${LIMIT}" \
    "${RESUME_ARGS[@]}"

for ratio in "${RATIOS[@]}"; do
    echo "stage: fetp ${ratio}"
    run_collect fetp "${ratio}"

    echo "stage: attention ${ratio}"
    run_collect attention "${ratio}"

    echo "stage: mmtok ${ratio}"
    run_collect mmtok "${ratio}"
done

echo "stage: merge"
bash "${SINK_SCRIPT_DIR}/merge.sh"

echo "stage: build-ablation"
python -m sink_analysis.cli --repo-root "${PROJECT_ROOT}" build-ablation
python -m sink_analysis.cli --repo-root "${PROJECT_ROOT}" rerun-ablation --config B
python -m sink_analysis.cli --repo-root "${PROJECT_ROOT}" rerun-ablation --config D

for ratio in "${RATIOS[@]}"; do
    echo "stage: ablation_b ${ratio}"
    run_collect \
        ablation_b \
        "${ratio}" \
        --override-file "${PROJECT_ROOT}/sink_analysis/data/b_overrides.json"

    echo "stage: ablation_d ${ratio}"
    run_collect \
        ablation_d \
        "${ratio}" \
        --override-file "${PROJECT_ROOT}/sink_analysis/data/d_overrides.json"
done

echo "stage: analyze"
bash "${SINK_SCRIPT_DIR}/analyze.sh"

echo "done"
echo "report: ${PROJECT_ROOT}/sink_analysis/report.md"
