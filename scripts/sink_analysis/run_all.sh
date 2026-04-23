#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

RESUME=false
TASKS_CSV="${TASKS_CSV:-gqa}"
LIMIT="${LIMIT:-8}"
RATIOS_CSV="${RATIOS_CSV:-25%,50%,75%}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            RESUME=true
            shift
            ;;
        --tasks)
            TASKS_CSV="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --ratios)
            RATIOS_CSV="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

IFS=',' read -r -a RATIOS <<< "${RATIOS_CSV}"
RESUME_ARGS=()
if [[ "${RESUME}" == "true" ]]; then
    RESUME_ARGS+=(--resume)
fi

mark_stage() {
    python - "$PROJECT_ROOT" "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
stage = sys.argv[2]
status = sys.argv[3]
state_path = repo_root / "sink_analysis" / "pipeline_state.json"
state = {}
if state_path.exists():
    state = json.loads(state_path.read_text(encoding="utf-8"))
state[stage] = status
state_path.parent.mkdir(parents=True, exist_ok=True)
state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

run_collect_family() {
    local script_name="$1"

    "$SCRIPT_DIR/${script_name}" --method full --tasks "$TASKS_CSV" --limit "$LIMIT" "${RESUME_ARGS[@]}"

    for ratio in "${RATIOS[@]}"; do
        "$SCRIPT_DIR/${script_name}" --method fetp --keep-ratio "$ratio" --tasks "$TASKS_CSV" --limit "$LIMIT" "${RESUME_ARGS[@]}"
        "$SCRIPT_DIR/${script_name}" --method attention --keep-ratio "$ratio" --tasks "$TASKS_CSV" --limit "$LIMIT" "${RESUME_ARGS[@]}"
        "$SCRIPT_DIR/${script_name}" --method mmtok --keep-ratio "$ratio" --tasks "$TASKS_CSV" --limit "$LIMIT" "${RESUME_ARGS[@]}"
    done
}

run_ablation_family() {
    local script_name="$1"
    local method_name="$2"
    local override_path="$3"

    for ratio in "${RATIOS[@]}"; do
        "$SCRIPT_DIR/${script_name}" \
            --method "$method_name" \
            --keep-ratio "$ratio" \
            --override-file "$override_path" \
            --tasks "$TASKS_CSV" \
            --limit "$LIMIT" \
            "${RESUME_ARGS[@]}"
    done
}

echo "resume flag: ${RESUME}"
echo "stage: fetp"
mark_stage "collection" "running"
run_collect_family "collect_qwen3.sh"
run_collect_family "collect_llava.sh"
mark_stage "collection" "complete"

echo "stage: merge"
mark_stage "merge" "running"
"$SCRIPT_DIR/merge.sh"
mark_stage "merge" "complete"

echo "stage: build-ablation"
mark_stage "build_ablation" "running"
python -m sink_analysis.cli --repo-root "$PROJECT_ROOT" build-ablation
python -m sink_analysis.cli --repo-root "$PROJECT_ROOT" rerun-ablation --config B
python -m sink_analysis.cli --repo-root "$PROJECT_ROOT" rerun-ablation --config D
mark_stage "build_ablation" "complete"

echo "stage: attention"
mark_stage "ablation_b" "running"
run_ablation_family "collect_qwen3.sh" "ablation_b" "$PROJECT_ROOT/sink_analysis/data/b_overrides.json"
run_ablation_family "collect_llava.sh" "ablation_b" "$PROJECT_ROOT/sink_analysis/data/b_overrides.json"
mark_stage "ablation_b" "complete"

echo "stage: mmtok"
echo "stage: analyze"
mark_stage "ablation_d" "running"
run_ablation_family "collect_qwen3.sh" "ablation_d" "$PROJECT_ROOT/sink_analysis/data/d_overrides.json"
run_ablation_family "collect_llava.sh" "ablation_d" "$PROJECT_ROOT/sink_analysis/data/d_overrides.json"
mark_stage "ablation_d" "complete"

mark_stage "analysis" "running"
"$SCRIPT_DIR/analyze.sh"
mark_stage "analysis" "complete"
