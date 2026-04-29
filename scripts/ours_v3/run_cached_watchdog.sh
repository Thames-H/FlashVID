#!/bin/bash
set -euo pipefail

QUEUE_FILE="${1:-}"

if [[ -z "$QUEUE_FILE" || ! -f "$QUEUE_FILE" ]]; then
    cat <<'USAGE'
Usage:
  bash scripts/ours_v3/run_cached_watchdog.sh /path/to/job_queue.tsv

Queue file format, tab separated:
  job_name<TAB>done_marker<TAB>command

Example:
  qwen_videomme_qonly<TAB>logs/watchdog_done/qwen_videomme_qonly.done<TAB>TARGET_LAYER=18 bash scripts/ours_v3/qwen3_vl_8b_videomme_question_only.sh

Environment variables:
  GPU_IDS            GPU ids to watch. Default: CUDA_VISIBLE_DEVICES, or 0.
                     Use GPU_IDS=all to watch all GPUs.
  POLL_INTERVAL      Seconds between checks. Default: 60.
  STABLE_CHECKS      Require this many consecutive idle checks. Default: 2.
  RETRY_DELAY        Seconds to sleep after a failed run before rechecking. Default: 30.
  MAX_RETRIES        Retry limit per job. 0 means unlimited. Default: 0.
  WATCHDOG_STATE_DIR Retry counter directory. Default: logs/watchdog_state.
  DRY_RUN            Print commands without running them. Default: false.

The watchdog does not change LMMS_EVAL_HOME or output paths. It calls the
original commands, so existing lmms-eval request cache remains reusable.
USAGE
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"
RETRY_DELAY="${RETRY_DELAY:-30}"
MAX_RETRIES="${MAX_RETRIES:-0}"
WATCHDOG_STATE_DIR="${WATCHDOG_STATE_DIR:-logs/watchdog_state}"
DRY_RUN="${DRY_RUN:-false}"
NVIDIA_SMI="${NVIDIA_SMI:-nvidia-smi}"

mkdir -p "$WATCHDOG_STATE_DIR"

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

describe_exit_status() {
    local status="$1"
    case "$status" in
        137) printf "exit code 137 (SIGKILL, likely OOM)" ;;
        143) printf "exit code 143 (SIGTERM)" ;;
        130) printf "exit code 130 (SIGINT)" ;;
        *)
            if (( status > 128 )); then
                printf "exit code %s (signal %s)" "$status" "$((status - 128))"
            else
                printf "exit code %s" "$status"
            fi
            ;;
    esac
}

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf "%s" "$value"
}

safe_name() {
    printf "%s" "$1" | tr -c 'A-Za-z0-9_.-' '_'
}

query_gpu_processes() {
    local output
    if [[ "$GPU_IDS" == "all" || -z "$GPU_IDS" ]]; then
        output="$(
            "$NVIDIA_SMI" \
                --query-compute-apps=pid,process_name,used_memory \
                --format=csv,noheader,nounits 2>/dev/null || true
        )"
    else
        output="$(
            "$NVIDIA_SMI" -i "$GPU_IDS" \
                --query-compute-apps=pid,process_name,used_memory \
                --format=csv,noheader,nounits 2>/dev/null || true
        )"
    fi

    printf "%s\n" "$output" \
        | sed '/^[[:space:]]*$/d' \
        | grep -v -E '^No running processes found' \
        || true
}

wait_until_gpu_free() {
    local stable_idle=0
    local processes

    echo "[$(timestamp)] Watching GPU_IDS=${GPU_IDS}; waiting for no visible compute process."
    while true; do
        processes="$(query_gpu_processes)"
        if [[ -z "$processes" ]]; then
            stable_idle=$((stable_idle + 1))
            echo "[$(timestamp)] GPU idle check ${stable_idle}/${STABLE_CHECKS}."
            if (( stable_idle >= STABLE_CHECKS )); then
                return 0
            fi
        else
            stable_idle=0
            echo "[$(timestamp)] GPU still busy:"
            printf "%s\n" "$processes" | sed 's/^/  /'
        fi
        sleep "$POLL_INTERVAL"
    done
}

retry_count_path() {
    local job_name="$1"
    printf "%s/%s.retry" "$WATCHDOG_STATE_DIR" "$(safe_name "$job_name")"
}

read_retry_count() {
    local retry_file="$1"
    if [[ -f "$retry_file" ]]; then
        cat "$retry_file"
    else
        printf "0"
    fi
}

write_done_marker() {
    local done_marker="$1"
    local job_name="$2"
    local command="$3"

    mkdir -p "$(dirname "$done_marker")"
    {
        printf "job_name=%s\n" "$job_name"
        printf "completed_at=%s\n" "$(timestamp)"
        printf "command=%s\n" "$command"
    } > "$done_marker"
}

run_job_until_done() {
    local job_name="$1"
    local done_marker="$2"
    local command="$3"
    local retry_file
    local retries
    local status

    if [[ -f "$done_marker" ]]; then
        echo "[$(timestamp)] Skip ${job_name}: done marker exists at ${done_marker}."
        return 0
    fi

    retry_file="$(retry_count_path "$job_name")"

    while [[ ! -f "$done_marker" ]]; do
        retries="$(read_retry_count "$retry_file")"
        if (( MAX_RETRIES > 0 && retries >= MAX_RETRIES )); then
            echo "[$(timestamp)] ${job_name}: reached MAX_RETRIES=${MAX_RETRIES}; stop." >&2
            return 1
        fi

        wait_until_gpu_free
        echo "[$(timestamp)] Starting ${job_name}: ${command}"
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "[$(timestamp)] DRY_RUN=true, marking ${job_name} as done."
            write_done_marker "$done_marker" "$job_name" "$command"
            return 0
        fi

        set +e
        bash -lc "$command"
        status=$?
        set -e

        if (( status == 0 )); then
            write_done_marker "$done_marker" "$job_name" "$command"
            rm -f "$retry_file"
            echo "[$(timestamp)] Finished ${job_name}; wrote ${done_marker}."
            return 0
        fi

        retries=$((retries + 1))
        printf "%s" "$retries" > "$retry_file"
        echo "[$(timestamp)] ${job_name} failed with $(describe_exit_status "$status"); retry ${retries} recorded." >&2
        echo "[$(timestamp)] Existing lmms-eval cache is preserved; next run will reuse cached samples." >&2
        sleep "$RETRY_DELAY"
    done
}

job_index=0
while IFS= read -r line || [[ -n "$line" ]]; do
    line="$(trim "$line")"
    if [[ -z "$line" || "$line" == \#* ]]; then
        continue
    fi

    IFS=$'\t' read -r job_name done_marker command <<< "$line"
    job_name="$(trim "${job_name:-}")"
    done_marker="$(trim "${done_marker:-}")"
    command="$(trim "${command:-}")"

    if [[ -z "$job_name" || -z "$done_marker" || -z "$command" ]]; then
        echo "[$(timestamp)] Invalid queue line; expected job_name<TAB>done_marker<TAB>command:" >&2
        echo "$line" >&2
        exit 2
    fi

    job_index=$((job_index + 1))
    echo "[$(timestamp)] Queue job ${job_index}: ${job_name}"
    run_job_until_done "$job_name" "$done_marker" "$command"
done < "$QUEUE_FILE"

echo "[$(timestamp)] All watchdog jobs are done."
