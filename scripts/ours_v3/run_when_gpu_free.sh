#!/bin/bash
set -euo pipefail

QUEUE_FILE="${1:-}"

if [[ -z "$QUEUE_FILE" || ! -f "$QUEUE_FILE" ]]; then
    cat <<'USAGE'
Usage:
  scripts/ours_v3/run_when_gpu_free.sh /path/to/job_queue.txt

Environment variables:
  GPU_IDS            GPU ids to watch. Default: CUDA_VISIBLE_DEVICES, or 0.
                     Use GPU_IDS=all to watch all GPUs.
  POLL_INTERVAL      Seconds between checks. Default: 60.
  STABLE_CHECKS      Require this many consecutive idle checks. Default: 2.
  CONTINUE_ON_ERROR  Continue after a failed command. Default: false.
  DRY_RUN            Print commands without running them. Default: false.

Queue file format:
  # comments and blank lines are ignored
  bash scripts/ours_v3/llava_onevision_7b_videomme_question_only.sh
  bash scripts/ours_v3/llava_onevision_7b_longvideobench_question_only.sh
USAGE
    exit 2
fi

GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-false}"
DRY_RUN="${DRY_RUN:-false}"
NVIDIA_SMI="${NVIDIA_SMI:-nvidia-smi}"

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
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

run_command() {
    local command="$1"
    echo "[$(timestamp)] Starting: $command"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[$(timestamp)] DRY_RUN=true, skipped."
        return 0
    fi
    bash -lc "$command"
}

job_index=0
while IFS= read -r command || [[ -n "$command" ]]; do
    command="${command#"${command%%[![:space:]]*}"}"
    command="${command%"${command##*[![:space:]]}"}"

    if [[ -z "$command" || "$command" == \#* ]]; then
        continue
    fi

    job_index=$((job_index + 1))
    wait_until_gpu_free

    if run_command "$command"; then
        echo "[$(timestamp)] Finished job ${job_index}."
    else
        status=$?
        echo "[$(timestamp)] Job ${job_index} failed with exit code ${status}: $command" >&2
        if [[ "$CONTINUE_ON_ERROR" != "true" ]]; then
            exit "$status"
        fi
    fi
done < "$QUEUE_FILE"

echo "[$(timestamp)] All queued jobs are done."
