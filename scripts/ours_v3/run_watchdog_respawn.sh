#!/bin/bash
set -euo pipefail

if [[ "${1:-}" != "--" || "$#" -lt 2 ]]; then
    cat <<'USAGE'
Usage:
  bash scripts/ours_v3/run_watchdog_respawn.sh -- command [args...]

Environment variables:
  WATCHDOG_RESPAWN_DELAY  Seconds to sleep before restarting. Default: 30.
  WATCHDOG_RESPAWN_MAX    Restart limit. 0 means unlimited. Default: 0.
  WATCHDOG_RESPAWN_CODES  Exit codes that trigger restart. Default: "137 143".

Exit code 137 usually means SIGKILL, often caused by OOM.
Exit code 143 means SIGTERM.
Ctrl+C exits without respawning.
USAGE
    exit 2
fi

shift

RESPAWN_DELAY="${WATCHDOG_RESPAWN_DELAY:-30}"
RESPAWN_MAX="${WATCHDOG_RESPAWN_MAX:-0}"
RESPAWN_CODES="${WATCHDOG_RESPAWN_CODES:-137 143}"
child_pid=""
restart_count=0
stop_requested="false"

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

code_in_list() {
    local code="$1"
    local candidate
    for candidate in $RESPAWN_CODES; do
        if [[ "$candidate" == "all" || "$candidate" == "$code" ]]; then
            return 0
        fi
    done
    return 1
}

signal_name() {
    local status="$1"
    case "$status" in
        130) printf "SIGINT" ;;
        137) printf "SIGKILL" ;;
        143) printf "SIGTERM" ;;
        *)
            if (( status > 128 )); then
                printf "signal %s" "$((status - 128))"
            else
                printf "exit code %s" "$status"
            fi
            ;;
    esac
}

stop_child() {
    stop_requested="true"
    if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
        echo "[$(timestamp)] Stop requested; terminating watchdog child pid=${child_pid}."
        if command -v setsid >/dev/null 2>&1; then
            kill -TERM "-${child_pid}" 2>/dev/null || kill -TERM "$child_pid" 2>/dev/null || true
        else
            kill -TERM "$child_pid" 2>/dev/null || true
        fi
        wait "$child_pid" 2>/dev/null || true
    fi
    exit 130
}

trap stop_child INT TERM

while true; do
    echo "[$(timestamp)] Launching supervised watchdog: $*"
    if command -v setsid >/dev/null 2>&1; then
        setsid "$@" &
    else
        "$@" &
    fi
    child_pid="$!"

    set +e
    wait "$child_pid"
    status="$?"
    set -e
    child_pid=""

    if [[ "$stop_requested" == "true" ]]; then
        exit 130
    fi

    if (( status == 0 )); then
        echo "[$(timestamp)] Supervised watchdog completed successfully."
        exit 0
    fi

    if (( status == 130 )); then
        echo "[$(timestamp)] Supervised watchdog stopped by Ctrl+C; not respawning."
        exit 130
    fi

    if ! code_in_list "$status"; then
        echo "[$(timestamp)] Supervised watchdog exited with $(signal_name "$status"); not in WATCHDOG_RESPAWN_CODES='${RESPAWN_CODES}'." >&2
        exit "$status"
    fi

    restart_count=$((restart_count + 1))
    if (( RESPAWN_MAX > 0 && restart_count > RESPAWN_MAX )); then
        echo "[$(timestamp)] Supervised watchdog reached WATCHDOG_RESPAWN_MAX=${RESPAWN_MAX}; stop." >&2
        exit "$status"
    fi

    echo "[$(timestamp)] Supervised watchdog ended with $(signal_name "$status"); restart ${restart_count} after ${RESPAWN_DELAY}s." >&2
    sleep "$RESPAWN_DELAY"
done
