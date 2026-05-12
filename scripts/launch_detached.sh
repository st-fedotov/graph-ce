#!/usr/bin/env bash
# Launch graph-ce fully detached from the controlling shell / Claude Code
# harness, so the coordinator survives any session-level cleanup.
#
# Usage: scripts/launch_detached.sh <run_name> [--override key=value ...]
#
# After launch:
#   - PID is written to runs/<run_name>.pid
#   - Pre-coordinator launch output (e.g. import errors) lands in
#     runs/<run_name>_launch.log
#   - Full per-process logs land under runs/<run_name>/ as usual
#
# Stopping the run cleanly:
#   kill "$(cat runs/<run_name>.pid)"
# This sends SIGTERM, which the coordinator's signal handler converts into
# stop_event.set() for a graceful island shutdown.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <run_name> [--override key=value ...]" >&2
    exit 2
fi

RUN_NAME="$1"
shift

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -x .venv/bin/graph-ce ]]; then
    echo "error: .venv/bin/graph-ce not found; install with: pip install -e ." >&2
    exit 1
fi

mkdir -p runs
LAUNCH_LOG="runs/${RUN_NAME}_launch.log"
PID_FILE="runs/${RUN_NAME}.pid"

if [[ -e "$PID_FILE" ]]; then
    OLD_PID="$(cat "$PID_FILE" 2>/dev/null || echo)"
    if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "error: PID file $PID_FILE points at live PID $OLD_PID. Stop it first:" >&2
        echo "         kill $OLD_PID" >&2
        exit 1
    fi
    rm -f "$PID_FILE"
fi

# setsid: detach from the controlling terminal and start a new session
# nohup: ignore SIGHUP so the process survives shell exit
# </dev/null: no stdin (don't keep terminal attached)
# >"$LAUNCH_LOG" 2>&1: redirect stdout/stderr (the coordinator also writes
#                     its own log file inside the run dir)
# &: background it
setsid nohup .venv/bin/graph-ce --run-name "$RUN_NAME" "$@" \
    </dev/null >"$LAUNCH_LOG" 2>&1 &
PID=$!
disown "$PID" 2>/dev/null || true

echo "$PID" > "$PID_FILE"

# Confirm it's actually running and reparented away from this shell.
sleep 1
if ! kill -0 "$PID" 2>/dev/null; then
    echo "error: process $PID exited immediately. Check $LAUNCH_LOG:" >&2
    tail -20 "$LAUNCH_LOG" >&2 || true
    exit 1
fi

echo "Launched graph-ce detached"
echo "  run_name : $RUN_NAME"
echo "  pid      : $PID  (saved to $PID_FILE)"
echo "  log      : $LAUNCH_LOG"
echo "  run dir  : runs/$RUN_NAME/  (will appear momentarily)"
echo
echo "Status:  .venv/bin/graph-ce-status --run-dir runs/$RUN_NAME"
echo "Stop  :  kill $PID"
