#!/bin/bash
# Minimal stop script: terminates backend (8083) and frontend (8501) processes.
set -e

kill_pid_file() {
    local f="$1"
    [ -f "$f" ] || return 0
    local pid
    pid="$(cat "$f" 2>/dev/null || true)"
    if [ -n "${pid}" ] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        sleep 1
        kill -9 "$pid" 2>/dev/null || true
        echo "stopped pid $pid ($f)"
    fi
    rm -f "$f"
}

kill_pid_file .backend.pid
kill_pid_file .frontend.pid

pkill -f 'uvicorn backend.api.main:app' 2>/dev/null || true
pkill -f 'streamlit run app.py' 2>/dev/null || true
pkill -f 'streamlit run frontend/app.py' 2>/dev/null || true

echo "services stopped"
