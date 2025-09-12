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

stop_port() {
    local port="$1"; local label="$2"
    if ss -tulpn 2>/dev/null | grep -q ":$port"; then
        echo "stopping $label (port $port)"
        fuser -k "$port"/tcp 2>/dev/null || true
        pkill -f "$port" 2>/dev/null || true
    else
        echo "$label not running"
    fi
}

kill_pid_file .backend.pid
kill_pid_file .frontend.pid

stop_port 8083 backend
stop_port 8501 frontend

pkill -f 'uvicorn backend.api.main:app' 2>/dev/null || true
pkill -f 'streamlit run app.py' 2>/dev/null || true
pkill -f 'streamlit run frontend/app.py' 2>/dev/null || true

echo "services stopped"
