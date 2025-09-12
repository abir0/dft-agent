#!/bin/bash
# Minimal run script: starts backend (8083) and frontend (8501).
set -e

if [ ! -d .venv ]; then
    echo ".venv not found. Run scripts/install.sh" >&2
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Load environment variables
set -a
source .env 2>/dev/null || true
set +a

PORT=${PORT:-8083}
HOST=${HOST:-0.0.0.0}

backend_running() { ss -tulpn 2>/dev/null | grep -q ":$PORT"; }
frontend_running() { ss -tulpn 2>/dev/null | grep -q ':8501'; }

if backend_running; then
    echo "backend already running ($PORT)"
else
    nohup uvicorn backend.api.main:app --host $HOST --port $PORT --reload > logs/service.log 2>&1 &
    echo $! > .backend.pid
    echo "backend started pid $(cat .backend.pid)"
fi

if frontend_running; then
    echo "frontend already running (8501)"
else
    nohup streamlit run frontend/app.py > logs/service.log 2>&1 &
    echo $! > .frontend.pid
    echo "frontend started pid $(cat .frontend.pid)"
fi

echo "services running: api http://localhost:$PORT  ui http://localhost:8501"
