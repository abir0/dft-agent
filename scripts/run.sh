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

# Function to check if port is in use (cross-platform)
check_port() {
    local port=$1
    if command -v lsof >/dev/null 2>&1; then
        # macOS and most Unix systems
        lsof -i :$port >/dev/null 2>&1
    elif command -v ss >/dev/null 2>&1; then
        # Linux systems with iproute2
        ss -tulpn | grep -q ":$port"
    elif command -v netstat >/dev/null 2>&1; then
        # Fallback to netstat
        netstat -an | grep -q ":$port"
    else
        # No port checking available
        return 1
    fi
}

# Create logs directory if it doesn't exist
mkdir -p logs

if check_port 8083; then
    echo "backend already running (8083)"
else
    nohup uvicorn backend.api.main:app --host 0.0.0.0 --port 8083 --reload > logs/service.log 2>&1 &
    echo $! > .backend.pid
    echo "backend started pid $(cat .backend.pid)"
fi

if check_port 8501; then
    echo "frontend already running (8501)"
else
    nohup streamlit run frontend/app.py > logs/app.log 2>&1 &
    echo $! > .frontend.pid
    echo "frontend started pid $(cat .frontend.pid)"
fi

echo "services running: api http://localhost:8083  ui http://localhost:8501"
