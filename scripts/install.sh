#!/bin/bash
set -e

# Python check
PY_BIN="$(command -v python3 || command -v python || true)"
[ -n "$PY_BIN" ] || { echo "python not found"; exit 1; }

# Install uv if not present
if ! command -v uv >/dev/null 2>&1; then
	curl -LsSf https://astral.sh/uv/install.sh | sh
	export PATH="$HOME/.local/bin:$PATH"
fi

# Create dot env file
[ -f .env ] || { [ -f env.example ] && cp env.example .env || true; }

# Install dependencies
uv sync

# Create directories
mkdir -p data/inputs logs WORKSPACE

# Add source path to python environment variable
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "Done. Edit .env to complete setup."
