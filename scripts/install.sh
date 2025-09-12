#!/bin/bash

# Install uv for virtual environment management
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate

# Add project src to python path
export PYTHONPATH="$(pwd):$PYTHONPATH"