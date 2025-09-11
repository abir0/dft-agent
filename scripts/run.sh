#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Add project src to python path
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Create logs dir
mkdir -p logs

# Run the API service
nohup python backend/run_service.py > logs/service.log 2>&1 &

# Run the Streamlit app
nohup streamlit run frontend/app.py > logs/app.log 2>&1 &