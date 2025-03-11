#!/bin/bash

# Stop the API service
kill $(pgrep -f run_service.py)

# Stop the Streamlit app
kill $(pgrep -f "streamlit run app.py")
