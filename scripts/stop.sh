#!/bin/bash
# DFT Agent Service Stop Script
# This script stops both the backend and frontend services

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🛑 Stopping DFT Agent Services...${NC}"

# Function to stop processes by port
stop_by_port() {
    local port=$1
    local service_name=$2
    
    if ss -tulpn | grep -q ":$port"; then
        echo -e "${YELLOW}🔄 Stopping $service_name on port $port...${NC}"
        
        # Get PIDs using ss and fuser
        local pids=$(fuser $port/tcp 2>/dev/null || echo "")
        if [ ! -z "$pids" ]; then
            echo "Found PIDs: $pids"
            kill $pids 2>/dev/null
            sleep 2
            
            # Force kill if still running
            pids=$(fuser $port/tcp 2>/dev/null || echo "")
            if [ ! -z "$pids" ]; then
                echo -e "${YELLOW}🔨 Force stopping $service_name...${NC}"
                kill -9 $pids 2>/dev/null
            fi
        fi
        
        # Double check with pkill
        pkill -f ":$port" 2>/dev/null || true
        
        echo -e "${GREEN}✅ $service_name stopped${NC}"
    else
        echo -e "${YELLOW}⚠️  No $service_name found on port $port${NC}"
    fi
}

# Stop services by port
stop_by_port 8083 "Backend API service"
stop_by_port 8501 "Streamlit frontend"

# Clean up any remaining processes
echo -e "${YELLOW}🧹 Cleaning up any remaining processes...${NC}"
pkill -f "uvicorn backend.api.main:app" 2>/dev/null || true
pkill -f "streamlit run app.py" 2>/dev/null || true
pkill -f "streamlit run frontend/app.py" 2>/dev/null || true

echo -e "${GREEN}🎉 All DFT Agent services have been stopped!${NC}"