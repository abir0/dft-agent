#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🛑 Stopping DFT Agent Services...${NC}"

# Function to stop processes by PID file
stop_by_pid_file() {
    local pid_file=$1
    local service_name=$2
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo -e "${YELLOW}🔄 Stopping $service_name (PID: $pid)...${NC}"
            kill $pid 2>/dev/null
            sleep 2
            # Force kill if still running
            if ps -p $pid > /dev/null 2>&1; then
                echo -e "${YELLOW}🔨 Force stopping $service_name...${NC}"
                kill -9 $pid 2>/dev/null
            fi
            echo -e "${GREEN}✅ $service_name stopped${NC}"
        else
            echo -e "${YELLOW}⚠️  $service_name was not running${NC}"
        fi
        rm -f "$pid_file"
    else
        echo -e "${YELLOW}⚠️  No PID file found for $service_name${NC}"
    fi
}

# Function to stop processes by port
stop_by_port() {
    local port=$1
    local service_name=$2
    
    local pids=$(lsof -ti :$port)
    if [ ! -z "$pids" ]; then
        echo -e "${YELLOW}🔄 Stopping $service_name on port $port...${NC}"
        kill $pids 2>/dev/null
        sleep 2
        # Force kill if still running
        pids=$(lsof -ti :$port)
        if [ ! -z "$pids" ]; then
            echo -e "${YELLOW}🔨 Force stopping $service_name...${NC}"
            kill -9 $pids 2>/dev/null
        fi
        echo -e "${GREEN}✅ $service_name stopped${NC}"
    else
        echo -e "${YELLOW}⚠️  No $service_name found on port $port${NC}"
    fi
}

# Stop services using PID files first (preferred method)
stop_by_pid_file "logs/backend.pid" "Backend API service"
stop_by_pid_file "logs/frontend.pid" "Streamlit frontend"

# Fallback: stop by port if PID files didn't work
stop_by_port 8080 "Backend API service"
stop_by_port 8501 "Streamlit frontend"

# Clean up any remaining processes
echo -e "${YELLOW}🧹 Cleaning up any remaining processes...${NC}"
pkill -f "run_service.py" 2>/dev/null
pkill -f "streamlit run frontend/app.py" 2>/dev/null

echo -e "${GREEN}🎉 All DFT Agent services have been stopped!${NC}"