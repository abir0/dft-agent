#!/bin/bash
# DFT Agent Service Startup Script
# This script starts both the backend and frontend services

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."  # Go to project root

echo -e "${BLUE}🚀 Starting DFT Agent Services${NC}"
echo "=================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}❌ Virtual environment not found. Please run setup first.${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "${YELLOW}📦 Activating virtual environment...${NC}"
source .venv/bin/activate

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

# Check if services are already running
if check_port 8083; then
    echo -e "${YELLOW}⚠️  Backend service already running on port 8083${NC}"
else
    echo -e "${GREEN}🔧 Starting backend service on port 8083...${NC}"
    PYTHONPATH="$SCRIPT_DIR/.." .venv/bin/uvicorn backend.api.main:app --host 0.0.0.0 --port 8083 --reload &
    BACKEND_PID=$!
    echo "Backend PID: $BACKEND_PID"
    
    # Wait for backend to start with retries
    echo -e "${YELLOW}⏳ Waiting for backend to start...${NC}"
    for i in {1..10}; do
        sleep 2
        if check_port 8083; then
            echo -e "${GREEN}✅ Backend service started successfully${NC}"
            break
        fi
        if [ $i -eq 10 ]; then
            echo -e "${RED}❌ Failed to start backend service after 20 seconds${NC}"
            exit 1
        fi
    done
fi

if check_port 8501; then
    echo -e "${YELLOW}⚠️  Frontend service already running on port 8501${NC}"
else
    echo -e "${GREEN}🎨 Starting frontend service on port 8501...${NC}"
    cd frontend
    "$SCRIPT_DIR/../.venv/bin/streamlit" run app.py &
    FRONTEND_PID=$!
    echo "Frontend PID: $FRONTEND_PID"
    cd ..
    
    # Wait for frontend to start with retries
    echo -e "${YELLOW}⏳ Waiting for frontend to start...${NC}"
    for i in {1..10}; do
        sleep 2
        if check_port 8501; then
            echo -e "${GREEN}✅ Frontend service started successfully${NC}"
            break
        fi
        if [ $i -eq 10 ]; then
            echo -e "${RED}❌ Failed to start frontend service after 20 seconds${NC}"
            exit 1
        fi
    done
fi

echo ""
echo -e "${GREEN}🎉 DFT Agent is ready!${NC}"
echo "=================================="
echo -e "${BLUE}📱 Web Interface:${NC} http://localhost:8501"
echo -e "${BLUE}🔧 API Endpoint:${NC} http://localhost:8083"
echo -e "${BLUE}📊 API Info:${NC} http://localhost:8083/info"
echo ""
echo -e "${YELLOW}💡 Tips:${NC}"
echo "• Use Ctrl+C to stop the services"
echo "• Check logs in the 'logs/' directory"
echo "• API documentation available at http://localhost:8083/docs"
echo ""
echo -e "${GREEN}Happy computing! 🧪⚛️${NC}"

# Keep the script running
wait