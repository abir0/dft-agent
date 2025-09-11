#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting DFT Agent Services...${NC}"

# Check if virtual environment exists
if [ ! -f ".venv/bin/activate" ]; then
    echo -e "${RED}❌ Virtual environment not found. Please run scripts/install.sh first.${NC}"
    exit 1
fi

# Activate the virtual environment
echo -e "${YELLOW}📦 Activating virtual environment...${NC}"
source .venv/bin/activate

# Add project src to python path
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Create logs dir
mkdir -p logs

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to kill processes on a port
kill_port() {
    local port=$1
    local pids=$(lsof -ti :$port)
    if [ ! -z "$pids" ]; then
        echo -e "${YELLOW}🔄 Stopping existing services on port $port...${NC}"
        kill $pids 2>/dev/null
        sleep 2
        # Force kill if still running
        pids=$(lsof -ti :$port)
        if [ ! -z "$pids" ]; then
            kill -9 $pids 2>/dev/null
        fi
    fi
}

# Check and handle port conflicts
if check_port 8080; then
    echo -e "${YELLOW}⚠️  Port 8080 is already in use. Stopping existing service...${NC}"
    kill_port 8080
fi

if check_port 8501; then
    echo -e "${YELLOW}⚠️  Port 8501 is already in use. Stopping existing service...${NC}"
    kill_port 8501
fi

# Wait a moment for ports to be released
sleep 2

# Run the API service
echo -e "${GREEN}🔧 Starting backend API service on port 8080...${NC}"
nohup python backend/run_service.py > logs/service.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > logs/backend.pid

# Wait for backend to start
echo -e "${YELLOW}⏳ Waiting for backend service to start...${NC}"
sleep 8

# Check if backend started successfully
if ! check_port 8080; then
    echo -e "${RED}❌ Failed to start backend service. Check logs/service.log for details.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Backend service started successfully!${NC}"

# Run the Streamlit app
echo -e "${GREEN}🎨 Starting Streamlit frontend on port 8501...${NC}"
nohup streamlit run frontend/app.py --server.port 8501 --browser.gatherUsageStats false > logs/app.log 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > logs/frontend.pid

# Wait for frontend to start
echo -e "${YELLOW}⏳ Waiting for frontend service to start...${NC}"
sleep 8

# Check if frontend started successfully
if ! check_port 8501; then
    echo -e "${RED}❌ Failed to start frontend service. Check logs/app.log for details.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Frontend service started successfully!${NC}"

# Open browser
echo -e "${BLUE}🌐 Opening browser to http://localhost:8501...${NC}"
if command -v open >/dev/null 2>&1; then
    # macOS
    open http://localhost:8501
elif command -v xdg-open >/dev/null 2>&1; then
    # Linux
    xdg-open http://localhost:8501
elif command -v start >/dev/null 2>&1; then
    # Windows
    start http://localhost:8501
else
    echo -e "${YELLOW}⚠️  Could not automatically open browser. Please visit http://localhost:8501 manually.${NC}"
fi

echo -e "${GREEN}🎉 DFT Agent is now running!${NC}"
echo -e "${BLUE}📊 Backend API: http://localhost:8080${NC}"
echo -e "${BLUE}🎨 Frontend UI: http://localhost:8501${NC}"
echo -e "${YELLOW}📝 Logs are available in the logs/ directory${NC}"
echo -e "${YELLOW}🛑 To stop services, run: scripts/stop.sh${NC}"