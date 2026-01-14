#!/usr/bin/env bash
#-------------------------------------------------------------------------
#
# neurondesktop-run.sh
#    NeuronDesktop Run Script
#
# Installs npm and Go dependencies, then runs both frontend and backend.
# Compatible with macOS, Rocky Linux, Ubuntu, and other Linux distributions.
# Handles dependency installation, environment setup, and application execution.
#
# Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
#
# IDENTIFICATION
#    NeuronDesktop/scripts/neurondesktop-run.sh
#
#-------------------------------------------------------------------------

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to script directory
cd "$SCRIPT_DIR"

# Colors for output (only if terminal supports it)
if [[ -t 1 ]] && [[ "${TERM:-}" != "dumb" ]]; then
    GREEN='\033[0;32m'
    BLUE='\033[0;34m'
    YELLOW='\033[1;33m'
    RED='\033[0;31m'
    NC='\033[0m'
else
    GREEN=''
    BLUE=''
    YELLOW=''
    RED=''
    NC=''
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}NeuronDesktop Startup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to cleanup background processes
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down services...${NC}"
    if [ -n "${BACKEND_PID:-}" ]; then
        kill "${BACKEND_PID}" 2>/dev/null || true
        wait "${BACKEND_PID}" 2>/dev/null || true
    fi
    if [ -n "${FRONTEND_PID:-}" ]; then
        kill "${FRONTEND_PID}" 2>/dev/null || true
        wait "${FRONTEND_PID}" 2>/dev/null || true
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Check if Node.js is installed
if ! command -v npm &> /dev/null; then
    echo -e "${RED}Error: npm is not installed${NC}" >&2
    echo -e "${YELLOW}Please install Node.js 20+ and npm${NC}" >&2
    exit 1
fi

# Check Node.js version
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version 2>/dev/null || echo "unknown")
    echo -e "${GREEN}✓ Node.js version: ${NODE_VERSION}${NC}"
fi

# Install frontend dependencies
if [ -d "frontend" ] && [ -f "frontend/package.json" ]; then
    echo -e "${BLUE}Installing frontend dependencies (npm)...${NC}"
    cd frontend
    if [ ! -d "node_modules" ]; then
        npm install --no-audit --no-fund
    else
        echo -e "${GREEN}✓ Frontend dependencies already installed${NC}"
    fi
    cd ..
    echo -e "${GREEN}✓ Frontend dependencies ready${NC}"
else
    echo -e "${YELLOW}Warning: frontend/package.json not found${NC}"
fi

# Check if Go is installed
if ! command -v go &> /dev/null; then
    echo -e "${RED}Error: Go is not installed${NC}" >&2
    echo -e "${YELLOW}Please install Go 1.23 or later${NC}" >&2
    exit 1
fi

# Check Go version
GO_VERSION=$(go version | awk '{print $3}' | sed 's/go//' 2>/dev/null || echo "unknown")
echo -e "${GREEN}✓ Go version: ${GO_VERSION}${NC}"

# Install backend Go dependencies
if [ -d "api" ] && [ -f "api/go.mod" ]; then
    echo -e "${BLUE}Installing backend Go dependencies...${NC}"
    cd api
    if go mod download && go mod tidy; then
        echo -e "${GREEN}✓ Backend dependencies installed${NC}"
    else
        echo -e "${YELLOW}Warning: Go dependencies installation had issues (continuing anyway)${NC}"
    fi
    cd ..
else
    echo -e "${YELLOW}Warning: api/go.mod not found${NC}"
fi

# Function to install Python dependencies (optional)
install_python_deps() {
    if [ ! -f "requirements.txt" ]; then
        return 0
    fi

    echo -e "${GREEN}✓ Found requirements.txt${NC}"
    echo -e "${BLUE}Installing Python dependencies...${NC}"
    
    if ! command -v python3 &> /dev/null; then
        echo -e "${YELLOW}Info: python3 not found, skipping Python dependencies${NC}"
        return 0
    fi

    local pip_cmd=""
    if command -v pip3 &> /dev/null; then
        pip_cmd="pip3"
    elif python3 -m pip --version &> /dev/null 2>&1; then
        pip_cmd="python3 -m pip"
    else
        echo -e "${YELLOW}Info: pip not found, skipping Python dependencies${NC}"
        return 0
    fi

    # Try installation (without --user first, then with --user if needed)
    if $pip_cmd install -r requirements.txt --quiet --disable-pip-version-check 2>/dev/null; then
        echo -e "${GREEN}✓ Python dependencies installed${NC}"
    elif $pip_cmd install --user -r requirements.txt --quiet --disable-pip-version-check 2>/dev/null; then
        echo -e "${GREEN}✓ Python dependencies installed (user install)${NC}"
    else
        echo -e "${YELLOW}Warning: Python dependencies installation had issues (continuing anyway)${NC}"
    fi
}

# Install Python dependencies if available (optional)
install_python_deps

# Set default environment variables if not already set
export DB_HOST="${DB_HOST:-localhost}"
export DB_PORT="${DB_PORT:-5433}"
export DB_NAME="${DB_NAME:-neurondesk}"
export DB_USER="${DB_USER:-neurondb}"
export DB_PASSWORD="${DB_PASSWORD:-neurondb}"
export SERVER_PORT="${SERVER_PORT:-8081}"

# Try to find the backend binary
BACKEND_BINARY=""
if [ -f "${SCRIPT_DIR}/bin/neurondesktop" ]; then
    BACKEND_BINARY="${SCRIPT_DIR}/bin/neurondesktop"
elif [ -f "${SCRIPT_DIR}/api/server" ]; then
    BACKEND_BINARY="${SCRIPT_DIR}/api/server"
fi

# If binary doesn't exist, try to build it
if [ -z "$BACKEND_BINARY" ] || [ ! -f "$BACKEND_BINARY" ]; then
    if [ -f "Makefile" ]; then
        echo -e "${BLUE}Backend binary not found, building from source...${NC}"
        if make build-api 2>/dev/null; then
            if [ -f "${SCRIPT_DIR}/bin/neurondesktop" ]; then
                BACKEND_BINARY="${SCRIPT_DIR}/bin/neurondesktop"
            fi
        else
            echo -e "${BLUE}Building backend manually...${NC}"
            cd api
            mkdir -p ../bin
            if go build -o ../bin/neurondesktop ./cmd/server 2>/dev/null; then
                BACKEND_BINARY="${SCRIPT_DIR}/bin/neurondesktop"
            fi
            cd ..
        fi
    fi
fi

# Make binary executable if it exists and isn't executable
if [ -n "$BACKEND_BINARY" ] && [ -f "$BACKEND_BINARY" ] && [ ! -x "$BACKEND_BINARY" ]; then
    chmod +x "$BACKEND_BINARY"
fi

# Display configuration
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Starting NeuronDesktop${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Database: ${DB_USER}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
echo -e "Backend API: http://localhost:${SERVER_PORT}"
echo -e "Frontend: http://localhost:3000"
echo -e "${BLUE}========================================${NC}"
echo ""

# Start backend
echo -e "${BLUE}Starting backend server...${NC}"
if [ -n "$BACKEND_BINARY" ] && [ -f "$BACKEND_BINARY" ]; then
    "$BACKEND_BINARY" > /dev/null 2>&1 &
    BACKEND_PID=$!
else
    cd api
    go run ./cmd/server > /dev/null 2>&1 &
    BACKEND_PID=$!
    cd ..
fi
echo -e "${GREEN}✓ Backend started (PID: $BACKEND_PID)${NC}"

# Wait a bit for backend to start
sleep 2

# Start frontend
if [ -d "frontend" ] && [ -f "frontend/package.json" ]; then
    echo -e "${BLUE}Starting frontend server...${NC}"
    cd frontend
    npm run dev > /dev/null 2>&1 &
    FRONTEND_PID=$!
    cd ..
    echo -e "${GREEN}✓ Frontend started (PID: $FRONTEND_PID)${NC}"
else
    echo -e "${YELLOW}Warning: Frontend directory not found, skipping frontend${NC}"
    FRONTEND_PID=""
fi

echo ""
echo -e "${GREEN}✓ NeuronDesktop is running!${NC}"
echo -e "${BLUE}Backend API: http://localhost:${SERVER_PORT}${NC}"
if [ -n "${FRONTEND_PID:-}" ]; then
    echo -e "${BLUE}Frontend: http://localhost:3000${NC}"
fi
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

# Wait for background processes (handle both cases)
if [ -n "${FRONTEND_PID:-}" ]; then
    wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || wait
else
    wait "$BACKEND_PID" 2>/dev/null || wait
fi

