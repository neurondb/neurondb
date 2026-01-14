#!/usr/bin/env bash
#-------------------------------------------------------------------------
#
# neuronagent-run.sh
#    NeuronAgent Run Script
#
# Installs Go dependencies and runs the agent server. Compatible with macOS,
# Rocky Linux, Ubuntu, and other Linux distributions. Handles dependency
# installation, environment setup, and server execution.
#
# Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
#
# IDENTIFICATION
#    NeuronAgent/scripts/neuronagent-run.sh
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
echo -e "${BLUE}NeuronAgent Server Startup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if Go is installed
if ! command -v go &> /dev/null; then
    echo -e "${RED}Error: Go is not installed${NC}" >&2
    echo -e "${YELLOW}Please install Go 1.23 or later${NC}" >&2
    exit 1
fi

# Check Go version (basic check)
GO_VERSION=$(go version | awk '{print $3}' | sed 's/go//' 2>/dev/null || echo "unknown")
echo -e "${GREEN}✓ Go version: ${GO_VERSION}${NC}"

# Install Go dependencies
if [ -f "go.mod" ]; then
    echo -e "${BLUE}Installing Go dependencies...${NC}"
    if go mod download && go mod tidy; then
        echo -e "${GREEN}✓ Go dependencies installed${NC}"
    else
        echo -e "${YELLOW}Warning: Go dependencies installation had issues (continuing anyway)${NC}"
    fi
else
    echo -e "${YELLOW}Warning: go.mod not found${NC}"
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

# Install Python dependencies if available (optional for agent server)
install_python_deps

# Set default environment variables if not already set
export DB_HOST="${DB_HOST:-localhost}"
export DB_PORT="${DB_PORT:-5433}"
export DB_NAME="${DB_NAME:-neurondb}"
export DB_USER="${DB_USER:-neurondb}"
export DB_PASSWORD="${DB_PASSWORD:-neurondb}"
export SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
export SERVER_PORT="${SERVER_PORT:-8080}"

# Try to find the binary
BINARY_PATH=""
if [ -f "${SCRIPT_DIR}/bin/neuronagent" ]; then
    BINARY_PATH="${SCRIPT_DIR}/bin/neuronagent"
elif [ -f "${SCRIPT_DIR}/agent-server" ]; then
    BINARY_PATH="${SCRIPT_DIR}/agent-server"
fi

# If binary doesn't exist, try to build it
if [ -z "$BINARY_PATH" ] || [ ! -f "$BINARY_PATH" ]; then
    if [ -f "Makefile" ]; then
        echo -e "${BLUE}Binary not found, building from source...${NC}"
        if make build 2>/dev/null; then
            if [ -f "${SCRIPT_DIR}/bin/neuronagent" ]; then
                BINARY_PATH="${SCRIPT_DIR}/bin/neuronagent"
            fi
        else
            echo -e "${YELLOW}Warning: Build failed, will try to use go run${NC}"
        fi
    fi
fi

# Make binary executable if it exists and isn't executable
if [ -n "$BINARY_PATH" ] && [ -f "$BINARY_PATH" ] && [ ! -x "$BINARY_PATH" ]; then
    chmod +x "$BINARY_PATH"
fi

# Display configuration
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Starting NeuronAgent Server${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Database: ${DB_USER}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
echo -e "Server: ${SERVER_HOST}:${SERVER_PORT}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Run the server (use binary if available, otherwise use go run)
if [ -n "$BINARY_PATH" ] && [ -f "$BINARY_PATH" ]; then
    exec "$BINARY_PATH"
else
    echo -e "${BLUE}Running with: go run cmd/agent-server/main.go${NC}"
    exec go run cmd/agent-server/main.go
fi

