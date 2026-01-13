#!/bin/bash
# Run NeuronMCP with Python virtual environment
# This script ensures the Python environment is activated before running NeuronMCP

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
    
    echo "Installing dependencies..."
    source .venv/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
    
    echo "✓ Virtual environment created and dependencies installed"
fi

# Activate virtual environment
source .venv/bin/activate

# Set PYTHON environment variable so Go code can find the right Python
export PYTHON="$(which python3)"

# Run NeuronMCP server
echo "Starting NeuronMCP with Python environment..."
echo "Python: $PYTHON"
echo ""

# Check if run_mcp_server.sh exists, otherwise run the binary directly
if [ -f "run_mcp_server.sh" ]; then
    exec ./run_mcp_server.sh "$@"
else
    # Try to find the binary
    if [ -f "neurondb-mcp" ]; then
        exec ./neurondb-mcp "$@"
    elif [ -f "bin/neuronmcp" ]; then
        exec ./bin/neuronmcp "$@"
    else
        echo "Error: Could not find NeuronMCP binary"
        echo "Please build NeuronMCP first: cd NeuronMCP && make"
        exit 1
    fi
fi



