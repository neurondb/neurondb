#!/bin/bash
#-------------------------------------------------------------------------
#
# neuronmcp-run-with-venv.sh
#    Run NeuronMCP with Python Virtual Environment
#
# Ensures the Python virtual environment is activated before running NeuronMCP.
# Creates and manages virtual environment, installs dependencies, and launches
# the server in an isolated Python environment.
#
# Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
#
# IDENTIFICATION
#    NeuronMCP/scripts/neuronmcp-run-with-venv.sh
#
#-------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT_NAME=$(basename "$0")
VERSION="1.0.0"
VERBOSE=false

# Source common CLI library
source "${PROJECT_ROOT}/scripts/lib/neuronmcp-cli.sh" || {
    echo "Error: Failed to load CLI library" >&2
    exit 1
}

# Parse arguments
parse_args() {
    local remaining_args
    remaining_args=$(parse_common_args "$@")
    
    eval set -- "$remaining_args"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            *)
                error_misuse "Unknown option: $1"
                ;;
        esac
    done
}

show_help() {
    cat << EOF
$SCRIPT_NAME - Run NeuronMCP with Python virtual environment

Usage:
    $SCRIPT_NAME [OPTIONS]

Description:
    Ensures the Python environment is activated before running NeuronMCP.

Options:
    -h, --help      Show this help message
    -V, --version   Show version information
    -v, --verbose   Enable verbose output

Examples:
    $SCRIPT_NAME
    $SCRIPT_NAME --verbose

EOF
}

main() {
    parse_args "$@"
    
    cd "$PROJECT_ROOT"

    # Check if virtual environment exists
    if [ ! -d ".venv" ]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv .venv
        
        log_info "Installing dependencies..."
        source .venv/bin/activate
        pip install --upgrade pip setuptools wheel
        pip install -r requirements.txt
        
        print_success "Virtual environment created and dependencies installed"
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Set PYTHON environment variable so Go code can find the right Python
    export PYTHON="$(which python3)"
    
    # Run NeuronMCP server
    log_info "Starting NeuronMCP with Python environment..."
    log_info "Python: $PYTHON"
    log_info ""
    
    # Check if run-server script exists, otherwise run the binary directly
    if [ -f "scripts/neuronmcp-run-server.sh" ]; then
        exec ./scripts/neuronmcp-run-server.sh "$@"
    else
        # Try to find the binary
        if [ -f "neurondb-mcp" ]; then
            exec ./neurondb-mcp "$@"
        elif [ -f "bin/neuronmcp" ]; then
            exec ./bin/neuronmcp "$@"
        elif [ -f "bin/neurondb-mcp" ]; then
            exec ./bin/neurondb-mcp "$@"
        else
            error "Could not find NeuronMCP binary. Please build NeuronMCP first: make build"
        fi
    fi
}

main "$@"



