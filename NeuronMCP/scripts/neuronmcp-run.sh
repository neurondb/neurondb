#!/usr/bin/env bash
#-------------------------------------------------------------------------
#
# neuronmcp-run.sh
#    NeuronMCP Run Script
#
# Installs dependencies from requirements.txt and runs the MCP server.
# Compatible with macOS, Rocky Linux, Ubuntu, and other Linux distributions.
# Handles dependency installation, environment setup, and server execution.
#
# Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
#
# IDENTIFICATION
#    NeuronMCP/scripts/neuronmcp-run.sh
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
$SCRIPT_NAME - NeuronMCP Run Script

Usage:
    $SCRIPT_NAME [OPTIONS]

Description:
    Installs dependencies from requirements.txt and runs the MCP server.
    Compatible with macOS, Rocky Linux, Ubuntu, and other Linux distributions.

Options:
    -h, --help      Show this help message
    -V, --version   Show version information
    -v, --verbose   Enable verbose output

Environment Variables:
    NEURONDB_HOST          Database host (default: localhost)
    NEURONDB_PORT          Database port (default: 5432)
    NEURONDB_DATABASE      Database name (default: neurondb)
    NEURONDB_USER          Database user (default: pgedge)
    NEURONDB_PASSWORD      Database password

Examples:
    $SCRIPT_NAME
    $SCRIPT_NAME --verbose

EOF
}

main() {
    parse_args "$@"
    
    cd "$PROJECT_ROOT"
    
    print_section "NeuronMCP Server Startup"

    # Function to install Python dependencies
    install_python_deps() {
        if [ ! -f "requirements.txt" ]; then
            print_warning "requirements.txt not found, skipping Python dependencies"
            return 0
        fi

        print_success "Found requirements.txt"
        log_info "Installing Python dependencies..."
        
        # Try different pip installation methods
        local pip_cmd=""
        if command -v pip3 &> /dev/null; then
            pip_cmd="pip3"
        elif python3 -m pip --version &> /dev/null 2>&1; then
            pip_cmd="python3 -m pip"
        else
            print_warning "pip not found, skipping Python dependencies"
            return 0
        fi

        # Try installation (without --user first, then with --user if needed)
        if $pip_cmd install -r requirements.txt --quiet --disable-pip-version-check 2>/dev/null; then
            print_success "Python dependencies installed"
        elif $pip_cmd install --user -r requirements.txt --quiet --disable-pip-version-check 2>/dev/null; then
            print_success "Python dependencies installed (user install)"
        else
            print_warning "Python dependencies installation had issues (continuing anyway)"
        fi
    }

    # Install Python dependencies if Python is available
    if command -v python3 &> /dev/null; then
        install_python_deps
    else
        log_info "python3 not found, skipping Python dependencies"
    fi

    # Check if Go is available (needed for building)
    if ! command -v go &> /dev/null; then
        print_warning "go is not installed, cannot build from source"
    fi

    # Try to find the binary
    BINARY_PATH=""
    if [ -f "${PROJECT_ROOT}/bin/neurondb-mcp" ]; then
        BINARY_PATH="${PROJECT_ROOT}/bin/neurondb-mcp"
    elif [ -f "${PROJECT_ROOT}/bin/neuronmcp" ]; then
        BINARY_PATH="${PROJECT_ROOT}/bin/neuronmcp"
    elif [ -f "${PROJECT_ROOT}/neurondb-mcp" ]; then
        BINARY_PATH="${PROJECT_ROOT}/neurondb-mcp"
    elif [ -f "${PROJECT_ROOT}/neuronmcp" ]; then
        BINARY_PATH="${PROJECT_ROOT}/neuronmcp"
    fi

    # If binary doesn't exist, try to build it
    if [ -z "$BINARY_PATH" ] || [ ! -f "$BINARY_PATH" ]; then
        if command -v go &> /dev/null && [ -f "Makefile" ]; then
            log_info "Binary not found, building from source..."
            if make build 2>/dev/null; then
                if [ -f "${PROJECT_ROOT}/bin/neurondb-mcp" ]; then
                    BINARY_PATH="${PROJECT_ROOT}/bin/neurondb-mcp"
                elif [ -f "${PROJECT_ROOT}/bin/neuronmcp" ]; then
                    BINARY_PATH="${PROJECT_ROOT}/bin/neuronmcp"
                fi
            else
                print_warning "Build failed, will try to use existing binary"
            fi
        fi
    fi

    # Check if binary exists
    if [ -z "$BINARY_PATH" ] || [ ! -f "$BINARY_PATH" ]; then
        error "MCP server binary not found. Please build the binary first with: make build"
    fi

    # Make binary executable if it isn't
    if [ ! -x "$BINARY_PATH" ]; then
        chmod +x "$BINARY_PATH"
    fi

    # Set default environment variables if not already set
    export NEURONDB_HOST="${NEURONDB_HOST:-localhost}"
    export NEURONDB_PORT="${NEURONDB_PORT:-5432}"
    export NEURONDB_DATABASE="${NEURONDB_DATABASE:-neurondb}"
    export NEURONDB_USER="${NEURONDB_USER:-pgedge}"
    export NEURONDB_PASSWORD="${NEURONDB_PASSWORD:-}"

    # Display configuration
    log_info ""
    print_section "Starting NeuronMCP Server"
    log_info "Binary: ${BINARY_PATH}"
    log_info "Database: ${NEURONDB_DATABASE}@${NEURONDB_HOST}:${NEURONDB_PORT}"
    log_info "User: ${NEURONDB_USER}"
    log_info ""

    # Run the MCP server
    exec "$BINARY_PATH"
}

main "$@"

