#!/bin/bash
#-------------------------------------------------------------------------
#
# neuronmcp-run-server.sh
#    NeuronMCP Server Startup Script
#
# Runs the MCP server with configured environment variables. Handles server
# startup, configuration loading, and graceful shutdown.
#
# Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
#
# IDENTIFICATION
#    NeuronMCP/scripts/neuronmcp-run-server.sh
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
$SCRIPT_NAME - NeuronMCP Server Startup Script

Usage:
    $SCRIPT_NAME [OPTIONS]

Description:
    Runs the MCP server with configured environment variables.

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
    # Try neurondb-mcp first, fall back to neuronmcp
    if [ -f "${PROJECT_ROOT}/bin/neurondb-mcp" ]; then
        BINARY_PATH="${PROJECT_ROOT}/bin/neurondb-mcp"
    elif [ -f "${PROJECT_ROOT}/bin/neuronmcp" ]; then
        BINARY_PATH="${PROJECT_ROOT}/bin/neuronmcp"
    elif [ -f "${PROJECT_ROOT}/neurondb-mcp" ]; then
        BINARY_PATH="${PROJECT_ROOT}/neurondb-mcp"
    else
        BINARY_PATH="${PROJECT_ROOT}/bin/neurondb-mcp"
    fi
    
    # Check if binary exists
    require_file "$BINARY_PATH" "MCP server binary"
    
    # Set environment variables (use defaults if not set)
    export NEURONDB_HOST="${NEURONDB_HOST:-localhost}"
    export NEURONDB_PORT="${NEURONDB_PORT:-5432}"
    export NEURONDB_DATABASE="${NEURONDB_DATABASE:-neurondb}"
    export NEURONDB_USER="${NEURONDB_USER:-pgedge}"
    export NEURONDB_PASSWORD="${NEURONDB_PASSWORD:-}"
    
    # Display configuration
    print_section "NeuronMCP Server Starting"
    log_info "Command: $BINARY_PATH"
    log_info "Database: $NEURONDB_DATABASE@$NEURONDB_HOST:$NEURONDB_PORT"
    log_info "User: $NEURONDB_USER"
    log_info ""
    log_info "Press Ctrl+C to shutdown gracefully"
    log_info ""
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Run the MCP server in the foreground using exec
    # This replaces the shell process, maintaining stdio connection
    # The binary itself handles SIGINT and SIGTERM signals for graceful shutdown
    exec "$BINARY_PATH"
}

main "$@"








