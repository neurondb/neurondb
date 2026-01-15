#!/bin/bash
# ====================================================================
# Unified Component Installation Script
# ====================================================================
# Installs NeuronMCP, NeuronAgent, and/or NeuronDesktop from source
# ====================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT_NAME=$(basename "$0")

# Source helper functions
source "$SCRIPT_DIR/install-helpers.sh"

# Default values
COMPONENTS=""
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"
ENABLE_SERVICE=false
SKIP_BUILD=false
SKIP_DB_SETUP=false
VERBOSE=false

show_help() {
    cat << EOF
Unified Component Installation Script

Usage: $SCRIPT_NAME [OPTIONS] [COMPONENTS...]

Components (default: all):
    neuronmcp       Install NeuronMCP
    neuronagent     Install NeuronAgent
    neurondesktop   Install NeuronDesktop
    all             Install all components (default)

Options:
    --prefix PATH      Installation prefix (default: /usr/local)
    --enable-service   Enable and start system services
    --skip-build       Skip building from source
    --skip-db-setup    Skip database setup
    -v, --verbose      Enable verbose output
    -h, --help         Show this help message

Examples:
    # Install all components
    $SCRIPT_NAME

    # Install specific components
    $SCRIPT_NAME neuronmcp neuronagent

    # Install with services enabled
    $SCRIPT_NAME --enable-service

    # Install to custom location
    $SCRIPT_NAME --prefix ~/neurondb

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prefix)
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        --enable-service)
            ENABLE_SERVICE=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-db-setup)
            SKIP_DB_SETUP=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        neuronmcp|neuronagent|neurondesktop|all)
            if [ -z "$COMPONENTS" ]; then
                COMPONENTS="$1"
            else
                COMPONENTS="$COMPONENTS $1"
            fi
            shift
            ;;
        *)
            print_error "Unknown option or component: $1"
            show_help
            exit 1
            ;;
    esac
done

# Default to all if no components specified
if [ -z "$COMPONENTS" ]; then
    COMPONENTS="all"
fi

print_info "Starting unified component installation..."
print_info "Components: $COMPONENTS"
print_info "Install prefix: $INSTALL_PREFIX"

# Export variables for sub-scripts
export INSTALL_PREFIX
export ENABLE_SERVICE
export SKIP_BUILD
export SKIP_DB_SETUP
export VERBOSE

# Install components
if [ "$COMPONENTS" = "all" ] || echo "$COMPONENTS" | grep -q "neuronmcp"; then
    print_info "Installing NeuronMCP..."
    "$SCRIPT_DIR/install-neuronmcp.sh" \
        --prefix "$INSTALL_PREFIX" \
        $([ "$ENABLE_SERVICE" = true ] && echo "--enable-service") \
        $([ "$SKIP_BUILD" = true ] && echo "--skip-build") \
        $([ "$SKIP_DB_SETUP" = true ] && echo "--skip-db-setup") \
        $([ "$VERBOSE" = true ] && echo "--verbose")
fi

if [ "$COMPONENTS" = "all" ] || echo "$COMPONENTS" | grep -q "neuronagent"; then
    print_info "Installing NeuronAgent..."
    "$SCRIPT_DIR/install-neuronagent.sh" \
        --prefix "$INSTALL_PREFIX" \
        $([ "$ENABLE_SERVICE" = true ] && echo "--enable-service") \
        $([ "$SKIP_BUILD" = true ] && echo "--skip-build") \
        $([ "$SKIP_DB_SETUP" = true ] && echo "--skip-db-setup") \
        $([ "$VERBOSE" = true ] && echo "--verbose")
fi

if [ "$COMPONENTS" = "all" ] || echo "$COMPONENTS" | grep -q "neurondesktop"; then
    print_info "Installing NeuronDesktop..."
    "$SCRIPT_DIR/install-neurondesktop.sh" \
        --prefix "$INSTALL_PREFIX" \
        $([ "$ENABLE_SERVICE" = true ] && echo "--enable-service") \
        $([ "$SKIP_BUILD" = true ] && echo "--skip-build") \
        $([ "$SKIP_DB_SETUP" = true ] && echo "--skip-db-setup") \
        $([ "$VERBOSE" = true ] && echo "--verbose")
fi

print_success "All component installations completed!"




