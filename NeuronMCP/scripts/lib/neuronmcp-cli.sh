#!/bin/bash
#-------------------------------------------------------------------------
#
# neuronmcp-cli.sh
#    NeuronMCP Common CLI Library
#
# Provides standard CLI functions for all NeuronMCP bash scripts including
# argument parsing, help message display, version information, verbose logging,
# color output, and error handling.
#
# Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
#
# IDENTIFICATION
#    NeuronMCP/scripts/lib/neuronmcp-cli.sh
#
#-------------------------------------------------------------------------

# Exit codes
EXIT_SUCCESS=0
EXIT_GENERAL_ERROR=1
EXIT_MISUSE=2

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Verbose flag (should be set by scripts)
VERBOSE="${VERBOSE:-false}"

# Print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1" >&2
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" >&2
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

print_debug() {
    if [ "$VERBOSE" = "true" ]; then
        echo -e "${CYAN}[DEBUG]${NC} $1" >&2
    fi
}

# Logging functions
log_info() {
    print_info "$1"
}

log_success() {
    print_success "$1"
}

log_warning() {
    print_warning "$1"
}

log_error() {
    print_error "$1"
}

log_debug() {
    print_debug "$1"
}

# Error handling
error() {
    print_error "$1"
    exit $EXIT_GENERAL_ERROR
}

error_misuse() {
    print_error "$1"
    exit $EXIT_MISUSE
}

# Version display
show_version() {
    local version="${1:-unknown}"
    echo "$version"
}

# Help message template
# Scripts should override this function
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
    -h, --help      Show this help message
    -V, --version   Show version information
    -v, --verbose   Enable verbose output

EOF
}

# Parse common arguments
# Scripts should call this and then parse their own arguments
parse_common_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit $EXIT_SUCCESS
                ;;
            -V|--version)
                if [ -n "${VERSION:-}" ]; then
                    show_version "$VERSION"
                else
                    show_version "unknown"
                fi
                exit $EXIT_SUCCESS
                ;;
            -v|--verbose)
                VERBOSE=true
                export VERBOSE=true
                shift
                ;;
            *)
                # Return remaining arguments for script-specific parsing
                echo "$@"
                return
                ;;
        esac
    done
    echo ""
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Require command to exist
require_command() {
    if ! command_exists "$1"; then
        error "Required command not found: $1"
    fi
}

# Check if file exists
file_exists() {
    [ -f "$1" ]
}

# Require file to exist
require_file() {
    if ! file_exists "$1"; then
        error "Required file not found: $1"
    fi
}

# Check if directory exists
dir_exists() {
    [ -d "$1" ]
}

# Require directory to exist
require_dir() {
    if ! dir_exists "$1"; then
        error "Required directory not found: $1"
    fi
}

# Print section header
print_section() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
}

# Print separator
print_separator() {
    echo "----------------------------------------"
}

# Success exit
exit_success() {
    exit $EXIT_SUCCESS
}

# Error exit
exit_error() {
    exit $EXIT_GENERAL_ERROR
}

# Misuse exit
exit_misuse() {
    exit $EXIT_MISUSE
}
