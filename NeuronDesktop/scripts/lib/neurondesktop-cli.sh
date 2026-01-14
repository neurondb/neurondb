#!/bin/bash
#-------------------------------------------------------------------------
#
# neurondesktop-cli.sh
#    NeuronDesktop Common CLI Library
#
# Provides standard CLI functions for all NeuronDesktop bash scripts including
# argument parsing, help message display, version information, verbose logging,
# color output, and error handling.
#
# Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
#
# IDENTIFICATION
#    NeuronDesktop/scripts/lib/neurondesktop-cli.sh
#
#-------------------------------------------------------------------------

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}[INFO]${NC} $1" >&2
    fi
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Function to display help message (to be implemented by individual scripts)
show_help() {
    echo "Help message not implemented for this script." >&2
}

# Function to display version
show_version() {
    echo "$SCRIPT_NAME version $1" >&2
}

# Function to display section headers
print_section() {
    echo "" >&2
    echo -e "${CYAN}========================================${NC}" >&2
    echo -e "${CYAN}$1${NC}" >&2
    echo -e "${CYAN}========================================${NC}" >&2
    echo "" >&2
}

# Exit codes
readonly EXIT_SUCCESS=0
readonly EXIT_FAILURE=1
readonly EXIT_MISUSE=2
