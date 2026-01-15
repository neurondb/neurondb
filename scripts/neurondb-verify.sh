#!/bin/bash
#
# NeuronDB Verification Script
# Unified script for all verification operations
#
# Usage:
#   ./neurondb-verify.sh COMMAND [OPTIONS]
#
# Commands:
#   packages    Verify DEB/RPM packages
#   docker      Verify Docker ecosystem
#   helm        Verify Helm chart
#   storage     Verify cloud storage configurations
#   all         Run all verifications
#
# This script is completely self-sufficient with no external dependencies.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT_NAME=$(basename "$0")

# Colors (inline - no external dependency)
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Default configuration
COMMAND=""
VERBOSE=false


log_info() {
    echo -e "${CYAN}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_debug() {
    [[ "$VERBOSE" == "true" ]] && echo -e "${MAGENTA}[DEBUG]${NC} $*"
}

print_header() {
    echo ""
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  ${BOLD}NeuronDB Verification${NC}                                    ${BLUE}║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}


show_help() {
    cat << EOF
${BOLD}NeuronDB Verification${NC}

${BOLD}Usage:${NC}
    ${SCRIPT_NAME} COMMAND [OPTIONS]

${BOLD}Commands:${NC}
    packages    Verify DEB/RPM packages
    docker      Verify Docker ecosystem
    helm        Verify Helm chart
    storage     Verify cloud storage configurations
    all         Run all verifications

${BOLD}Packages Command:${NC}
    ${SCRIPT_NAME} packages [OPTIONS]
    
    Delegates to: neurondb-pkgs.sh verify

${BOLD}Docker Command:${NC}
    ${SCRIPT_NAME} docker [OPTIONS]
    
    Delegates to: neurondb-docker.sh verify

${BOLD}Helm Command:${NC}
    ${SCRIPT_NAME} helm [OPTIONS]
    
    Delegates to: neurondb-helm.sh validate

${BOLD}Storage Command:${NC}
    ${SCRIPT_NAME} storage [OPTIONS]
    
    Delegates to: neurondb-storage.sh validate

${BOLD}All Command:${NC}
    ${SCRIPT_NAME} all [OPTIONS]
    
    Runs all verification commands in sequence.

${BOLD}Global Options:${NC}
    -h, --help              Show this help message
    -v, --verbose           Enable verbose output
    -V, --version           Show version information

${BOLD}Examples:${NC}
    # Verify packages
    ${SCRIPT_NAME} packages

    # Verify Docker ecosystem
    ${SCRIPT_NAME} docker

    # Verify Helm chart
    ${SCRIPT_NAME} helm

    # Verify cloud storage
    ${SCRIPT_NAME} storage

    # Run all verifications
    ${SCRIPT_NAME} all

EOF
}


packages_command() {
    shift
    if [[ -f "$SCRIPT_DIR/neurondb-pkgs.sh" ]]; then
        log_info "Running package verification..."
        "$SCRIPT_DIR/neurondb-pkgs.sh" verify "$@"
    else
        log_error "Package verification script not found: neurondb-pkgs.sh"
        exit 1
    fi
}

docker_command() {
    shift
    if [[ -f "$SCRIPT_DIR/neurondb-docker.sh" ]]; then
        log_info "Running Docker ecosystem verification..."
        "$SCRIPT_DIR/neurondb-docker.sh" verify "$@"
    else
        log_error "Docker verification script not found: neurondb-docker.sh"
        exit 1
    fi
}

helm_command() {
    shift
    if [[ -f "$SCRIPT_DIR/neurondb-helm.sh" ]]; then
        log_info "Running Helm chart validation..."
        "$SCRIPT_DIR/neurondb-helm.sh" validate "$@"
    else
        log_error "Helm verification script not found: neurondb-helm.sh"
        exit 1
    fi
}

storage_command() {
    shift
    if [[ -f "$SCRIPT_DIR/neurondb-storage.sh" ]]; then
        log_info "Running cloud storage validation..."
        "$SCRIPT_DIR/neurondb-storage.sh" validate "$@"
    else
        log_error "Storage verification script not found: neurondb-storage.sh"
        exit 1
    fi
}

all_command() {
    shift
    print_header
    
    local failed=0
    
    log_info "Running all verifications..."
    echo ""
    
    # Packages
    if [[ -f "$SCRIPT_DIR/neurondb-pkgs.sh" ]]; then
        log_info "=== Verifying Packages ==="
        if ! "$SCRIPT_DIR/neurondb-pkgs.sh" verify "$@" 2>&1; then
            ((failed++))
            log_warning "Package verification failed"
        fi
        echo ""
    else
        log_warning "Package verification script not found"
    fi
    
    # Docker
    if [[ -f "$SCRIPT_DIR/neurondb-docker.sh" ]]; then
        log_info "=== Verifying Docker Ecosystem ==="
        if ! "$SCRIPT_DIR/neurondb-docker.sh" verify "$@" 2>&1; then
            ((failed++))
            log_warning "Docker verification failed"
        fi
        echo ""
    else
        log_warning "Docker verification script not found"
    fi
    
    # Helm
    if [[ -f "$SCRIPT_DIR/neurondb-helm.sh" ]]; then
        log_info "=== Verifying Helm Chart ==="
        if ! "$SCRIPT_DIR/neurondb-helm.sh" validate "$@" 2>&1; then
            ((failed++))
            log_warning "Helm verification failed"
        fi
        echo ""
    else
        log_warning "Helm verification script not found"
    fi
    
    # Storage
    if [[ -f "$SCRIPT_DIR/neurondb-storage.sh" ]]; then
        log_info "=== Verifying Cloud Storage ==="
        if ! "$SCRIPT_DIR/neurondb-storage.sh" validate "$@" 2>&1; then
            ((failed++))
            log_warning "Storage verification failed"
        fi
        echo ""
    else
        log_warning "Storage verification script not found"
    fi
    
    # Summary
    echo ""
    if [ "$failed" -eq 0 ]; then
        log_success "All verifications completed successfully"
        exit 0
    else
        log_error "$failed verification(s) failed"
        exit 1
    fi
}


parse_arguments() {
    # Check for help/version flags before command
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                show_help
                exit 0
                ;;
            -V|--version)
                echo "${SCRIPT_NAME} version 2.0.0"
                exit 0
                ;;
            *)
                break
                ;;
        esac
    done
    
    if [[ $# -eq 0 ]]; then
        show_help
        exit 0
    fi
    
    COMMAND="$1"
    shift
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -V|--version)
                echo "${SCRIPT_NAME} version 2.0.0"
                exit 0
                ;;
            *)
                break
                ;;
        esac
    done
}


main() {
    parse_arguments "$@"
    
    case "$COMMAND" in
        packages)
            packages_command "$@"
            ;;
        docker)
            docker_command "$@"
            ;;
        helm)
            helm_command "$@"
            ;;
        storage)
            storage_command "$@"
            ;;
        all)
            all_command "$@"
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

main "$@"






