#!/bin/bash
#-------------------------------------------------------------------------
#
# neuronmcp-run-all-verifications.sh
#    Run All NeuronMCP Verification Scripts
#
# Runs all verification tests in sequence to validate NeuronMCP compatibility
# and integration with NeuronDB. Executes database connection, vector operations,
# schema validation, version compatibility, and tool execution verifications.
#
# Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
#
# IDENTIFICATION
#    NeuronMCP/tests/neuronmcp-run-all-verifications.sh
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
$SCRIPT_NAME - Run All NeuronMCP Verification Scripts

Usage:
    $SCRIPT_NAME [OPTIONS]

Description:
    Runs all verification tests in sequence to validate NeuronMCP
    compatibility and integration with NeuronDB.

Options:
    -h, --help      Show this help message
    -V, --version   Show version information
    -v, --verbose   Enable verbose output

Environment Variables:
    NEURONDB_HOST          Database host (default: localhost)
    NEURONDB_PORT          Database port (default: 5432)
    NEURONDB_DATABASE      Database name
    NEURONDB_USER          Database user

Examples:
    $SCRIPT_NAME
    NEURONDB_HOST=localhost NEURONDB_PORT=5432 $SCRIPT_NAME --verbose

EOF
}

main() {
    parse_args "$@"
    
    # Check if Go is available
    require_command go
    
    print_section "NeuronMCP Comprehensive Verification Suite"

    # Get the verification directory (test/verification relative to NeuronMCP root)
    VERIFICATION_DIR="$(cd "$(dirname "$0")/.." && pwd)/test/verification"
    
    # Test 1: Database Connection Verification
    print_section "Test 1: Database Connection Verification"
    log_info "Testing database connection handling and retry logic..."
    if go run -tags verification "$VERIFICATION_DIR/neuronmcp-verify-connection.go" 2>&1 | log_debug; then
        print_success "Database connection verification passed"
    else
        print_error "Database connection verification failed"
        exit_error
    fi
    
    # Test 2: Vector Operations Verification
    print_section "Test 2: Vector Operations Verification"
    log_info "Testing vector search operations with different distance metrics..."
    if go run -tags verification_vector_operations "$VERIFICATION_DIR/neuronmcp-verify-vector-operations.go" 2>&1 | log_debug; then
        print_success "Vector operations verification passed"
    else
        print_error "Vector operations verification failed"
        exit_error
    fi
    
    # Test 3: Schema Setup Validation
    print_section "Test 3: Schema Setup Validation"
    log_info "Validating NeuronMCP configuration schema setup..."
    if go run -tags verification_schema "$VERIFICATION_DIR/neuronmcp-verify-schema.go" 2>&1 | log_debug; then
        print_success "Schema setup validation passed"
    else
        print_warning "Schema setup validation failed"
        log_info "Note: Some failures may be expected if schema is not yet set up"
        log_info "Run: ./scripts/neuronmcp-setup.sh"
    fi
    
    # Test 4: Version Compatibility Verification
    print_section "Test 4: Version Compatibility Verification"
    log_info "Verifying PostgreSQL version compatibility (16, 17, 18)..."
    if go run -tags verification_version_compatibility "$VERIFICATION_DIR/neuronmcp-verify-version-compatibility.go" 2>&1 | log_debug; then
        print_success "Version compatibility verification passed"
    else
        print_error "Version compatibility verification failed"
        exit_error
    fi
    
    # Test 5: Tool Execution Flow Verification
    print_section "Test 5: Tool Execution Flow Verification"
    log_info "Testing tool execution flow from MCP client to NeuronDB..."
    if go run -tags verification_tool_execution "$VERIFICATION_DIR/neuronmcp-verify-tool-execution.go" 2>&1 | log_debug; then
        print_success "Tool execution flow verification passed"
    else
        print_error "Tool execution flow verification failed"
        exit_error
    fi
    
    # Summary
    print_section "Verification Summary"
    print_success "All verification tests completed!"
    log_info ""
    log_info "Next steps:"
    log_info "  1. Review any warnings or skipped tests above"
    log_info "  2. If schema validation failed, run: ./scripts/neuronmcp-setup.sh"
    log_info "  3. Set API keys: SELECT neurondb_set_model_key('model_name', 'api_key');"
    log_info "  4. View verification summary: cat VERIFICATION_SUMMARY.md"
    log_info ""
}

main "$@"



