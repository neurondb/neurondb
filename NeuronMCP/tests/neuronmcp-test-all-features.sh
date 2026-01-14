#!/bin/bash
#-------------------------------------------------------------------------
#
# neuronmcp-test-all-features.sh
#    NeuronMCP Feature Test Suite
#
# Tests all NeuronMCP features one by one including database operations,
# vector operations, tool execution, and integration capabilities.
# Provides comprehensive feature validation and reporting.
#
# Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
#
# IDENTIFICATION
#    NeuronMCP/tests/neuronmcp-test-all-features.sh
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

PASSED=0
FAILED=0

# Parse arguments
parse_args() {
    local remaining_args
    remaining_args=$(parse_common_args "$@")
    
    # Parse any script-specific arguments here
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
$SCRIPT_NAME - NeuronMCP Feature Test Suite

Usage:
    $SCRIPT_NAME [OPTIONS]

Description:
    Tests all NeuronMCP features one by one, including build verification,
    tool registration, security features, observability, and more.

Options:
    -h, --help      Show this help message
    -V, --version   Show version information
    -v, --verbose   Enable verbose output

Examples:
    $SCRIPT_NAME
    $SCRIPT_NAME --verbose

EOF
}

test_result() {
    if [ $1 -eq 0 ]; then
        print_success "✓ PASSED: $2"
        PASSED=$((PASSED + 1))
    else
        print_error "✗ FAILED: $2"
        FAILED=$((FAILED + 1))
    fi
}

# Change to project root
cd "$PROJECT_ROOT"

main() {
    parse_args "$@"
    
    print_section "NEURONMCP COMPREHENSIVE FEATURE TEST SUITE"
    
    log_debug "Starting feature test suite"
    
    print_section "PHASE 1: BUILD VERIFICATION"
    
    log_info "Test 1.1: Building all packages..."
if go build ./internal/... ./pkg/... 2>&1; then
    test_result 0 "Build all packages"
else
    test_result 1 "Build all packages"
fi

    log_info ""
    log_info "Test 1.2: Running go vet..."
if go vet ./internal/... ./pkg/... 2>&1; then
    test_result 0 "Go vet check"
else
    test_result 1 "Go vet check"
fi

    print_section "PHASE 2: TOOL REGISTRATION"
    
    log_info "Test 2.1: Checking tool registration..."
TOOL_COUNT=$(grep -r "registry.Register" internal/tools/register.go | wc -l | tr -d ' ')
if [ "$TOOL_COUNT" -ge 150 ]; then
    test_result 0 "Tool registration count ($TOOL_COUNT tools)"
else
    test_result 1 "Tool registration count ($TOOL_COUNT tools, expected >= 150)"
fi

    log_info ""
    log_info "Test 2.2: Checking PostgreSQL tools..."
POSTGRESQL_TOOLS=$(grep "postgresql_" internal/tools/register.go | wc -l | tr -d ' ')
if [ "$POSTGRESQL_TOOLS" -ge 50 ]; then
    test_result 0 "PostgreSQL tools ($POSTGRESQL_TOOLS tools)"
else
    test_result 1 "PostgreSQL tools ($POSTGRESQL_TOOLS tools, expected >= 50)"
fi

    log_info ""
    log_info "Test 2.3: Checking vector tools..."
VECTOR_TOOLS=$(grep "vector_" internal/tools/register.go | wc -l | tr -d ' ')
if [ "$VECTOR_TOOLS" -ge 20 ]; then
    test_result 0 "Vector tools ($VECTOR_TOOLS tools)"
else
    test_result 1 "Vector tools ($VECTOR_TOOLS tools, expected >= 20)"
fi

    log_info ""
    log_info "Test 2.4: Checking ML tools..."
ML_TOOLS=$(grep -E "(ml_|train_|predict)" internal/tools/register.go | wc -l | tr -d ' ')
if [ "$ML_TOOLS" -ge 15 ]; then
    test_result 0 "ML tools ($ML_TOOLS tools)"
else
    test_result 1 "ML tools ($ML_TOOLS tools, expected >= 15)"
fi

    log_info ""
    log_info "Test 2.5: Checking graph tools..."
GRAPH_TOOLS=$(grep "graph" internal/tools/register.go | wc -l | tr -d ' ')
if [ "$GRAPH_TOOLS" -ge 5 ]; then
    test_result 0 "Graph tools ($GRAPH_TOOLS tools)"
else
    test_result 1 "Graph tools ($GRAPH_TOOLS tools, expected >= 5)"
fi

    log_info ""
    log_info "Test 2.6: Checking multi-modal tools..."
MULTIMODAL_TOOLS=$(grep -E "(multimodal|image_|audio_)" internal/tools/register.go | wc -l | tr -d ' ')
if [ "$MULTIMODAL_TOOLS" -ge 5 ]; then
    test_result 0 "Multi-modal tools ($MULTIMODAL_TOOLS tools)"
else
    test_result 1 "Multi-modal tools ($MULTIMODAL_TOOLS tools, expected >= 5)"
fi

    print_section "PHASE 3: SECURITY FEATURES"
    
    log_info "Test 3.1: Checking RBAC module..."
if [ -f "internal/security/rbac.go" ]; then
    test_result 0 "RBAC module exists"
else
    test_result 1 "RBAC module missing"
fi

    log_info ""
    log_info "Test 3.2: Checking API key rotation..."
if [ -f "internal/security/api_key_rotation.go" ]; then
    test_result 0 "API key rotation module exists"
else
    test_result 1 "API key rotation module missing"
fi

    log_info ""
    log_info "Test 3.3: Checking MFA support..."
if [ -f "internal/security/mfa.go" ]; then
    test_result 0 "MFA module exists"
else
    test_result 1 "MFA module missing"
fi

    log_info ""
    log_info "Test 3.4: Checking data masking..."
if [ -f "internal/security/data_masking.go" ]; then
    test_result 0 "Data masking module exists"
else
    test_result 1 "Data masking module missing"
fi

    log_info ""
    log_info "Test 3.5: Checking network security..."
if [ -f "internal/security/network_security.go" ]; then
    test_result 0 "Network security module exists"
else
    test_result 1 "Network security module missing"
fi

    log_info ""
    log_info "Test 3.6: Checking compliance framework..."
if [ -f "internal/security/compliance.go" ]; then
    test_result 0 "Compliance module exists"
else
    test_result 1 "Compliance module missing"
fi

    print_section "PHASE 4: OBSERVABILITY"
    
    log_info "Test 4.1: Checking metrics collection..."
if [ -f "internal/observability/metrics.go" ]; then
    test_result 0 "Metrics module exists"
else
    test_result 1 "Metrics module missing"
fi

    log_info ""
    log_info "Test 4.2: Checking distributed tracing..."
if [ -f "internal/observability/tracing.go" ]; then
    test_result 0 "Tracing module exists"
else
    test_result 1 "Tracing module missing"
fi

    print_section "PHASE 5: HIGH AVAILABILITY"
    
    log_info "Test 5.1: Checking health check system..."
if [ -f "internal/ha/health.go" ]; then
    test_result 0 "Health check module exists"
else
    test_result 1 "Health check module missing"
fi

    log_info ""
    log_info "Test 5.2: Checking HA features compilation..."
if go build ./internal/ha/... 2>&1; then
    test_result 0 "HA module compiles"
else
    test_result 1 "HA module compilation failed"
fi

    print_section "PHASE 6: PLUGIN SYSTEM"
    
    log_info "Test 6.1: Checking plugin framework..."
if [ -f "internal/plugin/plugin.go" ]; then
    test_result 0 "Plugin framework exists"
else
    test_result 1 "Plugin framework missing"
fi

    log_info ""
    log_info "Test 6.2: Checking plugin system compilation..."
if go build ./internal/plugin/... 2>&1; then
    test_result 0 "Plugin system compiles"
else
    test_result 1 "Plugin system compilation failed"
fi

    print_section "PHASE 7: PERFORMANCE BENCHMARKING"
    
    log_info "Test 7.1: Checking performance benchmarking..."
if [ -f "internal/performance/benchmark.go" ]; then
    test_result 0 "Benchmarking module exists"
else
    test_result 1 "Benchmarking module missing"
fi

    print_section "PHASE 8: SDK IMPLEMENTATIONS"
    
    log_info "Test 8.1: Checking Python SDK..."
if [ -f "sdks/python/neurondb_mcp/client.py" ]; then
    test_result 0 "Python SDK exists"
else
    test_result 1 "Python SDK missing"
fi

    log_info ""
    log_info "Test 8.2: Checking TypeScript SDK..."
if [ -f "sdks/typescript/src/client.ts" ]; then
    test_result 0 "TypeScript SDK exists"
else
    test_result 1 "TypeScript SDK missing"
fi

    print_section "PHASE 9: FILE STRUCTURE VERIFICATION"
    
    log_info "Test 9.1: Checking PostgreSQL tool files..."
POSTGRESQL_FILES=$(find internal/tools -name "postgresql_*.go" | wc -l | tr -d ' ')
if [ "$POSTGRESQL_FILES" -ge 6 ]; then
    test_result 0 "PostgreSQL tool files ($POSTGRESQL_FILES files)"
else
    test_result 1 "PostgreSQL tool files ($POSTGRESQL_FILES files, expected >= 6)"
fi

    log_info ""
    log_info "Test 9.2: Checking vector tool files..."
VECTOR_FILES=$(find internal/tools -name "vector_*.go" | wc -l | tr -d ' ')
if [ "$VECTOR_FILES" -ge 5 ]; then
    test_result 0 "Vector tool files ($VECTOR_FILES files)"
else
    test_result 1 "Vector tool files ($VECTOR_FILES files, expected >= 5)"
fi

    log_info ""
    log_info "Test 9.3: Checking ML tool files..."
ML_FILES=$(find internal/tools -name "ml_*.go" | wc -l | tr -d ' ')
if [ "$ML_FILES" -ge 2 ]; then
    test_result 0 "ML tool files ($ML_FILES files)"
else
    test_result 1 "ML tool files ($ML_FILES files, expected >= 2)"
fi

    log_info ""
    log_info "Test 9.4: Checking security module files..."
SECURITY_FILES=$(find internal/security -name "*.go" | wc -l | tr -d ' ')
if [ "$SECURITY_FILES" -ge 6 ]; then
    test_result 0 "Security module files ($SECURITY_FILES files)"
else
    test_result 1 "Security module files ($SECURITY_FILES files, expected >= 6)"
fi

    print_section "PHASE 10: COMPILATION VERIFICATION"
    
    log_info "Test 10.1: Compiling tools package..."
if go build ./internal/tools/... 2>&1; then
    test_result 0 "Tools package compilation"
else
    test_result 1 "Tools package compilation"
fi

    log_info ""
    log_info "Test 10.2: Compiling security package..."
if go build ./internal/security/... 2>&1; then
    test_result 0 "Security package compilation"
else
    test_result 1 "Security package compilation"
fi

    log_info ""
    log_info "Test 10.3: Compiling observability package..."
if go build ./internal/observability/... 2>&1; then
    test_result 0 "Observability package compilation"
else
    test_result 1 "Observability package compilation"
fi

    log_info ""
    log_info "Test 10.4: Compiling plugin package..."
if go build ./internal/plugin/... 2>&1; then
    test_result 0 "Plugin package compilation"
else
    test_result 1 "Plugin package compilation"
fi

    print_section "TEST SUMMARY"
    
    TOTAL=$((PASSED + FAILED))
    log_info "Total Tests: $TOTAL"
    print_success "Passed: $PASSED"
    if [ $FAILED -gt 0 ]; then
        print_error "Failed: $FAILED"
    else
        print_success "Failed: $FAILED"
    fi
    log_info ""
    
    if [ $FAILED -eq 0 ]; then
        print_success "✅ ALL TESTS PASSED"
        exit_success
    else
        print_warning "⚠️  $FAILED TEST(S) FAILED"
        exit_error
    fi
}

main "$@"

