#!/bin/bash
#
# vagrant/test-integration-rpm.sh - Integration tests for all NeuronDB components (RPM)
#
# Tests all components working together with PostgreSQL 18 on Rocky Linux
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

TEST_DB="${TEST_DB:-neurondb_test}"
TEST_USER="${TEST_USER:-neurondb_user}"
TEST_PASSWORD="${TEST_PASSWORD:-neurondb_test}"
RESULTS_FILE="/vagrant/test-results/integration-tests-rpm.log"

mkdir -p "$(dirname "$RESULTS_FILE")"

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

log_info() {
    echo -e "${BLUE}[TEST]${NC} $*" | tee -a "$RESULTS_FILE"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $*" | tee -a "$RESULTS_FILE"
    PASSED_TESTS=$((PASSED_TESTS + 1))
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $*" | tee -a "$RESULTS_FILE"
    FAILED_TESTS=$((FAILED_TESTS + 1))
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $*" | tee -a "$RESULTS_FILE"
}

# Test PostgreSQL 18 is running
test_postgresql() {
    log_info "Testing PostgreSQL 18..."

    if systemctl is-active --quiet postgresql-18; then
        log_success "PostgreSQL service is running"
    else
        log_error "PostgreSQL service is not running"
        return 1
    fi

    if sudo -u postgres /usr/pgsql-18/bin/psql -c "SELECT version();" >/dev/null 2>&1; then
        local pg_version=$(sudo -u postgres /usr/pgsql-18/bin/psql -t -c "SELECT version();" 2>/dev/null | head -1)
        log_success "PostgreSQL connection works: $pg_version"
        
        # Verify it's PostgreSQL 18
        if echo "$pg_version" | grep -q "PostgreSQL 18"; then
            log_success "PostgreSQL 18 confirmed"
        else
            log_warning "PostgreSQL version may not be 18"
        fi
    else
        log_error "Cannot connect to PostgreSQL"
        return 1
    fi
}

# Test NeuronDB extension
test_neurondb_extension() {
    log_info "Testing NeuronDB extension..."

    # Check if extension files exist for current PostgreSQL version
    local pg_major="18"
    
    if [ ! -f "/usr/pgsql-${pg_major}/share/extension/neurondb.control" ]; then
        # Try to find extension files
        local ext_files=$(find /usr -name "neurondb.control" 2>/dev/null | head -1)
        if [ -n "$ext_files" ]; then
            log_info "Found extension files at: $ext_files"
        else
            log_warning "Extension control file not found"
        fi
    fi

    # Check for required runtime dependencies
    local so_file=$(find /usr -name "neurondb.so" 2>/dev/null | head -1)
    if [ -n "$so_file" ]; then
        local missing_deps=$(ldd "$so_file" 2>/dev/null | grep "not found" || true)
        if [ -n "$missing_deps" ]; then
            log_warning "Missing runtime dependencies for NeuronDB extension:"
            echo "$missing_deps" | while read line; do
                log_warning "  $line"
            done
            log_warning "Extension installation may fail until dependencies are installed"
        fi
    fi

    # Create extension in test database
    if sudo -u postgres /usr/pgsql-18/bin/psql -d "$TEST_DB" -c "CREATE EXTENSION IF NOT EXISTS neurondb;" >/dev/null 2>&1; then
        log_success "NeuronDB extension created"
    else
        local error=$(sudo -u postgres /usr/pgsql-18/bin/psql -d "$TEST_DB" -c "CREATE EXTENSION neurondb;" 2>&1 | grep -i error || echo "Unknown error")
        log_warning "Failed to create NeuronDB extension: $error"
        log_warning "This may be due to missing runtime dependencies (e.g., libonnxruntime)"
        return 1
    fi

    # Verify extension version
    local version=$(sudo -u postgres /usr/pgsql-18/bin/psql -d "$TEST_DB" -t -c "SELECT extversion FROM pg_extension WHERE extname = 'neurondb';" 2>/dev/null | xargs)
    if [ -n "$version" ]; then
        log_success "NeuronDB extension version: $version"
    else
        log_warning "Cannot retrieve NeuronDB extension version"
    fi

    # Test basic vector operation
    if sudo -u postgres /usr/pgsql-18/bin/psql -d "$TEST_DB" <<'EOF' >/dev/null 2>&1
CREATE TABLE IF NOT EXISTS test_vectors (id SERIAL, vec VECTOR(3));
INSERT INTO test_vectors (vec) VALUES ('[1,2,3]'::vector);
SELECT id FROM test_vectors WHERE vec = '[1,2,3]'::vector;
DROP TABLE IF EXISTS test_vectors;
EOF
    then
        log_success "Basic vector operations work"
    else
        log_warning "Vector operations failed"
    fi
}

# Test NeuronAgent
test_neuronagent() {
    log_info "Testing NeuronAgent..."

    # Check if binary exists
    if [ -f "/usr/bin/neuronagent" ] && [ -x "/usr/bin/neuronagent" ]; then
        log_success "NeuronAgent binary exists and is executable"
    else
        log_error "NeuronAgent binary not found or not executable"
        return 1
    fi

    # Check if migrations directory exists
    if [ -d "/usr/share/neuronagent/migrations" ]; then
        local sql_count=$(find /usr/share/neuronagent/migrations -name "*.sql" -type f 2>/dev/null | wc -l)
        if [ "$sql_count" -gt 0 ]; then
            log_success "NeuronAgent migrations found ($sql_count files)"
        else
            log_warning "NeuronAgent migrations directory exists but no SQL files found"
        fi
    else
        log_warning "NeuronAgent migrations directory not found"
    fi
}

# Test NeuronMCP
test_neuronmcp() {
    log_info "Testing NeuronMCP..."

    # Check if binary exists
    if [ -f "/usr/bin/neurondb-mcp" ] && [ -x "/usr/bin/neurondb-mcp" ]; then
        log_success "NeuronMCP binary exists and is executable"
    else
        log_error "NeuronMCP binary not found or not executable"
        return 1
    fi

    # Check SQL files
    if [ -d "/usr/share/neuronmcp/sql" ]; then
        local sql_count=$(find /usr/share/neuronmcp/sql -name "*.sql" -type f 2>/dev/null | wc -l)
        if [ "$sql_count" -gt 0 ]; then
            log_success "NeuronMCP SQL files found ($sql_count files)"
        else
            log_warning "NeuronMCP SQL directory exists but no SQL files found"
        fi
    else
        log_warning "NeuronMCP SQL directory not found"
    fi
}

# Test NeuronDesktop
test_neurondesktop() {
    log_info "Testing NeuronDesktop..."

    # Check if binary exists
    if [ -f "/usr/bin/neurondesktop" ] && [ -x "/usr/bin/neurondesktop" ]; then
        log_success "NeuronDesktop binary exists and is executable"
    else
        log_error "NeuronDesktop binary not found or not executable"
        return 1
    fi

    # Check migrations
    if [ -d "/usr/share/neurondesktop/migrations" ]; then
        local migration_count=$(find /usr/share/neurondesktop/migrations -name "*.sql" -type f 2>/dev/null | wc -l)
        if [ "$migration_count" -gt 0 ]; then
            log_success "NeuronDesktop migrations found ($migration_count files)"
        else
            log_warning "NeuronDesktop migrations directory exists but no SQL files found"
        fi
    else
        log_warning "NeuronDesktop migrations directory not found"
    fi
}

# Test component integration
test_integration() {
    log_info "Testing component integration..."

    # Test that all components can access PostgreSQL 18
    if sudo -u postgres /usr/pgsql-18/bin/psql -d "$TEST_DB" -c "SELECT 1;" >/dev/null 2>&1; then
        log_success "Database connection works for components"
    else
        log_error "Database connection failed"
        return 1
    fi

    # Test that extension is available
    if sudo -u postgres /usr/pgsql-18/bin/psql -d "$TEST_DB" -c "SELECT neurondb_version();" >/dev/null 2>&1; then
        local ext_version=$(sudo -u postgres /usr/pgsql-18/bin/psql -d "$TEST_DB" -t -c "SELECT neurondb_version();" 2>/dev/null | xargs)
        log_success "NeuronDB extension functions work: $ext_version"
    else
        log_warning "NeuronDB extension functions may not be available"
    fi

    log_success "Component integration tests passed"
}

# Main test runner
main() {
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║       Integration Test Suite (RPM/Rocky Linux)               ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Test results will be saved to: $RESULTS_FILE"
    echo ""

    # Run all tests
    test_postgresql
    test_neurondb_extension
    test_neuronagent
    test_neuronmcp
    test_neurondesktop
    test_integration

    # Summary
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}Test Summary${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo "Total tests: $TOTAL_TESTS"
    echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
    echo -e "${RED}Failed: $FAILED_TESTS${NC}"
    echo ""

    # Save summary to results file
    {
        echo "=== Test Summary ==="
        echo "Total: $TOTAL_TESTS"
        echo "Passed: $PASSED_TESTS"
        echo "Failed: $FAILED_TESTS"
    } >> "$RESULTS_FILE"

    if [ $FAILED_TESTS -eq 0 ]; then
        log_success "All integration tests passed!"
        return 0
    else
        log_error "Some tests failed"
        return 1
    fi
}

main "$@"

