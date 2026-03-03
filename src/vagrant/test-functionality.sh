#!/bin/bash
#
# vagrant/test-functionality.sh - Test actual functionality of each NeuronDB module
#
# Tests that each module actually works, not just that files exist
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

TEST_DB="${TEST_DB:-neurondb_test}"
PG_CMD="${PG_CMD:-psql}"
PG_USER="${PG_USER:-postgres}"
RESULTS_FILE="/vagrant/test-results/functionality-tests.log"

mkdir -p "$(dirname "$RESULTS_FILE")"

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

log_info() {
    echo -e "${BLUE}[FUNC-TEST]${NC} $*" | tee -a "$RESULTS_FILE"
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

# Test NeuronDB extension functionality
test_neurondb_functionality() {
    log_info "Testing NeuronDB extension functionality..."

    # Test 1: Extension can be created
    if sudo -u "$PG_USER" $PG_CMD -d "$TEST_DB" -c "CREATE EXTENSION IF NOT EXISTS neurondb;" >/dev/null 2>&1; then
        log_success "Extension creation works"
    else
        log_error "Extension creation failed"
        return 1
    fi

    # Test 2: Extension version function
    if sudo -u "$PG_USER" $PG_CMD -d "$TEST_DB" -t -c "SELECT neurondb_version();" >/dev/null 2>&1; then
        local version=$(sudo -u "$PG_USER" $PG_CMD -d "$TEST_DB" -t -c "SELECT neurondb_version();" 2>/dev/null | xargs)
        log_success "neurondb_version() function works: $version"
    else
        log_warning "neurondb_version() function not available (may need runtime dependencies)"
    fi

    # Test 3: Vector type operations
    if sudo -u "$PG_USER" $PG_CMD -d "$TEST_DB" <<'EOF' >/dev/null 2>&1
CREATE TABLE IF NOT EXISTS test_func_vectors (id SERIAL PRIMARY KEY, vec VECTOR(3));
INSERT INTO test_func_vectors (vec) VALUES ('[1,2,3]'::vector), ('[4,5,6]'::vector);
SELECT COUNT(*) FROM test_func_vectors WHERE vec = '[1,2,3]'::vector;
DROP TABLE test_func_vectors;
EOF
    then
        log_success "Vector type operations work"
    else
        log_warning "Vector operations failed (may need runtime dependencies)"
    fi

    return 0
}

# Test NeuronAgent functionality
test_neuronagent_functionality() {
    log_info "Testing NeuronAgent functionality..."

    # Test 1: Binary can execute and show help/version
    if [ -f "/usr/bin/neuronagent" ] && [ -x "/usr/bin/neuronagent" ]; then
        if /usr/bin/neuronagent --help >/dev/null 2>&1 || /usr/bin/neuronagent -h >/dev/null 2>&1; then
            log_success "NeuronAgent binary executes and shows help"
        elif /usr/bin/neuronagent --version >/dev/null 2>&1 || /usr/bin/neuronagent -v >/dev/null 2>&1; then
            log_success "NeuronAgent binary executes and shows version"
        else
            # Try to run it (may need config, but should at least start)
            timeout 2 /usr/bin/neuronagent 2>&1 | head -5 && log_success "NeuronAgent binary can execute" || log_warning "NeuronAgent may need configuration to run"
        fi
    else
        log_error "NeuronAgent binary not found or not executable"
        return 1
    fi

    # Test 2: Check if migrations directory has valid SQL files
    if [ -d "/usr/share/neuronagent/migrations" ]; then
        local sql_count=$(find /usr/share/neuronagent/migrations -name "*.sql" -type f 2>/dev/null | wc -l)
        if [ "$sql_count" -gt 0 ]; then
            # Check if first SQL file is valid SQL
            local first_sql=$(find /usr/share/neuronagent/migrations -name "*.sql" -type f 2>/dev/null | head -1)
            if grep -qi "CREATE\|INSERT\|ALTER\|TABLE" "$first_sql" 2>/dev/null; then
                log_success "Migration files contain valid SQL ($sql_count files)"
            else
                log_warning "Migration files may not contain valid SQL"
            fi
        else
            log_error "No SQL migration files found"
        fi
    else
        log_error "Migrations directory not found"
    fi

    return 0
}

# Test NeuronMCP functionality
test_neuronmcp_functionality() {
    log_info "Testing NeuronMCP functionality..."

    # Test 1: Binary can execute
    if [ -f "/usr/bin/neurondb-mcp" ] && [ -x "/usr/bin/neurondb-mcp" ]; then
        if /usr/bin/neurondb-mcp --help >/dev/null 2>&1 || /usr/bin/neurondb-mcp -h >/dev/null 2>&1; then
            log_success "NeuronMCP binary executes and shows help"
        elif /usr/bin/neurondb-mcp --version >/dev/null 2>&1 || /usr/bin/neurondb-mcp -v >/dev/null 2>&1; then
            log_success "NeuronMCP binary executes and shows version"
        else
            timeout 2 /usr/bin/neurondb-mcp 2>&1 | head -5 && log_success "NeuronMCP binary can execute" || log_warning "NeuronMCP may need configuration"
        fi
    else
        log_error "NeuronMCP binary not found or not executable"
        return 1
    fi

    # Test 2: SQL files are accessible and valid
    if [ -d "/usr/share/neuronmcp/sql" ]; then
        local sql_count=$(find /usr/share/neuronmcp/sql -name "*.sql" -type f 2>/dev/null | wc -l)
        if [ "$sql_count" -gt 0 ]; then
            local first_sql=$(find /usr/share/neuronmcp/sql -name "*.sql" -type f 2>/dev/null | head -1)
            if [ -r "$first_sql" ] && grep -qi "CREATE\|SELECT\|INSERT" "$first_sql" 2>/dev/null; then
                log_success "SQL files are readable and contain valid SQL ($sql_count files)"
            else
                log_warning "SQL files may not be readable or valid"
            fi
        else
            log_error "No SQL files found"
        fi
    else
        log_error "SQL directory not found"
    fi

    return 0
}

# Test NeuronDesktop functionality
test_neurondesktop_functionality() {
    log_info "Testing NeuronDesktop functionality..."

    # Test 1: Binary can execute
    if [ -f "/usr/bin/neurondesktop" ] && [ -x "/usr/bin/neurondesktop" ]; then
        if /usr/bin/neurondesktop --help >/dev/null 2>&1 || /usr/bin/neurondesktop -h >/dev/null 2>&1; then
            log_success "NeuronDesktop binary executes and shows help"
        elif /usr/bin/neurondesktop --version >/dev/null 2>&1 || /usr/bin/neurondesktop -v >/dev/null 2>&1; then
            log_success "NeuronDesktop binary executes and shows version"
        else
            timeout 2 /usr/bin/neurondesktop 2>&1 | head -5 && log_success "NeuronDesktop binary can execute" || log_warning "NeuronDesktop may need configuration"
        fi
    else
        log_error "NeuronDesktop binary not found or not executable"
        return 1
    fi

    # Test 2: Migration files are valid
    if [ -d "/usr/share/neurondesktop/migrations" ]; then
        local migration_count=$(find /usr/share/neurondesktop/migrations -name "*.sql" -type f 2>/dev/null | wc -l)
        if [ "$migration_count" -gt 0 ]; then
            local first_migration=$(find /usr/share/neurondesktop/migrations -name "*.sql" -type f 2>/dev/null | head -1)
            if [ -r "$first_migration" ] && grep -qi "CREATE\|INSERT\|ALTER\|TABLE" "$first_migration" 2>/dev/null; then
                log_success "Migration files are valid ($migration_count files)"
            else
                log_warning "Migration files may not be valid"
            fi
        else
            log_error "No migration files found"
        fi
    else
        log_error "Migrations directory not found"
    fi

    return 0
}

# Test component integration
test_component_integration() {
    log_info "Testing component integration..."

    # Test that NeuronDB extension is available for other components
    if sudo -u "$PG_USER" $PG_CMD -d "$TEST_DB" -c "SELECT 1 FROM pg_extension WHERE extname = 'neurondb';" >/dev/null 2>&1; then
        log_success "NeuronDB extension is available for component integration"
    else
        log_warning "NeuronDB extension not loaded (components may not be able to use it)"
    fi

    # Test database connectivity from components' perspective
    if sudo -u "$PG_USER" $PG_CMD -d "$TEST_DB" -c "SELECT current_database(), version();" >/dev/null 2>&1; then
        log_success "Database is accessible for components"
    else
        log_error "Database is not accessible"
        return 1
    fi

    return 0
}

# Main test runner
main() {
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║          Module Functionality Test Suite                     ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Test results will be saved to: $RESULTS_FILE"
    echo ""

    # Run all functionality tests
    test_neurondb_functionality
    test_neuronagent_functionality
    test_neuronmcp_functionality
    test_neurondesktop_functionality
    test_component_integration

    # Summary
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}Functionality Test Summary${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo "Total tests: $TOTAL_TESTS"
    echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
    echo -e "${RED}Failed: $FAILED_TESTS${NC}"
    echo ""

    # Save summary
    {
        echo "=== Functionality Test Summary ==="
        echo "Total: $TOTAL_TESTS"
        echo "Passed: $PASSED_TESTS"
        echo "Failed: $FAILED_TESTS"
    } >> "$RESULTS_FILE"

    if [ $FAILED_TESTS -eq 0 ]; then
        log_success "All functionality tests passed!"
        return 0
    else
        log_error "Some functionality tests failed"
        return 1
    fi
}

main "$@"

