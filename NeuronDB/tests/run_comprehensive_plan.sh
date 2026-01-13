#!/bin/bash
# Comprehensive NeuronDB Test Plan Execution Script
# Implements the comprehensive test plan phase by phase

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Configuration
export PGHOST="${PGHOST:-localhost}"
export PGPORT="${PGPORT:-5432}"
export PGUSER="${PGUSER:-postgres}"
export PGDATABASE="${PGDATABASE:-neurondb_test}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Test counters
TOTAL_PHASES=18
CURRENT_PHASE=1
TESTS_PASSED=0
TESTS_FAILED=0
PHASES_COMPLETED=0

# Logging
LOG_DIR="$PROJECT_ROOT/tests/comprehensive_plan_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/comprehensive_test_${TIMESTAMP}.log"

log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

log_section() {
    log ""
    log "${CYAN}================================================================"
    log "$1"
    log "================================================================${NC}"
    log ""
}

log_info() {
    log "${BLUE}ℹ️  $1${NC}"
}

log_pass() {
    log "${GREEN}✓ PASS${NC}: $1"
    ((TESTS_PASSED++))
}

log_fail() {
    log "${RED}✗ FAIL${NC}: $1"
    ((TESTS_FAILED++))
}

log_warn() {
    log "${YELLOW}⚠️  WARN${NC}: $1"
}

# Check prerequisites
check_prerequisites() {
    log_section "Checking Prerequisites"
    
    # Check PostgreSQL
    if command -v psql >/dev/null 2>&1; then
        log_pass "psql found"
    else
        log_fail "psql not found"
        exit 1
    fi
    
    # Check database connection
    if psql -d "$PGDATABASE" -c "SELECT 1;" >/dev/null 2>&1; then
        log_pass "Database connection successful"
    else
        log_fail "Database connection failed"
        log_info "Create test database: createdb $PGDATABASE"
        exit 1
    fi
    
    # Check extension
    if psql -d "$PGDATABASE" -t -c "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'neurondb');" | grep -q t; then
        log_pass "NeuronDB extension installed"
    else
        log_warn "NeuronDB extension not installed"
        log_info "Installing extension..."
        psql -d "$PGDATABASE" -c "CREATE EXTENSION IF NOT EXISTS neurondb;" || log_fail "Failed to install extension"
    fi
    
    # Check make
    if command -v make >/dev/null 2>&1; then
        log_pass "make found"
    else
        log_fail "make not found"
        exit 1
    fi
    
    log ""
}

# Phase 1: Foundation & Core Types
phase_1() {
    log_section "Phase 1: Foundation & Core Types"
    
    # 1.1 Extension Installation & Setup
    log_info "1.1 Extension Installation & Setup"
    
    # Test extension creation
    if psql -d "$PGDATABASE" -c "CREATE EXTENSION IF NOT EXISTS neurondb;" >/dev/null 2>&1; then
        log_pass "Extension creation works"
    else
        log_fail "Extension creation failed"
    fi
    
    # Verify extension version
    VERSION=$(psql -d "$PGDATABASE" -t -c "SELECT extversion FROM pg_extension WHERE extname = 'neurondb';" | xargs)
    if [ -n "$VERSION" ]; then
        log_pass "Extension version: $VERSION"
    else
        log_fail "Extension version not found"
    fi
    
    # Verify schema exists
    if psql -d "$PGDATABASE" -t -c "SELECT EXISTS(SELECT 1 FROM pg_namespace WHERE nspname = 'neurondb');" | grep -q t; then
        log_pass "NeuronDB schema exists"
    else
        log_fail "NeuronDB schema not found"
    fi
    
    # 1.2 Core Vector Types
    log_info "1.2 Core Vector Types"
    
    # Test vector type
    if psql -d "$PGDATABASE" -c "SELECT '[1,2,3]'::vector(3);" >/dev/null 2>&1; then
        log_pass "vector type creation works"
    else
        log_fail "vector type creation failed"
    fi
    
    # Test vector dimensions
    DIMS=$(psql -d "$PGDATABASE" -t -c "SELECT vector_dims('[1,2,3,4,5]'::vector);" | xargs)
    if [ "$DIMS" = "5" ]; then
        log_pass "vector_dims works correctly"
    else
        log_fail "vector_dims returned: $DIMS (expected: 5)"
    fi
    
    # 1.3 Vector Operations
    log_info "1.3 Vector Operations"
    
    # Test vector arithmetic
    if psql -d "$PGDATABASE" -c "SELECT '[1,2,3]'::vector + '[4,5,6]'::vector;" >/dev/null 2>&1; then
        log_pass "Vector addition works"
    else
        log_fail "Vector addition failed"
    fi
    
    # Test vector norm
    NORM=$(psql -d "$PGDATABASE" -t -c "SELECT vector_norm('[3,4]'::vector);" | xargs)
    if [ -n "$NORM" ]; then
        log_pass "vector_norm works: $NORM"
    else
        log_fail "vector_norm failed"
    fi
    
    ((PHASES_COMPLETED++))
    log_info "Phase 1 completed: $TESTS_PASSED passed, $TESTS_FAILED failed"
}

# Phase 2: Distance Metrics & Indexes
phase_2() {
    log_section "Phase 2: Distance Metrics & Indexes"
    
    # Create test table
    psql -d "$PGDATABASE" -c "DROP TABLE IF EXISTS test_vectors; CREATE TABLE test_vectors (id SERIAL PRIMARY KEY, vec vector(3));" >/dev/null 2>&1
    psql -d "$PGDATABASE" -c "INSERT INTO test_vectors (vec) VALUES ('[1,2,3]'::vector), ('[4,5,6]'::vector), ('[7,8,9]'::vector);" >/dev/null 2>&1
    
    # 2.1 Distance Metrics
    log_info "2.1 Distance Metrics"
    
    # Test L2 distance
    if psql -d "$PGDATABASE" -c "SELECT vector_l2_distance('[1,2,3]'::vector, '[4,5,6]'::vector);" >/dev/null 2>&1; then
        log_pass "L2 distance works"
    else
        log_fail "L2 distance failed"
    fi
    
    # Test cosine distance
    if psql -d "$PGDATABASE" -c "SELECT vector_cosine_distance('[1,2,3]'::vector, '[4,5,6]'::vector);" >/dev/null 2>&1; then
        log_pass "Cosine distance works"
    else
        log_fail "Cosine distance failed"
    fi
    
    # 2.2 HNSW Indexes
    log_info "2.2 HNSW Indexes"
    
    # Create HNSW index
    if psql -d "$PGDATABASE" -c "CREATE INDEX ON test_vectors USING hnsw (vec vector_l2_ops);" >/dev/null 2>&1; then
        log_pass "HNSW index creation works"
    else
        log_fail "HNSW index creation failed"
    fi
    
    # Test HNSW search
    if psql -d "$PGDATABASE" -c "SELECT id FROM test_vectors ORDER BY vec <=> '[1,2,3]'::vector LIMIT 1;" >/dev/null 2>&1; then
        log_pass "HNSW search works"
    else
        log_fail "HNSW search failed"
    fi
    
    # 2.3 IVF Indexes
    log_info "2.3 IVF Indexes"
    
    # Create IVF index (if supported)
    psql -d "$PGDATABASE" -c "DROP INDEX IF EXISTS test_vectors_vec_idx;" >/dev/null 2>&1
    if psql -d "$PGDATABASE" -c "CREATE INDEX ON test_vectors USING ivfflat (vec vector_l2_ops) WITH (lists = 10);" >/dev/null 2>&1; then
        log_pass "IVF index creation works"
    else
        log_warn "IVF index creation not supported or failed (may be optional)"
    fi
    
    ((PHASES_COMPLETED++))
    log_info "Phase 2 completed: $TESTS_PASSED passed, $TESTS_FAILED failed"
}

# Main execution
main() {
    log_section "Comprehensive NeuronDB Test Plan Execution"
    log "Started: $(date)"
    log "Database: $PGDATABASE"
    log "Log file: $LOG_FILE"
    log ""
    
    check_prerequisites
    
    # Execute phases
    phase_1
    phase_2
    
    # Summary
    log_section "Test Execution Summary"
    log "Phases completed: $PHASES_COMPLETED / $TOTAL_PHASES"
    log "Tests passed: $TESTS_PASSED"
    log "Tests failed: $TESTS_FAILED"
    log "Ended: $(date)"
    
    if [ $TESTS_FAILED -eq 0 ]; then
        log "${GREEN}All tests passed!${NC}"
        exit 0
    else
        log "${RED}Some tests failed. Check log: $LOG_FILE${NC}"
        exit 1
    fi
}

# Run main
main "$@"



