#!/bin/bash

# run_stress_test.sh - NeuronDB Vector Stress Test with Crash Detection
# Usage: ./run_stress_test.sh [OPTIONS]
#
# Environment variables:
#   CORE_PATTERN    - Core dump pattern (default: ./cores/core.%e.%p.%t)
#   DB_NAME         - Database name (default: test_db)
#   PGDATA          - PostgreSQL data directory (auto-detected if not set)
#   PG_CTL          - Path to pg_ctl (auto-detected if not set)
#   PSQL            - Path to psql (default: psql)
#   PGUSER          - PostgreSQL user (default: $USER)
#   PGHOST          - PostgreSQL host (default: localhost)
#   PGPORT          - PostgreSQL port (default: 5432)
#   MAKE_JOBS       - Number of parallel make jobs (default: 12)
#   SKIP_BUILD      - Skip rebuild if set to 1 (default: 0)
#   NEURONDB_DIR    - NeuronDB source directory (default: ../../)

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_PATTERN="${CORE_PATTERN:-${SCRIPT_DIR}/cores/core.%e.%p.%t}"
DB_NAME="${DB_NAME:-test_db}"

# Use PATH for psql if not explicitly set (respect user's environment)
PSQL="${PSQL:-$(command -v psql 2>/dev/null || echo 'psql')}"

# Use environment variables for PostgreSQL connection (with defaults)
PGUSER="${PGUSER:-${USER:-postgres}}"
PGHOST="${PGHOST:-localhost}"
PGPORT="${PGPORT:-5432}"

# Build configuration
MAKE_JOBS="${MAKE_JOBS:-12}"
SKIP_BUILD="${SKIP_BUILD:-0}"
NEURONDB_DIR="${NEURONDB_DIR:-${SCRIPT_DIR}/../../}"
STRESS_SQL="${SCRIPT_DIR}/neurondb_vector_stress.sql"

# Initialize variables
PG_CTL="${PG_CTL:-}"
PGDATA="${PGDATA:-}"
CORE_DIR="$(dirname "${CORE_PATTERN}")"
CORE_BASENAME="$(basename "${CORE_PATTERN}")"
BUILD_DIR="${NEURONDB_DIR}"

# Track crash detection
CRASH_DETECTED=0
CORE_DUMP_FOUND=""
CRASH_TIME=""
PG_LOG=""

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

log_step() {
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $*${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════${NC}"
}

# Find pg_ctl executable
find_pg_ctl() {
    # Priority 1: User-specified environment variable
    if [ -n "${PG_CTL}" ] && [ -x "${PG_CTL}" ]; then
        return 0
    fi
    
    # Priority 2: Use PG_CONFIG to find bindir
    if [ -n "${PG_CONFIG:-}" ]; then
        local bindir="$(${PG_CONFIG} --bindir 2>/dev/null || true)"
        if [ -n "${bindir}" ] && [ -x "${bindir}/pg_ctl" ]; then
            PG_CTL="${bindir}/pg_ctl"
            return 0
        fi
    fi
    
    # Priority 3: Use PATH (respect user's environment)
    local which_pg_ctl="$(command -v pg_ctl 2>/dev/null || true)"
    if [ -n "${which_pg_ctl}" ] && [ -x "${which_pg_ctl}" ]; then
        PG_CTL="${which_pg_ctl}"
        return 0
    fi
    
    # Priority 4: Fall back to common locations if PATH doesn't have it
    # Try exact paths first (prioritize PostgreSQL 18-pge)
    for path in "/usr/local/pgsql.18-pge/bin/pg_ctl" "/usr/local/pgsql.18/bin/pg_ctl"; do
        if [ -x "${path}" ]; then
            PG_CTL="${path}"
            return 0
        fi
    done
    
    # Try glob patterns (prioritize pgsql.*-pge versions)
    shopt -s nullglob 2>/dev/null || true
    # First try pgsql.*-pge versions (custom builds)
    for match in /usr/local/pgsql.*-pge/bin/pg_ctl; do
        if [ -x "${match}" ]; then
            PG_CTL="${match}"
            shopt -u nullglob 2>/dev/null || true
            return 0
        fi
    done
    # Then try standard pgsql versions
    for base_dir in "/usr/lib/postgresql" "/usr/local/pgsql" "/opt/homebrew/opt/postgresql@" "/usr/pgsql-"; do
        for match in "${base_dir}"*/bin/pg_ctl; do
            if [ -x "${match}" ]; then
                PG_CTL="${match}"
                shopt -u nullglob 2>/dev/null || true
                return 0
            fi
        done
    done
    shopt -u nullglob 2>/dev/null || true
    
    return 1
}

# Find PostgreSQL data directory
find_pgdata() {
    # Priority 1: Use PGDATA environment variable (respect user's environment)
    if [ -n "${PGDATA}" ] && [ -d "${PGDATA}" ] && [ -f "${PGDATA}/postgresql.conf" ]; then
        return 0
    fi
    
    # Priority 2: Try to query PostgreSQL for data directory (requires working connection)
    if command -v "${PSQL}" >/dev/null 2>&1; then
        local data_dir="$(${PSQL} -h "${PGHOST}" -p "${PGPORT}" -U "${PGUSER}" -d postgres -t -A -c "SHOW data_directory;" -w 2>/dev/null | xargs || true)"
        if [ -n "${data_dir}" ] && [ -d "${data_dir}" ] && [ -f "${data_dir}/postgresql.conf" ]; then
            PGDATA="${data_dir}"
            return 0
        fi
    fi
    
    # Priority 3: Fall back to common locations if PGDATA not set
    for path in "${HOME}/data/pge18-data" "/usr/local/pgsql.18-pge/data" "/usr/local/pgsql.18/data" "${HOME}/pgdata" "${HOME}/postgres_data"; do
        if [ -d "${path}" ] && [ -f "${path}/postgresql.conf" ]; then
            PGDATA="${path}"
            return 0
        fi
    done
    
    # Try glob patterns
    shopt -s nullglob 2>/dev/null || true
    for base_dir in "/var/lib/postgresql" "/usr/local/pgsql" "/opt/homebrew/var/postgresql@" "${HOME}"; do
        if [ "${base_dir}" = "${HOME}" ]; then
            for match in "${HOME}"/neurondb_data*; do
                if [ -d "${match}" ] && [ -f "${match}/postgresql.conf" ]; then
                    PGDATA="${match}"
                    shopt -u nullglob 2>/dev/null || true
                    return 0
                fi
            done
        elif [ "${base_dir}" = "/var/lib/postgresql" ]; then
            for match in /var/lib/postgresql/*/main; do
                if [ -d "${match}" ] && [ -f "${match}/postgresql.conf" ]; then
                    PGDATA="${match}"
                    shopt -u nullglob 2>/dev/null || true
                    return 0
                fi
            done
        else
            for match in "${base_dir}"*/data; do
                if [ -d "${match}" ] && [ -f "${match}/postgresql.conf" ]; then
                    PGDATA="${match}"
                    shopt -u nullglob 2>/dev/null || true
                    return 0
                fi
            done
        fi
    done
    shopt -u nullglob 2>/dev/null || true
    
    
    return 1
}

# Wait for PostgreSQL to be ready
wait_for_postgres() {
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if ${PSQL} -h "${PGHOST}" -p "${PGPORT}" -U "${PGUSER}" -d postgres -c "SELECT 1;" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
    done
    
    return 1
}

# Setup core dumps
setup_core_dumps() {
    log_step "Setting up core dumps"
    
    # Set ulimit
    ulimit -c unlimited
    if [ $? -eq 0 ]; then
        log_success "Core dumps enabled (ulimit -c unlimited)"
    else
        log_warning "Failed to set ulimit -c unlimited (may need root or special permissions)"
    fi
    
    # Set kernel.core_pattern (requires root or special permissions)
    if [ -w /proc/sys/kernel/core_pattern ] 2>/dev/null; then
        echo "${CORE_PATTERN}" > /proc/sys/kernel/core_pattern
        log_success "Core pattern set to: ${CORE_PATTERN}"
    else
        local current_pattern="$(cat /proc/sys/kernel/core_pattern 2>/dev/null || echo 'unknown')"
        log_warning "Cannot set core pattern (requires root). Current: ${current_pattern}"
        log_warning "To set manually: sudo sysctl -w kernel.core_pattern='${CORE_PATTERN}'"
    fi
    
    # Create core dump directory
    mkdir -p "${CORE_DIR}"
    log_success "Core dump directory: ${CORE_DIR}"
    
    # Show current core limit
    local core_limit="$(ulimit -c)"
    log_info "Current core limit: ${core_limit}"
    
    echo ""
}

# Build and install NeuronDB
build_neurondb() {
    if [ "${SKIP_BUILD}" = "1" ]; then
        log_warning "Skipping build (SKIP_BUILD=1)"
        return 0
    fi
    
    log_step "Building and installing NeuronDB"
    
    if [ ! -d "${BUILD_DIR}" ]; then
        log_error "NeuronDB directory not found: ${BUILD_DIR}"
        return 1
    fi
    
    cd "${BUILD_DIR}"
    log_info "Building in: ${BUILD_DIR}"
    log_info "Using ${MAKE_JOBS} parallel jobs"
    
    if make -j"${MAKE_JOBS}" install; then
        log_success "Build and install completed"
        echo ""
        return 0
    else
        log_error "Build failed"
        echo ""
        return 1
    fi
}

# Restart PostgreSQL
restart_postgresql() {
    log_step "Restarting PostgreSQL"
    
    if ! find_pg_ctl; then
        log_error "Could not find pg_ctl executable"
        log_info "Set PG_CTL environment variable or install PostgreSQL"
        return 1
    fi
    
    log_info "Found pg_ctl: ${PG_CTL}"
    
    if ! find_pgdata; then
        log_error "Could not find PostgreSQL data directory"
        log_info "Set PGDATA environment variable"
        return 1
    fi
    
    log_info "Found PGDATA: ${PGDATA}"
    
    # Set log file path (use relative path pg.log in PGDATA, as per pg_ctl convention)
    PG_LOG="${PGDATA}/pg.log"
    # Ensure log file path is absolute
    if [[ ! "${PG_LOG}" = /* ]]; then
        PG_LOG="$(cd "${PGDATA}" && pwd)/pg.log"
    fi
    
    # Restart PostgreSQL using pg_ctl with log file in PGDATA
    log_info "Restarting PostgreSQL with pg_ctl..."
    log_info "Using log file: ${PG_LOG}"
    if ${PG_CTL} restart -D "${PGDATA}" -l "${PG_LOG}" -w -m fast; then
        log_success "PostgreSQL restarted"
        
        # Wait for PostgreSQL to be ready
        log_info "Waiting for PostgreSQL to be ready..."
        if wait_for_postgres; then
            log_success "PostgreSQL is ready"
            echo ""
            return 0
        else
            log_error "PostgreSQL did not become ready after restart"
            echo ""
            return 1
        fi
    else
        log_error "Failed to restart PostgreSQL"
        echo ""
        return 1
    fi
}

# Setup database and extension
setup_database() {
    log_step "Setting up database and extension"
    
    # Drop and create database
    # First check if database exists
    local db_exists=$(${PSQL} -h "${PGHOST}" -p "${PGPORT}" -U "${PGUSER}" -d postgres -t -A -c "SELECT 1 FROM pg_database WHERE datname = '${DB_NAME}';" -w 2>/dev/null | tr -d ' ' || echo "")
    
    if [ "${db_exists}" = "1" ]; then
        log_info "Database '${DB_NAME}' exists, terminating connections..."
        ${PSQL} -h "${PGHOST}" -p "${PGPORT}" -U "${PGUSER}" -d postgres -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '${DB_NAME}' AND pid <> pg_backend_pid();" -w >/dev/null 2>&1 || true
        sleep 1
        
        log_info "Dropping database '${DB_NAME}'..."
        ${PSQL} -h "${PGHOST}" -p "${PGPORT}" -U "${PGUSER}" -d postgres -c "DROP DATABASE ${DB_NAME};" -w >/dev/null 2>&1 || true
        sleep 1
    fi
    
    # Check again if database exists (after drop attempt)
    db_exists=$(${PSQL} -h "${PGHOST}" -p "${PGPORT}" -U "${PGUSER}" -d postgres -t -A -c "SELECT 1 FROM pg_database WHERE datname = '${DB_NAME}';" -w 2>/dev/null | tr -d ' ' || echo "")
    
    # Create database if it doesn't exist
    if [ "${db_exists}" != "1" ]; then
        log_info "Creating database '${DB_NAME}'..."
        if ${PSQL} -h "${PGHOST}" -p "${PGPORT}" -U "${PGUSER}" -d postgres -c "CREATE DATABASE ${DB_NAME};" -w 2>&1; then
            log_success "Database '${DB_NAME}' created"
        else
            log_error "Failed to create database '${DB_NAME}'"
            return 1
        fi
    else
        log_info "Database '${DB_NAME}' already exists, using existing database"
    fi
    
    # Drop and create extension
    log_info "Dropping extension 'neurondb' if exists..."
    ${PSQL} -h "${PGHOST}" -p "${PGPORT}" -U "${PGUSER}" -d "${DB_NAME}" -c "DROP EXTENSION IF EXISTS neurondb CASCADE;" -w >/dev/null 2>&1 || true
    
    log_info "Creating extension 'neurondb'..."
    if ${PSQL} -h "${PGHOST}" -p "${PGPORT}" -U "${PGUSER}" -d "${DB_NAME}" -c "CREATE EXTENSION neurondb CASCADE;" -w; then
        log_success "Extension 'neurondb' created"
        echo ""
        return 0
    else
        log_error "Failed to create extension 'neurondb'"
        echo ""
        return 1
    fi
}

# Check for core dumps
check_for_core_dumps() {
    local cores_found=0
    local cores=()
    
    # Check in core directory
    if [ -d "${CORE_DIR}" ]; then
        # Look for core files (matching pattern or generic core.*)
        while IFS= read -r -d '' core_file; do
            cores+=("${core_file}")
            cores_found=$((cores_found + 1))
        done < <(find "${CORE_DIR}" -maxdepth 1 -name "core.*" -type f -print0 2>/dev/null || true)
    fi
    
    # Also check in PGDATA directory
    if [ -n "${PGDATA}" ] && [ -d "${PGDATA}" ]; then
        while IFS= read -r -d '' core_file; do
            cores+=("${core_file}")
            cores_found=$((cores_found + 1))
        done < <(find "${PGDATA}" -maxdepth 1 -name "core.*" -type f -print0 2>/dev/null || true)
    fi
    
    # Check current directory as fallback
    while IFS= read -r -d '' core_file; do
        cores+=("${core_file}")
        cores_found=$((cores_found + 1))
    done < <(find . -maxdepth 1 -name "core.*" -type f -print0 2>/dev/null || true)
    
    if [ $cores_found -gt 0 ]; then
        CRASH_DETECTED=1
        CORE_DUMP_FOUND="${cores[0]}"
        CRASH_TIME="$(date -r "${CORE_DUMP_FOUND}" 2>/dev/null || stat -c %y "${CORE_DUMP_FOUND}" 2>/dev/null || echo 'unknown')"
        log_error "CRASH DETECTED: Found ${cores_found} core dump(s)"
        for core in "${cores[@]}"; do
            log_error "  Core dump: ${core}"
        done
        return 1
    fi
    
    return 0
}

# Parse PostgreSQL log for errors
parse_pg_log() {
    if [ ! -f "${PG_LOG}" ]; then
        log_warning "PostgreSQL log not found: ${PG_LOG}"
        return 1
    fi
    
    log_info "Checking PostgreSQL log for errors..."
    
    # Extract recent errors (last 100 lines with ERROR, FATAL, PANIC, or segfault)
    local errors="$(grep -i -E "(ERROR|FATAL|PANIC|segfault|signal|crash|abort|SIGSEGV|SIGABRT|core dumped)" "${PG_LOG}" | tail -50 || true)"
    
    if [ -n "${errors}" ]; then
        log_error "Recent errors from PostgreSQL log:"
        echo "${errors}" | while IFS= read -r line; do
            echo "  ${line}"
        done
        return 1
    else
        log_info "No obvious errors found in recent log entries"
        return 0
    fi
}

# Run stress test
run_stress_test() {
    log_step "Running stress test"
    
    if [ ! -f "${STRESS_SQL}" ]; then
        log_error "Stress test SQL file not found: ${STRESS_SQL}"
        return 1
    fi
    
    log_info "Executing: ${STRESS_SQL}"
    log_info "Database: ${DB_NAME}"
    log_info "This may take several minutes..."
    echo ""
    
    # Record start time
    local start_time="$(date +%s)"
    local core_count_before=0
    
    # Count existing cores before test
    if [ -d "${CORE_DIR}" ]; then
        core_count_before="$(find "${CORE_DIR}" -maxdepth 1 -name "core.*" -type f 2>/dev/null | wc -l)"
    fi
    
    # Run the stress test and capture output
    local test_output=""
    local test_exit_code=0
    
    if test_output="$(${PSQL} -h "${PGHOST}" -p "${PGPORT}" -U "${PGUSER}" -d "${DB_NAME}" -f "${STRESS_SQL}" -w 2>&1)"; then
        test_exit_code=0
    else
        test_exit_code=$?
    fi
    
    local end_time="$(date +%s)"
    local duration=$((end_time - start_time))
    
    echo ""
    
    # Check for core dumps
    if check_for_core_dumps; then
        if [ $test_exit_code -eq 0 ]; then
            log_success "Stress test completed successfully in ${duration} seconds"
            echo ""
            echo "${test_output}"
            return 0
        else
            log_warning "Stress test exited with code ${test_exit_code}"
            echo "${test_output}"
            return $test_exit_code
        fi
    else
        # Crash detected
        CRASH_DETECTED=1
        log_error "CRASH DETECTED during stress test!"
        log_error "Test duration: ${duration} seconds"
        log_error "Exit code: ${test_exit_code}"
        echo ""
        echo "Test output (may be incomplete):"
        echo "${test_output}"
        return 1
    fi
}

# Print crash investigation help
print_crash_help() {
    if [ $CRASH_DETECTED -eq 0 ]; then
        return 0
    fi
    
    log_step "Crash Investigation Help"
    
    echo ""
    log_info "Core dump information:"
    echo "  Location: ${CORE_DUMP_FOUND}"
    echo "  Crash time: ${CRASH_TIME}"
    echo "  Size: $(du -h "${CORE_DUMP_FOUND}" 2>/dev/null | cut -f1 || echo 'unknown')"
    echo ""
    
    # Find PostgreSQL binary
    local postgres_binary=""
    if [ -n "${PG_CTL}" ]; then
        local bindir="$(dirname "${PG_CTL}")"
        if [ -f "${bindir}/postgres" ]; then
            postgres_binary="${bindir}/postgres"
        fi
    fi
    
    if [ -z "${postgres_binary}" ]; then
        postgres_binary="$(command -v postgres 2>/dev/null || echo '/path/to/postgres')"
    fi
    
    log_info "Debugging commands:"
    echo ""
    echo "  # Load core dump in GDB:"
    echo "  gdb ${postgres_binary} ${CORE_DUMP_FOUND}"
    echo ""
    echo "  # Or use LLDB (on macOS):"
    echo "  lldb ${postgres_binary} -c ${CORE_DUMP_FOUND}"
    echo ""
    echo "  # Or use coredumpctl (systemd systems):"
    echo "  coredumpctl info ${CORE_DUMP_FOUND}"
    echo "  coredumpctl gdb ${CORE_DUMP_FOUND}"
    echo ""
    echo "  # Useful GDB commands after loading:"
    echo "  (gdb) bt              # Print backtrace"
    echo "  (gdb) thread apply all bt  # Backtrace for all threads"
    echo "  (gdb) info registers  # Show register values"
    echo "  (gdb) list            # Show source code around crash"
    echo ""
    
    log_info "PostgreSQL log:"
    echo "  ${PG_LOG}"
    echo ""
    
    # Parse log for errors
    parse_pg_log
    
    log_info "System information:"
    echo "  OS: $(uname -a)"
    echo "  PostgreSQL version: $(${PSQL} --version 2>/dev/null || echo 'unknown')"
    if [ -n "${PG_CTL}" ]; then
        echo "  pg_ctl: ${PG_CTL}"
    fi
    if [ -n "${PGDATA}" ]; then
        echo "  PGDATA: ${PGDATA}"
    fi
    echo ""
}

# Main execution
main() {
    echo "════════════════════════════════════════════════════════════════════"
    echo "  NeuronDB Vector Stress Test with Crash Detection"
    echo "════════════════════════════════════════════════════════════════════"
    echo ""
    echo "Configuration:"
    echo "  Database: ${DB_NAME}"
    echo "  Core pattern: ${CORE_PATTERN}"
    echo "  PostgreSQL: ${PGHOST}:${PGPORT} (user: ${PGUSER})"
    echo "  Build jobs: ${MAKE_JOBS}"
    echo "  Skip build: ${SKIP_BUILD}"
    echo ""
    
    # Setup core dumps
    setup_core_dumps
    
    # Build and install
    if ! build_neurondb; then
        log_error "Build failed. Aborting."
        exit 1
    fi
    
    # Restart PostgreSQL
    if ! restart_postgresql; then
        log_error "PostgreSQL restart failed. Aborting."
        exit 1
    fi
    
    # Setup database
    if ! setup_database; then
        log_error "Database setup failed. Aborting."
        exit 1
    fi
    
    # Run stress test
    if ! run_stress_test; then
        if [ $CRASH_DETECTED -eq 1 ]; then
            print_crash_help
            exit 1
        else
            log_error "Stress test failed (no crash detected)"
            exit 1
        fi
    fi
    
    # Final check for crashes (in case crash happened after test)
    if check_for_core_dumps; then
        log_success "No crashes detected"
        echo ""
        log_success "Stress test completed successfully!"
        exit 0
    else
        print_crash_help
        exit 1
    fi
}

# Run main function
main "$@"
