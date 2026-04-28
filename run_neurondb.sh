#!/usr/bin/env bash
#-------------------------------------------------------------------------
#
# run_neurondb.sh
#    NeuronDB Run Script
#
# Runs NeuronDB (PostgreSQL with NeuronDB extension) natively or in Docker mode.
# By default runs natively with full dependency checking and setup.
# Use --docker flag to run in Docker Compose mode.
#
# Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
#
# IDENTIFICATION
#    run_neurondb.sh
#
#-------------------------------------------------------------------------

set -euo pipefail

# Script metadata
SCRIPT_NAME=$(basename "$0")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
VERSION="3.1.0"
VERBOSE=false
DOCKER_MODE=false

# Component directories
NEURONDB_DIR="${PROJECT_ROOT}/NeuronDB"
DOCKER_COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.yml"

# Color codes (only if terminal supports it)
if [[ -t 1 ]] && [[ "${TERM:-}" != "dumb" ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    NC=''
fi

#-------------------------------------------------------------------------------
# Utility Functions
#-------------------------------------------------------------------------------

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

error_exit() {
    print_error "$1"
    exit 1
}

error_usage() {
    print_error "$1"
    echo ""
    show_help
    exit 2
}

#-------------------------------------------------------------------------------
# Docker Functions
#-------------------------------------------------------------------------------

check_docker() {
    if ! command -v docker &> /dev/null; then
        error_exit "Docker is not installed. Please install Docker to use --docker mode."
    fi
    
    if ! command -v docker compose &> /dev/null && ! command -v docker-compose &> /dev/null; then
        error_exit "Docker Compose is not installed. Please install Docker Compose to use --docker mode."
    fi
    
    if ! docker info &> /dev/null; then
        error_exit "Docker daemon is not running. Please start Docker to use --docker mode."
    fi
    
    print_success "Docker and Docker Compose are available"
}

start_docker_service() {
    local service_name="$1"
    local compose_cmd="docker compose"
    if ! command -v docker compose &> /dev/null; then
        compose_cmd="docker-compose"
    fi
    
    print_info "Starting Docker service: $service_name"
    
    if [ ! -f "$DOCKER_COMPOSE_FILE" ]; then
        error_exit "docker-compose.yml not found at $DOCKER_COMPOSE_FILE"
    fi
    
    if $compose_cmd up -d "$service_name"; then
        print_success "Service $service_name started"
        print_info "View logs with: $compose_cmd logs -f $service_name"
        print_info "Stop service with: $compose_cmd stop $service_name"
        print_info "Connection: postgresql://${POSTGRES_USER:-neurondb}:${POSTGRES_PASSWORD:-neurondb}@localhost:${POSTGRES_PORT:-5433}/${POSTGRES_DB:-neurondb}"
    else
        error_exit "Failed to start service $service_name"
    fi
}

run_docker_mode() {
    local service_name="neurondb"
    
    print_info "Running NeuronDB in Docker mode"
    check_docker
    
    start_docker_service "$service_name"
    
    # Follow logs
    local compose_cmd="docker compose"
    if ! command -v docker compose &> /dev/null; then
        compose_cmd="docker-compose"
    fi
    
    print_info "Following logs (Ctrl+C to stop)..."
    print_info "Waiting for PostgreSQL to be ready..."
    sleep 5
    
    # Wait for PostgreSQL to be ready
    local max_attempts=30
    local attempt=0
    local pg_user="${POSTGRES_USER:-neurondb}"
    local pg_db="${POSTGRES_DB:-neurondb}"
    local pg_password="${POSTGRES_PASSWORD:-neurondb}"
    local pg_port="${POSTGRES_PORT:-5433}"
    
    while [ $attempt -lt $max_attempts ]; do
        if PGPASSWORD="$pg_password" psql -h localhost -p "$pg_port" -U "$pg_user" -d "$pg_db" -c "SELECT 1;" &> /dev/null; then
            print_success "PostgreSQL is ready"
            break
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    
    if [ $attempt -eq $max_attempts ]; then
        print_warning "PostgreSQL may not be ready yet, but continuing..."
    fi
    
    # Ensure extension exists
    print_info "Ensuring NeuronDB extension exists..."
    if PGPASSWORD="$pg_password" psql -h localhost -p "$pg_port" -U "$pg_user" -d "$pg_db" -c "CREATE EXTENSION IF NOT EXISTS neurondb;" &> /dev/null; then
        print_success "NeuronDB extension ready"
    else
        print_warning "Could not create extension automatically (may already exist)"
    fi
    
    $compose_cmd logs -f "$service_name"
}

#-------------------------------------------------------------------------------
# Dependency Checking Functions
#-------------------------------------------------------------------------------

check_dependency() {
    local name="$1"
    local command="$2"
    local version_flag="${3:---version}"
    local min_version="${4:-}"
    
    if ! command -v "$command" &> /dev/null; then
        return 1
    fi
    
    if [ -n "$min_version" ]; then
        local version
        version=$($command $version_flag 2>&1 | head -n1 || echo "")
        print_debug "Found $name version: $version"
    fi
    
    return 0
}

check_postgresql() {
    if ! check_dependency "PostgreSQL" "psql" "--version" "16"; then
        error_exit "PostgreSQL client (psql) is required but not found. Please install PostgreSQL 16+."
    fi
    
    local pg_version
    pg_version=$(psql --version 2>&1 | head -n1)
    print_success "PostgreSQL client: $pg_version"
    
    # Check if PostgreSQL server is running
    if ! command -v pg_isready &> /dev/null; then
        print_warning "pg_isready not found, cannot check if PostgreSQL server is running"
        return 0
    fi
    
    local pg_host="${POSTGRES_HOST:-localhost}"
    local pg_port="${POSTGRES_PORT:-5432}"
    local pg_user="${POSTGRES_USER:-postgres}"
    
    if pg_isready -h "$pg_host" -p "$pg_port" -U "$pg_user" &> /dev/null; then
        print_success "PostgreSQL server is running on $pg_host:$pg_port"
        return 0
    else
        print_warning "PostgreSQL server may not be running on $pg_host:$pg_port"
        print_info "You may need to start PostgreSQL manually:"
        print_info "  Linux: sudo systemctl start postgresql"
        print_info "  macOS: brew services start postgresql@17"
        return 1
    fi
}

check_neurondb_extension() {
    local pg_host="${POSTGRES_HOST:-localhost}"
    local pg_port="${POSTGRES_PORT:-5432}"
    local pg_user="${POSTGRES_USER:-postgres}"
    local pg_db="${POSTGRES_DB:-postgres}"
    local pg_password="${POSTGRES_PASSWORD:-}"
    
    if [ -z "$pg_password" ]; then
        print_debug "POSTGRES_PASSWORD not set, skipping extension check"
        return 0
    fi
    
    print_debug "Checking if NeuronDB extension is installed..."
    
    if PGPASSWORD="$pg_password" psql -h "$pg_host" -p "$pg_port" -U "$pg_user" -d "$pg_db" -tAc "SELECT 1 FROM pg_extension WHERE extname = 'neurondb';" 2>/dev/null | grep -q 1; then
        print_success "NeuronDB extension is installed"
        return 0
    else
        print_warning "NeuronDB extension is not installed in database $pg_db"
        print_info "You can install it with:"
        print_info "  psql -h $pg_host -p $pg_port -U $pg_user -d $pg_db -c 'CREATE EXTENSION neurondb;'"
        return 1
    fi
}

#-------------------------------------------------------------------------------
# Native Execution
#-------------------------------------------------------------------------------

run_native_mode() {
    print_info "Running NeuronDB in native mode"
    
    # Check dependencies
    check_postgresql || print_warning "PostgreSQL server check failed, but continuing..."
    
    # Setup environment
    export POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
    export POSTGRES_PORT="${POSTGRES_PORT:-5432}"
    export POSTGRES_USER="${POSTGRES_USER:-postgres}"
    export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-}"
    export POSTGRES_DB="${POSTGRES_DB:-postgres}"
    
    print_debug "Environment:"
    print_debug "  POSTGRES_HOST=$POSTGRES_HOST"
    print_debug "  POSTGRES_PORT=$POSTGRES_PORT"
    print_debug "  POSTGRES_USER=$POSTGRES_USER"
    print_debug "  POSTGRES_DB=$POSTGRES_DB"
    
    # Check if extension is installed (non-fatal)
    check_neurondb_extension || true
    
    # Display configuration
    echo ""
    print_info "=========================================="
    print_success "NeuronDB Configuration"
    print_info "=========================================="
    print_info "PostgreSQL: ${POSTGRES_USER}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"
    
    if [ -n "$POSTGRES_PASSWORD" ]; then
        print_info "Connection: postgresql://${POSTGRES_USER}:***@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"
    else
        print_info "Connection: postgresql://${POSTGRES_USER}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"
    fi
    
    print_info "=========================================="
    echo ""
    
    # Check if PostgreSQL is accessible
    if [ -n "$POSTGRES_PASSWORD" ]; then
        if PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT version();" &> /dev/null; then
            print_success "PostgreSQL connection successful"
            
            # Try to create extension if it doesn't exist
            if ! PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc "SELECT 1 FROM pg_extension WHERE extname = 'neurondb';" 2>/dev/null | grep -q 1; then
                print_info "Creating NeuronDB extension..."
                if PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "CREATE EXTENSION IF NOT EXISTS neurondb;" 2>/dev/null; then
                    print_success "NeuronDB extension created"
                else
                    print_warning "Could not create NeuronDB extension. It may need to be installed first."
                    print_info "To install NeuronDB extension, see: NeuronDB/docs/getting-started/installation.md"
                fi
            else
                print_success "NeuronDB extension is already installed"
            fi
            
            # Show extension version if available
            local ext_version
            ext_version=$(PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc "SELECT extversion FROM pg_extension WHERE extname = 'neurondb';" 2>/dev/null || echo "")
            if [ -n "$ext_version" ]; then
                print_success "NeuronDB extension version: $ext_version"
            fi
        else
            print_warning "PostgreSQL connection failed. Please check your configuration."
            print_info "Make sure PostgreSQL is running and credentials are correct."
        fi
    else
        print_warning "POSTGRES_PASSWORD not set. Cannot verify connection or extension."
        print_info "Set POSTGRES_PASSWORD to enable automatic extension setup."
    fi
    
    echo ""
    print_info "NeuronDB is ready!"
    print_info "You can now connect to PostgreSQL and use NeuronDB functions."
    print_info ""
    print_info "Example connection:"
    if [ -n "$POSTGRES_PASSWORD" ]; then
        print_info "  psql postgresql://${POSTGRES_USER}:***@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"
    else
        print_info "  psql -h ${POSTGRES_HOST} -p ${POSTGRES_PORT} -U ${POSTGRES_USER} -d ${POSTGRES_DB}"
    fi
    print_info ""
    print_info "Example query:"
    print_info "  SELECT neurondb.version();"
    echo ""
}

#-------------------------------------------------------------------------------
# Help and Version
#-------------------------------------------------------------------------------

show_version() {
    echo "$SCRIPT_NAME version $VERSION"
}

show_help() {
    cat << EOF
$SCRIPT_NAME - NeuronDB Run Script

Usage:
    $SCRIPT_NAME [OPTIONS]

Description:
    Runs NeuronDB (PostgreSQL with NeuronDB extension) natively or in Docker mode.
    By default, runs natively and verifies PostgreSQL connection and extension setup.
    Use --docker flag to run in Docker Compose mode.

Options:
    -h, --help          Show this help message
    -V, --version       Show version information
    -v, --verbose       Enable verbose output
    --docker            Run in Docker mode (default: native)

Environment Variables:
    POSTGRES_HOST          PostgreSQL host (default: localhost for native, neurondb for Docker)
    POSTGRES_PORT          PostgreSQL port (default: 5432 for native, 5433 for Docker)
    POSTGRES_USER          PostgreSQL user (default: postgres for native, neurondb for Docker)
    POSTGRES_PASSWORD      PostgreSQL password (required for extension setup)
    POSTGRES_DB            PostgreSQL database (default: postgres for native, neurondb for Docker)

Examples:
    $SCRIPT_NAME
    $SCRIPT_NAME --docker
    $SCRIPT_NAME --verbose
    POSTGRES_PASSWORD=secret $SCRIPT_NAME
    POSTGRES_PASSWORD=secret $SCRIPT_NAME --docker
    POSTGRES_HOST=localhost POSTGRES_PORT=5433 POSTGRES_USER=neurondb POSTGRES_PASSWORD=secret POSTGRES_DB=neurondb $SCRIPT_NAME

EOF
}

#-------------------------------------------------------------------------------
# Argument Parsing
#-------------------------------------------------------------------------------

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -V|--version)
                show_version
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            --docker)
                DOCKER_MODE=true
                shift
                ;;
            *)
                error_usage "Unknown option: $1"
                ;;
        esac
    done
}

#-------------------------------------------------------------------------------
# Main Entry Point
#-------------------------------------------------------------------------------

main() {
    parse_args "$@"
    
    if [ "$DOCKER_MODE" = "true" ]; then
        run_docker_mode
    else
        run_native_mode
    fi
}

main "$@"
