#!/usr/bin/env bash
#-------------------------------------------------------------------------
#
# run_neuronmcp.sh
#    NeuronMCP Run Script
#
# Runs NeuronMCP server natively or in Docker mode.
# By default runs natively with full dependency checking and installation.
# Use --docker flag to run in Docker Compose mode.
#
# Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
#
# IDENTIFICATION
#    run_neuronmcp.sh
#
#-------------------------------------------------------------------------

set -euo pipefail

# Script metadata
SCRIPT_NAME=$(basename "$0")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
VERSION="3.0.0-devel"
VERBOSE=false
DOCKER_MODE=false

# Component directories
NEURONMCP_DIR="${PROJECT_ROOT}/NeuronMCP"
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

check_docker_service_dependency() {
    local service_name="$1"
    local compose_cmd="docker compose"
    if ! command -v docker compose &> /dev/null; then
        compose_cmd="docker-compose"
    fi
    
    # Check if neurondb service is running
    if [ "$service_name" != "neurondb" ]; then
        if ! $compose_cmd ps neurondb 2>/dev/null | grep -q "Up"; then
            print_warning "neurondb service is not running. Starting it..."
            $compose_cmd up -d neurondb || error_exit "Failed to start neurondb service"
            print_info "Waiting for neurondb to be healthy..."
            sleep 5
        fi
    fi
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
    
    check_docker_service_dependency "$service_name"
    
    if $compose_cmd up -d "$service_name"; then
        print_success "Service $service_name started"
        print_info "View logs with: $compose_cmd logs -f $service_name"
        print_info "Stop service with: $compose_cmd stop $service_name"
    else
        error_exit "Failed to start service $service_name"
    fi
}

run_docker_mode() {
    local service_name="neuronmcp"
    
    print_info "Running NeuronMCP in Docker mode"
    check_docker
    
    start_docker_service "$service_name"
    
    # Follow logs
    local compose_cmd="docker compose"
    if ! command -v docker compose &> /dev/null; then
        compose_cmd="docker-compose"
    fi
    
    print_info "Following logs (Ctrl+C to stop)..."
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
        # Basic version check (can be enhanced)
    fi
    
    return 0
}

check_go() {
    if ! check_dependency "Go" "go" "version" "1.23"; then
        error_exit "Go 1.23+ is required but not found. Please install Go."
    fi
    
    local go_version
    go_version=$(go version | awk '{print $3}' | sed 's/go//')
    print_success "Go version: $go_version"
}

check_python() {
    if ! check_dependency "Python" "python3" "--version" "3.8"; then
        print_warning "Python 3.8+ not found. Some features may not work."
        return 1
    fi
    
    local py_version
    py_version=$(python3 --version 2>&1)
    print_success "Python: $py_version"
    return 0
}

check_postgresql_connection() {
    local host="${NEURONDB_HOST:-localhost}"
    local port="${NEURONDB_PORT:-5432}"
    local database="${NEURONDB_DATABASE:-neurondb}"
    local user="${NEURONDB_USER:-pgedge}"
    local password="${NEURONDB_PASSWORD:-}"
    
    if [ -z "$password" ]; then
        print_warning "NEURONDB_PASSWORD not set. Connection check skipped."
        return 0
    fi
    
    print_debug "Checking PostgreSQL connection to $host:$port/$database"
    
    if command -v psql &> /dev/null; then
        if PGPASSWORD="$password" psql -h "$host" -p "$port" -U "$user" -d "$database" -c "SELECT 1;" &> /dev/null; then
            print_success "PostgreSQL connection successful"
            return 0
        else
            print_warning "PostgreSQL connection check failed. Continuing anyway..."
            return 1
        fi
    else
        print_warning "psql not found. Skipping connection check."
        return 0
    fi
}

#-------------------------------------------------------------------------------
# Dependency Installation Functions
#-------------------------------------------------------------------------------

install_go_deps() {
    local project_dir="$1"
    
    if [ ! -f "$project_dir/go.mod" ]; then
        print_warning "go.mod not found in $project_dir"
        return 0
    fi
    
    print_info "Installing Go dependencies..."
    cd "$project_dir"
    
    if go mod download && go mod tidy; then
        print_success "Go dependencies installed"
    else
        print_warning "Go dependencies installation had issues (continuing anyway)"
    fi
}

install_python_deps() {
    local requirements_file="$1"
    
    if [ ! -f "$requirements_file" ]; then
        print_debug "requirements.txt not found: $requirements_file"
        return 0
    fi
    
    if ! command -v python3 &> /dev/null; then
        print_debug "python3 not found, skipping Python dependencies"
        return 0
    fi
    
    print_info "Installing Python dependencies..."
    
    local pip_cmd=""
    if command -v pip3 &> /dev/null; then
        pip_cmd="pip3"
    elif python3 -m pip --version &> /dev/null 2>&1; then
        pip_cmd="python3 -m pip"
    else
        print_warning "pip not found, skipping Python dependencies"
        return 0
    fi
    
    if $pip_cmd install -r "$requirements_file" --quiet --disable-pip-version-check 2>/dev/null; then
        print_success "Python dependencies installed"
    elif $pip_cmd install --user -r "$requirements_file" --quiet --disable-pip-version-check 2>/dev/null; then
        print_success "Python dependencies installed (user install)"
    else
        print_warning "Python dependencies installation had issues (continuing anyway)"
    fi
}

#-------------------------------------------------------------------------------
# Build Functions
#-------------------------------------------------------------------------------

build_binary() {
    local project_dir="$1"
    local binary_name="$2"
    
    if [ ! -d "$project_dir" ]; then
        error_exit "Project directory not found: $project_dir"
    fi
    
    cd "$project_dir"
    
    print_info "Building binary: $binary_name"
    
    if [ -f "Makefile" ]; then
        if make build 2>/dev/null; then
            print_success "Binary built successfully"
            return 0
        else
            print_warning "Make build failed, trying direct go build"
        fi
    fi
    
    # Try direct go build if Makefile build failed
    if [ -d "cmd/neurondb-mcp" ]; then
        if go build -o "bin/$binary_name" "./cmd/neurondb-mcp" 2>/dev/null; then
            print_success "Binary built successfully"
            return 0
        fi
    fi
    
    print_warning "Could not build binary automatically"
    return 1
}

find_binary() {
    local project_dir="$1"
    local binary_name="$2"
    
    # Try multiple possible locations
    local possible_paths=(
        "$project_dir/bin/neurondb-mcp"
        "$project_dir/bin/neuronmcp"
        "$project_dir/bin/$binary_name"
        "$project_dir/neurondb-mcp"
        "$project_dir/neuronmcp"
    )
    
    for path in "${possible_paths[@]}"; do
        if [ -f "$path" ] && [ -x "$path" ]; then
            echo "$path"
            return 0
        fi
    done
    
    return 1
}

#-------------------------------------------------------------------------------
# Environment Setup
#-------------------------------------------------------------------------------

setup_environment() {
    # Set defaults for native mode
    export NEURONDB_HOST="${NEURONDB_HOST:-localhost}"
    export NEURONDB_PORT="${NEURONDB_PORT:-5432}"
    export NEURONDB_DATABASE="${NEURONDB_DATABASE:-neurondb}"
    export NEURONDB_USER="${NEURONDB_USER:-pgedge}"
    export NEURONDB_PASSWORD="${NEURONDB_PASSWORD:-}"
    
    print_debug "Environment:"
    print_debug "  NEURONDB_HOST=$NEURONDB_HOST"
    print_debug "  NEURONDB_PORT=$NEURONDB_PORT"
    print_debug "  NEURONDB_DATABASE=$NEURONDB_DATABASE"
    print_debug "  NEURONDB_USER=$NEURONDB_USER"
}

#-------------------------------------------------------------------------------
# Native Execution
#-------------------------------------------------------------------------------

run_native_mode() {
    print_info "Running NeuronMCP in native mode"
    
    # Check dependencies
    check_go
    check_python || true  # Python is optional
    
    # Install dependencies
    if [ -d "$NEURONMCP_DIR" ]; then
        install_go_deps "$NEURONMCP_DIR"
        
        if [ -f "$NEURONMCP_DIR/requirements.txt" ]; then
            install_python_deps "$NEURONMCP_DIR/requirements.txt"
        fi
    else
        error_exit "NeuronMCP directory not found: $NEURONMCP_DIR"
    fi
    
    # Find or build binary
    local binary_path
    if binary_path=$(find_binary "$NEURONMCP_DIR" "neurondb-mcp"); then
        print_success "Found binary: $binary_path"
    else
        print_info "Binary not found, attempting to build..."
        if build_binary "$NEURONMCP_DIR" "neurondb-mcp"; then
            if binary_path=$(find_binary "$NEURONMCP_DIR" "neurondb-mcp"); then
                print_success "Binary built: $binary_path"
            else
                error_exit "Binary was built but cannot be found"
            fi
        else
            error_exit "Binary not found and build failed. Please build manually: cd $NEURONMCP_DIR && make build"
        fi
    fi
    
    # Setup environment
    setup_environment
    
    # Check PostgreSQL connection (non-fatal)
    check_postgresql_connection || true
    
    # Display configuration
    echo ""
    print_info "=========================================="
    print_success "Starting NeuronMCP Server"
    print_info "=========================================="
    print_info "Binary: $binary_path"
    print_info "Database: ${NEURONDB_DATABASE}@${NEURONDB_HOST}:${NEURONDB_PORT}"
    print_info "User: ${NEURONDB_USER}"
    print_info "=========================================="
    echo ""
    
    # Execute binary
    cd "$NEURONMCP_DIR"
    exec "$binary_path"
}

#-------------------------------------------------------------------------------
# Help and Version
#-------------------------------------------------------------------------------

show_version() {
    echo "$SCRIPT_NAME version $VERSION"
}

show_help() {
    cat << EOF
$SCRIPT_NAME - NeuronMCP Run Script

Usage:
    $SCRIPT_NAME [OPTIONS]

Description:
    Runs NeuronMCP server natively or in Docker mode.
    By default, runs natively with full dependency checking and installation.
    Use --docker flag to run in Docker Compose mode.

Options:
    -h, --help          Show this help message
    -V, --version       Show version information
    -v, --verbose       Enable verbose output
    --docker            Run in Docker mode (default: native)

Environment Variables:
    NEURONDB_HOST          Database host (default: localhost for native, neurondb for Docker)
    NEURONDB_PORT          Database port (default: 5432)
    NEURONDB_DATABASE      Database name (default: neurondb)
    NEURONDB_USER          Database user (default: pgedge for native, neurondb for Docker)
    NEURONDB_PASSWORD      Database password (required)

Examples:
    $SCRIPT_NAME
    $SCRIPT_NAME --docker
    $SCRIPT_NAME --verbose
    NEURONDB_PASSWORD=secret $SCRIPT_NAME
    NEURONDB_PASSWORD=secret $SCRIPT_NAME --docker

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
