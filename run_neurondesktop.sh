#!/usr/bin/env bash
#-------------------------------------------------------------------------
#
# run_neurondesktop.sh
#    NeuronDesktop Run Script
#
# Runs NeuronDesktop (API and frontend) natively or in Docker mode.
# By default runs natively with full dependency checking and installation.
# Use --docker flag to run in Docker Compose mode.
#
# Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
#
# IDENTIFICATION
#    run_neurondesktop.sh
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
NEURONDESKTOP_DIR="${PROJECT_ROOT}/NeuronDesktop"
NEURONDESKTOP_API_DIR="${NEURONDESKTOP_DIR}/api"
NEURONDESKTOP_FRONTEND_DIR="${NEURONDESKTOP_DIR}/frontend"
DOCKER_COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.yml"

# Process management
BACKEND_PID=""
FRONTEND_PID=""

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
# Cleanup Function
#-------------------------------------------------------------------------------

cleanup() {
    if [ -n "${BACKEND_PID:-}" ]; then
        print_info "Stopping backend (PID: $BACKEND_PID)..."
        kill "$BACKEND_PID" 2>/dev/null || true
        wait "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "${FRONTEND_PID:-}" ]; then
        print_info "Stopping frontend (PID: $FRONTEND_PID)..."
        kill "$FRONTEND_PID" 2>/dev/null || true
        wait "$FRONTEND_PID" 2>/dev/null || true
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

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
    local compose_cmd="docker compose"
    if ! command -v docker compose &> /dev/null; then
        compose_cmd="docker-compose"
    fi
    
    # Check if required services are running
    local required_services=("neurondb" "neuronagent" "neuronmcp")
    for service in "${required_services[@]}"; do
        if ! $compose_cmd ps "$service" 2>/dev/null | grep -q "Up"; then
            print_warning "$service service is not running. Starting it..."
            $compose_cmd up -d "$service" || error_exit "Failed to start $service service"
            print_info "Waiting for $service to be healthy..."
            sleep 5
        fi
    done
}

start_docker_services() {
    local compose_cmd="docker compose"
    if ! command -v docker compose &> /dev/null; then
        compose_cmd="docker-compose"
    fi
    
    print_info "Starting Docker services: neurondesk-api, neurondesk-frontend"
    
    if [ ! -f "$DOCKER_COMPOSE_FILE" ]; then
        error_exit "docker-compose.yml not found at $DOCKER_COMPOSE_FILE"
    fi
    
    check_docker_service_dependency
    
    # Start API first
    if $compose_cmd up -d neurondesk-api; then
        print_success "Service neurondesk-api started"
    else
        error_exit "Failed to start service neurondesk-api"
    fi
    
    # Wait for API to be healthy
    print_info "Waiting for API to be healthy..."
    sleep 5
    
    # Start frontend
    if $compose_cmd up -d neurondesk-frontend; then
        print_success "Service neurondesk-frontend started"
    else
        error_exit "Failed to start service neurondesk-frontend"
    fi
    
    print_info "View API logs with: $compose_cmd logs -f neurondesk-api"
    print_info "View frontend logs with: $compose_cmd logs -f neurondesk-frontend"
    print_info "Stop services with: $compose_cmd stop neurondesk-api neurondesk-frontend"
}

run_docker_mode() {
    print_info "Running NeuronDesktop in Docker mode"
    check_docker
    
    start_docker_services
    
    # Follow logs
    local compose_cmd="docker compose"
    if ! command -v docker compose &> /dev/null; then
        compose_cmd="docker-compose"
    fi
    
    print_info "Following logs (Ctrl+C to stop)..."
    print_info "API: http://localhost:8081"
    print_info "Frontend: http://localhost:3000"
    $compose_cmd logs -f neurondesk-api neurondesk-frontend
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

check_go() {
    if ! check_dependency "Go" "go" "version" "1.24"; then
        error_exit "Go 1.24+ is required but not found. Please install Go."
    fi
    
    local go_version
    go_version=$(go version | awk '{print $3}' | sed 's/go//')
    print_success "Go version: $go_version"
}

check_node() {
    if ! check_dependency "Node.js" "node" "--version" "18"; then
        error_exit "Node.js 18+ is required but not found. Please install Node.js."
    fi
    
    local node_version
    node_version=$(node --version 2>&1)
    print_success "Node.js: $node_version"
}

check_npm() {
    if ! check_dependency "npm" "npm" "--version" ""; then
        error_exit "npm is required but not found. Please install npm."
    fi
    
    local npm_version
    npm_version=$(npm --version 2>&1)
    print_success "npm: $npm_version"
}

check_python() {
    if ! check_dependency "Python" "python3" "--version" "3.8"; then
        print_debug "Python 3.8+ not found (optional)"
        return 1
    fi
    
    local py_version
    py_version=$(python3 --version 2>&1)
    print_debug "Python: $py_version (optional)"
    return 0
}

check_postgresql_connection() {
    local host="${DB_HOST:-localhost}"
    local port="${DB_PORT:-5433}"
    local database="${DB_NAME:-neurondesk}"
    local user="${DB_USER:-neurondb}"
    local password="${DB_PASSWORD:-neurondb}"
    
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

install_npm_deps() {
    local project_dir="$1"
    
    if [ ! -f "$project_dir/package.json" ]; then
        print_warning "package.json not found in $project_dir"
        return 0
    fi
    
    print_info "Installing npm dependencies..."
    cd "$project_dir"
    
    if [ -d "node_modules" ]; then
        print_debug "node_modules already exists, skipping install"
        return 0
    fi
    
    if npm install --no-audit --no-fund; then
        print_success "npm dependencies installed"
    else
        print_warning "npm dependencies installation had issues (continuing anyway)"
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

build_backend_binary() {
    local project_dir="$1"
    local binary_name="$2"
    
    if [ ! -d "$project_dir" ]; then
        error_exit "Project directory not found: $project_dir"
    fi
    
    cd "$project_dir"
    
    print_info "Building backend binary: $binary_name"
    
    if [ -f "Makefile" ]; then
        if make build-api 2>/dev/null; then
            print_success "Backend binary built successfully"
            return 0
        else
            print_warning "Make build failed, trying direct go build"
        fi
    fi
    
    # Try direct go build
    if [ -f "api/cmd/server/main.go" ]; then
        mkdir -p bin
        if go build -o "bin/$binary_name" "./api/cmd/server/main.go" 2>/dev/null; then
            print_success "Backend binary built successfully"
            return 0
        fi
    fi
    
    print_warning "Could not build backend binary automatically"
    return 1
}

find_backend_binary() {
    local project_dir="$1"
    local binary_name="$2"
    
    # Try multiple possible locations
    local possible_paths=(
        "$project_dir/bin/neurondesktop"
        "$project_dir/bin/$binary_name"
        "$project_dir/api/server"
        "$project_dir/bin/server"
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
    export DB_HOST="${DB_HOST:-localhost}"
    export DB_PORT="${DB_PORT:-5433}"
    export DB_NAME="${DB_NAME:-neurondesk}"
    export DB_USER="${DB_USER:-neurondb}"
    export DB_PASSWORD="${DB_PASSWORD:-neurondb}"
    export SERVER_PORT="${SERVER_PORT:-8081}"
    
    print_debug "Environment:"
    print_debug "  DB_HOST=$DB_HOST"
    print_debug "  DB_PORT=$DB_PORT"
    print_debug "  DB_NAME=$DB_NAME"
    print_debug "  DB_USER=$DB_USER"
    print_debug "  SERVER_PORT=$SERVER_PORT"
}

#-------------------------------------------------------------------------------
# Native Execution
#-------------------------------------------------------------------------------

run_native_mode() {
    print_info "Running NeuronDesktop in native mode"
    
    # Check dependencies
    check_go
    check_node
    check_npm
    check_python || true  # Python is optional
    
    # Install dependencies
    if [ ! -d "$NEURONDESKTOP_DIR" ]; then
        error_exit "NeuronDesktop directory not found: $NEURONDESKTOP_DIR"
    fi
    
    # Backend dependencies
    if [ -d "$NEURONDESKTOP_API_DIR" ]; then
        install_go_deps "$NEURONDESKTOP_API_DIR"
    else
        error_exit "NeuronDesktop API directory not found: $NEURONDESKTOP_API_DIR"
    fi
    
    # Frontend dependencies
    if [ -d "$NEURONDESKTOP_FRONTEND_DIR" ]; then
        install_npm_deps "$NEURONDESKTOP_FRONTEND_DIR"
    else
        print_warning "NeuronDesktop frontend directory not found: $NEURONDESKTOP_FRONTEND_DIR"
    fi
    
    # Python dependencies (optional)
    if [ -f "$NEURONDESKTOP_DIR/requirements.txt" ]; then
        install_python_deps "$NEURONDESKTOP_DIR/requirements.txt"
    fi
    
    # Find or build backend binary
    local backend_binary_path
    if backend_binary_path=$(find_backend_binary "$NEURONDESKTOP_DIR" "neurondesktop"); then
        print_success "Found backend binary: $backend_binary_path"
    else
        print_info "Backend binary not found, attempting to build..."
        if build_backend_binary "$NEURONDESKTOP_DIR" "neurondesktop"; then
            if backend_binary_path=$(find_backend_binary "$NEURONDESKTOP_DIR" "neurondesktop"); then
                print_success "Backend binary built: $backend_binary_path"
            else
                print_warning "Binary was built but cannot be found, will use 'go run'"
                backend_binary_path=""
            fi
        else
            print_warning "Build failed, will use 'go run' as fallback"
            backend_binary_path=""
        fi
    fi
    
    # Setup environment
    setup_environment
    
    # Check PostgreSQL connection (non-fatal)
    check_postgresql_connection || true
    
    # Display configuration
    echo ""
    print_info "=========================================="
    print_success "Starting NeuronDesktop"
    print_info "=========================================="
    if [ -n "$backend_binary_path" ]; then
        print_info "Backend binary: $backend_binary_path"
    else
        print_info "Backend: go run api/cmd/server/main.go"
    fi
    print_info "Frontend: npm run dev"
    print_info "Database: ${DB_USER}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
    print_info "Backend API: http://localhost:${SERVER_PORT}"
    print_info "Frontend: http://localhost:3000"
    print_info "=========================================="
    echo ""
    
    # Start backend
    print_info "Starting backend server..."
    cd "$NEURONDESKTOP_DIR"
    if [ -n "$backend_binary_path" ]; then
        "$backend_binary_path" > /dev/null 2>&1 &
        BACKEND_PID=$!
    else
        cd api
        go run ./cmd/server > /dev/null 2>&1 &
        BACKEND_PID=$!
        cd ..
    fi
    print_success "Backend started (PID: $BACKEND_PID)"
    
    # Wait a bit for backend to start
    sleep 2
    
    # Start frontend
    if [ -d "$NEURONDESKTOP_FRONTEND_DIR" ] && [ -f "$NEURONDESKTOP_FRONTEND_DIR/package.json" ]; then
        print_info "Starting frontend server..."
        cd "$NEURONDESKTOP_FRONTEND_DIR"
        npm run dev > /dev/null 2>&1 &
        FRONTEND_PID=$!
        cd ..
        print_success "Frontend started (PID: $FRONTEND_PID)"
    else
        print_warning "Frontend directory not found, skipping frontend"
        FRONTEND_PID=""
    fi
    
    echo ""
    print_success "NeuronDesktop is running!"
    print_info "Backend API: http://localhost:${SERVER_PORT}"
    if [ -n "${FRONTEND_PID:-}" ]; then
        print_info "Frontend: http://localhost:3000"
    fi
    print_info "Press Ctrl+C to stop"
    echo ""
    
    # Wait for background processes
    if [ -n "${FRONTEND_PID:-}" ]; then
        wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || wait
    else
        wait "$BACKEND_PID" 2>/dev/null || wait
    fi
}

#-------------------------------------------------------------------------------
# Help and Version
#-------------------------------------------------------------------------------

show_version() {
    echo "$SCRIPT_NAME version $VERSION"
}

show_help() {
    cat << EOF
$SCRIPT_NAME - NeuronDesktop Run Script

Usage:
    $SCRIPT_NAME [OPTIONS]

Description:
    Runs NeuronDesktop (API and frontend) natively or in Docker mode.
    By default, runs natively with full dependency checking and installation.
    Use --docker flag to run in Docker Compose mode.

Options:
    -h, --help          Show this help message
    -V, --version       Show version information
    -v, --verbose       Enable verbose output
    --docker            Run in Docker mode (default: native)

Environment Variables:
    DB_HOST              Database host (default: localhost for native, neurondb for Docker)
    DB_PORT              Database port (default: 5433 for native, 5432 for Docker)
    DB_NAME              Database name (default: neurondesk)
    DB_USER              Database user (default: neurondb)
    DB_PASSWORD          Database password (default: neurondb)
    SERVER_PORT          Backend API port (default: 8081)

Examples:
    $SCRIPT_NAME
    $SCRIPT_NAME --docker
    $SCRIPT_NAME --verbose
    DB_PASSWORD=secret $SCRIPT_NAME
    DB_PASSWORD=secret $SCRIPT_NAME --docker

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
