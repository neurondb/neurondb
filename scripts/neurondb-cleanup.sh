#!/bin/bash
#
# NeuronDB Cleanup Script
# Self-sufficient script for cleaning build artifacts, Docker resources, logs, and temporary files
#
# Usage:
#   ./neurondb-cleanup.sh [OPTIONS]
#
# Options:
#   --all                    Clean everything (Docker, logs, build artifacts, cache)
#   --docker                 Clean Docker containers, images, and volumes
#   --logs                   Clean log files
#   --build                  Clean build artifacts
#   --cache                  Clean cache directories
#   --dry-run                Show what would be cleaned without doing it
#   --help, -h               Show this help
#
# This script is completely self-sufficient with no external dependencies.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT_NAME=$(basename "$0")

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Options
CLEAN_ALL=false
CLEAN_DOCKER=false
CLEAN_LOGS=false
CLEAN_BUILD=false
CLEAN_CACHE=false
DRY_RUN=false

# Statistics
CLEANED_FILES=0
CLEANED_SIZE=0

log_info() {
    echo -e "${CYAN}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $*"
}

log_error() {
    echo -e "${RED}[✗]${NC} $*" >&2
}

print_header() {
    echo ""
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  ${BOLD}NeuronDB Ecosystem Cleanup${NC}                               ${BLUE}║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

show_help() {
    cat << EOF
${BOLD}NeuronDB Ecosystem Cleanup Script${NC}

${BOLD}Usage:${NC}
    $SCRIPT_NAME [OPTIONS]

${BOLD}Options:${NC}
    --all            Clean everything (Docker, logs, build artifacts, cache)
    --docker         Clean Docker containers, images, and volumes
    --logs           Clean log files
    --build          Clean build artifacts
    --cache          Clean cache directories
    --dry-run        Show what would be cleaned without doing it
    --help, -h       Show this help message

${BOLD}Examples:${NC}
    # Clean everything (dry run first)
    $SCRIPT_NAME --all --dry-run
    $SCRIPT_NAME --all

    # Clean only Docker resources
    $SCRIPT_NAME --docker

    # Clean logs and build artifacts
    $SCRIPT_NAME --logs --build

    # Clean cache
    $SCRIPT_NAME --cache

${BOLD}Warning:${NC}
    --docker will stop and remove all NeuronDB containers and volumes.
    This will result in data loss. Use with caution!

EOF
}

get_size() {
    local path="$1"
    if [ -e "$path" ]; then
        du -sh "$path" 2>/dev/null | cut -f1 || echo "0B"
    else
        echo "0B"
    fi
}

remove_item() {
    local item="$1"
    local description="$2"
    
    if [ ! -e "$item" ]; then
        return 0
    fi
    
    local size=$(get_size "$item")
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would remove: $description ($size)"
        return 0
    fi
    
    log_info "Removing: $description ($size)"
    rm -rf "$item"
    
    if [ $? -eq 0 ]; then
        log_success "Removed: $description"
        ((CLEANED_FILES++))
    else
        log_warning "Failed to remove: $description"
    fi
}

cleanup_docker() {
    log_info "Cleaning Docker resources..."
    echo ""
    
    if ! command -v docker &> /dev/null; then
        log_warning "Docker not found, skipping Docker cleanup"
        return 0
    fi
    
    cd "$PROJECT_ROOT"
    
    # Stop containers
    local containers=$(docker ps -a --format "{{.Names}}" | grep -E "neuron(db|agent|mcp|desktop)" || true)
    
    if [ -n "$containers" ]; then
        log_info "Found NeuronDB containers:"
        echo "$containers" | sed 's/^/  - /'
        echo ""
        
        if [ "$DRY_RUN" = false ]; then
            log_warning "Stopping and removing containers..."
            
            if [ -f "docker-compose.yml" ]; then
                docker compose down -v 2>/dev/null || docker-compose down -v 2>/dev/null || true
                log_success "Containers removed via Docker Compose"
            else
                echo "$containers" | while read container; do
                    docker stop "$container" 2>/dev/null || true
                    docker rm "$container" 2>/dev/null || true
                done
                log_success "Containers removed"
            fi
        else
            log_info "[DRY RUN] Would stop and remove containers"
        fi
    else
        log_info "No NeuronDB containers found"
    fi
    
    # Remove images
    local images=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep -E "neuron(db|agent|mcp|desktop)" || true)
    
    if [ -n "$images" ]; then
        echo ""
        log_info "Found NeuronDB images:"
        echo "$images" | sed 's/^/  - /'
        echo ""
        
        if [ "$DRY_RUN" = false ]; then
            log_info "Removing images..."
            echo "$images" | while read image; do
                docker rmi "$image" 2>/dev/null || true
            done
            log_success "Images removed"
        else
            log_info "[DRY RUN] Would remove images"
        fi
    else
        log_info "No NeuronDB images found"
    fi
    
    # Remove volumes
    local volumes=$(docker volume ls --format "{{.Name}}" | grep -E "neuron" || true)
    
    if [ -n "$volumes" ]; then
        echo ""
        log_info "Found NeuronDB volumes:"
        echo "$volumes" | sed 's/^/  - /'
        echo ""
        
        if [ "$DRY_RUN" = false ]; then
            log_warning "Removing volumes (DATA WILL BE LOST)..."
            echo "$volumes" | while read volume; do
                docker volume rm "$volume" 2>/dev/null || true
            done
            log_success "Volumes removed"
        else
            log_info "[DRY RUN] Would remove volumes"
        fi
    else
        log_info "No NeuronDB volumes found"
    fi
    
    echo ""
}

cleanup_logs() {
    log_info "Cleaning log files..."
    echo ""
    
    cd "$PROJECT_ROOT"
    
    # Find and remove log files
    local log_patterns=(
        "*.log"
        "*.log.*"
        "*.out"
        ".model_server.pid"
        "postgres_llm_server.log"
        "nohup.out"
    )
    
    for pattern in "${log_patterns[@]}"; do
        while IFS= read -r -d '' logfile; do
            remove_item "$logfile" "Log file: $(basename "$logfile")"
        done < <(find "$PROJECT_ROOT" -name "$pattern" -type f -print0 2>/dev/null)
    done
    
    # Component-specific logs
    local log_dirs=(
        "NeuronDB/logs"
    )
    
    for log_dir in "${log_dirs[@]}"; do
        if [ -d "$PROJECT_ROOT/$log_dir" ]; then
            remove_item "$PROJECT_ROOT/$log_dir" "Log directory: $log_dir"
        fi
    done
    
    # Temp logs
    if [ -d "/tmp" ]; then
        while IFS= read -r -d '' logfile; do
            remove_item "$logfile" "Temp log: $(basename "$logfile")"
        done < <(find /tmp -name "neurondb*.log" -type f -user "$(whoami)" -print0 2>/dev/null)
    fi
    
    echo ""
}

cleanup_build() {
    log_info "Cleaning build artifacts..."
    echo ""
    
    cd "$PROJECT_ROOT"
    
    # NeuronDB build artifacts
    if [ -d "NeuronDB" ]; then
        remove_item "NeuronDB/neurondb.so" "NeuronDB extension"
        remove_item "NeuronDB/neurondb.dylib" "NeuronDB extension (macOS)"
        remove_item "NeuronDB/*.o" "NeuronDB object files"
    fi
    
    # Go build artifacts (NeuronAgent/MCP/Desktop are in separate repos; keep for any local dev)
    for component in "NeuronAgent" "NeuronMCP" "NeuronDesktop"; do
        if [ -d "$component" ]; then
            remove_item "$component/bin" "$component binaries"
            remove_item "$component/$component" "$component binary"
        fi
    done
    
    # Frontend build artifacts (NeuronDesktop is in neuron-desktop repo)
    if [ -d "NeuronDesktop/frontend" ]; then
        remove_item "NeuronDesktop/frontend/.next" "Next.js build"
        remove_item "NeuronDesktop/frontend/out" "Next.js output"
        remove_item "NeuronDesktop/frontend/.turbo" "Turborepo cache"
    fi
    
    # Root bin directory
    remove_item "bin" "Root bin directory"
    
    # Python artifacts
    while IFS= read -r -d '' pycache; do
        remove_item "$pycache" "Python cache: $(basename "$(dirname "$pycache")")/__pycache__"
    done < <(find "$PROJECT_ROOT" -name "__pycache__" -type d -print0 2>/dev/null)
    
    while IFS= read -r -d '' pyc; do
        remove_item "$pyc" "Python compiled: $(basename "$pyc")"
    done < <(find "$PROJECT_ROOT" -name "*.pyc" -type f -print0 2>/dev/null)
    
    echo ""
}

cleanup_cache() {
    log_info "Cleaning cache directories..."
    echo ""
    
    cd "$PROJECT_ROOT"
    
    # Node modules (can be large)
    while IFS= read -r -d '' node_modules; do
        local dir_path=$(dirname "$node_modules")
        remove_item "$node_modules" "node_modules: $dir_path"
    done < <(find "$PROJECT_ROOT" -name "node_modules" -type d -print0 2>/dev/null)
    
    # Go module cache (in project)
    for component in "NeuronAgent" "NeuronMCP" "NeuronDesktop"; do
        if [ -d "$component" ]; then
            remove_item "$component/vendor" "$component vendor directory"
        fi
    done
    
    # Python virtual environments
    while IFS= read -r -d '' venv; do
        local dir_path=$(dirname "$venv")
        remove_item "$venv" "Python venv: $dir_path"
    done < <(find "$PROJECT_ROOT" -name "venv" -o -name ".venv" -o -name "env" -type d -print0 2>/dev/null)
    
    # Build caches
    remove_item "$PROJECT_ROOT/.cache" "Project cache"
    
    echo ""
}

parse_arguments() {
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --all)
                CLEAN_ALL=true
                CLEAN_DOCKER=true
                CLEAN_LOGS=true
                CLEAN_BUILD=true
                CLEAN_CACHE=true
                shift
                ;;
            --docker)
                CLEAN_DOCKER=true
                shift
                ;;
            --logs)
                CLEAN_LOGS=true
                shift
                ;;
            --build)
                CLEAN_BUILD=true
                shift
                ;;
            --cache)
                CLEAN_CACHE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

confirm_action() {
    if [ "$DRY_RUN" = true ]; then
        return 0
    fi
    
    echo -e "${YELLOW}Warning:${NC} This will perform the following cleanup actions:"
    echo ""
    
    if [ "$CLEAN_DOCKER" = true ]; then
        echo "  • Stop and remove Docker containers, images, and volumes"
        echo "    ${RED}(This will result in data loss!)${NC}"
    fi
    if [ "$CLEAN_LOGS" = true ]; then
        echo "  • Remove all log files"
    fi
    if [ "$CLEAN_BUILD" = true ]; then
        echo "  • Remove build artifacts and binaries"
    fi
    if [ "$CLEAN_CACHE" = true ]; then
        echo "  • Remove cache directories (node_modules, venv, etc.)"
    fi
    
    echo ""
    echo -n "Are you sure you want to proceed? Type 'yes' to confirm: "
    read -r confirmation
    
    if [ "$confirmation" != "yes" ]; then
        log_info "Cleanup cancelled"
        exit 0
    fi
}

main() {
    parse_arguments "$@"
    
    print_header
    
    if [ "$DRY_RUN" = true ]; then
        log_info "Running in DRY RUN mode (no changes will be made)"
        echo ""
    fi
    
    confirm_action
    
    echo ""
    
    if [ "$CLEAN_DOCKER" = true ]; then
        cleanup_docker
    fi
    
    if [ "$CLEAN_LOGS" = true ]; then
        cleanup_logs
    fi
    
    if [ "$CLEAN_BUILD" = true ]; then
        cleanup_build
    fi
    
    if [ "$CLEAN_CACHE" = true ]; then
        cleanup_cache
    fi
    
    # Summary
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║${NC}  ${BOLD}Cleanup Complete!${NC}                                          ${GREEN}║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    if [ "$DRY_RUN" = false ]; then
        echo -e "${BOLD}Summary:${NC}"
        echo "  Items cleaned: $CLEANED_FILES"
        echo ""
        
        if [ "$CLEAN_BUILD" = true ] || [ "$CLEAN_CACHE" = true ]; then
            log_info "You may need to rebuild components after cleanup:"
            echo "    make build              # Build all components"
            echo "    make docker-build       # Build Docker images"
        fi
    else
        log_info "This was a dry run. No changes were made."
        log_info "Run without --dry-run to perform actual cleanup."
    fi
    
    echo ""
}

main "$@"



