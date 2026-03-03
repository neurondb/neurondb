#!/bin/bash
#
# NeuronDB Docker Management Script
# Modular and clean script for managing NeuronDB ecosystem containers
#
# Usage:
#   ./docker.sh --version | -v                    # Show version
#   ./docker.sh --list                             # List available services
#   ./docker.sh --build [service] [--profile]      # Build containers
#   ./docker.sh --run [service] [--profile]        # Run containers
#   ./docker.sh --all --build [--profile]          # Build all services
#   ./docker.sh --all --run [--profile]            # Run all services
#

set -euo pipefail

# Script metadata
SCRIPT_VERSION="1.0.0"
SCRIPT_NAME="docker.sh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# Canonical compose file lives at repo root. We intentionally run that to avoid drift.
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.yml"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Available services
AVAILABLE_SERVICES=("neurondb" "neuronagent" "neuronmcp" "neurondesktop")
AVAILABLE_PROFILES=("cpu" "cuda" "rocm" "metal" "default")

# Default values
PROFILE="default"
PG_MAJOR="18"
ACTION=""
SERVICES=()
ALL_SERVICES=false
DELETE_RESOURCE=""

# ============================================================================
# Utility Functions
# ============================================================================

print_error() {
    echo -e "${RED}ERROR:${NC} $1" >&2
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# ============================================================================
# Validation Functions
# ============================================================================

validate_compose_file() {
    if [[ ! -f "${COMPOSE_FILE}" ]]; then
        print_error "Docker compose file not found: ${COMPOSE_FILE}"
        exit 1
    fi
}

validate_service() {
    local service=$1
    if [[ ! " ${AVAILABLE_SERVICES[@]} " =~ " ${service} " ]]; then
        print_error "Unknown service: ${service}"
        print_info "Available services: ${AVAILABLE_SERVICES[*]}"
        exit 1
    fi
}

validate_profile() {
    local profile=$1
    if [[ ! " ${AVAILABLE_PROFILES[@]} " =~ " ${profile} " ]]; then
        print_error "Unknown profile: ${profile}"
        print_info "Available profiles: ${AVAILABLE_PROFILES[*]}"
        exit 1
    fi
}

validate_postgres_version() {
    local version=$1
    if [[ ! "${version}" =~ ^[0-9]+$ ]] || [[ "${version}" -lt 12 ]] || [[ "${version}" -gt 20 ]]; then
        print_error "Invalid PostgreSQL version: ${version}"
        print_info "PostgreSQL version must be a number between 12 and 20"
        exit 1
    fi
}

# ============================================================================
# Core Functions
# ============================================================================

get_compose_cmd() {
    if command -v docker-compose &> /dev/null; then
        echo "docker-compose"
    elif docker compose version &> /dev/null 2>&1; then
        echo "docker compose"
    else
        print_error "Docker Compose is not available"
        exit 1
    fi
}

is_compose_v2() {
    local cmd=$(get_compose_cmd)
    if [[ "${cmd}" == "docker compose" ]]; then
        return 0
    else
        return 1
    fi
}

run_compose() {
    local compose_cmd=$(get_compose_cmd)
    cd "${PROJECT_ROOT}"
    
    # Export PG_MAJOR for docker compose to use
    export PG_MAJOR
    
    # Calculate version-specific ports (54XX where XX is PG_MAJOR)
    # e.g., PG 18 -> 5418, PG 17 -> 5417, PG 16 -> 5416
    export POSTGRES_PORT="${POSTGRES_PORT:-54${PG_MAJOR}}"
    export POSTGRES_CUDA_PORT="${POSTGRES_CUDA_PORT:-54${PG_MAJOR}}"
    export POSTGRES_ROCM_PORT="${POSTGRES_ROCM_PORT:-54${PG_MAJOR}}"
    export POSTGRES_METAL_PORT="${POSTGRES_METAL_PORT:-54${PG_MAJOR}}"
    
    # For docker-compose v1, profile flag comes before the subcommand
    # For docker compose v2, profile flag comes after the subcommand
    if is_compose_v2; then
        ${compose_cmd} -f "${COMPOSE_FILE}" "$@"
    else
        # docker-compose v1 syntax
        ${compose_cmd} -f "${COMPOSE_FILE}" "$@"
    fi
    
    local exit_code=$?
    cd - > /dev/null
    return ${exit_code}
}

show_version() {
    echo "NeuronDB Docker Management Script"
    echo "Version: ${SCRIPT_VERSION}"
    echo ""
    echo "Docker Compose file: ${COMPOSE_FILE}"
    if command -v docker-compose &> /dev/null || command -v docker &> /dev/null; then
        if command -v docker &> /dev/null; then
            echo "Docker version: $(docker --version)"
        fi
        if docker compose version &> /dev/null; then
            echo "Docker Compose version: $(docker compose version)"
        fi
    fi
}

list_services() {
    print_info "Available services:"
    for service in "${AVAILABLE_SERVICES[@]}"; do
        echo "  - ${service}"
    done
    echo ""
    print_info "Available profiles:"
    for profile in "${AVAILABLE_PROFILES[@]}"; do
        echo "  - ${profile}"
    done
}

list_docker_resources() {
    print_info "NeuronDB Docker Containers:"
    local containers=$(docker ps -a --filter "name=neurondb" --filter "name=neuronagent" --filter "name=neuronmcp" --filter "name=neurondesk" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null)
    if [[ -n "${containers}" ]]; then
        echo "${containers}"
    else
        echo "  (no containers found)"
    fi
    echo ""
    
    print_info "NeuronDB Docker Images:"
    local images=$(docker images --filter "reference=neurondb*" --filter "reference=neuronagent*" --filter "reference=neurondb-mcp*" --filter "reference=neurondesk*" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" 2>/dev/null)
    if [[ -n "${images}" ]]; then
        echo "${images}"
    else
        echo "  (no images found)"
    fi
    echo ""
    
    print_info "NeuronDB Docker Volumes:"
    local volumes=$(docker volume ls --format "table {{.Name}}\t{{.Driver}}" 2>/dev/null | grep -iE "(neurondb|neuronagent|neuronmcp|neurondesk|NAME)" || true)
    if [[ -n "${volumes}" ]]; then
        echo "${volumes}"
    else
        echo "  (no volumes found)"
    fi
    echo ""
    
    print_info "NeuronDB Docker Networks:"
    local networks=$(docker network ls --filter "name=neurondb" --format "table {{.Name}}\t{{.Driver}}\t{{.Scope}}" 2>/dev/null)
    if [[ -n "${networks}" ]]; then
        echo "${networks}"
    else
        echo "  (no networks found)"
    fi
}

delete_docker_resource() {
    local resource=$1
    local resource_type=""
    local resource_name=""
    
    # Parse resource type and name (format: type:name or just name for container)
    if [[ "${resource}" == *":"* ]]; then
        resource_type="${resource%%:*}"
        resource_name="${resource#*:}"
    else
        # Default to container if no type specified
        resource_type="container"
        resource_name="${resource}"
    fi
    
    case "${resource_type}" in
        container|ctr|c)
            if docker ps -a --format '{{.Names}}' | grep -q "^${resource_name}$"; then
                print_info "Stopping and removing container: ${resource_name}"
                docker stop "${resource_name}" 2>/dev/null || true
                docker rm "${resource_name}" 2>/dev/null || true
                if docker ps -a --format '{{.Names}}' | grep -q "^${resource_name}$"; then
                    print_error "Failed to remove container: ${resource_name}"
                    return 1
                else
                    print_success "Removed container: ${resource_name}"
                fi
            else
                print_warning "Container not found: ${resource_name}"
                return 1
            fi
            ;;
        image|img|i)
            if docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${resource_name}$" || docker images --format '{{.ID}}' | grep -q "^${resource_name}$"; then
                print_info "Removing image: ${resource_name}"
                docker rmi "${resource_name}" 2>/dev/null || true
                if docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${resource_name}$" && ! docker images --format '{{.ID}}' | grep -q "^${resource_name}$"; then
                    print_error "Failed to remove image: ${resource_name}"
                    return 1
                else
                    print_success "Removed image: ${resource_name}"
                fi
            else
                print_warning "Image not found: ${resource_name}"
                return 1
            fi
            ;;
        volume|vol|v)
            if docker volume ls --format '{{.Name}}' | grep -q "^${resource_name}$"; then
                print_info "Removing volume: ${resource_name}"
                docker volume rm "${resource_name}" 2>/dev/null || true
                if docker volume ls --format '{{.Name}}' | grep -q "^${resource_name}$"; then
                    print_error "Failed to remove volume: ${resource_name}"
                    return 1
                else
                    print_success "Removed volume: ${resource_name}"
                fi
            else
                print_warning "Volume not found: ${resource_name}"
                return 1
            fi
            ;;
        network|net|n)
            if docker network ls --format '{{.Name}}' | grep -q "^${resource_name}$"; then
                print_info "Removing network: ${resource_name}"
                docker network rm "${resource_name}" 2>/dev/null || true
                if docker network ls --format '{{.Name}}' | grep -q "^${resource_name}$"; then
                    print_error "Failed to remove network: ${resource_name}"
                    return 1
                else
                    print_success "Removed network: ${resource_name}"
                fi
            else
                print_warning "Network not found: ${resource_name}"
                return 1
            fi
            ;;
        *)
            print_error "Unknown resource type: ${resource_type}"
            print_info "Valid types: container, image, volume, network"
            print_info "Format: --delete=type:name or --delete=name (defaults to container)"
            return 1
            ;;
    esac
}

build_service() {
    local service=$1
    local profile=$2
    
    print_info "Building ${service} with profile: ${profile}"
    
    validate_service "${service}"
    validate_profile "${profile}"
    
    # Map service names to compose service names
    local compose_service=""
    case "${service}" in
        neurondb)
            case "${profile}" in
                cuda) compose_service="neurondb-cuda" ;;
                rocm) compose_service="neurondb-rocm" ;;
                metal) compose_service="neurondb-metal" ;;
                *) compose_service="neurondb" ;;
            esac
            ;;
        neuronagent)
            case "${profile}" in
                cuda) compose_service="neuronagent-cuda" ;;
                rocm) compose_service="neuronagent-rocm" ;;
                metal) compose_service="neuronagent-metal" ;;
                *) compose_service="neuronagent" ;;
            esac
            ;;
        neuronmcp)
            case "${profile}" in
                cuda) compose_service="neuronmcp-cuda" ;;
                rocm) compose_service="neuronmcp-rocm" ;;
                metal) compose_service="neuronmcp-metal" ;;
                *) compose_service="neuronmcp" ;;
            esac
            ;;
        neurondesktop)
            compose_service="neurondesk-api neurondesk-frontend"
            ;;
    esac
    
    if [[ -z "${compose_service}" ]]; then
        print_error "No compose service mapped for ${service} with profile ${profile}"
        return 1
    fi
    
    # Split multiple services if needed
    local services_array=(${compose_service})
    local compose_cmd=$(get_compose_cmd)
    
    if is_compose_v2; then
        # docker compose v2: --profile comes after subcommand
        cd "${PROJECT_ROOT}"
        ${compose_cmd} -f "${COMPOSE_FILE}" build --profile "${profile}" "${services_array[@]}"
        local exit_code=$?
        cd - > /dev/null
    else
        # docker-compose v1: --profile comes before subcommand
        cd "${PROJECT_ROOT}"
        ${compose_cmd} -f "${COMPOSE_FILE}" --profile "${profile}" build "${services_array[@]}"
        local exit_code=$?
        cd - > /dev/null
    fi
    
    if [[ ${exit_code} -eq 0 ]]; then
        print_success "Built ${service} (${profile})"
    else
        print_error "Failed to build ${service} (${profile})"
        return 1
    fi
}

run_service() {
    local service=$1
    local profile=$2
    
    print_info "Running ${service} with profile: ${profile}"
    
    validate_service "${service}"
    validate_profile "${profile}"
    
    # Map service names to compose service names
    local compose_service=""
    case "${service}" in
        neurondb)
            case "${profile}" in
                cuda) compose_service="neurondb-cuda" ;;
                rocm) compose_service="neurondb-rocm" ;;
                metal) compose_service="neurondb-metal" ;;
                *) compose_service="neurondb" ;;
            esac
            ;;
        neuronagent)
            case "${profile}" in
                cuda) compose_service="neuronagent-cuda" ;;
                rocm) compose_service="neuronagent-rocm" ;;
                metal) compose_service="neuronagent-metal" ;;
                *) compose_service="neuronagent" ;;
            esac
            ;;
        neuronmcp)
            case "${profile}" in
                cuda) compose_service="neuronmcp-cuda" ;;
                rocm) compose_service="neuronmcp-rocm" ;;
                metal) compose_service="neuronmcp-metal" ;;
                *) compose_service="neuronmcp" ;;
            esac
            ;;
        neurondesktop)
            compose_service="neurondesk-init neurondesk-api neurondesk-frontend"
            ;;
    esac
    
    if [[ -z "${compose_service}" ]]; then
        print_error "No compose service mapped for ${service} with profile ${profile}"
        return 1
    fi
    
    # Split multiple services if needed
    local services_array=(${compose_service})
    local compose_cmd=$(get_compose_cmd)
    
    # Stop and remove existing containers by name to avoid conflicts
    # This handles containers that were started outside of docker-compose
    # Map compose service names to container names
    for svc in "${services_array[@]}"; do
        local container_name=""
        case "${svc}" in
            neurondb)
                case "${profile}" in
                    cuda) container_name="neurondb-${PG_MAJOR}-cuda" ;;
                    rocm) container_name="neurondb-${PG_MAJOR}-rocm" ;;
                    metal) container_name="neurondb-${PG_MAJOR}-metal" ;;
                    *) container_name="neurondb-${PG_MAJOR}-cpu" ;;
                esac
                ;;
            neurondb-cuda) container_name="neurondb-${PG_MAJOR}-cuda" ;;
            neurondb-rocm) container_name="neurondb-${PG_MAJOR}-rocm" ;;
            neurondb-metal) container_name="neurondb-${PG_MAJOR}-metal" ;;
            neuronagent)
                case "${profile}" in
                    cuda) container_name="neuronagent-cuda" ;;
                    rocm) container_name="neuronagent-rocm" ;;
                    metal) container_name="neuronagent-metal" ;;
                    *) container_name="neuronagent" ;;
                esac
                ;;
            neuronagent-cuda) container_name="neuronagent-cuda" ;;
            neuronagent-rocm) container_name="neuronagent-rocm" ;;
            neuronagent-metal) container_name="neuronagent-metal" ;;
            neuronmcp)
                case "${profile}" in
                    cuda) container_name="neurondb-mcp-cuda" ;;
                    rocm) container_name="neurondb-mcp-rocm" ;;
                    metal) container_name="neurondb-mcp-metal" ;;
                    *) container_name="neurondb-mcp" ;;
                esac
                ;;
            neuronmcp-cuda) container_name="neurondb-mcp-cuda" ;;
            neuronmcp-rocm) container_name="neurondb-mcp-rocm" ;;
            neuronmcp-metal) container_name="neurondb-mcp-metal" ;;
            neurondesk-api) container_name="neurondesk-api" ;;
            neurondesk-frontend) container_name="neurondesk-frontend" ;;
            neurondesk-init) container_name="neurondesk-init" ;;
        esac
        
        if [[ -n "${container_name}" ]]; then
            # Stop and remove container if it exists
            if docker ps -a --format '{{.Names}}' | grep -q "^${container_name}$"; then
                print_info "Stopping existing container: ${container_name}"
                docker stop "${container_name}" 2>/dev/null || true
                docker rm "${container_name}" 2>/dev/null || true
            fi
        fi
    done
    
    if is_compose_v2; then
        # docker compose v2: --profile comes after subcommand
        cd "${PROJECT_ROOT}"
        ${compose_cmd} -f "${COMPOSE_FILE}" up -d --force-recreate --profile "${profile}" "${services_array[@]}"
        local exit_code=$?
        cd - > /dev/null
    else
        # docker-compose v1: --profile comes before subcommand
        cd "${PROJECT_ROOT}"
        ${compose_cmd} -f "${COMPOSE_FILE}" --profile "${profile}" up -d --force-recreate "${services_array[@]}"
        local exit_code=$?
        cd - > /dev/null
    fi
    
    if [[ ${exit_code} -eq 0 ]]; then
        print_success "Started ${service} (${profile})"
    else
        print_error "Failed to start ${service} (${profile})"
        return 1
    fi
}

build_all() {
    local profile=$1
    print_info "Building all services with profile: ${profile}"
    
    validate_profile "${profile}"
    
    local compose_cmd=$(get_compose_cmd)
    
    if is_compose_v2; then
        cd "${PROJECT_ROOT}"
        ${compose_cmd} -f "${COMPOSE_FILE}" build --profile "${profile}"
        local exit_code=$?
        cd - > /dev/null
    else
        cd "${PROJECT_ROOT}"
        ${compose_cmd} -f "${COMPOSE_FILE}" --profile "${profile}" build
        local exit_code=$?
        cd - > /dev/null
    fi
    
    if [[ ${exit_code} -eq 0 ]]; then
        print_success "Built all services (${profile})"
    else
        print_error "Failed to build all services (${profile})"
        return 1
    fi
}

run_all() {
    local profile=$1
    print_info "Running all services with profile: ${profile}"
    
    validate_profile "${profile}"
    
    local compose_cmd=$(get_compose_cmd)
    
    if is_compose_v2; then
        cd "${PROJECT_ROOT}"
        ${compose_cmd} -f "${COMPOSE_FILE}" up -d --profile "${profile}"
        local exit_code=$?
        cd - > /dev/null
    else
        cd "${PROJECT_ROOT}"
        ${compose_cmd} -f "${COMPOSE_FILE}" --profile "${profile}" up -d
        local exit_code=$?
        cd - > /dev/null
    fi
    
    if [[ ${exit_code} -eq 0 ]]; then
        print_success "Started all services (${profile})"
    else
        print_error "Failed to start all services (${profile})"
        return 1
    fi
}

# ============================================================================
# Argument Parsing
# ============================================================================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --version|-v)
                show_version
                exit 0
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            --list)
                list_services
                exit 0
                ;;
            --list-docker|--lsit-docker)
                list_docker_resources
                exit 0
                ;;
            --delete=*)
                DELETE_RESOURCE="${1#*=}"
                shift
                ;;
            --build)
                ACTION="build"
                shift
                ;;
            --run)
                ACTION="run"
                shift
                ;;
            --all)
                ALL_SERVICES=true
                shift
                ;;
            --profile)
                PROFILE="$2"
                shift 2
                ;;
            --postgres-version|--pg-version)
                PG_MAJOR="$2"
                validate_postgres_version "${PG_MAJOR}"
                shift 2
                ;;
            neurondb|neuronagent|neuronmcp|neurondesktop)
                if [[ "${ALL_SERVICES}" == false ]]; then
                    SERVICES+=("$1")
                fi
                shift
                ;;
            *)
                print_error "Unknown argument: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

show_usage() {
    cat << EOF
Usage: ${SCRIPT_NAME} [OPTIONS] [SERVICE...]

Options:
    --version, -v              Show script version and docker info
    --list                     List available services and profiles
    --list-docker               List all NeuronDB docker containers, images, volumes, and networks
    --delete=RESOURCE           Delete a docker resource (format: type:name or name)
                                Types: container, image, volume, network
                                Examples: --delete=container:neurondb-18-cpu
                                         --delete=image:neurondb:cpu-pg18
                                         --delete=volume:dockers_neurondb-data
    --build                    Build container(s)
    --run                      Run container(s)
    --all                      Apply action to all services
    --profile PROFILE          Specify profile (cpu, cuda, rocm, metal, default)
    --postgres-version VERSION Specify PostgreSQL major version (default: 18)
    
Services:
    neurondb                   NeuronDB database service
    neuronagent                NeuronAgent service
    neuronmcp                  NeuronMCP service
    neurondesktop              NeuronDesktop service

Examples:
    ${SCRIPT_NAME} --version
    ${SCRIPT_NAME} --list
    ${SCRIPT_NAME} --list-docker
    ${SCRIPT_NAME} --delete=container:neurondb-18-cpu
    ${SCRIPT_NAME} --delete=image:neurondb:cpu-pg18
    ${SCRIPT_NAME} --delete=volume:dockers_neurondb-data
    ${SCRIPT_NAME} --build neurondb --profile cpu
    ${SCRIPT_NAME} --run neuronagent --profile cuda
    ${SCRIPT_NAME} --build neurondb --postgres-version 18
    ${SCRIPT_NAME} --all --build --profile default --postgres-version 17
    ${SCRIPT_NAME} --all --run --profile cuda
    ${SCRIPT_NAME} neurondb neuronagent --build --profile rocm

EOF
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    # Check if docker is available
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check if docker compose is available (will be checked by get_compose_cmd when needed)
    
    # Validate compose file exists
    validate_compose_file
    
    # Parse arguments
    if [[ $# -eq 0 ]]; then
        show_usage
        exit 0
    fi
    
    parse_arguments "$@"
    
    # Handle delete action
    if [[ -n "${DELETE_RESOURCE}" ]]; then
        delete_docker_resource "${DELETE_RESOURCE}"
        exit $?
    fi
    
    # Validate PostgreSQL version
    validate_postgres_version "${PG_MAJOR}"
    
    # Execute action
    if [[ -z "${ACTION}" ]]; then
        print_error "No action specified. Use --build or --run"
        show_usage
        exit 1
    fi
    
    if [[ "${ALL_SERVICES}" == true ]]; then
        # Handle --all
        case "${ACTION}" in
            build)
                build_all "${PROFILE}"
                ;;
            run)
                run_all "${PROFILE}"
                ;;
        esac
    elif [[ ${#SERVICES[@]} -eq 0 ]]; then
        print_error "No services specified. Use --all or specify service names"
        show_usage
        exit 1
    else
        # Handle specific services
        local failed=0
        for service in "${SERVICES[@]}"; do
            case "${ACTION}" in
                build)
                    if ! build_service "${service}" "${PROFILE}"; then
                        ((failed++))
                    fi
                    ;;
                run)
                    if ! run_service "${service}" "${PROFILE}"; then
                        ((failed++))
                    fi
                    ;;
            esac
        done
        
        if [[ ${failed} -gt 0 ]]; then
            print_error "${failed} service(s) failed"
            exit 1
        fi
    fi
}

# Run main function
main "$@"

