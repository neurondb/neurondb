#!/usr/bin/env bash
#-------------------------------------------------------------------------
# build.sh - Unified build script for NeuronDB Ecosystem
#-------------------------------------------------------------------------
# Copies all required files from modules to bin/ directory
# Supports: -v, -v1, -v2 verbose levels (default: clean output)
#-------------------------------------------------------------------------

set -euo pipefail
IFS=$'\n\t'

#=========================================================================
# CONFIGURATION
#=========================================================================

readonly SCRIPT_VERSION="1.0.0"
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
readonly REPO_ROOT="$SCRIPT_DIR"

# Verbosity levels: 0=clean, 1=normal, 2=verbose, 3=debug
VERBOSE="${VERBOSE:-0}"

# Color support - only enable if output is to a terminal
if [[ -t 1 ]] && [[ "${TERM:-}" != "dumb" ]] && [[ -z "${NO_COLOR:-}" ]]; then
    readonly RED='\033[0;31m'
    readonly GREEN='\033[0;32m'
    readonly YELLOW='\033[1;33m'
    readonly BLUE='\033[0;34m'
    readonly MAGENTA='\033[0;35m'
    readonly CYAN='\033[0;36m'
    readonly NC='\033[0m'
    readonly BOLD='\033[1m'
    readonly DIM='\033[2m'
else
    readonly RED=''
    readonly GREEN=''
    readonly YELLOW=''
    readonly BLUE=''
    readonly MAGENTA=''
    readonly CYAN=''
    readonly NC=''
    readonly BOLD=''
    readonly DIM=''
fi

#=========================================================================
# LOGGING FUNCTIONS
#=========================================================================

log_info() {
    [[ ${VERBOSE} -ge 1 ]] && printf "[${BLUE}ℹ${NC}] %s\n" "$*" || true
}

log_success() {
    [[ ${VERBOSE} -ge 0 ]] && printf "[${GREEN}✓${NC}] %s\n" "$*" || true
}

log_warn() {
    [[ ${VERBOSE} -ge 1 ]] && printf "[${YELLOW}⚠${NC}] %s\n" "$*" >&2 || true
}

log_error() {
    [[ ${VERBOSE} -ge 0 ]] && printf "[${RED}✗${NC}] %s\n" "$*" >&2 || true
}

log_verbose() {
    [[ ${VERBOSE} -ge 2 ]] && printf "[${DIM}DEBUG${NC}] %s\n" "$*" || true
}

log_debug() {
    [[ ${VERBOSE} -ge 3 ]] && printf "[${DIM}DEBUG${NC}] %s\n" "$*" || true
}

section() {
    [[ ${VERBOSE} -ge 1 ]] && {
        echo ""
        printf "${BOLD}${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
        printf "${BOLD}${MAGENTA}  %s${NC}\n" "$*"
        printf "${BOLD}${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
        echo ""
    } || true
}

#=========================================================================
# FILE OPERATIONS
#=========================================================================

copy_file() {
    local src="$1"
    local dst="$2"
    local desc="${3:-}"
    
    if [[ ! -f "$src" ]]; then
        log_verbose "File not found: $src (skipping)"
        return 1
    fi
    
    local dst_dir
    dst_dir=$(dirname "$dst")
    [[ ! -d "$dst_dir" ]] && mkdir -p "$dst_dir"
    
    if cp "$src" "$dst" 2>/dev/null; then
        [[ ${VERBOSE} -ge 1 ]] && {
            if [[ -n "$desc" ]]; then
                printf "  ${GREEN}✓${NC} %s\n" "$desc"
            else
                printf "  ${GREEN}✓${NC} $(basename "$dst")\n"
            fi
        } || true
        log_debug "Copied: $src -> $dst"
        return 0
    else
        log_warn "Failed to copy: $src -> $dst"
        return 1
    fi
}

copy_dir() {
    local src="$1"
    local dst="$2"
    local desc="${3:-}"
    
    if [[ ! -d "$src" ]]; then
        log_verbose "Directory not found: $src (skipping)"
        return 1
    fi
    
    [[ ! -d "$dst" ]] && mkdir -p "$dst"
    
    if cp -r "$src"/* "$dst"/ 2>/dev/null; then
        [[ ${VERBOSE} -ge 1 ]] && {
            if [[ -n "$desc" ]]; then
                printf "  ${GREEN}✓${NC} %s\n" "$desc"
            else
                printf "  ${GREEN}✓${NC} $(basename "$dst")/\n"
            fi
        } || true
        log_debug "Copied directory: $src -> $dst"
        return 0
    else
        log_warn "Failed to copy directory: $src -> $dst"
        return 1
    fi
}

verify_file() {
    local file="$1"
    local desc="${2:-$(basename "$file")}"
    
    if [[ -f "$file" ]]; then
        log_verbose "Verified: $desc"
        return 0
    else
        log_warn "Missing: $desc"
        return 1
    fi
}

#=========================================================================
# MODULE COPY FUNCTIONS
#=========================================================================

copy_neurondb() {
    section "NeuronDB"
    
    local bin_dir="$REPO_ROOT/bin/neurondb"
    mkdir -p "$bin_dir"
    
    local copied=0
    local missing=0
    
    # Copy library files (required)
    local lib_copied=false
    if [[ -f "$REPO_ROOT/NeuronDB/neurondb.so" ]]; then
        if copy_file "$REPO_ROOT/NeuronDB/neurondb.so" "$bin_dir/neurondb.so" "neurondb.so"; then
            ((copied++))
            lib_copied=true
        else
            ((missing++))
        fi
    elif [[ -f "$REPO_ROOT/NeuronDB/neurondb.dylib" ]]; then
        if copy_file "$REPO_ROOT/NeuronDB/neurondb.dylib" "$bin_dir/neurondb.dylib" "neurondb.dylib"; then
            ((copied++))
            lib_copied=true
        else
            ((missing++))
        fi
    else
        log_warn "NeuronDB library not found (.so or .dylib)"
        ((missing++))
    fi
    
    # Copy control file (required)
    if copy_file "$REPO_ROOT/NeuronDB/neurondb.control" "$bin_dir/neurondb.control" "neurondb.control"; then
        ((copied++))
    else
        ((missing++))
    fi
    
    # Copy SQL files (optional)
    copy_file "$REPO_ROOT/NeuronDB/neurondb--1.0.sql" "$bin_dir/neurondb--1.0.sql" "neurondb--1.0.sql" && ((copied++)) || true
    copy_file "$REPO_ROOT/NeuronDB/neurondb--2.0.sql" "$bin_dir/neurondb--2.0.sql" "neurondb--2.0.sql" && ((copied++)) || true
    copy_file "$REPO_ROOT/NeuronDB/neurondb--1.0--2.0.sql" "$bin_dir/neurondb--1.0--2.0.sql" "neurondb--1.0--2.0.sql" && ((copied++)) || true
    
    if [[ $missing -eq 0 ]]; then
        log_success "NeuronDB: $copied files copied"
        return 0
    else
        log_error "NeuronDB: $missing required files missing"
        return 1
    fi
}

copy_neuronagent() {
    section "NeuronAgent"
    
    local bin_dir="$REPO_ROOT/bin/neuronagent"
    mkdir -p "$bin_dir"
    
    local copied=0
    local missing=0
    
    # Copy binary (required)
    if [[ -f "$REPO_ROOT/NeuronAgent/bin/neuronagent" ]]; then
        if copy_file "$REPO_ROOT/NeuronAgent/bin/neuronagent" "$bin_dir/neuronagent" "neuronagent binary"; then
            ((copied++))
            chmod +x "$bin_dir/neuronagent" 2>/dev/null || true
        else
            ((missing++))
        fi
    else
        log_warn "NeuronAgent binary not found"
        ((missing++))
    fi
    
    # Copy configuration files
    if [[ -f "$REPO_ROOT/NeuronAgent/conf/neuronagent-config.yaml" ]]; then
        copy_file "$REPO_ROOT/NeuronAgent/conf/neuronagent-config.yaml" "$bin_dir/neuronagent-config.yaml" "neuronagent-config.yaml" && ((copied++)) || true
    fi
    if [[ -f "$REPO_ROOT/NeuronAgent/conf/agent-profiles.yaml" ]]; then
        copy_file "$REPO_ROOT/NeuronAgent/conf/agent-profiles.yaml" "$bin_dir/agent-profiles.yaml" "agent-profiles.yaml" && ((copied++)) || true
    fi
    
    # Copy conf directory
    if [[ -d "$REPO_ROOT/NeuronAgent/bin/conf" ]]; then
        copy_dir "$REPO_ROOT/NeuronAgent/bin/conf" "$bin_dir/conf" "conf/" && ((copied++)) || true
    elif [[ -d "$REPO_ROOT/NeuronAgent/conf" ]]; then
        copy_dir "$REPO_ROOT/NeuronAgent/conf" "$bin_dir/conf" "conf/" && ((copied++)) || true
    fi
    
    # Copy SQL files
    if [[ -f "$REPO_ROOT/NeuronAgent/bin/sql/neuron-agent.sql" ]]; then
        copy_file "$REPO_ROOT/NeuronAgent/bin/sql/neuron-agent.sql" "$bin_dir/neuron-agent.sql" "neuron-agent.sql" && ((copied++)) || true
    elif [[ -f "$REPO_ROOT/NeuronAgent/neuron-agent.sql" ]]; then
        copy_file "$REPO_ROOT/NeuronAgent/neuron-agent.sql" "$bin_dir/neuron-agent.sql" "neuron-agent.sql" && ((copied++)) || true
    fi
    
    # Copy scripts
    if [[ -d "$REPO_ROOT/NeuronAgent/bin/scripts" ]]; then
        copy_dir "$REPO_ROOT/NeuronAgent/bin/scripts" "$bin_dir/scripts" "scripts/" && ((copied++)) || true
        find "$bin_dir/scripts" -type f -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true
    elif [[ -d "$REPO_ROOT/NeuronAgent/scripts" ]]; then
        copy_dir "$REPO_ROOT/NeuronAgent/scripts" "$bin_dir/scripts" "scripts/" && ((copied++)) || true
        find "$bin_dir/scripts" -type f -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true
    fi
    
    
    if [[ $missing -eq 0 ]]; then
        log_success "NeuronAgent: $copied files copied"
        return 0
    else
        log_error "NeuronAgent: $missing required files missing"
        return 1
    fi
}

copy_neuronmcp() {
    section "NeuronMCP"
    
    local bin_dir="$REPO_ROOT/bin/neuronmcp"
    mkdir -p "$bin_dir"
    
    local copied=0
    local missing=0
    
    # Copy binary (required) - check multiple possible names
    local binary_copied=false
    if [[ -f "$REPO_ROOT/NeuronMCP/bin/neuronmcp" ]]; then
        if copy_file "$REPO_ROOT/NeuronMCP/bin/neuronmcp" "$bin_dir/neuronmcp" "neuronmcp binary"; then
            ((copied++))
            binary_copied=true
            chmod +x "$bin_dir/neuronmcp" 2>/dev/null || true
        else
            ((missing++))
        fi
    elif [[ -f "$REPO_ROOT/NeuronMCP/bin/neurondb-mcp" ]]; then
        if copy_file "$REPO_ROOT/NeuronMCP/bin/neurondb-mcp" "$bin_dir/neuronmcp" "neuronmcp binary"; then
            ((copied++))
            binary_copied=true
            chmod +x "$bin_dir/neuronmcp" 2>/dev/null || true
        else
            ((missing++))
        fi
    elif [[ -f "$REPO_ROOT/NeuronMCP/neurondb-mcp" ]]; then
        if copy_file "$REPO_ROOT/NeuronMCP/neurondb-mcp" "$bin_dir/neuronmcp" "neuronmcp binary"; then
            ((copied++))
            binary_copied=true
            chmod +x "$bin_dir/neuronmcp" 2>/dev/null || true
        else
            ((missing++))
        fi
    else
        log_warn "NeuronMCP binary not found"
        ((missing++))
    fi
    
    # Copy SQL files
    if [[ -f "$REPO_ROOT/NeuronMCP/bin/sql/neuron-mcp.sql" ]]; then
        copy_file "$REPO_ROOT/NeuronMCP/bin/sql/neuron-mcp.sql" "$bin_dir/neuron-mcp.sql" "neuron-mcp.sql" && ((copied++)) || true
    elif [[ -f "$REPO_ROOT/NeuronMCP/sql/neuron-mcp.sql" ]]; then
        copy_file "$REPO_ROOT/NeuronMCP/sql/neuron-mcp.sql" "$bin_dir/neuron-mcp.sql" "neuron-mcp.sql" && ((copied++)) || true
    elif [[ -f "$REPO_ROOT/NeuronMCP/neuron-mcp.sql" ]]; then
        copy_file "$REPO_ROOT/NeuronMCP/neuron-mcp.sql" "$bin_dir/neuron-mcp.sql" "neuron-mcp.sql" && ((copied++)) || true
    fi
    
    # Copy config example
    if [[ -f "$REPO_ROOT/NeuronMCP/conf/mcp-config.json.example" ]]; then
        copy_file "$REPO_ROOT/NeuronMCP/conf/mcp-config.json.example" "$bin_dir/mcp-config.json.example" "mcp-config.json.example" && ((copied++)) || true
    elif [[ -f "$REPO_ROOT/NeuronMCP/mcp-config.json.example" ]]; then
        copy_file "$REPO_ROOT/NeuronMCP/mcp-config.json.example" "$bin_dir/mcp-config.json.example" "mcp-config.json.example" && ((copied++)) || true
    fi
    
    # Copy scripts if they exist
    if [[ -d "$REPO_ROOT/NeuronMCP/bin/scripts" ]]; then
        copy_dir "$REPO_ROOT/NeuronMCP/bin/scripts" "$bin_dir/scripts" "scripts/" && ((copied++)) || true
        find "$bin_dir/scripts" -type f -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true
    elif [[ -d "$REPO_ROOT/NeuronMCP/scripts" ]]; then
        copy_dir "$REPO_ROOT/NeuronMCP/scripts" "$bin_dir/scripts" "scripts/" && ((copied++)) || true
        find "$bin_dir/scripts" -type f -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true
    fi
    
    
    if [[ $missing -eq 0 ]]; then
        log_success "NeuronMCP: $copied files copied"
        return 0
    else
        log_error "NeuronMCP: $missing required files missing"
        return 1
    fi
}

copy_neurondesktop() {
    section "NeuronDesktop"
    
    local bin_dir="$REPO_ROOT/bin/neurondesktop"
    mkdir -p "$bin_dir"
    
    local copied=0
    local missing=0
    
    # Copy binary (required)
    if [[ -f "$REPO_ROOT/NeuronDesktop/bin/neurondesktop" ]]; then
        if copy_file "$REPO_ROOT/NeuronDesktop/bin/neurondesktop" "$bin_dir/neurondesktop" "neurondesktop binary"; then
            ((copied++))
            chmod +x "$bin_dir/neurondesktop" 2>/dev/null || true
        else
            ((missing++))
        fi
    else
        log_warn "NeuronDesktop binary not found"
        ((missing++))
    fi
    
    # Copy SQL files
    if [[ -f "$REPO_ROOT/NeuronDesktop/bin/sql/neuron-desktop.sql" ]]; then
        copy_file "$REPO_ROOT/NeuronDesktop/bin/sql/neuron-desktop.sql" "$bin_dir/neuron-desktop.sql" "neuron-desktop.sql" && ((copied++)) || true
    elif [[ -f "$REPO_ROOT/NeuronDesktop/bin/sql/neurondesktop.sql" ]]; then
        copy_file "$REPO_ROOT/NeuronDesktop/bin/sql/neurondesktop.sql" "$bin_dir/neuron-desktop.sql" "neuron-desktop.sql" && ((copied++)) || true
    elif [[ -f "$REPO_ROOT/NeuronDesktop/neuron-desktop.sql" ]]; then
        copy_file "$REPO_ROOT/NeuronDesktop/neuron-desktop.sql" "$bin_dir/neuron-desktop.sql" "neuron-desktop.sql" && ((copied++)) || true
    fi
    
    # Copy scripts
    if [[ -d "$REPO_ROOT/NeuronDesktop/bin/scripts" ]]; then
        copy_dir "$REPO_ROOT/NeuronDesktop/bin/scripts" "$bin_dir/scripts" "scripts/" && ((copied++)) || true
        find "$bin_dir/scripts" -type f -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true
    elif [[ -d "$REPO_ROOT/NeuronDesktop/scripts" ]]; then
        copy_dir "$REPO_ROOT/NeuronDesktop/scripts" "$bin_dir/scripts" "scripts/" && ((copied++)) || true
        find "$bin_dir/scripts" -type f -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true
    fi
    
    # Copy frontend build artifacts
    if [[ -d "$REPO_ROOT/NeuronDesktop/frontend/.next" ]]; then
        copy_dir "$REPO_ROOT/NeuronDesktop/frontend/.next" "$bin_dir/.next" "frontend/.next/" && ((copied++)) || true
    fi
    
    # Copy frontend package.json for reference
    if [[ -f "$REPO_ROOT/NeuronDesktop/frontend/package.json" ]]; then
        copy_file "$REPO_ROOT/NeuronDesktop/frontend/package.json" "$bin_dir/frontend-package.json" "frontend-package.json" && ((copied++)) || true
    fi
    
    
    if [[ $missing -eq 0 ]]; then
        log_success "NeuronDesktop: $copied files copied"
        return 0
    else
        log_error "NeuronDesktop: $missing required files missing"
        return 1
    fi
}

copy_sdks() {
    section "SDKs"
    
    local bin_dir="$REPO_ROOT/bin"
    local copied=0
    
    # Copy Python SDK if it exists
    if [[ -d "$REPO_ROOT/sdks/python" ]]; then
        log_verbose "Python SDK found, checking if needed..."
        # Only copy if any module explicitly needs it
        # For now, we'll just log it - modules can reference it from sdks/
        log_verbose "Python SDK available at: sdks/python/"
        ((copied++))
    fi
    
    # Copy TypeScript SDK if it exists
    if [[ -d "$REPO_ROOT/sdks/typescript" ]]; then
        log_verbose "TypeScript SDK found, checking if needed..."
        # Only copy if any module explicitly needs it
        log_verbose "TypeScript SDK available at: sdks/typescript/"
        ((copied++))
    fi
    
    if [[ $copied -gt 0 ]]; then
        log_success "SDKs: Available in sdks/ directory"
    else
        log_verbose "No SDKs to copy"
    fi
    
    return 0
}

#=========================================================================
# MAIN BUILD FUNCTION
#=========================================================================

build_all() {
    section "Building NeuronDB Ecosystem"
    
    local total_errors=0
    local total_success=0
    
    # Copy all modules
    if copy_neurondb; then
        ((total_success++))
    else
        ((total_errors++))
    fi
    
    if copy_neuronagent; then
        ((total_success++))
    else
        ((total_errors++))
    fi
    
    if copy_neuronmcp; then
        ((total_success++))
    else
        ((total_errors++))
    fi
    
    if copy_neurondesktop; then
        ((total_success++))
    else
        ((total_errors++))
    fi
    
    # Copy SDKs (informational)
    copy_sdks || true
    
    # Summary
    echo ""
    if [[ $total_errors -eq 0 ]]; then
        log_success "Build complete: All $total_success modules copied successfully"
        return 0
    else
        log_error "Build complete: $total_success modules succeeded, $total_errors modules failed"
        return 1
    fi
}

#=========================================================================
# USAGE AND ARGUMENT PARSING
#=========================================================================

usage() {
    cat <<EOF
NeuronDB Ecosystem Build Script v${SCRIPT_VERSION}

Usage: $0 [OPTIONS] <command>

Commands:
  build              Copy all required files from modules to bin/

Options:
  -v, --verbose      Enable verbose output (level 1)
  -v1                Verbose level 1 (normal verbose)
  -v2                Verbose level 2 (detailed verbose)
  -v3                Verbose level 3 (debug verbose)
  -h, --help         Show this help message

Verbosity Levels:
  0 (default)        Clean output - only success/error messages
  1 (-v, -v1)        Normal verbose - shows file copying progress
  2 (-v2)            Detailed verbose - shows all operations
  3 (-v3)            Debug verbose - shows everything including skipped files

Examples:
  $0 build                    # Clean output
  $0 -v build                # Verbose output
  $0 -v2 build               # Detailed verbose output

EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -v|--verbose)
                VERBOSE=1
                shift
                ;;
            -v1)
                VERBOSE=1
                shift
                ;;
            -v2)
                VERBOSE=2
                shift
                ;;
            -v3)
                VERBOSE=3
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            build)
                build_all
                exit $?
                ;;
            *)
                log_error "Unknown option or command: $1"
                usage
                exit 1
                ;;
        esac
    done
}

#=========================================================================
# MAIN
#=========================================================================

main() {
    cd "$REPO_ROOT"
    
    if [[ $# -eq 0 ]]; then
        usage
        exit 1
    fi
    
    parse_args "$@"
}

main "$@"

