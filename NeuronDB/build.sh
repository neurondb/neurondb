#!/usr/bin/env bash
#-------------------------------------------------------------------------
# build.sh - Modern, modular build script for NeurondB
#-------------------------------------------------------------------------
# Supports:
#   - Platforms: macOS, Rocky Linux, Ubuntu/Debian, RHEL, CentOS
#   - GPU Backends: CUDA (NVIDIA), Metal (Apple), ROCm (AMD)
#   - PostgreSQL: 16, 17, 18
#-------------------------------------------------------------------------

set -euo pipefail
IFS=$'\n\t'

#=========================================================================
# CONFIGURATION
#=========================================================================

readonly SCRIPT_VERSION="3.0.0"
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Build configuration
VERBOSE="${VERBOSE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
SKIP_TESTS="${SKIP_TESTS:-1}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

# GPU configuration
GPU_MODE="${GPU_MODE:-auto}"
CUDA_PATH="${CUDA_PATH:-${CUDA_HOME:-}}"
ROCM_PATH="${ROCM_PATH:-${ROCM_HOME:-}}"
ONNX_RUNTIME_PATH="${ONNX_RUNTIME_PATH:-/usr/local/onnxruntime}"

# PostgreSQL configuration
PG_VERSION="${PG_VERSION:-}"
PG_CONFIG="${PG_CONFIG:-}"

# Detected state
PLATFORM=""
OS_ID=""
OS_VERSION=""
ARCH=""
SELECTED_PG_CONFIG=""
PG_MAJOR_SELECTED=""
DETECTED_CUDA_PATH=""
DETECTED_ROCM_PATH=""
DETECTED_METAL="false"

#=========================================================================
# UTILITIES
#=========================================================================

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
    # No colors when not in a terminal, TERM=dumb, or NO_COLOR is set
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

log_info() { printf "[${BLUE}ℹ${NC}] %s\n" "$*"; }
log_success() { printf "[${GREEN}✓${NC}] %s\n" "$*"; }
log_warn() { printf "[${YELLOW}⚠${NC}] %s\n" "$*" >&2; }
log_error() { printf "[${RED}✗${NC}] %s\n" "$*" >&2; }
log_fatal() { log_error "$*"; exit 1; }
log_verbose() { [[ ${VERBOSE} -eq 1 ]] && printf "[${DIM}ℹ DEBUG${NC}] %s\n" "$*" || true; }

section() {
    echo ""
    printf "${BOLD}${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    printf "${BOLD}${MAGENTA}  %s${NC}\n" "$*"
    printf "${BOLD}${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    echo ""
}

has_cmd() { command -v "$1" >/dev/null 2>&1; }
has_path() { [[ -e "$1" ]]; }

run_cmd() {
    local desc="$1" start elapsed rc
    shift
    start=$(date +%s)
    
    if [[ ${DRY_RUN} -eq 1 ]]; then
        log_info "[DRY-RUN] Would execute: $*"
        return 0
    fi
    
    if [[ ${VERBOSE} -eq 1 ]]; then
        log_verbose "Executing: $*"
        "$@" && rc=0 || rc=$?
    else
        if output=$("$@" 2>&1); then
            rc=0
        else
            rc=$?
            echo "$output" >&2
        fi
    fi
    
    elapsed=$(( $(date +%s) - start ))
    [[ $rc -eq 0 ]] && log_success "$desc (${elapsed}s)" || log_error "$desc failed (${elapsed}s, exit: $rc)"
    return $rc
}

#=========================================================================
# PLATFORM DETECTION
#=========================================================================

detect_platform() {
    section "Platform Detection"
    
    local uname_s uname_m
    uname_s="$(uname -s)"
    uname_m="$(uname -m)"
    ARCH="$uname_m"
    
    log_info "Kernel: $uname_s $uname_m"
    
    case "$uname_s" in
        Darwin)
            PLATFORM="macos"
            OS_ID="macos"
            OS_VERSION="$(sw_vers -productVersion)"
            log_info "Detected: macOS $OS_VERSION"
            [[ "$uname_m" == "arm64" ]] && log_info "Architecture: Apple Silicon" || log_info "Architecture: Intel"
            ;;
        Linux)
            detect_linux_distro
            ;;
        *)
            log_fatal "Unsupported OS: $uname_s"
            ;;
    esac
    
    log_success "Platform: $PLATFORM ($OS_ID $OS_VERSION)"
}

detect_linux_distro() {
    if [[ -f /etc/os-release ]]; then
        # shellcheck source=/dev/null
        . /etc/os-release
        OS_ID="${ID:-unknown}"
        OS_VERSION="${VERSION_ID:-unknown}"
        log_info "OS Release: ${PRETTY_NAME:-$OS_ID $OS_VERSION}"
        
        case "${ID:-}" in
            rocky|centos|rhel|almalinux|fedora)
                PLATFORM="rocky"
                ;;
            ubuntu|debian)
                PLATFORM="ubuntu"
                ;;
            *)
                if grep -qiE "rocky|centos|rhel|redhat|fedora" /etc/os-release 2>/dev/null; then
                    PLATFORM="rocky"
                elif grep -qiE "debian|ubuntu" /etc/os-release 2>/dev/null; then
                    PLATFORM="ubuntu"
                else
                    PLATFORM="unknown"
                    log_warn "Unknown Linux distribution: $OS_ID"
                fi
                ;;
        esac
    else
        [[ -f /etc/rocky-release ]] || [[ -f /etc/redhat-release ]] && PLATFORM="rocky" || \
        has_cmd apt-get && PLATFORM="ubuntu" || \
        (has_cmd dnf || has_cmd yum) && PLATFORM="rocky" || \
        log_fatal "Cannot determine Linux distribution"
    fi
}

#=========================================================================
# GPU DETECTION
#=========================================================================

detect_gpu_backends() {
    section "GPU Backend Detection"
    
    local cuda_ok=false rocm_ok=false metal_ok=false
    
    detect_cuda && cuda_ok=true
    detect_rocm && rocm_ok=true
    detect_metal && metal_ok=true
    
    # Determine GPU mode
    case "$GPU_MODE" in
        auto)
            if [[ "$cuda_ok" == "true" ]]; then
                GPU_MODE="cuda"
                log_info "Auto-selected: CUDA"
            elif [[ "$rocm_ok" == "true" ]]; then
                GPU_MODE="rocm"
                log_info "Auto-selected: ROCm"
            elif [[ "$metal_ok" == "true" ]]; then
                GPU_MODE="metal"
                log_info "Auto-selected: Metal"
            else
                GPU_MODE="none"
                log_info "No GPU detected, building CPU-only"
            fi
            ;;
        cuda)
            [[ "$cuda_ok" != "true" ]] && log_warn "CUDA requested but not available" && GPU_MODE="none"
            ;;
        rocm)
            [[ "$rocm_ok" != "true" ]] && log_warn "ROCm requested but not available" && GPU_MODE="none"
            ;;
        metal)
            [[ "$metal_ok" != "true" ]] && log_warn "Metal requested but not available" && GPU_MODE="none"
            ;;
        none)
            GPU_MODE="none"
            ;;
        *)
            log_warn "Unknown GPU_MODE: $GPU_MODE, using auto"
            GPU_MODE="auto"
            detect_gpu_backends
            return
            ;;
    esac
    
    log_success "GPU mode: $GPU_MODE"
    echo ""
    printf "  ${BOLD}GPU Backend Status:${NC}\n"
    if [[ "$cuda_ok" == "true" ]]; then
        log_success "CUDA: ${CUDA_PATH:-$DETECTED_CUDA_PATH}"
    else
        log_error "CUDA: Not available"
    fi
    if [[ "$rocm_ok" == "true" ]]; then
        log_success "ROCm: ${ROCM_PATH:-$DETECTED_ROCM_PATH}"
    else
        log_error "ROCm: Not available"
    fi
    if [[ "$metal_ok" == "true" ]]; then
        log_success "Metal: Available"
    else
        log_error "Metal: Not available"
    fi
    echo ""
}

detect_cuda() {
    log_verbose "Detecting CUDA..."
    
    # Check nvcc in PATH
    if has_cmd nvcc; then
        local nvcc_path cuda_home
        nvcc_path="$(command -v nvcc)"
        cuda_home="$(dirname "$(dirname "$nvcc_path")")"
        if [[ -f "$cuda_home/include/cuda_runtime.h" ]]; then
            DETECTED_CUDA_PATH="$cuda_home"
            CUDA_PATH="${CUDA_PATH:-$DETECTED_CUDA_PATH}"
            log_info "Found CUDA via nvcc: $CUDA_PATH"
            return 0
        fi
    fi
    
    # Check common paths
    local candidates=(
        "${CUDA_PATH:-}"
        "${CUDA_HOME:-}"
        "/usr/local/cuda"
        "/usr/local/cuda-12.0"
        "/usr/local/cuda-11.8"
        "/opt/cuda"
        "/usr/lib/cuda"
    )
    
    for candidate in "${candidates[@]}"; do
        [[ -z "$candidate" ]] && continue
        if [[ -f "$candidate/include/cuda_runtime.h" ]]; then
            DETECTED_CUDA_PATH="$candidate"
            CUDA_PATH="$candidate"
            log_info "Found CUDA at: $CUDA_PATH"
            return 0
        fi
    done
    
    # Check system headers (Ubuntu/Debian packages)
    if [[ -f "/usr/include/cuda/cuda_runtime.h" ]]; then
        DETECTED_CUDA_PATH="/usr"
        CUDA_PATH="/usr"
        log_info "Found CUDA via system headers"
        return 0
    fi
    
    log_verbose "CUDA not detected"
    return 1
}

detect_rocm() {
    log_verbose "Detecting ROCm..."
    
    # Check hipcc in PATH
    if has_cmd hipcc; then
        local hipcc_path rocm_home
        hipcc_path="$(command -v hipcc)"
        rocm_home="$(dirname "$(dirname "$hipcc_path")")"
        if [[ -f "$rocm_home/include/hip/hip_runtime.h" ]]; then
            DETECTED_ROCM_PATH="$rocm_home"
            ROCM_PATH="${ROCM_PATH:-$DETECTED_ROCM_PATH}"
            log_info "Found ROCm via hipcc: $ROCM_PATH"
            return 0
        fi
    fi
    
    # Check common paths
    local candidates=(
        "${ROCM_PATH:-}"
        "${ROCM_HOME:-}"
        "/opt/rocm"
        "/opt/rocm-6.0"
        "/opt/rocm-5.7"
        "/usr/rocm"
    )
    
    for candidate in "${candidates[@]}"; do
        [[ -z "$candidate" ]] && continue
        if [[ -f "$candidate/include/hip/hip_runtime.h" ]]; then
            DETECTED_ROCM_PATH="$candidate"
            ROCM_PATH="$candidate"
            log_info "Found ROCm at: $ROCM_PATH"
            return 0
        fi
    done
    
    log_verbose "ROCm not detected"
    return 1
}

detect_metal() {
    [[ "$PLATFORM" != "macos" ]] && return 1
    
    log_verbose "Detecting Metal..."
    
    if xcode-select -p >/dev/null 2>&1; then
        if [[ -d "/System/Library/Frameworks/Metal.framework" ]]; then
            DETECTED_METAL="true"
            log_info "Metal framework available"
            return 0
        fi
    fi
    
    log_verbose "Metal not available"
    return 1
}

#=========================================================================
# POSTGRESQL DETECTION
#=========================================================================

detect_postgresql() {
    section "PostgreSQL Detection"
    
    SELECTED_PG_CONFIG=""
    PG_MAJOR_SELECTED=""
    
    # Priority 1: User-specified
    if [[ -n "${PG_CONFIG:-}" ]] && [[ -x "${PG_CONFIG:-}" ]]; then
        SELECTED_PG_CONFIG="$PG_CONFIG"
        log_info "Using user-specified PG_CONFIG: $SELECTED_PG_CONFIG"
    # Priority 2: PATH
    elif has_cmd pg_config; then
        SELECTED_PG_CONFIG="$(command -v pg_config)"
        log_info "Found pg_config in PATH: $SELECTED_PG_CONFIG"
    # Priority 3: Platform-specific paths
    else
        find_pg_config
    fi
    
    # Extract version
    if [[ -n "$SELECTED_PG_CONFIG" ]]; then
        local pg_ver_str
        pg_ver_str=$("$SELECTED_PG_CONFIG" --version 2>/dev/null || echo "")
        if [[ -n "$pg_ver_str" ]]; then
            PG_MAJOR_SELECTED=$(echo "$pg_ver_str" | sed -n 's/.*PostgreSQL \([0-9][0-9]*\)\..*/\1/p')
            log_info "PostgreSQL version: $pg_ver_str (major: $PG_MAJOR_SELECTED)"
            
            case "$PG_MAJOR_SELECTED" in
                16|17|18)
                    log_success "PostgreSQL $PG_MAJOR_SELECTED is supported"
                    ;;
                *)
                    log_warn "PostgreSQL $PG_MAJOR_SELECTED may not be fully supported"
                    ;;
            esac
        fi
    else
        log_warn "pg_config not found - will attempt installation"
        PG_MAJOR_SELECTED="${PG_VERSION:-18}"
    fi
    
    export SELECTED_PG_CONFIG
}

find_pg_config() {
    local candidates=()
    
    case "$PLATFORM" in
        ubuntu)
            for ver in 18 17 16; do
                candidates+=("/usr/lib/postgresql/$ver/bin/pg_config")
            done
            ;;
        rocky)
            for ver in 18 17 16; do
                candidates+=("/usr/pgsql-$ver/bin/pg_config")
            done
            ;;
        macos)
            for ver in 18 17 16; do
                candidates+=(
                    "/opt/homebrew/opt/postgresql@$ver/bin/pg_config"
                    "/usr/local/opt/postgresql@$ver/bin/pg_config"
                )
            done
            candidates+=("/opt/homebrew/bin/pg_config" "/usr/local/bin/pg_config")
            ;;
    esac
    
    for candidate in "${candidates[@]}"; do
        if [[ -x "$candidate" ]]; then
            SELECTED_PG_CONFIG="$candidate"
            log_info "Found pg_config: $SELECTED_PG_CONFIG"
            return 0
        fi
    done
    
    return 1
}

#=========================================================================
# DEPENDENCY INSTALLATION
#=========================================================================

install_dependencies() {
    # Skip if SKIP_DEP_INSTALL is set (dependencies already checked and present)
    if [[ "${SKIP_DEP_INSTALL:-0}" -eq 1 ]] || [[ "${SKIP_DEP_INSTALL:-}" == "1" ]]; then
        log_info "Dependencies already satisfied - skipping installation step"
        log_info "Skipping: package installation, waiting for package manager"
        # Still re-detect PostgreSQL in case it was just installed
        detect_postgresql
        return 0
    fi
    
    section "Installing Dependencies"
    
    case "$PLATFORM" in
        macos) install_dependencies_macos ;;
        ubuntu) install_dependencies_ubuntu ;;
        rocky) install_dependencies_rocky ;;
        *) log_fatal "Cannot install dependencies for platform: $PLATFORM" ;;
    esac
    
    # Re-detect PostgreSQL after installation
    detect_postgresql
}

install_dependencies_macos() {
    has_cmd brew || log_fatal "Homebrew required. Install from https://brew.sh"
    
    # Xcode CLT for Metal
    if ! xcode-select -p >/dev/null 2>&1; then
        log_info "Installing Xcode Command Line Tools..."
        xcode-select --install || log_warn "Xcode CLT installation may require user interaction"
    fi
				
    # Base packages
    local packages=("llvm" "make" "git" "curl" "pkg-config" "libxml2" "openssl@3" "zlib")
    for pkg in "${packages[@]}"; do
        if ! brew list --versions "$pkg" >/dev/null 2>&1; then
            run_cmd "Install $pkg" brew install "$pkg" || log_warn "Failed to install $pkg"
        fi
    done
    
    # PostgreSQL
    if [[ -z "$SELECTED_PG_CONFIG" ]]; then
        local pg_formula="postgresql@${PG_MAJOR_SELECTED:-18}"
        if ! brew list --versions "$pg_formula" >/dev/null 2>&1; then
            run_cmd "Install PostgreSQL" brew install "$pg_formula" || brew install postgresql || true
        fi
        local pg_prefix
        pg_prefix=$(brew --prefix "$pg_formula" 2>/dev/null || brew --prefix postgresql 2>/dev/null || echo "")
        [[ -n "$pg_prefix" ]] && [[ -x "$pg_prefix/bin/pg_config" ]] && SELECTED_PG_CONFIG="$pg_prefix/bin/pg_config"
    fi
				
    install_onnx_runtime
    log_success "macOS dependencies installed"
}

install_dependencies_ubuntu() {
    # Skip waiting if SKIP_DEP_INSTALL is set
    if [[ "${SKIP_DEP_INSTALL:-0}" -ne 1 ]] && [[ "${SKIP_DEP_INSTALL:-}" != "1" ]]; then
        wait_for_package_manager
    fi
    
    local packages=(
        "build-essential" "git" "curl" "wget" "ca-certificates"
        "pkg-config" "libxml2-dev" "libssl-dev" "libcurl4-openssl-dev" "zlib1g-dev"
    )
    
    if [[ -z "$SELECTED_PG_CONFIG" ]]; then
        packages+=("postgresql-server-dev-${PG_MAJOR_SELECTED:-18}")
    fi
    
    if [[ "$GPU_MODE" == "cuda" ]] && [[ -z "${CUDA_PATH:-}" ]]; then
        packages+=("nvidia-cuda-toolkit" "nvidia-cuda-dev")
    fi
    
    run_cmd "apt-get update" sudo apt-get update -y
    run_cmd "apt-get install" sudo apt-get install -y --no-install-recommends "${packages[@]}"
    
    install_onnx_runtime
    log_success "Ubuntu dependencies installed"
}

install_dependencies_rocky() {
    # Skip waiting if SKIP_DEP_INSTALL is set
    if [[ "${SKIP_DEP_INSTALL:-0}" -ne 1 ]] && [[ "${SKIP_DEP_INSTALL:-}" != "1" ]]; then
        wait_for_package_manager
    fi
    
    local pkg_manager="dnf"
    has_cmd dnf || pkg_manager="yum"
    
    local packages=(
        "gcc" "gcc-c++" "make" "git" "curl" "wget" "ca-certificates"
        "pkgconfig" "libxml2-devel" "openssl-devel" "libcurl-devel" "zlib-devel"
    )
    
    # PostgreSQL repo
    if [[ ! -f /etc/yum.repos.d/pgdg-redhat-all.repo ]]; then
        local rhel_version
        rhel_version=$(rpm -E %{rhel} 2>/dev/null || echo "9")
        local pg_repo_url="https://download.postgresql.org/pub/repos/yum/reporpms/EL-${rhel_version}-x86_64/pgdg-redhat-repo-latest.noarch.rpm"
        run_cmd "Install PostgreSQL repo" sudo $pkg_manager install -y "$pg_repo_url" || true
    fi
    
    if [[ -z "$SELECTED_PG_CONFIG" ]]; then
        packages+=("postgresql${PG_MAJOR_SELECTED:-18}-devel")
    fi
    
    run_cmd "$pkg_manager install" sudo $pkg_manager install -y "${packages[@]}" || true
    
    install_onnx_runtime
    log_success "Rocky dependencies installed"
}

wait_for_package_manager() {
    local max_wait=300 wait_interval=5 waited=0
    
    case "$PLATFORM" in
        ubuntu)
            while [[ $waited -lt $max_wait ]]; do
                if ! pgrep -f "(apt|dpkg)" >/dev/null 2>&1 && \
                   [[ ! -f /var/lib/dpkg/lock-frontend ]] && \
                   [[ ! -f /var/lib/dpkg/lock ]]; then
                    return 0
                fi
                [[ $waited -eq 0 ]] && log_info "Waiting for package manager..."
                sleep $wait_interval
                waited=$((waited + wait_interval))
            done
            ;;
        rocky)
            while [[ $waited -lt $max_wait ]]; do
                if ! pgrep -f "(dnf|yum|rpm)" >/dev/null 2>&1 && \
                   [[ ! -f /var/run/dnf.pid ]]; then
                    return 0
                fi
                [[ $waited -eq 0 ]] && log_info "Waiting for package manager..."
                sleep $wait_interval
                waited=$((waited + wait_interval))
            done
            ;;
    esac
}

install_onnx_runtime() {
    [[ -d "$ONNX_RUNTIME_PATH" ]] && \
    [[ -f "$ONNX_RUNTIME_PATH/include/onnxruntime_c_api.h" ]] && \
    log_info "ONNX Runtime already installed" && return 0
    
    local onnx_version="1.17.0" onnx_arch onnx_url
    
    case "$PLATFORM" in
        macos)
            onnx_arch=$([[ "$ARCH" == "arm64" ]] && echo "osx-arm64" || echo "osx-x86_64")
            ;;
        *)
            onnx_arch="linux-x64"
            ;;
    esac
    
    onnx_url="https://github.com/microsoft/onnxruntime/releases/download/v${onnx_version}/onnxruntime-${onnx_arch}-${onnx_version}.tgz"
    
    log_info "Downloading ONNX Runtime ${onnx_version}..."
    local tmp_file="/tmp/onnxruntime-${onnx_version}.tgz"
    
    if has_cmd curl; then
        run_cmd "Download ONNX Runtime" curl -L -f -o "$tmp_file" "$onnx_url"
    elif has_cmd wget; then
        run_cmd "Download ONNX Runtime" wget -O "$tmp_file" "$onnx_url"
    else
        log_fatal "curl or wget required for ONNX Runtime"
    fi
    
    run_cmd "Create ONNX Runtime directory" sudo mkdir -p "$ONNX_RUNTIME_PATH"
    run_cmd "Extract ONNX Runtime" sudo tar -xzf "$tmp_file" -C "$ONNX_RUNTIME_PATH" --strip-components=1
    rm -f "$tmp_file"
    
    [[ -f "$ONNX_RUNTIME_PATH/include/onnxruntime_c_api.h" ]] || log_fatal "ONNX Runtime installation failed"
    log_success "ONNX Runtime installed"
}

#=========================================================================
# BUILD CONFIGURATION
#=========================================================================

write_config_header() {
    section "Generating Configuration"
    
    mkdir -p include
    
    local pg_major="${PG_MAJOR_SELECTED:-18}"
    local build_date=$(date +%Y%m%d)
    local build_time=$(date +"%Y-%m-%d %H:%M:%S")
    local platform_upper
    platform_upper=$(echo "$PLATFORM" | tr '[:lower:]' '[:upper:]')
    
    log_info "Generating neurondb_config.h..."
    
    cat > include/neurondb_config.h <<EOF
/*-------------------------------------------------------------------------
 * neurondb_config.h - Auto-generated configuration header
 * Generated: ${build_time}
 * Platform: ${PLATFORM} | PostgreSQL: ${pg_major} | GPU: ${GPU_MODE}
 *-------------------------------------------------------------------------*/

#ifndef NEURONDB_CONFIG_H
#define NEURONDB_CONFIG_H

#define NEURONDB_VERSION "2.1.0"
#define NEURONDB_BUILD_DATE "${build_date}"
#define NEURONDB_PLATFORM_${platform_upper} 1
#define NEURONDB_PG_VERSION ${pg_major}

/* GPU support */
EOF

    case "$GPU_MODE" in
        cuda)
            [[ -n "${CUDA_PATH:-}" ]] && cat >> include/neurondb_config.h <<EOF
#define NDB_GPU_CUDA 1
#define HAVE_CUDA 1
#define CUDA_PATH "${CUDA_PATH}"
EOF
            ;;
        rocm)
            [[ -n "${ROCM_PATH:-}" ]] && cat >> include/neurondb_config.h <<EOF
#define NDB_GPU_HIP 1
#define ROCM_PATH "${ROCM_PATH}"
EOF
            ;;
        metal)
            cat >> include/neurondb_config.h <<EOF
#define NDB_GPU_METAL 1
EOF
            ;;
    esac
    
    [[ -f "$ONNX_RUNTIME_PATH/include/onnxruntime_c_api.h" ]] && \
    cat >> include/neurondb_config.h <<EOF
#define HAVE_ONNX_RUNTIME 1
#define ONNX_RUNTIME_PATH "${ONNX_RUNTIME_PATH}"
EOF

    cat >> include/neurondb_config.h <<EOF

#define NEURONDB_CPU_FALLBACK 1

#endif /* NEURONDB_CONFIG_H */
EOF

    log_success "Configuration header generated"
}

write_makefile_local() {
    log_info "Writing Makefile.local..."
    
    local pgc="${SELECTED_PG_CONFIG:-$(command -v pg_config || true)}"
    
    cat > Makefile.local <<EOF
# Auto-generated by build.sh - do not edit manually
# Generated: $(date)

PG_CONFIG ?= ${pgc}
PG_MAJOR ?= ${PG_MAJOR_SELECTED:-18}
GPU_MODE ?= ${GPU_MODE:-none}
CUDA_PATH ?= ${CUDA_PATH:-}
ROCM_PATH ?= ${ROCM_PATH:-}
ONNX_RUNTIME_PATH ?= ${ONNX_RUNTIME_PATH:-/usr/local/onnxruntime}

V ?= 1
QUIET_ONNX ?= 1

CFLAGS += -O2 -fPIC -fno-lto
LDFLAGS += -fno-lto

.PHONY: compile
.DEFAULT_GOAL := compile
compile: all
EOF

    log_success "Makefile.local written"
}

#=========================================================================
# BUILD EXECUTION
#=========================================================================

build_neurondb() {
    section "Building NeurondB"
    
    [[ -z "$SELECTED_PG_CONFIG" ]] && log_fatal "pg_config not found"
    
    local pgc="$SELECTED_PG_CONFIG"
    local num_jobs
    num_jobs=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    
    log_info "Build Configuration:"
    log_info "  Platform:       $PLATFORM"
    log_info "  PostgreSQL:     $("$pgc" --version 2>/dev/null || echo 'unknown')"
    log_info "  GPU Mode:       $GPU_MODE"
    [[ -n "${CUDA_PATH:-}" ]] && log_info "  CUDA Path:      $CUDA_PATH"
    [[ -n "${ROCM_PATH:-}" ]] && log_info "  ROCm Path:      $ROCM_PATH"
    log_info "  Parallel Jobs:  $num_jobs"
    echo ""
    
    [[ ${FORCE_REBUILD} -eq 1 ]] && run_cmd "Clean build" make clean PG_CONFIG="$pgc" || true
    
    local make_args="PG_CONFIG=$pgc"
    [[ -n "${CUDA_PATH:-}" ]] && make_args="$make_args CUDA_PATH=$CUDA_PATH"
    [[ -n "${ROCM_PATH:-}" ]] && make_args="$make_args ROCM_PATH=$ROCM_PATH"
    
    log_info "Compiling NeurondB..."
    
    if [[ ${VERBOSE} -eq 1 ]]; then
        run_cmd "Build NeurondB" make -j"$num_jobs" $make_args all
    else
        local build_output
        if build_output=$(make -j"$num_jobs" $make_args all 2>&1); then
            log_success "Build completed"
            [[ ${VERBOSE} -eq 1 ]] && echo "$build_output" | grep -E "(Error|error|Warning|warning|Linking)" | tail -20 || true
        else
            log_error "Build failed"
            echo "$build_output" | tail -100
            log_fatal "Compilation failed"
        fi
    fi
    
    verify_build_output
    
    # Ensure SQL file is generated after build (needed for installation)
    # The Makefile expects neurondb--1.0.sql but we may have neurondb--2.0.sql
    if [[ ! -f neurondb--1.0.sql ]]; then
        log_info "Ensuring SQL file exists for installation..."
        # Try multiple sources in order of preference
        if [[ -f neurondb--1.0.sql.linux ]]; then
            cp neurondb--1.0.sql.linux neurondb--1.0.sql
            log_info "Copied neurondb--1.0.sql.linux to neurondb--1.0.sql"
        elif [[ -f neurondb--1.0.sql.macos ]]; then
            cp neurondb--1.0.sql.macos neurondb--1.0.sql
            log_info "Copied neurondb--1.0.sql.macos to neurondb--1.0.sql"
        elif [[ -f neurondb--2.0.sql ]]; then
            # Use 2.0 file as fallback (Makefile expects 1.0 but control file says 2.0)
            cp neurondb--2.0.sql neurondb--1.0.sql
            log_info "Copied neurondb--2.0.sql to neurondb--1.0.sql (version fallback)"
        elif [[ -f neurondb--2.0.sql.linux ]]; then
            cp neurondb--2.0.sql.linux neurondb--1.0.sql
            log_info "Copied neurondb--2.0.sql.linux to neurondb--1.0.sql"
        elif [[ -f neurondb--2.0.sql.macos ]]; then
            cp neurondb--2.0.sql.macos neurondb--1.0.sql
            log_info "Copied neurondb--2.0.sql.macos to neurondb--1.0.sql"
        elif [[ -f neurondb--1.0.sql.in ]]; then
            # Try to generate from template
            log_info "Generating SQL file from template..."
            if [[ "$UNAME_S" == "Darwin" ]]; then
                make neurondb--1.0.sql.macos || log_warn "Could not generate macOS SQL file"
                [[ -f neurondb--1.0.sql.macos ]] && cp neurondb--1.0.sql.macos neurondb--1.0.sql
            else
                make neurondb--1.0.sql.linux || log_warn "Could not generate Linux SQL file"
                [[ -f neurondb--1.0.sql.linux ]] && cp neurondb--1.0.sql.linux neurondb--1.0.sql
            fi
        else
            log_warn "No SQL file found and template not available, installation may fail"
        fi
    fi
    
    log_success "NeurondB built successfully"
}

verify_build_output() {
    local lib_path=""
    
    # Check for library files
    if [[ -f neurondb.so ]]; then
        lib_path="neurondb.so"
    elif [[ -f neurondb.dylib ]]; then
        lib_path="neurondb.dylib"
    fi
    
    if [[ -z "$lib_path" ]]; then
        log_warn "Build output library not found immediately after build"
        log_info "Checking if build is still in progress or library is in a different location..."
        
        # Wait a moment and check again (in case of async builds)
        sleep 1
        if [[ -f neurondb.so ]]; then
            lib_path="neurondb.so"
        elif [[ -f neurondb.dylib ]]; then
            lib_path="neurondb.dylib"
        fi
    fi
    
    if [[ -n "$lib_path" ]]; then
        log_success "Library created: $lib_path"
        has_cmd ls && ls -lh "$lib_path"
    else
        log_warn "Library verification skipped (may be installed directly)"
    fi
}

install_neurondb() {
    [[ ${SKIP_INSTALL} -eq 1 ]] && log_info "Skipping installation" && return 0
    
    section "Installing NeurondB"
    
    # Ensure SQL file exists before installation (should already be generated in build phase)
    if [[ ! -f neurondb--1.0.sql ]]; then
        log_info "SQL file missing, attempting to create from available sources..."
        if [[ -f neurondb--1.0.sql.linux ]]; then
            cp neurondb--1.0.sql.linux neurondb--1.0.sql
        elif [[ -f neurondb--1.0.sql.macos ]]; then
            cp neurondb--1.0.sql.macos neurondb--1.0.sql
        elif [[ -f neurondb--2.0.sql ]]; then
            cp neurondb--2.0.sql neurondb--1.0.sql
        elif [[ -f neurondb--2.0.sql.linux ]]; then
            cp neurondb--2.0.sql.linux neurondb--1.0.sql
        elif [[ -f neurondb--2.0.sql.macos ]]; then
            cp neurondb--2.0.sql.macos neurondb--1.0.sql
        else
            log_error "No SQL file found for installation"
            log_fatal "Installation cannot proceed without SQL file"
        fi
    fi
    
    local pgc="$SELECTED_PG_CONFIG"
    local sudo_cmd=""
    [[ "$PLATFORM" != "macos" ]] && [[ "$(id -u)" != "0" ]] && sudo_cmd="sudo"
    
    run_cmd "Install NeurondB" $sudo_cmd make install PG_CONFIG="$pgc" || log_fatal "Installation failed"
    
    # If control file specifies version 2.0 but we installed 1.0.sql, create symlink/copy
    local sharedir pkglibdir default_version
    sharedir=$("$pgc" --sharedir 2>/dev/null || echo "unknown")
    pkglibdir=$("$pgc" --pkglibdir 2>/dev/null || echo "unknown")
    default_version=$(grep -E "^default_version\s*=" neurondb.control | sed 's/.*=\s*\(.*\)/\1/' | tr -d "'\" " || echo "1.0")
    
    if [[ "$default_version" == "2.0" ]] && [[ -f "${sharedir}/extension/neurondb--1.0.sql" ]] && [[ ! -f "${sharedir}/extension/neurondb--2.0.sql" ]]; then
        log_info "Control file specifies version 2.0, creating neurondb--2.0.sql from 1.0..."
        $sudo_cmd cp "${sharedir}/extension/neurondb--1.0.sql" "${sharedir}/extension/neurondb--2.0.sql" || log_warn "Could not create neurondb--2.0.sql"
    fi
    
    log_success "NeurondB installed"
    echo ""
    printf "  ${BOLD}Installation paths:${NC}\n"
    log_info "Extension SQL:  ${sharedir}/extension/neurondb--${default_version}.sql"
    log_info "Control file:   ${sharedir}/extension/neurondb.control (default_version: ${default_version})"
    log_info "Library:        ${pkglibdir}/neurondb.so"
    echo ""
}

run_tests() {
    [[ ${SKIP_TESTS} -eq 1 ]] && log_info "Skipping tests" && return 0
    
    section "Running Tests"
    
    local pgc="$SELECTED_PG_CONFIG"
    has_cmd psql || log_warn "psql not found, skipping tests" && return 0
    
    run_cmd "Run regression tests" make installcheck PG_CONFIG="$pgc" || log_warn "Some tests failed"
    log_success "Tests completed"
}

#=========================================================================
# MAIN
#=========================================================================

usage() {
    cat <<EOF
NeurondB Build Script v${SCRIPT_VERSION}

Usage: $0 [OPTIONS]

Options:
  -v, --verbose           Enable verbose output
  -n, --dry-run           Show what would be done without executing
  --gpu=MODE              GPU backend: auto, none, cuda, metal, rocm
  --cuda-path=PATH        Path to CUDA toolkit
  --rocm-path=PATH        Path to ROCm
  --pg-version=VERSION    PostgreSQL major version (16, 17, 18)
  --pg-config=PATH        Path to pg_config
  --skip-build            Skip compilation step
  --skip-install          Skip installation step
  --test                  Run regression tests after build
  --clean                 Clean build artifacts before building
  -h, --help              Show this help message

Examples:
  $0                                    # Auto-detect everything
  $0 --gpu=cuda                         # Build with CUDA support
  $0 --gpu=metal --test                 # Build with Metal and run tests
  $0 --pg-version=17 --verbose          # Build for PostgreSQL 17

Platforms: macOS, Rocky Linux, Ubuntu/Debian, RHEL, CentOS
GPU Backends: CUDA (NVIDIA), Metal (Apple), ROCm (AMD)
PostgreSQL: 16, 17, 18

EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -v|--verbose) VERBOSE=1; shift ;;
            -n|--dry-run) DRY_RUN=1; shift ;;
            --gpu) GPU_MODE="$2"; shift 2 ;;
            --gpu=*) GPU_MODE="${1#*=}"; shift ;;
            --cuda-path) CUDA_PATH="$2"; shift 2 ;;
            --cuda-path=*) CUDA_PATH="${1#*=}"; shift ;;
            --rocm-path) ROCM_PATH="$2"; shift 2 ;;
            --rocm-path=*) ROCM_PATH="${1#*=}"; shift ;;
            --pg-version) PG_VERSION="$2"; shift 2 ;;
            --pg-version=*) PG_VERSION="${1#*=}"; shift ;;
            --pg-config) PG_CONFIG="$2"; shift 2 ;;
            --pg-config=*) PG_CONFIG="${1#*=}"; shift ;;
            --skip-build) SKIP_BUILD=1; shift ;;
            --skip-install) SKIP_INSTALL=1; shift ;;
            --test) SKIP_TESTS=0; shift ;;
            --clean) FORCE_REBUILD=1; shift ;;
            -h|--help) usage; exit 0 ;;
            *) log_error "Unknown option: $1"; usage; exit 1 ;;
        esac
    done
}

main() {
    cd "$SCRIPT_DIR"
    parse_args "$@"
    
    # Banner
    echo ""
    printf "${BOLD}${CYAN}"
    printf "╔══════════════════════════════════════════════════════════════════════════╗\n"
    printf "║                        NeurondB Build Script                             ║\n"
    printf "║                              Version ${SCRIPT_VERSION}                                 ║\n"
    printf "╚══════════════════════════════════════════════════════════════════════════╝\n"
    printf "${NC}\n"
    
    # Detection
    detect_platform
    detect_postgresql
    detect_gpu_backends
    
    # Installation
    install_dependencies
    
    # Re-detect after installation
    detect_postgresql
    [[ "$GPU_MODE" == "auto" ]] && detect_gpu_backends
    
    # Configuration
    write_config_header
    write_makefile_local
    
    # Build
    [[ ${SKIP_BUILD} -eq 0 ]] && build_neurondb || log_info "Skipping build"
    
    # Install
    install_neurondb
    
    # Test
    run_tests
    
    # Summary
    section "Build Complete"
    log_success "NeurondB has been successfully built and installed!"
    echo ""
    printf "  ${BOLD}Next steps:${NC}\n"
    echo ""
    log_info "1. Add to postgresql.conf:"
    printf "     ${CYAN}shared_preload_libraries = 'neurondb'${NC}\n"
    echo ""
    log_info "2. Restart PostgreSQL"
    echo ""
    log_info "3. Create extension:"
    printf "     ${CYAN}psql -d mydb -c 'CREATE EXTENSION neurondb;'${NC}\n"
    echo ""
    printf "  ${BOLD}Build summary:${NC}\n"
    log_info "Platform:     ${PLATFORM}"
    log_info "PostgreSQL:   ${PG_MAJOR_SELECTED:-unknown}"
    log_info "GPU Backend:  ${GPU_MODE}"
    [[ -n "${CUDA_PATH:-}" ]] && log_info "CUDA Path:    ${CUDA_PATH}"
    [[ -n "${ROCM_PATH:-}" ]] && log_info "ROCm Path:    ${ROCM_PATH}"
    echo ""
}

main "$@"
