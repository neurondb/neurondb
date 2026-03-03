#!/bin/bash
#
# vagrant/download-packages.sh - Download packages from GitHub Actions or copy from local
#
# This script attempts to download packages using GitHub CLI, with fallback
# to copying from the shared folder if gh CLI is not available or fails.
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PACKAGES_DIR="/vagrant/packages"
LOCAL_PACKAGES_DIR="/vagrant/packages-local"
MODULES=("neurondb" "neuronagent" "neuronmcp" "neurondesktop")
REPO="${GITHUB_REPO:-neurondb/neurondb}"

log_info() {
    echo -e "${BLUE}[DOWNLOAD]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[DOWNLOAD]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[DOWNLOAD]${NC} $*"
}

log_error() {
    echo -e "${RED}[DOWNLOAD]${NC} $*" >&2
}

# Get latest successful workflow run ID
get_latest_run() {
    local workflow="$1"
    local run_id

    run_id=$(gh run list --repo "$REPO" --workflow="$workflow" --limit 10 \
        --json databaseId,conclusion,status \
        --jq '.[] | select(.conclusion=="success") | .databaseId' 2>/dev/null | head -1)

    if [ -z "$run_id" ]; then
        return 1
    fi

    echo "$run_id"
}

# Download packages via GitHub CLI
download_via_gh() {
    local module="$1"
    local workflow="${module^} - Packages"
    local module_dir="$PACKAGES_DIR/$module"

    log_info "Attempting to download $module via GitHub CLI..."

    # Check if gh CLI is available and authenticated
    if ! command -v gh >/dev/null 2>&1; then
        log_warning "GitHub CLI not found"
        return 1
    fi

    if ! gh auth status >/dev/null 2>&1; then
        log_warning "GitHub CLI not authenticated"
        return 1
    fi

    # Get latest successful run
    local run_id
    run_id=$(get_latest_run "$workflow")
    if [ -z "$run_id" ]; then
        log_warning "No successful run found for $workflow"
        return 1
    fi

    log_info "Found run ID: $run_id"

    # Download artifacts
    mkdir -p "$module_dir"
    if gh run download --repo "$REPO" "$run_id" --dir "$module_dir" 2>/dev/null; then
        # Check if we got packages
        if find "$module_dir" -name "*.deb" -type f | grep -q .; then
            log_success "Downloaded $module packages via GitHub CLI"
            return 0
        else
            log_warning "Download completed but no .deb files found"
            return 1
        fi
    else
        log_warning "Failed to download via GitHub CLI"
        return 1
    fi
}

# Copy packages from local shared folder
copy_from_local() {
    local module="$1"
    local module_local_dir="$LOCAL_PACKAGES_DIR/$module"
    local module_dir="$PACKAGES_DIR/$module"

    log_info "Attempting to copy $module from local shared folder..."

    if [ ! -d "$module_local_dir" ]; then
        log_warning "Local packages directory not found: $module_local_dir"
        return 1
    fi

    # Find packages in the local directory structure
    # Could be in subdirectories like: module-local/module-deb-packages/module/*.deb
    local packages=$(find "$module_local_dir" -name "*.deb" -type f)
    if [ -z "$packages" ]; then
        log_warning "No .deb files found in $module_local_dir"
        return 1
    fi

    # Create destination directory
    mkdir -p "$module_dir"

    # Copy packages and preserve structure
    log_info "Copying packages from $module_local_dir to $module_dir"
    cp -r "$module_local_dir"/* "$module_dir/" 2>/dev/null || {
        # Try to find packages in nested structure
        local pkg_count=$(find "$module_local_dir" -name "*.deb" -type f | wc -l)
        if [ "$pkg_count" -gt 0 ]; then
            find "$module_local_dir" -name "*.deb" -type f -exec cp --parents {} "$module_dir/" \;
            find "$module_local_dir" -name "SHA256SUMS*" -type f -exec cp --parents {} "$module_dir/" \;
            log_success "Copied $pkg_count package(s) from local directory"
            return 0
        else
            return 1
        fi
    }

    log_success "Copied $module packages from local directory"
    return 0
}

# Download or copy packages for a module
download_module() {
    local module="$1"
    local success=false

    log_info "Processing $module..."

    # Try GitHub CLI first
    if download_via_gh "$module"; then
        success=true
    else
        # Fallback to local copy
        log_info "GitHub CLI download failed, trying local copy..."
        if copy_from_local "$module"; then
            success=true
        fi
    fi

    if [ "$success" = true ]; then
        # Verify we have packages
        local pkg_count=$(find "$PACKAGES_DIR/$module" -name "*.deb" -type f 2>/dev/null | wc -l)
        if [ "$pkg_count" -gt 0 ]; then
            log_success "$module: $pkg_count package(s) available"
            return 0
        fi
    fi

    log_error "$module: Failed to obtain packages"
    return 1
}

# Main
main() {
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║          Package Download Script                             ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    mkdir -p "$PACKAGES_DIR"

    local failed_modules=()
    for module in "${MODULES[@]}"; do
        if ! download_module "$module"; then
            failed_modules+=("$module")
        fi
    done

    echo ""
    if [ ${#failed_modules[@]} -eq 0 ]; then
        log_success "All packages downloaded/copied successfully!"
        log_info "Packages are available in: $PACKAGES_DIR"
        return 0
    else
        log_error "Failed to download/copy: ${failed_modules[*]}"
        return 1
    fi
}

main "$@"

