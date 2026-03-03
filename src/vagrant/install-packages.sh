#!/bin/bash
#
# vagrant/install-packages.sh - Install DEB packages in correct dependency order
#
# Installs packages in order: NeuronDB → NeuronAgent/NeuronMCP → NeuronDesktop
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PACKAGES_DIR="/vagrant/packages"

log_info() {
    echo -e "${BLUE}[INSTALL]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[INSTALL]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[INSTALL]${NC} $*"
}

log_error() {
    echo -e "${RED}[INSTALL]${NC} $*" >&2
}

# Find package file for a module
find_package() {
    local module="$1"
    local pkg_file

    # Search recursively in nested directories - check path or filename
    pkg_file=$(find "$PACKAGES_DIR" -type f -name "*.deb" 2>/dev/null | grep -E "(/${module}[_-]|/${module}/|${module}_)" | head -1)
    
    if [ -z "$pkg_file" ]; then
        # Try case-insensitive search in full path
        pkg_file=$(find "$PACKAGES_DIR" -type f -name "*.deb" 2>/dev/null | grep -i "$module" | head -1)
    fi

    if [ -z "$pkg_file" ]; then
        # Try direct module directory
        pkg_file=$(find "$PACKAGES_DIR/$module" -name "${module}_*.deb" -type f 2>/dev/null | head -1)
    fi

    if [ -z "$pkg_file" ]; then
        # Try any DEB file in module directory
        pkg_file=$(find "$PACKAGES_DIR/$module" -name "*.deb" -type f 2>/dev/null | head -1)
    fi

    echo "$pkg_file"
}

# Install a single package
install_package() {
    local module="$1"
    local pkg_file

    log_info "Installing $module..."

    pkg_file=$(find_package "$module")
    if [ -z "$pkg_file" ] || [ ! -f "$pkg_file" ]; then
        log_error "Package file not found for $module"
        return 1
    fi

    log_info "Package file: $(basename "$pkg_file")"

    # Install package
    if sudo dpkg -i "$pkg_file" 2>&1; then
        log_success "$module installed successfully"
        return 0
    else
        # Try to fix dependencies
        log_info "Attempting to fix dependencies..."
        if sudo apt-get install -f -y; then
            log_success "$module installed successfully (dependencies fixed)"
            return 0
        else
            log_error "Failed to install $module"
            return 1
        fi
    fi
}

# Verify package installation
verify_installation() {
    local module="$1"

    # Check if package is installed - use exact match
    local pkg_status=$(dpkg -l | awk -v pkg="$module" '$2 == pkg && $1 == "ii" {print $2 " " $3; exit}')
    if [ -n "$pkg_status" ]; then
        local version=$(echo "$pkg_status" | awk '{print $2}')
        log_success "$module is installed (version: $version)"
        return 0
    else
        # Try alternative check
        if dpkg-query -W -f='${Status}' "$module" 2>/dev/null | grep -q "install ok installed"; then
            local version=$(dpkg-query -W -f='${Version}' "$module" 2>/dev/null)
            log_success "$module is installed (version: $version)"
            return 0
        else
            log_error "$module is not installed"
            return 1
        fi
    fi
}

# Main installation workflow
main() {
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║          Package Installation Script                         ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Update package cache
    log_info "Updating package cache..."
    sudo apt-get update

    local failed_modules=()

    # Step 1: Install NeuronDB (base extension)
    if ! install_package "neurondb"; then
        failed_modules+=("neurondb")
    else
        verify_installation "neurondb"
    fi

    # Step 2: Install NeuronAgent and NeuronMCP (depend on NeuronDB)
    if ! install_package "neuronagent"; then
        failed_modules+=("neuronagent")
    else
        verify_installation "neuronagent"
    fi

    if ! install_package "neuronmcp"; then
        failed_modules+=("neuronmcp")
    else
        verify_installation "neuronmcp"
    fi

    # Step 3: Install NeuronDesktop (depends on NeuronDB)
    if ! install_package "neurondesktop"; then
        failed_modules+=("neurondesktop")
    else
        verify_installation "neurondesktop"
    fi

    # Final verification
    echo ""
    log_info "Installation summary:"
    dpkg -l | grep -E "^ii.*neuron" || log_warning "No neuron packages found in dpkg list"

    if [ ${#failed_modules[@]} -eq 0 ]; then
        log_success "All packages installed successfully!"
        return 0
    else
        log_error "Failed to install: ${failed_modules[*]}"
        return 1
    fi
}

main "$@"

