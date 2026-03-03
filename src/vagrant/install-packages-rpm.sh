#!/bin/bash
#
# vagrant/install-packages-rpm.sh - Install RPM packages in correct dependency order
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

    # Try different RPM naming patterns - search recursively
    pkg_file=$(find "$PACKAGES_DIR/$module" -name "${module}-*.rpm" -type f 2>/dev/null | grep -v ".src.rpm" | head -1)

    if [ -z "$pkg_file" ]; then
        # Try alternative naming with version
        pkg_file=$(find "$PACKAGES_DIR/$module" -name "${module}_*.rpm" -type f 2>/dev/null | grep -v ".src.rpm" | head -1)
    fi

    if [ -z "$pkg_file" ]; then
        # Try any RPM file in the module directory
        pkg_file=$(find "$PACKAGES_DIR/$module" -name "*.rpm" -type f 2>/dev/null | grep -v ".src.rpm" | head -1)
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
    if sudo rpm -ivh "$pkg_file" 2>&1; then
        log_success "$module installed successfully"
        return 0
    else
        # Try to fix dependencies or upgrade if already installed
        log_info "Attempting to upgrade or fix dependencies..."
        if sudo rpm -Uvh "$pkg_file" 2>&1; then
            log_success "$module installed/upgraded successfully"
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

    # Check if package is installed using rpm
    if rpm -q "$module" >/dev/null 2>&1; then
        local version=$(rpm -q --qf '%{VERSION}-%{RELEASE}' "$module" 2>/dev/null)
        log_success "$module is installed (version: $version)"
        return 0
    else
        log_error "$module is not installed"
        return 1
    fi
}

# Main installation workflow
main() {
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║          RPM Package Installation Script                     ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

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
    rpm -qa | grep -E "neurondb|neuronagent|neuronmcp|neurondesktop" || log_warning "No neuron packages found in rpm list"

    if [ ${#failed_modules[@]} -eq 0 ]; then
        log_success "All packages installed successfully!"
        return 0
    else
        log_error "Failed to install: ${failed_modules[*]}"
        return 1
    fi
}

main "$@"

