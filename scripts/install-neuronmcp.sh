#!/bin/bash
# ====================================================================
# NeuronMCP Installation Script
# ====================================================================
# Installs NeuronMCP from source, sets up database, and optionally
# configures as a system service.
# ====================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT_NAME=$(basename "$0")

# Source helper functions
source "$SCRIPT_DIR/install-helpers.sh"

# Default values
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"
SERVICE_DIR="/etc/systemd/system"
CONFIG_DIR="/etc/neurondb"
ENABLE_SERVICE=false
SKIP_BUILD=false
SKIP_DB_SETUP=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prefix)
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        --enable-service)
            ENABLE_SERVICE=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-db-setup)
            SKIP_DB_SETUP=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            cat << EOF
NeuronMCP Installation Script

Usage: $SCRIPT_NAME [OPTIONS]

Options:
    --prefix PATH         Installation prefix (default: /usr/local)
    --enable-service      Enable and start system service
    --skip-build          Skip building from source
    --skip-db-setup       Skip database setup
    -v, --verbose         Enable verbose output
    -h, --help            Show this help message

Examples:
    # Basic installation
    $SCRIPT_NAME

    # Install with service enabled
    $SCRIPT_NAME --enable-service

    # Install to custom location
    $SCRIPT_NAME --prefix ~/neurondb

EOF
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_info "Starting NeuronMCP installation..."

# Check prerequisites
print_info "Checking prerequisites..."
if ! check_go; then
    exit 1
fi

if ! check_postgres; then
    exit 1
fi

# Build from source
if [ "$SKIP_BUILD" = false ]; then
    print_info "Building NeuronMCP from source..."
    cd "$PROJECT_ROOT/NeuronMCP"
    
    if [ -f Makefile ]; then
        make build
    else
        go build -o bin/neurondb-mcp ./cmd/neurondb-mcp
    fi
    
    if [ ! -f bin/neurondb-mcp ] && [ ! -f bin/neuronmcp ]; then
        print_error "Build failed: binary not found"
        exit 1
    fi
    
    print_success "Build completed"
fi

# Install binary
BINARY_NAME="neurondb-mcp"
BINARY_SRC=""
if [ -f "$PROJECT_ROOT/NeuronMCP/bin/neurondb-mcp" ]; then
    BINARY_SRC="$PROJECT_ROOT/NeuronMCP/bin/neurondb-mcp"
elif [ -f "$PROJECT_ROOT/NeuronMCP/bin/neuronmcp" ]; then
    BINARY_SRC="$PROJECT_ROOT/NeuronMCP/bin/neuronmcp"
    BINARY_NAME="neuronmcp"
else
    print_error "Binary not found. Please build first or use --skip-build"
    exit 1
fi

print_info "Installing binary to $INSTALL_PREFIX/bin/$BINARY_NAME..."
mkdir -p "$INSTALL_PREFIX/bin"
cp "$BINARY_SRC" "$INSTALL_PREFIX/bin/$BINARY_NAME"
chmod +x "$INSTALL_PREFIX/bin/$BINARY_NAME"
print_success "Binary installed"

# Database setup
if [ "$SKIP_DB_SETUP" = false ]; then
    print_info "Setting up database schema..."
    if [ -f "$PROJECT_ROOT/NeuronMCP/scripts/neuronmcp-setup.sh" ]; then
        "$PROJECT_ROOT/NeuronMCP/scripts/neuronmcp-setup.sh"
        print_success "Database setup completed"
    else
        print_warning "Database setup script not found. Skipping database setup."
    fi
fi

# Install service configuration (if root and systemd)
INIT_SYSTEM=$(detect_init_system)
if [ "$INIT_SYSTEM" = "systemd" ] && [ "$(id -u)" -eq 0 ]; then
    print_info "Installing systemd service..."
    mkdir -p "$SERVICE_DIR"
    cp "$PROJECT_ROOT/scripts/services/systemd/neuronmcp.service" "$SERVICE_DIR/"
    
    # Update service file paths
    sed -i "s|/usr/local/bin/neurondb-mcp|$INSTALL_PREFIX/bin/$BINARY_NAME|g" "$SERVICE_DIR/neuronmcp.service"
    
    # Create config directory
    mkdir -p "$CONFIG_DIR"
    if [ ! -f "$CONFIG_DIR/neuronmcp.env" ]; then
        cp "$PROJECT_ROOT/scripts/config/neuronmcp.env.example" "$CONFIG_DIR/neuronmcp.env"
        print_warning "Created $CONFIG_DIR/neuronmcp.env - please update with your configuration"
    fi
    
    systemctl daemon-reload
    
    if [ "$ENABLE_SERVICE" = true ]; then
        systemctl enable neuronmcp
        systemctl start neuronmcp
        print_success "Service enabled and started"
    else
        print_info "Service installed. Enable with: sudo systemctl enable neuronmcp"
        print_info "Start with: sudo systemctl start neuronmcp"
    fi
elif [ "$INIT_SYSTEM" = "launchd" ]; then
    print_info "For macOS, see scripts/services/launchd/README.md for launchd setup"
else
    print_info "Skipping service installation (requires root and systemd/launchd)"
fi

print_success "NeuronMCP installation completed!"
print_info "Binary location: $INSTALL_PREFIX/bin/$BINARY_NAME"




