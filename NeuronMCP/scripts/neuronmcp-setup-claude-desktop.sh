#!/bin/bash
#-------------------------------------------------------------------------
#
# neuronmcp-setup-claude-desktop.sh
#    NeuronMCP Setup for Claude Desktop
#
# Prepares the environment for NeuronMCP to work with Claude Desktop by
# installing Python dependencies and verifying configuration. This script
# should be run ONCE before configuring Claude Desktop. Handles platform-specific
# configuration for Linux, macOS, and Windows.
#
# Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
#
# IDENTIFICATION
#    NeuronMCP/scripts/neuronmcp-setup-claude-desktop.sh
#
#-------------------------------------------------------------------------

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"
SCRIPT_NAME=$(basename "$0")

# Version
VERSION="2.0.0"

# Default values
VERBOSE=false
SKIP_PYTHON_CHECK=false
SKIP_INSTALL=false
CHECK_EMBEDDINGS=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Database connection (optional, for embedding config check)
DB_HOST="${NEURONDB_HOST:-localhost}"
DB_PORT="${NEURONDB_PORT:-5432}"
DB_NAME="${NEURONDB_DATABASE:-neurondb}"
DB_USER="${NEURONDB_USER:-neurondb}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
	case $1 in
		--skip-python-check)
			SKIP_PYTHON_CHECK=true
			shift
			;;
		--skip-install)
			SKIP_INSTALL=true
			shift
			;;
		--check-embeddings)
			CHECK_EMBEDDINGS=true
			shift
			;;
		-v|--verbose)
			VERBOSE=true
			shift
			;;
		-V|--version)
			echo "neuronmcp_setup_claude_desktop.sh version $VERSION"
			exit 0
			;;
		-h|--help)
			cat << EOF
NeuronMCP Setup for Claude Desktop

Usage:
    $SCRIPT_NAME [OPTIONS]

Description:
    Prepares the environment for NeuronMCP to work with Claude Desktop by:
    1. Checking for Python 3
    2. Installing Python dependencies (datasets, pandas, psycopg2-binary, etc.)
    3. Verifying installation
    4. Optionally checking embedding configuration

    IMPORTANT: Run this script ONCE before configuring Claude Desktop.
    Claude Desktop starts the MCP server automatically without running setup scripts,
    so all dependencies must be pre-installed.

Options:
    --skip-python-check    Skip Python version check
    --skip-install         Skip dependency installation (only verify)
    --check-embeddings      Check embedding configuration in PostgreSQL
    -v, --verbose          Enable verbose output
    -V, --version          Show version information
    -h, --help             Show this help message

Environment Variables:
    NEURONDB_HOST          Database host (for embedding check, default: localhost)
    NEURONDB_PORT          Database port (for embedding check, default: 5432)
    NEURONDB_DATABASE      Database name (for embedding check, default: neurondb)
    NEURONDB_USER          Database user (for embedding check, default: neurondb)
    NEURONDB_PASSWORD      Database password (for embedding check)

Examples:
    # Basic setup
    $SCRIPT_NAME

    # Setup with embedding configuration check
    $SCRIPT_NAME --check-embeddings

    # Verify only (skip installation)
    $SCRIPT_NAME --skip-install

EOF
			exit 0
			;;
		*)
			echo -e "${RED}Unknown option: $1${NC}" >&2
			echo "Use -h or --help for usage information" >&2
			exit 1
			;;
	esac
done

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_section() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
}

# Function to check Python 3
check_python() {
    print_section "Checking Python Installation"
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        # Check if it's Python 3
        if python --version 2>&1 | grep -q "Python 3"; then
            PYTHON_CMD="python"
        else
            print_error "Python 3 is required but only Python 2 found"
            return 1
        fi
    else
        print_error "Python 3 is not installed"
        print_info "Please install Python 3.8 or later:"
        print_info "  macOS: brew install python3"
        print_info "  Ubuntu/Debian: sudo apt-get install python3 python3-pip"
        print_info "  Windows: Download from https://www.python.org/downloads/"
        return 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    print_success "Found Python: $PYTHON_CMD ($PYTHON_VERSION)"
    
    # Check version (3.8+)
    MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
    
    if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 8 ]); then
        print_error "Python 3.8 or later is required (found $PYTHON_VERSION)"
        return 1
    fi
    
    # Check pip
    if $PYTHON_CMD -m pip --version &> /dev/null; then
        PIP_CMD="$PYTHON_CMD -m pip"
        print_success "Found pip"
    elif command -v pip3 &> /dev/null; then
        PIP_CMD="pip3"
        print_success "Found pip3"
    else
        print_error "pip is not installed"
        print_info "Please install pip:"
        print_info "  macOS: python3 -m ensurepip --upgrade"
        print_info "  Ubuntu/Debian: sudo apt-get install python3-pip"
        return 1
    fi
    
    return 0
}

# Function to check requirements file
check_requirements_file() {
    print_section "Checking Requirements File"
    
    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        print_error "Requirements file not found: $REQUIREMENTS_FILE"
        return 1
    fi
    
    print_success "Found requirements file: $REQUIREMENTS_FILE"
    
    if [ "$VERBOSE" = true ]; then
        print_info "Required packages:"
        grep -v "^#" "$REQUIREMENTS_FILE" | grep -v "^$" | while read -r line; do
            echo "  - $line"
        done
    fi
    
    return 0
}

# Function to install dependencies
install_dependencies() {
    print_section "Installing Python Dependencies"
    
    print_info "Installing packages from $REQUIREMENTS_FILE..."
    print_info "This may take a few minutes..."
    
    if [ "$VERBOSE" = true ]; then
        $PIP_CMD install -r "$REQUIREMENTS_FILE"
    else
        $PIP_CMD install -r "$REQUIREMENTS_FILE" --quiet --disable-pip-version-check
    fi
    
    if [ $? -eq 0 ]; then
        print_success "Dependencies installed successfully"
        return 0
    else
        print_error "Failed to install dependencies"
        print_info "Try running with --verbose to see detailed error messages"
        return 1
    fi
}

# Function to verify installation
verify_installation() {
    print_section "Verifying Installation"
    
    local all_ok=true
    
    # Check core packages
    print_info "Checking core packages..."
    
    if $PYTHON_CMD -c "import psycopg2" 2>/dev/null; then
        print_success "  ✓ psycopg2-binary"
    else
        print_error "  ✗ psycopg2-binary (required for database connectivity)"
        all_ok=false
    fi
    
    if $PYTHON_CMD -c "import pandas" 2>/dev/null; then
        print_success "  ✓ pandas"
    else
        print_error "  ✗ pandas (required for data processing)"
        all_ok=false
    fi
    
    if $PYTHON_CMD -c "from datasets import load_dataset" 2>/dev/null; then
        print_success "  ✓ datasets (required for Hugging Face dataset loading)"
    else
        print_error "  ✗ datasets (required for Hugging Face dataset loading)"
        all_ok=false
    fi
    
    if $PYTHON_CMD -c "import huggingface_hub" 2>/dev/null; then
        print_success "  ✓ huggingface-hub"
    else
        print_warning "  ⚠ huggingface-hub (optional but recommended)"
    fi
    
    if $PYTHON_CMD -c "import requests" 2>/dev/null; then
        print_success "  ✓ requests (optional but recommended)"
    else
        print_warning "  ⚠ requests (optional but recommended)"
    fi
    
    if $PYTHON_CMD -c "import boto3" 2>/dev/null; then
        print_success "  ✓ boto3 (optional, for S3 support)"
    else
        print_warning "  ⚠ boto3 (optional, for S3 support)"
    fi
    
    if $PYTHON_CMD -c "import pyarrow" 2>/dev/null; then
        print_success "  ✓ pyarrow (optional, for Parquet support)"
    else
        print_warning "  ⚠ pyarrow (optional, for Parquet support)"
    fi
    
    echo ""
    
    if [ "$all_ok" = true ]; then
        print_success "All required packages are installed"
        return 0
    else
        print_error "Some required packages are missing"
        print_info "Please install missing packages manually:"
        print_info "  $PIP_CMD install -r $REQUIREMENTS_FILE"
        return 1
    fi
}

# Function to check embedding configuration
check_embedding_config() {
    print_section "Checking Embedding Configuration"
    
    if ! command -v psql &> /dev/null; then
        print_warning "psql not found, skipping embedding configuration check"
        print_info "To check embedding configuration manually, run:"
        print_info "  psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c \"SHOW neurondb.llm_api_key;\""
        return 0
    fi
    
    print_info "Checking PostgreSQL embedding configuration..."
    
    # Set password if provided
    if [ -n "$NEURONDB_PASSWORD" ]; then
        export PGPASSWORD="$NEURONDB_PASSWORD"
    fi
    
    # Check if we can connect
    if ! psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1" > /dev/null 2>&1; then
        print_warning "Cannot connect to PostgreSQL, skipping embedding configuration check"
        print_info "Connection details: $DB_HOST:$DB_PORT/$DB_NAME as $DB_USER"
        return 0
    fi
    
    # Check NeuronDB extension
    if ! psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT 1 FROM pg_extension WHERE extname = 'neurondb'" | grep -q 1; then
        print_warning "NeuronDB extension not found"
        print_info "Install with: CREATE EXTENSION neurondb;"
        return 0
    fi
    
    print_success "NeuronDB extension is installed"
    
    # Check API key
    local api_key=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT current_setting('neurondb.llm_api_key', true)" 2>/dev/null || echo "")
    
    if [ -n "$api_key" ] && [ "$api_key" != "" ]; then
        print_success "  ✓ Embedding API key is configured"
    else
        print_warning "  ⚠ Embedding API key is not set"
        print_info "To set API key (for Hugging Face API):"
        print_info "  ALTER SYSTEM SET neurondb.llm_api_key = 'your-api-key';"
        print_info "  SELECT pg_reload_conf();"
        print_info ""
        print_info "Or use GUC variable:"
        print_info "  SET neurondb.llm_api_key = 'your-api-key';"
    fi
    
    # Check provider
    local provider=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT current_setting('neurondb.llm_provider', true)" 2>/dev/null || echo "huggingface")
    print_info "  Provider: ${provider:-huggingface (default)}"
    
    # Check endpoint
    local endpoint=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT current_setting('neurondb.llm_endpoint', true)" 2>/dev/null || echo "https://api-inference.huggingface.co")
    print_info "  Endpoint: ${endpoint:-https://api-inference.huggingface.co (default)}"
    
    # Check model
    local model=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT current_setting('neurondb.llm_model', true)" 2>/dev/null || echo "sentence-transformers/all-MiniLM-L6-v2")
    print_info "  Model: ${model:-sentence-transformers/all-MiniLM-L6-v2 (default)}"
    
    return 0
}

# Function to show next steps
show_next_steps() {
    print_section "Next Steps"
    
    echo "1. Configure Claude Desktop:"
    echo ""
    echo "   macOS:"
    echo "     mkdir -p ~/Library/Application\\ Support/Claude"
    echo "     cp $PROJECT_ROOT/conf/claude-desktop-config-macos.json \\"
    echo "        ~/Library/Application\\ Support/Claude/claude_desktop_config.json"
    echo ""
    echo "   Linux:"
    echo "     mkdir -p ~/.config/Claude"
    echo "     cp $PROJECT_ROOT/conf/claude-desktop-config-linux.json \\"
    echo "        ~/.config/Claude/claude_desktop_config.json"
    echo ""
    echo "   Windows:"
    echo "     Copy $PROJECT_ROOT/conf/claude-desktop-config-windows.json to"
    echo "     %APPDATA%\\Claude\\claude_desktop_config.json"
    echo ""
    echo "2. Edit the configuration file and update:"
    echo "   - Path to neurondb-mcp binary"
    echo "   - Database connection settings (NEURONDB_* environment variables)"
    echo ""
    echo "3. (Optional) Configure embedding API key in PostgreSQL:"
    echo "   ALTER SYSTEM SET neurondb.llm_api_key = 'your-api-key';"
    echo "   SELECT pg_reload_conf();"
    echo ""
    echo "4. Restart Claude Desktop"
    echo ""
    echo "For more information, see:"
    echo "  $PROJECT_ROOT/SETUP_GUIDE.md"
    echo ""
}

# Main execution
main() {
    echo "=========================================="
    echo "NeuronMCP Setup for Claude Desktop"
    echo "=========================================="
    echo ""
    echo "This script prepares your environment for NeuronMCP to work"
    echo "with Claude Desktop. Run this ONCE before configuring Claude Desktop."
    echo ""
    
    # Check Python
    if [ "$SKIP_PYTHON_CHECK" = false ]; then
        if ! check_python; then
            exit 1
        fi
    fi
    
    # Check requirements file
    if ! check_requirements_file; then
        exit 1
    fi
    
    # Install dependencies
    if [ "$SKIP_INSTALL" = false ]; then
        if ! install_dependencies; then
            exit 1
        fi
    fi
    
    # Verify installation
    if ! verify_installation; then
        exit 1
    fi
    
    # Check embedding configuration (optional)
    if [ "$CHECK_EMBEDDINGS" = true ]; then
        check_embedding_config
    fi
    
    # Show next steps
    show_next_steps
    
    print_success "Setup completed successfully!"
    echo ""
    print_info "You can now configure Claude Desktop to use NeuronMCP."
    echo ""
}

# Run main function
main "$@"

