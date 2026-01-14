#!/bin/bash
#-------------------------------------------------------------------------
#
# neuronmcp-setup.sh
#    NeuronMCP Configuration Schema Setup
#
# Sets up comprehensive database schema for NeuronMCP with all configurations,
# tables, functions, and pre-populated defaults.
#
# Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
#
# IDENTIFICATION
#    NeuronMCP/scripts/neuronmcp-setup.sh
#
#-------------------------------------------------------------------------

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SQL_DIR="$PROJECT_ROOT/sql"
SCRIPT_NAME=$(basename "$0")

# Version
VERSION="2.0.0"

# Default values
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values (can be overridden by environment variables)
# Note: Default port 5433 matches Docker Compose setup
# For native PostgreSQL, set DB_PORT=5432 or NEURONDB_PORT=5432
DB_HOST="${NEURONDB_HOST:-${DB_HOST:-localhost}}"
DB_PORT="${NEURONDB_PORT:-${DB_PORT:-5433}}"  # Docker Compose default
DB_NAME="${NEURONDB_DATABASE:-${DB_NAME:-neurondb}}"
DB_USER="${NEURONDB_USER:-${DB_USER:-neurondb}}"  # Docker Compose default user
DB_PASSWORD="${NEURONDB_PASSWORD:-${DB_PASSWORD:-neurondb}}"  # Docker Compose default password

# Parse command line arguments
while [[ $# -gt 0 ]]; do
	case $1 in
		-D|--database)
			DB_NAME="$2"
			shift 2
			;;
		-U|--user)
			DB_USER="$2"
			shift 2
			;;
		-H|--host)
			DB_HOST="$2"
			shift 2
			;;
		-p|--port)
			DB_PORT="$2"
			shift 2
			;;
		--password)
			DB_PASSWORD="$2"
			shift 2
			;;
		-v|--verbose)
			VERBOSE=true
			shift
			;;
		-V|--version)
			echo "neuronmcp_setup.sh version $VERSION"
			exit 0
			;;
		-h|--help)
			cat << EOF
NeuronMCP Configuration Schema Setup

Usage:
    $SCRIPT_NAME [OPTIONS]

Description:
    Sets up comprehensive database schema for NeuronMCP with all configurations,
    tables, functions, and pre-populated defaults.

Options:
    -D, --database DB     Database name (default: neurondb)
    -U, --user USER       Database user (default: neurondb)
    -H, --host HOST       Database host (default: localhost)
    -p, --port PORT       Database port (default: 5433)
    --password PASSWORD   Database password
    -v, --verbose         Enable verbose output
    -V, --version         Show version information
    -h, --help            Show this help message

Environment Variables:
    NEURONDB_HOST / DB_HOST       Database host (default: localhost)
    NEURONDB_PORT / DB_PORT       Database port (default: 5433)
    NEURONDB_DATABASE / DB_NAME   Database name (default: neurondb)
    NEURONDB_USER / DB_USER       Database user (default: neurondb)
    NEURONDB_PASSWORD / DB_PASSWORD   Database password

Examples:
    # Basic usage (Docker Compose defaults)
    $SCRIPT_NAME

    # Native PostgreSQL
    $SCRIPT_NAME -p 5432 -U postgres

    # Custom database
    $SCRIPT_NAME -D mydb -U myuser -H localhost -p 5432

    # With verbose output
    $SCRIPT_NAME --verbose

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

if [ "$VERBOSE" = true ]; then
	echo "========================================"
	echo "NeuronMCP Configuration Schema Setup"
	echo "========================================"
	echo "Database: $DB_HOST:$DB_PORT/$DB_NAME"
	echo "User: $DB_USER"
	echo "========================================"
	echo ""
fi

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

# Function to check if PostgreSQL is accessible
check_postgres() {
    print_info "Checking PostgreSQL connection..."
    
    if [ -n "$DB_PASSWORD" ]; then
        export PGPASSWORD="$DB_PASSWORD"
    fi
    
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1" > /dev/null 2>&1; then
        print_success "Connected to PostgreSQL"
        return 0
    else
        print_error "Cannot connect to PostgreSQL"
        print_info "Connection details:"
        print_info "  Host: $DB_HOST"
        print_info "  Port: $DB_PORT"
        print_info "  Database: $DB_NAME"
        print_info "  User: $DB_USER"
        return 1
    fi
}

# Function to check if NeuronDB extension exists
check_neurondb_extension() {
    print_info "Checking NeuronDB extension..."
    
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT 1 FROM pg_extension WHERE extname = 'neurondb'" | grep -q 1; then
        print_success "NeuronDB extension is installed"
        return 0
    else
        print_warning "NeuronDB extension not found. Attempting to create..."
        if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c \
            "CREATE EXTENSION IF NOT EXISTS neurondb;" 2>/dev/null; then
            print_success "NeuronDB extension created"
            return 0
        else
            print_error "Failed to create NeuronDB extension"
            print_info "Please ensure NeuronDB is installed: CREATE EXTENSION neurondb;"
            return 1
        fi
    fi
}

# Function to run SQL file
run_sql_file() {
    local sql_file="$1"
    local description="$2"
    
    if [ ! -f "$sql_file" ]; then
        print_error "SQL file not found: $sql_file"
        return 1
    fi
    
    print_info "Running $description..."
    
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$sql_file" > /dev/null 2>&1; then
        print_success "$description completed"
        return 0
    else
        print_error "Failed to run $description"
        print_info "Attempting to show error details..."
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$sql_file" 2>&1 | tail -20
        return 1
    fi
}

# Function to verify tables
verify_tables() {
    print_info "Verifying tables..."
    
    local tables=(
        "llm_providers"
        "llm_models"
        "llm_model_keys"
        "llm_model_configs"
        "llm_model_usage"
        "index_configs"
        "index_templates"
        "worker_configs"
        "worker_schedules"
        "ml_default_configs"
        "ml_model_templates"
        "tool_configs"
        "system_configs"
    )
    
    local all_exist=true
    for table in "${tables[@]}"; do
        if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
            "SELECT 1 FROM information_schema.tables WHERE table_schema = 'neurondb' AND table_name = '$table'" | grep -q 1; then
            print_success "  ✓ Table neurondb.$table exists"
        else
            print_error "  ✗ Table neurondb.$table missing"
            all_exist=false
        fi
    done
    
    return $([ "$all_exist" = true ] && echo 0 || echo 1)
}

# Function to verify views
verify_views() {
    print_info "Verifying views..."
    
    local views=(
        "v_llm_models_active"
        "v_llm_models_ready"
    )
    
    local all_exist=true
    for view in "${views[@]}"; do
        if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
            "SELECT 1 FROM information_schema.views WHERE table_schema = 'neurondb' AND table_name = '$view'" | grep -q 1; then
            print_success "  ✓ View neurondb.$view exists"
        else
            print_error "  ✗ View neurondb.$view missing"
            all_exist=false
        fi
    done
    
    return $([ "$all_exist" = true ] && echo 0 || echo 1)
}

# Function to verify functions
verify_functions() {
    print_info "Verifying functions..."
    
    local functions=(
        "neurondb_set_model_key"
        "neurondb_get_model_key"
        "neurondb_list_models"
        "neurondb_get_index_config"
        "neurondb_get_worker_config"
        "neurondb_get_ml_defaults"
        "neurondb_get_tool_config"
        "neurondb_get_system_config"
        "neurondb_get_all_configs"
    )
    
    local count=0
    for func in "${functions[@]}"; do
        if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
            "SELECT 1 FROM pg_proc WHERE proname = '$func' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb')" | grep -q 1; then
            ((count++))
        fi
    done
    
    if [ $count -eq ${#functions[@]} ]; then
        print_success "  ✓ All key functions exist ($count/${#functions[@]})"
        return 0
    else
        print_warning "  ⚠ Some functions missing ($count/${#functions[@]})"
        return 1
    fi
}

# Function to check pre-populated data
check_prepopulated_data() {
    print_info "Checking pre-populated data..."
    
    local provider_count=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT COUNT(*) FROM neurondb.llm_providers")
    local model_count=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT COUNT(*) FROM neurondb.llm_models")
    local template_count=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT COUNT(*) FROM neurondb.index_templates")
    local worker_count=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT COUNT(*) FROM neurondb.worker_configs")
    local ml_defaults_count=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT COUNT(*) FROM neurondb.ml_default_configs")
    local tool_count=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT COUNT(*) FROM neurondb.tool_configs")
    
    print_info "  Providers: $provider_count"
    print_info "  LLM Models: $model_count"
    print_info "  Index Templates: $template_count"
    print_info "  Worker Configs: $worker_count"
    print_info "  ML Defaults: $ml_defaults_count"
    print_info "  Tool Configs: $tool_count"
    
    if [ "$provider_count" -ge 5 ] && [ "$model_count" -ge 50 ]; then
        print_success "Pre-populated data looks good"
        return 0
    else
        print_warning "Some pre-populated data may be missing"
        return 1
    fi
}

# Function to show status summary
show_status() {
    print_info "Status Summary:"
    echo ""
    
    local ready_models=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT COUNT(*) FROM neurondb.v_llm_models_ready")
    local total_models=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT COUNT(*) FROM neurondb.llm_models")
    
    echo "  LLM Models: $ready_models/$total_models configured with API keys"
    echo ""
    echo "Next steps:"
    echo "  1. Set API keys: SELECT neurondb_set_model_key('model_name', 'api_key');"
    echo "  2. View active models: SELECT * FROM neurondb.v_llm_models_active;"
    echo "  3. View ready models: SELECT * FROM neurondb.v_llm_models_ready;"
    echo "  4. Get all configs: SELECT neurondb_get_all_configs();"
    echo ""
}

# Main execution
main() {
    echo "=========================================="
    echo "NeuronMCP Configuration Schema Setup"
    echo "=========================================="
    echo ""
    
    # Check prerequisites
    if ! check_postgres; then
        exit 1
    fi
    
    if ! check_neurondb_extension; then
        exit 1
    fi
    
    # Run schema SQL
    if ! run_sql_file "$SQL_DIR/neuronmcp_initial_schema.sql" "Schema setup"; then
        exit 1
    fi
    
    # Run functions SQL
    if ! run_sql_file "$SQL_DIR/neuronmcp_functions.sql" "Functions setup"; then
        exit 1
    fi
    
    # Verify installation
    echo ""
    print_info "Verifying installation..."
    echo ""
    
    verify_tables
    echo ""
    verify_views
    echo ""
    verify_functions
    echo ""
    check_prepopulated_data
    echo ""
    
    # Show status
    show_status
    
    print_success "NeuronMCP configuration schema setup completed!"
    echo ""
}

# Run main function
main "$@"



