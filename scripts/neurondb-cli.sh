#!/bin/bash
#
# NeuronDB CLI Helper
# Command-line tool for common NeuronDB operations
#
# Usage:
#   ./neurondb-cli.sh COMMAND [SUBCOMMAND] [OPTIONS]
#
# Commands:
#   index       Index management (create, list, drop)
#   model       Model management (load, list)
#   quickstart  Load quickstart data pack
#
# This script is self-sufficient with no external dependencies.

set -euo pipefail

# Get script directory
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

# Default configuration
DB_HOST=""
DB_PORT=""
DB_NAME="neurondb"
DB_USER="neurondb"
DB_PASSWORD="neurondb"
USE_DOCKER=false
VERBOSE=false

# Functions
log_info() {
    echo -e "${CYAN}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

# Auto-detect Docker Compose setup
detect_docker() {
    if [[ -z "$DB_HOST" ]] && [[ -z "$DB_PORT" ]]; then
        if command -v docker &> /dev/null && (command -v docker-compose &> /dev/null || docker compose version &> /dev/null 2>&1); then
            if docker compose ps neurondb 2>/dev/null | grep -q "running\|healthy"; then
                USE_DOCKER=true
                return 0
            fi
        fi
    fi
    return 1
}

# Execute SQL command
execute_sql() {
    local sql="$1"
    
    if [[ "$USE_DOCKER" == "true" ]]; then
        docker compose exec -T neurondb psql -U "$DB_USER" -d "$DB_NAME" -c "$sql"
    else
        if [[ -z "$DB_HOST" ]]; then
            DB_HOST="localhost"
        fi
        if [[ -z "$DB_PORT" ]]; then
            DB_PORT="5433"
        fi
        export PGPASSWORD="$DB_PASSWORD"
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "$sql"
    fi
}

# Show help
show_help() {
    cat << EOF
${BOLD}NeuronDB CLI Helper${NC}

${BOLD}Usage:${NC}
    ${SCRIPT_NAME} COMMAND [SUBCOMMAND] [OPTIONS]

${BOLD}Commands:${NC}
    index create <table> <column> [OPTIONS]   Create vector index
    index list                                 List all vector indexes
    index drop <index_name>                    Drop an index
    model load <model_name>                    Load/prepare an embedding model
    model list                                 List available models
    quickstart                                 Load quickstart data pack

${BOLD}Index Options:${NC}
    --type TYPE            Index type: hnsw or ivf (default: hnsw)
    --distance METRIC      Distance metric: l2, cosine, ip (default: cosine)
    --m VALUE              HNSW m parameter (default: 16)
    --ef_construction VAL  HNSW ef_construction parameter (default: 64)
    --lists VALUE          IVF lists parameter (default: 100)

${BOLD}Connection Options:${NC}
    -h, --host HOST        Database host (default: auto-detect)
    -p, --port PORT        Database port (default: auto-detect)
    -d, --database DB      Database name (default: neurondb)
    -U, --user USER        Database user (default: neurondb)
    -W, --password PASS    Database password (default: neurondb)

${BOLD}Global Options:${NC}
    -v, --verbose          Enable verbose output
    -h, --help             Show this help message

${BOLD}Examples:${NC}
    # Create HNSW index with default parameters
    ${SCRIPT_NAME} index create documents embedding

    # Create HNSW index with custom parameters
    ${SCRIPT_NAME} index create documents embedding --type hnsw --m 16 --ef_construction 200

    # Create IVF index
    ${SCRIPT_NAME} index create documents embedding --type ivf --lists 100

    # List all indexes
    ${SCRIPT_NAME} index list

    # Drop an index
    ${SCRIPT_NAME} index drop idx_documents_hnsw

    # Load quickstart data
    ${SCRIPT_NAME} quickstart

    # Use custom connection
    ${SCRIPT_NAME} index list -h localhost -p 5432 -d mydb -U postgres

EOF
}

# Parse connection options (modifies globals, shifts consumed args)
parse_connection_opts() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--host)
                DB_HOST="$2"
                USE_DOCKER=false
                shift 2
                ;;
            -p|--port)
                DB_PORT="$2"
                USE_DOCKER=false
                shift 2
                ;;
            -d|--database)
                DB_NAME="$2"
                shift 2
                ;;
            -U|--user)
                DB_USER="$2"
                shift 2
                ;;
            -W|--password)
                DB_PASSWORD="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                return 0  # Stop parsing, leave remaining args in $@
                ;;
        esac
    done
}

# Index create command
index_create() {
    local table=""
    local column=""
    local index_type="hnsw"
    local distance="cosine"
    local m=16
    local ef_construction=64
    local lists=100
    
    # Parse arguments (connection options already parsed, these are remaining)
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --type)
                index_type="$2"
                shift 2
                ;;
            --distance)
                distance="$2"
                shift 2
                ;;
            --m)
                m="$2"
                shift 2
                ;;
            --ef_construction)
                ef_construction="$2"
                shift 2
                ;;
            --lists)
                lists="$2"
                shift 2
                ;;
            -h|--host|-p|--port|-d|--database|-U|--user|-W|--password|-v|--verbose)
                # Connection options already parsed, skip
                shift
                ;;
            -*)
                shift
                ;;
            *)
                if [[ -z "$table" ]]; then
                    table="$1"
                elif [[ -z "$column" ]]; then
                    column="$1"
                fi
                shift
                ;;
        esac
    done
    
    if [[ -z "$table" ]] || [[ -z "$column" ]]; then
        log_error "Usage: index create <table> <column> [OPTIONS]"
        exit 1
    fi
    
    # Auto-detect Docker if not specified
    detect_docker
    
    # Validate table and column exist
    log_info "Validating table and column..."
    local table_exists
    table_exists=$(execute_sql "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '$table');" | grep -E 't|f' | head -1)
    
    if [[ "$table_exists" != "t" ]]; then
        log_error "Table '$table' does not exist"
        exit 1
    fi
    
    # Determine operator class
    local opclass
    case "$distance" in
        l2)
            opclass="vector_l2_ops"
            ;;
        cosine)
            opclass="vector_cosine_ops"
            ;;
        ip)
            opclass="vector_ip_ops"
            ;;
        *)
            log_error "Invalid distance metric: $distance (must be: l2, cosine, ip)"
            exit 1
            ;;
    esac
    
    # Generate index name
    local index_name="idx_${table}_${column}_${index_type}_${distance}"
    
    # Create index SQL
    local sql
    if [[ "$index_type" == "hnsw" ]]; then
        sql="CREATE INDEX IF NOT EXISTS $index_name ON $table USING hnsw ($column $opclass) WITH (m = $m, ef_construction = $ef_construction);"
    elif [[ "$index_type" == "ivf" ]]; then
        sql="CREATE INDEX IF NOT EXISTS $index_name ON $table USING ivf ($column $opclass) WITH (lists = $lists);"
    else
        log_error "Invalid index type: $index_type (must be: hnsw, ivf)"
        exit 1
    fi
    
    log_info "Creating $index_type index on $table.$column..."
    if [[ "$VERBOSE" == "true" ]]; then
        log_info "SQL: $sql"
    fi
    
    execute_sql "$sql"
    
    if [[ $? -eq 0 ]]; then
        log_success "Index '$index_name' created successfully"
    else
        log_error "Failed to create index"
        exit 1
    fi
}

# Index list command
index_list() {
    # Auto-detect Docker if not specified
    detect_docker
    
    log_info "Listing vector indexes..."
    
    local sql="SELECT schemaname, tablename, indexname, indexdef FROM pg_indexes WHERE indexdef LIKE '%hnsw%' OR indexdef LIKE '%ivf%' ORDER BY schemaname, tablename, indexname;"
    
    execute_sql "$sql"
}

# Index drop command
index_drop() {
    local index_name="$1"
    
    if [[ -z "$index_name" ]]; then
        log_error "Usage: index drop <index_name>"
        exit 1
    fi
    
    # Auto-detect Docker if not specified
    detect_docker
    
    log_info "Dropping index '$index_name'..."
    
    local sql="DROP INDEX IF EXISTS $index_name;"
    
    if [[ "$VERBOSE" == "true" ]]; then
        log_info "SQL: $sql"
    fi
    
    execute_sql "$sql"
    
    if [[ $? -eq 0 ]]; then
        log_success "Index '$index_name' dropped successfully"
    else
        log_error "Failed to drop index"
        exit 1
    fi
}

# Model load command (placeholder - actual implementation depends on NeuronDB model management)
model_load() {
    local model_name="$1"
    
    if [[ -z "$model_name" ]]; then
        log_error "Usage: model load <model_name>"
        exit 1
    fi
    
    log_warning "Model loading is not yet implemented. Use NeuronDB model management functions directly."
    log_info "Example SQL: SELECT neurondb_ensure_model('$model_name', 'public', 'huggingface', 'embedding', 'onnx', '/path/to/model', current_user);"
}

# Model list command (placeholder)
model_list() {
    log_warning "Model listing is not yet implemented. Query neurondb.models table directly."
    log_info "Example SQL: SELECT model_name, provider, model_type FROM neurondb.models;"
}

# Quickstart command
quickstart() {
    local quickstart_script="${PROJECT_ROOT}/src/examples/quickstart/load_quickstart.sh"
    
    if [[ ! -f "$quickstart_script" ]]; then
        log_error "Quickstart script not found: $quickstart_script"
        exit 1
    fi
    
    log_info "Loading quickstart data pack..."
    
    # Pass connection options to quickstart script if specified
    if [[ -n "$DB_HOST" ]] || [[ -n "$DB_PORT" ]]; then
        if [[ -n "$DB_HOST" ]]; then
            "$quickstart_script" -h "$DB_HOST" -p "${DB_PORT:-5433}" -d "$DB_NAME" -U "$DB_USER" -W "$DB_PASSWORD"
        else
            "$quickstart_script" -p "$DB_PORT" -d "$DB_NAME" -U "$DB_USER" -W "$DB_PASSWORD"
        fi
    else
        "$quickstart_script"
    fi
}

# Main command router
main() {
    # Get command (before parsing connection options)
    local command="${1:-}"
    
    if [[ -z "$command" ]] || [[ "$command" == "-h" ]] || [[ "$command" == "--help" ]]; then
        show_help
        exit 0
    fi
    
    shift
    
    # Parse connection options (they can appear anywhere)
    parse_connection_opts "$@"
    # Remaining args are still in $@ after parse_connection_opts returns
    
    case "$command" in
        index)
            local subcommand="${1:-}"
            if [[ -z "$subcommand" ]]; then
                log_error "No index subcommand specified"
                log_error "Use: index create|list|drop"
                exit 1
            fi
            shift
            parse_connection_opts "$@"  # Parse again for remaining options
            case "$subcommand" in
                create)
                    index_create "$@"
                    ;;
                list)
                    index_list
                    ;;
                drop)
                    index_drop "$@"
                    ;;
                *)
                    log_error "Unknown index subcommand: $subcommand"
                    log_error "Use: index create|list|drop"
                    exit 1
                    ;;
            esac
            ;;
        model)
            local subcommand="${1:-}"
            if [[ -z "$subcommand" ]]; then
                log_error "No model subcommand specified"
                log_error "Use: model load|list"
                exit 1
            fi
            shift
            parse_connection_opts "$@"  # Parse again for remaining options
            case "$subcommand" in
                load)
                    model_load "$@"
                    ;;
                list)
                    model_list
                    ;;
                *)
                    log_error "Unknown model subcommand: $subcommand"
                    log_error "Use: model load|list"
                    exit 1
                    ;;
            esac
            ;;
        quickstart)
            parse_connection_opts "$@"  # Parse again for remaining options
            quickstart
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main
main "$@"

