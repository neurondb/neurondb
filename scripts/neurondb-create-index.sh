#!/bin/bash
#
# NeuronDB Index Creation Helper
# ===============================
# Simplified CLI for creating HNSW and IVF vector indexes
#
# Usage:
#   ./neurondb-create-index.sh [OPTIONS]
#
# Options:
#   --table TABLE         Table name (required)
#   --column COLUMN       Vector column name (required)
#   --type TYPE           Index type: hnsw or ivf (default: hnsw)
#   --metric METRIC       Distance metric: cosine, l2, or ip (default: cosine)
#   --name NAME           Index name (auto-generated if not specified)
#   --m M                 HNSW m parameter (default: 16)
#   --ef-construction EF  HNSW ef_construction parameter (default: 64)
#   --lists LISTS         IVF lists parameter (default: 100)
#   --concurrent          Create index concurrently (non-blocking)
#   --if-not-exists       Only create if index doesn't exist
#   --database DB         Database name (default: neurondb)
#   --user USER           Database user (default: neurondb)
#   --host HOST           Database host (default: localhost)
#   --port PORT           Database port (default: 5433)
#   --dry-run             Show SQL without executing
#   -h, --help            Show this help message
#

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
DB_HOST="${PGHOST:-localhost}"
DB_PORT="${PGPORT:-5433}"
DB_NAME="${PGDATABASE:-neurondb}"
DB_USER="${PGUSER:-neurondb}"
DB_PASSWORD="${PGPASSWORD:-neurondb}"

TABLE=""
COLUMN=""
INDEX_TYPE="hnsw"
METRIC="cosine"
INDEX_NAME=""
M_PARAM=16
EF_CONSTRUCTION=64
LISTS_PARAM=100
CONCURRENT=false
IF_NOT_EXISTS=false
DRY_RUN=false

# Logging functions
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

show_help() {
    cat << EOF
${BOLD}NeuronDB Index Creation Helper${NC}

${BOLD}Usage:${NC}
    $0 [OPTIONS]

${BOLD}Required Options:${NC}
    --table TABLE         Table name
    --column COLUMN       Vector column name

${BOLD}Index Options:${NC}
    --type TYPE           Index type: hnsw or ivf (default: hnsw)
    --metric METRIC       Distance metric: cosine, l2, or ip (default: cosine)
    --name NAME           Custom index name (auto-generated if not specified)
    --m M                 HNSW m parameter (default: 16, range: 4-64)
    --ef-construction EF  HNSW ef_construction (default: 64, range: 4-1000)
    --lists LISTS         IVF lists parameter (default: 100)
    --concurrent          Create index concurrently (non-blocking, slower)
    --if-not-exists       Only create if index doesn't exist

${BOLD}Database Options:${NC}
    --database DB         Database name (default: neurondb)
    --user USER           Database user (default: neurondb)
    --host HOST           Database host (default: localhost)
    --port PORT           Database port (default: 5433)

${BOLD}Other Options:${NC}
    --dry-run             Show SQL without executing
    -h, --help            Show this help message

${BOLD}Examples:${NC}
    # Create HNSW index (default)
    $0 --table documents --column embedding

    # Create HNSW index with custom parameters
    $0 --table documents --column embedding --m 32 --ef-construction 128

    # Create IVF index
    $0 --table documents --column embedding --type ivf --lists 50

    # Create index for L2 distance
    $0 --table documents --column embedding --metric l2

    # Concurrent index creation
    $0 --table documents --column embedding --concurrent

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --table)
            TABLE="$2"
            shift 2
            ;;
        --column)
            COLUMN="$2"
            shift 2
            ;;
        --type)
            INDEX_TYPE="$2"
            if [[ "$INDEX_TYPE" != "hnsw" && "$INDEX_TYPE" != "ivf" ]]; then
                log_error "Index type must be 'hnsw' or 'ivf'"
                exit 1
            fi
            shift 2
            ;;
        --metric)
            METRIC="$2"
            if [[ "$METRIC" != "cosine" && "$METRIC" != "l2" && "$METRIC" != "ip" ]]; then
                log_error "Metric must be 'cosine', 'l2', or 'ip'"
                exit 1
            fi
            shift 2
            ;;
        --name)
            INDEX_NAME="$2"
            shift 2
            ;;
        --m)
            M_PARAM="$2"
            shift 2
            ;;
        --ef-construction)
            EF_CONSTRUCTION="$2"
            shift 2
            ;;
        --lists)
            LISTS_PARAM="$2"
            shift 2
            ;;
        --concurrent)
            CONCURRENT=true
            shift
            ;;
        --if-not-exists)
            IF_NOT_EXISTS=true
            shift
            ;;
        --database)
            DB_NAME="$2"
            shift 2
            ;;
        --user)
            DB_USER="$2"
            shift 2
            ;;
        --host)
            DB_HOST="$2"
            shift 2
            ;;
        --port)
            DB_PORT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$TABLE" || -z "$COLUMN" ]]; then
    log_error "Table and column are required"
    show_help
    exit 1
fi

# Generate index name if not provided
if [[ -z "$INDEX_NAME" ]]; then
    INDEX_NAME="${TABLE}_${COLUMN}_${INDEX_TYPE}_idx"
fi

# Determine operator class based on metric
case "$METRIC" in
    cosine)
        OP_CLASS="vector_cosine_ops"
        ;;
    l2)
        OP_CLASS="vector_l2_ops"
        ;;
    ip)
        OP_CLASS="vector_ip_ops"
        ;;
esac

# Build CREATE INDEX SQL
IF_EXISTS_CLAUSE=""
if [[ "$IF_NOT_EXISTS" == true ]]; then
    IF_EXISTS_CLAUSE="IF NOT EXISTS"
fi

CONCURRENT_CLAUSE=""
if [[ "$CONCURRENT" == true ]]; then
    CONCURRENT_CLAUSE="CONCURRENTLY"
fi

if [[ "$INDEX_TYPE" == "hnsw" ]]; then
    CREATE_SQL="CREATE INDEX $IF_EXISTS_CLAUSE $CONCURRENT_CLAUSE ${INDEX_NAME}
ON ${TABLE} USING hnsw (${COLUMN} ${OP_CLASS})
WITH (m = ${M_PARAM}, ef_construction = ${EF_CONSTRUCTION});"
else
    CREATE_SQL="CREATE INDEX $IF_EXISTS_CLAUSE $CONCURRENT_CLAUSE ${INDEX_NAME}
ON ${TABLE} USING ivfflat (${COLUMN} ${OP_CLASS})
WITH (lists = ${LISTS_PARAM});"
fi

# Show configuration
log_info "Index creation configuration:"
echo "  Table: ${TABLE}"
echo "  Column: ${COLUMN}"
echo "  Index name: ${INDEX_NAME}"
echo "  Type: ${INDEX_TYPE}"
echo "  Metric: ${METRIC}"
if [[ "$INDEX_TYPE" == "hnsw" ]]; then
    echo "  m: ${M_PARAM}"
    echo "  ef_construction: ${EF_CONSTRUCTION}"
else
    echo "  lists: ${LISTS_PARAM}"
fi
echo "  Concurrent: ${CONCURRENT}"
echo ""

# Dry run mode
if [[ "$DRY_RUN" == true ]]; then
    log_info "Dry run mode - SQL that would be executed:"
    echo ""
    echo "$CREATE_SQL"
    exit 0
fi

# Export password for psql
export PGPASSWORD="${DB_PASSWORD}"

# Check if psql is available
if ! command -v psql &> /dev/null; then
    log_error "psql is not installed or not in PATH"
    exit 1
fi

# Test database connection
log_info "Testing database connection..."
if ! psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -c "SELECT 1;" > /dev/null 2>&1; then
    log_error "Cannot connect to database"
    log_error "Check connection parameters: ${DB_USER}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
    exit 1
fi
log_success "Database connection successful"

# Verify table exists
log_info "Verifying table exists..."
if ! psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -tAc "SELECT 1 FROM pg_tables WHERE tablename = '${TABLE}';" | grep -q 1; then
    log_error "Table '${TABLE}' does not exist"
    exit 1
fi
log_success "Table '${TABLE}' found"

# Verify column exists
log_info "Verifying column exists..."
if ! psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -tAc "SELECT 1 FROM information_schema.columns WHERE table_name = '${TABLE}' AND column_name = '${COLUMN}';" | grep -q 1; then
    log_error "Column '${COLUMN}' does not exist in table '${TABLE}'"
    exit 1
fi
log_success "Column '${COLUMN}' found"

# Check if index already exists
if psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -tAc "SELECT 1 FROM pg_indexes WHERE indexname = '${INDEX_NAME}';" | grep -q 1; then
    if [[ "$IF_NOT_EXISTS" == false ]]; then
        log_warning "Index '${INDEX_NAME}' already exists"
        read -p "Drop existing index and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Dropping existing index..."
            psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -c "DROP INDEX ${INDEX_NAME};" > /dev/null
            log_success "Index dropped"
        else
            log_info "Cancelled"
            exit 0
        fi
    else
        log_info "Index '${INDEX_NAME}' already exists, skipping (--if-not-exists)"
        exit 0
    fi
fi

# Create index
log_info "Creating index '${INDEX_NAME}'..."
log_warning "This may take a while depending on data size..."

if psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -c "${CREATE_SQL}" > /dev/null 2>&1; then
    log_success "Index created successfully"
    
    # Verify index
    log_info "Verifying index..."
    INDEX_SIZE=$(psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -tAc "SELECT pg_size_pretty(pg_relation_size('${INDEX_NAME}'::regclass));" 2>/dev/null || echo "unknown")
    log_success "Index verified (size: ${INDEX_SIZE})"
    
    echo ""
    log_success "Index creation complete!"
    echo "  Index: ${INDEX_NAME}"
    echo "  Type: ${INDEX_TYPE}"
    echo "  Size: ${INDEX_SIZE}"
else
    log_error "Index creation failed"
    exit 1
fi



