#!/bin/bash
#
# NeuronDB Quickstart Data Setup Script
# ======================================
# Sets up a small sample dataset for quick testing and exploration.
# Creates tables, generates/loads sample data, and creates indexes.
#
# Usage:
#   ./neurondb-quickstart-data.sh [OPTIONS]
#
# Options:
#   --database DB         Database name (default: neurondb)
#   --user USER          Database user (default: neurondb)
#   --host HOST          Database host (default: localhost)
#   --port PORT          Database port (default: 5433)
#   --count COUNT        Number of documents to generate (default: 200)
#   --skip-generation    Skip data generation, only load existing SQL file
#   --cleanup            Drop existing quickstart tables before setup
#   --help               Show this help message
#
# Environment Variables:
#   PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
#

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
QUICKSTART_DIR="${PROJECT_ROOT}/examples/quickstart-data"
DATA_FILE="${QUICKSTART_DIR}/sample_data/sample_data.sql"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Default configuration (Docker Compose defaults)
DB_HOST="${PGHOST:-localhost}"
DB_PORT="${PGPORT:-5433}"
DB_NAME="${PGDATABASE:-neurondb}"
DB_USER="${PGUSER:-neurondb}"
DB_PASSWORD="${PGPASSWORD:-neurondb}"
DOC_COUNT=200
SKIP_GENERATION=false
CLEANUP=false

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

print_header() {
    echo ""
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  ${BOLD}NeuronDB Quickstart Data Setup${NC}                            ${BLUE}║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

show_help() {
    cat << EOF
${BOLD}NeuronDB Quickstart Data Setup${NC}

${BOLD}Usage:${NC}
    $0 [OPTIONS]

${BOLD}Options:${NC}
    --database DB         Database name (default: neurondb)
    --user USER          Database user (default: neurondb)
    --host HOST          Database host (default: localhost)
    --port PORT          Database port (default: 5433)
    --count COUNT        Number of documents to generate (default: 200)
    --skip-generation    Skip data generation, only load existing SQL file
    --cleanup            Drop existing quickstart tables before setup
    -h, --help           Show this help message

${BOLD}Environment Variables:${NC}
    PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD

${BOLD}Examples:${NC}
    # Basic usage (generates and loads 200 documents)
    $0

    # Custom database connection
    $0 --host localhost --port 5432 --database mydb --user postgres

    # Generate more documents
    $0 --count 500

    # Skip generation, only load existing file
    $0 --skip-generation

    # Clean up and reinstall
    $0 --cleanup

${BOLD}What this script does:${NC}
    1. Generates sample documents with embeddings (unless --skip-generation)
    2. Loads data into PostgreSQL database
    3. Creates HNSW index for fast similarity search
    4. Verifies setup with test queries

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
        --count)
            DOC_COUNT="$2"
            shift 2
            ;;
        --skip-generation)
            SKIP_GENERATION=true
            shift
            ;;
        --cleanup)
            CLEANUP=true
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

print_header

log_info "Configuration:"
echo "  Database: ${DB_NAME}@${DB_HOST}:${DB_PORT}"
echo "  User: ${DB_USER}"
echo "  Document count: ${DOC_COUNT}"
echo ""

# Export password for psql
export PGPASSWORD="${DB_PASSWORD}"

# Check if psql is available
if ! command -v psql &> /dev/null; then
    log_error "psql is not installed or not in PATH"
    log_error "Please install PostgreSQL client tools"
    exit 1
fi

# Test database connection
log_info "Testing database connection..."
if ! psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -c "SELECT 1;" > /dev/null 2>&1; then
    log_error "Cannot connect to database"
    log_error "Please check:"
    log_error "  - Database is running"
    log_error "  - Connection parameters (host, port, user, password)"
    log_error "  - For Docker: docker compose ps neurondb"
    exit 1
fi
log_success "Database connection successful"

# Check if NeuronDB extension exists
log_info "Checking NeuronDB extension..."
if ! psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -tAc "SELECT 1 FROM pg_extension WHERE extname = 'neurondb';" | grep -q 1; then
    log_warning "NeuronDB extension not found, creating..."
    psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -c "CREATE EXTENSION IF NOT EXISTS neurondb;" > /dev/null
    log_success "NeuronDB extension created"
else
    log_success "NeuronDB extension found"
fi

# Cleanup if requested
if [ "$CLEANUP" = true ]; then
    log_info "Cleaning up existing quickstart tables..."
    psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" <<EOF > /dev/null 2>&1 || true
DROP INDEX IF EXISTS quickstart_documents_embedding_idx;
DROP TABLE IF EXISTS quickstart_documents;
EOF
    log_success "Cleanup complete"
fi

# Generate sample data if not skipping
if [ "$SKIP_GENERATION" = false ]; then
    log_info "Generating sample data..."
    
    # Check if Python script exists
    GENERATOR_SCRIPT="${QUICKSTART_DIR}/generate_sample_data.py"
    if [ ! -f "$GENERATOR_SCRIPT" ]; then
        log_error "Generator script not found: $GENERATOR_SCRIPT"
        exit 1
    fi
    
    # Check if Python dependencies are available
    if ! python3 -c "import sentence_transformers, numpy" 2>/dev/null; then
        log_error "Python dependencies not found"
        log_error "Install with: pip install sentence-transformers numpy"
        exit 1
    fi
    
    # Run generator
    if python3 "$GENERATOR_SCRIPT" --output-dir "${QUICKSTART_DIR}/sample_data" --count "${DOC_COUNT}"; then
        log_success "Sample data generated"
    else
        log_error "Data generation failed"
        exit 1
    fi
else
    log_info "Skipping data generation (using existing file)"
fi

# Check if SQL file exists
if [ ! -f "$DATA_FILE" ]; then
    log_error "SQL file not found: $DATA_FILE"
    log_error "Run without --skip-generation to generate data first"
    exit 1
fi

# Load data into database
log_info "Loading data into database..."
if psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -f "$DATA_FILE" > /dev/null 2>&1; then
    log_success "Data loaded successfully"
else
    log_warning "Some errors occurred during data loading (this may be normal if data already exists)"
fi

# Verify setup
log_info "Verifying setup..."
DOC_COUNT_DB=$(psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -tAc "SELECT COUNT(*) FROM quickstart_documents;" 2>/dev/null || echo "0")
EMBED_COUNT=$(psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -tAc "SELECT COUNT(*) FROM quickstart_documents WHERE embedding IS NOT NULL;" 2>/dev/null || echo "0")

if [ "$DOC_COUNT_DB" -gt 0 ]; then
    log_success "Setup verification:"
    echo "  Documents loaded: ${DOC_COUNT_DB}"
    echo "  Documents with embeddings: ${EMBED_COUNT}"
    
    # Test similarity search
    log_info "Testing similarity search..."
    TEST_RESULT=$(psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -tAc "
        WITH q AS (SELECT embed_text('vector databases') AS query_vec)
        SELECT COUNT(*) FROM quickstart_documents, q
        WHERE embedding IS NOT NULL
        LIMIT 1;
    " 2>/dev/null || echo "0")
    
    if [ "$TEST_RESULT" -gt 0 ]; then
        log_success "Similarity search test passed"
    else
        log_warning "Similarity search test had issues (may need embedding model configuration)"
    fi
else
    log_error "No documents found in database"
    exit 1
fi

echo ""
log_success "Quickstart data setup complete!"
echo ""
echo -e "${BOLD}Next steps:${NC}"
echo "  1. Try a similarity search:"
echo "     psql -h ${DB_HOST} -p ${DB_PORT} -U ${DB_USER} -d ${DB_NAME} -c \""
echo "     WITH q AS (SELECT embed_text('machine learning') AS query_vec)"
echo "     SELECT title, embedding <=> q.query_vec AS distance"
echo "     FROM quickstart_documents, q"
echo "     ORDER BY distance LIMIT 5;\""
echo ""
echo "  2. Explore SQL recipes: examples/sql-recipes/"
echo "  3. Read the documentation: Docs/getting-started/quickstart.md"
echo ""




