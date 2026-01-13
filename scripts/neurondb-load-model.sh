#!/bin/bash
#
# NeuronDB Model Loading Helper
# ==============================
# Simplified CLI for loading and configuring embedding models
#
# Usage:
#   ./neurondb-load-model.sh [OPTIONS]
#
# Options:
#   --name NAME           Model name/identifier (required)
#   --model MODEL         HuggingFace model path (required, e.g., all-MiniLM-L6-v2)
#   --source SOURCE       Model source: huggingface (default: huggingface)
#   --config JSON         Additional model configuration as JSON
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

MODEL_NAME=""
MODEL_PATH=""
MODEL_SOURCE="huggingface"
MODEL_CONFIG="{}"
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
${BOLD}NeuronDB Model Loading Helper${NC}

${BOLD}Usage:${NC}
    $0 [OPTIONS]

${BOLD}Required Options:${NC}
    --name NAME           Model name/identifier (how you'll reference it)
    --model MODEL         HuggingFace model path (e.g., sentence-transformers/all-MiniLM-L6-v2)

${BOLD}Model Options:${NC}
    --source SOURCE       Model source (default: huggingface)
    --config JSON         Additional configuration as JSON string
                         Example: '{"batch_size": 32, "normalize": true}'

${BOLD}Database Options:${NC}
    --database DB         Database name (default: neurondb)
    --user USER           Database user (default: neurondb)
    --host HOST           Database host (default: localhost)
    --port PORT           Database port (default: 5433)

${BOLD}Other Options:${NC}
    --dry-run             Show SQL without executing
    -h, --help            Show this help message

${BOLD}Examples:${NC}
    # Load a HuggingFace model
    $0 --name mini_lm --model sentence-transformers/all-MiniLM-L6-v2

    # Load with custom configuration
    $0 --name mini_lm --model all-MiniLM-L6-v2 \\
       --config '{"batch_size": 64, "device": "cpu"}'

    # Load different model
    $0 --name mpnet --model sentence-transformers/all-mpnet-base-v2

${BOLD}Note:${NC}
    This script configures the model for use with embed_text() function.
    The model will be downloaded from HuggingFace on first use.
    NeuronDB must be configured with HuggingFace API access if needed.

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --source)
            MODEL_SOURCE="$2"
            shift 2
            ;;
        --config)
            MODEL_CONFIG="$2"
            shift 2
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
if [[ -z "$MODEL_NAME" || -z "$MODEL_PATH" ]]; then
    log_error "Model name and model path are required"
    show_help
    exit 1
fi

# Validate JSON config if provided
if [[ "$MODEL_CONFIG" != "{}" ]]; then
    if ! echo "$MODEL_CONFIG" | python3 -m json.tool > /dev/null 2>&1; then
        log_error "Invalid JSON configuration: $MODEL_CONFIG"
        exit 1
    fi
fi

# Build configuration JSON
FULL_CONFIG=$(cat <<EOF | python3 -c "import sys, json; print(json.dumps(json.load(sys.stdin)))" 2>/dev/null || echo "$MODEL_CONFIG"
{
    "model_path": "${MODEL_PATH}",
    "source": "${MODEL_SOURCE}",
    "model_name": "${MODEL_NAME}",
    "config": ${MODEL_CONFIG}
}
EOF
)

# Show configuration
log_info "Model loading configuration:"
echo "  Model name: ${MODEL_NAME}"
echo "  Model path: ${MODEL_PATH}"
echo "  Source: ${MODEL_SOURCE}"
echo "  Config: ${FULL_CONFIG}"
echo ""

# Dry run mode
if [[ "$DRY_RUN" == true ]]; then
    log_info "Dry run mode - SQL that would be executed:"
    echo ""
    echo "INSERT INTO neurondb.embedding_model_config (model_name, config_json)"
    echo "VALUES ('${MODEL_NAME}', '${FULL_CONFIG}'::jsonb)"
    echo "ON CONFLICT (model_name) DO UPDATE"
    echo "SET config_json = EXCLUDED.config_json, updated_at = CURRENT_TIMESTAMP;"
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

# Check if NeuronDB extension exists
log_info "Checking NeuronDB extension..."
if ! psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -tAc "SELECT 1 FROM pg_extension WHERE extname = 'neurondb';" | grep -q 1; then
    log_error "NeuronDB extension not found"
    log_error "Please install the extension first: CREATE EXTENSION neurondb;"
    exit 1
fi
log_success "NeuronDB extension found"

# Check if embedding_model_config table exists, create if not
log_info "Checking embedding_model_config table..."
if ! psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -tAc "SELECT 1 FROM information_schema.tables WHERE table_schema = 'neurondb' AND table_name = 'embedding_model_config';" | grep -q 1; then
    log_warning "embedding_model_config table not found, creating..."
    psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" <<EOF > /dev/null 2>&1 || true
CREATE TABLE IF NOT EXISTS neurondb.embedding_model_config (
    model_name text PRIMARY KEY,
    config_json jsonb NOT NULL,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);
EOF
    log_success "Table created"
else
    log_success "Table exists"
fi

# Check if model already exists
if psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -tAc "SELECT 1 FROM neurondb.embedding_model_config WHERE model_name = '${MODEL_NAME}';" | grep -q 1; then
    log_warning "Model '${MODEL_NAME}' already exists"
    read -p "Update existing model configuration? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Cancelled"
        exit 0
    fi
fi

# Insert or update model configuration
log_info "Registering model '${MODEL_NAME}'..."

# Escape JSON for SQL
ESCAPED_CONFIG=$(echo "$FULL_CONFIG" | sed "s/'/''/g")

SQL="INSERT INTO neurondb.embedding_model_config (model_name, config_json)
VALUES ('${MODEL_NAME}', '${ESCAPED_CONFIG}'::jsonb)
ON CONFLICT (model_name) 
DO UPDATE SET 
    config_json = EXCLUDED.config_json,
    updated_at = CURRENT_TIMESTAMP
RETURNING model_name, created_at, updated_at;"

if RESULT=$(psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -tAc "${SQL}" 2>&1); then
    log_success "Model registered successfully"
    
    # Show registered model info
    log_info "Model information:"
    psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -c "
        SELECT 
            model_name,
            config_json->>'model_path' AS model_path,
            config_json->>'source' AS source,
            created_at,
            updated_at
        FROM neurondb.embedding_model_config
        WHERE model_name = '${MODEL_NAME}';
    " 2>/dev/null || true
    
    echo ""
    log_success "Model loading complete!"
    log_info "You can now use this model with:"
    echo "  embed_text('your text', '${MODEL_NAME}')"
else
    log_error "Model registration failed: $RESULT"
    exit 1
fi



