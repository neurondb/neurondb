#!/bin/bash
#
# NeuronDB Quickstart Data Pack Loader
# Loads sample data into NeuronDB for quick testing and learning
#
# Usage:
#   ./load_quickstart.sh [OPTIONS]
#
# Options:
#   -h, --host HOST       Database host (default: auto-detect)
#   -p, --port PORT       Database port (default: auto-detect)
#   -d, --database DB     Database name (default: neurondb)
#   -U, --user USER       Database user (default: neurondb)
#   -W, --password PASS   Database password (default: neurondb)
#   -f, --file FILE       SQL file path (default: quickstart_data.sql)
#   --help                Show this help message
#
# Examples:
#   ./load_quickstart.sh
#   ./load_quickstart.sh -h localhost -p 5432 -d mydb -U postgres
#

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME=$(basename "$0")

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Default configuration
DB_HOST=""
DB_PORT=""
DB_NAME="neurondb"
DB_USER="neurondb"
DB_PASSWORD="neurondb"
SQL_FILE="${SCRIPT_DIR}/quickstart_data.sql"
USE_DOCKER=false

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

show_help() {
    cat << EOF
${SCRIPT_NAME} - Load NeuronDB Quickstart Data Pack

Usage:
    ${SCRIPT_NAME} [OPTIONS]

Options:
    -h, --host HOST       Database host (default: auto-detect)
    -p, --port PORT       Database port (default: auto-detect)
    -d, --database DB     Database name (default: neurondb)
    -U, --user USER       Database user (default: neurondb)
    -W, --password PASS   Database password (default: neurondb)
    -f, --file FILE       SQL file path (default: quickstart_data.sql)
    --help                Show this help message

Examples:
    # Auto-detect Docker Compose setup
    ./load_quickstart.sh

    # Use native PostgreSQL
    ./load_quickstart.sh -h localhost -p 5432 -d mydb -U postgres

    # Use custom connection
    ./load_quickstart.sh -h db.example.com -p 5432 -d neurondb -U admin -W secret
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host)
            DB_HOST="$2"
            shift 2
            ;;
        -p|--port)
            DB_PORT="$2"
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
        -f|--file)
            SQL_FILE="$2"
            shift 2
            ;;
        --help)
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

# Check if SQL file exists
if [[ ! -f "$SQL_FILE" ]]; then
    log_error "SQL file not found: $SQL_FILE"
    exit 1
fi

# Auto-detect Docker Compose setup if host/port not specified
if [[ -z "$DB_HOST" ]] && [[ -z "$DB_PORT" ]]; then
    # Check if docker compose is available and neurondb service is running
    if command -v docker &> /dev/null && (command -v docker-compose &> /dev/null || docker compose version &> /dev/null 2>&1); then
        # Check if neurondb service exists and is running
        if docker compose ps neurondb 2>/dev/null | grep -q "running\|healthy"; then
            USE_DOCKER=true
            DB_HOST=""
            DB_PORT=""
            log_info "Detected Docker Compose setup"
        fi
    fi
fi

# Execute SQL
log_info "Loading quickstart data pack..."
echo ""

if [[ "$USE_DOCKER" == "true" ]]; then
    # Use docker compose exec
    docker compose exec -T neurondb psql -U "$DB_USER" -d "$DB_NAME" < "$SQL_FILE"
    if [[ $? -eq 0 ]]; then
        log_success "Quickstart data pack loaded successfully!"
        echo ""
        log_info "You can now query the quickstart_documents table:"
        echo "  docker compose exec neurondb psql -U $DB_USER -d $DB_NAME -c \"SELECT COUNT(*) FROM quickstart_documents;\""
    else
        log_error "Failed to load quickstart data pack"
        exit 1
    fi
else
    # Use psql directly
    if [[ -z "$DB_HOST" ]]; then
        DB_HOST="localhost"
    fi
    if [[ -z "$DB_PORT" ]]; then
        DB_PORT="5433"  # Default Docker Compose port
    fi

    export PGPASSWORD="$DB_PASSWORD"
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$SQL_FILE"
    
    if [[ $? -eq 0 ]]; then
        log_success "Quickstart data pack loaded successfully!"
        echo ""
        log_info "You can now query the quickstart_documents table:"
        echo "  psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c \"SELECT COUNT(*) FROM quickstart_documents;\""
    else
        log_error "Failed to load quickstart data pack"
        exit 1
    fi
fi



