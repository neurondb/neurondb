#!/bin/bash
# ====================================================================
# NeuronAgent Migration Runner
# ====================================================================
# Runs database migrations for NeuronAgent in order
# ====================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIGRATIONS_DIR="$SCRIPT_DIR/../sql"
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

# Database configuration
DB_NAME="${DB_NAME:-neurondb}"
DB_USER="${DB_USER:-postgres}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"

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
		-v|--verbose)
			VERBOSE=true
			shift
			;;
		-V|--version)
			echo "neuronagent_migrate.sh version $VERSION"
			exit 0
			;;
		-h|--help)
			cat << EOF
NeuronAgent Migration Runner

Usage:
    $SCRIPT_NAME [OPTIONS]

Description:
    Runs database migrations for NeuronAgent in order

Options:
    -D, --database DB    Database name (default: neurondb)
    -U, --user USER      Database user (default: postgres)
    -H, --host HOST      Database host (default: localhost)
    -p, --port PORT      Database port (default: 5432)
    -v, --verbose        Enable verbose output
    -V, --version        Show version information
    -h, --help           Show this help message

Environment Variables:
    DB_NAME     Database name (default: neurondb)
    DB_USER     Database user (default: postgres)
    DB_HOST     Database host (default: localhost)
    DB_PORT     Database port (default: 5432)

Examples:
    # Basic usage
    $SCRIPT_NAME

    # Custom database
    $SCRIPT_NAME -D mydb -U myuser

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
	echo "NeuronAgent Migration Runner"
	echo "========================================"
	echo "Database: $DB_HOST:$DB_PORT/$DB_NAME"
	echo "User: $DB_USER"
	echo "Migrations Directory: $MIGRATIONS_DIR"
	echo "========================================"
fi

echo "Running migrations..."

# Run migrations in order
for migration in "$MIGRATIONS_DIR"/*.sql; do
    if [ -f "$migration" ]; then
        if [ "$VERBOSE" = true ]; then
            echo "Running $(basename "$migration")..."
        fi
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$migration"
    fi
done

echo "Migrations complete!"
