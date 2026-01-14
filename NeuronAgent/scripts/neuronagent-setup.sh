#!/bin/bash
#-------------------------------------------------------------------------
#
# neuronagent-setup.sh
#    NeuronAgent Database Setup
#
# Sets up database and extension for NeuronAgent. Creates the database if
# it doesn't exist and installs the NeuronDB extension.
#
# Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
#
# IDENTIFICATION
#    NeuronAgent/scripts/neuronagent-setup.sh
#
#-------------------------------------------------------------------------

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT_NAME=$(basename "$0")

# Source common CLI library
source "${SCRIPT_DIR}/lib/neuronagent-cli.sh" || {
    echo "Error: Failed to load CLI library" >&2
    exit 1
}

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
			echo "neuronagent_setup.sh version $VERSION"
			exit 0
			;;
		-h|--help)
			cat << EOF
NeuronAgent Database Setup

Usage:
    $SCRIPT_NAME [OPTIONS]

Description:
    Sets up database and extension for NeuronAgent

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
	echo "NeuronAgent Database Setup"
	echo "========================================"
	echo "Database: $DB_HOST:$DB_PORT/$DB_NAME"
	echo "User: $DB_USER"
	echo "========================================"
fi

echo "Setting up NeuronAgent database..."

# Create database if it doesn't exist
if [ "$VERBOSE" = true ]; then
	echo "Checking if database exists..."
fi
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -tc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME'" | grep -q 1 || \
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "CREATE DATABASE $DB_NAME"

# Create extension
if [ "$VERBOSE" = true ]; then
	echo "Creating NeuronDB extension..."
fi
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS neurondb;"

echo "Database setup complete!"
