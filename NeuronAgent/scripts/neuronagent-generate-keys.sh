#!/bin/bash
# ====================================================================
# NeuronAgent API Key Generator
# ====================================================================
# Generates API keys for NeuronAgent authentication
# ====================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$SCRIPT_DIR/../agent-server"
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

# Default parameters
ORGANIZATION_ID="default"
USER_ID="default"
RATE_LIMIT="60"
ROLES="user"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
	case $1 in
		-o|--org|--organization)
			ORGANIZATION_ID="$2"
			shift 2
			;;
		-u|--user)
			USER_ID="$2"
			shift 2
			;;
		-r|--rate|--rate-limit)
			RATE_LIMIT="$2"
			shift 2
			;;
		--roles)
			ROLES="$2"
			shift 2
			;;
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
			echo "neuronagent_generate_keys.sh version $VERSION"
			exit 0
			;;
		-h|--help)
			cat << EOF
NeuronAgent API Key Generator

Usage:
    $SCRIPT_NAME [OPTIONS] [ORGANIZATION_ID] [USER_ID] [RATE_LIMIT] [ROLES]

Description:
    Generates API keys for NeuronAgent authentication

Options:
    -o, --org, --organization ORG    Organization ID (default: default)
    -u, --user USER                  User ID (default: default)
    -r, --rate, --rate-limit LIMIT   Rate limit per minute (default: 60)
    --roles ROLES                    Comma-separated roles (default: user)
    -D, --database DB                Database name (default: neurondb)
    -U, --user USER                  Database user (default: postgres)
    -H, --host HOST                  Database host (default: localhost)
    -p, --port PORT                  Database port (default: 5432)
    -v, --verbose                    Enable verbose output
    -V, --version                    Show version information
    -h, --help                       Show this help message

Environment Variables:
    DB_NAME         Database name (default: neurondb)
    DB_USER         Database user (default: postgres)
    DB_HOST         Database host (default: localhost)
    DB_PORT         Database port (default: 5432)

Examples:
    # Basic usage with defaults
    $SCRIPT_NAME

    # Custom organization and user
    $SCRIPT_NAME -o myorg -u myuser

    # High rate limit for admin
    $SCRIPT_NAME --org admin --user admin --rate 1000 --roles admin,user

    # With verbose output
    $SCRIPT_NAME -v -o myorg -u myuser

EOF
			exit 0
			;;
		*)
			# Positional arguments for backward compatibility
			if [ "$ORGANIZATION_ID" = "default" ]; then
				ORGANIZATION_ID="$1"
			elif [ "$USER_ID" = "default" ]; then
				USER_ID="$1"
			elif [ "$RATE_LIMIT" = "60" ]; then
				RATE_LIMIT="$1"
			elif [ "$ROLES" = "user" ]; then
				ROLES="$1"
			else
				echo -e "${RED}Unknown option: $1${NC}" >&2
				echo "Use -h or --help for usage information" >&2
				exit 1
			fi
			shift
			;;
	esac
done

if [ "$VERBOSE" = true ]; then
	echo "========================================"
	echo "NeuronAgent API Key Generator"
	echo "========================================"
	echo "Organization: $ORGANIZATION_ID"
	echo "User: $USER_ID"
	echo "Rate Limit: $RATE_LIMIT/min"
	echo "Roles: $ROLES"
	echo "Database: $DB_HOST:$DB_PORT/$DB_NAME"
	echo "========================================"
fi

echo "Generating API key..."
if [ "$VERBOSE" = true ]; then
	echo "Organization: $ORGANIZATION_ID"
	echo "User: $USER_ID"
	echo "Rate Limit: $RATE_LIMIT/min"
	echo "Roles: $ROLES"
fi

# Generate key using Go program
go run -tags tools "$SCRIPT_DIR/../cmd/generate-key/main.go" \
    -org "$ORGANIZATION_ID" \
    -user "$USER_ID" \
    -rate "$RATE_LIMIT" \
    -roles "$ROLES" \
    -db-host "$DB_HOST" \
    -db-port "$DB_PORT" \
    -db-name "$DB_NAME" \
    -db-user "$DB_USER"
