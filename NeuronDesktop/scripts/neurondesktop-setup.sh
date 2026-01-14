#!/bin/bash
# ====================================================================
# NeuronDesktop Setup
# ====================================================================
# Unified setup script for NeuronDesktop
# Orchestrates database migrations, default profile creation, and sample agent setup
# ====================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEURONDESKTOP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT_NAME=$(basename "$0")

# Version
VERSION="2.0.0"

# Default values
VERBOSE=false

# Colors for output
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Database connection from environment variables
# Note: Defaults match Docker Compose setup
# For native PostgreSQL, override these variables
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5433}"  # Docker Compose default port
DB_NAME="${DB_NAME:-neurondesk}"
DB_USER="${DB_USER:-neurondb}"  # Docker Compose default user
DB_PASSWORD="${DB_PASSWORD:-neurondb}"  # Docker Compose default password

# Export for sub-scripts
export DB_HOST DB_PORT DB_NAME DB_USER DB_PASSWORD

# Parse command line arguments
while [[ $# -gt 0 ]]; do
	case $1 in
		-D|--database)
			DB_NAME="$2"
			export DB_NAME
			shift 2
			;;
		-U|--user)
			DB_USER="$2"
			export DB_USER
			shift 2
			;;
		-H|--host)
			DB_HOST="$2"
			export DB_HOST
			shift 2
			;;
		-p|--port)
			DB_PORT="$2"
			export DB_PORT
			shift 2
			;;
		--password)
			DB_PASSWORD="$2"
			export DB_PASSWORD
			shift 2
			;;
		-v|--verbose)
			VERBOSE=true
			shift
			;;
		-V|--version)
			echo "neurondesktop_setup.sh version $VERSION"
			exit 0
			;;
		-h|--help)
			cat << EOF
NeuronDesktop Setup

Usage:
    $SCRIPT_NAME [OPTIONS]

Description:
    Unified setup script for NeuronDesktop. Orchestrates database migrations,
    default profile creation, and sample agent setup.

Options:
    -D, --database DB     Database name (default: neurondesk)
    -U, --user USER       Database user (default: neurondb)
    -H, --host HOST       Database host (default: localhost)
    -p, --port PORT       Database port (default: 5433)
    --password PASSWORD   Database password
    -v, --verbose         Enable verbose output
    -V, --version         Show version information
    -h, --help            Show this help message

Environment Variables:
    DB_HOST       Database host (default: localhost)
    DB_PORT       Database port (default: 5433)
    DB_NAME       Database name (default: neurondesk)
    DB_USER       Database user (default: neurondb)
    DB_PASSWORD   Database password

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

# Print header
if [ "$VERBOSE" = true ]; then
	echo "========================================"
	echo "NeuronDesktop Setup"
	echo "========================================"
	echo "Database: $DB_HOST:$DB_PORT/$DB_NAME"
	echo "User: $DB_USER"
	echo "========================================"
	echo ""
fi

# Step 1: Check database connection
echo -e "${CYAN}Step 1: Checking database connection...${NC}"
if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Database connection successful${NC}"
else
    echo -e "${RED}✗ Cannot connect to database${NC}"
    echo -e "${YELLOW}Please ensure:${NC}"
    echo "  - Database exists: createdb $DB_NAME"
    echo "  - Connection parameters are correct"
    echo "  - Database is accessible"
    exit 1
fi
echo ""

# Step 2: Run database migrations
echo -e "${CYAN}Step 2: Running database migrations...${NC}"
MIGRATIONS_DIR="$NEURONDESKTOP_ROOT/api/migrations"

if [ -d "$MIGRATIONS_DIR" ]; then
    for migration in "$MIGRATIONS_DIR"/*.sql; do
        if [ -f "$migration" ]; then
            migration_name=$(basename "$migration")
            echo -e "${CYAN}  Running $migration_name...${NC}"
            if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$migration" >/dev/null 2>&1; then
                echo -e "${GREEN}  ✓ $migration_name completed${NC}"
            else
                echo -e "${YELLOW}  ⚠ $migration_name may have had errors (check if objects already exist)${NC}"
            fi
        fi
    done
    echo -e "${GREEN}✓ Migrations completed${NC}"
else
    echo -e "${YELLOW}⚠ Migrations directory not found: $MIGRATIONS_DIR${NC}"
fi
echo ""

# Step 3: Build NeuronMCP if needed
echo -e "${CYAN}Step 3: Checking NeuronMCP binary...${NC}"
PROJECT_ROOT="$(cd "$NEURONDESKTOP_ROOT/.." && pwd)"
MCP_DIR="$PROJECT_ROOT/NeuronMCP"
MCP_BINARY="$MCP_DIR/bin/neurondb-mcp"

if [ -f "$MCP_BINARY" ] && [ -x "$MCP_BINARY" ]; then
    echo -e "${GREEN}✓ NeuronMCP binary found: $MCP_BINARY${NC}"
elif [ -d "$MCP_DIR" ] && [ -f "$MCP_DIR/cmd/neurondb-mcp/main.go" ]; then
    echo -e "${CYAN}  Building NeuronMCP...${NC}"
    cd "$MCP_DIR"
    if make build 2>/dev/null || go build -o bin/neurondb-mcp ./cmd/neurondb-mcp 2>/dev/null; then
        if [ -f "$MCP_BINARY" ] && [ -x "$MCP_BINARY" ]; then
            echo -e "${GREEN}✓ NeuronMCP built successfully${NC}"
        else
            echo -e "${YELLOW}⚠ NeuronMCP build may have failed${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ Could not build NeuronMCP (Go may not be installed)${NC}"
    fi
    cd - >/dev/null
else
    echo -e "${YELLOW}⚠ NeuronMCP source not found. Profile will be created without MCP config.${NC}"
    echo -e "${YELLOW}  You can set NEURONMCP_BINARY_PATH to specify the binary location.${NC}"
fi
echo ""

# Step 4: Create default profile
echo -e "${CYAN}Step 4: Creating default profile...${NC}"
if [ -f "$SCRIPT_DIR/setup_default_profile.sh" ]; then
    bash "$SCRIPT_DIR/setup_default_profile.sh"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Default profile setup complete${NC}"
    else
        echo -e "${RED}✗ Default profile setup failed${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ neurondesktop_profile.sh not found${NC}"
    exit 1
fi
echo ""

# Step 5: Create sample NeuronAgent (optional)
echo -e "${CYAN}Step 5: Creating sample NeuronAgent (optional)...${NC}"
if [ -f "$SCRIPT_DIR/neurondesktop_create_agent.sh" ]; then
    # Check if NeuronAgent endpoint is configured
    if [ -n "${NEURONAGENT_ENDPOINT:-}" ] || curl -s -o /dev/null -w "%{http_code}" "http://localhost:8080/health" 2>/dev/null | grep -q "200"; then
        bash "$SCRIPT_DIR/neurondesktop_create_agent.sh"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Sample agent setup complete${NC}"
        else
            echo -e "${YELLOW}⚠ Sample agent creation skipped or failed (this is optional)${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ NeuronAgent not detected. Skipping sample agent creation.${NC}"
        echo -e "${YELLOW}  Set NEURONAGENT_ENDPOINT to create a sample agent.${NC}"
    fi
else
    echo -e "${YELLOW}⚠ neurondesktop_create_agent.sh not found${NC}"
fi
echo ""

# Step 6: Verify setup
echo -e "${CYAN}Step 6: Verifying setup...${NC}"

# Check profiles table
PROFILE_COUNT=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM profiles WHERE is_default = true;" 2>/dev/null | xargs || echo "0")
if [ "$PROFILE_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓ Default profile exists${NC}"
else
    echo -e "${RED}✗ No default profile found${NC}"
fi

# Check tables
TABLES=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_name IN ('profiles', 'api_keys', 'request_logs');" 2>/dev/null | xargs || echo "0")
if [ "$TABLES" -ge 3 ]; then
    echo -e "${GREEN}✓ Required tables exist${NC}"
else
    echo -e "${YELLOW}⚠ Some tables may be missing${NC}"
fi

echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Setup Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}NeuronDesktop is ready to use!${NC}"
echo ""
echo -e "${CYAN}Next steps:${NC}"
echo "  1. Start NeuronDesktop API:"
echo "     cd $NEURONDESKTOP_ROOT/api && go run cmd/server/main.go"
echo ""
echo "  2. Start NeuronDesktop Frontend:"
echo "     cd $NEURONDESKTOP_ROOT/frontend && npm run dev"
echo ""
echo "  3. Access NeuronDesktop at:"
echo "     http://localhost:3000"
echo ""
echo -e "${CYAN}Configuration:${NC}"
echo "  Database: $DB_NAME @ $DB_HOST:$DB_PORT"
echo "  Default profile: Created and configured"
if [ -n "${NEURONAGENT_ENDPOINT:-}" ]; then
    echo "  NeuronAgent: $NEURONAGENT_ENDPOINT"
fi
echo ""






