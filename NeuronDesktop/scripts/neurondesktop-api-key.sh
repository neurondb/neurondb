#!/bin/bash
# ====================================================================
# NeuronDesktop API Key Generator
# ====================================================================
# Generates and displays API key for NeuronDesktop
# ====================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

# Configuration
DB_DSN="${DB_DSN:-postgresql://neurondesk:neurondesk@localhost:5432/neurondesk}"
USER_ID="${USER_ID:-nbduser}"
RATE_LIMIT="${RATE_LIMIT:-1000}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
	case $1 in
		-d|--dsn|--database-dsn)
			DB_DSN="$2"
			shift 2
			;;
		-u|--user|--user-id)
			USER_ID="$2"
			shift 2
			;;
		-r|--rate|--rate-limit)
			RATE_LIMIT="$2"
			shift 2
			;;
		-v|--verbose)
			VERBOSE=true
			shift
			;;
		-V|--version)
			echo "neurondesktop_api_key.sh version $VERSION"
			exit 0
			;;
		-h|--help)
			cat << EOF
NeuronDesktop API Key Generator

Usage:
    $SCRIPT_NAME [OPTIONS]

Description:
    Generates and displays API key for NeuronDesktop

Options:
    -d, --dsn, --database-dsn DSN    Database DSN (default: postgresql://neurondesk:neurondesk@localhost:5432/neurondesk)
    -u, --user, --user-id USER       User ID (default: nbduser)
    -r, --rate, --rate-limit LIMIT   Rate limit (default: 1000)
    -v, --verbose                    Enable verbose output
    -V, --version                    Show version information
    -h, --help                       Show this help message

Environment Variables:
    DB_DSN       Database DSN (default: postgresql://neurondesk:neurondesk@localhost:5432/neurondesk)
    USER_ID      User ID (default: nbduser)
    RATE_LIMIT   Rate limit (default: 1000)

Examples:
    # Basic usage
    $SCRIPT_NAME

    # Custom user and rate limit
    $SCRIPT_NAME -u myuser -r 5000

    # Custom database DSN
    $SCRIPT_NAME -d "postgresql://user:pass@localhost:5432/mydb"

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
	echo "NeuronDesktop API Key Generator"
	echo "========================================"
	echo "Database: $DB_DSN"
	echo "User: $USER_ID"
	echo "Rate Limit: $RATE_LIMIT"
	echo "========================================"
fi

echo "Generating API key for NeuronDesktop..."
if [ "$VERBOSE" = true ]; then
	echo "Database: $DB_DSN"
	echo "User: $USER_ID"
	echo ""
fi

cd "$SCRIPT_DIR/../api"

# Generate API key using Go
go run -exec "cd $(pwd)" << 'EOF'
package main

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/base64"
	"fmt"
	"os"
	"strings"
	"time"

	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/google/uuid"
	"golang.org/x/crypto/bcrypt"
)

func main() {
	dsn := os.Getenv("DB_DSN")
	if dsn == "" {
		dsn = "postgresql://neurondesk:neurondesk@localhost:5432/neurondesk"
	}
	userID := os.Getenv("USER_ID")
	if userID == "" {
		userID = "nbduser"
	}

	db, err := sql.Open("pgx", dsn)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to connect: %v\n", err)
		os.Exit(1)
	}
	defer db.Close()

	ctx := context.Background()
	if err := db.PingContext(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to ping database: %v\n", err)
		os.Exit(1)
	}

	// Generate API key
	keyBytes := make([]byte, 32)
	if _, err := rand.Read(keyBytes); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to generate key: %v\n", err)
		os.Exit(1)
	}
	apiKey := base64.URLEncoding.EncodeToString(keyBytes)

	// Hash the key for storage
	hashedKey, err := bcrypt.GenerateFromPassword([]byte(apiKey), bcrypt.DefaultCost)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to hash key: %v\n", err)
		os.Exit(1)
	}

	// Store in database
	keyID := uuid.New().String()
	rateLimit := 1000
	if rl := os.Getenv("RATE_LIMIT"); rl != "" {
		fmt.Sscanf(rl, "%d", &rateLimit)
	}

	query := `
		INSERT INTO neurondesk.api_keys (id, user_id, key_hash, rate_limit, created_at)
		VALUES ($1, $2, $3, $4, $5)
		ON CONFLICT (user_id) DO UPDATE SET
			key_hash = EXCLUDED.key_hash,
			rate_limit = EXCLUDED.rate_limit,
			updated_at = $5
	`
	_, err = db.ExecContext(ctx, query, keyID, userID, string(hashedKey), rateLimit, time.Now())
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to store key: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(apiKey)
}
EOF
