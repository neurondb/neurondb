#!/bin/bash
# Start NeuronAgent server with proper configuration

cd "$(dirname "$0")"

export DB_HOST="${DB_HOST:-localhost}"
export DB_PORT="${DB_PORT:-5432}"
export DB_NAME="${DB_NAME:-neurondb}"
export DB_USER="${DB_USER:-pge}"
export DB_PASSWORD="${DB_PASSWORD:-}"
export SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
export SERVER_PORT="${SERVER_PORT:-8080}"

echo "Starting NeuronAgent server..."
echo "Database: $DB_USER@$DB_HOST:$DB_PORT/$DB_NAME"
echo "Server: $SERVER_HOST:$SERVER_PORT"

go run cmd/agent-server/main.go

