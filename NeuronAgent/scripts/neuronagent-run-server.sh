#!/bin/bash
# Start NeuronAgent server with proper configuration

cd "$(dirname "$0")"

# Database connection (defaults match Docker Compose setup)
# For native PostgreSQL, override these environment variables
export DB_HOST="${DB_HOST:-localhost}"
export DB_PORT="${DB_PORT:-5433}"  # Docker Compose default port
export DB_NAME="${DB_NAME:-neurondb}"
export DB_USER="${DB_USER:-neurondb}"  # Docker Compose default user
export DB_PASSWORD="${DB_PASSWORD:-neurondb}"  # Docker Compose default password
export SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
export SERVER_PORT="${SERVER_PORT:-8080}"

echo "Starting NeuronAgent server..."
echo "Database: $DB_USER@$DB_HOST:$DB_PORT/$DB_NAME"
echo "Server: $SERVER_HOST:$SERVER_PORT"

go run cmd/agent-server/main.go









