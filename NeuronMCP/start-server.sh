#!/bin/bash

# Neurondb MCP Server Startup Script
# Configured for local PostgreSQL connection

export NEURONDB_HOST="localhost"
export NEURONDB_PORT="5432"
export NEURONDB_DATABASE="postgres"
export NEURONDB_USER="ndbuser"

echo "Starting Neurondb MCP Server..."
echo "Database: $NEURONDB_DATABASE@$NEURONDB_HOST:$NEURONDB_PORT"
echo "User: $NEURONDB_USER"
echo ""

cd "$(dirname "$0")"
node dist/index.js

