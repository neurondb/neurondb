#!/bin/bash
set -euo pipefail

# NeuronDB Ecosystem Network Setup Script
# Creates a shared Docker network for NeuronDB, NeuronAgent, and NeuronMCP
# This allows services to communicate via container names

NETWORK_NAME="neurondb-network"

echo "=========================================="
echo "NeuronDB Ecosystem Network Setup"
echo "=========================================="
echo ""

# Check if network already exists
if docker network ls | grep -q "^[a-f0-9]*[[:space:]]*${NETWORK_NAME}[[:space:]]"; then
    echo "✓ Network '${NETWORK_NAME}' already exists"
    echo ""
    echo "To connect existing containers to this network:"
    echo "  docker network connect ${NETWORK_NAME} neurondb-cpu"
    echo "  docker network connect ${NETWORK_NAME} neuronagent"
    echo "  docker network connect ${NETWORK_NAME} neurondb-mcp"
    echo ""
else
    echo "Creating shared Docker network '${NETWORK_NAME}'..."
    docker network create ${NETWORK_NAME}
    echo "✓ Network '${NETWORK_NAME}' created successfully"
    echo ""
fi

echo "Network Configuration:"
echo "  Name: ${NETWORK_NAME}"
echo "  Driver: bridge"
echo ""
echo "Next Steps:"
echo "1. Start NeuronDB:"
echo "   cd NeuronDB/docker && docker compose up -d neurondb"
echo ""
echo "2. Connect NeuronDB to network (if not already connected):"
echo "   docker network connect ${NETWORK_NAME} neurondb-cpu"
echo ""
echo "3. Configure NeuronAgent:"
echo "   cd NeuronAgent/docker"
echo "   cp .env.example .env"
echo "   # Edit .env: DB_HOST=neurondb-cpu, DB_PORT=5432"
echo "   docker compose up -d agent-server"
echo "   docker network connect ${NETWORK_NAME} neuronagent"
echo ""
echo "4. Configure NeuronMCP:"
echo "   cd NeuronMCP/docker"
echo "   cp .env.example .env"
echo "   # Edit .env: NEURONDB_HOST=neurondb-cpu, NEURONDB_PORT=5432"
echo "   docker compose up -d neurondb-mcp"
echo "   docker network connect ${NETWORK_NAME} neurondb-mcp"
echo ""
echo "=========================================="
echo "Network setup complete!"
echo "=========================================="

