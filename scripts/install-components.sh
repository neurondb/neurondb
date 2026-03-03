#!/bin/bash
# ====================================================================
# Component Installation Script (NeuronDB repo)
# ====================================================================
# NeuronAgent, NeuronMCP, and NeuronDesktop are maintained in separate
# repositories: neuron-agent, neuron-mcp, neuron-desktop.
# Install them from those repos. This script only documents the layout.
# ====================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cat << EOF
NeuronDB multi-repo layout:

  neurondb       (this repo)  - PostgreSQL extension + Python SDK
  neuron-agent   - Agent runtime; install from neuron-agent repo
  neuron-mcp     - MCP server; install from neuron-mcp repo
  neuron-desktop - Web UI; install from neuron-desktop repo
  neuron-cloud   - SaaS control plane; install from neuron-cloud repo
  neuron-hub      - Agent builder; install from neuron-hub repo
  neuron-deploy   - Unified deploy (Docker, K8s, Terraform)

To install the extension only (this repo):
  cd $PROJECT_ROOT/NeuronDB && make install

To run full stack via Docker, use neuron-deploy:
  cd /path/to/neuron-deploy && docker compose -f local/docker-compose.yml up -d
EOF
