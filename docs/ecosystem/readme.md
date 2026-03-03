# NeuronDB Ecosystem Overview

The NeuronDB ecosystem consists of three products that can run together or separately:

| Product | Purpose | When to use |
|---------|---------|-------------|
| **neurondb** (this repo) | Core: PostgreSQL extension (vector/ML). Agent, MCP, Desktop are in neuron-agent, neuron-mcp, neuron-desktop repos | Self-hosted vector DB; add agent/MCP/desktop from their repos or use neuron-deploy |
| **neuron-cloud** | SaaS control plane: auth, tenants, provisioning, billing, metering | Multi-tenant managed service; provisions neurondb stacks per tenant |
| **neuron-hub** | Agent builder and embed platform: create agents, paste widget to any site; backend uses NeuronAgent (and optionally Cloud) | Let users build agents and embed a chat widget; Hub backend talks to NeuronAgent and optionally to neuron-cloud |

## Full platform (all three)

To run **neurondb**, **neuron-cloud**, and **neuron-hub** on a single machine with a shared Docker network:

1. Clone the three repos as siblings (e.g. under `pge/`: `neurondb/`, `neuron-cloud/`, `neuron-hub/`).  
   For the core ecosystem layout (neurondb, neuron-agent, neuron-desktop, neuron-mcp), see [Directory structure](../directory-structure.md).
2. From the **neurondb** repo, run:
   ```bash
   ./scripts/deploy-all.sh TARGET_HOST [LOCAL_BASE_DIR]
   ```
   Example: `./scripts/deploy-all.sh user@192.168.1.100`
3. Prerequisites: passwordless SSH to `TARGET_HOST`; Docker (and Docker Compose) on the remote. The script can install Docker if missing.

**Ports:** neurondb 5433, 8080, 8081, 3000 | neuron-cloud 5435, 8083 | neuron-hub 5434, 8084, 8085, 3001.

See [Component Integration](integration.md) for how the core neurondb components (NeuronDB, NeuronAgent, NeuronMCP, NeuronDesktop) connect. For Cloud provisioning and data plane, see the neuron-cloud repo `docs/` (e.g. `data-plane.md`, `provisioning-flow.md`).
