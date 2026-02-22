# NeuronDB Ecosystem Overview

The NeuronDB ecosystem consists of three products that can run together or separately:

| Product | Purpose | When to use |
|---------|---------|-------------|
| **neurondb** (this repo) | Core: PostgreSQL extension (vector/ML), NeuronAgent, NeuronMCP, NeuronDesktop | Self-hosted vector DB, agents, MCP, and desktop UI |
| **neurondb-cloud** | SaaS control plane: auth, tenants, provisioning, billing, metering | Multi-tenant managed service; provisions neurondb stacks per tenant |
| **neurondb-hub** | Agent builder and embed platform: create agents, paste widget to any site; backend uses NeuronAgent (and optionally Cloud) | Let users build agents and embed a chat widget; Hub backend talks to NeuronAgent and optionally to neurondb-cloud |

## Full platform (all three)

To run **neurondb**, **neurondb-cloud**, and **neurondb-hub** on a single machine with a shared Docker network:

1. Clone the three repos as siblings (e.g. `neurondb/`, `neurondb-cloud/`, `neurondb-hub/` under one parent).
2. From the **neurondb** repo, run:
   ```bash
   ./scripts/deploy-all.sh TARGET_HOST [LOCAL_BASE_DIR]
   ```
   Example: `./scripts/deploy-all.sh user@192.168.1.100`
3. Prerequisites: passwordless SSH to `TARGET_HOST`; Docker (and Docker Compose) on the remote. The script can install Docker if missing.

**Ports:** neurondb 5433, 8080, 8081, 3000 | neurondb-cloud 5435, 8083 | neurondb-hub 5434, 8084, 8085, 3001.

See [Component Integration](integration.md) for how the core neurondb components (NeuronDB, NeuronAgent, NeuronMCP, NeuronDesktop) connect. For Cloud provisioning and data plane, see the neurondb-cloud repo `docs/` (e.g. `data-plane.md`, `provisioning-flow.md`).
