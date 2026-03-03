# Directory structure

This document describes the recommended layout for the NeuronDB ecosystem when working with multiple repositories.

## Canonical layout under `pge/`

When developing or deploying the full stack, use a single parent directory (e.g. `pge/`) with each component in **its own folder** (separate repos):

```
pge/
├── neurondb/          # This repo – core DB extension, dockers, docs, scripts
├── neuron-agent/      # NeuronAgent service (API, agents)
├── neuron-desktop/    # NeuronDesktop – web UI API + frontend
├── neuron-mcp/        # NeuronMCP server (MCP tools)
├── neuron-deploy/     # Full-stack deployment (compose, k8s, etc.)
├── neuron-cloud/      # Control plane (SaaS)
├── neuron-hub/        # Agent builder / embed platform
├── neuron-llm/       # LLM SQL training (neurondbpy-llm Python package)
├── neuron-www/       # Marketing website (Next.js)
└── dataset/          # Dataset utilities and assets
```

- **neurondb** – PostgreSQL extension, Docker images for DB/agent/MCP/desktop, documentation, Python SDK, benchmarks.
- **neuron-agent** – Agent server (lives in its own repo; run from `neuron-agent/` or via neuron-deploy).
- **neuron-desktop** – Desktop UI (API + frontend). For Docker builds that include the desktop, neurondb’s `docker/docker-compose.yml` expects this repo as a **sibling**: `../neuron-desktop` relative to the neurondb repo.
- **neuron-mcp** – MCP server (own repo).
- **neuron-deploy** – Use for full-stack orchestration (neurondb + neuron-agent + neuron-mcp + neuron-desktop) when all components are in sibling folders.

## Why separate folders?

- **Agent** and **desktop** each have their own repo under `pge/` (e.g. `neuron-agent/`, `neuron-desktop/`). They are not subdirectories inside `neurondb/`.
- Clear separation: one repo per product, easier CI/CD and versioning.
- neurondb’s root `docker-compose.yml` runs only the **neurondb** service (PostgreSQL extension). For the full ecosystem, use **neuron-deploy** or run `docker/docker-compose.yml` from neurondb with sibling repos present (see below).

## Using neurondb’s Docker Compose with desktop

The compose file at `docker/docker-compose.yml` can run the full stack (neurondb, neuronagent, neuronmcp, neurondesk-api, neurondesk-frontend). It expects **NeuronDesktop** to be available as a sibling repo:

- **Path:** `../neuron-desktop` (relative to the **neurondb** repo root).
- **From neurondb root:**
  ```bash
  cd /path/to/pge/neurondb
  docker compose -f docker/docker-compose.yml --profile default up -d
  ```
  This uses `../neuron-desktop` for schema file and for building the desktop API and frontend images (when that path exists).

If you use a different layout (e.g. monorepo or different parent path), set the environment variable **`NEURONDESKTOP_DIR`** to the path to the neuron-desktop repo (absolute or relative to where you run compose). The compose file uses this for build context and volume mounts when set.

## Summary

| Component       | Folder (under pge/)  | Purpose                          |
|----------------|----------------------|----------------------------------|
| neurondb       | `neurondb/`          | Core extension, dockers, docs    |
| NeuronAgent    | `neuron-agent/`      | Agent service (own repo)         |
| NeuronDesktop  | `neuron-desktop/`    | Web UI – API + frontend (own repo) |
| NeuronMCP      | `neuron-mcp/`        | MCP server (own repo)            |
| neuron-cloud   | `neuron-cloud/`      | Control plane (SaaS)            |
| neuron-hub     | `neuron-hub/`        | Agent builder / embed platform   |
| neuron-deploy  | `neuron-deploy/`     | Orchestration over all components |
| neuron-llm     | `neuron-llm/`        | LLM SQL training (Python package) |
| neuron-www     | `neuron-www/`        | Marketing website                |
| dataset        | `dataset/`           | Dataset utilities and assets     |

Keep **agent** and **desktop** in their own folders under `pge/`; neurondb references the desktop via the sibling path `../neuron-desktop` (or `NEURONDESKTOP_DIR`) for a clean directory structure.
