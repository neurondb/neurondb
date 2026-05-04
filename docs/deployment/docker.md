# Docker Deployment Guide

Deploy the NeuronDB PostgreSQL extension using Docker and the Compose file in this repository.

---

## Overview

Docker provides a consistent way to run PostgreSQL with the NeuronDB extension. The Compose file is **`docker/docker-compose.yml`** (no root `docker-compose.yml`).

**Benefits:**
- No manual PostgreSQL install or extension build on the host
- Isolated environment
- **Pulled** images: official **CUDA** (`neurondb/neurondb-cuda`); this repo’s Compose can also **build** CPU and other GPU variants from source

## Prerequisites

| Requirement | Minimum | Notes |
|-------------|---------|--------|
| Docker | 20.10+ | Required |
| Docker Compose | 2.0+ | Required |
| NVIDIA Docker | Latest | For CUDA |
| ROCm | 5.7+ | For AMD GPU |
| macOS | 13+ | For Metal (Apple Silicon) |

**Registry default:** CUDA only (NVIDIA). The Compose file can still build a **CPU** image locally without a GPU.

## Quick start

From the repository root:

```bash
# Start NeuronDB (CPU, default)
docker compose -f docker/docker-compose.yml up -d

# Wait for healthy (e.g. 30–60 seconds), then verify
docker compose -f docker/docker-compose.yml ps
docker compose -f docker/docker-compose.yml exec neurondb psql -U neurondb -d neurondb -c "SELECT neurondb.version();"
```

**Expected:** One or more `neurondb*` containers running (e.g. `neurondb-cpu`).

## Compose file location

The canonical Compose file is **`docker/docker-compose.yml`**. When running from the repository root, always use:

```bash
docker compose -f docker/docker-compose.yml <command>
```

Or run from the `docker/` directory:

```bash
cd docker && docker compose up -d
```

## Services (this repository)

The Compose file defines only NeuronDB-related services:

| Service | Profile | Port | Description |
|---------|---------|------|-------------|
| neurondb | default, cpu | 5433 | PostgreSQL + NeuronDB (CPU) |
| neurondb-cuda | cuda | 5434 | PostgreSQL + NeuronDB (CUDA) |
| neurondb-rocm | rocm | 5435 | PostgreSQL + NeuronDB (ROCm) |
| neurondb-metal | metal | 5436 | PostgreSQL + NeuronDB (Metal) |

## GPU variants

```bash
# CUDA
docker compose -f docker/docker-compose.yml --profile cuda up -d

# ROCm
docker compose -f docker/docker-compose.yml --profile rocm up -d

# Metal (Apple Silicon)
docker compose -f docker/docker-compose.yml --profile metal up -d
```

See [Unified Docker Guide](docker-unified.md) and [GPU feature matrix](../gpu/gpu-feature-matrix.md) for details.

## Configuration

Use a `.env` file (e.g. from `.env.example`). Key variables:

- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` — defaults: neurondb / neurondb / neurondb
- `POSTGRES_PORT` — host port (default 5433)
- `NEURONDB_LLM_API_KEY` — optional, for embeddings/LLM

Do not commit real secrets. For production, set a strong `POSTGRES_PASSWORD`.

## Service management

```bash
# Start
docker compose -f docker/docker-compose.yml up -d

# Stop (keep volumes)
docker compose -f docker/docker-compose.yml down

# Stop and remove volumes (data loss)
docker compose -f docker/docker-compose.yml down -v

# Logs
docker compose -f docker/docker-compose.yml logs -f neurondb
```

## Health check

```bash
docker compose -f docker/docker-compose.yml exec neurondb pg_isready -U neurondb -d neurondb
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" -c "SELECT neurondb.version();"
```

## Data persistence

PostgreSQL data is stored in Docker volumes (e.g. `neurondb-data`). Back up with `pg_dump`; see [Backup and restore](backup-restore.md).

## Troubleshooting

- **Port in use:** Change `POSTGRES_PORT` in `.env` or stop the conflicting service.
- **Extension not found:** Ensure you are using the Neurondb image and that the extension is created: `CREATE EXTENSION neurondb;`
- **GPU:** Verify Docker GPU runtime (e.g. `nvidia-smi` inside container for CUDA).

See [Operations troubleshooting](../operations/troubleshooting.md) and [Getting started troubleshooting](../getting-started/troubleshooting.md).

## Related documentation

| Document | Description |
|----------|-------------|
| [Unified Docker Guide](docker-unified.md) | Build, run, and verify |
| [Docker ecosystem](docker-ecosystem.md) | Setup and verification |
| [NeuronDB Docker](../../docker/README.md) | Dockerfile and build notes |
| [Backup and restore](backup-restore.md) | Backup procedures |

---

[Deployment](README.md) · [Documentation](../readme.md)
