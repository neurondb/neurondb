# Docker Ecosystem Setup and Verification

Run and verify the NeuronDB PostgreSQL extension in Docker using the Compose file in this repository.

---

## Overview

| Topic | Description |
|-------|-------------|
| **Starting** | Start NeuronDB (CPU or GPU) with Docker Compose |
| **Verification** | Confirm the extension is loaded and working |
| **Troubleshooting** | Common issues and fixes |

## Compose file

**Path:** `docker/docker-compose.yml` (no root `docker-compose.yml`).

**Services defined:**
- `neurondb` — PostgreSQL with NeuronDB extension (CPU, default)
- `neurondb-cuda` — CUDA GPU (profile `cuda`)
- `neurondb-rocm` — ROCm GPU (profile `rocm`)
- `neurondb-metal` — Metal GPU, Apple Silicon (profile `metal`)

## Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- 4GB+ RAM
- Port 5433 (or 5434/5435/5436 for GPU variants)
- For GPU: NVIDIA Container Toolkit (CUDA), ROCm, or Metal as appropriate

## Quick start

### 1. Start NeuronDB

From the repository root:

```bash
# CPU (default)
docker compose -f docker/docker-compose.yml up -d

# Or from docker/
cd docker && docker compose up -d

# GPU variants
docker compose -f docker/docker-compose.yml --profile cuda up -d   # CUDA
docker compose -f docker/docker-compose.yml --profile rocm up -d   # ROCm
docker compose -f docker/docker-compose.yml --profile metal up -d  # Metal (Apple Silicon)
```

First run will build images; subsequent starts are faster.

### 2. Verify

**Health:**

```bash
docker compose -f docker/docker-compose.yml ps
docker compose -f docker/docker-compose.yml exec neurondb pg_isready -U neurondb -d neurondb
```

**Extension:**

```bash
docker compose -f docker/docker-compose.yml exec neurondb \
  psql -U neurondb -d neurondb -c "SELECT neurondb.version();"
```

**Optional script (if present):**

```bash
./scripts/verify-docker-ecosystem.sh
```

This script checks containers and extension; it does not cover services from other repositories.

### 3. Connect

```bash
# From host
psql -h localhost -p 5433 -U neurondb -d neurondb

# Or via exec
docker compose -f docker/docker-compose.yml exec neurondb psql -U neurondb -d neurondb
```

Default password is `neurondb` (development only). Set `POSTGRES_PASSWORD` in `.env` for production.

## Logs and debugging

```bash
docker compose -f docker/docker-compose.yml logs -f neurondb
docker compose -f docker/docker-compose.yml logs neurondb --tail 100
```

## Stopping

```bash
docker compose -f docker/docker-compose.yml down
# With volumes: docker compose -f docker/docker-compose.yml down -v
```

## Related

- [Docker deployment](docker.md)
- [Container images](container-images.md)
- [Quick start](../getting-started/quickstart.md)
- [Verification](../integration/verification.md)

---

[Deployment](README.md) · [Documentation](../readme.md)
