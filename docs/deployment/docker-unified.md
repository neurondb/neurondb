# Unified Docker Orchestration

Build and run the NeuronDB PostgreSQL extension in Docker using the Compose file in this repository.

---

## Overview

The Compose file `docker/docker-compose.yml` defines NeuronDB services only (CPU and GPU variants). Use Docker Compose directly; there are no `make build` or `make run` targets for Docker.

**Published registry images** (pull, do not build): **`neurondb/neurondb-cuda`** / **`ghcr.io/neurondb/neurondb-cuda`** — see [Container images](container-images.md). This page describes **building from the repo** for development.

## Quick start

### Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- For GPU: NVIDIA Container Toolkit (CUDA), ROCm, or Metal as appropriate

### 1. Build

From the **repository root**:

```bash
# Build CPU variant (default)
docker compose -f docker/docker-compose.yml build

# Or build a GPU variant
docker compose -f docker/docker-compose.yml --profile cuda build
docker compose -f docker/docker-compose.yml --profile rocm build
docker compose -f docker/docker-compose.yml --profile metal build
```

### 2. Run

```bash
# Run CPU (default)
docker compose -f docker/docker-compose.yml up -d

# Or run a GPU variant
docker compose -f docker/docker-compose.yml --profile cuda up -d
docker compose -f docker/docker-compose.yml --profile rocm up -d
docker compose -f docker/docker-compose.yml --profile metal up -d
```

### 3. Verify

```bash
docker compose -f docker/docker-compose.yml ps
docker compose -f docker/docker-compose.yml exec neurondb pg_isready -U neurondb -d neurondb
docker compose -f docker/docker-compose.yml exec neurondb psql -U neurondb -d neurondb -c "SELECT neurondb.version();"
```

## Architecture

The Compose file defines one or more NeuronDB PostgreSQL containers (depending on profile):

- **neurondb** (default/cpu) — port 5433
- **neurondb-cuda** (profile `cuda`) — port 5434
- **neurondb-rocm** (profile `rocm`) — port 5435
- **neurondb-metal** (profile `metal`) — port 5436

**External access:** Connect to PostgreSQL at `localhost:5433` (or the port for the variant you started).

## Configuration

### Environment variables

Use a `.env` file in the repository root (e.g. copy from `.env.example`). Common variables:

- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` — database credentials (defaults: neurondb/neurondb/neurondb)
- `POSTGRES_PORT` — host port for PostgreSQL (default 5433)
- `NEURONDB_LLM_API_KEY` — optional, for embedding/LLM features

For production, set a strong `POSTGRES_PASSWORD`; do not commit `.env` with real secrets.

## Usage

### Management

```bash
# Status
docker compose -f docker/docker-compose.yml ps

# Logs
docker compose -f docker/docker-compose.yml logs neurondb
docker compose -f docker/docker-compose.yml logs -f neurondb

# Stop (keep volumes)
docker compose -f docker/docker-compose.yml down

# Stop and remove volumes
docker compose -f docker/docker-compose.yml down -v
```

### Running one variant

```bash
# Start only the CPU service
docker compose -f docker/docker-compose.yml up -d neurondb

# Or only a GPU service
docker compose -f docker/docker-compose.yml --profile cuda up -d neurondb-cuda
```

### Testing connection

```bash
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" -c "SELECT neurondb.version();"
```

## GPU variants

- **CUDA:** `--profile cuda`, NVIDIA GPU and nvidia-container-toolkit.
- **ROCm:** `--profile rocm`, AMD GPU and ROCm drivers.
- **Metal:** `--profile metal`, Apple Silicon, macOS 13+.

See [GPU feature matrix](../gpu/gpu-feature-matrix.md) and `docker/README.md` for details.

## Troubleshooting

- **Port in use:** Set `POSTGRES_PORT` in `.env` or stop the process using the port.
- **GPU not detected:** Verify the Docker runtime (e.g. `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` for CUDA).
- **Build failures:** Ensure build context is correct; use `docker compose -f docker/docker-compose.yml build neurondb` to build only the main image.

## Related documentation

- [Docker ecosystem](docker-ecosystem.md)
- [NeuronDB Docker](../../docker/README.md)
- [Container images](container-images.md)

---

[Deployment](README.md) · [Documentation](../readme.md)
