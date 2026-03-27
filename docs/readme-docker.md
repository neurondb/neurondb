# Running NeuronDB Tests with Docker PostgreSQL

This guide explains how to run NeuronDB tests against a Docker PostgreSQL container or a local PostgreSQL with the extension installed.

**Note:** This repository does **not** include a script named `scripts/run_tests_docker.sh`. Use the methods below.

## Quick Start

### Option 1: Docker + make installcheck (recommended)

1. Start NeuronDB in Docker from the repository root:

   ```bash
   docker compose -f docker/docker-compose.yml up -d neurondb
   ```

2. Wait for the container to be healthy, then run the extension regression tests from the repo root:

   ```bash
   PGHOST=localhost PGPORT=5433 PGDATABASE=neurondb PGUSER=neurondb PGPASSWORD=neurondb make installcheck
   ```

   Or with a single connection string:

   ```bash
   PGCONNSTRING="postgresql://neurondb:neurondb@localhost:5433/neurondb" make installcheck
   ```

   (If your Makefile supports `PGCONNSTRING`; otherwise use the individual `PG*` variables.)

### Option 2: Local PostgreSQL (latest source code)

If you have built and installed the extension locally (e.g. `./build.sh` or `make install`):

1. Ensure PostgreSQL is running and `shared_preload_libraries = 'neurondb'` is set, then restart.
2. From the repository root:

   ```bash
   make installcheck
   ```

   Or with explicit connection:

   ```bash
   PGHOST=localhost PGPORT=5432 PGDATABASE=neurondb PGUSER=your_user make installcheck
   ```

## Docker PostgreSQL configuration

When using this repo’s Docker Compose (`docker/docker-compose.yml`), default ports are:

- **CPU variant**: host port `5433` → container `5432`
- **CUDA variant**: host port `5434`
- **ROCm variant**: host port `5435`
- **Metal variant**: host port `5436`

Default credentials: user `neurondb`, password `neurondb`, database `neurondb`.

## Manual test run (Python runner)

You can also run the Python test runner from this repo (tests live under `src/tests/`):

```bash
# From repository root; PostgreSQL must be running (Docker or local)
python3 src/tests/run_test.py \
  --compute=cpu \
  --port=5433 \
  --host=localhost \
  --user=neurondb \
  --password=neurondb \
  --db=neurondb \
  --category=basic
```

Adjust `--port` to match your setup (e.g. 5433 for Docker CPU, 5432 for local).

## Finding the Docker port

```bash
docker compose -f docker/docker-compose.yml ps
# or
docker ps --format "table {{.Names}}\t{{.Ports}}"
```

## See also

- [INSTALL.md](../../INSTALL.md) — Build and install the extension
- [getting-started/installation.md](getting-started/installation.md) — Installation options
- [Troubleshooting](getting-started/troubleshooting.md) — Common issues
