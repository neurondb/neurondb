# Installing NeuronDB

The default path is **Docker** — see the [root README](../README.md) and [docker.md](docker.md).

## Docker (recommended)

1. **One command:**  
   `curl -fsSL https://raw.githubusercontent.com/neurondb/neurondb/main/scripts/install-docker.sh | bash`
2. **Manual run:**  
   Use image **`neurondb/neurondb:latest`** (CPU) or **`neurondb/neurondb-cuda:latest`** (NVIDIA; add `--gpus all`), plus `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, and a volume on `/var/lib/postgresql/data`. See [docker.md](docker.md).

## Docker Compose (from a clone)

```bash
docker compose -f docker/docker-compose.yml up -d neurondb
```

See [docker.md](docker.md) for using the Hub image instead of a local build.

## Native / build from source

For developers or custom PostgreSQL installations:

1. Read [INSTALL.md](../INSTALL.md) for toolchain and optional ML dependencies.
2. Follow [getting-started/installation.md](getting-started/installation.md) for step-by-step native install.
3. From the repository root, typical flow:

```bash
./build.sh
# or: make && sudo make install
```

You need `pg_config` for the target PostgreSQL version (16–18).

## Packages

DEB/RPM workflows may exist in CI; see `src/packaging/` and release artifacts on GitHub when available.
