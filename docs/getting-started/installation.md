# Installation Guide

Install the NeuronDB PostgreSQL extension via Docker or from source.

> **New here?** Start with [Simple Start](simple-start.md).

---

## Choose your method

| Method | Best for | Time | Difficulty |
|--------|----------|------|------------|
| **[Docker](#method-1-docker-recommended)** | Most users | 5–15 min | Easy |
| **[Source build](#method-2-source-build)** | Developers, custom builds | 30+ min | Advanced |

---

## Prerequisites

- **PostgreSQL:** 16, 17, or 18 (for native install)
- **OS:** Linux, macOS, or Windows (WSL2)
- **Docker:** For containerized install

**Source build only:** C toolchain, `pg_config`, `make`. See [INSTALL.md](../../INSTALL.md) for ML library prerequisites (XGBoost, LightGBM, CatBoost). Optional: GPU support (CUDA, ROCm, Metal).

## Method 1: Docker

```bash
git clone <repository-url>
cd neurondb

docker compose -f docker/docker-compose.yml up -d

# Or: cd docker && docker compose up -d
```

- Compose file: **`docker/docker-compose.yml`**. See [docker/README.md](../../docker/README.md) and `docker/docker.sh` for GPU profiles.

## Method 2: Native installation

```bash
# From repository root
./build.sh
# Or: PG_CONFIG=/path/to/pg_config make && sudo make install
```

See [INSTALL.md](../../INSTALL.md) and [Native Installation](installation-native.md) for details.

## Method 3: Source build (manual)

From the repository root:

```bash
PG_CONFIG=/path/to/pg_config make
sudo PG_CONFIG=/path/to/pg_config make install
```

Then in PostgreSQL:

```sql
CREATE EXTENSION neurondb;
```

See [INSTALL.md](../../INSTALL.md) for full steps.

### Method 4: Package Installation

Install using platform-specific packages (DEB/RPM), if available.

```bash
# Debian/Ubuntu
sudo dpkg -i neurondb_*.deb

# RHEL/CentOS
sudo rpm -i neurondb_*.rpm
```

See [Packaging Documentation](../deployment/package.md) for package build instructions.

## Database Setup

### Create Database

```bash
createdb neurondb
```

### Install Extension

```bash
psql -d neurondb -c "CREATE EXTENSION neurondb;"
```

For NeuronAgent or NeuronDesktop database migrations, see their respective repositories.

## Verification

### Verify NeuronDB

```bash
psql -d neurondb -c "SELECT neurondb.version();"
```

For verifying NeuronAgent, NeuronMCP, or NeuronDesktop, see their repositories.

## Configuration

### NeuronDB configuration

- [Configuration reference](../configuration.md) — GUCs, `shared_preload_libraries`, etc.
- Environment variables for Docker are in `.env.example` at the repository root.

Configuration for NeuronAgent, NeuronMCP, and NeuronDesktop is documented in their respective repositories.

## Next Steps

1. **[Quick Start Guide](quickstart.md)** — Run your first queries
2. **[Components](components/README.md)** — NeuronDB component overview

## Troubleshooting

### Common Issues

- **Connection Errors**: Verify database is running and connection parameters are correct
- **Extension Not Found**: Ensure NeuronDB extension is installed in the database
- **Port Conflicts**: Check if ports 5432, 8080, 8081, 3000 are available
- **Build Errors**: Verify all prerequisites are installed

For detailed troubleshooting, see:
- [Troubleshooting](troubleshooting.md) (this repo)
- [Official Documentation](https://www.neurondb.ai/docs/troubleshooting)

## Official Documentation

For comprehensive installation guides and platform-specific instructions:
** [https://www.neurondb.ai/docs/installation](https://www.neurondb.ai/docs/installation)**

