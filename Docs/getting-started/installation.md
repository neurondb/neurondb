# Installation Guide

Complete installation guide for the NeuronDB ecosystem, based on what is present in this repository.

> **New here?** Start with **[simple-start.md](simple-start.md)** first.

> **Technical user?** Continue below for detailed installation options.

---

## Choose Your Installation Method

| Method | Best For | Time Required | Difficulty |
|--------|----------|---------------|------------|
| **[Docker](#method-1-docker-recommended)** | Most users | 5-15 minutes | Easy |
| **[Source build](#method-2-source-build)** | Developers, custom builds | 30+ minutes | Advanced |

**Recommendation:** Use Docker unless you have a specific reason not to.

---

## Prerequisites (by component)

### System requirements

- **PostgreSQL**: 16, 17, or 18
- **Operating System**: Linux, macOS, or Windows (with WSL2)
- **Docker**: required for containerized deployment

### Build requirements (only if building from source)

- **NeuronDB (extension)**: C toolchain + PostgreSQL server development headers (`pg_config`), `make`
- **NeuronAgent**: Go `1.24.x` (see `NeuronAgent/go.mod`)
- **NeuronMCP**: Go `1.23.x` (see `NeuronMCP/go.mod`)
- **NeuronDesktop API**: Go `1.24.x` (see `NeuronDesktop/api/go.mod`)
- **NeuronDesktop frontend**: Node.js (see `NeuronDesktop/frontend/package.json`)

### Optional requirements

- **GPU Support**: CUDA (NVIDIA), ROCm (AMD), or Metal (Apple Silicon) for GPU acceleration
- **Build Tools**: C compiler (GCC or Clang), Make, PostgreSQL development headers

## Installation Methods

### Method 1: Docker (Recommended)

Docker installation provides the easiest and most consistent setup.

#### Quick Start

```bash
# Clone the repository (or use your existing checkout)
git clone <repository-url>
cd neurondb

# Start all services
docker compose up -d
```

#### Notes

- The canonical compose file is `docker-compose.yml` at repo root.
- For the full Docker layout and the helper script, see `dockers/readme.md` and `dockers/docker.sh`.

### Method 2: Native installation (without Docker)

Install components directly on your system using automated installation scripts or manual setup.

**Quick Installation:**

```bash
# Install all components
sudo ./scripts/install-components.sh

# Install specific components
sudo ./scripts/install-components.sh neuronmcp neuronagent
```

For detailed instructions, see [Native Installation Guide](installation-native.md) and [Service Management Guide](installation-services.md).

### Method 3: Source build (manual)

Build and install components manually:

#### Build/install NeuronDB extension

See `NeuronDB/INSTALL.md` (and `NeuronDB/Makefile`) for the authoritative steps.

```bash
cd NeuronDB
PG_CONFIG=/path/to/pg_config make
sudo PG_CONFIG=/path/to/pg_config make install

# Then, in Postgres:
# CREATE EXTENSION neurondb;
```

#### Build NeuronAgent

```bash
cd NeuronAgent
go build ./cmd/agent-server

# Run with configuration
./agent-server -config configs/config.yaml
```

See `NeuronAgent/readme.md` and `NeuronAgent/openapi/` for details.

#### Build NeuronMCP

```bash
cd NeuronMCP
go build ./cmd/neurondb-mcp

# Run with environment variables (see `docker-compose.yml` for the env names)
# For Docker Compose setup (connecting from host):
export NEURONDB_HOST=localhost
export NEURONDB_PORT=5433        # Docker Compose default host port
export NEURONDB_DATABASE=neurondb
export NEURONDB_USER=neurondb
export NEURONDB_PASSWORD=neurondb
./neurondb-mcp

# For native PostgreSQL or inside Docker network:
export NEURONDB_HOST=localhost
export NEURONDB_PORT=5432        # Native PostgreSQL default port
export NEURONDB_DATABASE=neurondb
./neurondb-mcp
```

See [NeuronMCP README](../../NeuronMCP/readme.md) for setup details.

#### Step 4: Build NeuronDesktop

**Backend:**
```bash
cd NeuronDesktop/api
go build ./cmd/server
./server
```

**Frontend:**
```bash
cd NeuronDesktop/frontend
npm install
npm run dev
```

See [NeuronDesktop README](../../NeuronDesktop/readme.md) for detailed setup.

### Method 3: Package Installation

Install using platform-specific packages (DEB/RPM).

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

### Install Extensions

```bash
psql -d neurondb -c "CREATE EXTENSION neurondb;"
```

### Run Migrations

**NeuronAgent:**
```bash
cd NeuronAgent
./NeuronAgent/scripts/neuronagent_migrate.sh
```
This runs all migrations including `migrations/001_initial_schema.sql` and subsequent migrations.

**NeuronDesktop:**
```bash
cd NeuronDesktop
createdb neurondesk
./scripts/setup_neurondesktop.sh
```
This runs all migrations including `api/migrations/001_initial_schema.sql` and subsequent migrations.

## Verification

### Verify NeuronDB

```bash
psql -d neurondb -c "SELECT neurondb.version();"
```

### Verify NeuronAgent

```bash
curl http://localhost:8080/health
```

### Verify NeuronMCP

```bash
# Check if binary exists and is executable
which neurondb-mcp
```

### Verify NeuronDesktop

```bash
# Frontend
curl http://localhost:3000

# Backend
curl http://localhost:8081/health
```

## Configuration

### Environment Variables

Each component requires specific environment variables. See component-specific documentation:

- [NeuronDB Configuration](../../NeuronDB/docs/configuration.md)
- [NeuronAgent Configuration](../../NeuronAgent/readme.md#configuration)
- [NeuronMCP Configuration](../../NeuronMCP/readme.md#configuration)
- [NeuronDesktop Configuration](../../NeuronDesktop/readme.md#configuration)

### Configuration Files

- **NeuronAgent**: `NeuronAgent/configs/config.yaml`
- **NeuronMCP**: `NeuronMCP/mcp-config.json`
- **NeuronDesktop**: Environment variables or `.env` file

## Next Steps

1. **[Quick Start Guide](quickstart.md)** - Run your first queries
2. **[Component Documentation](../components/README.md)** - Learn about each component
3. **[Integration Guide](../ecosystem/integration.md)** - Connect components together

## Troubleshooting

### Common Issues

- **Connection Errors**: Verify database is running and connection parameters are correct
- **Extension Not Found**: Ensure NeuronDB extension is installed in the database
- **Port Conflicts**: Check if ports 5432, 8080, 8081, 3000 are available
- **Build Errors**: Verify all prerequisites are installed

For detailed troubleshooting, see:
- [NeuronDB Troubleshooting](../../NeuronDB/docs/troubleshooting.md)
- [Official Documentation](https://www.neurondb.ai/docs/troubleshooting)

## Official Documentation

For comprehensive installation guides and platform-specific instructions:
** [https://www.neurondb.ai/docs/installation](https://www.neurondb.ai/docs/installation)**

