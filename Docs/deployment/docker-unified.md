# NeuronDB Ecosystem - Unified Docker Orchestration

Complete guide for building and running NeuronDB, NeuronAgent, NeuronMCP, and NeuronDesktop together using Docker Compose.

## Overview

This unified Docker orchestration system provides simple commands to build and run all four services (NeuronDB, NeuronAgent, NeuronMCP, and NeuronDesktop) with automatic networking and default connection settings. The system uses Docker Compose profiles to support CPU and GPU variants (CUDA, ROCm, Metal).

## Quick Start

### Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- For GPU support: NVIDIA Docker runtime (CUDA) or ROCm drivers (ROCm)

### 1. Build All Services

```bash
# Build CPU variant (default)
make build

# Or build specific GPU variant
make build-cuda   # CUDA GPU
make build-rocm   # ROCm GPU
make build-metal  # Metal GPU (macOS/Apple Silicon)
```

### 2. Run All Services

```bash
# Run CPU variant (default)
make run

# Or run specific GPU variant
make run-cuda     # CUDA GPU
make run-rocm     # ROCm GPU
make run-metal    # Metal GPU
```

### 3. Verify Services

```bash
# Check service status
make status

# Check service health
make health

# View logs
make logs
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Docker Network: neurondb-network                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐      ┌────────────┐│
│  │  NeuronDB    │◄─────┤  NeuronAgent │      │  NeuronMCP  ││
│  │  (PostgreSQL)│      │  (REST API)  │      │  (MCP)      ││
│  │  Port: 5433  │      │  Port: 8080  │      │  (stdio)    ││
│  └──────────────┘      └──────────────┘      └────────────┘│
│                                                               │
│  All services connect via shared network using container     │
│  names for automatic service discovery                        │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Service Communication

- **NeuronAgent → NeuronDB**: Connects via service name `neurondb:5432` (internal Docker network)
- **NeuronMCP → NeuronDB**: Connects via service name `neurondb:5432` (internal Docker network)
- **NeuronDesktop → NeuronDB**: Connects via service name `neurondb:5432` (internal Docker network)
- **External Access**: Use host ports (`localhost:5433` for PostgreSQL, `localhost:8080` for NeuronAgent, `localhost:8081` for NeuronDesktop API, `localhost:3000` for NeuronDesktop UI)

**Important:** Inside Docker network, services use the service name (`neurondb`) not container name (`neurondb-cpu`). From your host machine, use `localhost` with the mapped ports.

## Configuration

### Environment Variables

Copy the example environment file and customize:

```bash
cp .env.example .env
```

Edit `.env` to customize settings:

```env
# NeuronDB Configuration
POSTGRES_USER=neurondb
POSTGRES_PASSWORD=neurondb
POSTGRES_DB=neurondb
POSTGRES_PORT=5433

# NeuronAgent Configuration (auto-connects to NeuronDB)
DB_HOST=neurondb            # Service name in Docker network (not container name)
DB_PORT=5432                # Internal container port (not host port 5433)
DB_NAME=neurondb
DB_USER=neurondb
DB_PASSWORD=neurondb
SERVER_PORT=8080

# NeuronMCP Configuration (auto-connects to NeuronDB)
NEURONDB_HOST=neurondb-cpu  # Container name for automatic connection
NEURONDB_PORT=5432          # Internal container port
NEURONDB_DATABASE=neurondb
NEURONDB_USER=neurondb
NEURONDB_PASSWORD=neurondb
```

### GPU Variant Configuration

When using GPU variants, update the connection hostnames in `.env`:

**For CUDA:**
```env
DB_HOST=neurondb-cuda
NEURONDB_HOST=neurondb-cuda
```

**For ROCm:**
```env
DB_HOST=neurondb-rocm
NEURONDB_HOST=neurondb-rocm
```

**For Metal:**
```env
DB_HOST=neurondb-metal
NEURONDB_HOST=neurondb-metal
```

## Usage

### Build Commands

```bash
make build          # Build all services (CPU)
make build-cpu      # Build CPU variant only
make build-cuda     # Build CUDA GPU variant
make build-rocm     # Build ROCm GPU variant
make build-metal    # Build Metal GPU variant
```

### Run Commands

```bash
make run            # Start all services (CPU)
make run-cuda       # Start all services with CUDA GPU
make run-rocm       # Start all services with ROCm GPU
make run-metal      # Start all services with Metal GPU
```

### Management Commands

```bash
make stop           # Stop all running services
make logs            # View logs from all services
make logs-neurondb   # View NeuronDB logs only
make logs-neuronagent # View NeuronAgent logs only
make logs-neuronmcp  # View NeuronMCP logs only
make ps              # Show running containers
make status          # Show service status
make health          # Check service health
```

### Cleanup Commands

```bash
make clean           # Stop and remove containers
make clean-all       # Stop, remove containers, networks, and volumes
```

## Service Endpoints

After starting services:

- **NeuronDB**: `localhost:5433` (PostgreSQL)
- **NeuronAgent**: `http://localhost:8080` (REST API)
- **NeuronMCP**: stdio protocol (for MCP clients)

### Testing Connections

**Test NeuronDB:**
```bash
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" \
  -c "SELECT neurondb.version();"
```

**Test NeuronAgent:**
```bash
curl http://localhost:8080/health
```

**Test NeuronMCP:**
```bash
docker exec neurondb-mcp test -f /app/neurondb-mcp -a -x /app/neurondb-mcp
```

## GPU Variants

### CUDA (NVIDIA)

**Requirements:**
- NVIDIA GPU with CUDA support
- NVIDIA Docker runtime installed

**Usage:**
```bash
make build-cuda
make run-cuda
```

**Configuration:**
Update `.env` with:
```env
DB_HOST=neurondb-cuda
NEURONDB_HOST=neurondb-cuda
```

### ROCm (AMD)

**Requirements:**
- AMD GPU with ROCm support
- ROCm drivers installed

**Usage:**
```bash
make build-rocm
make run-rocm
```

**Configuration:**
Update `.env` with:
```env
DB_HOST=neurondb-rocm
NEURONDB_HOST=neurondb-rocm
```

### Metal (Apple Silicon)

**Requirements:**
- macOS with Apple Silicon (M1/M2/M3)
- Docker Desktop for Mac

**Usage:**
```bash
make build-metal
make run-metal
```

**Configuration:**
Update `.env` with:
```env
DB_HOST=neurondb-metal
NEURONDB_HOST=neurondb-metal
```

## Troubleshooting

### Services Cannot Connect to NeuronDB

**Symptoms:**
- Connection timeout errors
- Authentication failures

**Solutions:**

1. Verify NeuronDB is running:
   ```bash
   make status
   docker ps | grep neurondb
   ```

2. Check network connectivity:
   ```bash
   docker exec neuronagent ping neurondb-cpu
   docker exec neurondb-mcp ping neurondb-cpu
   ```

3. Verify environment variables:
   ```bash
   docker exec neuronagent env | grep DB_
   docker exec neurondb-mcp env | grep NEURONDB_
   ```

4. Check container names match:
   - For CPU: `DB_HOST=neurondb-cpu`
   - For CUDA: `DB_HOST=neurondb-cuda`
   - For ROCm: `DB_HOST=neurondb-rocm`
   - For Metal: `DB_HOST=neurondb-metal`

### Port Already in Use

**Symptoms:**
- Port binding errors
- Service fails to start

**Solutions:**

1. Change ports in `.env`:
   ```env
   POSTGRES_PORT=5434
   SERVER_PORT=8081
   ```

2. Stop conflicting services:
   ```bash
   docker ps | grep -E "5433|8080"
   docker stop <container-id>
   ```

### GPU Not Detected

**Symptoms:**
- GPU services fail to start
- No GPU acceleration

**Solutions:**

1. **CUDA:**
   ```bash
   # Verify NVIDIA Docker runtime
   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```

2. **ROCm:**
   ```bash
   # Verify ROCm devices
   ls -la /dev/kfd /dev/dri
   ```

3. Check Docker Compose profiles:
   ```bash
   docker compose --profile cuda config | grep -i gpu
   ```

### Service Health Checks Fail

**Symptoms:**
- Services show as unhealthy
- Health check timeouts

**Solutions:**

1. Check service logs:
   ```bash
   make logs-neurondb
   make logs-neuronagent
   make logs-neuronmcp
   ```

2. Increase health check timeout in `docker-compose.yml` if needed

3. Verify database is ready:
   ```bash
   docker exec neurondb-cpu pg_isready -U neurondb
   ```

### Build Failures

**Symptoms:**
- Docker build errors
- Missing dependencies

**Solutions:**

1. Check Docker build context:
   ```bash
   ls -la dockers/neurondb/Dockerfile
   ls -la dockers/neuronagent/Dockerfile
   ls -la dockers/neuronmcp/Dockerfile
   ```

2. Clear Docker build cache:
   ```bash
   docker builder prune -a
   ```

3. Build individual services:
   ```bash
   docker compose build neurondb
   docker compose build neuronagent
   docker compose build neuronmcp
   ```

## Extended Usage

### Using Docker Compose Directly

You can also use `docker compose` commands directly:

```bash
# Build
docker compose build

# Run
docker compose up -d

# Run with GPU profile
docker compose --profile cuda up -d

# Stop
docker compose down

# View logs
docker compose logs -f
```

### Custom Build Arguments

Override build arguments:

```bash
docker compose build \
  --build-arg PG_MAJOR=18 \
  --build-arg CUDA_VERSION=12.4.1 \
  neurondb-cuda
```

### Running Individual Services

Start only specific services:

```bash
# Start only NeuronDB
docker compose up -d neurondb

# Start NeuronDB and NeuronAgent
docker compose up -d neurondb neuronagent
```

### Network Inspection

Inspect the shared network:

```bash
docker network inspect neurondb-network
```

## Best Practices

1. **Use `.env` file**: Copy `.env.example` to `.env` and customize settings
2. **Check health**: Use `make health` to verify all services are running
3. **Monitor logs**: Use `make logs` to monitor service activity
4. **GPU configuration**: Update `.env` with correct hostnames when using GPU variants
5. **Resource limits**: Adjust CPU and memory limits in `docker-compose.yml` based on your system
6. **Backup data**: NeuronDB data is stored in Docker volumes - back up volumes regularly

## Related Documentation

- [NeuronDB Docker Guide](../../dockers/neurondb/README.md)
- [NeuronAgent Docker Guide](../../dockers/neuronagent/README.md)
- [NeuronMCP Docker Guide](../../dockers/neuronmcp/README.md)
- [Ecosystem Docker Guide](../../dockers/neurondb/ECOSYSTEM.md)
- [Main README](../../README.md)

## Support

For issues and questions:
- GitHub Issues: [Report Issues](https://github.com/neurondb/NeurondB/issues)
- Documentation: [Full Documentation](https://neurondb.ai/docs)
- Email: support@neurondb.ai



