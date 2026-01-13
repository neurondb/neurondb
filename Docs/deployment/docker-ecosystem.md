# Docker Ecosystem Setup and Verification Guide

Complete guide for running and verifying NeuronDB, NeuronAgent, NeuronDesktop, and NeuronMCP together in Docker containers.

## Overview

This guide covers:
- Starting all services with Docker Compose
- Verifying all components are working
- Testing integrations between services
- Troubleshooting common issues

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Docker Network: neurondb-network          │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐      ┌──────────────┐      ┌─────────┐│
│  │  NeuronDB    │◄─────┤  NeuronAgent │      │ NeuronMCP││
│  │  (PostgreSQL)│      │  (REST API)  │      │  (stdio) ││
│  │  Port: 5433  │      │  Port: 8080  │      │          ││
│  └──────┬───────┘      └──────┬───────┘      └────┬─────┘│
│         │                      │                    │      │
│         │                      │                    │      │
│  ┌──────▼──────────────────────▼────────────────────▼─────┐│
│  │         NeuronDesktop                                  ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ ││
│  │  │   Frontend   │  │     API      │  │   Init DB   │ ││
│  │  │  Port: 3000  │  │  Port: 8081  │  │             │ ││
│  │  └──────────────┘  └──────────────┘  └─────────────┘ ││
│  └───────────────────────────────────────────────────────┘│
└───────────────────────────────────────────────────────────┘
```

## Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- 4GB+ RAM available
- Ports available: 5433, 8080, 8081, 3000
- For GPU support: NVIDIA Docker runtime (CUDA) or ROCm drivers

## Quick Start

### 1. Start All Services

From the repository root:

```bash
# Start all services (CPU variant, default)
docker compose up -d

# Or start with GPU support
docker compose --profile cuda up -d    # CUDA
docker compose --profile rocm up -d    # ROCm
docker compose --profile metal up -d   # Metal (Apple Silicon)
```

This will:
- Build Docker images (first time only, takes 5-10 minutes)
- Start all services in the correct order
- Configure networking between components
- Initialize databases

### 2. Verify Services

Run the comprehensive verification script:

```bash
./scripts/verify-docker-ecosystem.sh
```

This checks:
- All containers are running and healthy
- NeuronDB extension loads and works
- NeuronAgent API responds
- NeuronMCP server works
- NeuronDesktop API and frontend are accessible
- All services can communicate

### 3. Run Integration Tests

Test complex integration scenarios:

```bash
./scripts/docker-integration-tests.sh
```

This tests:
- NeuronAgent querying NeuronDB
- NeuronMCP querying NeuronDB
- NeuronDesktop querying NeuronDB
- NeuronDesktop proxying to NeuronAgent
- NeuronDesktop spawning NeuronMCP
- End-to-end workflows

## Service Details

### NeuronDB (PostgreSQL with Extension)

- **Container**: `neurondb-cpu`
- **Port**: `5433` (host) → `5432` (container)
- **Database**: `neurondb`
- **User**: `neurondb`
- **Password**: `neurondb` (default, change for production)
- **Connection**: `postgresql://neurondb:neurondb@localhost:5433/neurondb`

**Test Connection:**
```bash
docker compose exec neurondb psql -U neurondb -d neurondb -c "SELECT neurondb.version();"
```

### NeuronAgent (REST API Server)

- **Container**: `neuronagent`
- **Port**: `8080`
- **Health**: `http://localhost:8080/health`
- **API**: `http://localhost:8080/api/v1/`

**Test Connection:**
```bash
curl http://localhost:8080/health
```

### NeuronMCP (MCP Protocol Server)

- **Container**: `neurondb-mcp`
- **Protocol**: stdio (JSON-RPC 2.0)
- **Binary**: `/app/neuronmcp` (in container)

**Test Connection:**
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}' | \
docker compose exec -T neurondb-mcp /app/neuronmcp
```

### NeuronDesktop (Web Interface)

- **API Container**: `neurondesk-api`
- **Frontend Container**: `neurondesk-frontend`
- **API Port**: `8081`
- **Frontend Port**: `3000`
- **API Health**: `http://localhost:8081/health`
- **Frontend**: `http://localhost:3000`

**Test Connection:**
```bash
# API
curl http://localhost:8081/health

# Frontend
curl http://localhost:3000
```

## Verification Scripts

### Basic Verification

```bash
./scripts/verify-docker-ecosystem.sh
```

Options:
- `--verbose`: Show detailed output
- `--skip-service SERVICE`: Skip checking a specific service

Example:
```bash
./scripts/verify-docker-ecosystem.sh --verbose --skip-service neuronmcp
```

### Integration Tests

```bash
./scripts/docker-integration-tests.sh
```

Options:
- `--verbose`: Show detailed output
- `--skip-test TEST_NAME`: Skip a specific test

Available tests:
- `neurondb-agent`: Test NeuronAgent querying NeuronDB
- `neurondb-mcp`: Test NeuronMCP querying NeuronDB
- `desktop-db`: Test NeuronDesktop querying NeuronDB
- `desktop-agent`: Test NeuronDesktop proxying to NeuronAgent
- `desktop-mcp`: Test NeuronDesktop spawning NeuronMCP
- `e2e-workflow`: End-to-end workflow test

Example:
```bash
./scripts/docker-integration-tests.sh --verbose --skip-test desktop-mcp
```

## Service Management

### View Service Status

```bash
docker compose ps
```

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f neurondb
docker compose logs -f neuronagent
docker compose logs -f neurondb-mcp
docker compose logs -f neurondesk-api
docker compose logs -f neurondesk-frontend
```

### Stop Services

```bash
# Stop all services
docker compose down

# Stop specific service
docker compose stop neurondb
```

### Restart Services

```bash
# Restart all services
docker compose restart

# Restart specific service
docker compose restart neurondesk-api
```

### Rebuild Services

```bash
# Rebuild all services
docker compose build

# Rebuild specific service
docker compose build neurondesk-api

# Rebuild without cache
docker compose build --no-cache neurondesk-api
```

## Configuration

### Environment Variables

Create a `.env` file in the repository root to customize settings:

```env
# NeuronDB Configuration
POSTGRES_USER=neurondb
POSTGRES_PASSWORD=your-secure-password
POSTGRES_DB=neurondb
POSTGRES_PORT=5433

# NeuronAgent Configuration
SERVER_PORT=8080
DB_HOST=neurondb
DB_PORT=5432
DB_NAME=neurondb
DB_USER=neurondb
DB_PASSWORD=your-secure-password

# NeuronDesktop Configuration
NEURONDB_HOST=neurondb
NEURONDB_PORT=5432
NEURONDB_DATABASE=neurondb
NEURONDB_USER=neurondb
NEURONDB_PASSWORD=your-secure-password
NEURONAGENT_ENDPOINT=http://neuronagent:8080
NEURONAGENT_API_KEY=your-api-key

# Logging
LOG_LEVEL=info
LOG_FORMAT=json
```

### NeuronMCP Binary in NeuronDesktop

NeuronDesktop needs the NeuronMCP binary to spawn MCP processes. The Dockerfile automatically copies the binary from the `neurondb-mcp` image during build.

**Important**: Build the `neuronmcp` service before building `neurondesk-api`:

```bash
# Build neuronmcp first
docker compose build neuronmcp

# Then build neurondesk-api (it will copy the binary)
docker compose build neurondesk-api
```

Or build all services together (they'll build in the correct order):

```bash
docker compose build
```

## Troubleshooting

### Services Won't Start

**Check logs:**
```bash
docker compose logs neurondb
docker compose logs neuronagent
docker compose logs neurondesk-api
```

**Common issues:**
- Port conflicts: Change ports in `docker-compose.yml` or `.env`
- Out of memory: Increase Docker memory limit
- Build failures: Check Docker has sufficient disk space

### Services Start But Health Checks Fail

**Wait for initialization:**
- NeuronDB may take 30-60 seconds to initialize
- NeuronAgent waits for NeuronDB to be ready
- NeuronDesktop waits for all dependencies

**Check service health:**
```bash
docker compose ps
```

All services should show "healthy" status.

### NeuronDesktop Cannot Spawn NeuronMCP

**Check binary exists:**
```bash
docker compose exec neurondesk-api ls -la /usr/local/bin/neurondb-mcp
```

**If missing, rebuild:**
```bash
# Build neuronmcp first
docker compose build neuronmcp

# Then rebuild neurondesk-api
docker compose build neurondesk-api
```

### Network Connectivity Issues

**Verify services are on the same network:**
```bash
docker network inspect neurondb-network
```

**Test connectivity:**
```bash
# From neuronagent to neurondb
docker compose exec neuronagent nc -z neurondb 5432

# From neurondesk-api to neuronagent
docker compose exec neurondesk-api nc -z neuronagent 8080
```

### Database Connection Errors

**Verify database is ready:**
```bash
docker compose exec neurondb pg_isready -U neurondb
```

**Check connection string:**
- Ensure `DB_HOST=neurondb` (container name, not localhost)
- Ensure `DB_PORT=5432` (container port, not host port)
- Verify password matches in all services

### Verification Scripts Fail

**Run with verbose output:**
```bash
./scripts/verify-docker-ecosystem.sh --verbose
```

**Check individual services:**
```bash
# Test NeuronDB
docker compose exec neurondb psql -U neurondb -d neurondb -c "SELECT neurondb.version();"

# Test NeuronAgent
curl http://localhost:8080/health

# Test NeuronDesktop API
curl http://localhost:8081/health
```

## Advanced Usage

### GPU Support

**CUDA:**
```bash
docker compose --profile cuda up -d
```

**ROCm:**
```bash
docker compose --profile rocm up -d
```

**Metal (Apple Silicon):**
```bash
docker compose --profile metal up -d
```

### Custom Build Arguments

```bash
docker compose build \
  --build-arg PG_MAJOR=18 \
  --build-arg VERSION=1.0.0 \
  neurondb
```

### Running Individual Services

```bash
# Start only NeuronDB
docker compose up -d neurondb

# Start NeuronDB and NeuronAgent
docker compose up -d neurondb neuronagent
```

### Accessing Services from Host

All services are accessible from the host machine:

- **NeuronDB**: `localhost:5433`
- **NeuronAgent**: `http://localhost:8080`
- **NeuronDesktop API**: `http://localhost:8081`
- **NeuronDesktop Frontend**: `http://localhost:3000`

### Accessing Services from Containers

Services communicate using container names:

- **NeuronDB**: `neurondb:5432`
- **NeuronAgent**: `neuronagent:8080`
- **NeuronMCP**: `neurondb-mcp` (stdio, not network)

## Cleanup

### Stop and Remove Containers

```bash
# Stop containers (keeps data)
docker compose down

# Stop and remove volumes (deletes data)
docker compose down -v
```

### Remove Images

```bash
# Remove all NeuronDB ecosystem images
docker rmi neurondb:cpu-pg17 neuronagent:latest neurondb-mcp:latest neurondesk-api:latest neurondesk-frontend:latest
```

### Full Cleanup

```bash
# Stop and remove everything
docker compose down -v

# Remove images
docker compose down --rmi all

# Remove network (if it persists)
docker network rm neurondb-network 2>/dev/null || true
```

## Best Practices

1. **Use `.env` file**: Never commit passwords to version control
2. **Check health**: Always verify services are healthy before use
3. **Monitor logs**: Use `docker compose logs -f` to monitor services
4. **Resource limits**: Adjust CPU/memory limits in `docker-compose.yml` based on your system
5. **Backup data**: NeuronDB data is in Docker volumes - back them up regularly
6. **Build order**: Build `neuronmcp` before `neurondesk-api` to ensure binary is available

## Related Documentation

- [Docker Unified Guide](docker-unified.md) - Unified Docker orchestration
- [NeuronDB Docker Guide](../../dockers/neurondb/readme.md) - NeuronDB-specific Docker info
- [NeuronAgent Docker Guide](../../dockers/neuronagent/readme.md) - NeuronAgent-specific Docker info
- [NeuronMCP Docker Guide](../../dockers/neuronmcp/readme.md) - NeuronMCP-specific Docker info
- [Quick Start Guide](../../QUICKSTART.md) - Quick start instructions

## Support

For issues and questions:
- Check service logs: `docker compose logs [service-name]`
- Run verification script: `./scripts/verify-docker-ecosystem.sh --verbose`
- GitHub Issues: [Report Issues](https://github.com/neurondb/NeurondB/issues)
- Documentation: [Full Documentation](https://neurondb.ai/docs)


