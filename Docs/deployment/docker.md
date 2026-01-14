# 🐳 Docker Deployment Guide

<div align="center">

**Complete guide for deploying the NeuronDB ecosystem using Docker**

[![Docker](https://img.shields.io/badge/docker-ready-blue)](.)
[![Difficulty](https://img.shields.io/badge/difficulty-easy-brightgreen)](.)
[![GPU Support](https://img.shields.io/badge/GPU-CUDA%20%7C%20ROCm%20%7C%20Metal-green)](.)

</div>

---

> [!TIP]
> For the fastest setup, use `docker compose up -d` from the repository root. This starts all services with defaults.

---

## 📋 Overview

Docker deployment provides the easiest and most consistent way to run all NeuronDB ecosystem components. Each component includes Docker configurations with support for CPU and GPU variants.

**Benefits:**
- ✅ **Easy setup** - No manual PostgreSQL configuration required
- ✅ **Isolated environment** - No conflicts with existing installations
- ✅ **GPU support** - CUDA, ROCm, and Metal variants available
- ✅ **Consistent** - Same environment across all machines

---

## ✅ Prerequisites

<details>
<summary><strong>📋 Prerequisites Checklist</strong></summary>

| Requirement | Minimum Version | Notes |
|-------------|----------------|-------|
| **Docker** | 20.10+ | Required for all deployments |
| **Docker Compose** | 2.0+ | Required for multi-service orchestration |
| **NVIDIA Docker** | Latest | Required for CUDA GPU support |
| **ROCm Drivers** | 5.7+ | Required for AMD GPU support |
| **macOS** | 13+ | Required for Metal GPU support (Apple Silicon) |

</details>

> [!NOTE]
> **GPU Support:** GPU variants are optional. CPU-only deployment works on any system with Docker installed.

## 🚀 Quick Start

### Start All Services

> [!TIP]
> The first run downloads Docker images. This takes 5 to 10 minutes depending on your connection speed.

```bash
# From repository root
docker compose up -d

# Wait for services to be healthy (30-60 seconds)
docker compose ps
```

**Expected output:**
```
NAME                STATUS
neurondb-cpu        healthy
neuronagent         healthy
neuronmcp           healthy
neurondesk-api      healthy
neurondesk-frontend healthy
```

### Component-Specific Deployment

<details>
<summary><strong>🐘 NeuronDB Only (Database Extension)</strong></summary>

```bash
# Start just the database
docker compose up -d neurondb

# Verify it's running
docker compose ps neurondb
```

> [!NOTE]
> This is the simplest deployment. The PostgreSQL database with NeuronDB extension. No additional services required.

</details>

<details>
<summary><strong>🤖 NeuronAgent (Agent Runtime)</strong></summary>

```bash
# Start database + agent
docker compose up -d neurondb neuronagent

# Test the API
curl http://localhost:8080/health
```

</details>

<details>
<summary><strong>🔌 NeuronMCP (MCP Server)</strong></summary>

```bash
# Start database + MCP server
docker compose up -d neurondb neuronmcp
```

> [!NOTE]
> NeuronMCP uses stdio for communication, not HTTP. It works with MCP-compatible clients like Claude Desktop.

</details>

<details>
<summary><strong>🖥️ NeuronDesktop (Web UI)</strong></summary>

```bash
# Start full stack (database + API + frontend)
docker compose up -d neurondb neurondesk-api neurondesk-frontend

# Access the web UI
open http://localhost:3000
```

</details>

## Unified Docker Orchestration

For unified orchestration of all services, see the [Unified Docker Guide](docker-unified.md) which provides:

- Single command to build and run all services
- Automatic networking between containers
- GPU variant support (CUDA, ROCm, Metal)
- Health checks and service management

## 🏗️ Architecture

<details>
<summary><strong>📐 Docker Network Architecture</strong></summary>

```mermaid
graph TB
    subgraph Network["Docker Network: neurondb-network"]
        DB[NeuronDB<br/>PostgreSQL Extension<br/>Port: 5433]
        AGENT[NeuronAgent<br/>REST/WebSocket API<br/>Port: 8080]
        MCP[NeuronMCP<br/>MCP Protocol<br/>stdio]
        DESKTOP_API[NeuronDesktop API<br/>REST API<br/>Port: 8081]
        DESKTOP_UI[NeuronDesktop UI<br/>Web Frontend<br/>Port: 3000]
    end
    
    AGENT -->|SQL| DB
    MCP -->|SQL| DB
    DESKTOP_API -->|SQL| DB
    DESKTOP_UI -->|HTTP| DESKTOP_API
    DESKTOP_API -->|HTTP| AGENT
    
    style DB fill:#e1f5ff
    style AGENT fill:#fff4e1
    style MCP fill:#e8f5e9
    style DESKTOP_API fill:#f3e5f5
    style DESKTOP_UI fill:#f3e5f5
```

</details>

> [!NOTE]
> **Network Isolation:** All services communicate via the Docker network using service names. External access is provided through mapped host ports.

## ⚙️ Configuration

### Environment Variables

Each component uses environment variables for configuration. See component-specific Docker guides:

| Component | Documentation | Key Configuration |
|-----------|---------------|-------------------|
| **NeuronDB** | [NeuronDB Docker](../../dockers/neurondb/README.md) | `POSTGRES_PASSWORD`, `POSTGRES_PORT` |
| **NeuronAgent** | [NeuronAgent Docker](../../dockers/neuronagent/README.md) | `NEURONDB_URL`, `API_KEY` |
| **NeuronMCP** | [NeuronMCP Docker](../../dockers/neuronmcp/README.md) | `NEURONDB_URL`, `MCP_CONFIG` |
| **NeuronDesktop** | [NeuronDesktop README](../../NeuronDesktop/README.md) | `API_URL`, `NEURONDB_URL` |

> [!TIP]
> Use `.env` file in the repository root for environment variables. See [`env.example`](../../env.example) for all available options.

### Network Configuration

<details>
<summary><strong>🌐 Network Communication</strong></summary>

Services communicate via Docker network using service names (not container names):

| From | To | Connection | Port |
|------|-----|------------|------|
| **NeuronAgent** | NeuronDB | `neurondb:5432` | Internal Docker network |
| **NeuronMCP** | NeuronDB | `neurondb:5432` | Internal Docker network |
| **NeuronDesktop** | NeuronDB | `neurondb:5432` | Internal Docker network |
| **External Access** | NeuronDB | `localhost:5433` | Host machine |

> [!IMPORTANT]
> Inside Docker network, use service name `neurondb`, not container name `neurondb-cpu`. From your host machine, use `localhost:5433`.

</details>

## 🎮 GPU Variants

NeuronDB supports multiple GPU backends via Docker Compose profiles:

| GPU Backend | Platform | Profile | Service Name | Port |
|-------------|----------|---------|--------------|------|
| **CUDA** | NVIDIA | `cuda` | `neurondb-cuda` | 5434 |
| **ROCm** | AMD | `rocm` | `neurondb-rocm` | 5435 |
| **Metal** | Apple Silicon | `metal` | `neurondb-metal` | 5436 |
| **CPU** | Any | `cpu` (default) | `neurondb-cpu` | 5433 |

<details>
<summary><strong>🚀 CUDA (NVIDIA)</strong></summary>

```bash
# Stop CPU services first
docker compose down

# Build and run CUDA variant
docker compose --profile cuda build neurondb-cuda
docker compose --profile cuda up -d neurondb-cuda

# Verify CUDA support
docker compose exec neurondb-cuda psql -U neurondb -d neurondb -c "SELECT neurondb.gpu_backend();"
```

> [!NOTE]
> **Requirements:** NVIDIA GPU with CUDA 12.2+ and `nvidia-container-toolkit` installed.

</details>

<details>
<summary><strong>🚀 ROCm (AMD)</strong></summary>

```bash
# Stop CPU services first
docker compose down

# Build and run ROCm variant
docker compose --profile rocm build neurondb-rocm
docker compose --profile rocm up -d neurondb-rocm

# Verify ROCm support
docker compose exec neurondb-rocm psql -U neurondb -d neurondb -c "SELECT neurondb.gpu_backend();"
```

> [!NOTE]
> **Requirements:** AMD GPU with ROCm 5.7+ and proper device access configured.

</details>

<details>
<summary><strong>🚀 Metal (Apple Silicon)</strong></summary>

```bash
# Stop CPU services first
docker compose down

# Build and run Metal variant
docker compose --profile metal build neurondb-metal
docker compose --profile metal up -d neurondb-metal

# Verify Metal support
docker compose exec neurondb-metal psql -U neurondb -d neurondb -c "SELECT neurondb.gpu_backend();"
```

> [!NOTE]
> **Requirements:** Apple Silicon (M1/M2/M3) Mac with macOS 13+ (Ventura or later).

</details>

## 🔧 Service Management

<details>
<summary><strong>▶️ Start Services</strong></summary>

```bash
# Start all services
docker compose up -d

# Start specific services
docker compose up -d neurondb neuronagent
```

</details>

<details>
<summary><strong>⏹️ Stop Services</strong></summary>

```bash
# Stop all services (keeps containers)
docker compose stop

# Stop specific services
docker compose stop neurondb
```

</details>

<details>
<summary><strong>📋 View Logs</strong></summary>

```bash
# All services (follow mode)
docker compose logs -f

# Specific service
docker compose logs -f neurondb
docker compose logs -f neuronagent
docker compose logs -f neuronmcp
docker compose logs -f neurondesk-api

# Last 100 lines
docker compose logs --tail=100 neurondb
```

</details>

<details>
<summary><strong>🔄 Restart Services</strong></summary>

```bash
# Restart all services
docker compose restart

# Restart specific service
docker compose restart neurondb
```

</details>

<details>
<summary><strong>🗑️ Remove Services</strong></summary>

```bash
# Stop and remove containers (keeps volumes)
docker compose down

# Remove containers, networks, and volumes (⚠️ deletes data!)
docker compose down -v
```

> [!WARNING]
> **Data Loss:** Using `docker compose down -v` will delete all data volumes. Make sure you have backups before running this command.

</details>

## 🏥 Health Checks

### Verify Services

<details>
<summary><strong>✅ Service Status Check</strong></summary>

```bash
# Check container status
docker compose ps

# Expected output shows all services as "healthy"
```

</details>

<details>
<summary><strong>🐘 NeuronDB Health Check</strong></summary>

```bash
# Test NeuronDB connection and version
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" \
  -c "SELECT neurondb.version();"

# Expected output: version number (e.g., "3.0.0-devel")
```

</details>

<details>
<summary><strong>🤖 NeuronAgent Health Check</strong></summary>

```bash
# Test NeuronAgent API
curl http://localhost:8080/health

# Expected output: {"status":"ok"}
```

</details>

<details>
<summary><strong>🖥️ NeuronDesktop Health Check</strong></summary>

```bash
# Test NeuronDesktop API
curl http://localhost:8081/health

# Expected output: {"status":"ok"}
```

</details>

## 💾 Data Persistence

### Volumes

Docker volumes are used for data persistence:

| Service | Volume Name | Purpose |
|---------|-------------|---------|
| **NeuronDB** | `neurondb_data` | PostgreSQL data directory |
| **NeuronDesktop** | `neurondesk_data` | Database and application data |

> [!NOTE]
> **Volume Location:** Volumes are stored in Docker's volume directory. Use `docker volume ls` to list all volumes.

### Backup

<details>
<summary><strong>💾 Backup Procedures</strong></summary>

```bash
# Backup NeuronDB data
docker exec neurondb-cpu pg_dump -U neurondb neurondb > backup_$(date +%Y%m%d).sql

# Backup NeuronDesktop data
docker exec neurondesk-api pg_dump -U neurondesk neurondesk > neurondesk_backup_$(date +%Y%m%d).sql
```

> [!TIP]
> **Automated Backups:** For production deployments, consider using the [Backup and Restore Guide](backup-restore.md) for automated backup strategies.

</details>

## 🔧 Troubleshooting

<details>
<summary><strong>❌ Services Cannot Connect</strong></summary>

**Symptoms:** Services fail to connect to each other or to the database.

**Solutions:**

1. **Verify containers are running:**
   ```bash
   docker compose ps
   ```

2. **Check network connectivity:**
   ```bash
   docker network inspect neurondb-network
   ```

3. **Verify environment variables:**
   ```bash
   docker compose config
   ```

4. **Check logs for errors:**
   ```bash
   docker compose logs neurondb
   docker compose logs neuronagent
   ```

</details>

<details>
<summary><strong>🚫 Port Conflicts</strong></summary>

**Symptoms:** Port already in use errors when starting services.

**Solutions:**

1. **Change ports in `.env` file:**
   ```bash
   POSTGRES_PORT=5434  # Change from default 5433
   ```

2. **Or modify `docker-compose.yml`:**
   ```yaml
   ports:
     - "5434:5432"  # Change external port
   ```

3. **Check what's using the port:**
   ```bash
   # Linux/macOS
   lsof -i :5433
   
   # Or use netstat
   netstat -tuln | grep 5433
   ```

</details>

<details>
<summary><strong>🎮 GPU Not Detected</strong></summary>

**Symptoms:** GPU features not working despite having GPU hardware.

**Solutions:**

1. **CUDA (NVIDIA):**
   ```bash
   # Verify NVIDIA Docker runtime
   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```

2. **ROCm (AMD):**
   ```bash
   # Check devices
   ls -la /dev/kfd /dev/dri
   ```

3. **Verify Docker Compose profiles:**
   ```bash
   docker compose --profile cuda config
   ```

> [!NOTE]
> **GPU Support:** See [GPU Feature Matrix](../../gpu/gpu_feature_matrix.md) for detailed GPU support information.

</details>

## 💡 Best Practices

<details>
<summary><strong>✅ Production Best Practices</strong></summary>

| Practice | Description | Priority |
|----------|-------------|----------|
| **Use `.env` files** | Store environment variables in `.env` files (never commit secrets) | ⚠️ Critical |
| **Health checks** | Monitor service health regularly | ⭐ High |
| **Resource limits** | Set appropriate CPU and memory limits | ⭐ High |
| **Backup strategy** | Implement regular backup procedures | ⚠️ Critical |
| **Network security** | Use Docker networks to isolate services | ⭐ High |
| **Secrets management** | Use Docker secrets or external secret management | ⚠️ Critical |

> [!IMPORTANT]
> **Security:** Never commit `.env` files with production secrets to version control. Use secret management systems in production.

</details>

---

## 🔗 Related Documentation

| Document | Description |
|----------|-------------|
| **[Unified Docker Guide](docker-unified.md)** | Complete unified orchestration guide |
| **[NeuronDB Docker](../../dockers/neurondb/README.md)** | NeuronDB-specific Docker guide |
| **[NeuronAgent Docker](../../dockers/neuronagent/README.md)** | NeuronAgent Docker guide |
| **[NeuronMCP Docker](../../dockers/neuronmcp/README.md)** | NeuronMCP Docker guide |
| **[Ecosystem Integration](../ecosystem/integration.md)** | Component integration guide |
| **[Backup and Restore](backup-restore.md)** | Backup and restore procedures |
| **[Troubleshooting](../operations/troubleshooting.md)** | Comprehensive troubleshooting guide |

---

## 🌐 Official Documentation

For comprehensive Docker deployment guides and latest updates:

**🌐 [https://www.neurondb.ai/docs/docker](https://www.neurondb.ai/docs/docker)**

---

<div align="center">

[⬆ Back to Top](#-docker-deployment-guide) · [📚 Deployment Index](README.md) · [📚 Main Documentation](../../README.md)

</div>

