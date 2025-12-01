# NeuronDB Docker Images

This directory contains comprehensive, modular Docker builds for NeuronDB supporting multiple PostgreSQL versions, GPU backends, and architectures.

## Overview

NeuronDB Docker images are provided in four variants:

- **CPU** (`Dockerfile`) – CPU-only image based on `postgres:${PG_MAJOR}-bookworm`
- **CUDA** (`Dockerfile.gpu.cuda`) – NVIDIA CUDA GPU support
- **ROCm** (`Dockerfile.gpu.rocm`) – AMD ROCm GPU support
- **Metal** (`Dockerfile.gpu.metal`) – Apple Silicon Metal GPU support

All variants support:
- **PostgreSQL**: 16, 17, 18 (configurable via `PG_MAJOR` build arg)
- **Architectures**: amd64, arm64 (automatic detection)
- **ONNX Runtime**: Configurable version (default: 1.17.0)

## Quick Start

### CPU Image

```bash
cd Neurondb/docker

# Build and run CPU image (PostgreSQL 17)
docker compose build neurondb
docker compose up neurondb

# Or build directly
docker build -f docker/Dockerfile --build-arg PG_MAJOR=17 -t neurondb:17-cpu ..
```

Connect:

```bash
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" -c '\dx neurondb'
```

### CUDA GPU Image

**Requirements**: NVIDIA driver ≥ 535, `nvidia-container-toolkit`, Docker Compose v2.

```bash
# Build and run CUDA image
docker compose --profile cuda build neurondb-cuda
docker compose --profile cuda up neurondb-cuda

# With RAPIDS support (optional)
docker compose --profile cuda build neurondb-cuda \
  --build-arg ENABLE_RAPIDS=1 \
  --build-arg PG_MAJOR=17

# Or build directly
docker build -f docker/Dockerfile.gpu.cuda \
  --build-arg PG_MAJOR=17 \
  --build-arg CUDA_VERSION=12.4.1 \
  --build-arg ENABLE_RAPIDS=0 \
  -t neurondb:17-cuda ..
```

Verify GPU:

```bash
docker exec -it neurondb-cuda nvidia-smi
psql "postgresql://neurondb:neurondb@localhost:5434/neurondb" \
  -c "SELECT neurondb.gpu_device_info();"
```

### ROCm GPU Image

**Requirements**: AMD GPU with ROCm drivers, Docker with device access.

```bash
# Build and run ROCm image
docker compose --profile rocm build neurondb-rocm
docker compose --profile rocm up neurondb-rocm

# Or build directly
docker build -f docker/Dockerfile.gpu.rocm \
  --build-arg PG_MAJOR=17 \
  --build-arg ROCM_VERSION=5.7 \
  -t neurondb:17-rocm ..
```

### Metal GPU Image (macOS/Apple Silicon)

**Requirements**: macOS with Apple Silicon, Docker Desktop.

```bash
# Build and run Metal image (arm64 only)
docker compose --profile metal build neurondb-metal
docker compose --profile metal up neurondb-metal

# Or build directly with buildx for arm64
docker buildx build --platform linux/arm64 \
  -f docker/Dockerfile.gpu.metal \
  --build-arg PG_MAJOR=17 \
  -t neurondb:17-metal ..
```

## Build Arguments

### Common Arguments (All Dockerfiles)

| Argument | Default | Description |
|----------|---------|-------------|
| `PG_MAJOR` | `17` | PostgreSQL major version: `16`, `17`, or `18` |
| `ONNX_VERSION` | `1.17.0` | ONNX Runtime version to embed |

### CUDA-Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `CUDA_VERSION` | `12.4.1` | CUDA toolkit version |
| `ENABLE_RAPIDS` | `0` | Enable RAPIDS/cuML stack: `0` or `1` |

### ROCm-Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `ROCM_VERSION` | `5.7` | ROCm version |

## Architecture Support

All Dockerfiles automatically detect and support both architectures:

- **amd64** (x86_64): Intel/AMD processors
- **arm64** (aarch64): ARM processors, Apple Silicon

The build process automatically:
- Detects the target architecture
- Downloads the appropriate ONNX Runtime build
- Configures library paths correctly

### Multi-Architecture Builds

Build for multiple architectures using Docker Buildx:

```bash
# Create buildx builder (if not exists)
docker buildx create --name multiarch --use

# Build for both architectures
docker buildx build --platform linux/amd64,linux/arm64 \
  -f docker/Dockerfile \
  --build-arg PG_MAJOR=17 \
  -t neurondb:17-cpu \
  --push .  # or --load for local use
```

## PostgreSQL Version Selection

All Dockerfiles support PostgreSQL 16, 17, and 18. Specify the version via build arg:

```bash
# PostgreSQL 16
docker build -f docker/Dockerfile --build-arg PG_MAJOR=16 -t neurondb:16-cpu ..

# PostgreSQL 17 (default)
docker build -f docker/Dockerfile --build-arg PG_MAJOR=17 -t neurondb:17-cpu ..

# PostgreSQL 18
docker build -f docker/Dockerfile --build-arg PG_MAJOR=18 -t neurondb:18-cpu ..
```

## Docker Compose Profiles

The `docker-compose.yml` file includes profiles for easy service management:

```bash
# CPU (default, no profile needed)
docker compose up neurondb

# CUDA
docker compose --profile cuda up neurondb-cuda

# ROCm
docker compose --profile rocm up neurondb-rocm

# Metal
docker compose --profile metal up neurondb-metal

# All GPU variants
docker compose --profile gpu up
```

### Ports

Each service uses a different port to avoid conflicts:

- **CPU**: `5433`
- **CUDA**: `5434`
- **ROCm**: `5435`
- **Metal**: `5436`

## Configuration

### Automatic Extension Creation

An initialization script (`docker-entrypoint-initdb.d/20_create_neurondb.sql`) automatically creates the NeuronDB extension on first boot.

### PostgreSQL Configuration

During `initdb`, the container sets the following defaults in `postgresql.conf`:

```conf
shared_preload_libraries = 'neurondb'
neurondb.gpu_enabled = off
neurondb.automl.use_gpu = off
```

Modify at runtime:

```sql
ALTER SYSTEM SET neurondb.gpu_enabled = on;
SELECT pg_reload_conf();
```

Or edit directly:

```bash
docker exec -it neurondb-cuda vi /var/lib/postgresql/data/postgresql.conf
docker exec -it neurondb-cuda pg_ctl reload
```

### Environment Variables

Standard PostgreSQL environment variables are supported:

- `POSTGRES_USER` (default: `postgres`)
- `POSTGRES_PASSWORD` (default: `postgres`)
- `POSTGRES_DB` (default: `postgres`)
- `POSTGRES_INITDB_ARGS`

GPU-specific variables:

- `NVIDIA_VISIBLE_DEVICES` (CUDA): GPU device selection
- `NVIDIA_DRIVER_CAPABILITIES` (CUDA): Driver capabilities
- `NEURONDB_GPU_ENABLED`: Enable GPU features

## Advanced Usage

### Custom Build with RAPIDS

Build CUDA image with RAPIDS/cuML support:

```bash
docker build -f docker/Dockerfile.gpu.cuda \
  --build-arg PG_MAJOR=17 \
  --build-arg ENABLE_RAPIDS=1 \
  -t neurondb:17-cuda-rapids ..
```

### Multi-Stage Build Optimization

All Dockerfiles use multi-stage builds:
- **Builder stage**: Compiles NeuronDB with all dependencies
- **Runtime stage**: Minimal image with only runtime dependencies

This results in smaller final images (~500MB-2GB depending on variant).

### Volume Management

Each variant uses separate volumes:

- `neurondb-data`: CPU variant
- `neurondb-cuda-data`: CUDA variant
- `neurondb-rocm-data`: ROCm variant
- `neurondb-metal-data`: Metal variant

Persistent data is stored in Docker volumes and persists across container restarts.

## Troubleshooting

### GPU Not Detected (CUDA)

```bash
# Verify NVIDIA driver
nvidia-smi

# Verify nvidia-container-toolkit
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

# Check container GPU access
docker exec -it neurondb-cuda nvidia-smi
```

### GPU Not Detected (ROCm)

```bash
# Verify ROCm installation
rocm-smi

# Check device access
docker exec -it neurondb-rocm ls -la /dev/kfd /dev/dri
```

### Build Failures

**Out of disk space**: GPU builds require ~4-8GB during compilation. Clean up:

```bash
docker system prune -a
```

**Architecture mismatch**: Ensure you're building for the correct architecture:

```bash
docker buildx inspect --bootstrap
```

**PostgreSQL version not found**: Verify the PG_MAJOR version is supported (16, 17, or 18).

### Connection Issues

**Port already in use**: Change the port mapping in `docker-compose.yml`:

```yaml
ports:
  - "5437:5432"  # Use different host port
```

**Extension not found**: Ensure the extension was built correctly:

```bash
docker exec -it neurondb psql -U neurondb -c '\dx'
```

## Lifecycle

### Building Images

```bash
# Build all variants
docker compose build

# Build specific variant
docker compose build neurondb-cuda

# Build with custom args
docker compose build neurondb-cuda --build-arg PG_MAJOR=18
```

### Running Containers

```bash
# Start in foreground
docker compose up neurondb

# Start in background
docker compose up -d neurondb

# View logs
docker compose logs -f neurondb
```

### Stopping and Cleaning

```bash
# Stop containers
docker compose down

# Remove volumes (WARNING: deletes data)
docker compose down -v

# Remove images
docker compose down --rmi all
```

## Health Checks

All services include health checks using `pg_isready`:

- **Interval**: 30 seconds
- **Timeout**: 5 seconds
- **Retries**: 5

Check health status:

```bash
docker compose ps
docker inspect neurondb-cuda | jq '.[0].State.Health'
```

## Notes

- **Disk Space**: GPU builds require significant disk space (4-8GB) during compilation
- **Build Time**: GPU images take longer to build (10-30 minutes depending on hardware)
- **Runtime**: Images inherit PostgreSQL entrypoint; standard PostgreSQL environment variables work
- **Metal Support**: Metal GPU support requires macOS host with Docker Desktop
- **Multi-Arch**: Use Docker Buildx for multi-architecture builds
- **RAPIDS**: RAPIDS support adds ~2GB to CUDA images and requires CUDA 12.x

## Examples

### Build for Production

```bash
# Build optimized CPU image for PostgreSQL 18
docker build -f docker/Dockerfile \
  --build-arg PG_MAJOR=18 \
  --build-arg ONNX_VERSION=1.17.0 \
  -t neurondb:18-cpu-prod \
  --target builder \
  ..
```

### Development Workflow

```bash
# Start development environment
docker compose up -d neurondb

# Connect and test
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb"

# Rebuild after code changes
docker compose build neurondb
docker compose up -d --force-recreate neurondb
```

### GPU Testing

```bash
# Start CUDA container
docker compose --profile cuda up -d neurondb-cuda

# Test GPU functions
psql "postgresql://neurondb:neurondb@localhost:5434/neurondb" <<EOF
SELECT neurondb.gpu_device_info();
SELECT neurondb.gpu_enabled();
EOF
```

## External Connections

NeuronDB Docker containers are designed to accept connections from other services. To connect from other Docker containers or external services:

### Connection Parameters

**Default Configuration (CPU variant):**
- **Host**: `localhost` (from host) or container name (from Docker network)
- **Port**: `5433` (mapped from container port 5432)
- **Database**: `neurondb` (or as set via `POSTGRES_DB`)
- **User**: `neurondb` (or as set via `POSTGRES_USER`)
- **Password**: `neurondb` (or as set via `POSTGRES_PASSWORD`)

### From Other Docker Containers

When connecting from other Docker containers in the same network:

```bash
# Connection string format
postgresql://neurondb:neurondb@neurondb-cpu:5432/neurondb

# Or using service name
DB_HOST=neurondb-cpu
DB_PORT=5432
DB_NAME=neurondb
DB_USER=neurondb
DB_PASSWORD=neurondb
```

### From Host Machine

When connecting from the host machine or external services:

```bash
# Connection string format
postgresql://neurondb:neurondb@localhost:5433/neurondb

# Or using environment variables
DB_HOST=localhost
DB_PORT=5433
DB_NAME=neurondb
DB_USER=neurondb
DB_PASSWORD=neurondb
```

### Network Configuration

By default, containers use the `default` bridge network. To enable service discovery between containers:

1. **Create a shared network:**
   ```bash
   docker network create neurondb-network
   ```

2. **Add NeuronDB to the network:**
   ```bash
   docker network connect neurondb-network neurondb-cpu
   ```

3. **Use container name as hostname** when connecting from other services

### Security Considerations

For production deployments:
- Change default passwords via environment variables
- Use Docker secrets for sensitive credentials
- Restrict network access using Docker networks
- Enable SSL/TLS connections if required
- Consider using connection pooling in applications

### Environment Variables for External Access

Ensure these are set appropriately:

```bash
# Allow external connections (default: enabled)
POSTGRES_HOST_AUTH_METHOD=md5  # or scram-sha-256 for better security

# Optional: configure listen addresses
# Add to postgresql.conf or via environment
```

## Support

For issues, questions, or contributions:

- **GitHub Issues**: [https://github.com/neurondb/NeurondB/issues](https://github.com/neurondb/NeurondB/issues)
- **Documentation**: [https://neurondb.ai/docs](https://neurondb.ai/docs)
- **Email**: support@neurondb.ai
