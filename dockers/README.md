# NeuronDB Docker Management

This directory contains all Docker-related files for the NeuronDB ecosystem, organized by component.

## Directory Structure

```
dockers/
├── docker.sh              # Main management script
├── docker-compose.yml     # Unified compose file for all services
├── .dockerignore          # Global docker ignore rules
├── neurondb/             # NeuronDB database service files
├── neuronagent/          # NeuronAgent service files
├── neuronmcp/            # NeuronMCP service files
└── neurondesktop/        # NeuronDesktop service files
```

## Quick Start

> **⚠️ Important:** The canonical `docker-compose.yml` file is at the repository root.
> 
> For new users, simply run from the repository root:
> ```bash
> docker compose up -d
> ```
> 
> This `dockers/` directory contains component-specific Docker files and the `docker.sh` management script.

### Using docker.sh (Recommended)

The `docker.sh` script provides a clean interface for managing all services:

```bash
# Show version and help
./docker.sh --version          # or -v
./docker.sh --help             # or -h

# List available services and profiles
./docker.sh --list

# Build services
./docker.sh --build neurondb --profile cpu
./docker.sh --build neuronagent --profile cuda
./docker.sh --build neuronmcp --profile rocm

# Run services
./docker.sh --run neurondb --profile cpu
./docker.sh --run neuronagent --profile cuda

# Build/run multiple services
./docker.sh neurondb neuronagent --build --profile cpu
./docker.sh neurondb neuronmcp --run --profile cuda

# Build/run all services
./docker.sh --all --build --profile default
./docker.sh --all --run --profile cpu
```

### Direct Docker Compose (Advanced)

**Recommended:** Use the root `docker-compose.yml` (canonical):
```bash
# From project root (recommended)
docker compose up -d
```

**If you're currently in `dockers/`** and want to run the canonical compose without changing directories:
```bash
# From dockers/
docker compose -f ../docker-compose.yml up -d
```

**Legacy/Reference:** `dockers/docker-compose.yml` exists for historical reasons, but it may lag behind the canonical root file.
Prefer the root compose above to avoid configuration drift.

# CUDA GPU profile
docker-compose -f dockers/docker-compose.yml --profile cuda up -d

# ROCm GPU profile
docker-compose -f dockers/docker-compose.yml --profile rocm up -d

# Metal GPU profile (macOS)
docker-compose -f dockers/docker-compose.yml --profile metal up -d
```

## Available Services

| Service | Description | Ports |
|---------|-------------|-------|
| `neurondb` | PostgreSQL with NeuronDB extension | 5433 (cpu), 5434 (cuda), 5435 (rocm), 5436 (metal) |
| `neuronagent` | AI agent service | 8080 |
| `neuronmcp` | Model Context Protocol server | stdio |
| `neurondesktop` | Web-based management UI | 3000 (frontend), 8081 (api) |

## Available Profiles

| Profile | Description | GPU Support |
|---------|-------------|-------------|
| `cpu` | CPU-only, default configuration | No |
| `default` | Alias for cpu | No |
| `cuda` | NVIDIA CUDA GPU acceleration | CUDA 12.x |
| `rocm` | AMD ROCm GPU acceleration | ROCm 5.7+ |
| `metal` | Apple Metal GPU acceleration | Metal (macOS) |

## Examples

### Example 1: Development Setup (CPU)

```bash
# Build and run CPU-based stack
./docker.sh --all --build --profile cpu
./docker.sh --all --run --profile cpu

# Check status
docker ps
```

### Example 2: GPU Workstation (CUDA)

```bash
# Build CUDA-enabled services
./docker.sh neurondb neuronagent --build --profile cuda

# Run services
./docker.sh neurondb neuronagent --run --profile cuda

# Verify GPU is detected
docker exec neurondb-cuda nvidia-smi
```

### Example 3: Individual Service Management

```bash
# Build only NeuronDB
./docker.sh --build neurondb --profile cpu

# Run only NeuronDB
./docker.sh --run neurondb --profile cpu

# Check database
docker exec neurondb-cpu psql -U neurondb -c "SELECT version();"
```

### Example 4: Multiple Profiles

```bash
# Run CPU and CUDA side-by-side (different ports)
./docker.sh --run neurondb --profile cpu   # Port 5433
./docker.sh --run neurondb --profile cuda  # Port 5434

# Both accessible simultaneously
psql -h localhost -p 5433 -U neurondb  # CPU instance
psql -h localhost -p 5434 -U neurondb  # CUDA instance
```

## Script Features

### Command-Line Options

```
Usage: docker.sh [OPTIONS] [SERVICE...]

Options:
    --version, -v              Show script version and docker info
    --help, -h                 Show this help message
    --list                     List available services and profiles
    --build                    Build container(s)
    --run                      Run container(s)
    --all                      Apply action to all services
    --profile PROFILE          Specify profile (cpu, cuda, rocm, metal, default)

Services:
    neurondb                   NeuronDB database service
    neuronagent                NeuronAgent service
    neuronmcp                  NeuronMCP service
    neurondesktop              NeuronDesktop service
```

### Validation

The script includes comprehensive validation:

- **Service validation**: Ensures service names are valid
- **Profile validation**: Ensures profile names are valid
- **Action validation**: Requires either --build or --run
- **Error handling**: Clear error messages with suggestions

### Features

- ✅ **Color-coded output**: Errors (red), success (green), info (blue)
- ✅ **Docker Compose v1 & v2**: Automatically detects and uses correct syntax
- ✅ **Profile mapping**: Maps services to compose service names with profiles
- ✅ **Multiple services**: Build/run multiple services in one command
- ✅ **All services**: Build/run entire stack with --all flag
- ✅ **Context-aware**: Runs from project root for correct build contexts

## Container Management

### Stop Services

```bash
# Stop individual service
docker stop neurondb-cpu

# Stop all services
docker compose -f ../docker-compose.yml down

# Stop and remove volumes
docker compose -f ../docker-compose.yml down -v
```

### View Logs

```bash
# View logs for a service
docker logs neurondb-cpu
docker logs -f neuronagent  # Follow logs

# View compose logs
docker compose -f ../docker-compose.yml logs neurondb
```

### Health Checks

```bash
# Check container health
docker ps --filter "name=neuron" --format "table {{.Names}}\t{{.Status}}"

# Check database health
docker exec neurondb-cpu pg_isready -U neurondb

# Check agent health
curl http://localhost:8080/health
```

## Environment Variables

Key environment variables (set in `.env` or shell):

```bash
# PostgreSQL
POSTGRES_USER=neurondb
POSTGRES_PASSWORD=neurondb
POSTGRES_DB=neurondb
POSTGRES_PORT=5433

# Build arguments
PG_MAJOR=17
PACKAGE_VERSION=3.0.0-devel
ONNX_VERSION=1.17.0

# GPU
NVIDIA_VISIBLE_DEVICES=all
NEURONDB_GPU_ENABLED=on
NEURONDB_GPU_BACKEND_TYPE=1  # 0=CPU, 1=CUDA, 2=ROCm, 3=Metal
```

## Troubleshooting

### Build Fails

```bash
# Clean and rebuild
docker-compose -f dockers/docker-compose.yml build --no-cache neurondb

# Check build context
docker-compose -f dockers/docker-compose.yml config
```

### Container Won't Start

```bash
# Check logs
docker logs neurondb-cpu

# Check if port is in use
lsof -i :5433

# Remove existing container
docker rm -f neurondb-cpu
```

### GPU Not Detected

```bash
# CUDA: Check nvidia-docker
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# ROCm: Check devices
docker run --rm --device=/dev/kfd --device=/dev/dri rocm/rocm-terminal rocm-smi
```

## Component Details

### NeuronDB

- **Base Image**: postgres:17-bookworm
- **Extensions**: NeuronDB, vector, plpython3u
- **ML Libraries**: ONNX Runtime, transformers, catboost
- **Variants**: CPU, CUDA, ROCm, Metal

### NeuronAgent

- **Language**: Go 1.23+
- **Frameworks**: OpenAPI 3.0, REST API
- **Dependencies**: PostgreSQL client

### NeuronMCP

- **Language**: Go 1.23+
- **Protocol**: Model Context Protocol
- **Transport**: stdio

### NeuronDesktop

- **Frontend**: Next.js, React, TypeScript
- **Backend**: Go, REST API
- **Features**: SQL console, query builder, schema browser

## Best Practices

1. **Use profiles**: Always specify a profile matching your hardware
2. **Health checks**: Wait for services to be healthy before connecting
3. **Resource limits**: Adjust CPU/memory limits in docker-compose.yml
4. **Volumes**: Use named volumes for data persistence
5. **Networks**: Services auto-connect via neurondb-network
6. **Logs**: Enable log rotation to prevent disk space issues

## Testing

Comprehensive testing of the docker.sh script:

```bash
# All tests passed ✓
# - Version and help commands
# - Service listing
# - Build commands (all services × all profiles)
# - Run commands (all services × all profiles)
# - Multiple service handling
# - All services flag
# - Input validation
# - Error handling
```

## Version

- **Script Version**: 1.0.0
- **Docker Compose**: v2 format (compatible with v1 and v2)
- **Last Updated**: 2025-12-31

## Support

For issues, questions, or contributions:

- Check logs: `docker logs [container-name]`
- Validate compose file: `docker-compose -f dockers/docker-compose.yml config`
- Report issues: Include output of `./docker.sh --version`


