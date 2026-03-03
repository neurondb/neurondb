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

All commands use package-based builds for faster, more reproducible images. Run commands from the repository root directory.

### CPU Image

**Build and Run:**

```bash
# From repository root
cd /path/to/neurondb

# Build CPU image (PostgreSQL 17)
sudo docker build -f NeuronDB/docker/Dockerfile.package \
  --build-arg PG_MAJOR=17 \
  --build-arg PACKAGE_VERSION=3.0.0-devel \
  -t neurondb:cpu-package-pg17 .

# Start container
sudo docker run -d --name neurondb-cpu \
  -p 5433:5432 \
  -e POSTGRES_USER=neurondb \
  -e POSTGRES_PASSWORD=neurondb \
  -e POSTGRES_DB=neurondb \
  -e POSTGRES_HOST_AUTH_METHOD=md5 \
  -v /tmp/neurondb-init:/docker-entrypoint-initdb.d \
  neurondb:cpu-package-pg17

# Wait for container to be ready
sleep 20

# Configure shared_preload_libraries
sudo docker exec neurondb-cpu bash -c "CONF_FILE=\$(find /var/lib/postgresql -name postgresql.conf -type f 2>/dev/null | head -1) && if [ -n \"\$CONF_FILE\" ]; then echo \"shared_preload_libraries = 'neurondb'\" >> \"\$CONF_FILE\" && echo \"neurondb.compute_mode = 0\" >> \"\$CONF_FILE\" && echo \"neurondb.gpu_backend_type = 0\" >> \"\$CONF_FILE\"; fi"

# Restart container to apply configuration
sudo docker restart neurondb-cpu
sleep 10

# Create extension
PGPASSWORD=neurondb psql -h localhost -p 5433 -U neurondb -d neurondb \
  -c "CREATE EXTENSION IF NOT EXISTS neurondb;"

# Verify installation
PGPASSWORD=neurondb psql -h localhost -p 5433 -U neurondb -d neurondb \
  -c "SELECT neurondb.version();"
```

**Connection String:**
```
postgresql://neurondb:neurondb@localhost:5433/neurondb
```

### CUDA GPU Image

**Requirements**: NVIDIA driver ≥ 535, `nvidia-container-toolkit` installed.

**Build and Run:**

```bash
# From repository root
cd /path/to/neurondb

# Build CUDA image (PostgreSQL 17)
sudo docker build -f NeuronDB/docker/Dockerfile.package.cuda \
  --build-arg PG_MAJOR=17 \
  --build-arg PACKAGE_VERSION=3.0.0-devel \
  --build-arg CUDA_VERSION=12.4.1 \
  -t neurondb:cuda-package-pg17 .

# Start container (with GPU support if nvidia-container-toolkit is configured)
sudo docker run -d --name neurondb-cuda \
  -p 5434:5432 \
  -e POSTGRES_USER=neurondb \
  -e POSTGRES_PASSWORD=neurondb \
  -e POSTGRES_DB=neurondb \
  -e POSTGRES_HOST_AUTH_METHOD=md5 \
  -v /tmp/neurondb-init:/docker-entrypoint-initdb.d \
  neurondb:cuda-package-pg17

# Wait for container to be ready
sleep 20

# Configure shared_preload_libraries with GPU settings
sudo docker exec neurondb-cuda bash -c "CONF_FILE=\$(find /var/lib/postgresql -name postgresql.conf -type f 2>/dev/null | head -1) && if [ -n \"\$CONF_FILE\" ]; then echo \"shared_preload_libraries = 'neurondb'\" >> \"\$CONF_FILE\" && echo \"neurondb.compute_mode = 1\" >> \"\$CONF_FILE\" && echo \"neurondb.gpu_backend_type = 1\" >> \"\$CONF_FILE\"; fi"

# Restart container to apply configuration
sudo docker restart neurondb-cuda
sleep 10

# Create extension
PGPASSWORD=neurondb psql -h localhost -p 5434 -U neurondb -d neurondb \
  -c "CREATE EXTENSION IF NOT EXISTS neurondb;"

# Verify installation
PGPASSWORD=neurondb psql -h localhost -p 5434 -U neurondb -d neurondb \
  -c "SELECT neurondb.version();"

# Verify GPU access (if nvidia-container-toolkit is configured)
sudo docker exec neurondb-cuda nvidia-smi
```

**Connection String:**
```
postgresql://neurondb:neurondb@localhost:5434/neurondb
```

### ROCm GPU Image

**Requirements**: AMD GPU with ROCm drivers, Docker with device access.

**Build and Run:**

```bash
# From repository root
cd /path/to/neurondb

# Build ROCm image (PostgreSQL 17)
sudo docker build -f NeuronDB/docker/Dockerfile.package.rocm \
  --build-arg PG_MAJOR=17 \
  --build-arg PACKAGE_VERSION=3.0.0-devel \
  --build-arg ROCM_VERSION=5.7 \
  -t neurondb:rocm-package-pg17 .

# Start container
sudo docker run -d --name neurondb-rocm \
  -p 5435:5432 \
  --device=/dev/kfd \
  --device=/dev/dri \
  -e POSTGRES_USER=neurondb \
  -e POSTGRES_PASSWORD=neurondb \
  -e POSTGRES_DB=neurondb \
  -e POSTGRES_HOST_AUTH_METHOD=md5 \
  -v /tmp/neurondb-init:/docker-entrypoint-initdb.d \
  neurondb:rocm-package-pg17

# Wait for container to be ready
sleep 20

# Configure shared_preload_libraries with GPU settings
sudo docker exec neurondb-rocm bash -c "CONF_FILE=\$(find /var/lib/postgresql -name postgresql.conf -type f 2>/dev/null | head -1) && if [ -n \"\$CONF_FILE\" ]; then echo \"shared_preload_libraries = 'neurondb'\" >> \"\$CONF_FILE\" && echo \"neurondb.compute_mode = 1\" >> \"\$CONF_FILE\" && echo \"neurondb.gpu_backend_type = 2\" >> \"\$CONF_FILE\"; fi"

# Restart container to apply configuration
sudo docker restart neurondb-rocm
sleep 10

# Create extension
PGPASSWORD=neurondb psql -h localhost -p 5435 -U neurondb -d neurondb \
  -c "CREATE EXTENSION IF NOT EXISTS neurondb;"

# Verify installation
PGPASSWORD=neurondb psql -h localhost -p 5435 -U neurondb -d neurondb \
  -c "SELECT neurondb.version();"
```

**Connection String:**
```
postgresql://neurondb:neurondb@localhost:5435/neurondb
```

### Metal GPU Image (macOS/Apple Silicon)

**Requirements**: macOS with Apple Silicon, Docker Desktop.

**Note**: Metal support uses source-based build (package-based not available yet).

**Build and Run:**

```bash
# From repository root
cd /path/to/neurondb

# Build Metal image (PostgreSQL 17, arm64 only)
docker buildx build --platform linux/arm64 \
  -f NeuronDB/docker/Dockerfile.gpu.metal \
  --build-arg PG_MAJOR=17 \
  --build-arg ONNX_VERSION=1.17.0 \
  -t neurondb:metal-pg17 \
  --load .

# Start container
docker run -d --name neurondb-metal \
  -p 5436:5432 \
  -e POSTGRES_USER=neurondb \
  -e POSTGRES_PASSWORD=neurondb \
  -e POSTGRES_DB=neurondb \
  -e POSTGRES_HOST_AUTH_METHOD=md5 \
  -v /tmp/neurondb-init:/docker-entrypoint-initdb.d \
  neurondb:metal-pg17

# Wait for container to be ready
sleep 20

# Configure shared_preload_libraries with GPU settings
docker exec neurondb-metal bash -c "CONF_FILE=\$(find /var/lib/postgresql -name postgresql.conf -type f 2>/dev/null | head -1) && if [ -n \"\$CONF_FILE\" ]; then echo \"shared_preload_libraries = 'neurondb'\" >> \"\$CONF_FILE\" && echo \"neurondb.compute_mode = 1\" >> \"\$CONF_FILE\" && echo \"neurondb.gpu_backend_type = 3\" >> \"\$CONF_FILE\"; fi"

# Restart container to apply configuration
docker restart neurondb-metal
sleep 10

# Create extension
PGPASSWORD=neurondb psql -h localhost -p 5436 -U neurondb -d neurondb \
  -c "CREATE EXTENSION IF NOT EXISTS neurondb;"

# Verify installation
PGPASSWORD=neurondb psql -h localhost -p 5436 -U neurondb -d neurondb \
  -c "SELECT neurondb.version();"
```

**Connection String:**
```
postgresql://neurondb:neurondb@localhost:5436/neurondb
```

## Build Arguments

### Common Arguments (All Dockerfiles)

| Argument | Default | Description |
|----------|---------|-------------|
| `PG_MAJOR` | `17` | PostgreSQL major version: `16`, `17`, or `18` |
| `ONNX_VERSION` | `1.17.0` | ONNX Runtime version to embed |
| `VERSION` | `latest` | Application version (used in labels) |
| `BUILD_DATE` | - | Build date (ISO 8601 format) |
| `VCS_REF` | - | Git commit hash or version control reference |

### CUDA-Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `CUDA_VERSION` | `12.4.1` | CUDA toolkit version |
| `ENABLE_RAPIDS` | `0` | Enable RAPIDS/cuML stack: `0` or `1` |

### ROCm-Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `ROCM_VERSION` | `5.7` | ROCm version |

### Production Build Example

For production builds with versioning:

```bash
VERSION=3.0.0-devel
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD)

docker build -f docker/Dockerfile \
  --build-arg PG_MAJOR=17 \
  --build-arg ONNX_VERSION=1.17.0 \
  --build-arg VERSION=${VERSION} \
  --build-arg BUILD_DATE=${BUILD_DATE} \
  --build-arg VCS_REF=${VCS_REF} \
  -t neurondb:${VERSION}-cpu \
  -t neurondb:latest-cpu ..
```

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
neurondb.compute_mode = off
neurondb.automl.use_gpu = off
```

Modify at runtime:

```sql
ALTER SYSTEM SET neurondb.compute_mode = on;
SELECT pg_reload_conf();
```

Or edit directly:

```bash
docker exec -it neurondb-cuda vi /var/lib/postgresql/data/postgresql.conf
docker exec -it neurondb-cuda pg_ctl reload
```

### Environment Variables

Standard PostgreSQL environment variables are supported:

- `POSTGRES_USER` (default: `neurondb`)
- `POSTGRES_PASSWORD` (default: `neurondb`)
- `POSTGRES_DB` (default: `neurondb`)
- `POSTGRES_HOST_AUTH_METHOD` (default: `md5`, use `scram-sha-256` for production)
- `POSTGRES_INITDB_ARGS`

GPU-specific variables:

- `NVIDIA_VISIBLE_DEVICES` (CUDA): GPU device selection
- `NVIDIA_DRIVER_CAPABILITIES` (CUDA): Driver capabilities
- `NEURONDB_GPU_ENABLED`: Enable GPU features

### Environment File

Create `.env` file from template:

```bash
cp .env.example .env
# Edit .env with your configuration
```

The `.env.example` file includes all configurable environment variables with documentation.

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
SELECT neurondb.compute_mode();
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

## Production Deployment

### Using Production Compose File

The `docker-compose.prod.yml` file extends the base configuration with production-focused settings:

- Enhanced resource limits
- Structured JSON logging
- Production restart policies
- Extended health checks
- Security enhancements

Start with production configuration:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Production Environment Variables

Create `.env.prod` for production:

```env
POSTGRES_USER=neurondb
POSTGRES_PASSWORD=<secure-password>
POSTGRES_DB=neurondb
POSTGRES_HOST_AUTH_METHOD=scram-sha-256
PG_MAJOR=17
ONNX_VERSION=1.17.0
```

Load production environment:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.prod up -d
```

## SSL/TLS Configuration

### Secure Database Connections

NeuronDB (PostgreSQL) supports SSL/TLS for secure connections. Configure SSL in PostgreSQL:

### Generate SSL Certificates

```bash
# Create certificates directory
mkdir -p ssl

# Generate CA key and certificate
openssl req -new -x509 -days 3650 -nodes \
  -out ssl/ca.crt -keyout ssl/ca.key \
  -subj "/CN=NeuronDB-CA"

# Generate server key
openssl genrsa -out ssl/server.key 2048

# Generate server certificate request
openssl req -new -key ssl/server.key -out ssl/server.csr \
  -subj "/CN=neurondb-cpu"

# Sign server certificate with CA
openssl x509 -req -in ssl/server.csr -CA ssl/ca.crt \
  -CAkey ssl/ca.key -CAcreateserial \
  -out ssl/server.crt -days 3650
```

### Mount SSL Certificates

In `docker-compose.yml`:

```yaml
services:
  neurondb:
    volumes:
      - neurondb-data:/var/lib/postgresql/data
      - ./ssl:/var/lib/postgresql/ssl:ro
    environment:
      POSTGRES_SSL_CERT: /var/lib/postgresql/ssl/server.crt
      POSTGRES_SSL_KEY: /var/lib/postgresql/ssl/server.key
      POSTGRES_SSL_CA: /var/lib/postgresql/ssl/ca.crt
```

### Configure PostgreSQL for SSL

Add to `postgresql.conf` (via init script or runtime):

```conf
ssl = on
ssl_cert_file = '/var/lib/postgresql/ssl/server.crt'
ssl_key_file = '/var/lib/postgresql/ssl/server.key'
ssl_ca_file = '/var/lib/postgresql/ssl/ca.crt'
```

### Client SSL Configuration

Connect with SSL:

```bash
psql "postgresql://neurondb:password@localhost:5433/neurondb?sslmode=require"
```

## Security

### Container Security Best Practices

The NeuronDB Docker images implement multiple security best practices:

#### Base Image Security
- Uses official PostgreSQL base images
- Regular security updates via base image updates
- Minimal attack surface

#### Authentication Methods

For production, use `scram-sha-256` authentication:

```env
POSTGRES_HOST_AUTH_METHOD=scram-sha-256
```

This provides better security than `md5` authentication.

#### Credential Management

#### Docker Secrets (Recommended for Production)

```yaml
version: "3.9"

secrets:
  postgres_password:
    file: ./secrets/postgres_password.txt
  postgres_user:
    file: ./secrets/postgres_user.txt

services:
  neurondb:
    secrets:
      - postgres_password
      - postgres_user
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
      POSTGRES_USER_FILE: /run/secrets/postgres_user
```

Create secrets directory:

```bash
mkdir -p secrets
chmod 700 secrets
echo "your-secure-password" > secrets/postgres_password.txt
chmod 600 secrets/postgres_password.txt
```

#### Environment Variables

For development, use `.env` file with restricted permissions:

```bash
chmod 600 .env
```

Never commit `.env` files to version control. Use `.env.example` as a template.

#### External Secrets Management

Integrate with external secrets management systems:

**HashiCorp Vault:**
```bash
export POSTGRES_PASSWORD=$(vault kv get -field=password secret/neurondb)
```

**AWS Secrets Manager:**
```bash
export POSTGRES_PASSWORD=$(aws secretsmanager get-secret-value \
  --secret-id neurondb/password --query SecretString --output text)
```

### Network Security

- **Isolated Networks**: Use Docker networks to isolate services
- **Firewall Rules**: Restrict external access to PostgreSQL port
- **TLS/SSL**: Enable TLS/SSL for all connections in production
- **Network Policies**: Implement network policies in orchestrated environments

### Security Checklist

Before deploying to production:

- [ ] Use Docker secrets or external secrets management
- [ ] Change default passwords
- [ ] Use `scram-sha-256` authentication method
- [ ] Enable SSL/TLS for database connections
- [ ] Restrict network access
- [ ] Enable structured logging for audit trails
- [ ] Set appropriate resource limits
- [ ] Configure health checks
- [ ] Use production restart policies
- [ ] Regularly update base images
- [ ] Scan images for vulnerabilities

## Logging Integration

### Structured JSON Logging

For production, configure PostgreSQL logging:

```yaml
services:
  neurondb:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
        compress: "true"
```

### Docker Logging Drivers

#### Syslog Driver

Send logs to syslog:

```yaml
services:
  neurondb:
    logging:
      driver: "syslog"
      options:
        syslog-address: "tcp://localhost:514"
        syslog-facility: "daemon"
        tag: "neurondb"
```

#### Fluentd Driver

Send logs to Fluentd:

```yaml
services:
  neurondb:
    logging:
      driver: "fluentd"
      options:
        fluentd-address: "localhost:24224"
        tag: "neurondb.postgresql"
```

#### Loki Driver

Send logs to Grafana Loki:

```yaml
services:
  neurondb:
    logging:
      driver: "loki"
      options:
        loki-url: "http://localhost:3100/loki/api/v1/push"
        loki-batch-size: "400"
```

#### CloudWatch Logs (AWS)

```yaml
services:
  neurondb:
    logging:
      driver: "awslogs"
      options:
        awslogs-group: "/docker/neurondb"
        awslogs-region: "us-east-1"
        awslogs-stream-prefix: "postgresql"
```

### PostgreSQL Log Configuration

Configure PostgreSQL logging in `postgresql.conf`:

```conf
logging_collector = on
log_directory = 'log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_rotation_age = 1d
log_rotation_size = 100MB
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
log_timezone = 'UTC'
```

## Monitoring

### Container Metrics

#### Docker Stats

Monitor real-time container metrics:

```bash
docker stats neurondb-cpu
```

#### cAdvisor

Use cAdvisor for detailed container metrics:

```yaml
services:
  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
```

#### Prometheus

Export PostgreSQL metrics to Prometheus using `postgres_exporter`:

```yaml
services:
  postgres-exporter:
    image: prometheuscommunity/postgres-exporter:latest
    environment:
      DATA_SOURCE_NAME: "postgresql://neurondb:password@neurondb-cpu:5432/neurondb?sslmode=disable"
    ports:
      - "9187:9187"
```

### Database Metrics

#### PostgreSQL Statistics

Query PostgreSQL statistics:

```sql
-- Connection statistics
SELECT * FROM pg_stat_activity;

-- Database statistics
SELECT * FROM pg_stat_database WHERE datname = 'neurondb';

-- Table statistics
SELECT * FROM pg_stat_user_tables;
```

#### NeuronDB Metrics

Query NeuronDB-specific metrics:

```sql
-- GPU device info (if GPU enabled)
SELECT neurondb.gpu_device_info();

-- Extension version
SELECT neurondb.version();
```

### Health Monitoring

#### Health Check Status

Check container health:

```bash
docker inspect --format='{{.State.Health.Status}}' neurondb-cpu
```

#### Database Health

Test database connectivity:

```bash
docker exec neurondb-cpu pg_isready -U neurondb
```

### Alerting

#### Prometheus Alertmanager

Set up alerts for:

- Container down
- High CPU usage (>80%)
- High memory usage (>90%)
- Health check failures
- Database connection errors
- Slow queries

Example alert rule:

```yaml
groups:
  - name: neurondb
    rules:
      - alert: NeuronDBDown
        expr: up{job="neurondb"} == 0
        for: 1m
        annotations:
          summary: "NeuronDB container is down"
      - alert: HighConnectionCount
        expr: pg_stat_database_numbackends{datname="neurondb"} > 80
        for: 5m
        annotations:
          summary: "High connection count detected"
```

### Monitoring Best Practices

1. **Set Up Alerts**: Configure alerts for critical metrics
2. **Log Aggregation**: Centralize logs for analysis
3. **Metrics Retention**: Configure appropriate retention periods
4. **Dashboard Creation**: Create dashboards for key metrics
5. **Regular Reviews**: Review metrics and logs regularly
6. **Capacity Planning**: Monitor trends for capacity planning
7. **Performance Baselines**: Establish performance baselines

## Support

For issues, questions, or contributions:

- **GitHub Issues**: [https://github.com/neurondb/NeurondB/issues](https://github.com/neurondb/NeurondB/issues)
- **Documentation**: [https://neurondb.ai/docs](https://neurondb.ai/docs)
- **Email**: support@neurondb.ai
