# NeuronMCP Docker Setup

[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](../../LICENSE)
[![MCP](https://img.shields.io/badge/MCP-Protocol-blue.svg)](https://modelcontextprotocol.io/)

Docker container for NeuronMCP service. Connects to external NeuronDB PostgreSQL instance. Implements Model Context Protocol over stdio for MCP-compatible clients.

## Overview

NeuronMCP Docker container runs the MCP server service. Connects to an external NeuronDB PostgreSQL database. Communicates via stdio using JSON-RPC 2.0 protocol. Compatible with MCP clients like Claude Desktop.

## Prerequisites

| Requirement | Description |
|-------------|-------------|
| Docker | Docker 20.10 or later |
| Docker Compose | Docker Compose 2.0 or later |
| NeuronDB | Running NeuronDB PostgreSQL instance |
| Network Access | Connectivity to NeuronDB database |
| MCP Client | MCP-compatible client (optional for testing) |

## Quick Start

**Prerequisites**: NeuronDB container must be running first. See [NeuronDB Docker README](../../NeuronDB/docker/README.md).

### Step 1: Navigate to Directory

```bash
cd /path/to/neurondb/NeuronMCP/docker
```

### Step 2: Build Image

```bash
# Build from repository root
cd /path/to/neurondb
sudo docker build -f NeuronMCP/docker/Dockerfile.package \
  --build-arg PACKAGE_VERSION=3.0.0-devel \
  -t neuronmcp:package .
```

### Step 3: Start Container

```bash
# Start NeuronMCP container
sudo docker run -d --name neuronmcp \
  -e NEURONDB_HOST=localhost \
  -e NEURONDB_PORT=5433 \
  -e NEURONDB_DATABASE=neurondb \
  -e NEURONDB_USER=neurondb \
  -e NEURONDB_PASSWORD=neurondb \
  --network host \
  neuronmcp:package
```

**Note**: Using `--network host` allows connection to `localhost:5433`. For Docker network, use container name:

```bash
# If NeuronDB is in Docker network
sudo docker run -d --name neuronmcp \
  -e NEURONDB_HOST=neurondb-cpu \
  -e NEURONDB_PORT=5432 \
  -e NEURONDB_DATABASE=neurondb \
  -e NEURONDB_USER=neurondb \
  -e NEURONDB_PASSWORD=neurondb \
  --network neurondb-network \
  neuronmcp:package
```

### Step 4: Verify Installation

Wait for container to be ready:

```bash
sleep 10
```

Check container status:

```bash
sudo docker ps | grep neuronmcp
```

View logs:

```bash
sudo docker logs neuronmcp
```

Test stdio communication (MCP protocol):

```bash
sudo docker exec -i neuronmcp ./neurondb-mcp
```

For MCP client configuration, see [Claude Desktop Configuration](../README.md#claude-desktop-configuration).

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEURONDB_HOST` | `localhost` | NeuronDB database hostname |
| `NEURONDB_PORT` | `5433` | NeuronDB database port |
| `NEURONDB_DATABASE` | `neurondb` | Database name |
| `NEURONDB_USER` | `neurondb` | Database username |
| `NEURONDB_PASSWORD` | `neurondb` | Database password |
| `NEURONDB_CONNECTION_STRING` | - | Full connection string (overrides above) |
| `NEURONDB_MCP_CONFIG` | `mcp-config.json` | Path to config file |
| `NEURONDB_LOG_LEVEL` | `info` | Log level (debug, info, warn, error) |
| `NEURONDB_LOG_FORMAT` | `text` | Log format (json, text) |
| `NEURONDB_LOG_OUTPUT` | `stderr` | Log output (stdout, stderr, file) |
| `NEURONDB_ENABLE_GPU` | `false` | Enable GPU acceleration |

Environment variables override configuration file values.

### Configuration File

Copy example configuration:

```bash
cp mcp-config.json.example mcp-config.json
```

Edit `mcp-config.json`:

```json
{
  "database": {
    "host": "localhost",
    "port": 5432,
    "database": "neurondb",
    "user": "neurondb",
    "password": "neurondb"
  },
  "server": {
    "name": "neurondb-mcp-server",
    "version": "1.0.0"
  },
  "logging": {
    "level": "info",
    "format": "text"
  },
  "features": {
    "vector": { "enabled": true },
    "ml": { "enabled": true },
    "analytics": { "enabled": true }
  }
}
```

Mount in `docker-compose.yml`:

```yaml
volumes:
  - ./mcp-config.json:/app/mcp-config.json:ro
```

Set environment variable:

```env
NEURONDB_MCP_CONFIG=/app/mcp-config.json
```

## Connecting to NeuronDB

### From Host Machine

Set connection parameters for localhost access:

```env
NEURONDB_HOST=localhost
NEURONDB_PORT=5433
```

Port 5433 is the default Docker port mapping for NeuronDB.

### From Docker Network

Create shared network:

```bash
docker network create neurondb-network
```

Connect NeuronDB container:

```bash
docker network connect neurondb-network neurondb-cpu
```

Connect NeuronMCP container:

```bash
docker network connect neurondb-network neurondb-mcp
```

Update environment variables:

```env
NEURONDB_HOST=neurondb-cpu
NEURONDB_PORT=5432
```

Use container name as hostname. Port 5432 is the internal container port.

### Network Configuration Example

Update `docker-compose.yml`:

```yaml
networks:
  neurondb-network:
    external: true

services:
  neurondb-mcp:
    networks:
      - neurondb-network
```

## MCP Protocol

NeuronMCP uses Model Context Protocol over stdio:

- No HTTP endpoints
- Communication via stdin and stdout
- Messages follow JSON-RPC 2.0 format
- Clients initiate all requests

### Stdio Configuration

Container configured for stdio communication:

```yaml
services:
  neurondb-mcp:
    stdin_open: true
    tty: true
    entrypoint: ["./neurondb-mcp"]
```

Required for MCP protocol communication.

## Using with Claude Desktop

### Configuration File Location

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

**Linux:** `~/.config/Claude/claude_desktop_config.json`

### Docker Configuration

Create Claude Desktop config file:

```json
{
  "mcpServers": {
    "neurondb": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "--network", "neurondb-network",
        "-e", "NEURONDB_HOST=neurondb-cpu",
        "-e", "NEURONDB_PORT=5432",
        "-e", "NEURONDB_DATABASE=neurondb",
        "-e", "NEURONDB_USER=neurondb",
        "-e", "NEURONDB_PASSWORD=neurondb",
        "neurondb-mcp:latest"
      ]
    }
  }
}
```

### Local Binary Configuration

If using local binary instead of Docker:

```json
{
  "mcpServers": {
    "neurondb": {
      "command": "/path/to/neurondb-mcp",
      "env": {
        "NEURONDB_HOST": "localhost",
        "NEURONDB_PORT": "5432",
        "NEURONDB_DATABASE": "neurondb",
        "NEURONDB_USER": "neurondb",
        "NEURONDB_PASSWORD": "neurondb"
      }
    }
  }
}
```

### Restart Claude Desktop

After configuration changes, restart Claude Desktop to load new MCP server.

## Using with Other MCP Clients

### Interactive Mode

Run container interactively for testing:

```bash
docker run -i --rm \
  -e NEURONDB_HOST=neurondb-cpu \
  -e NEURONDB_PORT=5432 \
  -e NEURONDB_DATABASE=neurondb \
  -e NEURONDB_USER=neurondb \
  -e NEURONDB_PASSWORD=neurondb \
  --network neurondb-network \
  neurondb-mcp:latest
```

Send JSON-RPC messages via stdin, receive responses via stdout.

### Docker Compose

For development, use docker-compose with stdio:

```yaml
services:
  neurondb-mcp:
    stdin_open: true
    tty: true
```

### MCP Client Integration

Configure your MCP client to execute:

```bash
docker run -i --rm \
  --network neurondb-network \
  -e NEURONDB_HOST=neurondb-cpu \
  -e NEURONDB_PORT=5432 \
  neurondb-mcp:latest
```

## Building the Image

### Standard Build

Using Docker Compose:

```bash
docker compose build
```

### Custom Build

Using Docker directly:

```bash
docker build -f docker/Dockerfile -t neurondb-mcp:latest ..
```

### Build Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `GO_VERSION` | `1.23` | Go version for builder stage |
| `VERSION` | `latest` | Application version (used in labels) |
| `BUILD_DATE` | - | Build date (ISO 8601 format) |
| `VCS_REF` | - | Git commit hash or version control reference |

Example with all build arguments:

```bash
docker build -f docker/Dockerfile \
  --build-arg GO_VERSION=1.23 \
  --build-arg VERSION=3.0.0-devel \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  --build-arg VCS_REF=$(git rev-parse --short HEAD) \
  -t neurondb-mcp:1.0.0 ..
```

### Production Build

For production builds with versioning:

```bash
VERSION=3.0.0-devel
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD)

docker build -f docker/Dockerfile \
  --build-arg GO_VERSION=1.23 \
  --build-arg VERSION=${VERSION} \
  --build-arg BUILD_DATE=${BUILD_DATE} \
  --build-arg VCS_REF=${VCS_REF} \
  -t neurondb-mcp:${VERSION} \
  -t neurondb-mcp:latest ..
```

## Production Deployment

### Using Production Compose File

The `docker-compose.prod.yml` file extends the base configuration with production-focused settings:

- Enhanced resource limits
- Structured JSON logging
- Security hardening (read-only filesystem, dropped capabilities)
- Production restart policies
- Health checks with extended timeouts

Start with production configuration:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Production Environment Variables

Create `.env.prod` for production:

```env
NEURONDB_HOST=neurondb-cpu
NEURONDB_PORT=5432
NEURONDB_DATABASE=neurondb
NEURONDB_USER=neurondb
NEURONDB_PASSWORD=<secure-password>
NEURONDB_LOG_LEVEL=info
NEURONDB_LOG_FORMAT=json
NEURONDB_SSL_MODE=require
```

Load production environment:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.prod up -d
```

### Docker Entrypoint Script

The container includes an optional entrypoint script (`docker-entrypoint.sh`) that:

- Validates binary existence and permissions
- Checks environment variables
- Validates configuration file (if provided)
- Logs startup information
- Executes the NeuronMCP binary

The entrypoint can be overridden in `docker-compose.yml`:

```yaml
services:
  neurondb-mcp:
    entrypoint: ["./neurondb-mcp"]  # Skip entrypoint script
```

## SSL/TLS Configuration

### Secure Database Connections

NeuronMCP supports SSL/TLS for secure database connections. Configure SSL in your environment:

```env
# SSL mode: disable, allow, prefer, require, verify-ca, verify-full
NEURONDB_SSL_MODE=require

# SSL certificate paths (inside container)
NEURONDB_SSL_CERT=/app/ssl/client-cert.pem
NEURONDB_SSL_KEY=/app/ssl/client-key.pem
NEURONDB_SSL_ROOT_CERT=/app/ssl/ca-cert.pem
```

### Using Connection String with SSL

```env
NEURONDB_CONNECTION_STRING=postgresql://user:password@host:port/database?sslmode=require&sslcert=/app/ssl/client-cert.pem&sslkey=/app/ssl/client-key.pem&sslrootcert=/app/ssl/ca-cert.pem
```

### Mounting SSL Certificates

In `docker-compose.yml`:

```yaml
services:
  neurondb-mcp:
    volumes:
      - ./ssl:/app/ssl:ro
    environment:
      NEURONDB_SSL_MODE: require
      NEURONDB_SSL_CERT: /app/ssl/client-cert.pem
      NEURONDB_SSL_KEY: /app/ssl/client-key.pem
      NEURONDB_SSL_ROOT_CERT: /app/ssl/ca-cert.pem
```

### SSL Certificate Permissions

Ensure certificates have correct permissions:

```bash
chmod 600 /path/to/ssl/client-key.pem
chmod 644 /path/to/ssl/client-cert.pem
chmod 644 /path/to/ssl/ca-cert.pem
```

## Container Management

### Start Container

```bash
docker compose up -d
```

Note: Container runs in background but requires stdio for MCP communication. Use with MCP clients that handle stdio.

### Stop Container

```bash
docker compose stop
```

### Restart Container

```bash
docker compose restart
```

### View Logs

Follow logs:

```bash
docker compose logs -f neurondb-mcp
```

View last 100 lines:

```bash
docker compose logs --tail=100 neurondb-mcp
```

### Execute Commands

Run shell in container:

```bash
docker compose exec neurondb-mcp /bin/sh
```

## Troubleshooting

### Container Will Not Start

Check container logs:

```bash
docker compose logs neurondb-mcp
```

Common issues:
- Missing environment variables
- Invalid database connection parameters
- Network connectivity issues
- Configuration file errors

### Database Connection Failed

Verify NeuronDB is running:

```bash
docker compose ps neurondb
```

Test connection manually:

```bash
psql -h localhost -p 5433 -U neurondb -d neurondb -c "SELECT 1;"
```

Check environment variables:

```bash
docker compose config | grep -A 10 NEURONDB_
```

Verify network connectivity:

```bash
docker exec neurondb-mcp ping neurondb-cpu
```

### Extension Not Found

Verify extension is installed:

```sql
SELECT * FROM pg_extension WHERE extname = 'neurondb';
```

Install if missing:

```sql
CREATE EXTENSION neurondb;
```

### Stdio Not Working

Ensure `stdin_open: true` and `tty: true` in `docker-compose.yml`:

```yaml
services:
  neurondb-mcp:
    stdin_open: true
    tty: true
```

For interactive use, run with:

```bash
docker run -i -t neurondb-mcp:latest
```

Do not redirect stdin or stdout:

```bash
./neurondb-mcp > output.log  # Incorrect - breaks MCP protocol
```

### MCP Client Connection Issues

Check container is running:

```bash
docker compose ps neurondb-mcp
```

Test stdio manually:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | \
  docker compose exec -T neurondb-mcp ./neurondb-mcp
```

Verify network connectivity:

```bash
docker network inspect neurondb-network
```

Check MCP client configuration file path and format.

### Configuration Issues

Verify config file path:

```bash
docker compose exec neurondb-mcp ls -la /app/mcp-config.json
```

Check environment variable names (must start with `NEURONDB_`):

```bash
docker compose config | grep -E "^NEURONDB_"
```

Ensure variables are set before container starts:

```bash
docker compose config
```

Check for typos in variable names.

## Security

### Container Security Best Practices

The NeuronMCP Docker image implements multiple security best practices:

#### Non-Root User
- Container runs as non-root user `neuronmcp` (UID 1000)
- Prevents privilege escalation attacks
- Minimal filesystem permissions

#### Minimal Base Image
- Uses Debian slim base image
- Reduced attack surface
- Only essential runtime dependencies

#### No Network Exposure
- No HTTP endpoints or exposed ports
- Communication via stdio only
- No external attack surface

#### Read-Only Filesystem (Production)

In `docker-compose.prod.yml`, enable read-only root filesystem:

```yaml
services:
  neurondb-mcp:
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=100m
      - /var/tmp:noexec,nosuid,size=100m
```

#### Dropped Capabilities

Production configuration drops all capabilities:

```yaml
services:
  neurondb-mcp:
    cap_drop:
      - ALL
    security_opt:
      - no-new-privileges:true
```

#### Image Scanning

Regularly scan images for vulnerabilities:

```bash
# Using Trivy
trivy image neurondb-mcp:latest

# Using Docker Scout
docker scout cves neurondb-mcp:latest

# Using Snyk
snyk test --docker neurondb-mcp:latest
```

### Credential Management

#### Docker Secrets (Recommended for Production)

```yaml
version: "3.9"

secrets:
  db_password:
    file: ./secrets/db_password.txt
  db_user:
    file: ./secrets/db_user.txt

services:
  neurondb-mcp:
    secrets:
      - db_password
      - db_user
    environment:
      NEURONDB_PASSWORD_FILE: /run/secrets/db_password
      NEURONDB_USER_FILE: /run/secrets/db_user
```

Create secrets directory:

```bash
mkdir -p secrets
chmod 700 secrets
echo "your-secure-password" > secrets/db_password.txt
chmod 600 secrets/db_password.txt
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
# Retrieve secret and set as environment variable
export NEURONDB_PASSWORD=$(vault kv get -field=password secret/neurondb)
```

**AWS Secrets Manager:**
```bash
export NEURONDB_PASSWORD=$(aws secretsmanager get-secret-value \
  --secret-id neurondb/password --query SecretString --output text)
```

**Kubernetes Secrets:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: neurondb-credentials
type: Opaque
stringData:
  password: your-secure-password
```

### Network Security

- **Isolated Networks**: Use Docker networks to isolate services
- **No External Exposure**: Container has no exposed ports
- **TLS/SSL**: Enable TLS/SSL for all database connections in production
- **Network Policies**: Implement network policies in orchestrated environments (Kubernetes, Docker Swarm)

### Security Checklist

Before deploying to production:

- [ ] Use Docker secrets or external secrets management
- [ ] Enable SSL/TLS for database connections
- [ ] Use read-only filesystem where possible
- [ ] Drop unnecessary capabilities
- [ ] Set `no-new-privileges: true`
- [ ] Regularly scan images for vulnerabilities
- [ ] Use non-root user (already configured)
- [ ] Restrict network access
- [ ] Enable structured logging for audit trails
- [ ] Set appropriate resource limits
- [ ] Configure health checks
- [ ] Use production restart policies

## Integration with NeuronDB

### Requirements

- PostgreSQL 16 or later
- NeuronDB extension installed and enabled
- Database user with appropriate permissions
- Network connectivity to database

### Setup Instructions

See [NeuronDB Docker README](../../NeuronDB/docker/README.md) for NeuronDB setup.

## MCP Tools and Resources

NeuronMCP exposes:

### Tools

- Vector operations: search, similarity, embedding generation
- ML operations: training, prediction, evaluation
- Analytics: data analysis, clustering, dimensionality reduction
- RAG operations: document processing, context retrieval

### Resources

- Schema information
- Model configurations
- Index configurations
- Worker status
- Database statistics

See [NeuronMCP README](../README.md) for complete documentation.

## Production Deployment

### Recommendations

| Practice | Implementation |
|----------|----------------|
| Use Secrets | Store credentials in Docker secrets or external system |
| Configure Logging | Set up log aggregation and monitoring |
| Resource Limits | Set CPU and memory limits |
| Restart Policy | Configure restart policy for reliability |
| MCP Client Configuration | Properly configure MCP clients for stdio communication |

### Resource Limits

Example `docker-compose.yml`:

```yaml
services:
  neurondb-mcp:
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 512M
        reservations:
          cpus: '0.5'
          memory: 256M
```

## Logging Integration

### Structured JSON Logging

For production, use JSON logging format for easier parsing:

```yaml
services:
  neurondb-mcp:
    environment:
      NEURONDB_LOG_FORMAT: json
      NEURONDB_LOG_OUTPUT: stderr
```

### Docker Logging Drivers

#### JSON File Driver (Default)

Configure log rotation:

```yaml
services:
  neurondb-mcp:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
        compress: "true"
```

#### Syslog Driver

Send logs to syslog:

```yaml
services:
  neurondb-mcp:
    logging:
      driver: "syslog"
      options:
        syslog-address: "tcp://localhost:514"
        syslog-facility: "daemon"
        tag: "neurondb-mcp"
```

#### Fluentd Driver

Send logs to Fluentd:

```yaml
services:
  neurondb-mcp:
    logging:
      driver: "fluentd"
      options:
        fluentd-address: "localhost:24224"
        tag: "neurondb.mcp"
```

#### Loki Driver

Send logs to Grafana Loki:

```yaml
services:
  neurondb-mcp:
    logging:
      driver: "loki"
      options:
        loki-url: "http://localhost:3100/loki/api/v1/push"
        loki-batch-size: "400"
```

#### GELF Driver (Graylog)

Send logs to Graylog:

```yaml
services:
  neurondb-mcp:
    logging:
      driver: "gelf"
      options:
        gelf-address: "udp://localhost:12201"
        tag: "neurondb-mcp"
```

### Log Aggregation Examples

#### ELK Stack (Elasticsearch, Logstash, Kibana)

```yaml
services:
  neurondb-mcp:
    logging:
      driver: "gelf"
      options:
        gelf-address: "udp://logstash:12201"
        tag: "neurondb-mcp"
```

#### Splunk

```yaml
services:
  neurondb-mcp:
    logging:
      driver: "splunk"
      options:
        splunk-token: "${SPLUNK_TOKEN}"
        splunk-url: "https://splunk.example.com:8088"
        splunk-source: "neurondb-mcp"
        splunk-sourcetype: "json"
```

#### CloudWatch Logs (AWS)

```yaml
services:
  neurondb-mcp:
    logging:
      driver: "awslogs"
      options:
        awslogs-group: "/docker/neurondb-mcp"
        awslogs-region: "us-east-1"
        awslogs-stream-prefix: "mcp"
```

#### Google Cloud Logging

```yaml
services:
  neurondb-mcp:
    logging:
      driver: "gcplogs"
      options:
        gcp-project: "your-project-id"
        gcp-log-cmd: "true"
```

### Log Parsing

With JSON logging, parse logs easily:

```bash
# Extract error logs
docker compose logs neurondb-mcp | jq 'select(.level == "error")'

# Count log entries by level
docker compose logs neurondb-mcp | jq -r '.level' | sort | uniq -c

# Filter by timestamp
docker compose logs neurondb-mcp | jq 'select(.time > "2024-01-01T00:00:00Z")'
```

## Monitoring

### Container Metrics

#### Docker Stats

Monitor real-time container metrics:

```bash
docker stats neurondb-mcp
```

Output includes:
- CPU usage percentage
- Memory usage and limits
- Network I/O
- Block I/O

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

Access metrics at `http://localhost:8080/containers/neurondb-mcp`

#### Prometheus

Export Docker metrics to Prometheus:

```yaml
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
```

Configure `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'docker'
    static_configs:
      - targets: ['host.docker.internal:9323']
```

### Health Monitoring

#### Health Check Status

Check container health:

```bash
docker inspect --format='{{.State.Health.Status}}' neurondb-mcp
```

#### Health Check Logs

View health check history:

```bash
docker inspect --format='{{json .State.Health}}' neurondb-mcp | jq
```

### Application Metrics

#### Custom Metrics Endpoint

If you need to expose application metrics, consider adding a sidecar container or using MCP resources:

```bash
# Query MCP resources for metrics
docker compose exec neurondb-mcp ./neurondb-mcp <<EOF
{"jsonrpc":"2.0","id":1,"method":"resources/read","params":{"uri":"neurondb://stats"}}
EOF
```

### Alerting

#### Prometheus Alertmanager

Set up alerts for:

- Container down
- High CPU usage (>80%)
- High memory usage (>90%)
- Health check failures
- Database connection errors

Example alert rule:

```yaml
groups:
  - name: neurondb_mcp
    rules:
      - alert: NeuronMCPDown
        expr: up{job="neurondb-mcp"} == 0
        for: 1m
        annotations:
          summary: "NeuronMCP container is down"
```

#### Grafana Dashboards

Create dashboards for:

- Container resource usage
- Log volume and patterns
- Error rates
- Response times (if applicable)

### Monitoring Best Practices

1. **Set Up Alerts**: Configure alerts for critical metrics
2. **Log Aggregation**: Centralize logs for analysis
3. **Metrics Retention**: Configure appropriate retention periods
4. **Dashboard Creation**: Create dashboards for key metrics
5. **Regular Reviews**: Review metrics and logs regularly
6. **Capacity Planning**: Monitor trends for capacity planning

## Support

- **Documentation**: [NeuronMCP README](../README.md)
- **GitHub Issues**: [Report Issues](https://github.com/neurondb/NeurondB/issues)
- **Email**: support@neurondb.ai

## License

See [LICENSE](../../LICENSE) file for license information.
