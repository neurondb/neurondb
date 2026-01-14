# NeuronAgent Docker Setup

[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](../../LICENSE)

Docker container for NeuronAgent service. Connects to external NeuronDB PostgreSQL instance. Provides REST API and WebSocket endpoints for agent runtime operations.

## Overview

NeuronAgent Docker container runs the agent server service. Connects to an external NeuronDB PostgreSQL database. Automatically runs database migrations on startup. Exposes HTTP API on port 8080.

## Prerequisites

| Requirement | Description |
|-------------|-------------|
| Docker | Docker 20.10 or later |
| Docker Compose | Docker Compose 2.0 or later |
| NeuronDB | Running NeuronDB PostgreSQL instance |
| Network Access | Connectivity to NeuronDB database |

## Quick Start

**Prerequisites**: NeuronDB container must be running first. See [NeuronDB Docker README](../../NeuronDB/docker/README.md).

### Step 1: Navigate to Directory

```bash
cd /path/to/neurondb/NeuronAgent/docker
```

### Step 2: Build Image

```bash
# Build from repository root
cd /path/to/neurondb
sudo docker build -f NeuronAgent/docker/Dockerfile.package \
  --build-arg PACKAGE_VERSION=1.0.0.beta \
  -t neuronagent:package .
```

### Step 3: Start Container

```bash
# Start NeuronAgent container
sudo docker run -d --name neuronagent \
  -p 8080:8080 \
  -e DB_HOST=localhost \
  -e DB_PORT=5433 \
  -e DB_NAME=neurondb \
  -e DB_USER=neurondb \
  -e DB_PASSWORD=neurondb \
  -e SERVER_HOST=0.0.0.0 \
  -e SERVER_PORT=8080 \
  --network host \
  neuronagent:package
```

**Note**: Using `--network host` allows connection to `localhost:5433`. For Docker network, use container name:

```bash
# If NeuronDB is in Docker network
sudo docker run -d --name neuronagent \
  -p 8080:8080 \
  -e DB_HOST=neurondb-cpu \
  -e DB_PORT=5432 \
  -e DB_NAME=neurondb \
  -e DB_USER=neurondb \
  -e DB_PASSWORD=neurondb \
  -e SERVER_HOST=0.0.0.0 \
  -e SERVER_PORT=8080 \
  --network neurondb-network \
  neuronagent:package
```

### Step 4: Verify Installation

Wait for container to be ready:

```bash
sleep 10
```

Check container status:

```bash
sudo docker ps | grep neuronagent
```

Test health endpoint:

```bash
curl http://localhost:8080/health
```

Expected response:
```json
{"status":"healthy","database":"connected"}
```

View logs:

```bash
sudo docker logs neuronagent
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | `localhost` | NeuronDB database hostname |
| `DB_PORT` | `5433` | NeuronDB database port |
| `DB_NAME` | `neurondb` | Database name |
| `DB_USER` | `neurondb` | Database username |
| `DB_PASSWORD` | `neurondb` | Database password |
| `DB_MAX_OPEN_CONNS` | `25` | Maximum open connections |
| `DB_MAX_IDLE_CONNS` | `5` | Maximum idle connections |
| `DB_CONN_MAX_LIFETIME` | `5m` | Connection max lifetime |
| `SERVER_HOST` | `0.0.0.0` | Server bind address |
| `SERVER_PORT` | `8080` | Server port number |
| `SERVER_READ_TIMEOUT` | `30s` | Read timeout duration |
| `SERVER_WRITE_TIMEOUT` | `30s` | Write timeout duration |
| `LOG_LEVEL` | `info` | Log level (debug, info, warn, error) |
| `LOG_FORMAT` | `json` | Log format (json, text) |
| `CONFIG_PATH` | - | Path to config.yaml file |

Environment variables override configuration file values.

### Configuration File

Create `config.yaml` and mount in container:

```yaml
database:
  host: localhost
  port: 5432
  name: neurondb
  user: neurondb
  password: neurondb

server:
  host: 0.0.0.0
  port: 8080
```

Mount in `docker-compose.yml`:

```yaml
volumes:
  - ./config.yaml:/app/config.yaml:ro
```

## Connecting to NeuronDB

### From Host Machine

Set connection parameters for localhost access:

```env
DB_HOST=localhost
DB_PORT=5433
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

Connect NeuronAgent container:

```bash
docker network connect neurondb-network neuronagent
```

Update environment variables:

```env
DB_HOST=neurondb-cpu
DB_PORT=5432
```

Use container name as hostname. Port 5432 is the internal container port.

### Network Configuration Example

Update `docker-compose.yml`:

```yaml
networks:
  neurondb-network:
    external: true

services:
  agent-server:
    networks:
      - neurondb-network
```

## Database Setup

### NeuronDB Extension

Ensure NeuronDB extension is installed:

```sql
CREATE EXTENSION IF NOT EXISTS neurondb;
```

### Permissions

Grant necessary permissions:

```sql
GRANT ALL PRIVILEGES ON DATABASE neurondb TO neurondb;
GRANT ALL ON SCHEMA neurondb_agent TO neurondb;
```

### Automatic Migrations

NeuronAgent runs migrations automatically on startup:

1. `001_initial_schema.sql` - Creates schema and tables
2. `002_add_indexes.sql` - Adds database indexes
3. `003_add_triggers.sql` - Adds triggers

Migrations execute in order. Service starts after successful migration.

## Building the Image

### Standard Build

Using Docker Compose:

```bash
docker compose build
```

### Custom Build

Using Docker directly:

```bash
docker build -f docker/Dockerfile -t neuronagent:latest ..
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
  --build-arg VERSION=1.0.0 \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  --build-arg VCS_REF=$(git rev-parse --short HEAD) \
  -t neuronagent:1.0.0 ..
```

### Production Build

For production builds with versioning:

```bash
VERSION=1.0.0
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD)

docker build -f docker/Dockerfile \
  --build-arg GO_VERSION=1.23 \
  --build-arg VERSION=${VERSION} \
  --build-arg BUILD_DATE=${BUILD_DATE} \
  --build-arg VCS_REF=${VCS_REF} \
  -t neuronagent:${VERSION} \
  -t neuronagent:latest ..
```

## Production Deployment

### Using Production Compose File

The `docker-compose.prod.yml` file extends the base configuration with production-focused settings:

- Enhanced resource limits
- Structured JSON logging
- Security hardening (read-only filesystem, dropped capabilities)
- Production restart policies
- Extended health checks

Start with production configuration:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Production Environment Variables

Create `.env.prod` for production:

```env
DB_HOST=neurondb-cpu
DB_PORT=5432
DB_NAME=neurondb
DB_USER=neurondb
DB_PASSWORD=<secure-password>
LOG_LEVEL=info
LOG_FORMAT=json
DB_SSL_MODE=require
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
- Checks for migration files
- Logs startup information
- Executes the NeuronAgent binary

The entrypoint can be overridden in `docker-compose.yml`:

```yaml
services:
  agent-server:
    entrypoint: ["./agent-server"]  # Skip entrypoint script
```

## SSL/TLS Configuration

### Secure Database Connections

NeuronAgent supports SSL/TLS for secure database connections. Configure SSL in your environment:

```env
# SSL mode: disable, allow, prefer, require, verify-ca, verify-full
DB_SSL_MODE=require

# SSL certificate paths (inside container)
DB_SSL_CERT=/app/ssl/client-cert.pem
DB_SSL_KEY=/app/ssl/client-key.pem
DB_SSL_ROOT_CERT=/app/ssl/ca-cert.pem
```

### Using Connection String with SSL

If your database connection library supports connection strings, you can include SSL parameters:

```env
DB_CONNECTION_STRING=postgresql://user:password@host:port/database?sslmode=require&sslcert=/app/ssl/client-cert.pem&sslkey=/app/ssl/client-key.pem&sslrootcert=/app/ssl/ca-cert.pem
```

### Mounting SSL Certificates

In `docker-compose.yml`:

```yaml
services:
  agent-server:
    volumes:
      - ./ssl:/app/ssl:ro
    environment:
      DB_SSL_MODE: require
      DB_SSL_CERT: /app/ssl/client-cert.pem
      DB_SSL_KEY: /app/ssl/client-key.pem
      DB_SSL_ROOT_CERT: /app/ssl/ca-cert.pem
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
docker compose logs -f agent-server
```

View last 100 lines:

```bash
docker compose logs --tail=100 agent-server
```

### Execute Commands

Run shell in container:

```bash
docker compose exec agent-server /bin/sh
```

## Health Checks

### Container Health Check

Container includes built-in health check. Check status:

```bash
docker inspect neuronagent | jq '.[0].State.Health'
```

### Manual Health Check

Test health endpoint:

```bash
curl http://localhost:8080/health
```

Expected response:

```json
{
  "status": "healthy",
  "database": "connected"
}
```

Health endpoint returns:
- `200 OK` if healthy and database connected
- `503 Service Unavailable` if database connection fails

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check endpoint |
| `/metrics` | GET | Prometheus metrics |
| `/api/v1/agents` | POST | Create new agent |
| `/api/v1/agents` | GET | List all agents |
| `/api/v1/agents/{id}` | GET | Get agent details |
| `/api/v1/agents/{id}` | PUT | Update agent |
| `/api/v1/agents/{id}` | DELETE | Delete agent |
| `/api/v1/sessions` | POST | Create new session |
| `/api/v1/sessions/{id}/messages` | POST | Send message to agent |
| `/ws` | WebSocket | Streaming agent responses |

See [API Documentation](../docs/API.md) for complete reference.

## Troubleshooting

### Container Will Not Start

Check container logs:

```bash
docker compose logs agent-server
```

Common issues:
- Missing environment variables
- Invalid database connection parameters
- Port already in use
- Network connectivity issues

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
docker compose config | grep -A 10 DB_
```

Verify network connectivity:

```bash
docker exec agent-server ping neurondb-cpu
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

### Port Already in Use

Change port mapping in `.env`:

```env
SERVER_PORT=8081
```

Or modify `docker-compose.yml`:

```yaml
ports:
  - "8081:8080"
```

### API Not Responding

Check service is running:

```bash
docker compose ps agent-server
```

Test health endpoint:

```bash
curl http://localhost:8080/health
```

Check logs for errors:

```bash
docker compose logs agent-server | grep -i error
```

Verify API key:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:8080/api/v1/agents
```

### Migration Errors

Check migration logs:

```bash
docker compose logs agent-server | grep -i migration
```

Verify database permissions:

```sql
GRANT ALL ON SCHEMA neurondb_agent TO neurondb;
```

## Security

### Container Security Best Practices

The NeuronAgent Docker image implements multiple security best practices:

#### Non-Root User
- Container runs as non-root user `neuronagent` (UID 1000)
- Prevents privilege escalation attacks
- Minimal filesystem permissions

#### Minimal Base Image
- Uses Debian slim base image
- Reduced attack surface
- Only essential runtime dependencies

#### Network Exposure
- Exposes only necessary port (8080)
- No unnecessary network services
- Can be restricted with firewall rules

#### Read-Only Filesystem (Production)

In `docker-compose.prod.yml`, enable read-only root filesystem:

```yaml
services:
  agent-server:
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=100m
      - /var/tmp:noexec,nosuid,size=100m
```

#### Dropped Capabilities

Production configuration drops all capabilities:

```yaml
services:
  agent-server:
    cap_drop:
      - ALL
    security_opt:
      - no-new-privileges:true
```

#### Image Scanning

Regularly scan images for vulnerabilities:

```bash
# Using Trivy
trivy image neuronagent:latest

# Using Docker Scout
docker scout cves neuronagent:latest

# Using Snyk
snyk test --docker neuronagent:latest
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
  agent-server:
    secrets:
      - db_password
      - db_user
    environment:
      DB_PASSWORD_FILE: /run/secrets/db_password
      DB_USER_FILE: /run/secrets/db_user
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
export DB_PASSWORD=$(vault kv get -field=password secret/neurondb)
```

**AWS Secrets Manager:**
```bash
export DB_PASSWORD=$(aws secretsmanager get-secret-value \
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
- **Firewall Rules**: Restrict external access to port 8080
- **TLS/SSL**: Enable TLS/SSL for all database connections in production
- **Network Policies**: Implement network policies in orchestrated environments (Kubernetes, Docker Swarm)
- **Reverse Proxy**: Use nginx or traefik as reverse proxy with TLS termination

### API Security

- **API Key Authentication**: Required for all API endpoints
- **Rate Limiting**: Enabled to prevent abuse
- **Secure Storage**: Store API keys securely (never in code or logs)
- **Key Rotation**: Rotate keys regularly
- **HTTPS**: Use reverse proxy with TLS for production

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
- [ ] Use reverse proxy with TLS for API access
- [ ] Implement rate limiting
- [ ] Rotate API keys regularly

## Integration with NeuronDB

### Requirements

- PostgreSQL 16 or later
- NeuronDB extension installed and enabled
- Database user with appropriate permissions
- Network connectivity to database

### Setup Instructions

See [NeuronDB Docker README](../../NeuronDB/docker/README.md) for NeuronDB setup.

## Production Deployment

### Recommendations

| Practice | Implementation |
|----------|----------------|
| Use Secrets | Store credentials in Docker secrets or external system |
| Enable Health Checks | Monitor container health status |
| Configure Logging | Set up log aggregation and monitoring |
| Resource Limits | Set CPU and memory limits |
| Restart Policy | Configure restart policy for reliability |

### Resource Limits

Example `docker-compose.yml`:

```yaml
services:
  agent-server:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
```

## Logging Integration

### Structured JSON Logging

For production, use JSON logging format for easier parsing:

```yaml
services:
  agent-server:
    environment:
      LOG_FORMAT: json
```

### Docker Logging Drivers

#### JSON File Driver (Default)

Configure log rotation:

```yaml
services:
  agent-server:
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
  agent-server:
    logging:
      driver: "syslog"
      options:
        syslog-address: "tcp://localhost:514"
        syslog-facility: "daemon"
        tag: "neuronagent"
```

#### Fluentd Driver

Send logs to Fluentd:

```yaml
services:
  agent-server:
    logging:
      driver: "fluentd"
      options:
        fluentd-address: "localhost:24224"
        tag: "neuronagent.server"
```

#### Loki Driver

Send logs to Grafana Loki:

```yaml
services:
  agent-server:
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
  agent-server:
    logging:
      driver: "gelf"
      options:
        gelf-address: "udp://localhost:12201"
        tag: "neuronagent"
```

### Log Aggregation Examples

#### ELK Stack (Elasticsearch, Logstash, Kibana)

```yaml
services:
  agent-server:
    logging:
      driver: "gelf"
      options:
        gelf-address: "udp://logstash:12201"
        tag: "neuronagent"
```

#### Splunk

```yaml
services:
  agent-server:
    logging:
      driver: "splunk"
      options:
        splunk-token: "${SPLUNK_TOKEN}"
        splunk-url: "https://splunk.example.com:8088"
        splunk-source: "neuronagent"
        splunk-sourcetype: "json"
```

#### CloudWatch Logs (AWS)

```yaml
services:
  agent-server:
    logging:
      driver: "awslogs"
      options:
        awslogs-group: "/docker/neuronagent"
        awslogs-region: "us-east-1"
        awslogs-stream-prefix: "agent"
```

#### Google Cloud Logging

```yaml
services:
  agent-server:
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
docker compose logs agent-server | jq 'select(.level == "error")'

# Count log entries by level
docker compose logs agent-server | jq -r '.level' | sort | uniq -c

# Filter by timestamp
docker compose logs agent-server | jq 'select(.time > "2024-01-01T00:00:00Z")'
```

## Monitoring

### Container Metrics

#### Docker Stats

Monitor real-time container metrics:

```bash
docker stats neuronagent
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

Access metrics at `http://localhost:8080/containers/neuronagent`

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
  - job_name: 'neuronagent'
    static_configs:
      - targets: ['agent-server:8080']
    metrics_path: '/metrics'
```

### Application Metrics

NeuronAgent exposes Prometheus metrics at `/metrics`:

```bash
curl http://localhost:8080/metrics
```

Metrics include:
- HTTP request counts and durations
- Database connection pool stats
- Agent execution metrics
- Session and message counts

### Health Monitoring

#### Health Check Status

Check container health:

```bash
docker inspect --format='{{.State.Health.Status}}' neuronagent
```

#### Health Check Logs

View health check history:

```bash
docker inspect --format='{{json .State.Health}}' neuronagent | jq
```

#### Application Health Endpoint

Test application health:

```bash
curl http://localhost:8080/health
```

Expected response:

```json
{
  "status": "healthy",
  "database": "connected"
}
```

### Alerting

#### Prometheus Alertmanager

Set up alerts for:

- Container down
- High CPU usage (>80%)
- High memory usage (>90%)
- Health check failures
- Database connection errors
- High API error rates

Example alert rule:

```yaml
groups:
  - name: neuronagent
    rules:
      - alert: NeuronAgentDown
        expr: up{job="neuronagent"} == 0
        for: 1m
        annotations:
          summary: "NeuronAgent container is down"
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        annotations:
          summary: "High error rate detected"
```

#### Grafana Dashboards

Create dashboards for:

- Container resource usage
- HTTP request rates and latencies
- Database connection pool metrics
- Agent execution metrics
- Error rates and types
- Log volume and patterns

### Monitoring Best Practices

1. **Set Up Alerts**: Configure alerts for critical metrics
2. **Log Aggregation**: Centralize logs for analysis
3. **Metrics Retention**: Configure appropriate retention periods
4. **Dashboard Creation**: Create dashboards for key metrics
5. **Regular Reviews**: Review metrics and logs regularly
6. **Capacity Planning**: Monitor trends for capacity planning
7. **Performance Baselines**: Establish performance baselines
8. **Anomaly Detection**: Set up anomaly detection for unusual patterns

## Support

- **Documentation**: [NeuronAgent README](../README.md)
- **API Docs**: [API Documentation](../docs/API.md)
- **GitHub Issues**: [Report Issues](https://github.com/neurondb/NeurondB/issues)
- **Email**: support@neurondb.ai

## License

See [LICENSE](../../LICENSE) file for license information.
