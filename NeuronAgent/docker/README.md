# NeuronAgent Docker Setup

[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](../../LICENSE)

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

### Step 1: Copy Environment File

```bash
cp .env.example .env
```

### Step 2: Configure Database Connection

Edit `.env` file:

```env
DB_HOST=localhost
DB_PORT=5433
DB_NAME=neurondb
DB_USER=neurondb
DB_PASSWORD=neurondb
SERVER_HOST=0.0.0.0
SERVER_PORT=8080
```

### Step 3: Build and Start

```bash
docker compose build
docker compose up -d
```

### Step 4: Verify Installation

Check container status:

```bash
docker compose ps
```

Test health endpoint:

```bash
curl http://localhost:8080/health
```

Test API:

```bash
curl -H "Authorization: Bearer <your-api-key>" \
  http://localhost:8080/api/v1/agents
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

Example:

```bash
docker build -f docker/Dockerfile \
  --build-arg GO_VERSION=1.23 \
  -t neuronagent:latest ..
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

### Container Security

- Container runs as non-root user `neuronagent`
- Uses Debian slim base image
- Minimal attack surface
- No unnecessary packages

### Credential Management

Store credentials securely:

**Docker Secrets:**

```yaml
secrets:
  db_password:
    file: ./secrets/db_password.txt

services:
  agent-server:
    secrets:
      - db_password
    environment:
      DB_PASSWORD_FILE: /run/secrets/db_password
```

**Environment Variables:**

Use `.env` file with restricted permissions:

```bash
chmod 600 .env
```

**External Secrets Management:**

Integrate with HashiCorp Vault, AWS Secrets Manager, or similar.

### Network Security

- Use Docker networks to isolate services
- Restrict container network access
- Use firewall rules for external access
- Enable TLS/SSL for database connections

### API Security

- API key authentication required
- Rate limiting enabled
- Store API keys securely
- Rotate keys regularly

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

### Monitoring

Monitor container metrics:

```bash
docker stats neuronagent
```

Expose Prometheus metrics:

```bash
curl http://localhost:8080/metrics
```

## Support

- **Documentation**: [NeuronAgent README](../README.md)
- **API Docs**: [API Documentation](../docs/API.md)
- **GitHub Issues**: [Report Issues](https://github.com/neurondb/NeurondB/issues)
- **Email**: support@neurondb.ai

## License

See [LICENSE](../../LICENSE) file for license information.
