# NeuronDB Ecosystem Docker Setup

[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](../../LICENSE)

Complete Docker deployment guide for running NeuronDB, NeuronAgent, and NeuronMCP as separate Docker services. Each service connects to NeuronDB PostgreSQL independently and can run in any order.

## Overview

The NeuronDB ecosystem consists of three independent Docker services:

| Service | Purpose | Port | Protocol |
|---------|---------|------|----------|
| **NeuronDB** | PostgreSQL extension with vector search and ML | 5433 | TCP (PostgreSQL) |
| **NeuronAgent** | Agent runtime with REST API and WebSocket | 8080 | HTTP/WebSocket |
| **NeuronMCP** | Model Context Protocol server | - | Stdio (JSON-RPC) |

All services connect to the same NeuronDB PostgreSQL instance. Services operate independently and can be started or stopped separately.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Network                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │  NeuronDB    │◄─────┤  NeuronAgent │      │  NeuronMCP   │  │
│  │  Container   │      │  Container   │      │  Container   │  │
│  │  Port: 5433  │      │  Port: 8080  │      │  Stdio Only  │  │
│  └──────┬───────┘      └──────┬───────┘      └──────┬───────┘  │
│         │                     │                     │            │
│         └─────────────────────┼─────────────────────┘            │
│                               │                                  │
│                    ┌───────────▼───────────┐                     │
│                    │  Shared PostgreSQL    │                     │
│                    │  Database (NeuronDB)   │                     │
│                    └───────────────────────┘                      │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Step 1: Start NeuronDB

Navigate to NeuronDB directory:

```bash
cd NeuronDB/docker
```

Build and start container:

```bash
docker compose build neurondb
docker compose up -d neurondb
```

Wait for healthy status:

```bash
docker compose ps neurondb
```

Verify connection:

```bash
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" \
  -c "SELECT neurondb.version();"
```

### Step 2: Start NeuronAgent

Navigate to NeuronAgent directory:

```bash
cd ../../NeuronAgent/docker
```

Copy environment file:

```bash
cp .env.example .env
```

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

Build and start:

```bash
docker compose build
docker compose up -d agent-server
```

Verify API:

```bash
curl http://localhost:8080/health
```

### Step 3: Start NeuronMCP

Navigate to NeuronMCP directory:

```bash
cd ../../NeuronMCP/docker
```

Copy environment file:

```bash
cp .env.example .env
```

Edit `.env` file:

```env
NEURONDB_HOST=localhost
NEURONDB_PORT=5433
NEURONDB_DATABASE=neurondb
NEURONDB_USER=neurondb
NEURONDB_PASSWORD=neurondb
```

Build and start:

```bash
docker compose build
docker compose up -d neurondb-mcp
```

Verify container:

```bash
docker compose ps neurondb-mcp
```

## Network Configuration

### Option 1: Host Networking

All services use `localhost` for connections.

**NeuronAgent Configuration:**

```env
DB_HOST=localhost
DB_PORT=5433
```

**NeuronMCP Configuration:**

```env
NEURONDB_HOST=localhost
NEURONDB_PORT=5433
```

**Advantages:**
- Simple setup
- No network configuration required
- Works for local development

**Disadvantages:**
- Services must run on same host
- Less isolation between services

### Option 2: Docker Network

Create shared Docker network for container communication.

**Quick Setup (Recommended):**

Use the provided setup script:

```bash
cd NeuronDB/docker
./setup-network.sh
```

This creates the `neurondb-network` and provides instructions for connecting services.

**Manual Setup:**

Create shared network:

```bash
docker network create neurondb-network
```

**Connect NeuronDB:**

```bash
docker network connect neurondb-network neurondb-cpu
```

**Connect NeuronAgent:**

```bash
docker network connect neurondb-network neuronagent
```

**Connect NeuronMCP:**

```bash
docker network connect neurondb-network neurondb-mcp
```

**Update Configuration:**

**NeuronAgent `.env`:**

```env
DB_HOST=neurondb-cpu
DB_PORT=5432
```

Note: Port 5432 is the internal container port when using Docker network. Use port 5433 when connecting from host.

**NeuronMCP `.env`:**

```env
NEURONDB_HOST=neurondb-cpu
NEURONDB_PORT=5432
```

Note: Port 5432 is the internal container port when using Docker network. Use port 5433 when connecting from host.

**Advantages:**
- Better isolation
- Service discovery via container names
- Production-ready setup

**Disadvantages:**
- Requires network configuration
- Container names must match

### Option 3: Docker Compose Networks (External Network)

Configure networks in each `docker-compose.yml` to use an external shared network:

**Step 1: Create the network (if not already created):**

```bash
cd NeuronDB/docker
./setup-network.sh
# Or manually: docker network create neurondb-network
```

**Step 2: Update NeuronDB `docker-compose.yml`:**

Uncomment the external network configuration in the networks section.

**Step 3: Update NeuronAgent `docker-compose.yml`:**

Uncomment the neurondb-network configuration in networks section and service networks.

**Step 4: Update NeuronMCP `docker-compose.yml`:**

Uncomment the neurondb-network configuration in networks section and service networks.

All services will then automatically connect to the shared network on startup.

## Service Connection Matrix

| Service | Host Port | Container Port | Protocol | Connection To |
|---------|-----------|----------------|----------|---------------|
| **NeuronDB** | 5433 | 5432 | TCP (PostgreSQL) | - |
| **NeuronAgent** | 8080 | 8080 | HTTP/WebSocket | NeuronDB (5432/5433) |
| **NeuronMCP** | - | - | Stdio (JSON-RPC) | NeuronDB (5432/5433) |

### Connection Details

**NeuronDB:**
- Exposed on host port `5433`
- Internal container port `5432`
- Accepts connections from other containers and host

**NeuronAgent:**
- Exposed on host port `8080`
- Connects to NeuronDB using `DB_HOST` and `DB_PORT`
- Accepts HTTP and WebSocket connections

**NeuronMCP:**
- No network ports (stdio only)
- Connects to NeuronDB using `NEURONDB_HOST` and `NEURONDB_PORT`
- Communicates via stdin/stdout with MCP clients

## Complete Setup Example

### Create Shared Network

**Option 1: Use setup script (recommended):**

```bash
cd NeuronDB/docker
./setup-network.sh
```

**Option 2: Create manually:**

```bash
docker network create neurondb-network
```

### Start NeuronDB

```bash
cd NeuronDB/docker
docker compose up -d neurondb
docker network connect neurondb-network neurondb-cpu
```

Wait for healthy status:

```bash
docker compose ps neurondb
```

### Start NeuronAgent

```bash
cd ../../NeuronAgent/docker

cat > .env << EOF
DB_HOST=neurondb-cpu
DB_PORT=5432
DB_NAME=neurondb
DB_USER=neurondb
DB_PASSWORD=neurondb
SERVER_HOST=0.0.0.0
SERVER_PORT=8080
LOG_LEVEL=info
LOG_FORMAT=json
EOF

docker compose build
docker compose up -d agent-server
docker network connect neurondb-network neuronagent
```

### Start NeuronMCP

```bash
cd ../../NeuronMCP/docker

cat > .env << EOF
NEURONDB_HOST=neurondb-cpu
NEURONDB_PORT=5432
NEURONDB_DATABASE=neurondb
NEURONDB_USER=neurondb
NEURONDB_PASSWORD=neurondb
NEURONDB_LOG_LEVEL=info
NEURONDB_LOG_FORMAT=text
EOF

docker compose build
docker compose up -d neurondb-mcp
docker network connect neurondb-network neurondb-mcp
```

### Verify All Services

Check NeuronDB:

```bash
docker compose -f NeuronDB/docker/docker-compose.yml ps neurondb
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" \
  -c "SELECT neurondb.version();"
```

Check NeuronAgent:

```bash
curl http://localhost:8080/health
curl -H "Authorization: Bearer <api-key>" \
  http://localhost:8080/api/v1/agents
```

Check NeuronMCP:

```bash
docker compose -f NeuronMCP/docker/docker-compose.yml ps neurondb-mcp
docker compose -f NeuronMCP/docker/docker-compose.yml logs neurondb-mcp
```

## Configuration Reference

### NeuronDB Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Port Mapping** | `5433:5432` | Host port to container port |
| **Database** | `neurondb` | Database name |
| **User** | `neurondb` | Database username |
| **Password** | `neurondb` | Database password |

**Change password in production:**

```bash
docker exec -it neurondb-cpu \
  psql -U neurondb -d neurondb \
  -c "ALTER USER neurondb WITH PASSWORD 'new_password';"
```

### NeuronAgent Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | `localhost` | NeuronDB hostname |
| `DB_PORT` | `5433` | NeuronDB port |
| `DB_NAME` | `neurondb` | Database name |
| `DB_USER` | `neurondb` | Database username |
| `DB_PASSWORD` | `neurondb` | Database password |
| `SERVER_PORT` | `8080` | API server port |

See [NeuronAgent Docker README](../../NeuronAgent/docker/README.md) for complete configuration options.

### NeuronMCP Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `NEURONDB_HOST` | `localhost` | NeuronDB hostname |
| `NEURONDB_PORT` | `5433` | NeuronDB port |
| `NEURONDB_DATABASE` | `neurondb` | Database name |
| `NEURONDB_USER` | `neurondb` | Database username |
| `NEURONDB_PASSWORD` | `neurondb` | Database password |

See [NeuronMCP Docker README](../../NeuronMCP/docker/README.md) for complete configuration options.

## Service Management

### Start All Services

Start services in any order:

```bash
# Start NeuronDB
cd NeuronDB/docker && docker compose up -d neurondb

# Start NeuronAgent
cd ../../NeuronAgent/docker && docker compose up -d agent-server

# Start NeuronMCP
cd ../../NeuronMCP/docker && docker compose up -d neurondb-mcp
```

### Stop All Services

Stop services independently:

```bash
# Stop NeuronMCP
cd NeuronMCP/docker && docker compose down

# Stop NeuronAgent
cd ../NeuronAgent/docker && docker compose down

# Stop NeuronDB
cd ../../NeuronDB/docker && docker compose down neurondb
```

### Restart Services

Restart individual services:

```bash
# Restart NeuronDB
cd NeuronDB/docker && docker compose restart neurondb

# Restart NeuronAgent
cd ../../NeuronAgent/docker && docker compose restart agent-server

# Restart NeuronMCP
cd ../../NeuronMCP/docker && docker compose restart neurondb-mcp
```

### View Logs

Follow logs for all services:

```bash
# NeuronDB logs
docker compose -f NeuronDB/docker/docker-compose.yml logs -f neurondb

# NeuronAgent logs
docker compose -f NeuronAgent/docker/docker-compose.yml logs -f agent-server

# NeuronMCP logs
docker compose -f NeuronMCP/docker/docker-compose.yml logs -f neurondb-mcp
```

### Check Health

Verify service health:

```bash
# NeuronDB health
docker inspect neurondb-cpu | jq '.[0].State.Health'

# NeuronAgent health
curl http://localhost:8080/health

# NeuronMCP status
docker compose -f NeuronMCP/docker/docker-compose.yml logs neurondb-mcp | tail -20
```

## Troubleshooting

### Services Cannot Connect to NeuronDB

**Symptoms:**
- Connection timeout errors
- Authentication failures
- Network unreachable errors

**Solutions:**

1. Verify NeuronDB is running:

```bash
docker compose ps neurondb
```

2. Check network connectivity:

```bash
# From NeuronAgent container
docker exec neuronagent ping neurondb-cpu

# From NeuronMCP container
docker exec neurondb-mcp ping neurondb-cpu
```

3. Test connection manually:

```bash
psql -h localhost -p 5433 -U neurondb -d neurondb -c "SELECT 1;"
```

4. Verify firewall settings:

```bash
sudo ufw status
```

5. Check environment variables match:

```bash
# NeuronAgent
docker compose -f NeuronAgent/docker/docker-compose.yml config | grep DB_

# NeuronMCP
docker compose -f NeuronMCP/docker/docker-compose.yml config | grep NEURONDB_
```

### Port Already in Use

**Symptoms:**
- Port binding errors
- Service fails to start

**Solutions:**

1. Change port mappings in `docker-compose.yml`:

```yaml
ports:
  - "5434:5432"  # Different host port
```

2. Stop conflicting services:

```bash
# Find process using port
sudo lsof -i :5433
sudo lsof -i :8080

# Stop conflicting service
docker compose stop <service-name>
```

3. Use different ports:

```env
# NeuronDB
POSTGRES_PORT=5434

# NeuronAgent
SERVER_PORT=8081
```

### NeuronDB Extension Not Found

**Symptoms:**
- Extension creation errors
- Function not found errors

**Solutions:**

1. Verify extension installed:

```sql
SELECT * FROM pg_extension WHERE extname = 'neurondb';
```

2. Create extension if missing:

```sql
CREATE EXTENSION neurondb;
```

3. Check database name matches configuration:

```bash
docker exec neurondb-cpu psql -U neurondb -l
```

4. Verify user has permissions:

```sql
GRANT ALL PRIVILEGES ON DATABASE neurondb TO neurondb;
```

### MCP Client Cannot Connect

**Symptoms:**
- MCP client connection failures
- Stdio communication errors

**Solutions:**

1. Verify container running with stdio enabled:

```yaml
services:
  neurondb-mcp:
    stdin_open: true
    tty: true
```

2. Check MCP client configuration path:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

**Linux:** `~/.config/Claude/claude_desktop_config.json`

3. Test stdio manually:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | \
  docker compose exec -T neurondb-mcp ./neurondb-mcp
```

4. Verify Docker command syntax in MCP client config:

```json
{
  "mcpServers": {
    "neurondb": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "--network", "neurondb-network", ...]
    }
  }
}
```

5. Check network connectivity:

```bash
docker network inspect neurondb-network
```

## Best Practices

### Security

| Practice | Implementation |
|----------|----------------|
| Change Default Passwords | Update database passwords in production |
| Use Docker Secrets | Store credentials in Docker secrets |
| Network Isolation | Use Docker networks to isolate services |
| Enable SSL/TLS | Configure encrypted database connections |
| API Key Management | Implement proper key rotation for NeuronAgent |

### Performance

| Practice | Implementation |
|----------|----------------|
| Resource Limits | Set CPU and memory limits per container |
| Connection Pooling | Configure connection pool settings |
| Health Checks | Monitor container health status |
| Logging | Set up log aggregation and monitoring |

### Reliability

| Practice | Implementation |
|----------|----------------|
| Restart Policies | Configure restart policies for containers |
| Health Checks | Implement health check endpoints |
| Backup Strategy | Set up database backup procedures |
| Monitoring | Monitor service metrics and logs |

## Support

- **Documentation**: [Main README](../../README.md)
- **NeuronDB Docker**: [NeuronDB Docker README](README.md)
- **NeuronAgent Docker**: [NeuronAgent Docker README](../../NeuronAgent/docker/README.md)
- **NeuronMCP Docker**: [NeuronMCP Docker README](../../NeuronMCP/docker/README.md)
- **GitHub Issues**: [Report Issues](https://github.com/neurondb/NeurondB/issues)
- **Email**: support@neurondb.ai

## License

See [LICENSE](../../LICENSE) file for license information.
