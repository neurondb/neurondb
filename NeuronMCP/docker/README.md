# NeuronMCP Docker Setup

[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](../../LICENSE)
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

### Step 1: Copy Environment File

```bash
cp .env.example .env
```

### Step 2: Configure Database Connection

Edit `.env` file:

```env
NEURONDB_HOST=localhost
NEURONDB_PORT=5433
NEURONDB_DATABASE=neurondb
NEURONDB_USER=neurondb
NEURONDB_PASSWORD=neurondb
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

View logs:

```bash
docker compose logs neurondb-mcp
```

Test stdio communication:

```bash
docker compose exec neurondb-mcp ./neurondb-mcp
```

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

Example:

```bash
docker build -f docker/Dockerfile \
  --build-arg GO_VERSION=1.23 \
  -t neurondb-mcp:latest ..
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

### Container Security

- Container runs as non-root user `neuronmcp`
- Uses Debian slim base image
- Minimal attack surface
- No network endpoints (stdio only)

### Credential Management

Store credentials securely:

**Docker Secrets:**

```yaml
secrets:
  db_password:
    file: ./secrets/db_password.txt

services:
  neurondb-mcp:
    secrets:
      - db_password
    environment:
      NEURONDB_PASSWORD_FILE: /run/secrets/db_password
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
- No external network exposure (stdio only)
- Enable TLS/SSL for database connections

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

### Monitoring

Monitor container metrics:

```bash
docker stats neurondb-mcp
```

View logs for debugging:

```bash
docker compose logs -f neurondb-mcp
```

## Support

- **Documentation**: [NeuronMCP README](../README.md)
- **GitHub Issues**: [Report Issues](https://github.com/neurondb/NeurondB/issues)
- **Email**: support@neurondb.ai

## License

See [LICENSE](../../LICENSE) file for license information.
