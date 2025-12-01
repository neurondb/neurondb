# NeuronMCP

[![Go](https://img.shields.io/badge/Go-1.23+-00ADD8.svg)](https://golang.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-blue.svg)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](../LICENSE)
[![MCP](https://img.shields.io/badge/MCP-Protocol-blue.svg)](https://modelcontextprotocol.io/)

Model Context Protocol server for NeuronDB PostgreSQL extension, implemented in Go. Enables MCP-compatible clients to access NeuronDB vector search, ML algorithms, and RAG capabilities.

## Overview

NeuronMCP implements the Model Context Protocol using JSON-RPC 2.0 over stdio. It provides tools and resources for MCP clients to interact with NeuronDB, including vector operations, ML model training, and database schema management.

## Features

| Feature | Description |
|---------|-------------|
| **MCP Protocol** | Full JSON-RPC 2.0 implementation with stdio transport |
| **Vector Operations** | Search, embedding generation, indexing tools |
| **ML Tools** | Training and prediction for various algorithms |
| **Resources** | Schema, models, indexes, config, workers, stats |
| **Middleware** | Validation, logging, timeout, error handling |
| **Configuration** | JSON config files with environment variable overrides |
| **Modular Architecture** | Clean separation of concerns |

## Architecture

```
┌─────────────────────────────────────────────┐
│          MCP Client                         │
│  (Claude Desktop, etc.)                     │
└──────────────┬──────────────────────────────┘
               │ stdio (JSON-RPC 2.0)
┌──────────────▼──────────────────────────────┐
│          NeuronMCP Server                   │
├─────────────────────────────────────────────┤
│  MCP Protocol Handler                       │
├─────────────────────────────────────────────┤
│  Tools │  Resources │  Middleware           │
├─────────────────────────────────────────────┤
│          NeuronDB PostgreSQL                │
│  (Vector Search │  ML │  Embeddings)        │
└─────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- PostgreSQL 16 or later
- NeuronDB extension installed
- Go 1.23 or later (for building from source)
- MCP-compatible client (e.g., Claude Desktop)

### Database Setup

Create database and extension:

```bash
createdb neurondb
psql -d neurondb -c "CREATE EXTENSION neurondb;"
```

### Configuration

Create `mcp-config.json`:

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

Or use environment variables:

```bash
export NEURONDB_HOST=localhost
export NEURONDB_PORT=5432
export NEURONDB_DATABASE=neurondb
export NEURONDB_USER=neurondb
export NEURONDB_PASSWORD=neurondb
```

### Build and Run

From source:

```bash
go build ./cmd/neurondb-mcp
./neurondb-mcp
```

Using Docker:

```bash
cd docker
cp .env.example .env
# Edit .env with your configuration
docker compose up -d
```

See [Docker Guide](docker/README.md) for Docker deployment details.

## MCP Protocol

NeuronMCP uses Model Context Protocol over stdio:

- Communication via stdin and stdout
- Messages follow JSON-RPC 2.0 format
- Clients initiate all requests
- Server responds with results or errors

Example request:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "vector_search",
    "arguments": {
      "query_vector": [0.1, 0.2, 0.3],
      "table": "documents",
      "limit": 10
    }
  }
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEURONDB_HOST` | `localhost` | Database hostname |
| `NEURONDB_PORT` | `5432` | Database port |
| `NEURONDB_DATABASE` | `neurondb` | Database name |
| `NEURONDB_USER` | `neurondb` | Database username |
| `NEURONDB_PASSWORD` | `neurondb` | Database password |
| `NEURONDB_CONNECTION_STRING` | - | Full connection string (overrides above) |
| `NEURONDB_MCP_CONFIG` | `mcp-config.json` | Path to config file |
| `NEURONDB_LOG_LEVEL` | `info` | Log level (debug, info, warn, error) |
| `NEURONDB_LOG_FORMAT` | `text` | Log format (json, text) |
| `NEURONDB_LOG_OUTPUT` | `stderr` | Log output (stdout, stderr, file) |
| `NEURONDB_ENABLE_GPU` | `false` | Enable GPU acceleration |

### Configuration File

See `mcp-config.json.example` for complete configuration structure. Environment variables override configuration file values.

## Tools

NeuronMCP provides the following tools:

| Tool Category | Tools |
|---------------|-------|
| **Vector Operations** | `vector_search`, `vector_similarity`, `generate_embedding`, `create_vector_index` |
| **ML Operations** | `train_model`, `predict`, `evaluate_model`, `list_models` |
| **Analytics** | `analyze_data`, `cluster_data`, `reduce_dimensionality` |
| **RAG Operations** | `process_document`, `retrieve_context`, `generate_response` |

See tool documentation for complete parameter lists and examples.

## Resources

NeuronMCP exposes the following resources:

| Resource | Description |
|----------|-------------|
| `schema` | Database schema information |
| `models` | Available ML models |
| `indexes` | Vector index configurations |
| `config` | Server configuration |
| `workers` | Background worker status |
| `stats` | Database and system statistics |

## Using with Claude Desktop

Create Claude Desktop configuration file:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

**Linux:** `~/.config/Claude/claude_desktop_config.json`

Example configuration:

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

Or use local binary:

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

Restart Claude Desktop after configuration changes.

## Using with Other MCP Clients

Run NeuronMCP interactively for testing:

```bash
./neurondb-mcp
```

Send JSON-RPC messages via stdin, receive responses via stdout.

For Docker:

```bash
docker run -i --rm \
  -e NEURONDB_HOST=localhost \
  -e NEURONDB_PORT=5432 \
  -e NEURONDB_DATABASE=neurondb \
  -e NEURONDB_USER=neurondb \
  -e NEURONDB_PASSWORD=neurondb \
  neurondb-mcp:latest
```

## Documentation

| Document | Description |
|----------|-------------|
| [Docker Guide](docker/README.md) | Container deployment guide |
| [MCP Specification](https://modelcontextprotocol.io/) | Model Context Protocol documentation |

## System Requirements

| Component | Requirement |
|-----------|-------------|
| PostgreSQL | 16 or later |
| NeuronDB Extension | Installed and enabled |
| Go | 1.23 or later (for building) |
| MCP Client | Compatible MCP client for connection |

## Integration with NeuronDB

NeuronMCP requires:

- PostgreSQL database with NeuronDB extension installed
- Database user with appropriate permissions
- Access to NeuronDB vector search, ML, and embedding functions

See [NeuronDB documentation](../NeuronDB/README.md) for installation instructions.

## Troubleshooting

### Stdio Not Working

Ensure stdin and stdout are not redirected:

```bash
./neurondb-mcp  # Correct
./neurondb-mcp > output.log  # Incorrect - breaks MCP protocol
```

For Docker, use interactive mode:

```bash
docker run -i --rm neurondb-mcp:latest
```

### Database Connection Failed

Verify connection parameters:

```bash
psql -h localhost -p 5432 -U neurondb -d neurondb -c "SELECT 1;"
```

Check environment variables:

```bash
env | grep NEURONDB
```

### MCP Client Connection Issues

Verify container is running:

```bash
docker compose ps neurondb-mcp
```

Test stdio manually:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | ./neurondb-mcp
```

Check client configuration file path and format.

### Configuration Issues

Verify config file path:

```bash
ls -la mcp-config.json
```

Check environment variable names (must start with `NEURONDB_`):

```bash
env | grep -E "^NEURONDB_"
```

## Security

- Database credentials stored securely via environment variables
- Supports TLS/SSL for encrypted database connections
- Non-root user in Docker containers
- No network endpoints (stdio only)

## Support

- **Documentation**: [Component Documentation](../README.md)
- **GitHub Issues**: [Report Issues](https://github.com/neurondb/NeurondB/issues)
- **Email**: support@neurondb.ai

## License

See [LICENSE](../LICENSE) file for license information.
