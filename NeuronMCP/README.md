# NeuronMCP

[![Go](https://img.shields.io/badge/Go-1.23+-00ADD8.svg)](https://golang.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-blue.svg)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](../LICENSE)
[![MCP](https://img.shields.io/badge/MCP-Protocol-blue.svg)](https://modelcontextprotocol.io/)

Model Context Protocol server for NeuronDB PostgreSQL extension, implemented in Go. Enables MCP-compatible clients to access NeuronDB vector search, ML algorithms, and RAG capabilities.

## Overview

NeuronMCP implements the Model Context Protocol using JSON-RPC 2.0 over stdio. It provides tools and resources for MCP clients to interact with NeuronDB, including vector operations, ML model training, and database schema management.

## Documentation

- **[Features](docs/features.md)** - Complete feature list and capabilities
- **[Tool & Resource Catalog](docs/tool-resource-catalog.md)** - Complete catalog of all tools and resources
- **[Setup Guide](setup_guide.md)** - Setup and configuration guide

## Tool Registration Modes

### Claude Desktop Compatibility

**Important**: Claude Desktop has compatibility issues with tools that have `neurondb_` prefixes in their names. By default, only PostgreSQL tools are registered for maximum compatibility.

#### Default Mode (PostgreSQL-only)
```json
{
  "mcpServers": {
    "neurondb_postgresql_mcp": {
      "command": "/path/to/neuronmcp",
      "env": {
        "NEURONDB_HOST": "localhost",
        "NEURONDB_PORT": "5432",
        "NEURONDB_DATABASE": "neurondb",
        "NEURONDB_USER": "pgedge"
      }
    }
  }
}
```
**Tools available**: 5 essential PostgreSQL tools (version, execute_query, tables, query_plan, cancel_query)

**Note**: Claude Desktop has a hard limit of 5 tools per MCP server. Additional tools can be enabled using category-based selection.

#### Enable NeuronDB Tools (Advanced Mode)
⚠️ **Warning**: `neurondb_` prefixed tools will NOT display in Claude Desktop, but work with other MCP clients.

```json
{
  "mcpServers": {
    "neurondb_postgresql_mcp": {
      "command": "/path/to/neuronmcp",
      "env": {
        "NEURONDB_HOST": "localhost",
        "NEURONDB_PORT": "5432",
        "NEURONDB_DATABASE": "neurondb",
        "NEURONDB_USER": "pgedge",
        "NEURONMCP_ALLOW_NEURONDB_TOOLS": "true"
      }
    }
  }
}
```
**Tools available**: 6 tools including vector and RAG tools

#### Category-Based Selection
```json
{
  "mcpServers": {
    "neurondb_postgresql_mcp": {
      "command": "/path/to/neuronmcp",
      "env": {
        "NEURONDB_HOST": "localhost",
        "NEURONDB_PORT": "5432",
        "NEURONDB_DATABASE": "neurondb",
        "NEURONDB_USER": "pgedge",
        "NEURONMCP_TOOL_CATEGORIES": "postgresql,vector"
      }
    }
  }
}
```

## Official Documentation

**For comprehensive documentation, detailed tutorials, complete tool references, and integration guides, visit:**

🌐 **[https://www.neurondb.ai/docs/neuronmcp](https://www.neurondb.ai/docs/neuronmcp)**

The official documentation provides:
- Complete MCP protocol implementation details
- All available tools and resources reference
- Claude Desktop integration guide
- Custom tool development
- Configuration and deployment guides
- Troubleshooting and best practices

## Features

| Feature | Description |
|---------|-------------|
| **MCP Protocol** | Full JSON-RPC 2.0 implementation with stdio, HTTP, and SSE transport |
| **Vector Operations** | 50+ tools for vector search (L2, cosine, inner product), embedding generation, indexing (HNSW, IVF), quantization (int8, fp16, binary, uint8, ternary, int4), and 7+ distance metrics |
| **ML Tools** | Complete ML pipeline: training (52+ algorithms), prediction, evaluation, AutoML, ONNX model support, time series analysis, and analytics |
| **RAG Operations** | Document processing, context retrieval, response generation with multiple reranking methods (cross-encoder, LLM, Cohere, ColBERT, LTR, ensemble) |
| **PostgreSQL Tools** | 100+ comprehensive PostgreSQL tools providing complete database control: DDL (CREATE/ALTER/DROP for databases, schemas, tables, indexes, views, functions, triggers, sequences, types, domains), DML (INSERT, UPDATE, DELETE, TRUNCATE, COPY), DCL (GRANT/REVOKE), user/role management, backup/restore, materialized views, partitioning, foreign tables, security validation, and full administration capabilities |
| **Dataset Loading** | Load datasets from HuggingFace, URLs, GitHub, S3, and local files with automatic schema detection, embedding generation, and index creation |
| **Resources** | Schema, models, indexes, config, workers, stats with real-time subscriptions |
| **Prompts Protocol** | Full prompts/list and prompts/get with template engine |
| **Sampling/Completions** | sampling/createMessage with streaming support |
| **Progress Tracking** | Long-running operation progress with progress/get |
| **Batch Operations** | Transactional batch tool calls (tools/call_batch) |
| **Tool Discovery** | Search and filter tools with categorization |
| **Middleware System** | Pluggable middleware pipeline: request validation, structured logging, configurable timeouts, comprehensive error handling, authentication (JWT, API keys, OAuth2), and rate limiting with per-key quotas |
| **Security** | Multiple authentication methods (JWT, API keys, OAuth2), rate limiting, request validation, and secure credential storage |
| **Performance** | TTL-based caching layer with idempotency support, connection pooling, and optimized query execution |
| **Enterprise Features** | Prometheus metrics export, webhook notifications, circuit breaker for resilience, retry mechanisms, health checks, and comprehensive monitoring |
| **Health Checks** | Database, tools, and resource availability monitoring |
| **Configuration** | JSON config files with environment variable overrides |
| **Modular Architecture** | 19 independent packages with clean separation of concerns |

> 📊 **See [COMPARISON.md](COMPARISON.md) for a detailed comparison with other MCP servers**

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

**Option 1: Using Docker Compose (Recommended for Quick Start)**

If using the root `docker-compose.yml`:
```bash
# From repository root
docker compose up -d neurondb

# Wait for service to be healthy
docker compose ps neurondb

# Create extension (if not already created)
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" -c "CREATE EXTENSION IF NOT EXISTS neurondb;"
```

**Option 2: Native PostgreSQL Installation**

```bash
createdb neurondb
psql -d neurondb -c "CREATE EXTENSION neurondb;"
```

### NeuronMCP Configuration Schema Setup

NeuronMCP requires a comprehensive database schema for managing LLM models, API keys, index configurations, worker settings, ML defaults, and tool configurations. This schema provides:

- **50+ pre-populated LLM models** (OpenAI, Anthropic, HuggingFace, local) with encrypted API key storage
- **Index templates** for HNSW and IVF vector indexes
- **Worker configurations** for background workers
- **ML algorithm defaults** for all supported algorithms
- **Tool-specific defaults** for all NeuronMCP tools
- **System-wide settings** and feature flags

**Quick Setup:**

```bash
cd NeuronMCP
./scripts/neuronmcp-setup.sh
```

**Set API Keys:**

```sql
-- Set API key for a model
SELECT neurondb_set_model_key('text-embedding-3-small', 'sk-your-api-key');

-- View configured models
SELECT * FROM neurondb.v_llm_models_ready;
```

**For complete documentation**, see [neurondb-mcp-setup.md](docs/neurondb-mcp-setup.md)

### Configuration

Create `mcp-config.json`:

```json
{
  "database": {
    "host": "localhost",
    "port": 5433,
    "database": "neurondb",
    "user": "neurondb",
    "password": "neurondb"
  },
  "server": {
    "name": "neurondb-mcp-server",
    "version": "2.0.0"
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

#### Automated Installation (Recommended)

Use the installation script for easy setup:

```bash
# From repository root
sudo ./scripts/install-neuronmcp.sh

# With system service enabled
sudo ./scripts/install-neuronmcp.sh --enable-service
```

#### Manual Build

From source:

```bash
go build ./cmd/neurondb-mcp
./neurondb-mcp
```

#### Using Docker

```bash
cd docker
# Optionally create .env file with your configuration
# Or use environment variables directly (docker-compose.yml has defaults)
docker compose up -d
```

See [Docker Guide](docker/readme.md) for Docker deployment details.

#### Running as a Service

For systemd (Linux) or launchd (macOS), see [Service Management Guide](../../Docs/getting-started/installation-services.md).

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

NeuronMCP provides comprehensive tools covering all NeuronDB capabilities:

| Tool Category | Tools |
|---------------|-------|
| **Vector Operations** | `vector_search`, `vector_search_l2`, `vector_search_cosine`, `vector_search_inner_product`, `vector_similarity`, `vector_arithmetic`, `vector_distance`, `vector_similarity_unified` |
| **Vector Quantization** | `vector_quantize`, `quantization_analyze` (int8, fp16, binary, uint8, ternary, int4) |
| **Embeddings** | `generate_embedding`, `batch_embedding`, `embed_image`, `embed_multimodal`, `embed_cached`, `configure_embedding_model`, `get_embedding_model_config`, `list_embedding_model_configs`, `delete_embedding_model_config` |
| **Hybrid Search** | `hybrid_search`, `reciprocal_rank_fusion`, `semantic_keyword_search`, `multi_vector_search`, `faceted_vector_search`, `temporal_vector_search`, `diverse_vector_search` |
| **Reranking** | `rerank_cross_encoder`, `rerank_llm`, `rerank_cohere`, `rerank_colbert`, `rerank_ltr`, `rerank_ensemble` |
| **ML Operations** | `train_model`, `predict`, `predict_batch`, `evaluate_model`, `list_models`, `get_model_info`, `delete_model`, `export_model` |
| **Analytics** | `analyze_data`, `cluster_data`, `reduce_dimensionality`, `detect_outliers`, `quality_metrics`, `detect_drift`, `topic_discovery` |
| **Time Series** | `timeseries_analysis` (ARIMA, forecasting, seasonal decomposition) |
| **AutoML** | `automl` (model selection, hyperparameter tuning, auto training) |
| **ONNX** | `onnx_model` (import, export, info, predict) |
| **Index Management** | `create_hnsw_index`, `create_ivf_index`, `index_status`, `drop_index`, `tune_hnsw_index`, `tune_ivf_index` |
| **RAG Operations** | `process_document`, `retrieve_context`, `generate_response`, `chunk_document` |
| **Workers & GPU** | `worker_management`, `gpu_info` |
| **Vector Graph** | `vector_graph` (BFS, DFS, PageRank, community detection) |
| **Vecmap Operations** | `vecmap_operations` (distances, arithmetic, norm on sparse vectors) |
| **Dataset Loading** | `load_dataset` (HuggingFace, URLs, GitHub, S3, local files with auto-embedding) |
| **PostgreSQL (100+ tools)** | Complete PostgreSQL control: **DDL** (CREATE/ALTER/DROP for databases, schemas, tables, indexes, views, functions, triggers, sequences, types, domains, materialized views, partitions, foreign tables), **DML** (INSERT, UPDATE, DELETE, TRUNCATE, COPY), **DCL** (GRANT/REVOKE), **User/Role Management** (CREATE/ALTER/DROP USER/ROLE), **Backup/Recovery** (pg_dump/pg_restore), **Security** (SQL validation, permission checking, audit), plus all administration, monitoring, and statistics tools |

**Comprehensive Documentation:**
- **[TOOLS_REFERENCE.md](TOOLS_REFERENCE.md)** - Complete reference for all 100+ tools with parameters, examples, and error codes
- **[POSTGRESQL_TOOLS.md](POSTGRESQL_TOOLS.md)** - Detailed documentation for all PostgreSQL tools (100+ tools covering DDL, DML, DCL, administration, backup, security)

For a comprehensive catalog of all tools and resources, see [docs/tool-resource-catalog.md](docs/tool-resource-catalog.md).

For example client usage and interaction transcripts, see [docs/examples/](docs/examples/).

### Dataset Loading Examples

The `load_dataset` tool supports multiple data sources with automatic schema detection, embedding generation, and index creation:

#### HuggingFace Datasets

```json
{
  "name": "load_dataset",
  "arguments": {
    "source_type": "huggingface",
    "source_path": "sentence-transformers/embedding-training-data",
    "split": "train",
    "limit": 10000,
    "auto_embed": true,
    "embedding_model": "default"
  }
}
```

#### URL Datasets (CSV, JSON, Parquet)

```json
{
  "name": "load_dataset",
  "arguments": {
    "source_type": "url",
    "source_path": "https://example.com/data.csv",
    "format": "csv",
    "auto_embed": true,
    "create_indexes": true
  }
}
```

#### GitHub Repositories

```json
{
  "name": "load_dataset",
  "arguments": {
    "source_type": "github",
    "source_path": "owner/repo/path/to/data.json",
    "auto_embed": true
  }
}
```

#### S3 Buckets

```json
{
  "name": "load_dataset",
  "arguments": {
    "source_type": "s3",
    "source_path": "s3://my-bucket/data.parquet",
    "auto_embed": true
  }
}
```

#### Local Files

```json
{
  "name": "load_dataset",
  "arguments": {
    "source_type": "local",
    "source_path": "/path/to/local/file.jsonl",
    "schema_name": "my_schema",
    "table_name": "my_table",
    "auto_embed": true
  }
}
```

**Key Features:**
- **Automatic Schema Detection**: Analyzes data types and creates optimized PostgreSQL tables
- **Auto-Embedding**: Automatically detects text columns and generates vector embeddings using NeuronDB
- **Index Creation**: Creates HNSW indexes for vectors, GIN indexes for full-text search
- **Batch Loading**: Efficient bulk loading with progress tracking
- **Multiple Formats**: Supports CSV, JSON, JSONL, Parquet, and HuggingFace datasets

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

NeuronMCP is fully compatible with Claude Desktop on macOS, Windows, and Linux.

Create Claude Desktop configuration file:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

**Linux:** `~/.config/Claude/claude_desktop_config.json`

See the example configuration files in this directory (`claude_desktop_config.*.json`) for platform-specific examples.

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

### Using neurondb-mcp-client

A simple MCP client that works exactly like Claude Desktop. It handles the full MCP protocol including initialize handshake.

Build the client:

```bash
make build-client
```

Usage:

```bash
# Initialize and list tools
./bin/neurondb-mcp-client ./bin/neurondb-mcp tools/list

# Call a tool
./bin/neurondb-mcp-client ./bin/neurondb-mcp tools/call '{"name":"vector_search","arguments":{}}'

# List resources
./bin/neurondb-mcp-client ./bin/neurondb-mcp resources/list
```

The client automatically:
- Sends initialize request with proper headers (exactly like Claude Desktop)
- Reads initialize response
- Reads initialized notification
- Then sends your requests and reads responses

Test script:

```bash
cd client
./example_usage.sh
```

Or use the Python client:

```bash
cd client
python neurondb_mcp_client.py -c ../../neuronmcp_server.json -e "list_tools"
```

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
| [Docker Guide](docker/readme.md) | Container deployment guide |
| [MCP Specification](https://modelcontextprotocol.io/) | Model Context Protocol documentation |
| [Claude Desktop Config Examples](claude_desktop_config.json) | Example configurations for macOS, Linux, and Windows |

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

See [NeuronDB documentation](../NeuronDB/readme.md) for installation instructions.

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

- **Documentation**: [Component Documentation](../readme.md)
- **GitHub Issues**: [Report Issues](https://github.com/neurondb/NeurondB/issues)
- **Email**: support@neurondb.ai

## License

See [LICENSE](../LICENSE) file for license information.
