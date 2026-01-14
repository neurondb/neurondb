# 🔌 NeuronMCP

<div align="center">

**Model Context Protocol (MCP) server with 100+ tools for NeuronDB**

[![Status](https://img.shields.io/badge/status-stable-brightgreen)](.)
[![Tools](https://img.shields.io/badge/tools-100+-green)](.)
[![Protocol](https://img.shields.io/badge/protocol-MCP-blue)](.)

</div>

---

> [!TIP]
> NeuronMCP provides a complete MCP protocol implementation. It includes 100+ tools for vector operations, ML, RAG, and PostgreSQL administration.

---

## 📋 What It Is

NeuronMCP is a Model Context Protocol (MCP) server providing comprehensive tools and resources for MCP-compatible clients to interact with NeuronDB.

| Feature | Description | Status |
|---------|-------------|--------|
| **MCP Protocol Server** | Full JSON-RPC 2.0 implementation with stdio, HTTP, and SSE transport | ✅ Stable |
| **Tool Server** | 100+ tools covering vector operations, ML, RAG, PostgreSQL administration, and dataset loading | ✅ Stable |
| **Resource Provider** | Schema, models, indexes, config, workers, and stats with real-time subscriptions | ✅ Stable |
| **Enterprise Platform** | Middleware system, authentication, caching, metrics, webhooks, and resilience features | ✅ Stable |

## Key Features & Modules

### MCP Protocol Implementation
- **JSON-RPC 2.0**: Full protocol implementation with stdio, HTTP, and SSE transport modes
- **Batch Operations**: Transactional batch tool calls (tools/call_batch) for efficient bulk operations
- **Progress Tracking**: Long-running operation progress with progress/get for monitoring
- **Tool Discovery**: Search and filter tools with categorization and metadata
- **Prompts Protocol**: Full prompts/list and prompts/get with template engine support
- **Sampling/Completions**: sampling/createMessage with streaming support for LLM interactions

### Vector Operations (50+ Tools)
- **Vector Search**: Multiple distance metrics (L2, cosine, inner product) with HNSW and IVF indexes
- **Embedding Generation**: Generate embeddings with 50+ pre-configured models (OpenAI, HuggingFace, local)
- **Index Management**: Create, tune, and manage HNSW and IVF indexes with optimization
- **Quantization**: Support for int8, fp16, binary, uint8, ternary, and int4 quantization
- **Vector Arithmetic**: Vector addition, subtraction, multiplication, and normalization
- **Distance Metrics**: 7+ distance metrics including L2, cosine, inner product, Hamming, Jaccard
- **Multi-Vector Search**: Search across multiple vector columns with fusion
- **Hybrid Search**: Combine vector search with full-text search and SQL filters

### ML Tools & Pipeline
- **Training**: 52+ ML algorithms (classification, regression, clustering, dimensionality reduction)
- **Prediction**: Batch and single prediction with model versioning
- **Evaluation**: Comprehensive model evaluation with metrics (accuracy, precision, recall, F1, etc.)
- **AutoML**: Automated model selection, hyperparameter tuning, and training
- **ONNX Support**: Import, export, and inference with ONNX models
- **Time Series**: ARIMA, forecasting, seasonal decomposition, and anomaly detection
- **Analytics**: Data analysis, clustering, outlier detection, drift detection, topic discovery

### RAG Operations
- **Document Processing**: Chunk documents, extract metadata, and generate embeddings
- **Context Retrieval**: Semantic search with reranking for relevant context
- **Response Generation**: Generate responses with LLM integration and context injection
- **Reranking**: Multiple reranking methods (cross-encoder, LLM, Cohere, ColBERT, LTR, ensemble)
- **Hybrid Retrieval**: Combine vector search with keyword search and filters

### PostgreSQL Administration (27 Tools)
- **Database Management**: Version, stats, databases, connections, replication
- **Schema Management**: Tables, indexes, schemas, views, sequences, functions, triggers, constraints
- **User Management**: Users, roles, permissions, and access control
- **Monitoring**: Active queries, wait events, table/index stats, sizes, bloat, vacuum stats
- **Performance**: Connection pooling, query optimization, and performance metrics

### Dataset Loading
- **HuggingFace**: Load datasets directly from HuggingFace with automatic schema detection
- **URL Sources**: Load CSV, JSON, JSONL, Parquet files from URLs
- **GitHub**: Load datasets from GitHub repositories
- **S3**: Load datasets from AWS S3 buckets
- **Local Files**: Load from local filesystem with path support
- **Auto-Embedding**: Automatic embedding generation for text columns
- **Index Creation**: Automatic HNSW and GIN index creation for vectors and full-text search

### Middleware System
- **Validation**: Request validation with JSON Schema and parameter checking
- **Logging**: Structured logging with configurable levels and formats
- **Timeout Handling**: Configurable timeouts for tool execution and requests
- **Error Handling**: Comprehensive error handling with detailed error messages
- **Authentication**: JWT, API keys, and OAuth2 authentication support
- **Rate Limiting**: Per-key rate limiting with configurable quotas and windows

### Enterprise Features
- **Prometheus Metrics**: Comprehensive metrics export for monitoring and alerting
- **Webhooks**: Outbound webhook notifications for events and tool executions
- **Circuit Breaker**: Resilience patterns with circuit breaker for fault tolerance
- **Caching Layer**: TTL-based caching with idempotency support for performance
- **Connection Pooling**: Efficient database connection pooling with health checks
- **Health Checks**: Database, tools, and resource availability monitoring

### Resources & Subscriptions
- **Schema Resources**: Database schema information with real-time updates
- **Model Resources**: ML model information and status
- **Index Resources**: Vector index configurations and statistics
- **Config Resources**: Server configuration and settings
- **Worker Resources**: Background worker status and metrics
- **Stats Resources**: Database and system statistics with subscriptions

### Security & Authentication
- **JWT Authentication**: JSON Web Token support for stateless authentication
- **API Keys**: API key-based authentication with secure storage
- **OAuth2**: OAuth2 integration for third-party authentication
- **Rate Limiting**: Per-key rate limiting with configurable quotas
- **Request Validation**: Input validation and sanitization
- **Secure Credentials**: Encrypted storage for API keys and credentials

### Configuration & Management
- **JSON Configuration**: Flexible JSON config files with environment variable overrides
- **Environment Variables**: Comprehensive environment variable support
- **Feature Flags**: Enable/disable features via configuration
- **Tool Configuration**: Per-tool configuration with defaults and overrides
- **Model Configuration**: 50+ pre-configured LLM models with encrypted API key storage

### Modular Architecture
- **19 Independent Packages**: Clean separation of concerns with modular design
- **Tool Registry**: Extensible tool registration system
- **Resource Manager**: Centralized resource management
- **Middleware Pipeline**: Pluggable middleware architecture
- **Protocol Handlers**: Separate handlers for different MCP protocol methods

## Documentation

- **Main README**: `NeuronMCP/README.md`
- **Tools Reference**: `NeuronMCP/REGISTERED_TOOLS.md`
- **PostgreSQL Tools**: `NeuronMCP/POSTGRESQL_TOOLS.md`
- **Setup Guide**: `NeuronMCP/docs/neurondb_mcp_setup.md`
- **Tool Catalog**: `NeuronMCP/docs/tool-resource-catalog.md`
- **Examples**: `NeuronMCP/docs/examples/`
- **Official Docs**: [https://www.neurondb.ai/docs/neuronmcp](https://www.neurondb.ai/docs/neuronmcp)

## Docker

- Compose service: `neuronmcp` (plus GPU-profile variants)
- From repo root: `docker compose up -d neuronmcp`
- See: `NeuronMCP/docker/README.md`

## Quick Start

### Minimal Verification

```bash
# Test MCP server (requires MCP client)
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | ./neurondb-mcp
```

### Using with Claude Desktop

Create Claude Desktop configuration file:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

**Linux:** `~/.config/Claude/claude_desktop_config.json`

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

For complete setup instructions, see `NeuronMCP/README.md`.
