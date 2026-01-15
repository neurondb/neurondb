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

Comprehensive vector operations with extensive tooling:

#### Vector Search (8 tools)
- **Distance Metrics**: L2, cosine, inner product, L1, Hamming, Chebyshev, Minkowski
- **Index Support**: HNSW and IVF indexes with optimized search performance
- **Unified Search**: Unified vector similarity search with configurable metrics

#### Embedding Generation (7 tools)
- **Generate Embedding**: Single embedding generation with 50+ pre-configured models
- **Batch Embedding**: Batch embedding generation for efficient processing
- **Image Embedding**: Image embedding with vision models
- **Multimodal Embedding**: Multimodal embedding for text-image pairs
- **Cached Embedding**: Embedding caching for performance optimization
- **Model Configuration**: Configure, list, and manage embedding model configurations

#### Index Management (6 tools)
- **HNSW Index**: Create, tune, and manage HNSW indexes
- **IVF Index**: Create, tune, and manage IVF indexes
- **Index Status**: Monitor index status and statistics
- **Drop Index**: Remove indexes with cleanup

#### Quantization (2 tools)
- **Vector Quantization**: Quantize vectors (int8, fp16, binary, uint8, ternary, int4)
- **Quantization Analysis**: Analyze quantization impact on search quality

#### Vector Operations (3 tools)
- **Vector Arithmetic**: Vector addition, subtraction, multiplication, normalization
- **Vector Distance**: Calculate distances between vectors with multiple metrics
- **Vector Similarity**: Unified similarity calculations

#### Advanced Vector Features
- **Multi-Vector Search**: Search across multiple vector columns with fusion strategies
- **Hybrid Search**: Combine vector search with full-text search and SQL filters
- **Vector Graph Operations**: Graph algorithms on vector spaces (BFS, DFS, PageRank, community detection)
- **Vecmap Operations**: Sparse vector operations with distance and arithmetic

### ML Tools & Pipeline

Complete machine learning pipeline with 52+ algorithms:

#### Training & Prediction (6 tools)
- **Train Model**: Train models with 52+ algorithms (classification, regression, clustering, dimensionality reduction, gradient boosting, random forest, recommendation systems)
- **Predict**: Single prediction with model versioning
- **Predict Batch**: Batch prediction for efficient processing
- **Evaluate Model**: Comprehensive model evaluation with metrics (accuracy, precision, recall, F1, ROC-AUC, etc.)
- **List Models**: List all trained models with metadata
- **Get Model Info**: Detailed model information and statistics
- **Delete Model**: Remove models with cleanup
- **Export Model**: Export models for deployment

#### Advanced ML Features
- **AutoML**: Automated model selection, hyperparameter tuning, and training
- **ONNX Support**: Import, export, and inference with ONNX models (4 operations: import, export, info, predict)
- **Time Series**: ARIMA, forecasting, seasonal decomposition, and anomaly detection
- **Analytics**: Data analysis, clustering, outlier detection, drift detection, topic discovery
- **Quality Metrics**: Comprehensive quality metrics for model performance

### RAG Operations
- **Document Processing**: Chunk documents, extract metadata, and generate embeddings
- **Context Retrieval**: Semantic search with reranking for relevant context
- **Response Generation**: Generate responses with LLM integration and context injection
- **Reranking**: Multiple reranking methods (cross-encoder, LLM, Cohere, ColBERT, LTR, ensemble)
- **Hybrid Retrieval**: Combine vector search with keyword search and filters

### PostgreSQL Administration (100+ Tools)

NeuronMCP provides comprehensive PostgreSQL administration with 100+ tools covering complete database control:

#### Server Information (8 tools)
- Version, stats, databases, connections, locks, replication, settings, extensions

#### Database Object Management (8 tools)
- Tables, indexes, schemas, views, sequences, functions, triggers, constraints

#### User and Role Management (9 tools)
- Users (create, alter, drop), roles (create, alter, drop), permissions (grant, revoke, grant role, revoke role)

#### Performance and Statistics (4 tools)
- Table stats, index stats, active queries, wait events

#### Size and Storage (4 tools)
- Table size, index size, bloat analysis, vacuum stats

#### Administration (16 tools)
- Explain, explain analyze, vacuum, analyze, reindex, transactions, terminate query, kill query, set config, reload config, slow queries, cache hit ratio, buffer stats, partitions, partition stats, FDW servers, FDW tables, logical replication slots

#### Query Execution & Management (6 tools)
- Execute query, query plan, cancel query, kill query, query history, query optimization

#### Database & Schema Management (6 tools)
- Create/alter/drop database, create/alter/drop schema

#### Permission Management (4 tools)
- Grant, revoke, grant role, revoke role

#### Backup & Recovery (6 tools)
- Backup database, restore database, backup table, list backups, verify backup, backup schedule

#### Schema Modification (7 tools)
- Create/alter/drop table, create index, create view, create function, create trigger

#### Object Management (17+ tools)
- Alter index, alter view, alter function, alter trigger, drop index, drop view, drop function, drop trigger, create sequence, alter sequence, drop sequence, create type, alter type, drop type, create domain, alter domain, drop domain, and more

#### Advanced DDL Operations
- Materialized views, partitioning, foreign tables, advanced constraints

#### Migration & Schema Evolution (2 tools)
- Schema evolution tracking, migration generation and execution

#### Optimization Tools (10 tools)
- Query optimizer, performance insights, index advisor, query plan analyzer, schema evolution, migration tool, connection pool optimizer, vacuum analyzer, replication lag monitor, wait event analyzer

#### Developer Experience Tools (10+ tools)
- NL to SQL, SQL to NL, query builder, code generator, test data generator, schema visualizer, query explainer, schema documentation, migration generator, SDK generator

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

Comprehensive enterprise-grade capabilities:

#### Multi-Tenant & Governance (6 tools)
- **Multi-Tenant Management**: Complete multi-tenant isolation and management
- **Data Governance**: Data governance policies and enforcement
- **Data Lineage**: Track data lineage and dependencies
- **Compliance Reporting**: Automated compliance reporting and auditing
- **Audit Analysis**: Comprehensive audit log analysis
- **Backup Automation**: Automated backup scheduling and management

#### Performance & Optimization (5 tools)
- **Query Result Cache**: Intelligent query result caching
- **Cache Optimizer**: Cache optimization and invalidation strategies
- **Performance Benchmark**: Performance benchmarking and comparison
- **Auto-Scaling Advisor**: Auto-scaling recommendations based on usage
- **Slow Query Analyzer**: Identify and analyze slow queries

#### Monitoring & Resilience
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

### AI Intelligence Layer Tools

Advanced AI-powered tools for model management and optimization:

| Tool | Description | Status |
|------|-------------|--------|
| **AI Model Orchestration** | Orchestrate multiple AI models for complex tasks | ✅ Stable |
| **AI Cost Tracking** | Track and analyze AI model usage costs | ✅ Stable |
| **AI Embedding Quality** | Assess and optimize embedding quality | ✅ Stable |
| **AI Model Comparison** | Compare multiple models side-by-side | ✅ Stable |
| **AI RAG Evaluation** | Evaluate RAG system performance | ✅ Stable |
| **AI Embedding Drift Detection** | Detect embedding distribution drift | ✅ Stable |
| **AI Model Finetuning** | Finetune models for specific tasks | ✅ Stable |
| **AI Prompt Versioning** | Version control for prompts | ✅ Stable |
| **AI Token Optimization** | Optimize token usage for cost reduction | ✅ Stable |
| **AI Multi-Model Ensemble** | Combine multiple models for better performance | ✅ Stable |

### Analytics & Monitoring Tools

Advanced analytics and monitoring capabilities:

| Tool | Description | Status |
|------|-------------|--------|
| **Real-Time Dashboard** | Real-time monitoring dashboard | ✅ Stable |
| **Anomaly Detection** | Detect anomalies in data and usage patterns | ✅ Stable |
| **Predictive Analytics** | Predictive analytics for forecasting | ✅ Stable |
| **Cost Forecasting** | Forecast costs based on usage patterns | ✅ Stable |
| **Usage Analytics** | Comprehensive usage analytics and reporting | ✅ Stable |
| **Alert Manager** | Centralized alert management and routing | ✅ Stable |

### Plugin System

Extensible plugin architecture for custom functionality:

| Feature | Description | Status |
|--------|-------------|--------|
| **Plugin Marketplace** | Discover and install plugins from marketplace | ✅ Stable |
| **Plugin Hot Reload** | Hot reload plugins without server restart | ✅ Stable |
| **Plugin Versioning** | Version control for plugins | ✅ Stable |
| **Plugin Sandbox** | Secure sandboxed plugin execution | ✅ Stable |
| **Plugin Testing** | Testing framework for plugins | ✅ Stable |
| **Plugin Builder** | Tools for building custom plugins | ✅ Stable |

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
- **Setup Guide**: `NeuronMCP/docs/neurondb-mcp-setup.md`
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
