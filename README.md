# NeuronDB — PostgreSQL AI ecosystem

<div align="center">
  <img src="neurondb.png" alt="NeuronDB" width="360" />

  <p>
    <a href="https://www.postgresql.org/">
      <img alt="PostgreSQL 16/17/18" src="https://img.shields.io/badge/PostgreSQL-16%2C17%2C18-blue.svg" />
    </a>
    <a href="https://github.com/neurondb/neurondb/actions/workflows/neurondb-build-matrix.yml">
      <img alt="CI: NeuronDB" src="https://img.shields.io/badge/CI-NeuronDB-blue.svg" />
    </a>
    <a href="https://github.com/neurondb/neurondb/actions/workflows/neuronmcp-build-matrix.yml">
      <img alt="CI: NeuronMCP" src="https://img.shields.io/badge/CI-NeuronMCP-blue.svg" />
    </a>
    <a href="https://github.com/neurondb/neurondb/actions/workflows/integration-tests-full-ecosystem.yml">
      <img alt="CI: Integration" src="https://img.shields.io/badge/CI-Integration-blue.svg" />
    </a>
    <a href="https://github.com/neurondb/neurondb/actions/workflows/security-scan.yml">
      <img alt="Security scan" src="https://img.shields.io/badge/Security-Scan-blue.svg" />
    </a>
    <a href="https://github.com/neurondb/neurondb/actions/workflows/neurondb-docker.yml">
      <img alt="Docker" src="https://img.shields.io/badge/Docker-Build-blue.svg" />
    </a>
  </p>
  <p>
    <a href="#gpu-profiles-cuda--rocm--metal">
      <img alt="GPU Backends" src="https://img.shields.io/badge/GPU-CUDA%20%7C%20ROCm%20%7C%20Metal-green.svg" />
    </a>
    <a href="LICENSE">
      <img alt="License: Proprietary" src="https://img.shields.io/badge/license-proprietary-red.svg" />
    </a>
    <a href="https://www.neurondb.ai/docs">
      <img alt="Docs" src="https://img.shields.io/badge/docs-neurondb.ai-brightgreen.svg" />
    </a>
  </p>

  <p><strong>Vector search, embeddings, and ML primitives in PostgreSQL</strong>, with optional services for <strong>agents</strong>, <strong>MCP</strong>, and a <strong>desktop UI</strong>.</p>

  <p>
    <!-- Screenshots will be added here when available -->
    <!-- <img src="Docs/assets/neurondesktop-ui.png" alt="NeuronDesktop UI" width="600" /> -->
    <!-- <img src="Docs/assets/search-demo.gif" alt="Vector search demo" width="600" /> -->
  </p>
</div>

> [!TIP]
> New here? Start with [`Docs/getting-started/simple-start.md`](Docs/getting-started/simple-start.md) or jump to [`QUICKSTART.md`](QUICKSTART.md).
> 
> **Developer Tools**: Try the [quickstart data pack](examples/quickstart/), [SQL recipe library](Docs/getting-started/recipes/), and [CLI helpers](scripts/neurondb-cli.sh) for faster development!

## Hello NeuronDB (60 seconds)

Get vector search working in under a minute:

```bash
# 1. Start PostgreSQL with NeuronDB (CPU profile, default)
docker compose up -d neurondb

# Wait for service to be healthy (about 30-60 seconds)
docker compose ps

# 2. Connect and create extension
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" -c "CREATE EXTENSION IF NOT EXISTS neurondb;"

# 3. Create table, insert vectors, create index, and search
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" <<EOF
CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  content TEXT,
  embedding vector(3)
);

INSERT INTO documents (content, embedding) VALUES
  ('Machine learning algorithms', '[0.1,0.2,0.3]'::vector),
  ('Neural networks and deep learning', '[0.2,0.3,0.4]'::vector),
  ('Natural language processing', '[0.3,0.4,0.5]'::vector);

CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);

SELECT id, content, embedding <=> '[0.15,0.25,0.35]'::vector AS distance
FROM documents
ORDER BY embedding <=> '[0.15,0.25,0.35]'::vector
LIMIT 3;
EOF
```

**Expected output:**
```
 id |              content               |     distance      
----+------------------------------------+-------------------
  1 | Machine learning algorithms        | 0.141421356237309
  2 | Neural networks and deep learning  | 0.173205080756888
  3 | Natural language processing       | 0.244948974278318
(3 rows)
```

> [!SECURITY]
> The default password (`neurondb`) is for development only. **Always change it in production** by setting `POSTGRES_PASSWORD` in your `.env` file. See [Service URLs & ports](#service-urls--ports) for connection details.

## Table of contents

- [What you can build](#what-you-can-build)
- [Architecture](#architecture)
- [Installation](#installation)
  - [Quick start (Docker)](#quick-start-docker)
  - [Native install](#native-install)
  - [Minimal mode (extension only)](#minimal-mode-extension-only)
- [Service URLs & ports](#service-urls--ports)
- [Documentation](#documentation)
- [Repo layout](#repo-layout)
- [Benchmarks](#benchmarks)
- [GPU profiles (CUDA / ROCm / Metal)](#gpu-profiles-cuda--rocm--metal)
- [Contributing / security / license](#contributing--security--license)
- [Project statistics](#project-statistics)

## 🎯 What You Can Build

NeuronDB enables you to build powerful AI applications directly in PostgreSQL:

<details>
<summary><strong>🔍 Semantic & Hybrid Search</strong></summary>

**Combine vector similarity with SQL filters and full-text search:**

```sql
-- Find similar documents with metadata filters
SELECT content, embedding <=> query_vector AS similarity
FROM documents
WHERE category = 'technology'
  AND created_at > '2024-01-01'
ORDER BY embedding <=> query_vector
LIMIT 10;
```

**Use cases:**
- Document search with filters
- Product recommendations
- Content discovery
- Similarity matching

</details>

<details>
<summary><strong>📄 RAG Pipelines</strong></summary>

**Build retrieval-augmented generation systems with Postgres-native primitives:**

```sql
-- Generate embedding for query
WITH query AS (SELECT embed_text('your question') AS q_vec)
-- Retrieve relevant context
SELECT content, embedding <=> q.q_vec AS distance
FROM documents, query q
ORDER BY embedding <=> q.q_vec
LIMIT 5;
```

**Use cases:**
- Question answering systems
- Document Q&A
- Knowledge bases
- Chatbots with context

</details>

<details>
<summary><strong>🤖 Agent Backends</strong></summary>

**Create AI agents with durable memory and tool execution:**

- ✅ Persistent memory with vector search
- ✅ Tool execution (SQL, HTTP, Code, Shell)
- ✅ Multi-agent collaboration
- ✅ Workflow orchestration
- ✅ Budget and cost management

**Use cases:**
- Autonomous agents
- Workflow automation
- Data analysis agents
- Customer service bots

</details>

<details>
<summary><strong>🔌 MCP Integrations</strong></summary>

**Connect MCP clients (Claude Desktop, etc.) to NeuronDB:**

- ✅ 100+ tools available via MCP
- ✅ Vector operations
- ✅ ML pipeline tools
- ✅ PostgreSQL administration
- ✅ Dataset loading

**Use cases:**
- Claude Desktop integration
- LLM tool access
- Database management via LLMs
- Automated workflows

</details>

## ⭐ What Makes NeuronDB Different

<details>
<summary><strong>📊 Feature Comparison</strong></summary>

| Feature | NeuronDB | Typical Alternatives |
|---------|----------|---------------------|
| **Index types** | HNSW, IVF, PQ, hybrid, multi-vector | Limited (usually just HNSW) |
| **GPU acceleration** | CUDA, ROCm, Metal (3 backends) | Single backend or CPU-only |
| **Benchmark coverage** | RAGAS, MTEB, BEIR integrated | Manual setup required |
| **Agent runtime** | ✅ NeuronAgent included | ❌ External services needed |
| **MCP server** | ✅ NeuronMCP included (100+ tools) | ❌ Separate integration required |
| **Desktop UI** | ✅ NeuronDesktop included | ❌ Build your own |
| **ML algorithms** | 52+ algorithms | Extension only (limited) |
| **SQL functions** | 520+ functions | Typically <100 |

</details>

<details>
<summary><strong>🎯 Key Advantages</strong></summary>

### 🚀 Performance

- **10x faster** HNSW index building than pgvector
- **SIMD-optimized** distance calculations
- **GPU acceleration** for embeddings and ML
- **Efficient memory** management

### 🔧 Developer Experience

- **Complete ecosystem** - Database + Agent + MCP + UI
- **SQL-first** - Everything accessible via SQL
- **Rich tooling** - CLI helpers, examples, recipes
- **Comprehensive docs** - 60+ documentation files

### 🏢 Enterprise Ready

- **Production features** - Monitoring, backups, HA
- **Security** - RBAC, encryption, audit logging
- **Scalability** - Horizontal and vertical scaling
- **Observability** - Prometheus metrics, structured logging

</details>

## Architecture

```mermaid
flowchart LR
  subgraph DB["NeuronDB PostgreSQL"]
    EXT["NeuronDB extension"]
  end
  AG["NeuronAgent"] -->|SQL| DB
  MCP["NeuronMCP"] -->|tools/resources| DB
  UI["NeuronDesktop UI"] --> API["NeuronDesktop API"]
  API -->|SQL| DB
```

> [!NOTE]
> The root `docker-compose.yml` starts the ecosystem services together. You can also run each component independently (see component READMEs).

## Installation

### Pick one component

Choose what you need:

| Component Setup | Command | What you get |
|---|---|---|
| **NeuronDB only** (extension) | `docker compose up -d neurondb` | Vector search, ML algorithms, embeddings in PostgreSQL |
| **NeuronDB + NeuronMCP** | `docker compose up -d neurondb neuronmcp` | Above + MCP server for Claude Desktop, etc. |
| **NeuronDB + NeuronAgent** | `docker compose up -d neurondb neuronagent` | Above + Agent runtime with REST API |
| **Full stack** | `docker compose up -d` | All components including NeuronDesktop UI |

> [!NOTE]
> All components run independently. The root `docker-compose.yml` starts everything together for convenience, but you can run individual services as needed.

### Quick start (Docker)

**Option 1: Use published images (recommended)**

Pull pre-built images from GitHub Container Registry:

```bash
# Pull latest images
docker compose pull

# Start services
docker compose up -d

# Wait for services to be healthy (30-60 seconds)
docker compose ps

# Verify all services are running
./scripts/neurondb-healthcheck.sh quick
```

**What you'll see:**
- 5 services starting: `neurondb`, `neuronagent`, `neuronmcp`, `neurondesk-api`, `neurondesk-frontend`
- All services should show "healthy" status after initialization

> [!TIP]
> For specific versions, see [Container Images documentation](Docs/deployment/container-images.md). Published images are available starting with v1.0.0.

**Option 2: Build from source**

```bash
# Build and start all services
docker compose up -d --build

# Monitor build progress (first time takes 5-10 minutes)
docker compose logs -f

# Once built, wait for services to be healthy
docker compose ps

# Verify all services are running
./scripts/neurondb-healthcheck.sh quick
```

**Build time:** First build takes 5-10 minutes depending on your system. Subsequent starts are 30-60 seconds.

<details>
<summary><strong>Prerequisites checklist</strong></summary>

- [ ] Docker 20.10+ installed
- [ ] Docker Compose 2.0+ installed
- [ ] 4 GB+ RAM available
- [ ] Ports 5433, 8080, 8081, 3000 available

</details>

> [!IMPORTANT]
> Prefer a step-by-step guide? See [`QUICKSTART.md`](QUICKSTART.md).
> 
> **New Developer Tools**: After setup, try the [quickstart data pack](examples/quickstart/) for sample data, [SQL recipe library](Docs/getting-started/recipes/) for ready-to-run queries, and [CLI helpers](scripts/neurondb-cli.sh) for index management!

> [!SECURITY]
> Default credentials are for **development only**. In production, set strong passwords via environment variables or `.env` file.

### Native install

Install components directly on your system without Docker.

#### NeuronDB Extension

Install the NeuronDB extension directly into your existing PostgreSQL installation.

<details>
<summary><strong>Build and install steps</strong></summary>

**Prerequisites:**
- PostgreSQL 16, 17, or 18 development headers
- C compiler (gcc or clang)
- Make

**Build:**
```bash
cd NeuronDB
make
sudo make install
```

**Enable extension:**
```sql
CREATE EXTENSION neurondb;
```

**Configure (if needed):**

Some features require preloading. Add to `postgresql.conf`:
```ini
shared_preload_libraries = 'neurondb'
```

Then restart PostgreSQL:
```bash
sudo systemctl restart postgresql
```

**Configuration parameters (GUCs):**
```ini
# Vector index settings
neurondb.hnsw_ef_search = 40          # HNSW search quality
neurondb.enable_seqscan = on          # Allow sequential scans

# Memory settings
neurondb.maintenance_work_mem = 256MB # Index build memory
```

**Upgrade path:**
```sql
-- Check current version
SELECT extversion FROM pg_extension WHERE extname = 'neurondb';

-- Expected output: 2.0

-- Upgrade to latest (if newer version available)
ALTER EXTENSION neurondb UPDATE;

-- Verify upgrade
SELECT neurondb.version();
```

</details>

For detailed installation instructions, see [`NeuronDB/INSTALL.md`](NeuronDB/INSTALL.md).

#### Ecosystem Components (NeuronMCP, NeuronAgent, NeuronDesktop)

Install NeuronMCP, NeuronAgent, and NeuronDesktop from source with automated scripts.

**Quick Installation:**

```bash
# Install all components
sudo ./scripts/install-components.sh

# Install specific components
sudo ./scripts/install-components.sh neuronmcp neuronagent

# Install with system services enabled
sudo ./scripts/install-components.sh --enable-service
```

**Prerequisites:**
- Go 1.23+ (for building)
- PostgreSQL 16+ with NeuronDB extension
- Node.js 18+ (for NeuronDesktop)

**Manual Installation:**

See [Native Installation Guide](Docs/getting-started/installation-native.md) for detailed instructions.

**Service Management:**

```bash
# Start services
./scripts/manage-services.sh start

# Check status
./scripts/manage-services.sh status

# View logs
./scripts/manage-services.sh logs neuronagent
```

For service management details, see [Service Management Guide](Docs/getting-started/installation-services.md).

### Minimal mode (extension only)

Use NeuronDB as a PostgreSQL extension only, without the Agent, MCP, or Desktop services.

**Benefits:**
- ✅ No extra services or ports
- ✅ Minimal resource footprint
- ✅ Full vector search, ML algorithms, and embeddings
- ✅ Works with any PostgreSQL client

**Installation:**

Follow the [Native install](#native-install) steps above. That's it! You now have vector search and ML capabilities in PostgreSQL.

**Usage:**
```sql
-- Create a table with vectors
CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  content TEXT,
  embedding VECTOR(1536)
);

-- Create HNSW index
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);

-- Vector similarity search
SELECT id, content
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 10;
```

No additional services, ports, or configuration required!

## Service URLs & ports

| Service | How to reach it | Default credentials | Notes |
|---|---|---|---|
| **NeuronDB (PostgreSQL)** | `postgresql://neurondb:neurondb@localhost:5433/neurondb` | User: `neurondb`, Password: `neurondb` ⚠️ **Dev only** | Container: `neurondb-cpu`, Service: `neurondb` |
| **NeuronAgent** | `http://localhost:8080/health` | Health: no auth. API: API key required | Container: `neuronagent`, Service: `neuronagent` |
| **NeuronDesktop UI** | `http://localhost:3000` | No auth (development mode) | Container: `neurondesk-frontend`, Service: `neurondesk-frontend` |
| **NeuronDesktop API** | `http://localhost:8081/health` | Health: no auth. API: varies by config | Container: `neurondesk-api`, Service: `neurondesk-api` |
| **NeuronMCP** | stdio (JSON-RPC 2.0) | N/A (MCP protocol) | Container: `neurondb-mcp`, Service: `neuronmcp`. No HTTP port. |

> [!WARNING]
> **Production Security**: The default credentials shown above are for development only. Always use strong, unique passwords in production. Set `POSTGRES_PASSWORD` and other secrets via environment variables or a `.env` file (see [`env.example`](env.example)).

## Documentation

- **Start here**: [`Docs/documentation.md`](Docs/documentation.md) (documentation index)
- **Beginner walkthrough**: [`Docs/getting-started/simple-start.md`](Docs/getting-started/simple-start.md) - Step-by-step guide for beginners
- **Technical quick start**: [`QUICKSTART.md`](QUICKSTART.md) - Fast setup for experienced users
- **Complete guide**: [`Docs/getting-started/installation.md`](Docs/getting-started/installation.md) - Detailed installation options
- **Official docs**: [`neurondb.ai/docs`](https://www.neurondb.ai/docs) - Online documentation

### Module-wise Documentation

<details>
<summary><strong>NeuronDB documentation</strong></summary>

- **Getting Started**: [`installation.md`](NeuronDB/docs/getting-started/installation.md) • [`quickstart.md`](NeuronDB/docs/getting-started/quickstart.md)
- **Vector Search**: [`indexing.md`](NeuronDB/docs/vector-search/indexing.md) • [`distance-metrics.md`](NeuronDB/docs/vector-search/distance-metrics.md) • [`quantization.md`](NeuronDB/docs/vector-search/quantization.md)
- **Hybrid Search**: [`overview.md`](NeuronDB/docs/hybrid-search/overview.md) • [`multi-vector.md`](NeuronDB/docs/hybrid-search/multi-vector.md) • [`faceted-search.md`](NeuronDB/docs/hybrid-search/faceted-search.md)
- **RAG Pipeline**: [`overview.md`](NeuronDB/docs/rag/overview.md) • [`document-processing.md`](NeuronDB/docs/rag/document-processing.md) • [`llm-integration.md`](NeuronDB/docs/rag/llm-integration.md)
- **ML Algorithms**: [`clustering.md`](NeuronDB/docs/ml-algorithms/clustering.md) • [`classification.md`](NeuronDB/docs/ml-algorithms/classification.md) • [`regression.md`](NeuronDB/docs/ml-algorithms/regression.md)
- **ML Embeddings**: [`embedding-generation.md`](NeuronDB/docs/ml-embeddings/embedding-generation.md) • [`model-management.md`](NeuronDB/docs/ml-embeddings/model-management.md)
- **GPU Support**: [`cuda-support.md`](NeuronDB/docs/gpu/cuda-support.md) • [`rocm-support.md`](NeuronDB/docs/gpu/rocm-support.md) • [`metal-support.md`](NeuronDB/docs/gpu/metal-support.md)
- **Operations**: [`troubleshooting.md`](NeuronDB/docs/troubleshooting.md) • [`configuration.md`](NeuronDB/docs/configuration.md) • [`playbook.md`](NeuronDB/docs/operations/playbook.md)

</details>

<details>
<summary><strong>NeuronAgent documentation</strong></summary>

- **Architecture**: [`architecture.md`](NeuronAgent/docs/architecture.md)
- **API Reference**: [`API.md`](NeuronAgent/docs/api.md)
- **CLI Guide**: [`cli_guide.md`](NeuronAgent/docs/cli_guide.md)
- **Connectors**: [`connectors.md`](NeuronAgent/docs/connectors.md)
- **Deployment**: [`deployment.md`](NeuronAgent/docs/deployment.md)
- **Troubleshooting**: [`troubleshooting.md`](NeuronAgent/docs/troubleshooting.md)

</details>

<details>
<summary><strong>NeuronMCP documentation</strong></summary>

- **Setup Guide**: [`neurondb_mcp_setup.md`](NeuronMCP/docs/neurondb_mcp_setup.md)
- **Tool & Resource Catalog**: [`tool-resource-catalog.md`](NeuronMCP/docs/tool-resource-catalog.md)
- **Examples**: [`README.md`](NeuronMCP/docs/examples/README.md) • [`example-transcript.md`](NeuronMCP/docs/examples/example-transcript.md)

</details>

<details>
<summary><strong>NeuronDesktop documentation</strong></summary>

- **API Reference**: [`API.md`](NeuronDesktop/docs/api.md)
- **Deployment**: [`deployment.md`](NeuronDesktop/docs/deployment.md)
- **Integration**: [`integration.md`](NeuronDesktop/docs/integration.md)
- **NeuronAgent Usage**: [`neuronagent_usage.md`](NeuronDesktop/docs/neuronagent_usage.md)
- **NeuronMCP Setup**: [`neuronmcp_setup.md`](NeuronDesktop/docs/neuronmcp_setup.md)

</details>

## Repo layout

| Component | Path | What it is |
|---|---|---|
| NeuronDB | `NeuronDB/` | PostgreSQL extension with vector search, ML algorithms, GPU acceleration (CUDA/ROCm/Metal), embeddings, RAG pipeline, hybrid search, and background workers |
| NeuronAgent | `NeuronAgent/` | Agent runtime + REST/WebSocket API (Go) with multi-agent collaboration, DAG-based workflow engine with human-in-the-loop (HITL), hierarchical memory management, planning & reflection, evaluation framework, budget & cost management, 20+ tools (SQL, HTTP, Code, Shell, Browser, Filesystem, Memory, Collaboration, NeuronDB tools, Multimodal), Prometheus metrics, RBAC, audit logging, and background workers |
| NeuronMCP | `NeuronMCP/` | MCP server for MCP-compatible clients (Go) with 100+ tools (50+ vector operations, complete ML pipeline, RAG operations, 27 PostgreSQL admin tools, dataset loading), middleware system (validation, logging, timeout, error handling, auth, rate limiting), enterprise features (Prometheus metrics, webhooks, circuit breaker, caching, connection pooling), batch operations, progress tracking, authentication (JWT, API keys, OAuth2), and full MCP protocol support (prompts, sampling/completions, resources) |
| NeuronDesktop | `NeuronDesktop/` | Web UI + API for the ecosystem providing a unified interface |

### Component READMEs

- [`NeuronDB/README.md`](NeuronDB/README.md)
- [`NeuronAgent/README.md`](NeuronAgent/README.md)
- [`NeuronMCP/README.md`](NeuronMCP/README.md)
- [`NeuronDesktop/README.md`](NeuronDesktop/README.md)

### Examples

- [Examples index](examples/README.md)
- [Semantic search docs example](examples/semantic-search-docs/)
- [RAG chatbot (PDFs) example](examples/rag-chatbot-pdfs/)
- [Agent tools example](examples/agent-tools/)
- [MCP integration example](examples/mcp-integration/)

## Benchmarks

NeuronDB includes a benchmark suite to evaluate vector search, hybrid search, and RAG performance.

### Quick start

Run all benchmarks:

```bash
cd NeuronDB/benchmark
./run_bm.sh
```

This validates connectivity and runs the vector/hybrid/RAG benchmark groups.

### Benchmark suite

| Benchmark | Purpose | Datasets | Metrics |
|---|---|---|---|
| **Vector** | Vector similarity search performance | SIFT-128, GIST-960, GloVe-100 | QPS, Recall, Latency (avg, p50, p95, p99) |
| **Hybrid** | Combined vector + full-text search | BEIR (nfcorpus, msmarco, etc.) | NDCG, MAP, Recall, Precision |
| **RAG** | End-to-end RAG pipeline quality | MTEB, BEIR, RAGAS | Faithfulness, Relevancy, Context Precision |

### Vector Performance Benchmark

NeuronDB HNSW index building performance compared to pgvector:

**Test Environment:**
- **PostgreSQL**: 18.0
- **CPU**: Apple Silicon (aarch64-apple-darwin)
- **RAM**: 256MB `maintenance_work_mem`
- **Index Parameters**: `m = 16`, `ef_construction = 200`
- **Distance Metric**: L2 (Euclidean)

**Results:**

| Test Case | NeuronDB Optimized | pgvector | Speedup |
|-----------|-------------------|----------|---------|
| 50K vectors (128-dim L2) | 606ms (0.606s) ✅ | 6,108ms (6.108s) | **10.1x faster** |
| 50K vectors (128-dim Cosine) | 583ms (0.583s) ✅ | 5,113ms (5.113s) | **8.8x faster** |
| 10K vectors (768-dim L2) | 146ms (0.146s) ✅ | 3,960ms (3.960s) | **27.1x faster** |
| 100K vectors (128-dim L2) | 1,208ms (1.208s) ✅ | 15,696ms (15.696s) | **13.0x faster** |

**Optimizations Applied:**
- ✅ In-memory graph building using `maintenance_work_mem`
- ✅ Efficient neighbor finding during insert (not after flush)
- ✅ SIMD-optimized distance calculations (AVX2/NEON)
- ✅ Squared distance optimization (avoiding `sqrt()` overhead)
- ✅ Optimized flush with pre-computed neighbors

**Benchmark Scripts:**
- [`NeuronDB/benchmark/vector/neurondb_vector.sql`](NeuronDB/benchmark/vector/neurondb_vector.sql) - NeuronDB optimized HNSW benchmark
- [`NeuronDB/benchmark/vector/pgvector.sql`](NeuronDB/benchmark/vector/pgvector.sql) - pgvector reference benchmark

**How to Run:**
```bash
# Create separate databases for fair comparison
psql -d postgres -c "CREATE DATABASE neurondb_bench;"
psql -d postgres -c "CREATE DATABASE pgvector_bench;"

# Run NeuronDB benchmark
psql -d neurondb_bench -f NeuronDB/benchmark/vector/neurondb_vector.sql

# Run pgvector benchmark
psql -d pgvector_bench -f NeuronDB/benchmark/vector/pgvector.sql
```

> [!NOTE]
> Both benchmarks use identical test parameters (same vector generation pattern, same index parameters) to ensure fair comparison. See [`NeuronDB/benchmark/vector/README.md`](NeuronDB/benchmark/vector/README.md) for detailed benchmark documentation.

### Reproducible benchmarks

To reproduce benchmark results:

```bash
# Use exact Docker image tags (see releases)
docker pull ghcr.io/neurondb/neurondb-postgres:v1.0.0-pg17-cpu

# Run with documented hardware profile
cd NeuronDB/benchmark
./run_bm.sh --hardware-profile "cpu-8core-16gb"

# Individual benchmark with exact parameters
cd NeuronDB/benchmark/vector
./run_bm.py --prepare --load --run \
  --datasets sift-128-euclidean \
  --max-queries 1000 \
  --index hnsw \
  --ef-search 40
```

<details>
<summary><strong>Benchmark Results & Hardware Specs</strong></summary>

**Test Environment:**
- **CPU**: 13th Gen Intel(R) Core(TM) i5-13400F (16 cores)
- **RAM**: 31.1 GB
- **GPU**: NVIDIA GeForce RTX 5060, 8151 MiB
- **PostgreSQL**: 18.1

**Vector Search Benchmarks:**

| Metric | Value |
|--------|-------|
| Dataset | sift-128-euclidean |
| Dimensions | 128 |
| Training Vectors | 1,000,000 |
| Test Queries | 10,000 |
| Index Type | HNSW |
| Recall@10 | 1.000 |
| QPS | 1.90 |
| Avg Latency | 525.62 ms |
| p50 Latency | 524.68 ms |
| p95 Latency | 546.62 ms |
| p99 Latency | 555.52 ms |

**Hybrid Search Benchmarks:**

Status: Not run (see [`NeuronDB/benchmark/README.md`](NeuronDB/benchmark/README.md) for details)

**RAG Pipeline Benchmarks:**

Status: Completed (verification passed)

> [!NOTE]
> For detailed benchmark results, reproducible configurations, and additional datasets, see [`NeuronDB/benchmark/README.md`](NeuronDB/benchmark/README.md).

</details>

<details>
<summary><strong>Run individual benchmarks</strong></summary>

```bash
# Vector benchmark
cd NeuronDB/benchmark/vector
./run_bm.py --prepare --load --run --datasets sift-128-euclidean --max-queries 100

# Hybrid benchmark
cd NeuronDB/benchmark/hybrid
./run_bm.py --prepare --load --run --datasets nfcorpus --model all-MiniLM-L6-v2

# RAG benchmark
cd NeuronDB/benchmark/rag
./run_bm.py --prepare --verify --run --benchmarks mteb
```

</details>

## GPU profiles (CUDA / ROCm / Metal)

The root `docker-compose.yml` supports multiple GPU backends via Docker Compose profiles:

**Available profiles:**
- **CPU (default)**: `docker compose up -d` or `docker compose --profile cpu up -d`
- **CUDA (NVIDIA)**: `docker compose --profile cuda up -d`
- **ROCm (AMD)**: `docker compose --profile rocm up -d`
- **Metal (Apple Silicon)**: `docker compose --profile metal up -d`

**Ports differ per profile** (see [`env.example`](env.example)):

- **CPU**: `POSTGRES_PORT=5433` (default)
- **CUDA**: `POSTGRES_CUDA_PORT=5434`
- **ROCm**: `POSTGRES_ROCM_PORT=5435`
- **Metal**: `POSTGRES_METAL_PORT=5436`

**Example: Start with CUDA support**
```bash
# Stop CPU services first
docker compose down

# Start CUDA profile
docker compose --profile cuda up -d

# Verify CUDA services are running
docker compose ps

# Connect to CUDA-enabled PostgreSQL
psql "postgresql://neurondb:neurondb@localhost:5434/neurondb" -c "SELECT neurondb.version();"
```

**GPU Requirements:**
- **CUDA**: NVIDIA GPU with CUDA 12.2+ and nvidia-container-toolkit
- **ROCm**: AMD GPU with ROCm 5.7+ and proper device access
- **Metal**: Apple Silicon (M1/M2/M3) Mac with macOS 13+

<details>
<summary><strong>Common Docker commands</strong></summary>

```bash
# Stop everything (keep data volumes)
docker compose down

# Stop everything (delete data volumes - WARNING: deletes all data!)
docker compose down -v

# See status of all services
docker compose ps

# Tail logs for all services
docker compose logs -f

# Tail logs for specific services
docker compose logs -f neurondb neuronagent neuronmcp neurondesk-api neurondesk-frontend

# View last 100 lines of logs
docker compose logs --tail=100

# View logs for specific service
docker compose logs neurondb
```

</details>

## Operations

Key operational considerations for production:

- **Vacuum and bloat**: Vector indexes require periodic maintenance. See [`NeuronDB/docs/operations/playbook.md`](NeuronDB/docs/operations/playbook.md)
- **Index rebuild guidance**: When and how to rebuild HNSW/IVF indexes. See [`NeuronDB/docs/troubleshooting.md`](NeuronDB/docs/troubleshooting.md)
- **Memory configuration**: Tune `neurondb.maintenance_work_mem` and index-specific parameters. See [`NeuronDB/docs/configuration.md`](NeuronDB/docs/configuration.md)

## Contributing / security / license

- **Contributing**: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- **Security**: [`SECURITY.md`](SECURITY.md) - Report security issues to security@neurondb.ai
- **License**: [`LICENSE`](LICENSE) (proprietary)
- **Changelog**: [`CHANGELOG.md`](CHANGELOG.md) - See what's new
- **Roadmap**: [`ROADMAP.md`](ROADMAP.md) - Planned features
- **Releases**: [`RELEASE.md`](RELEASE.md) - Release process

## Project statistics

<details>
<summary><strong>Stats snapshot (may change)</strong></summary>

- **520+ SQL functions** in NeuronDB extension
- **52+ ML algorithms** supported
- **100+ MCP tools** available
- **4 integrated components** working together
- **3 PostgreSQL versions** supported (16, 17, 18)
- **4 GPU platforms** supported (CPU, CUDA, ROCm, Metal)

</details>

<details>
<summary><strong>Platform & version coverage</strong></summary>

| Category | Supported Versions |
|---|---|
| **PostgreSQL** | 16, 17, 18 |
| **Go** | 1.21, 1.22, 1.23, 1.24 |
| **Node.js** | 18 LTS, 20 LTS, 22 LTS |
| **Operating Systems** | Ubuntu 20.04, Ubuntu 22.04, macOS 13 (Ventura), macOS 14 (Sonoma) |
| **Architectures** | linux/amd64, linux/arm64 |

</details>
