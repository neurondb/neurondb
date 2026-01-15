# 🚀 Quick Start Guide

<div align="center">

**Get NeuronDB up and running in minutes with this step-by-step guide**

[![Quick Start](https://img.shields.io/badge/quick--start-5_min-green)](.)
[![Difficulty](https://img.shields.io/badge/difficulty-easy-brightgreen)](.)

</div>

---

> [!TIP]
> **New here?** Start with **[Simple Start Guide](Docs/getting-started/simple-start.md)** instead - it explains everything in plain English!

> [!NOTE]
> **TECHNICAL USER?** Continue below for a streamlined technical setup.

---

## 📋 Prerequisites

**Before starting, verify you have:**

- [ ] **Docker** 20.10+ and **Docker Compose** 2.0+
- [ ] **5-10 minutes** for setup and verification
- [ ] **4GB RAM** minimum (8GB recommended)
- [ ] Ports **5433, 8080, 8081, 3000** available

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Docker** | 20.10+ | Latest |
| **Docker Compose** | 2.0+ | Latest |
| **RAM** | 4GB | 8GB+ |
| **Disk Space** | 5GB | 10GB+ |
| **Time** | 5 min | 10 min |

<details>
<summary><strong>Verify Docker installation</strong></summary>

```bash
docker --version
docker compose version
```

**Expected output:**
```
Docker version 20.10.0 or higher
Docker Compose version v2.0.0 or higher
```

</details>

## 🚀 Step 1: Start All Services

Start the complete NeuronDB ecosystem with a single command:

```bash
# From the repository root
# Option 1: Use published images (recommended if available)
docker compose pull
docker compose up -d

# Option 2: Build from source
docker compose up -d --build
```

> [!NOTE]
> Published images from GitHub Container Registry (GHCR) are available starting with v2.0.0. See [Container Images documentation](Docs/deployment/container-images.md) for image names and tags.

This command will:

- [x] Build all Docker images (first time only, takes a few minutes)
- [x] Start PostgreSQL with NeuronDB extension
- [x] Start NeuronAgent (REST/WebSocket API server with multi-agent collaboration, workflow engine, HITL, and 16+ tools)
- [x] Start NeuronMCP (MCP protocol server with 100+ tools, middleware, and enterprise features)
- [x] Start NeuronDesktop (web interface with API and frontend)
- [x] Configure networking between all components

**What to expect:**
- First run: 5-10 minutes (building images)
- Subsequent runs: 30-60 seconds (containers already built)

**Check service status:**

```bash
docker compose ps
```

You should see five services running:

| Service | Status | Description |
|---------|--------|-------------|
| `neurondb` | healthy | PostgreSQL with NeuronDB extension |
| `neuronagent` | healthy | REST/WebSocket API server with multi-agent collaboration, workflow engine (DAG-based with HITL), hierarchical memory, evaluation framework, budget management, and 16+ tools |
| `neuronmcp` | healthy | MCP protocol server with 100+ tools, middleware system, batch operations, progress tracking, and enterprise features |
| `neurondesk-api` | healthy | NeuronDesktop API server |
| `neurondesk-frontend` | healthy | NeuronDesktop web interface |

**Wait for all services to show "healthy" status** (may take 30-60 seconds)

## ✅ Step 2: Run Verification Tests

Verify everything works with these quick smoke tests.

### Test 1: SQL Query (NeuronDB Extension)

Test that NeuronDB extension is loaded and functional:

```bash
# Using docker compose exec (service name)
docker compose exec neurondb psql -U neurondb -d neurondb -c "SELECT neurondb.version();"

# Or connect directly from host
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" -c "SELECT neurondb.version();"
```

**Expected output:**
```
version
--------
2.0
(1 row)
```

### Test 2: REST API Call (NeuronAgent)

Test that NeuronAgent REST API is responding:

```bash
# Health check endpoint (no authentication required)
curl -s http://localhost:8080/health

# Pretty print JSON response
curl -s http://localhost:8080/health | jq .
```

**Expected output:**
```json
{"status":"ok"}
```

**If you don't have jq installed, the raw response is:**
```
{"status":"ok"}
```

### Test 3: MCP Protocol Call (NeuronMCP)

Test that NeuronMCP server responds to MCP protocol:

```bash
# Test MCP server initialization via JSON-RPC
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}' | \
docker compose exec -T neuronmcp /app/neurondb-mcp | head -20
```

**Expected output:** JSON-RPC response with server information including protocol version and server capabilities.

**Alternative: Use the Python MCP client (if available):**

```bash
cd NeuronMCP/client
pip install -r requirements.txt
python neurondb_mcp_client.py -c ../../neuronmcp_server.json -e "list_tools" 2>/dev/null | head -30
```

**Note:** NeuronMCP communicates via stdio (standard input/output), not HTTP. It's designed to be used with MCP-compatible clients like Claude Desktop.

## 🔍 Step 3: Quick Verification

Run the verification commands manually:

```bash
# Test 1: SQL query
docker compose exec neurondb psql -U neurondb -d neurondb -c "SELECT neurondb.version();"

# Test 2: REST API
curl -s http://localhost:8080/health

# Test 3: MCP server (if using Python client)
cd NeuronMCP/client && python neurondb_mcp_client.py -c ../../neuronmcp_server.json -e "list_tools" 2>/dev/null | head -30
```

**Expected output:**
```
NeuronDB SQL query successful
NeuronAgent REST API responding
NeuronMCP server responding
All smoke tests passed!
```

## Next Steps

**Congratulations!** Your NeuronDB ecosystem is running. Try these examples:

<details>
<summary><strong>Example 1: Create a Vector Table</strong></summary>

```bash
# Create extension first (if not already created)
docker compose exec neurondb psql -U neurondb -d neurondb -c "CREATE EXTENSION IF NOT EXISTS neurondb;"

# Create table, insert data, and query
docker compose exec neurondb psql -U neurondb -d neurondb <<EOF
CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  content TEXT,
  embedding vector(3)
);

INSERT INTO documents (content, embedding)
VALUES ('Hello, world!', '[0.1, 0.2, 0.3]'::vector);

SELECT id, content FROM documents;
EOF
```

**Expected output:**
```
 id |    content    
----+---------------
  1 | Hello, world!
(1 row)
```

**Or connect directly from your host machine:**
```bash
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" <<EOF
CREATE EXTENSION IF NOT EXISTS neurondb;
CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  content TEXT,
  embedding vector(3)
);
INSERT INTO documents (content, embedding)
VALUES ('Hello, world!', '[0.1, 0.2, 0.3]'::vector);
SELECT id, content FROM documents;
EOF
```

</details>

<details>
<summary><strong>Example 2: Create an Agent via REST API</strong></summary>

```bash
# Step 1: Create an API key first (see NeuronAgent documentation for key generation)
# For testing, you can use the default development setup

# Step 2: Create an agent
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-agent",
    "system_prompt": "You are a helpful assistant",
    "model_name": "gpt-4",
    "enabled_tools": [],
    "config": {}
  }'
```

**Expected output:**
```json
{
  "id": "agent-uuid-here",
  "name": "test-agent",
  "status": "active",
  "created_at": "2026-01-08T..."
}
```

**Note:** 
- Replace `YOUR_API_KEY` with an actual API key from NeuronAgent
- See [NeuronAgent API documentation](NeuronAgent/docs/api.md) for authentication setup
- For development, check if API key authentication is enabled in your configuration

</details>

## 🔧 Troubleshooting

**Having issues?** Check these common problems:

### Services Won't Start

**Check logs:**
```bash
# View logs for all services
docker compose logs

# View logs for specific service
docker compose logs neurondb
docker compose logs neuronagent
docker compose logs neuronmcp
docker compose logs neurondesk-api
docker compose logs neurondesk-frontend

# Follow logs in real-time
docker compose logs -f neurondb

# View last 50 lines of logs
docker compose logs --tail=50 neurondb
```

**Common issues:**

<details>
<summary><strong>Port already in use</strong></summary>

- Change ports in `docker-compose.yml` or stop conflicting services
- Default ports: **5433** (PostgreSQL), **8080** (NeuronAgent), **8081** (NeuronDesktop API), **3000** (NeuronDesktop UI)

</details>

<details>
<summary><strong>Out of memory</strong></summary>

- Ensure Docker has at least **4GB RAM** allocated
- Check: Docker Desktop → Settings → Resources
- Recommended: **8GB+** for optimal performance

</details>

<details>
<summary><strong>Build failures</strong></summary>

- Ensure Docker has sufficient disk space (**10GB+** recommended)
- Try: `docker compose build --no-cache`
- Check Docker logs: `docker compose logs --tail=50`

</details>

### Services Start But Tests Fail

**Check service health:**
```bash
docker compose ps
```

All services should show "healthy" status. If not, check logs:

```bash
docker compose logs --tail=50 neurondb
docker compose logs --tail=50 neuronagent
docker compose logs --tail=50 neuronmcp
docker compose logs --tail=50 neurondesk-api
docker compose logs --tail=50 neurondesk-frontend
```

**Wait for initialization:**
- NeuronDB may take 30-60 seconds to initialize
- NeuronAgent waits for NeuronDB to be ready
- NeuronMCP waits for NeuronDB to be ready

## 🗑️ Uninstall and Cleanup

When you're done, clean up all resources:

### Stop Services

```bash
docker compose down
```

This stops all containers but preserves data volumes.

### Remove All Resources (Full Cleanup)

**Warning: This will delete all data!**

```bash
# Stop and remove containers, networks, and volumes
docker compose down -v

# Remove Docker images (optional)
docker rmi neurondb:cpu-pg17 neuronagent:latest neurondb-mcp:latest neurondesk-api:latest neurondesk-frontend:latest

# Remove Docker network (if it persists)
docker network rm neurondb-network 2>/dev/null || true
```

### Partial Cleanup Options

**Keep data, remove containers:**
```bash
docker compose down
```

**Remove data volumes but keep images:**
```bash
docker compose down -v
```

**Remove specific service only:**
```bash
docker compose stop neuronagent
docker compose rm neuronagent
```

## 📊 What's Running?

After `docker compose up -d`, you have:

| Service Name | Container Name | Port | Description | Connection String |
|--------------|---------------|------|-------------|-------------------|
| **NeuronDB** | `neurondb-cpu` | 5433 | PostgreSQL with NeuronDB extension | `postgresql://neurondb:neurondb@localhost:5433/neurondb` |
| **NeuronAgent** | `neuronagent` | 8080 | REST API server for agent runtime | `http://localhost:8080` |
| **NeuronMCP** | `neurondb-mcp` | - | MCP protocol server (stdio) | stdio (JSON-RPC 2.0) |
| **NeuronDesktop API** | `neurondesk-api` | 8081 | NeuronDesktop backend API | `http://localhost:8081` |
| **NeuronDesktop Frontend** | `neurondesk-frontend` | 3000 | NeuronDesktop web interface | `http://localhost:3000` |

**Network:**All services communicate via `neurondb-network` Docker network.

**Data:**Persistent data stored in Docker volumes:
- `neurondb-data` (PostgreSQL data)

## 🔌 Accessing Services

### PostgreSQL (NeuronDB)

```bash
# Connect via docker compose exec (inside container)
docker compose exec neurondb psql -U neurondb -d neurondb

# Or connect directly from host machine
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb"

# Quick test query
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" -c "SELECT neurondb.version();"

# Create extension (if not already created)
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" -c "CREATE EXTENSION IF NOT EXISTS neurondb;"
```

> [!WARNING]
> Default password `neurondb` is for **development only**. Always use strong passwords in production!

### NeuronAgent REST API

```bash
# Health check (no authentication required)
curl -s http://localhost:8080/health

# Pretty print JSON response
curl -s http://localhost:8080/health | jq .

# List agents (authentication required - replace YOUR_API_KEY)
curl -s -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:8080/api/v1/agents | jq .

# Get agent by ID
curl -s -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:8080/api/v1/agents/AGENT_ID | jq .
```

**Expected health check response:**
```json
{"status":"ok"}
```

### NeuronMCP Server

MCP server communicates via stdio (JSON-RPC 2.0). Use MCP clients or the Python client:

```bash
cd NeuronMCP/client
./neurondb_mcp_client.py -c ../../neuronmcp_server.json -e "list_tools"
```

### NeuronDesktop Web Interface

Access the unified web interface:

```bash
# Web UI (open in browser)
open http://localhost:3000
# Or visit: http://localhost:3000

# API health check endpoint
curl -s http://localhost:8081/health

# Pretty print JSON response
curl -s http://localhost:8081/health | jq .
```

**Expected health check response:**
```json
{"status":"ok","service":"neurondesk-api"}
```

**NeuronDesktop provides:**
- Unified web interface for all NeuronDB ecosystem components
- Real-time monitoring and metrics
- SQL console for direct database queries
- Agent management interface
- Vector search and RAG pipeline tools

## ⚙️ Configuration

All services use default configuration suitable for development. To customize:

1. **Edit `docker-compose.yml`**for service-level changes
2. **Set environment variables**for runtime configuration
3. **Mount configuration files**for advanced setups

See component-specific documentation for detailed configuration options.

## 💬 Getting Help

- **Documentation:**See [README.md](README.md) for detailed documentation
- **Issues:**Check service logs: `docker compose logs [service-name]`
- **Support:**Contact support@neurondb.ai

## 📦 Quickstart with Sample Data

Get started immediately with pre-generated sample data:

```bash
# Set up quickstart data pack (200 sample documents with embeddings)
./scripts/neurondb-quickstart-data.sh
```

This script will:
- Generate 200 sample documents with pre-computed embeddings
- Create the `quickstart_documents` table
- Create an HNSW index for fast similarity search
- Verify the setup

**Try a similarity search:**

```bash
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" <<EOF
WITH q AS (SELECT embed_text('machine learning') AS query_vec)
SELECT title, 1 - (embedding <=> q.query_vec) AS similarity
FROM quickstart_documents, q
ORDER BY embedding <=> q.query_vec
LIMIT 5;
EOF
```

For more details, see [Quickstart Data Pack documentation](examples/quickstart-data/README.md).

## 📝 SQL Recipes

Explore ready-to-run SQL recipes for common patterns:

**Vector Search Recipes:**
- Basic similarity search
- Filtered search with metadata
- Multiple distance metrics
- Performance tuning

**Hybrid Search Recipes:**
- Text + vector combination
- Weighted scoring
- Reciprocal Rank Fusion (RRF)
- Faceted search

**Indexing Recipes:**
- Create HNSW indexes
- Create IVF indexes
- Tune parameters
- Index maintenance

**RAG Patterns:**
- Document chunking
- Context retrieval
- Reranking
- Complete RAG pipeline

**Quick start:**
```bash
# Run a vector search recipe
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" \
  -f examples/sql-recipes/vector-search/01_basic_similarity.sql
```

See [SQL Recipe Library](examples/sql-recipes/README.md) for all recipes.

## 🛠️ CLI Helpers

Use command-line helpers for common operations:

### Create Vector Indexes

```bash
# Create HNSW index (default)
./scripts/neurondb-create-index.sh \
  --table quickstart_documents \
  --column embedding

# Create with custom parameters
./scripts/neurondb-create-index.sh \
  --table documents \
  --column embedding \
  --type hnsw \
  --metric cosine \
  --m 32 \
  --ef-construction 128

# Create IVF index
./scripts/neurondb-create-index.sh \
  --table documents \
  --column embedding \
  --type ivf \
  --lists 50
```

### Load Embedding Models

```bash
# Load a HuggingFace embedding model
./scripts/neurondb-load-model.sh \
  --name mini_lm \
  --model sentence-transformers/all-MiniLM-L6-v2

# Load with custom configuration
./scripts/neurondb-load-model.sh \
  --name mpnet \
  --model sentence-transformers/all-mpnet-base-v2 \
  --config '{"batch_size": 64}'
```

For more details, see the [Scripts README](scripts/README.md).

## Next Steps

- Read the [full documentation](README.md)
- Explore [NeuronDB examples](NeuronDB/demo/)
- Try [NeuronAgent examples](NeuronAgent/examples/)
- Check out [NeuronMCP documentation](NeuronMCP/README.md)
- Access [NeuronDesktop web interface](http://localhost:3000) and see [NeuronDesktop documentation](NeuronDesktop/README.md)

