# NeuronAgent

[![Go](https://img.shields.io/badge/Go-1.23+-00ADD8.svg)](https://golang.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-blue.svg)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](../LICENSE)

AI agent runtime system providing REST API and WebSocket endpoints for building applications with long-term memory and tool execution.

## Overview

NeuronAgent integrates with NeuronDB PostgreSQL extension to provide agent runtime capabilities. Use it to build autonomous agent systems with persistent memory, tool execution, and streaming responses.

## Documentation

**For comprehensive documentation, detailed tutorials, complete API references, and best practices, visit:**

🌐 **[https://www.neurondb.ai/docs/neuronagent](https://www.neurondb.ai/docs/neuronagent)**

### Local Documentation

- **[Features](docs/features.md)** - Complete feature list and capabilities
- **[API Reference](docs/API.md)** - Complete REST API documentation
- **[Architecture](docs/ARCHITECTURE.md)** - System architecture and design
- **[CLI Guide](docs/CLI_GUIDE.md)** - Command-line interface guide
- **[Deployment](docs/DEPLOYMENT.md)** - Deployment and configuration
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions

The official documentation provides:
- Complete REST API reference with examples
- WebSocket integration guides
- Agent configuration and profiles
- Tool development and registration
- Production deployment guides
- Performance optimization tips

## Features

| Feature | Description |
|---------|-------------|
| **Agent Runtime** | Complete state machine for autonomous task execution with persistent memory |
| **Multi-Agent Collaboration** | Agent-to-agent communication, task delegation, shared workspaces, and hierarchical agent structures |
| **Workflow Engine** | DAG-based workflow execution with agent, tool, HTTP, approval, and conditional steps |
| **Human-in-the-Loop (HITL)** | Approval gates, feedback loops, and human oversight in workflows with email/webhook notifications |
| **Hierarchical Memory** | Multi-level memory organization with HNSW-based vector search for better context retrieval |
| **Long-term Memory** | HNSW-based vector search for context retrieval with memory promotion |
| **Planning & Reflection** | LLM-based planning with task decomposition, agent self-reflection, and quality assessment |
| **Evaluation Framework** | Built-in evaluation system for agent performance with automated quality scoring |
| **Budget & Cost Management** | Real-time cost tracking, per-agent and per-session budget controls, and budget alerts |
| **Tool System (16+ Tools)** | Extensible tool registry with SQL, HTTP, Code, Shell, Browser (Playwright), Visualization, Filesystem (virtual), Memory, Collaboration, NeuronDB tools (RAG, Hybrid Search, Reranking, Vector, ML, Analytics), and Multimodal processing. Supports custom tool registration. |
| **REST API** | Full CRUD API for agents, sessions, messages, workflows, plans, budgets, and collaborations |
| **WebSocket Support** | Streaming agent responses in real-time with event streaming |
| **Authentication & Security** | API key-based authentication with bcrypt hashing, RBAC, fine-grained permissions, and audit logging |
| **Background Jobs** | PostgreSQL-based job queue with worker pool, async task execution, and memory promotion |
| **Observability** | Prometheus metrics, structured logging, distributed tracing, and debugging tools |
| **NeuronDB Integration** | Direct integration with NeuronDB embedding, LLM, vector search, and ML functions |
| **Virtual Filesystem** | Isolated filesystem for agents with secure file operations |
| **Versioning & History** | Version control for agents, execution replay, and state snapshots |

## Architecture

```
┌─────────────────────────────────────────────┐
│          NeuronAgent Service                │
├─────────────────────────────────────────────┤
│  REST API     │  WebSocket  │  Health      │
├─────────────────────────────────────────────┤
│  Agent State Machine │  Session Management  │
├─────────────────────────────────────────────┤
│  Tool Registry │  Memory Store │  Job Queue │
├─────────────────────────────────────────────┤
│          NeuronDB PostgreSQL                │
│  (Vector Search │  Embeddings │  LLM)       │
└─────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- PostgreSQL 16 or later
- NeuronDB extension installed
- Go 1.23 or later (for building from source)

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

# Run NeuronAgent migrations
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" -f NeuronAgent/sql/neuronagent_initial_schema.sql
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" -f NeuronAgent/sql/neuronagent_add_indexes.sql
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" -f NeuronAgent/sql/neuronagent_add_triggers.sql
```

**Option 2: Native PostgreSQL Installation**

```bash
createdb neurondb
psql -d neurondb -c "CREATE EXTENSION neurondb;"

# Run migrations
psql -d neurondb -f sql/neuronagent_initial_schema.sql
psql -d neurondb -f sql/neuronagent_add_indexes.sql
psql -d neurondb -f sql/neuronagent_add_triggers.sql
```

### Configuration

Set environment variables or create `config.yaml`:

**For Docker Compose setup (default):**
```bash
export DB_HOST=neurondb  # Service name in Docker network
export DB_PORT=5432       # Container port (not host port)
export DB_NAME=neurondb
export DB_USER=neurondb
export DB_PASSWORD=neurondb
export SERVER_PORT=8080
```

**For native PostgreSQL or connecting from host:**
```bash
export DB_HOST=localhost
export DB_PORT=5433       # Host port (Docker Compose default)
export DB_NAME=neurondb
export DB_USER=neurondb
export DB_PASSWORD=neurondb
export SERVER_PORT=8080
```

See [Deployment Guide](docs/DEPLOYMENT.md) for complete configuration options.

### Run Service

#### Automated Installation (Recommended)

Use the installation script for easy setup:

```bash
# From repository root
sudo ./scripts/install-neuronagent.sh

# With system service enabled
sudo ./scripts/install-neuronagent.sh --enable-service
```

#### Manual Build and Run

From source:

```bash
go run cmd/agent-server/main.go
```

Or build and run:

```bash
make build
./bin/neuronagent
```

#### Using Docker

**Option 1: Root docker-compose.yml (Recommended)**
```bash
# From repository root
docker compose up -d neuronagent

# Check status
docker compose ps neuronagent

# View logs
docker compose logs -f neuronagent
```

**Option 2: NeuronAgent-specific docker-compose**
```bash
cd docker
# Optionally create .env file with your configuration
# Or use environment variables directly (docker-compose.yml has defaults)
docker compose up -d
```

See [Docker Guide](docker/readme.md) for Docker deployment details.

#### Running as a Service

For systemd (Linux) or launchd (macOS), see [Service Management Guide](../../Docs/getting-started/installation-services.md).

### Verify Installation

Test health endpoint (no authentication required):

```bash
curl -s http://localhost:8080/health
```

**Expected output:**
```json
{"status":"ok"}
```

Test API with authentication:

```bash
# Replace YOUR_API_KEY with actual API key
curl -s -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:8080/api/v1/agents | jq .
```

**Expected output:**
```json
[]
```

(Empty array if no agents created yet)

**Create your first agent:**
```bash
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-first-agent",
    "system_prompt": "You are a helpful assistant",
    "model_name": "gpt-4",
    "enabled_tools": [],
    "config": {}
  }' | jq .
```

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

See [API Documentation](docs/API.md) for complete API reference.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | `localhost` | Database hostname |
| `DB_PORT` | `5432` | Database port |
| `DB_NAME` | `neurondb` | Database name |
| `DB_USER` | `neurondb` | Database username |
| `DB_PASSWORD` | `neurondb` | Database password |
| `DB_MAX_OPEN_CONNS` | `25` | Maximum open connections |
| `DB_MAX_IDLE_CONNS` | `5` | Maximum idle connections |
| `DB_CONN_MAX_LIFETIME` | `5m` | Connection max lifetime |
| `SERVER_HOST` | `0.0.0.0` | Server bind address |
| `SERVER_PORT` | `8080` | Server port |
| `SERVER_READ_TIMEOUT` | `30s` | Read timeout |
| `SERVER_WRITE_TIMEOUT` | `30s` | Write timeout |
| `LOG_LEVEL` | `info` | Log level (debug, info, warn, error) |
| `LOG_FORMAT` | `json` | Log format (json, text) |
| `CONFIG_PATH` | - | Path to config.yaml file |

### Configuration File

Create `config.yaml`:

```yaml
database:
  host: localhost
  port: 5432
  name: neurondb
  user: neurondb
  password: neurondb
  max_open_conns: 25
  max_idle_conns: 5
  conn_max_lifetime: 5m

server:
  host: 0.0.0.0
  port: 8080
  read_timeout: 30s
  write_timeout: 30s

logging:
  level: info
  format: json
```

Environment variables override configuration file values.

## Usage Examples

### Create Agent

```bash
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "research_agent",
    "profile": "research",
    "tools": ["sql", "http"]
  }'
```

### Create Session

```bash
curl -X POST http://localhost:8080/api/v1/sessions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agent_123"
  }'
```

### Send Message

```bash
curl -X POST http://localhost:8080/api/v1/sessions/SESSION_ID/messages \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Find documents about machine learning"
  }'
```

### WebSocket Connection

Connect to WebSocket endpoint for streaming responses:

```javascript
const ws = new WebSocket('ws://localhost:8080/ws?session_id=SESSION_ID');
ws.onmessage = (event) => {
  console.log('Agent response:', JSON.parse(event.data));
};
```

## Documentation

| Document | Description |
|----------|-------------|
| [API Reference](docs/API.md) | Complete REST API documentation |
| [Architecture](docs/ARCHITECTURE.md) | System design and structure |
| [Deployment Guide](docs/DEPLOYMENT.md) | Production deployment instructions |
| [Docker Guide](docker/readme.md) | Container deployment guide |

## System Requirements

| Component | Requirement |
|-----------|-------------|
| PostgreSQL | 16 or later |
| NeuronDB Extension | Installed and enabled |
| Go | 1.23 or later (for building) |
| Network | Port 8080 available (configurable) |

## Integration with NeuronDB

NeuronAgent requires:

- PostgreSQL database with NeuronDB extension installed
- Database user with appropriate permissions
- Access to NeuronDB vector search and embedding functions

See [NeuronDB documentation](../NeuronDB/readme.md) for installation instructions.

## Security

- API key authentication required for all API endpoints
- Rate limiting configured per API key
- Database credentials stored securely via environment variables
- Supports TLS/SSL for encrypted connections
- Non-root user in Docker containers

See [Deployment Guide](docs/DEPLOYMENT.md) for security best practices.

## Troubleshooting

### Service Won't Start

Check database connection:

```bash
psql -h localhost -p 5432 -U neurondb -d neurondb -c "SELECT 1;"
```

Verify environment variables:

```bash
env | grep -E "DB_|SERVER_"
```

Check logs:

```bash
docker compose logs agent-server
```

### Database Connection Failed

Verify NeuronDB extension:

```sql
SELECT * FROM pg_extension WHERE extname = 'neurondb';
```

Check database permissions:

```sql
GRANT ALL PRIVILEGES ON DATABASE neurondb TO neurondb;
GRANT ALL ON SCHEMA neurondb_agent TO neurondb;
```

### API Not Responding

Test health endpoint:

```bash
curl http://localhost:8080/health
```

Verify API key:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:8080/api/v1/agents
```

## Support

- **Documentation**: [Component Documentation](../readme.md)
- **GitHub Issues**: [Report Issues](https://github.com/neurondb/NeurondB/issues)
- **Email**: support@neurondb.ai

## License

See [LICENSE](../LICENSE) file for license information.
