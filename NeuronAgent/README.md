# NeuronAgent

<div align="center">

**AI agent runtime system providing REST API and WebSocket endpoints for building applications with long-term memory and tool execution**

[![Go](https://img.shields.io/badge/Go-1.23+-00ADD8.svg)](https://golang.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-blue.svg)](https://www.postgresql.org/)
[![Version](https://img.shields.io/badge/version-3.0.0--devel-blue.svg)](https://github.com/neurondb/neurondb)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](../LICENSE)
[![Documentation](https://img.shields.io/badge/docs-neurondb.ai-brightgreen.svg)](https://www.neurondb.ai/docs/neuronagent)

</div>

## 📑 Table of Contents

<details>
<summary><strong>Expand full table of contents</strong></summary>

- [Overview](#overview)
  - [Key Capabilities](#key-capabilities)
- [Documentation](#documentation)
- [Features](#features)
- [Architecture](#architecture)
  - [System Architecture](#system-architecture)
  - [Agent Execution Flow](#agent-execution-flow)
  - [Workflow Engine (DAG)](#workflow-engine-dag)
  - [Memory Hierarchy](#memory-hierarchy)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Database Setup](#database-setup)
  - [Configuration](#configuration)
  - [Run Service](#run-service)
  - [Verify Installation](#verify-installation)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration-1)
- [Usage Examples](#usage-examples)
- [Documentation](#documentation-1)
- [System Requirements](#system-requirements)
- [Integration with NeuronDB](#integration-with-neurondb)
- [Security](#security)
- [Troubleshooting](#troubleshooting)
- [Support](#support)
- [License](#license)

</details>

---

## Overview

NeuronAgent integrates with NeuronDB PostgreSQL extension to provide agent runtime capabilities. Use it to build autonomous agent systems with persistent memory, tool execution, and streaming responses.

### Key Capabilities

- 🤖 **Autonomous Agents** - Build agents that can reason, plan, and execute complex tasks
- 🧠 **Persistent Memory** - Long-term memory with vector search for context retrieval
- 🔧 **Tool Execution** - 18+ built-in tools plus custom tool registration
- 🔄 **Workflow Orchestration** - DAG-based workflows with human-in-the-loop support
- 👥 **Multi-Agent Collaboration** - Agents can communicate and collaborate on tasks
- 💰 **Cost Management** - Real-time budget tracking and cost controls
- 📊 **Observability** - Prometheus metrics, structured logging, and distributed tracing

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

<details>
<summary><strong>📊 Complete Feature List</strong></summary>

| Feature | Description | Status |
|:--------|:------------|:-------|
| **Agent Runtime** | Complete state machine for autonomous task execution with persistent memory | ✅ Stable |
| **Multi-Agent Collaboration** | Agent-to-agent communication, task delegation, shared workspaces, and hierarchical agent structures | ✅ Stable |
| **Workflow Engine** | DAG-based workflow execution with agent, tool, HTTP, approval, and conditional steps | ✅ Stable |
| **Human-in-the-Loop (HITL)** | Approval gates, feedback loops, and human oversight in workflows with email/webhook notifications | ✅ Stable |
| **Hierarchical Memory** | Multi-level memory organization with HNSW-based vector search for better context retrieval | ✅ Stable |
| **Long-term Memory** | HNSW-based vector search for context retrieval with memory promotion | ✅ Stable |
| **Agentic RAG** | Intelligent retrieval where agent decides when and where to retrieve information | ✅ Stable |
| **Agent Memory** | Read/write memory with learning from interactions and personalization | ✅ Stable |
| **Memory Feedback** | User feedback system to improve memory quality over time | ✅ Stable |
| **Adaptive Memory** | Usage-based importance adjustment, consolidation, and compression | ✅ Stable |
| **Cross-Session Memory** | Share memories across sessions with automatic relevance detection | ✅ Stable |
| **Planning & Reflection** | LLM-based planning with task decomposition, agent self-reflection, and quality assessment | ✅ Stable |
| **Evaluation Framework** | Built-in evaluation system for agent performance with automated quality scoring | ✅ Stable |
| **Budget & Cost Management** | Real-time cost tracking, per-agent and per-session budget controls, and budget alerts | ✅ Stable |
| **Tool System** | 18+ tools: SQL, HTTP, Code, Shell, Browser, Visualization, Filesystem, Memory, Collaboration, NeuronDB tools (ML, Vector, RAG, Hybrid Search, Reranking, Analytics), Multimodal, Web Search, Retrieval | ✅ Stable |
| **REST API** | Full CRUD API for agents, sessions, messages, workflows, plans, budgets, and collaborations | ✅ Stable |
| **WebSocket Support** | Streaming agent responses in real-time with event streaming | ✅ Stable |
| **Authentication & Security** | API key-based authentication with bcrypt hashing, RBAC, fine-grained permissions, and audit logging | ✅ Stable |
| **Background Jobs** | PostgreSQL-based job queue with worker pool, async task execution, and memory promotion | ✅ Stable |
| **Observability** | Prometheus metrics, structured logging, distributed tracing, and debugging tools | ✅ Stable |
| **NeuronDB Integration** | Direct integration with NeuronDB embedding, LLM, vector search, and ML functions | ✅ Stable |
| **Virtual Filesystem** | Isolated filesystem for agents with secure file operations | ✅ Stable |
| **Versioning & History** | Version control for agents, execution replay, and state snapshots | ✅ Stable |

</details>

## Architecture

### System Architecture

```mermaid
graph TB
    subgraph API["API Layer"]
        REST[REST API<br/>Port 8080]
        WS[WebSocket<br/>Real-time Streaming]
        HEALTH[Health Check<br/>/health, /metrics]
    end
    
    subgraph CORE["Core Services"]
        STATE[Agent State Machine<br/>Task Execution]
        SESSION[Session Management<br/>Conversation Context]
        MEMORY[Memory Store<br/>HNSW Vector Search]
        TOOLS[Tool Registry<br/>18+ Tools]
        WORKFLOW[Workflow Engine<br/>DAG Execution]
    end
    
    subgraph WORKERS["Background Workers"]
        JOBQ[Job Queue<br/>PostgreSQL-based]
        PROMOTER[Memory Promoter<br/>Long-term Storage]
        VERIFIER[Verifier Worker<br/>Quality Checks]
    end
    
    subgraph DB["NeuronDB PostgreSQL"]
        VECTOR[Vector Search<br/>HNSW Indexes]
        EMBED[Embeddings<br/>Text/Image/Multimodal]
        LLM[LLM Integration<br/>OpenAI/Anthropic]
        ML[ML Functions<br/>52+ Algorithms]
    end
    
    REST --> STATE
    WS --> STATE
    STATE --> SESSION
    STATE --> MEMORY
    STATE --> TOOLS
    STATE --> WORKFLOW
    MEMORY --> VECTOR
    TOOLS --> DB
    WORKFLOW --> TOOLS
    JOBQ --> PROMOTER
    JOBQ --> VERIFIER
    PROMOTER --> MEMORY
    MEMORY --> VECTOR
    
    style API fill:#e3f2fd
    style CORE fill:#fff3e0
    style WORKERS fill:#f3e5f5
    style DB fill:#e8f5e9
```

### Agent Execution Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as REST/WebSocket API
    participant Agent as Agent Runtime
    participant Memory as Memory Store
    participant Tools as Tool Registry
    participant DB as NeuronDB
    
    Client->>API: POST /api/v1/sessions/{id}/messages
    API->>Agent: Process message
    
    Agent->>Memory: Retrieve context (vector search)
    Memory->>DB: Query similar memories
    DB-->>Memory: Return relevant context
    Memory-->>Agent: Context + history
    
    Agent->>Agent: Generate response plan
    Agent->>Tools: Execute tool (if needed)
    Tools->>DB: Execute SQL/HTTP/Code
    DB-->>Tools: Tool results
    Tools-->>Agent: Tool output
    
    Agent->>Memory: Store new memory
    Memory->>DB: Store with embedding
    Agent->>API: Stream response
    API-->>Client: Real-time updates
```

### Workflow Engine (DAG)

```mermaid
graph LR
    START([Start]) --> STEP1[Agent Step<br/>LLM Processing]
    STEP1 --> COND{Conditional<br/>Check}
    COND -->|True| STEP2[Tool Step<br/>SQL Query]
    COND -->|False| STEP3[HTTP Step<br/>API Call]
    STEP2 --> APPROVAL{Approval<br/>Gate}
    STEP3 --> APPROVAL
    APPROVAL -->|Approved| STEP4[Agent Step<br/>Final Response]
    APPROVAL -->|Rejected| STEP1
    STEP4 --> END([End])
    
    style START fill:#c8e6c9
    style END fill:#ffcdd2
    style COND fill:#fff9c4
    style APPROVAL fill:#f8bbd0
```

### Memory Hierarchy

```mermaid
graph TD
    subgraph MEM["Memory System"]
        EPISODIC[Episodic Memory<br/>Recent conversations<br/>Short-term]
        SEMANTIC[Semantic Memory<br/>Vector embeddings<br/>HNSW search]
        WORKING[Working Memory<br/>Current session<br/>Active context]
    end
    
    WORKING -->|Promote| EPISODIC
    EPISODIC -->|Promote| SEMANTIC
    SEMANTIC -->|Retrieve| WORKING
    
    style EPISODIC fill:#ffebee
    style SEMANTIC fill:#e8f5e9
    style WORKING fill:#e3f2fd
```

> [!NOTE]
> Memory promotion follows a hierarchical structure: working memory (current session) → episodic memory (recent conversations) → semantic memory (long-term knowledge). HNSW indexes enable fast similarity search across all memory levels.

## Quick Start

### Prerequisites

<details>
<summary><strong>📋 Prerequisites Checklist</strong></summary>

- [ ] PostgreSQL 16 or later installed
- [ ] NeuronDB extension installed and enabled
- [ ] Go 1.23 or later (for building from source)
- [ ] Port 8080 available (configurable)
- [ ] API key generated (for authentication)

</details>

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

See [Deployment Guide](docs/deployment.md) for complete configuration options.

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

See [Docker Guide](docker/README.md) for Docker deployment details.

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

**Expected output:**
```json
{
  "id": "agent_123",
  "name": "my-first-agent",
  "system_prompt": "You are a helpful assistant",
  "model_name": "gpt-4",
  "enabled_tools": [],
  "created_at": "2024-01-01T00:00:00Z"
}
```

> [!SUCCESS]
> **Agent created!** You can now create a session and start chatting with your agent.

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

<details>
<summary><strong>📝 Complete Usage Examples</strong></summary>

### Create Agent

```bash
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "research_agent",
    "system_prompt": "You are a research assistant that helps find and analyze information.",
    "model_name": "gpt-4",
    "profile": "research",
    "enabled_tools": ["sql", "http", "memory"],
    "config": {
      "temperature": 0.7,
      "max_tokens": 2000
    }
  }' | jq .
```

**Expected output:**
```json
{
  "id": "agent_research_001",
  "name": "research_agent",
  "system_prompt": "You are a research assistant...",
  "model_name": "gpt-4",
  "enabled_tools": ["sql", "http", "memory"],
  "created_at": "2024-01-01T00:00:00Z"
}
```

### Create Session

```bash
curl -X POST http://localhost:8080/api/v1/sessions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agent_research_001",
    "metadata": {
      "user_id": "user_123",
      "context": "research_project"
    }
  }' | jq .
```

**Expected output:**
```json
{
  "id": "session_abc123",
  "agent_id": "agent_research_001",
  "created_at": "2024-01-01T00:00:00Z",
  "metadata": {
    "user_id": "user_123",
    "context": "research_project"
  }
}
```

### Send Message

```bash
curl -X POST http://localhost:8080/api/v1/sessions/session_abc123/messages \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Find documents about machine learning",
    "metadata": {
      "priority": "high"
    }
  }' | jq .
```

**Expected output:**
```json
{
  "id": "msg_xyz789",
  "session_id": "session_abc123",
  "role": "user",
  "content": "Find documents about machine learning",
  "created_at": "2024-01-01T00:01:00Z"
}
```

### WebSocket Connection

Connect to WebSocket endpoint for streaming responses:

```javascript
const ws = new WebSocket('ws://localhost:8080/ws?session_id=session_abc123');

ws.onopen = () => {
  console.log('WebSocket connected');
  // Send message
  ws.send(JSON.stringify({
    type: 'message',
    content: 'Find documents about machine learning'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Agent response:', data);
  
  if (data.type === 'token') {
    process.stdout.write(data.content); // Stream tokens
  } else if (data.type === 'complete') {
    console.log('\nResponse complete');
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('WebSocket closed');
};
```

### Python Example

```python
import requests
import json

API_BASE = "http://localhost:8080/api/v1"
API_KEY = "YOUR_API_KEY"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Create agent
agent_data = {
    "name": "python_agent",
    "system_prompt": "You are a helpful Python assistant",
    "model_name": "gpt-4",
    "enabled_tools": ["sql", "code"]
}

response = requests.post(
    f"{API_BASE}/agents",
    headers=HEADERS,
    json=agent_data
)
agent = response.json()
print(f"Created agent: {agent['id']}")

# Create session
session_data = {"agent_id": agent['id']}
response = requests.post(
    f"{API_BASE}/sessions",
    headers=HEADERS,
    json=session_data
)
session = response.json()
print(f"Created session: {session['id']}")

# Send message
message_data = {
    "content": "Write a Python function to calculate fibonacci numbers"
}
response = requests.post(
    f"{API_BASE}/sessions/{session['id']}/messages",
    headers=HEADERS,
    json=message_data
)
message = response.json()
print(f"Message sent: {message['id']}")
```

</details>

## Documentation

| Document | Description |
|----------|-------------|
| [API Reference](docs/API.md) | Complete REST API documentation |
| [Architecture](docs/architecture.md) | System design and structure |
| [Deployment Guide](docs/deployment.md) | Production deployment instructions |
| [Docker Guide](docker/README.md) | Container deployment guide |

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

See [NeuronDB documentation](../NeuronDB/README.md) for installation instructions.

## Security

- API key authentication required for all API endpoints
- Rate limiting configured per API key
- Database credentials stored securely via environment variables
- Supports TLS/SSL for encrypted connections
- Non-root user in Docker containers

See [Deployment Guide](docs/deployment.md) for security best practices.

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

- **Documentation**: [Component Documentation](../README.md)
- **GitHub Issues**: [Report Issues](https://github.com/neurondb/NeurondB/issues)
- **Email**: support@neurondb.ai

## License

See [LICENSE](../LICENSE) file for license information.

---

<div align="center">

[⬆ Back to Top](#neuronagent)

</div>
