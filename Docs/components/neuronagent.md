# 🤖 NeuronAgent

<div align="center">

**AI agent runtime system with REST API and WebSocket endpoints**

[![Status](https://img.shields.io/badge/status-stable-brightgreen)](.)
[![API](https://img.shields.io/badge/API-REST%20%7C%20WebSocket-blue)](.)
[![Tools](https://img.shields.io/badge/tools-20+-green)](.)

</div>

---

> [!TIP]
> NeuronAgent provides a complete platform for building autonomous AI agents. It includes persistent memory, tool execution, and multi-agent collaboration.

---

## 📋 What It Is

NeuronAgent is an AI agent runtime system providing REST API and WebSocket endpoints for building autonomous agent applications.

| Feature | Description | Status |
|---------|-------------|--------|
| **Agent Runtime** | Complete state machine for autonomous task execution with persistent memory | ✅ Stable |
| **REST API** | Full CRUD API for agents, sessions, messages, and advanced features | ✅ Stable |
| **WebSocket Support** | Real-time streaming agent responses | ✅ Stable |
| **Tool System** | Extensible tool registry with 20+ built-in tools | ✅ Stable |
| **Multi-Agent Collaboration** | Agent-to-agent communication and task delegation | ✅ Stable |
| **Workflow Engine** | DAG-based workflow execution with human-in-the-loop support | ✅ Stable |
| **Memory Management** | HNSW-based vector search for long-term memory with hierarchical organization | ✅ Stable |
| **Integration** | Direct integration with NeuronDB for embeddings, LLM, and vector operations | ✅ Stable |

## 🎯 Key Features & Modules

<details>
<summary><strong>⚙️ Core Agent Runtime</strong></summary>

| Feature | Description | Status |
|---------|-------------|--------|
| **Agent State Machine** | Complete execution engine with state management | ✅ Stable |
| **Session Management** | Multi-session support with caching and cleanup | ✅ Stable |
| **Context Management** | Intelligent context loading from messages and memory | ✅ Stable |
| **Prompt Engineering** | Advanced prompt construction with templating | ✅ Stable |
| **LLM Integration** | Integration with NeuronDB LLM functions (OpenAI, HuggingFace) | ✅ Stable |

</details>

<details>
<summary><strong>🧠 Memory & Knowledge</strong></summary>

| Feature | Description | Status |
|---------|-------------|--------|
| **Long-term Memory** | HNSW-based vector search for context retrieval | ✅ Stable |
| **Hierarchical Memory** | Multi-level memory organization for better recall | ✅ Stable |
| **Memory Promotion** | Background worker for promoting important memories | ✅ Stable |
| **Event Streaming** | Real-time event capture and summarization | ✅ Stable |

</details>

<details>
<summary><strong>🔧 Tool System (20+ Tools)</strong></summary>

| Category | Tools | Description | Status |
|----------|-------|-------------|--------|
| **Core Tools** | SQL, HTTP, Code, Shell | SQL (read-only), HTTP (with allowlist), Code (sandboxed), Shell (whitelisted) | ✅ Stable |
| **Browser Tool** | Browser | Web automation with Playwright for DOM interaction and navigation | ✅ Stable |
| **Filesystem Tool** | Filesystem | Virtual filesystem integration for file operations | ✅ Stable |
| **Memory Tool** | Memory | Direct memory manipulation and retrieval | ✅ Stable |
| **Collaboration Tool** | Collaboration | Multi-agent communication and task delegation | ✅ Stable |
| **NeuronDB Tools** | RAG, Hybrid Search, Reranking, Vector, ML, Analytics, Visualization | Complete NeuronDB integration | ✅ Stable |
| **Multimodal Tool** | Multimodal | Image and multimedia processing | ✅ Stable |
| **Tool Registry** | Custom Tools | Extensible system for registering custom tools | ✅ Stable |

</details>

<details>
<summary><strong>👥 Multi-Agent Collaboration</strong></summary>

| Feature | Description | Status |
|---------|-------------|--------|
| **Agent Delegation** | Delegate tasks to specialized agents | ✅ Stable |
| **Inter-Agent Communication** | Message passing between agents | ✅ Stable |
| **Workspace Management** | Shared workspaces for collaborative agents | ✅ Stable |
| **Sub-Agents** | Hierarchical agent structures for complex tasks | ✅ Stable |
| **Collaboration API** | REST endpoints for managing agent collaborations | ✅ Stable |

</details>

<details>
<summary><strong>🔄 Workflow Engine</strong></summary>

| Feature | Description | Status |
|---------|-------------|--------|
| **DAG Workflows** | Directed acyclic graph workflow execution | ✅ Stable |
| **Workflow Steps** | Agent, tool, HTTP, approval, and conditional steps | ✅ Stable |
| **Human-in-the-Loop (HITL)** | Approval gates and feedback loops | ✅ Stable |
| **Idempotency** | Idempotent step execution with key-based caching | ✅ Stable |
| **Retries** | Configurable retry logic for workflow steps | ✅ Stable |
| **Workflow API** | Complete CRUD API for workflows and executions | ✅ Stable |

</details>

### Planning & Task Management
- **LLM-Based Planning**: Advanced planning with task decomposition
- **Task Plans**: Multi-step plan creation and execution
- **Async Tasks**: Background task execution with job queue
- **Task Notifications**: Alerts and notifications for task events
- **Plans API**: Endpoints for creating, managing, and executing plans

<details>
<summary><strong>📊 Quality & Evaluation</strong></summary>

| Feature | Description | Status |
|---------|-------------|--------|
| **Reflections** | Agent self-reflection and quality assessment | ✅ Stable |
| **Quality Scoring** | Automated quality scoring for agent responses | ✅ Stable |
| **Evaluation Framework** | Built-in evaluation system for agent performance | ✅ Stable |
| **Verification Agent** | Dedicated agent for verifying outputs | ✅ Stable |
| **Execution Snapshots** | Capture and replay agent execution states | ✅ Stable |

</details>

<details>
<summary><strong>💰 Budget & Cost Management</strong></summary>

| Feature | Description | Status |
|---------|-------------|--------|
| **Cost Tracking** | Real-time cost tracking for LLM usage | ✅ Stable |
| **Budget Management** | Per-agent and per-session budget controls | ✅ Stable |
| **Budget Alerts** | Configurable alerts for budget thresholds | ✅ Stable |
| **Budget API** | Complete API for managing budgets and tracking costs | ✅ Stable |

</details>

<details>
<summary><strong>👤 Human-in-the-Loop (HITL)</strong></summary>

| Feature | Description | Status |
|---------|-------------|--------|
| **Approval Workflows** | Human approval gates in workflows | ✅ Stable |
| **Feedback System** | Collect and integrate human feedback | ✅ Stable |
| **Alert Preferences** | Configurable alert preferences for users | ✅ Stable |
| **HumanLoop API** | Endpoints for approvals and feedback | ✅ Stable |

</details>

<details>
<summary><strong>📜 Versioning & History</strong></summary>

| Feature | Description | Status |
|---------|-------------|--------|
| **Version Management** | Version control for agents and configurations | ✅ Stable |
| **Execution Replay** | Replay previous agent executions | ✅ Stable |
| **Execution Snapshots** | Capture and restore agent states | ✅ Stable |
| **Versions API** | API for managing versions and viewing history | ✅ Stable |

</details>

<details>
<summary><strong>📊 Observability & Monitoring</strong></summary>

| Feature | Description | Status |
|---------|-------------|--------|
| **Prometheus Metrics** | Comprehensive metrics export | ✅ Stable |
| **Structured Logging** | JSON-formatted logs with context | ✅ Stable |
| **Tracing** | Distributed tracing support | ✅ Stable |
| **Debugging Tools** | Advanced debugging capabilities | ✅ Stable |
| **Event Streaming** | Real-time event capture and analysis | ✅ Stable |

</details>

<details>
<summary><strong>🔒 Security & Safety</strong></summary>

| Feature | Description | Status |
|---------|-------------|--------|
| **API Key Authentication** | Bcrypt-hashed API keys with rate limiting | ✅ Stable |
| **RBAC** | Role-based access control with fine-grained permissions | ✅ Stable |
| **Data Permissions** | Per-principal data access controls | ✅ Stable |
| **Tool Permissions** | Granular tool access permissions | ✅ Stable |
| **Audit Logging** | Comprehensive audit trail for all operations | ✅ Stable |
| **Safety Moderation** | Content moderation and safety checks | ✅ Stable |

</details>

### Integrations & Connectors
- **S3 Connector**: AWS S3 integration for storage
- **GitHub Connector**: GitHub API integration
- **GitLab Connector**: GitLab API integration
- **Slack Connector**: Slack webhook integration
- **Webhooks**: Outbound webhook support for events
- **Secrets Management**: AWS Secrets Manager and HashiCorp Vault integration

### Storage & Persistence
- **Database Storage**: PostgreSQL-based persistence
- **S3 Storage**: Object storage for large files
- **Multimodal Storage**: Specialized storage for images and media
- **Session Caching**: Redis-compatible session caching

### Background Workers
- **Job Queue**: PostgreSQL-based job queue with SKIP LOCKED
- **Worker Pool**: Configurable worker pool with graceful shutdown
- **Async Task Worker**: Background execution of async tasks
- **Memory Promoter**: Promotes important memories to long-term storage
- **Verifier Worker**: Background verification of agent outputs

### Advanced Features
- **Batch Operations**: Batch processing for multiple requests
- **Virtual Filesystem**: Isolated filesystem for agents
- **Token Counting**: Accurate token counting for cost tracking
- **Relationship Management**: Manage relationships between entities
- **Advanced Handlers**: Specialized handlers for complex operations

---

## 📚 Documentation

| Resource | Location | Description |
|----------|----------|-------------|
| **Main README** | `NeuronAgent/README.md` | Component overview |
| **API Reference** | `NeuronAgent/docs/api.md` | Complete API documentation |
| **Architecture** | `NeuronAgent/docs/architecture.md` | Architecture details |
| **Deployment** | `NeuronAgent/docs/deployment.md` | Deployment guide |
| **OpenAPI Spec** | `NeuronAgent/openapi/openapi.yaml` | OpenAPI 3.0 specification |
| **Official Docs** | [https://www.neurondb.ai/docs/neuronagent](https://www.neurondb.ai/docs/neuronagent) | Online documentation |

---

## 🐳 Docker

| Service | Description |
|---------|-------------|
| **neuronagent** | Main service (CPU) |
| **neuronagent-cuda** | NVIDIA GPU variant |
| **neuronagent-rocm** | AMD GPU variant |
| **neuronagent-metal** | Apple Silicon GPU variant |

> [!TIP]
> **Docker Setup:** See [`NeuronAgent/docker/README.md`](../../NeuronAgent/docker/README.md) for detailed Docker deployment instructions.

---

## 🚀 Quick Start

<details>
<summary><strong>✅ Minimal Verification</strong></summary>

```bash
# Check health endpoint
curl -sS http://localhost:8080/health

# Expected output: {"status":"ok"}
```

</details>

<details>
<summary><strong>🤖 Create Agent</strong></summary>

```bash
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_agent",
    "profile": "general-assistant",
    "tools": ["sql", "http", "browser"]
  }'
```

> [!NOTE]
> **API Key:** Replace `YOUR_API_KEY` with your actual API key. See [NeuronAgent API Reference](reference/neuronagent-api.md) for authentication details.

</details>

> [!TIP]
> **Complete Setup:** For complete setup instructions, see [`NeuronAgent/README.md`](../../NeuronAgent/README.md).

---

## 🔗 Related Documentation

| Document | Description |
|----------|-------------|
| **[Components Overview](README.md)** | All components overview |
| **[API Reference](reference/neuronagent-api.md)** | Complete API reference |
| **[Architecture Guide](../internals/neuronagent-architecture.md)** | Internal architecture |
| **[Deployment Guide](../deployment/docker.md)** | Docker deployment |

---

<div align="center">

[⬆ Back to Top](#-neuronagent) · [📚 Components Index](README.md) · [📚 Main Documentation](../../README.md)

</div>
