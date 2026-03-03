# 🤖 NeuronAgent

<div align="center">

**AI agent runtime system with REST API and WebSocket endpoints**

[![Status](https://img.shields.io/badge/status-stable-brightgreen)](.)
[![API](https://img.shields.io/badge/API-REST%20%7C%20WebSocket-blue)](.)
[![Tools](https://img.shields.io/badge/tools-16+-green)](.)

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
| **Tool System** | Extensible tool registry with 16+ built-in tools (extensible via custom registration) | ✅ Stable |
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
<summary><strong>🔧 Tool System (16+ Base Tools, Extensible)</strong></summary>

| Category | Tools | Description | Status |
|----------|-------|-------------|--------|
| **Core Tools** | SQL, HTTP, Code, Shell, Browser, Visualization | SQL (read-only queries), HTTP (with allowlist), Code (sandboxed execution), Shell (whitelisted commands), Browser (Playwright web automation), Visualization (data visualization) | ✅ Stable |
| **Virtual Filesystem Tool** | Filesystem | Isolated virtual filesystem for secure file operations per agent/session | ✅ Stable |
| **Memory Tool** | Memory | Direct hierarchical memory manipulation, retrieval, and management | ✅ Stable |
| **Collaboration Tool** | Collaboration | Multi-agent communication, task delegation, and workspace coordination | ✅ Stable |
| **NeuronDB Integration Tools** | ML, Vector, RAG, Analytics, Hybrid Search, Reranking | Complete NeuronDB integration: ML model training/prediction, vector search, RAG operations, analytics, hybrid search, and reranking | ✅ Stable |
| **Multimodal Tool** | Multimodal | Image and multimedia processing with embedding generation | ✅ Stable |
| **Tool Registry** | Custom Tools | Extensible system for registering custom tools with JSON Schema validation | ✅ Stable |

**Total**: 16+ base tools (SQL, HTTP, Code, Shell, Browser, Visualization, Filesystem, Memory, Collaboration, ML, Vector, RAG, Analytics, Hybrid Search, Reranking, Multimodal), with support for custom tool registration.

</details>

<details>
<summary><strong>👥 Multi-Agent Collaboration</strong></summary>

| Feature | Description | Status |
|---------|-------------|--------|
| **Agent Delegation** | Delegate tasks to specialized agents with automatic routing | ✅ Stable |
| **Inter-Agent Communication** | Message passing between agents with structured protocols | ✅ Stable |
| **Workspace Management** | Shared workspaces for collaborative agents with isolation and permissions | ✅ Stable |
| **Sub-Agents** | Hierarchical agent structures for complex multi-level task decomposition | ✅ Stable |
| **Task Coordination** | Coordinate parallel and sequential task execution across agents | ✅ Stable |
| **Collaboration API** | REST endpoints for managing agent collaborations, workspaces, and delegations | ✅ Stable |
| **Agent Discovery** | Discover and select appropriate agents for task delegation | ✅ Stable |
| **Shared Context** | Shared context and state management across collaborating agents | ✅ Stable |

</details>

<details>
<summary><strong>🔄 Workflow Engine</strong></summary>

| Feature | Description | Status |
|---------|-------------|--------|
| **DAG Workflows** | Directed acyclic graph workflow execution with dependency resolution | ✅ Stable |
| **Workflow Steps** | Multiple step types: agent (execute agent), tool (execute tool), HTTP (HTTP requests), approval (human approval gates), conditional (branching logic) | ✅ Stable |
| **Dependency Management** | Step dependencies with automatic parallel execution where possible | ✅ Stable |
| **Input/Output Mapping** | Step input/output mapping with data transformation | ✅ Stable |
| **Compensation Steps** | Rollback and compensation logic for failed workflow steps | ✅ Stable |
| **Human-in-the-Loop (HITL)** | Approval gates with email/webhook notifications and feedback loops | ✅ Stable |
| **Idempotency** | Idempotent step execution with key-based caching to prevent duplicate execution | ✅ Stable |
| **Retries** | Configurable retry logic with exponential backoff for workflow steps | ✅ Stable |
| **Workflow Scheduling** | Schedule workflows for future execution with cron-like syntax | ✅ Stable |
| **Workflow API** | Complete CRUD API for workflows, executions, and schedules | ✅ Stable |
| **Execution Monitoring** | Real-time workflow execution monitoring and status tracking | ✅ Stable |

</details>

<details>
<summary><strong>📋 Planning & Task Management</strong></summary>

| Feature | Description | Status |
|---------|-------------|--------|
| **LLM-Based Planning** | Advanced planning with LLM-powered task decomposition and strategy generation | ✅ Stable |
| **Task Decomposition** | Automatic breakdown of complex tasks into manageable sub-tasks | ✅ Stable |
| **Task Plans** | Multi-step plan creation, validation, and execution with dependency tracking | ✅ Stable |
| **Plan Templates** | Reusable plan templates for common task patterns | ✅ Stable |
| **Async Tasks** | Background task execution with PostgreSQL-based job queue | ✅ Stable |
| **Task Prioritization** | Priority-based task scheduling and execution | ✅ Stable |
| **Task Notifications** | Alerts and notifications for task events (start, complete, failure) | ✅ Stable |
| **Plans API** | Complete REST API for creating, managing, executing, and monitoring plans | ✅ Stable |
| **Plan Execution Tracking** | Real-time tracking of plan execution progress and status | ✅ Stable |

</details>

<details>
<summary><strong>📊 Quality & Evaluation</strong></summary>

| Feature | Description | Status |
|---------|-------------|--------|
| **Reflections** | Agent self-reflection and quality assessment with LLM-powered analysis | ✅ Stable |
| **Quality Scoring** | Automated quality scoring for agent responses using multiple metrics | ✅ Stable |
| **Evaluation Framework** | Built-in evaluation system for agent performance with configurable metrics | ✅ Stable |
| **Performance Metrics** | Comprehensive performance metrics: accuracy, relevance, completeness, latency | ✅ Stable |
| **Verification Agent** | Dedicated verification agent for validating and cross-checking outputs | ✅ Stable |
| **Evaluation API** | REST API for running evaluations, viewing results, and comparing agent performance | ✅ Stable |
| **Execution Snapshots** | Capture and replay agent execution states for debugging and analysis | ✅ Stable |
| **Quality Reports** | Automated quality reports with trends and recommendations | ✅ Stable |

</details>

<details>
<summary><strong>💰 Budget & Cost Management</strong></summary>

| Feature | Description | Status |
|---------|-------------|--------|
| **Cost Tracking** | Real-time cost tracking for LLM usage with per-request, per-session, and per-agent aggregation | ✅ Stable |
| **Token Counting** | Accurate token counting for input/output with model-specific tokenizers | ✅ Stable |
| **Cost Analytics** | Detailed cost analytics with breakdowns by agent, session, model, and time period | ✅ Stable |
| **Budget Management** | Per-agent and per-session budget controls with hard and soft limits | ✅ Stable |
| **Budget Alerts** | Configurable alerts for budget thresholds via email and webhooks | ✅ Stable |
| **Cost Forecasting** | Predictive cost forecasting based on usage patterns | ✅ Stable |
| **Budget API** | Complete REST API for managing budgets, tracking costs, and viewing analytics | ✅ Stable |
| **Cost Optimization** | Recommendations for cost optimization based on usage patterns | ✅ Stable |

</details>

<details>
<summary><strong>👤 Human-in-the-Loop (HITL)</strong></summary>

| Feature | Description | Status |
|---------|-------------|--------|
| **Approval Workflows** | Human approval gates in workflows with configurable approval rules | ✅ Stable |
| **Approval Notifications** | Email and webhook notifications for pending approvals with approval links | ✅ Stable |
| **Approval Timeouts** | Configurable timeouts for approvals with automatic escalation | ✅ Stable |
| **Feedback System** | Collect and integrate human feedback with structured feedback forms | ✅ Stable |
| **Feedback Integration** | Automatic integration of feedback into agent learning and improvement | ✅ Stable |
| **Alert Preferences** | Configurable alert preferences for users with multiple notification channels | ✅ Stable |
| **HumanLoop API** | Complete REST API for approvals, feedback, and alert management | ✅ Stable |
| **Approval History** | Complete audit trail of all approvals and feedback | ✅ Stable |

</details>

<details>
<summary><strong>📜 Versioning & History</strong></summary>

| Feature | Description | Status |
|---------|-------------|--------|
| **Version Management** | Version control for agents, configurations, and prompts with semantic versioning | ✅ Stable |
| **Version Comparison** | Compare versions side-by-side with diff visualization | ✅ Stable |
| **Version Rollback** | Rollback to previous versions with one-click restore | ✅ Stable |
| **Execution Replay** | Replay previous agent executions with full state reconstruction | ✅ Stable |
| **Execution History** | Complete execution history with search and filtering | ✅ Stable |
| **Execution Snapshots** | Capture and restore agent states at any point in execution | ✅ Stable |
| **State Diff** | View differences between execution states for debugging | ✅ Stable |
| **Versions API** | Complete REST API for managing versions, viewing history, and replaying executions | ✅ Stable |

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

<details>
<summary><strong>🔌 Integrations & Connectors</strong></summary>

| Connector | Description | Status |
|-----------|-------------|--------|
| **S3 Connector** | AWS S3 integration for object storage with automatic file management | ✅ Stable |
| **GitHub Connector** | GitHub API integration for repository access, issue management, and webhooks | ✅ Stable |
| **GitLab Connector** | GitLab API integration for repository access, CI/CD, and project management | ✅ Stable |
| **Slack Connector** | Slack webhook integration for notifications and bot interactions | ✅ Stable |
| **Webhooks** | Outbound webhook support for events with retry logic and authentication | ✅ Stable |
| **Secrets Management** | AWS Secrets Manager and HashiCorp Vault integration for secure credential storage | ✅ Stable |
| **Email Service** | SMTP email service for notifications and alerts | ✅ Stable |
| **Custom Connectors** | Extensible connector framework for custom integrations | ✅ Stable |

</details>

### Storage & Persistence
- **Database Storage**: PostgreSQL-based persistence
- **S3 Storage**: Object storage for large files
- **Multimodal Storage**: Specialized storage for images and media
- **Session Caching**: Redis-compatible session caching

<details>
<summary><strong>⚙️ Background Workers</strong></summary>

| Worker | Description | Status |
|--------|-------------|--------|
| **Job Queue** | PostgreSQL-based job queue with SKIP LOCKED for efficient concurrent processing | ✅ Stable |
| **Worker Pool** | Configurable worker pool with graceful shutdown and health monitoring | ✅ Stable |
| **Async Task Worker** | Background execution of async tasks with priority queuing | ✅ Stable |
| **Memory Promoter** | Promotes important memories to long-term storage based on usage patterns | ✅ Stable |
| **Verifier Worker** | Background verification of agent outputs with quality checks | ✅ Stable |
| **Cleanup Worker** | Automatic cleanup of expired sessions, old messages, and temporary data | ✅ Stable |
| **Metrics Worker** | Background collection and aggregation of metrics and statistics | ✅ Stable |
| **Notification Worker** | Background processing of email and webhook notifications | ✅ Stable |

</details>

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
| **Main README** | [neuron-agent repo](https://github.com/neurondb/neuron-agent) | Component overview |
| **API Reference** | neuron-agent repo `docs/api.md` | Complete API documentation |
| **Architecture** | neuron-agent repo `docs/architecture.md` | Architecture details |
| **Deployment** | neuron-agent repo `docs/deployment.md` | Deployment guide |
| **OpenAPI Spec** | neuron-agent repo `openapi/openapi.yaml` | OpenAPI 3.0 specification |
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
> **Docker Setup:** See the **neuron-agent** repo for Docker deployment instructions.

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
> **Complete Setup:** For complete setup instructions, see the **neuron-agent** repo.

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
