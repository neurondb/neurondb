# 📚 API Reference

<div align="center">

**Complete API reference for the NeuronDB ecosystem**

[![APIs](https://img.shields.io/badge/APIs-3-blue)](.)
[![Status](https://img.shields.io/badge/status-stable-brightgreen)](.)

</div>

---

> [!TIP]
> **Component APIs:** Each component has its own API. See component-specific documentation for detailed guides and examples.

---

## 📑 Table of Contents

| Section | Description |
|---------|-------------|
| [NeuronDesktop API](#neurondesktop-api) | Web UI REST and WebSocket API |
| [NeuronAgent API](#neuronagent-api) | Agent runtime REST and WebSocket API |
| [NeuronMCP API](#neuronmcp-api) | MCP protocol tools and resources |
| [Authentication](#authentication) | Authentication methods |
| [Error Handling](#error-handling) | Error codes and handling |

---

## 🖥️ NeuronDesktop API

<details>
<summary><strong>📡 API Overview</strong></summary>

**Base URL:** `http://localhost:8081/api/v1`

| Feature | Description | Status |
|---------|-------------|--------|
| **REST API** | HTTP REST endpoints | ✅ Stable |
| **WebSocket** | Real-time WebSocket connections | ✅ Stable |
| **Authentication** | Local auth and OIDC support | ✅ Stable |
| **Profiles** | Database connection profiles | ✅ Stable |

> [!NOTE]
> **Default Port:** NeuronDesktop API runs on port `8081` by default. The web UI runs on port `3000`.

</details>

### Authentication

#### Register (Local Auth)
```http
POST /auth/register
Content-Type: application/json

{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "secure-password"
}
```

#### Login (Local Auth)
```http
POST /auth/login
Content-Type: application/json

{
  "username": "john_doe",
  "password": "secure-password"
}
```

**Response**: Sets `access_token` and `refresh_token` cookies

#### OIDC Start
```http
GET /auth/oidc/start?redirect_uri=http://localhost:3000/callback
```

**Response**: Redirects to OIDC provider

#### OIDC Callback
```http
GET /auth/oidc/callback?code=...&state=...
```

**Response**: Sets cookies and redirects to `redirect_uri`

#### Refresh Token
```http
POST /auth/refresh
```

**Response**: New access and refresh tokens in cookies

#### Logout
```http
POST /auth/logout
```

#### Get Current User
```http
GET /auth/me
```

**Response**:
```json
{
  "id": "uuid",
  "username": "john_doe",
  "email": "john@example.com",
  "is_admin": false
}
```

### Profiles

#### List Profiles
```http
GET /profiles
```

**Response**:
```json
[
  {
    "id": "uuid",
    "name": "Default",
    "neurondb_dsn": "postgresql://...",
    "mcp_config": {...},
    "agent_endpoint": "http://localhost:8080",
    "is_default": true
  }
]
```

#### Create Profile
```http
POST /profiles
Content-Type: application/json

{
  "name": "Production",
  "neurondb_dsn": "postgresql://...",
  "mcp_config": {...},
  "agent_endpoint": "http://localhost:8080",
  "agent_api_key": "key"
}
```

#### Get Profile
```http
GET /profiles/{id}
```

#### Update Profile
```http
PUT /profiles/{id}
Content-Type: application/json

{
  "name": "Updated Name",
  "neurondb_dsn": "postgresql://..."
}
```

#### Delete Profile
```http
DELETE /profiles/{id}
```

#### Clone Profile
```http
POST /profiles/{id}/clone
Content-Type: application/json

{
  "name": "Cloned Profile"
}
```

#### Validate Profile
```http
POST /profiles/validate
Content-Type: application/json

{
  "neurondb_dsn": "postgresql://...",
  "mcp_config": {...},
  "agent_endpoint": "http://localhost:8080"
}
```

#### Health Check Profile
```http
GET /profiles/{id}/health
```

**Response**:
```json
{
  "status": "healthy",
  "neurondb": "connected",
  "mcp": "connected",
  "agent": "connected"
}
```

### NeuronDB Operations

#### List Collections
```http
GET /profiles/{profile_id}/neurondb/collections
```

**Response**:
```json
[
  {
    "name": "documents",
    "schema": "public",
    "vector_dim": 1536,
    "index_type": "hnsw"
  }
]
```

#### Search
```http
POST /profiles/{profile_id}/neurondb/search
Content-Type: application/json

{
  "collection": "documents",
  "query_vector": [0.1, 0.2, ...],
  "limit": 10,
  "distance_type": "cosine"
}
```

**Response**:
```json
[
  {
    "id": "uuid",
    "content": "text",
    "distance": 0.123,
    "metadata": {...}
  }
]
```

#### Execute SQL (Read-Only)
```http
POST /profiles/{profile_id}/neurondb/sql
Content-Type: application/json

{
  "query": "SELECT * FROM documents LIMIT 10"
}
```

**Response**:
```json
[
  {"id": "...", "content": "..."},
  ...
]
```

#### Execute SQL (Full - Admin Only)
```http
POST /profiles/{profile_id}/neurondb/sql/execute
Content-Type: application/json

{
  "query": "CREATE TABLE ..."
}
```

### Dataset Ingestion

#### Ingest Dataset
```http
POST /profiles/{profile_id}/neurondb/ingest
Content-Type: application/json

{
  "source_type": "file",
  "source_path": "/path/to/data.csv",
  "format": "csv",
  "table_name": "documents",
  "auto_embed": true,
  "embedding_model": "text-embedding-3-small",
  "create_index": true
}
```

**Response**:
```json
{
  "job_id": "ingest_1234567890",
  "status": "queued",
  "table_name": "documents",
  "created_at": "2025-01-01T12:00:00Z"
}
```

#### Get Ingestion Status
```http
GET /profiles/{profile_id}/neurondb/ingest/{job_id}
```

**Response**:
```json
{
  "job_id": "ingest_1234567890",
  "status": "completed",
  "progress": 100,
  "rows_ingested": 1000
}
```

#### List Ingestion Jobs
```http
GET /profiles/{profile_id}/neurondb/ingest
```

### Model Management

#### List Models
```http
GET /profiles/{profile_id}/llm-models
```

**Response**:
```json
[
  {
    "id": "uuid",
    "name": "gpt-4",
    "provider": "openai",
    "model_type": "chat",
    "api_key_set": true,
    "config": {...}
  }
]
```

#### Add Model
```http
POST /profiles/{profile_id}/llm-models
Content-Type: application/json

{
  "name": "gpt-4",
  "provider": "openai",
  "model_type": "chat",
  "config": {
    "temperature": 0.7,
    "max_tokens": 2000
  }
}
```

#### Set Model API Key
```http
POST /profiles/{profile_id}/llm-models/{model_name}/key
Content-Type: application/json

{
  "api_key": "sk-..."
}
```

#### Get Model Info
```http
GET /profiles/{profile_id}/llm-models/{model_id}
```

#### Delete Model
```http
DELETE /profiles/{profile_id}/llm-models/{model_id}
```

### Observability

#### Database Health
```http
GET /profiles/{profile_id}/observability/db-health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "PostgreSQL 15.0",
  "connections": 10,
  "uptime": "7 days",
  "metrics": {
    "extension": "loaded",
    "extension_version": "1.0.0"
  }
}
```

#### Index Health
```http
GET /profiles/{profile_id}/observability/indexes
```

**Response**:
```json
[
  {
    "table_name": "documents",
    "index_name": "hnsw_documents_embedding",
    "index_type": "HNSW",
    "status": "healthy",
    "size": "128 MB"
  }
]
```

#### Worker Status
```http
GET /profiles/{profile_id}/observability/workers
```

**Response**:
```json
[
  {
    "worker_name": "embedding_worker",
    "status": "running",
    "last_run": "2025-01-01T12:00:00Z",
    "jobs_processed": 1000,
    "errors": 0
  }
]
```

#### Usage Statistics
```http
GET /profiles/{profile_id}/observability/usage
```

**Response**:
```json
{
  "total_requests": 10000,
  "errors": 50,
  "avg_duration_ms": 123.45,
  "total_tokens": 5000000
}
```

### MCP Operations

#### List Connections
```http
GET /mcp/connections
```

#### List Tools
```http
GET /profiles/{profile_id}/mcp/tools
```

**Response**:
```json
{
  "tools": [
    {
      "name": "search_documents",
      "description": "Search documents",
      "input_schema": {...}
    }
  ]
}
```

#### Call Tool
```http
POST /profiles/{profile_id}/mcp/tools/call
Content-Type: application/json

{
  "name": "search_documents",
  "arguments": {
    "query": "test"
  }
}
```

#### Test MCP Config
```http
POST /mcp/test
Content-Type: application/json

{
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-pg"],
  "env": {
    "DATABASE_URL": "postgresql://..."
  }
}
```

### Agent Operations

#### List Agents
```http
GET /profiles/{profile_id}/agent/agents
```

**Response**:
```json
[
  {
    "id": "uuid",
    "name": "my-agent",
    "description": "Helpful assistant",
    "system_prompt": "You are helpful",
    "model_name": "gpt-4",
    "enabled_tools": ["search", "calculator"]
  }
]
```

#### Create Agent
```http
POST /profiles/{profile_id}/agent/agents
Content-Type: application/json

{
  "name": "my-agent",
  "description": "Helpful assistant",
  "system_prompt": "You are helpful",
  "model_name": "gpt-4",
  "enabled_tools": ["search", "calculator"],
  "config": {}
}
```

#### Get Agent
```http
GET /profiles/{profile_id}/agent/agents/{agent_id}
```

#### Update Agent
```http
PUT /profiles/{profile_id}/agent/agents/{agent_id}
Content-Type: application/json

{
  "name": "updated-name",
  "system_prompt": "Updated prompt"
}
```

#### Delete Agent
```http
DELETE /profiles/{profile_id}/agent/agents/{agent_id}
```

#### Create Session
```http
POST /profiles/{profile_id}/agent/sessions
Content-Type: application/json

{
  "agent_id": "uuid",
  "external_user_id": "user-123"
}
```

**Response**:
```json
{
  "id": "session-uuid",
  "agent_id": "uuid",
  "external_user_id": "user-123",
  "created_at": "2025-01-01T12:00:00Z"
}
```

#### Send Message
```http
POST /profiles/{profile_id}/agent/sessions/{session_id}/messages
Content-Type: application/json

{
  "role": "user",
  "content": "Hello, agent!"
}
```

**Response**:
```json
{
  "id": "message-uuid",
  "session_id": "session-uuid",
  "role": "assistant",
  "content": "Hello! How can I help?",
  "created_at": "2025-01-01T12:00:00Z"
}
```

#### Get Messages
```http
GET /profiles/{profile_id}/agent/sessions/{session_id}/messages
```

### API Keys

#### Generate API Key
```http
POST /api-keys
Content-Type: application/json

{
  "name": "my-api-key",
  "rate_limit_per_min": 100,
  "roles": ["agent", "tool"]
}
```

**Response**:
```json
{
  "id": "uuid",
  "key": "nd_...",
  "name": "my-api-key",
  "created_at": "2025-01-01T12:00:00Z"
}
```

#### List API Keys
```http
GET /api-keys
```

#### Delete API Key
```http
DELETE /api-keys/{id}
```

### Audit Logs

#### List Audit Logs (Admin Only)
```http
GET /audit-logs?limit=100&offset=0
```

**Response**:
```json
[
  {
    "id": "uuid",
    "event_type": "api_key.created",
    "user_id": "uuid",
    "metadata": {...},
    "created_at": "2025-01-01T12:00:00Z"
  }
]
```

#### Get Audit Log
```http
GET /audit-logs/{id}
```

---

## NeuronAgent API

Base URL: `http://localhost:8080/api/v1`

### Authentication

All endpoints require an API key in the `Authorization` header:
```http
Authorization: Bearer nd_...
```

### Agents

#### List Agents
```http
GET /agents
```

#### Create Agent
```http
POST /agents
Content-Type: application/json

{
  "name": "my-agent",
  "system_prompt": "You are helpful",
  "model_name": "gpt-4",
  "enabled_tools": ["search"]
}
```

#### Get Agent
```http
GET /agents/{id}
```

#### Update Agent
```http
PUT /agents/{id}
```

#### Delete Agent
```http
DELETE /agents/{id}
```

### Sessions

#### Create Session
```http
POST /sessions
Content-Type: application/json

{
  "agent_id": "uuid",
  "external_user_id": "user-123"
}
```

#### Get Session
```http
GET /sessions/{id}
```

#### Send Message
```http
POST /sessions/{id}/messages
Content-Type: application/json

{
  "role": "user",
  "content": "Hello"
}
```

#### Get Messages
```http
GET /sessions/{id}/messages
```

---

## NeuronMCP API

NeuronMCP implements the Model Context Protocol (MCP) specification.

### Tools

Tools are discovered via MCP protocol:
```json
{
  "jsonrpc": "2.0",
  "method": "tools/list",
  "id": 1
}
```

### Resources

Resources are discovered via MCP protocol:
```json
{
  "jsonrpc": "2.0",
  "method": "resources/list",
  "id": 2
}
```

---

## Authentication

### Session-Based (OIDC)

1. Start OIDC flow: `GET /auth/oidc/start`
2. Redirect to OIDC provider
3. Callback: `GET /auth/oidc/callback`
4. Cookies set automatically
5. Use cookies for subsequent requests

### API Key-Based

Include API key in `Authorization` header:
```http
Authorization: Bearer nd_...
```

### JWT (Legacy)

Include JWT in `Authorization` header:
```http
Authorization: Bearer eyJhbGc...
```

---

## ⚠️ Error Handling

<details>
<summary><strong>📋 Error Response Format</strong></summary>

All errors follow a consistent format:

```json
{
  "error": "Error message",
  "code": "ERROR_CODE",
  "details": {...}
}
```

</details>

### HTTP Status Codes

| Code | Status | Description |
|------|--------|-------------|
| **200** | OK | Success |
| **201** | Created | Resource created |
| **400** | Bad Request | Invalid request |
| **401** | Unauthorized | Authentication required |
| **403** | Forbidden | Insufficient permissions |
| **404** | Not Found | Resource not found |
| **500** | Internal Server Error | Server error |

### Common Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `INVALID_REQUEST` | Request validation failed | 400 |
| `UNAUTHORIZED` | Authentication required | 401 |
| `FORBIDDEN` | Insufficient permissions | 403 |
| `NOT_FOUND` | Resource not found | 404 |
| `RATE_LIMIT_EXCEEDED` | Rate limit exceeded | 429 |
| `INTERNAL_ERROR` | Internal server error | 500 |

---

## 🚦 Rate Limiting

<details>
<summary><strong>📊 Default Rate Limits</strong></summary>

| Authentication Type | Rate Limit | Description |
|---------------------|------------|-------------|
| **API Keys** | 100 requests/minute | Standard API key authentication |
| **Sessions** | 1000 requests/minute | User session authentication |
| **Admin** | 10000 requests/minute | Administrative access |

</details>

### Rate Limit Headers

All responses include rate limit headers:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

> [!NOTE]
> **Rate Limit Exceeded:** When rate limit is exceeded, the API returns HTTP 429 with `RATE_LIMIT_EXCEEDED` error code.

---

## Pagination

List endpoints support pagination:
```http
GET /profiles?limit=10&offset=0
```

**Response**:
```json
{
  "data": [...],
  "pagination": {
    "limit": 10,
    "offset": 0,
    "total": 100,
    "has_more": true
  }
}
```

---

## WebSocket Endpoints

### Agent WebSocket
```http
GET /profiles/{profile_id}/agent/ws?session_id={session_id}
Upgrade: websocket
```

### MCP WebSocket
```http
GET /profiles/{profile_id}/mcp/ws
Upgrade: websocket
```

---

## SDKs

### Python SDK

```python
from neuronagent import NeuronAgentClient

client = NeuronAgentClient(
    base_url="http://localhost:8080",
    api_key="nd_..."
)

agent = client.agents.create_agent(
    name="my-agent",
    system_prompt="You are helpful",
    model_name="gpt-4"
)
```

### TypeScript SDK

```typescript
import { NeuronAgentClient } from '@neurondb/neuronagent'

const client = new NeuronAgentClient({
  baseURL: 'http://localhost:8080',
  apiKey: 'nd_...'
})

const agent = await client.agents.createAgent({
  name: 'my-agent',
  systemPrompt: 'You are helpful',
  modelName: 'gpt-4'
})
```

---

## Examples

See `examples/` directory for complete examples:
- `examples/basics/` - Basic usage
- `examples/rag-chatbot-pdfs/` - RAG chatbot
- `examples/llm_training/` - LLM training
- `examples/agent-tools/` - Agent tools

---

---

## 🔗 Related Documentation

| Document | Description |
|----------|-------------|
| **[NeuronAgent API](neuronagent-api.md)** | Complete NeuronAgent API reference |
| **[Top Functions](top_functions.md)** | Most commonly used SQL functions |
| **[Data Types](data-types.md)** | Complete data types reference |
| **[Component Documentation](../components/README.md)** | Component overviews |

---

## 💬 Support

| Resource | Link |
|---------|------|
| **Documentation** | `Docs/` directory |
| **GitHub Issues** | [Report Issues](https://github.com/neurondb/neurondb2/issues) |
| **Email Support** | support@neurondb.ai |

---

<div align="center">

[⬆ Back to Top](#-api-reference) · [📚 Reference Index](README.md) · [📚 Main Documentation](../../README.md)

</div>


