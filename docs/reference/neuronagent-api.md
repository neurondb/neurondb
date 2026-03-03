# NeuronAgent API Reference

## Overview

NeuronAgent provides a comprehensive REST API for managing AI agents, sessions, messages, tools, and advanced features. All API endpoints are versioned under `/api/v1` and require API key authentication.

## Base URL

```
http://localhost:8080/api/v1
```

## Authentication

All API requests require an API key in the `Authorization` header:

```
Authorization: Bearer YOUR_API_KEY
```

API keys can be generated using the `generate-key` command or through your NeuronAgent administration interface.

## API Endpoints

### Agents

#### Create Agent
`POST /api/v1/agents`

Creates a new agent with specified configuration.

**Request Body:**
```json
{
  "name": "research_agent",
  "description": "Agent for research tasks",
  "system_prompt": "You are a helpful research assistant.",
  "model_name": "gpt-4",
  "memory_table": "research_memory",
  "enabled_tools": ["sql", "http", "browser"],
  "config": {
    "temperature": 0.7,
    "max_tokens": 2000
  }
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "research_agent",
    "description": "Agent for research tasks",
    "system_prompt": "You are a helpful research assistant.",
    "model_name": "gpt-4",
    "memory_table": "research_memory",
    "enabled_tools": ["sql", "http", "browser"],
    "config": {
      "temperature": 0.7,
      "max_tokens": 2000
    }
  }'
```

**Response:** `201 Created`
```json
{
  "id": "uuid",
  "name": "research_agent",
  "description": "Agent for research tasks",
  "system_prompt": "You are a helpful research assistant.",
  "model_name": "gpt-4",
  "memory_table": "research_memory",
  "enabled_tools": ["sql", "http", "browser"],
  "config": {},
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-01-01T00:00:00Z"
}
```

#### List Agents
`GET /api/v1/agents?search=query`

Lists all agents, optionally filtered by search query.

**Example (curl):**
```bash
# List all agents
curl -X GET http://localhost:8080/api/v1/agents \
  -H "Authorization: Bearer YOUR_API_KEY"

# Search agents
curl -X GET "http://localhost:8080/api/v1/agents?search=research" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response:** `200 OK`
```json
[
  {
    "id": "uuid",
    "name": "research_agent",
    "description": "Agent for research tasks",
    "created_at": "2025-01-01T00:00:00Z"
  }
]
```

#### Get Agent
`GET /api/v1/agents/{id}`

Retrieves detailed information about a specific agent.

**Example (curl):**
```bash
curl -X GET http://localhost:8080/api/v1/agents/{agent_id} \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response:** `200 OK`

#### Update Agent
`PUT /api/v1/agents/{id}`

Updates an existing agent configuration.

**Request Body:** Same as Create Agent

**Response:** `200 OK`

#### Delete Agent
`DELETE /api/v1/agents/{id}`

Deletes an agent and all associated data.

**Response:** `204 No Content`

#### Clone Agent
`POST /api/v1/agents/{id}/clone`

Creates a copy of an existing agent with a new ID.

**Response:** `201 Created`

#### Generate Plan
`POST /api/v1/agents/{id}/plan`

Generates an execution plan for a given task.

**Request Body:**
```json
{
  "task": "Research machine learning algorithms"
}
```

**Response:** `200 OK`

#### Reflect on Response
`POST /api/v1/agents/{id}/reflect`

Submits agent response for reflection and improvement.

**Request Body:**
```json
{
  "session_id": "uuid",
  "message_id": 123,
  "reflection": "User feedback or reflection"
}
```

**Response:** `200 OK`

#### Delegate to Agent
`POST /api/v1/agents/{id}/delegate`

Delegates a task to another agent or sub-agent.

**Request Body:**
```json
{
  "target_agent_id": "uuid",
  "task": "Task description",
  "context": {}
}
```

**Response:** `200 OK`

#### Get Agent Metrics
`GET /api/v1/agents/{id}/metrics`

Retrieves performance metrics for an agent.

**Response:** `200 OK`
```json
{
  "messages_count": 100,
  "sessions_count": 10,
  "avg_response_time": 1.5,
  "token_usage": {
    "total": 50000,
    "prompt": 30000,
    "completion": 20000
  }
}
```

#### Get Agent Costs
`GET /api/v1/agents/{id}/costs`

Retrieves cost tracking information for an agent.

**Response:** `200 OK`

#### Agent Versions

##### List Agent Versions
`GET /api/v1/agents/{id}/versions`

Lists all versions of an agent.

**Response:** `200 OK`

##### Create Agent Version
`POST /api/v1/agents/{id}/versions`

Creates a new version of an agent.

**Response:** `201 Created`

##### Get Agent Version
`GET /api/v1/agents/{id}/versions/{version}`

Retrieves a specific agent version.

**Response:** `200 OK`

##### Activate Agent Version
`PUT /api/v1/agents/{id}/versions/{version}/activate`

Activates a specific agent version.

**Response:** `200 OK`

#### Agent Relationships

##### List Agent Relationships
`GET /api/v1/agents/{id}/relationships`

Lists relationships between agents.

**Response:** `200 OK`

##### Create Agent Relationship
`POST /api/v1/agents/{id}/relationships`

Creates a relationship between agents.

**Response:** `201 Created`

##### Delete Agent Relationship
`DELETE /api/v1/agents/{id}/relationships/{relationship_id}`

Deletes an agent relationship.

**Response:** `204 No Content`

### Sessions

#### Create Session
`POST /api/v1/sessions`

Creates a new conversation session with an agent.

**Request Body:**
```json
{
  "agent_id": "uuid",
  "external_user_id": "user123",
  "metadata": {
    "source": "web_app"
  }
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:8080/api/v1/sessions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "your-agent-id",
    "external_user_id": "user123",
    "metadata": {
      "source": "web_app"
    }
  }'
```

**Response:** `201 Created`

#### Get Session
`GET /api/v1/sessions/{id}`

Retrieves session details.

**Response:** `200 OK`

#### Update Session
`PUT /api/v1/sessions/{id}`

Updates session metadata.

**Response:** `200 OK`

#### Delete Session
`DELETE /api/v1/sessions/{id}`

Deletes a session and all associated messages.

**Response:** `204 No Content`

#### List Sessions
`GET /api/v1/agents/{agent_id}/sessions`

Lists all sessions for an agent.

**Response:** `200 OK`

### Messages

#### Send Message
`POST /api/v1/sessions/{session_id}/messages`

Sends a message to an agent in a session.

**Request Body:**
```json
{
  "role": "user",
  "content": "What is machine learning?",
  "stream": false,
  "metadata": {}
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:8080/api/v1/sessions/{session_id}/messages \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "role": "user",
    "content": "What is machine learning?",
    "stream": false,
    "metadata": {}
  }'
```

**Response:** `200 OK`
```json
{
  "id": 123,
  "session_id": "uuid",
  "role": "assistant",
  "content": "Machine learning is...",
  "token_count": 150,
  "created_at": "2025-01-01T00:00:00Z"
}
```

#### Get Messages
`GET /api/v1/sessions/{session_id}/messages`

Retrieves message history for a session.

**Query Parameters:**
- `limit` (optional): Maximum number of messages to return
- `offset` (optional): Offset for pagination

**Response:** `200 OK`

#### Get Message
`GET /api/v1/messages/{id}`

Retrieves a specific message.

**Response:** `200 OK`

#### Update Message
`PUT /api/v1/messages/{id}`

Updates message content or metadata.

**Response:** `200 OK`

#### Delete Message
`DELETE /api/v1/messages/{id}`

Deletes a message.

**Response:** `204 No Content`

### Tools

#### List Tools
`GET /api/v1/tools`

Lists all available tools.

**Response:** `200 OK`

#### Create Tool
`POST /api/v1/tools`

Registers a new custom tool.

**Request Body:**
```json
{
  "name": "custom_tool",
  "description": "Custom tool description",
  "handler_type": "http",
  "handler_config": {},
  "schema": {
    "type": "object",
    "properties": {
      "url": {"type": "string"}
    }
  }
}
```

**Response:** `201 Created`

#### Get Tool
`GET /api/v1/tools/{name}`

Retrieves tool details.

**Response:** `200 OK`

#### Update Tool
`PUT /api/v1/tools/{name}`

Updates tool configuration.

**Response:** `200 OK`

#### Delete Tool
`DELETE /api/v1/tools/{name}`

Deletes a tool.

**Response:** `204 No Content`

#### Get Tool Analytics
`GET /api/v1/tools/{name}/analytics`

Retrieves usage analytics for a tool.

**Response:** `200 OK`

### Memory

#### List Memory Chunks
`GET /api/v1/agents/{id}/memory`

Lists memory chunks for an agent.

**Response:** `200 OK`

#### Search Memory
`POST /api/v1/agents/{id}/memory/search`

Searches agent memory using vector similarity.

**Request Body:**
```json
{
  "query": "machine learning algorithms",
  "top_k": 10
}
```

**Response:** `200 OK`

#### Get Memory Chunk
`GET /api/v1/memory/{chunk_id}`

Retrieves a specific memory chunk.

**Response:** `200 OK`

#### Delete Memory Chunk
`DELETE /api/v1/memory/{chunk_id}`

Deletes a memory chunk.

**Response:** `204 No Content`

#### Summarize Memory
`POST /api/v1/memory/{id}/summarize`

Generates a summary of memory chunks.

**Response:** `200 OK`

### Plans

#### List Plans
`GET /api/v1/plans`

Lists execution plans.

**Response:** `200 OK`

#### Get Plan
`GET /api/v1/plans/{id}`

Retrieves plan details.

**Response:** `200 OK`

#### Update Plan Status
`PUT /api/v1/plans/{id}`

Updates plan execution status.

**Response:** `200 OK`

### Reflections

#### List Reflections
`GET /api/v1/reflections`

Lists agent reflections.

**Response:** `200 OK`

#### Get Reflection
`GET /api/v1/reflections/{id}`

Retrieves reflection details.

**Response:** `200 OK`

### Budgets

#### Get Budget
`GET /api/v1/agents/{id}/budget`

Retrieves budget configuration for an agent.

**Response:** `200 OK`

#### Set Budget
`POST /api/v1/agents/{id}/budget`

Sets budget limits for an agent.

**Request Body:**
```json
{
  "monthly_limit": 1000.00,
  "currency": "USD"
}
```

**Response:** `201 Created`

#### Update Budget
`PUT /api/v1/agents/{id}/budget`

Updates budget configuration.

**Response:** `200 OK`

### Webhooks

#### List Webhooks
`GET /api/v1/webhooks`

Lists all webhooks.

**Response:** `200 OK`

#### Create Webhook
`POST /api/v1/webhooks`

Creates a new webhook.

**Request Body:**
```json
{
  "url": "https://example.com/webhook",
  "events": ["message.created", "session.updated"],
  "secret": "webhook_secret"
}
```

**Response:** `201 Created`

#### Get Webhook
`GET /api/v1/webhooks/{id}`

Retrieves webhook details.

**Response:** `200 OK`

#### Update Webhook
`PUT /api/v1/webhooks/{id}`

Updates webhook configuration.

**Response:** `200 OK`

#### Delete Webhook
`DELETE /api/v1/webhooks/{id}`

Deletes a webhook.

**Response:** `204 No Content`

#### List Webhook Deliveries
`GET /api/v1/webhooks/{id}/deliveries`

Lists webhook delivery attempts.

**Response:** `200 OK`

### Human-in-the-Loop

#### List Approval Requests
`GET /api/v1/approval-requests`

Lists pending approval requests.

**Response:** `200 OK`

#### Get Approval Request
`GET /api/v1/approval-requests/{id}`

Retrieves approval request details.

**Response:** `200 OK`

#### Approve Request
`POST /api/v1/approval-requests/{id}/approve`

Approves a pending request.

**Response:** `200 OK`

#### Reject Request
`POST /api/v1/approval-requests/{id}/reject`

Rejects a pending request.

**Response:** `200 OK`

#### Submit Feedback
`POST /api/v1/feedback`

Submits feedback on agent responses.

**Request Body:**
```json
{
  "session_id": "uuid",
  "message_id": 123,
  "rating": 5,
  "comment": "Great response!"
}
```

**Response:** `201 Created`

#### List Feedback
`GET /api/v1/feedback`

Lists all feedback submissions.

**Response:** `200 OK`

#### Get Feedback Stats
`GET /api/v1/feedback/stats`

Retrieves feedback statistics.

**Response:** `200 OK`

### Collaboration Workspaces

#### Create Workspace
`POST /api/v1/workspaces`

Creates a new collaboration workspace.

**Request Body:**
```json
{
  "name": "Project Workspace",
  "description": "Shared workspace for collaboration",
  "members": ["user1", "user2"]
}
```

**Response:** `201 Created`

#### Get Workspace
`GET /api/v1/workspaces/{id}`

Retrieves workspace details.

**Response:** `200 OK`

#### Update Workspace
`PUT /api/v1/workspaces/{id}`

Updates workspace configuration.

**Response:** `200 OK`

#### Delete Workspace
`DELETE /api/v1/workspaces/{id}`

Deletes a workspace.

**Response:** `204 No Content`

#### List Workspaces
`GET /api/v1/workspaces`

Lists all workspaces.

**Response:** `200 OK`

### Async Tasks

#### List Async Tasks
`GET /api/v1/async-tasks`

Lists asynchronous tasks.

**Response:** `200 OK`

#### Get Async Task
`GET /api/v1/async-tasks/{id}`

Retrieves async task details.

**Response:** `200 OK`

#### Create Async Task
`POST /api/v1/async-tasks`

Creates a new async task.

**Response:** `201 Created`

### Alert Preferences

#### Get Alert Preferences
`GET /api/v1/alert-preferences`

Retrieves alert preferences.

**Response:** `200 OK`

#### Update Alert Preferences
`PUT /api/v1/alert-preferences`

Updates alert preferences.

**Response:** `200 OK`

### Batch Operations

#### Batch Create Agents
`POST /api/v1/agents/batch`

Creates multiple agents in a single request.

**Response:** `201 Created`

#### Batch Delete Agents
`POST /api/v1/agents/batch/delete`

Deletes multiple agents.

**Response:** `200 OK`

#### Batch Delete Messages
`POST /api/v1/messages/batch/delete`

Deletes multiple messages.

**Response:** `200 OK`

#### Batch Delete Tools
`POST /api/v1/tools/batch/delete`

Deletes multiple tools.

**Response:** `200 OK`

### Analytics

#### Get Analytics Overview
`GET /api/v1/analytics/overview`

Retrieves system-wide analytics.

**Response:** `200 OK`

### WebSocket

#### WebSocket Connection
`GET /ws?session_id={session_id}`

Establishes a WebSocket connection for streaming agent responses.

**Protocol:** WebSocket

**Message Format:**
```json
{
  "type": "message",
  "content": "Agent response chunk",
  "done": false
}
```

## Error Responses

All errors follow a consistent format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "request_id": "uuid",
    "details": {}
  }
}
```

**HTTP Status Codes:**
- `400 Bad Request` - Invalid request parameters
- `401 Unauthorized` - Missing or invalid API key
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Service temporarily unavailable

## Rate Limiting

API requests are rate-limited per API key. Rate limits are configurable and enforced per endpoint. When rate limits are exceeded, the API returns `429 Too Many Requests` with a `Retry-After` header indicating when to retry.

## Pagination

List endpoints support pagination using query parameters:

- `limit` - Maximum number of items to return (default: 50, max: 1000)
- `offset` - Number of items to skip (default: 0)

**Example:**
```
GET /api/v1/agents?limit=20&offset=40
```

## Filtering and Search

Many list endpoints support filtering and search:

- `search` - Text search query
- `filter` - Additional filter parameters (format varies by endpoint)

## WebSocket Streaming

For real-time streaming responses, connect to the WebSocket endpoint:

```javascript
const ws = new WebSocket('ws://localhost:8080/ws?session_id=YOUR_SESSION_ID');

ws.onopen = () => {
  // Connection established
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.done) {
    // Stream complete
  } else {
    // Process chunk
    console.log(data.content);
  }
};
```

## OpenAPI Specification

A complete OpenAPI 3.0 specification is available at:

```
http://localhost:8080/openapi.yaml
```

Use this specification to generate client libraries, explore the API interactively, or integrate with API documentation tools.



