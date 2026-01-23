# NeuronAgent API Documentation

## Table of Contents

- [Base URL](#base-url)
- [OpenAPI Specification](#openapi-specification)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [Agents](#agents)
  - [Sessions](#sessions)
  - [Messages](#messages)
  - [Workflows](#workflows)
  - [Plans](#plans)
  - [Budgets](#budgets)
  - [Collaborations](#collaborations)
  - [Tools](#tools)
  - [Memory](#memory)
  - [WebSocket](#websocket)
- [Reflections](#reflections)
- [Webhooks](#webhooks)
- [Approval Requests](#approval-requests)
- [Feedback](#feedback)
- [Marketplace](#marketplace)
- [Compliance](#compliance)
- [Observability](#observability)
- [Async Tasks](#async-tasks)
- [Event Streams](#event-streams)
- [Verification](#verification)
- [Virtual Filesystem](#virtual-filesystem)
- [Evaluation](#evaluation)
- [Replay](#replay)
- [RAG](#rag)
- [Embeddings](#embeddings)
- [Analytics](#analytics)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)

## Base URL

```
http://localhost:8080/api/v1
```

## OpenAPI Specification

For machine-readable API specification, see the [OpenAPI 3.0 specification](../openapi/openapi.yaml).

The OpenAPI spec includes:
- Complete endpoint definitions
- Request/response schemas
- Authentication requirements
- Error responses
- Example requests and responses

You can use the OpenAPI spec to:
- Generate client libraries
- View interactive API documentation (Swagger UI, Redoc)
- Validate API requests/responses
- Import into API testing tools

## Authentication

All API requests require authentication using an API key in the Authorization header:

```
Authorization: Bearer <api_key>
```

## Endpoints

### Agents

#### Create Agent
```
POST /api/v1/agents
```

Request body:
```json
{
  "name": "my-agent",
  "description": "A helpful agent",
  "system_prompt": "You are a helpful assistant.",
  "model_name": "gpt-4",
  "enabled_tools": ["sql", "http"],
  "config": {
    "temperature": 0.7,
    "max_tokens": 1000
  }
}
```

#### List Agents
```
GET /api/v1/agents
```

#### Get Agent
```
GET /api/v1/agents/{id}
```

#### Update Agent
```
PUT /api/v1/agents/{id}
```

#### Delete Agent
```
DELETE /api/v1/agents/{id}
```

#### Clone Agent
```
POST /api/v1/agents/{id}/clone
```

#### Generate Plan
```
POST /api/v1/agents/{id}/plan
```

#### Reflect on Response
```
POST /api/v1/agents/{id}/reflect
```

#### Delegate to Agent
```
POST /api/v1/agents/{id}/delegate
```

#### Get Agent Metrics
```
GET /api/v1/agents/{id}/metrics
```

#### Get Agent Costs
```
GET /api/v1/agents/{id}/costs
```

#### List Agent Versions
```
GET /api/v1/agents/{id}/versions
```

#### Create Agent Version
```
POST /api/v1/agents/{id}/versions
```

#### Get Agent Version
```
GET /api/v1/agents/{id}/versions/{version}
```

#### Activate Agent Version
```
PUT /api/v1/agents/{id}/versions/{version}/activate
```

#### List Agent Relationships
```
GET /api/v1/agents/{id}/relationships
```

#### Create Agent Relationship
```
POST /api/v1/agents/{id}/relationships
```

#### Delete Agent Relationship
```
DELETE /api/v1/agents/{id}/relationships/{relationship_id}
```

#### Batch Create Agents
```
POST /api/v1/agents/batch
```

#### Batch Delete Agents
```
POST /api/v1/agents/batch/delete
```

### Sessions

#### Create Session
```
POST /api/v1/sessions
```

Request body:
```json
{
  "agent_id": "uuid",
  "external_user_id": "user123",
  "metadata": {}
}
```

#### Get Session
```
GET /api/v1/sessions/{id}
```

#### Update Session
```
PUT /api/v1/sessions/{id}
```

#### Delete Session
```
DELETE /api/v1/sessions/{id}
```

#### List Sessions
```
GET /api/v1/agents/{agent_id}/sessions
```

### Messages

#### Send Message
```
POST /api/v1/sessions/{session_id}/messages
```

Request body:
```json
{
  "role": "user",
  "content": "Hello, how are you?",
  "stream": false
}
```

#### Get Messages
```
GET /api/v1/sessions/{session_id}/messages
```

#### Get Message
```
GET /api/v1/messages/{id}
```

#### Update Message
```
PUT /api/v1/messages/{id}
```

#### Delete Message
```
DELETE /api/v1/messages/{id}
```

#### Batch Delete Messages
```
POST /api/v1/messages/batch/delete
```

### WebSocket

#### Connect to WebSocket
```
WS /ws?session_id={session_id}&api_key={api_key}
```

Or use Authorization header:
```
WS /ws?session_id={session_id}
Headers: Authorization: Bearer {api_key}
```

**Features:**
- API key authentication (query parameter or header)
- Ping/pong keepalive (60s timeout)
- Message queue for concurrent requests
- Graceful error handling

**Message Format:**
```json
{
  "content": "Your message here"
}
```

**Response Format:**
```json
{
  "type": "chunk",
  "content": "Response chunk..."
}
```

```json
{
  "type": "response",
  "content": "Full response",
  "complete": true,
  "tokens_used": 150,
  "tool_calls": [],
  "tool_results": []
}
```

**Error Format:**
```json
{
  "type": "error",
  "error": "Error message"
}
```

Send messages:
```json
{
  "content": "Hello"
}
```

Receive responses:
```json
{
  "type": "response",
  "content": "Hello! How can I help you?",
  "complete": true
}
```

## Evaluation Framework

### Create Evaluation Task
```
POST /api/v1/eval/tasks
```

Request body:
```json
{
  "task_type": "end_to_end",
  "input": "What is the capital of France?",
  "expected_output": "Paris",
  "expected_tool_sequence": {},
  "metadata": {}
}
```

### List Evaluation Tasks
```
GET /api/v1/eval/tasks?task_type=end_to_end&limit=100&offset=0
```

### Create Evaluation Run
```
POST /api/v1/eval/runs
```

### Execute Evaluation Run
```
POST /api/v1/eval/runs/{run_id}/execute
```

### Get Evaluation Run Results
```
GET /api/v1/eval/runs/{run_id}/results
```

## Execution Snapshots and Replay

### Create Execution Snapshot
```
POST /api/v1/sessions/{session_id}/snapshots
```

Request body:
```json
{
  "user_message": "Hello, agent!",
  "deterministic_mode": false
}
```

### List Snapshots
```
GET /api/v1/sessions/{session_id}/snapshots
GET /api/v1/agents/{agent_id}/snapshots
```

### Replay Execution
```
POST /api/v1/snapshots/{id}/replay
```

### Delete Snapshot
```
DELETE /api/v1/snapshots/{id}
```

## Workflow Schedules

### Create/Update Workflow Schedule
```
POST /api/v1/workflows/{workflow_id}/schedule
```

Request body:
```json
{
  "cron_expression": "0 0 * * *",
  "timezone": "UTC",
  "enabled": true
}
```

### Get Workflow Schedule
```
GET /api/v1/workflows/{workflow_id}/schedule
```

### List Workflow Schedules
```
GET /api/v1/workflow-schedules
```

### Delete Workflow Schedule
```
DELETE /api/v1/workflows/{workflow_id}/schedule
```

## Agent Specializations

### Create Agent Specialization
```
POST /api/v1/agents/{agent_id}/specialization
```

Request body:
```json
{
  "specialization_type": "coding",
  "capabilities": ["python", "javascript", "sql"],
  "config": {}
}
```

### Get Agent Specialization
```
GET /api/v1/agents/{agent_id}/specialization
```

### List Specializations
```
GET /api/v1/specializations?specialization_type=coding
```

### Update Specialization
```
PUT /api/v1/agents/{agent_id}/specialization
```

### Delete Specialization
```
DELETE /api/v1/agents/{agent_id}/specialization
```

## Memory Management

### Submit Memory Feedback
```
POST /api/v1/memory/{memory_id}/feedback
```

Submit user feedback on a memory retrieval to improve memory quality over time.

Request body:
```json
{
  "agent_id": "uuid",
  "session_id": "uuid (optional)",
  "memory_tier": "chunk|stm|mtm|lpm",
  "feedback_type": "positive|negative|neutral|correction",
  "feedback_text": "Optional feedback text",
  "query": "Query that led to this memory retrieval (optional)",
  "relevance_score": 0.85,
  "metadata": {}
}
```

Response:
```json
{
  "feedback_id": "uuid",
  "memory_id": "uuid",
  "agent_id": "uuid",
  "feedback_type": "positive",
  "status": "recorded",
  "message": "Feedback recorded and memory quality updated",
  "duration_ms": 45,
  "created_at": "2024-01-01T00:00:00Z"
}
```

### Get Retrieval Statistics
```
GET /api/v1/agents/{id}/retrieval-stats?days=30
```

Get statistics about retrieval decisions for an agent. Useful for understanding retrieval patterns and improving routing.

Query parameters:
- `days` (optional): Number of days to analyze (default: 30, max: 365)

Response:
```json
{
  "agent_id": "uuid",
  "days": 30,
  "total_decisions": 150,
  "avg_confidence": 0.75,
  "source_usage": {
    "vector_db": 80,
    "web": 50,
    "api": 20
  },
  "avg_quality_score": 0.82,
  "duration_ms": 120
}
```

### Consolidate Memory
```
POST /api/v1/agents/{id}/memory/consolidate
```

Consolidate similar memories to reduce duplication and improve storage efficiency.

Request body:
```json
{
  "tier": "stm|mtm|lpm",
  "similarity_threshold": 0.9
}
```

Response:
```json
{
  "agent_id": "uuid",
  "tier": "mtm",
  "similarity_threshold": 0.9,
  "consolidated_count": 15,
  "status": "completed",
  "duration_ms": 2500,
  "completed_at": "2024-01-01T00:00:00Z"
}
```

### Get Memory Quality
```
GET /api/v1/agents/{id}/memory/quality?memory_id={uuid}&tier={tier}
```

Get quality metrics for a specific memory or update all memory quality scores.

Query parameters:
- `memory_id` (optional): Specific memory ID
- `tier` (optional): Memory tier (stm, mtm, lpm) - required if memory_id provided

For complete API documentation including all endpoints, request/response schemas, and examples, see the [OpenAPI specification](../openapi/openapi.yaml).

---

<div align="center">

[⬆ Back to Top](#neuronagent-api-documentation)

</div>
