# NeuronAgent API Documentation

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

#### Delete Agent
```
DELETE /api/v1/agents/{id}
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

For complete API documentation including all endpoints, request/response schemas, and examples, see the [OpenAPI specification](../openapi/openapi.yaml).

