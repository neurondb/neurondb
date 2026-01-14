# NeuronDesktop API Documentation

## Base URL

```
http://localhost:8081/api/v1
```

## Authentication

All API requests require authentication using an API key in the Authorization header:

```
Authorization: Bearer <api_key>
```

## Rate Limiting

Rate limits are applied per API key:
- **Limit**: 100 requests per minute
- **Headers**: 
  - `X-RateLimit-Limit`: Maximum requests allowed
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Time when limit resets

## Error Responses

All errors follow this format:

```json
{
  "error": "Error Type",
  "message": "Detailed error message",
  "code": "ERROR_CODE",
  "details": {
    "field": "field_name",
    "message": "Validation message"
  }
}
```

## Endpoints

### Health Check

#### GET /health

Health check endpoint (no authentication required).

**Response:**
```json
{
  "status": "ok",
  "service": "neurondesk-api",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Profiles

#### GET /api/v1/profiles

List all profiles for the authenticated user.

**Response:**
```json
[
  {
    "id": "uuid",
    "name": "My Profile",
    "user_id": "user123",
    "mcp_config": {
      "command": "neurondb-mcp",
      "args": [],
      "env": {}
    },
    "neurondb_dsn": "host=localhost port=5432 user=neurondb dbname=neurondb",
    "agent_endpoint": "http://localhost:8080",
    "agent_api_key": "key",
    "default_collection": "documents",
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z"
  }
]
```

#### POST /api/v1/profiles

Create a new profile.

**Request Body:**
```json
{
  "name": "My Profile",
  "mcp_config": {
    "command": "neurondb-mcp",
    "args": [],
    "env": {
      "NEURONDB_HOST": "localhost"
    }
  },
  "neurondb_dsn": "host=localhost port=5432 user=neurondb dbname=neurondb",
  "agent_endpoint": "http://localhost:8080",
  "agent_api_key": "key",
  "default_collection": "documents"
}
```

#### GET /api/v1/profiles/{id}

Get a specific profile.

#### PUT /api/v1/profiles/{id}

Update a profile.

#### DELETE /api/v1/profiles/{id}

Delete a profile.

### MCP

#### GET /api/v1/mcp/connections

List active MCP connections.

**Response:**
```json
[
  {
    "profile_id": "uuid",
    "alive": true
  }
]
```

#### GET /api/v1/profiles/{profile_id}/mcp/tools

List available tools from MCP server.

**Response:**
```json
{
  "tools": [
    {
      "name": "vector_search",
      "description": "Perform vector search",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query_vector": {
            "type": "array",
            "items": {"type": "number"}
          }
        }
      }
    }
  ]
}
```

#### POST /api/v1/profiles/{profile_id}/mcp/tools/call

Call a tool on the MCP server.

**Request Body:**
```json
{
  "name": "vector_search",
  "arguments": {
    "query_vector": [0.1, 0.2, 0.3],
    "table": "documents",
    "limit": 10
  }
}
```

**Response:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "Search results..."
    }
  ],
  "isError": false,
  "metadata": {}
}
```

#### WebSocket /api/v1/profiles/{profile_id}/mcp/ws

WebSocket connection for real-time MCP communication.

**Message Format:**
```json
{
  "type": "tool_call",
  "name": "vector_search",
  "arguments": {}
}
```

### NeuronDB

#### GET /api/v1/profiles/{profile_id}/neurondb/collections

List all collections (tables with vector columns).

**Response:**
```json
[
  {
    "name": "documents",
    "schema": "public",
    "vector_col": "embedding",
    "indexes": [
      {
        "name": "documents_embedding_idx",
        "type": "hnsw",
        "definition": "CREATE INDEX ...",
        "size": "10 MB"
      }
    ],
    "row_count": 1000
  }
]
```

#### POST /api/v1/profiles/{profile_id}/neurondb/search

Perform a vector search.

**Request Body:**
```json
{
  "collection": "documents",
  "schema": "public",
  "query_vector": [0.1, 0.2, 0.3],
  "query_text": "search query",
  "limit": 10,
  "distance_type": "cosine",
  "filter": {
    "category": "tech"
  }
}
```

**Response:**
```json
[
  {
    "id": 1,
    "score": 0.95,
    "distance": 0.05,
    "data": {
      "id": 1,
      "content": "Document content...",
      "metadata": {}
    }
  }
]
```

#### POST /api/v1/profiles/{profile_id}/neurondb/sql

Execute a SQL query (SELECT only, with guardrails).

**Request Body:**
```json
{
  "query": "SELECT * FROM documents LIMIT 10"
}
```

### Agent

#### GET /api/v1/profiles/{profile_id}/agent/agents

List agents from NeuronAgent.

#### POST /api/v1/profiles/{profile_id}/agent/sessions

Create a new session.

**Request Body:**
```json
{
  "agent_id": "uuid",
  "external_user_id": "user123",
  "metadata": {}
}
```

#### POST /api/v1/profiles/{profile_id}/agent/sessions/{session_id}/messages

Send a message to an agent.

**Request Body:**
```json
{
  "role": "user",
  "content": "Hello, how are you?",
  "stream": false
}
```

#### WebSocket /api/v1/profiles/{profile_id}/agent/ws?session_id={session_id}

WebSocket connection for real-time agent communication.

### Metrics

#### GET /api/v1/metrics

Get application metrics.

**Response:**
```json
{
  "requests": {
    "total": 1000,
    "successful": 950,
    "failed": 50
  },
  "response_time": {
    "avg_ms": 45,
    "min_ms": 10,
    "max_ms": 500
  },
  "connections": {
    "mcp": 2,
    "neurondb": 3,
    "agent": 1
  },
  "endpoints": {
    "/api/v1/profiles": 100,
    "/api/v1/mcp/tools": 200
  },
  "errors": {
    "VALIDATION_ERROR": 10,
    "INTERNAL_ERROR": 5
  }
}
```

#### POST /api/v1/metrics/reset

Reset all metrics.

## Error Codes

- `VALIDATION_ERROR`: Request validation failed
- `UNAUTHORIZED`: Authentication failed
- `NOT_FOUND`: Resource not found
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `INTERNAL_ERROR`: Server error

## Status Codes

- `200 OK`: Success
- `201 Created`: Resource created
- `400 Bad Request`: Invalid request
- `401 Unauthorized`: Authentication required
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

