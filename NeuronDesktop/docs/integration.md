# NeuronDesktop Integration Guide

## Integration Status

NeuronDesktop is **100% integrated** with both NeuronMCP and NeuronAgent.

## NeuronMCP Integration

### How It Works

NeuronDesktop integrates with NeuronMCP by:

1. **Process Spawning**: The MCP client (`api/internal/mcp/client.go`) spawns the NeuronMCP server process via stdio
2. **JSON-RPC 2.0 Communication**: Communicates using the Model Context Protocol over stdin/stdout
3. **Default Configuration**: Defaults to `neurondb-mcp` command (the NeuronMCP binary)

### Integration Points

**Backend (`api/internal/mcp/client.go`)**:
- Spawns MCP server processes (e.g., `neurondb-mcp`)
- Handles JSON-RPC 2.0 protocol
- Supports `tools/list`, `tools/call`, `resources/list`, `resources/read`
- WebSocket support for real-time communication

**API Endpoints**:
- `GET /api/v1/profiles/{profile_id}/mcp/tools` - List tools from NeuronMCP
- `POST /api/v1/profiles/{profile_id}/mcp/tools/call` - Call NeuronMCP tools
- `WS /api/v1/profiles/{profile_id}/mcp/ws` - WebSocket for streaming

**Frontend (`frontend/app/mcp/page.tsx`)**:
- MCP Console UI
- Tool inspector
- Real-time tool calling via WebSocket

### Configuration

In a profile's `mcp_config`:
```json
{
  "command": "neurondb-mcp",
  "args": [],
  "env": {
    "NEURONDB_HOST": "localhost",
    "NEURONDB_PORT": "5432",
    "NEURONDB_DATABASE": "neurondb",
    "NEURONDB_USER": "neurondb",
    "NEURONDB_PASSWORD": "neurondb"
  }
}
```

### Usage

1. Create a profile with MCP configuration
2. The backend spawns `neurondb-mcp` process
3. Frontend can list and call all NeuronMCP tools
4. All NeuronMCP capabilities are available through the UI

## NeuronAgent Integration

### How It Works

NeuronDesktop integrates with NeuronAgent by:

1. **HTTP Client**: Uses HTTP REST API to communicate with NeuronAgent
2. **API Key Authentication**: Forwards API keys from profiles
3. **WebSocket Proxy**: Proxies WebSocket connections for real-time agent communication

### Integration Points

**Backend (`api/internal/agent/client.go`)**:
- HTTP client for NeuronAgent REST API
- Supports all NeuronAgent endpoints:
  - `/api/v1/agents` (CRUD)
  - `/api/v1/sessions` (create, get)
  - `/api/v1/sessions/{id}/messages` (send, list)
  - `/ws` (WebSocket proxy)

**API Endpoints**:
- `GET /api/v1/profiles/{profile_id}/agent/agents` - List agents
- `POST /api/v1/profiles/{profile_id}/agent/sessions` - Create session
- `POST /api/v1/profiles/{profile_id}/agent/sessions/{session_id}/messages` - Send message
- `WS /api/v1/profiles/{profile_id}/agent/ws` - WebSocket proxy

**Frontend**:
- Can be extended to show agent management UI
- WebSocket support for streaming agent responses

### Configuration

In a profile:
```json
{
  "agent_endpoint": "http://localhost:8080",
  "agent_api_key": "your-neuronagent-api-key"
}
```

### Usage

1. Create a profile with NeuronAgent endpoint and API key
2. Backend proxies all requests to NeuronAgent
3. WebSocket connections are proxied for real-time communication
4. All NeuronAgent functionality is available through NeuronDesktop

## Integration Architecture

```
┌─────────────────────────────────────────────────────────┐
│              NeuronDesktop Frontend                      │
│              (Next.js + React)                           │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP/WebSocket
┌──────────────────────▼──────────────────────────────────┐
│         NeuronDesktop Backend API                        │
│         (Go REST + WebSocket Server)                     │
├─────────────────────────────────────────────────────────┤
│  MCP Proxy          │  Agent Client  │  NeuronDB Client │
└──────┬──────────────┴───────┬────────┴────────┬────────┘
       │                      │                 │
       │ stdio                 │ HTTP            │ Postgres
       │ (JSON-RPC 2.0)        │ (REST API)      │ (SQL)
       │                       │                 │
┌──────▼──────────┐  ┌─────────▼─────────┐  ┌───▼────────┐
│  NeuronMCP     │  │  NeuronAgent      │  │  NeuronDB  │
│  (neurondb-mcp)│  │  (HTTP API)       │  │  (Postgres) │
│  Process       │  │  Port 8080        │  │  Extension │
└────────────────┘  └───────────────────┘  └────────────┘
```

## Verification

### Check NeuronMCP Integration

1. Ensure `neurondb-mcp` binary exists in PATH or specify full path in profile
2. Create a profile with MCP config pointing to NeuronMCP
3. Test: `GET /api/v1/profiles/{id}/mcp/tools` should return NeuronMCP tools

### Check NeuronAgent Integration

1. Ensure NeuronAgent is running on port 8080 (or configured port)
2. Create a profile with `agent_endpoint` and `agent_api_key`
3. Test: `GET /api/v1/profiles/{id}/agent/agents` should return agents from NeuronAgent

## Example Profile Configuration

```json
{
  "name": "Full Integration Profile",
  "mcp_config": {
    "command": "/path/to/neurondb-mcp",
    "env": {
      "NEURONDB_HOST": "localhost",
      "NEURONDB_PORT": "5432",
      "NEURONDB_DATABASE": "neurondb",
      "NEURONDB_USER": "neurondb",
      "NEURONDB_PASSWORD": "neurondb"
    }
  },
  "neurondb_dsn": "host=localhost port=5432 user=neurondb dbname=neurondb",
  "agent_endpoint": "http://localhost:8080",
  "agent_api_key": "your-neuronagent-api-key"
}
```

## Summary

✅ **NeuronMCP**: Fully integrated via stdio process spawning and JSON-RPC 2.0  
✅ **NeuronAgent**: Fully integrated via HTTP REST API and WebSocket proxy  
✅ **NeuronDB**: Fully integrated via direct Postgres connection  

All four components (NeuronDB, NeuronAgent, NeuronMCP, and NeuronDesktop) are seamlessly integrated and accessible through the unified NeuronDesktop interface.

