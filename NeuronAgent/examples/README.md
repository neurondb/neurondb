# NeuronAgent Examples

This directory contains comprehensive examples demonstrating how to use NeuronAgent to create and interact with AI agents.

## Overview

NeuronAgent is an AI agent runtime system that provides:
- **Agent Runtime**: Complete state machine for autonomous task execution
- **Long-term Memory**: HNSW-based vector search for context retrieval
- **Tool System**: Extensible tool registry with SQL, HTTP, Code, and Shell tools
- **REST API**: Full CRUD API for agents, sessions, and messages
- **WebSocket Support**: Streaming agent responses in real-time

## Prerequisites

1. **NeuronAgent Server Running**
   ```bash
   # From NeuronAgent directory
   go run cmd/agent-server/main.go
   # Or using Docker
   cd docker && docker compose up -d
   ```

2. **Database Setup**
   - PostgreSQL 16+ with NeuronDB extension
   - Database migrations run (see main README)

3. **API Key**
   ```bash
   # Generate an API key
   ./scripts/generate_api_keys.sh
   
   # Or set environment variable
   export NEURONAGENT_API_KEY=your_api_key_here
   ```

## Examples

### Python Client (`python_client.py`)

A comprehensive Python client demonstrating:
- Creating agents with custom configurations
- Creating sessions for conversations
- Sending messages and receiving responses
- Using WebSocket for streaming responses
- Working with different agent types (general, research, data analyst)
- Error handling and best practices

**Run the example:**
```bash
# Install dependencies
pip install requests websocket-client

# Set API key
export NEURONAGENT_API_KEY=your_api_key

# Run examples
python3 python_client.py
```

**Example output:**
```
============================================================
NeuronAgent Python Client Examples
============================================================

✅ Server is healthy
📝 Creating agent...
✅ Agent created: example-assistant
💬 Creating session...
✅ Session created: <session-id>
💭 Sending message 1: Hello! Can you introduce yourself?...
🤖 Agent response: Hello! I'm an AI assistant...
```

### Go Client (`go_client.go`)

A native Go client demonstrating:
- Type-safe API interactions
- Agent creation and management
- Session handling
- Message sending and retrieval
- Error handling

**Run the example:**
```bash
# Set API key
export NEURONAGENT_API_KEY=your_api_key

# Run
go run go_client.go
```

## Example Use Cases

### 1. Basic Assistant Agent

Create a general-purpose assistant:

```python
from python_client import NeuronAgentClient

client = NeuronAgentClient()

agent = client.create_agent(
    name="my-assistant",
    system_prompt="You are a helpful assistant.",
    model_name="gpt-4",
    enabled_tools=['sql', 'http']
)

session = client.create_session(agent_id=agent['id'])

response = client.send_message(
    session_id=session['id'],
    content="Hello! What can you help me with?"
)

print(response['response'])
```

### 2. Research Agent with HTTP Tools

Create an agent that can search the web:

```python
agent = client.create_agent(
    name="research-assistant",
    system_prompt="""You are a research assistant. 
    Use HTTP tools to search for current information.""",
    enabled_tools=['http'],
    config={'temperature': 0.5}
)

session = client.create_session(agent_id=agent['id'])

response = client.send_message(
    session_id=session['id'],
    content="What are the latest AI developments?"
)
```

### 3. Data Analyst with SQL Tools

Create an agent that can query databases:

```python
agent = client.create_agent(
    name="data-analyst",
    system_prompt="""You are a data analyst. 
    Write and execute SQL queries to analyze data.""",
    enabled_tools=['sql'],
    config={'temperature': 0.2}  # Lower for precise SQL
)

session = client.create_session(agent_id=agent['id'])

response = client.send_message(
    session_id=session['id'],
    content="Find the top 10 customers by revenue"
)
```

### 4. Streaming Responses

Use WebSocket for real-time streaming:

```python
def on_message_chunk(data):
    if data.get('type') == 'response':
        print(data.get('content', ''), end='', flush=True)

client.stream_message(
    session_id=session['id'],
    content="Explain neural networks in detail",
    on_message=on_message_chunk
)
```

## Agent Profiles

NeuronAgent comes with predefined agent profiles (see `configs/agent_profiles.yaml`):

- **general-assistant**: General purpose assistant
- **code-assistant**: Specialized for code analysis
- **data-analyst**: Data analysis and SQL queries
- **research-assistant**: Research and information gathering

You can use these profiles or create custom agents with your own configuration.

## Available Tools

NeuronAgent supports several tool types:

- **sql**: Execute SQL queries on the database
- **http**: Make HTTP requests to external APIs
- **code**: Execute code in a sandboxed environment
- **shell**: Execute shell commands (use with caution)

Enable tools when creating an agent:

```python
agent = client.create_agent(
    name="my-agent",
    system_prompt="...",
    enabled_tools=['sql', 'http']  # Enable specific tools
)
```

## Configuration

### Agent Configuration

When creating an agent, you can configure:

```python
config = {
    'temperature': 0.7,      # Creativity (0.0-2.0)
    'max_tokens': 2000,      # Maximum response length
    'top_p': 0.9,            # Nucleus sampling
    # Add other model-specific parameters
}
```

### Server Configuration

Configure the server via environment variables or `config.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8080

database:
  host: "localhost"
  port: 5432
  database: "neurondb"
  user: "postgres"
  password: "postgres"
```

## Error Handling

Always handle errors appropriately:

```python
try:
    response = client.send_message(
        session_id=session['id'],
        content="Hello"
    )
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 401:
        print("Authentication failed. Check your API key.")
    elif e.response.status_code == 404:
        print("Session not found.")
    else:
        print(f"Error: {e}")
```

## Best Practices

1. **Reuse Sessions**: Create one session per conversation thread
2. **Manage API Keys**: Store API keys securely (environment variables, secrets manager)
3. **Handle Rate Limits**: Implement retry logic with exponential backoff
4. **Monitor Token Usage**: Track token consumption for cost management
5. **Use Streaming**: For long responses, use WebSocket streaming
6. **Error Handling**: Always handle errors and provide user feedback
7. **Tool Selection**: Only enable tools that your agent actually needs

## API Reference

See the [API Documentation](../docs/API.md) for complete API reference.

### Key Endpoints

- `POST /api/v1/agents` - Create agent
- `GET /api/v1/agents` - List agents
- `GET /api/v1/agents/{id}` - Get agent
- `POST /api/v1/sessions` - Create session
- `POST /api/v1/sessions/{id}/messages` - Send message
- `GET /api/v1/sessions/{id}/messages` - Get messages
- `WS /ws?session_id={id}` - WebSocket streaming

## Troubleshooting

### Server Not Responding

```bash
# Check if server is running
curl http://localhost:8080/health

# Check logs
docker compose logs agent-server
```

### Authentication Errors

```bash
# Verify API key is set
echo $NEURONAGENT_API_KEY

# Test authentication
curl -H "Authorization: Bearer $NEURONAGENT_API_KEY" \
  http://localhost:8080/api/v1/agents
```

### Database Connection Issues

```bash
# Test database connection
psql -h localhost -p 5432 -U postgres -d neurondb -c "SELECT 1;"

# Verify NeuronDB extension
psql -d neurondb -c "SELECT * FROM pg_extension WHERE extname = 'neurondb';"
```

## Next Steps

- Read the [Architecture Documentation](../docs/ARCHITECTURE.md)
- Check the [Deployment Guide](../docs/DEPLOYMENT.md)
- Explore the [NeuronDB Documentation](../../NeuronDB/README.md)
- Build your own custom agents!

## Support

- **Documentation**: See [main README](../README.md)
- **Issues**: Report on GitHub
- **Email**: support@neurondb.ai

## License

See [LICENSE](../../LICENSE) for license information.

