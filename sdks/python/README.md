# NeuronDB Python SDKs

Dependencies are pinned in `setup.py` for reproducible installs. A TypeScript/JavaScript SDK for NeuronAgent and NeuronMCP is planned.

This package includes:

- **neurondb_mcp**: NeuronMCP (Model Context Protocol) client – tools, async.
- **neuronagent**: NeuronAgent HTTP client – agents, sessions, messages, and **streaming**.

## Features

- ✅ Full MCP protocol support
- ✅ Async/await support
- ✅ Type hints throughout
- ✅ Comprehensive error handling
- ✅ Automatic retry logic
- ✅ Connection pooling
- ✅ Request/response logging

## Installation

```bash
pip install neurondb-mcp-python
```

## Quick Start

```python
import asyncio
from neurondb_mcp import NeuronMCPClient

async def main():
    async with NeuronMCPClient("http://localhost:8080", api_key="your-api-key") as client:
        # List all available tools
        tools = await client.list_tools()
        print(f"Available tools: {len(tools)}")
        
        # Call a tool
        result = await client.call_tool("vector_search", {
            "table": "documents",
            "vector_column": "embedding",
            "query_vector": [0.1, 0.2, 0.3],
            "limit": 10
        })
        
        print(f"Results: {result.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

## NeuronAgent client (sessions and streaming)

```python
from neuronagent import NeuronAgentClient

client = NeuronAgentClient(base_url="http://localhost:8080", api_key="your-api-key")
agent = client.agents.create_agent(name="my-agent", system_prompt="You are helpful.")
session = client.sessions.create_session(agent_id=agent.id)
response = client.sessions.send_message(session_id=session.id, content="Hello")
print(response.response)

# Streaming
for chunk in client.send_message_stream(session_id=session.id, content="Explain vectors"):
    print(chunk, end="")
```

One-off: `from neuronagent import run; print(run("http://localhost:8080", agent_id, "Hello"))`

## Examples

See the `examples/` directory for more comprehensive examples.

## Documentation

Full documentation available at: https://docs.neurondb.com/mcp/python-sdk

## License

MIT License
