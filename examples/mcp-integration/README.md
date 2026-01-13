# MCP Integration Examples

Example configurations and demonstrations for integrating NeuronMCP with MCP-compatible clients.

## Overview

This example provides:
- Claude Desktop configuration
- Example integration with other MCP clients
- Tool demonstration scripts
- Setup verification

## Quick Start

### Prerequisites

- NeuronMCP server running
- MCP-compatible client (Claude Desktop, etc.)
- PostgreSQL 16+ with NeuronDB extension

### Claude Desktop Setup

1. **Locate Claude Desktop config:**
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - Linux: `~/.config/Claude/claude_desktop_config.json`

2. **Add NeuronMCP configuration:**
   ```json
   {
     "mcpServers": {
       "neurondb": {
         "command": "neurondb-mcp",
         "args": [],
         "env": {
           "NEURONDB_HOST": "localhost",
           "NEURONDB_PORT": "5432",
           "NEURONDB_DATABASE": "neurondb",
           "NEURONDB_USER": "postgres",
           "NEURONDB_PASSWORD": "your_password"
         }
       }
     }
   }
   ```

3. **Restart Claude Desktop**

### Docker-based Setup

If using Docker:

```json
{
  "mcpServers": {
    "neurondb": {
      "command": "docker",
      "args": ["exec", "-i", "neurondb-mcp", "/app/neurondb-mcp"],
      "env": {
        "NEURONDB_HOST": "neurondb-cpu",
        "NEURONDB_PORT": "5432",
        "NEURONDB_DATABASE": "neurondb",
        "NEURONDB_USER": "neurondb",
        "NEURONDB_PASSWORD": "neurondb"
      }
    }
  }
}
```

## Files

- `claude_desktop_config.json` - Complete Claude Desktop configuration
- `test_mcp_connection.py` - Test MCP server connection
- `list_tools.py` - List available MCP tools
- `call_tool_example.py` - Example tool call
- `readme.md` - This file

## Testing MCP Connection

```bash
# Test connection
python test_mcp_connection.py

# List available tools
python list_tools.py

# Call a tool
python call_tool_example.py vector_search '{"query_vector": "[0.1,0.2,0.3]", "limit": 5}'
```

## MCP Tools Available

NeuronMCP provides these tools:

- `vector_search` - Search vectors using similarity
- `embed_text` - Generate text embeddings
- `train_model` - Train ML models
- `predict` - Make predictions with trained models
- `list_indexes` - List vector indexes
- `create_index` - Create new vector index

See [NeuronMCP Tool Catalog](../../NeuronMCP/docs/tool-resource-catalog.md) for complete list.

## Other MCP Clients

### LangChain Integration

```python
from langchain.agents import Agent
from langchain_mcp import MCPTool

# Create MCP tool
mcp_tool = MCPTool(
    server_name="neurondb",
    command="neurondb-mcp",
    env={
        "NEURONDB_HOST": "localhost",
        "NEURONDB_PORT": "5432",
        "NEURONDB_DATABASE": "neurondb"
    }
)

# Use in agent
agent = Agent(tools=[mcp_tool])
```

## Troubleshooting

### Connection Issues

1. Verify NeuronMCP is running:
   ```bash
   ps aux | grep neurondb-mcp
   ```

2. Test stdio communication:
   ```bash
   echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | neurondb-mcp
   ```

3. Check environment variables:
   ```bash
   env | grep NEURONDB
   ```

### Tool Not Found

- Verify tool is listed: `python list_tools.py`
- Check NeuronMCP logs for errors
- Ensure database connection is working

## Related Documentation

- [NeuronMCP README](../../NeuronMCP/readme.md) - Complete MCP documentation
- [Tool Catalog](../../NeuronMCP/docs/tool-resource-catalog.md) - All available tools
- [MCP Protocol](https://modelcontextprotocol.io) - MCP specification







