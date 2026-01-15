"""
NeuronMCP Python SDK

A comprehensive Python SDK for interacting with NeuronMCP (Model Context Protocol) server.
Provides full MCP protocol support with async/await, type hints, and comprehensive documentation.

Example:
    ```python
    from neurondb_mcp import NeuronMCPClient
    
    async with NeuronMCPClient("http://localhost:8080") as client:
        result = await client.call_tool("vector_search", {
            "table": "documents",
            "vector_column": "embedding",
            "query_vector": [0.1, 0.2, 0.3],
            "limit": 10
        })
        print(result)
    ```
"""

__version__ = "1.0.0"
__author__ = "neurondb, Inc."

from .client import NeuronMCPClient
from .types import ToolResult, ToolError, MCPRequest, MCPResponse
from .exceptions import MCPError, MCPConnectionError, MCPTimeoutError, MCPToolError

__all__ = [
    "NeuronMCPClient",
    "ToolResult",
    "ToolError",
    "MCPRequest",
    "MCPResponse",
    "MCPError",
    "MCPConnectionError",
    "MCPTimeoutError",
    "MCPToolError",
]




