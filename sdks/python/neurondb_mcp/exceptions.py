"""
Exception classes for NeuronMCP SDK.
"""


class MCPError(Exception):
    """Base exception for all MCP errors."""
    pass


class MCPConnectionError(MCPError):
    """Raised when connection to MCP server fails."""
    pass


class MCPTimeoutError(MCPError):
    """Raised when request times out."""
    pass


class MCPToolError(MCPError):
    """Raised when tool execution fails."""
    
    def __init__(self, message: str, code: str = None):
        self.message = message
        self.code = code
        super().__init__(self.message)



