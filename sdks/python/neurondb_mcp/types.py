"""
Type definitions for NeuronMCP SDK.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolResult:
    """Result from a tool execution."""
    
    content: List[Dict[str, Any]]
    is_error: bool = False
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_mcp_response(cls, response: "MCPResponse") -> "ToolResult":
        """Create ToolResult from MCPResponse."""
        result = response.result or {}
        content = result.get("content", [])
        is_error = result.get("isError", False)
        metadata = result.get("metadata")
        
        return cls(
            content=content,
            is_error=is_error,
            metadata=metadata,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "isError": self.is_error,
            "metadata": self.metadata,
        }


@dataclass
class ToolError:
    """Error from tool execution."""
    
    message: str
    code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class MCPRequest:
    """MCP JSON-RPC request."""
    
    jsonrpc: str = "2.0"
    method: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    id: str = ""
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
            "params": self.params,
            "id": self.id,
        }


@dataclass
class MCPResponse:
    """MCP JSON-RPC response."""
    
    jsonrpc: str = "2.0"
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[str] = None




