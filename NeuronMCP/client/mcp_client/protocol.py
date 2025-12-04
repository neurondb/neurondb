"""
MCP Protocol implementation (JSON-RPC 2.0).
"""

import json
import uuid
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, asdict


@dataclass
class JSONRPCRequest:
    """JSON-RPC 2.0 request."""

    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    method: str = ""
    params: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Generate ID if not provided."""
        if self.id is None:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
        }
        # Only include ID if it's not None (notifications don't have IDs)
        if self.id is not None:
            result["id"] = self.id
        if self.params:
            result["params"] = self.params
        return result

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JSONRPCRequest":
        """Create from dictionary."""
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id"),
            method=data.get("method", ""),
            params=data.get("params"),
        )


@dataclass
class JSONRPCError:
    """JSON-RPC 2.0 error."""

    code: int
    message: str
    data: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "code": self.code,
            "message": self.message,
        }
        if self.data is not None:
            result["data"] = self.data
        return result


@dataclass
class JSONRPCResponse:
    """JSON-RPC 2.0 response."""

    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    result: Optional[Any] = None
    error: Optional[JSONRPCError] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
        }
        if self.error:
            result["error"] = self.error.to_dict()
        elif self.result is not None:
            result["result"] = self.result
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JSONRPCResponse":
        """Create from dictionary."""
        error = None
        if "error" in data:
            err_data = data["error"]
            error = JSONRPCError(
                code=err_data.get("code", -32603),
                message=err_data.get("message", "Internal error"),
                data=err_data.get("data"),
            )

        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id"),
            result=data.get("result"),
            error=error,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "JSONRPCResponse":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def matches_id(self, request_id: Any) -> bool:
        """Check if response ID matches request ID."""
        if self.id is None or request_id is None:
            return False
        
        # Convert both to strings for comparison
        response_id = self.id
        if isinstance(response_id, int):
            response_id = str(response_id)
        elif isinstance(response_id, bytes):
            response_id = response_id.decode('utf-8')
        
        req_id = request_id
        if isinstance(req_id, int):
            req_id = str(req_id)
        elif isinstance(req_id, bytes):
            req_id = req_id.decode('utf-8')
        
        return str(response_id) == str(req_id)

    def is_error(self) -> bool:
        """Check if response is an error."""
        return self.error is not None

    def get_error_message(self) -> str:
        """Get error message."""
        if self.error:
            return f"{self.error.message} (code: {self.error.code})"
        return ""

