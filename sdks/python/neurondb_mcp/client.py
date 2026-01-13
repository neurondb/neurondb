"""
NeuronMCP Client Implementation

Provides async/await support for MCP protocol communication.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import aiohttp

from .exceptions import MCPError, MCPConnectionError, MCPTimeoutError, MCPToolError
from .types import ToolResult, MCPRequest, MCPResponse


class NeuronMCPClient:
    """
    NeuronMCP Client for interacting with MCP server.
    
    Supports async/await, connection pooling, retry logic, and comprehensive error handling.
    
    Args:
        base_url: Base URL of the MCP server
        api_key: Optional API key for authentication
        timeout: Request timeout in seconds (default: 30)
        max_retries: Maximum number of retries (default: 3)
        retry_backoff: Backoff multiplier for retries (default: 1.5)
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_backoff: float = 1.5,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def connect(self):
        """Establish connection to MCP server."""
        if self._session is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers=headers,
            )
    
    async def close(self):
        """Close connection to MCP server."""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def call_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        retry: bool = True,
    ) -> ToolResult:
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            params: Tool parameters
            retry: Whether to retry on failure
            
        Returns:
            ToolResult containing the tool execution result
            
        Raises:
            MCPConnectionError: If connection fails
            MCPTimeoutError: If request times out
            MCPToolError: If tool execution fails
        """
        if self._session is None:
            await self.connect()
        
        request = MCPRequest(
            jsonrpc="2.0",
            method="tools/call",
            params={
                "name": tool_name,
                "arguments": params,
            },
            id=self._generate_id(),
        )
        
        if retry:
            return await self._call_with_retry(request)
        else:
            return await self._call_once(request)
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all available tools.
        
        Returns:
            List of tool definitions
        """
        if self._session is None:
            await self.connect()
        
        request = MCPRequest(
            jsonrpc="2.0",
            method="tools/list",
            params={},
            id=self._generate_id(),
        )
        
        response = await self._call_once(request)
        if response.result and "tools" in response.result:
            return response.result["tools"]
        return []
    
    async def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """
        Get schema for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool schema definition
        """
        tools = await self.list_tools()
        for tool in tools:
            if tool.get("name") == tool_name:
                return tool
        raise MCPToolError(f"Tool '{tool_name}' not found")
    
    async def _call_with_retry(self, request: MCPRequest) -> ToolResult:
        """Call with retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return await self._call_once(request)
            except (MCPConnectionError, MCPTimeoutError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_backoff ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    raise
            except MCPToolError:
                # Don't retry tool errors
                raise
        
        raise last_error
    
    async def _call_once(self, request: MCPRequest) -> ToolResult:
        """Make a single MCP request."""
        url = urljoin(self.base_url, "/mcp")
        
        try:
            async with self._session.post(
                url,
                json=request.dict(),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    mcp_response = MCPResponse(**data)
                    
                    if mcp_response.error:
                        raise MCPToolError(
                            mcp_response.error.get("message", "Unknown error"),
                            code=mcp_response.error.get("code"),
                        )
                    
                    return ToolResult.from_mcp_response(mcp_response)
                elif response.status == 408 or response.status == 504:
                    raise MCPTimeoutError(f"Request timeout: {response.status}")
                else:
                    error_text = await response.text()
                    raise MCPConnectionError(
                        f"HTTP {response.status}: {error_text}"
                    )
        except aiohttp.ClientError as e:
            raise MCPConnectionError(f"Connection error: {str(e)}")
        except asyncio.TimeoutError:
            raise MCPTimeoutError("Request timeout")
    
    def _generate_id(self) -> str:
        """Generate a unique request ID."""
        import time
        return f"req_{int(time.time() * 1000000)}"



