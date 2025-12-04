"""
Main MCP client implementation.
"""

import json
import time
from typing import Any, Dict, Optional
from .config import MCPConfig
from .transport import StdioTransport
from .protocol import JSONRPCRequest, JSONRPCResponse
from .commands import parse_command, build_tool_call_request


class MCPClient:
    """MCP client for communicating with MCP servers."""

    def __init__(self, config: MCPConfig, verbose: bool = False):
        """
        Initialize MCP client.

        Args:
            config: MCP configuration
            verbose: Enable verbose output
        """
        self.config = config
        self.verbose = verbose
        self.transport: Optional[StdioTransport] = None
        self.initialized = False

    def connect(self):
        """Connect to the MCP server."""
        if self.transport is not None:
            raise RuntimeError("Already connected")

        # Create transport
        self.transport = StdioTransport(
            command=self.config.command,
            env=self.config.get_env(),
            args=self.config.args,
        )

        # Start server process
        if self.verbose:
            print(f"Starting MCP server: {self.config.command}")
        self.transport.start()

        # Initialize connection
        self._initialize()

    def _initialize(self):
        """Initialize MCP connection."""
        if self.initialized:
            return

        # Send initialize request
        init_request = JSONRPCRequest(
            method="initialize",
            params={
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {
                    "name": "neurondb-mcp-client",
                    "version": "1.0.0",
                },
            },
        )

        response = self.transport.send_request(init_request)
        if response.is_error():
            raise RuntimeError(f"Initialize failed: {response.get_error_message()}")

        if self.verbose:
            print("MCP connection initialized")

        # Send initialized notification
        # Notifications don't have IDs in JSON-RPC 2.0
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }
        # For notifications, we don't wait for response
        # Just send the message
        init_json = json.dumps(notification)
        init_bytes = init_json.encode('utf-8')
        header = f"Content-Length: {len(init_bytes)}\r\n\r\n"
        self.transport.process.stdin.write(header.encode('utf-8'))
        self.transport.process.stdin.write(init_bytes)
        self.transport.process.stdin.flush()

        self.initialized = True

    def disconnect(self):
        """Disconnect from the MCP server."""
        if self.transport:
            if self.verbose:
                print("Disconnecting from MCP server")
            self.transport.stop()
            self.transport = None
            self.initialized = False

    def list_tools(self) -> Dict[str, Any]:
        """
        List available tools.

        Returns:
            Dictionary containing list of tools
        """
        request = JSONRPCRequest(
            method="tools/list",
            params={},
        )

        response = self.transport.send_request(request)
        if response.is_error():
            return {"error": response.get_error_message()}

        return response.result or {}

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Tool result
        """
        params = build_tool_call_request(tool_name, arguments)

        request = JSONRPCRequest(
            method="tools/call",
            params=params,
        )

        response = self.transport.send_request(request)
        if response.is_error():
            return {"error": response.get_error_message()}

        return response.result or {}

    def execute_command(self, command_str: str) -> Dict[str, Any]:
        """
        Execute a command string.

        Args:
            command_str: Command string (format: tool_name or tool_name:arg1=val1,arg2=val2)

        Returns:
            Command result
        """
        # Parse command
        try:
            tool_name, arguments = parse_command(command_str)
        except Exception as e:
            return {"error": f"Failed to parse command: {e}"}

        # Handle special commands
        if tool_name == "list_tools":
            return self.list_tools()
        elif tool_name == "resources/list":
            return self.list_resources()
        elif tool_name.startswith("resources/read"):
            # Handle resources/read:uri=...
            uri = arguments.get("uri")
            if not uri:
                return {"error": "Missing 'uri' parameter for resources/read"}
            return self.read_resource(uri)

        # Call tool
        return self.call_tool(tool_name, arguments)

    def list_resources(self) -> Dict[str, Any]:
        """
        List available resources.

        Returns:
            Dictionary containing list of resources
        """
        request = JSONRPCRequest(
            method="resources/list",
            params={},
        )

        response = self.transport.send_request(request)
        if response.is_error():
            return {"error": response.get_error_message()}

        return response.result or {}

    def read_resource(self, uri: str) -> Dict[str, Any]:
        """
        Read a resource.

        Args:
            uri: Resource URI

        Returns:
            Resource content
        """
        request = JSONRPCRequest(
            method="resources/read",
            params={"uri": uri},
        )

        response = self.transport.send_request(request)
        if response.is_error():
            return {"error": response.get_error_message()}

        return response.result or {}

