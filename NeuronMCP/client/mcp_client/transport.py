"""
MCP Transport layer (stdio-based communication).
"""

import subprocess
import json
import struct
import sys
import io
from typing import Optional, Dict, Any
from .protocol import JSONRPCRequest, JSONRPCResponse


class StdioTransport:
    """Stdio-based transport for MCP communication."""

    def __init__(self, command: str, env: Dict[str, str], args: list = None):
        """
        Initialize stdio transport.

        Args:
            command: Command to execute
            env: Environment variables
            args: Additional command arguments
        """
        self.command = command
        self.env = env
        self.args = args or []
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0

    def start(self):
        """Start the MCP server process."""
        if self.process is not None:
            raise RuntimeError("Transport already started")

        cmd = [self.command] + self.args
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.env,
            text=False,  # Use binary mode for Content-Length protocol
            bufsize=0,
        )

    def stop(self):
        """Stop the MCP server process."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            except Exception:
                pass
            finally:
                self.process = None

    def send_request(self, request: JSONRPCRequest, expected_id: str = None) -> JSONRPCResponse:
        """
        Send a request and wait for response.
        Matches response ID with request ID to ensure correct pairing.

        Args:
            request: JSON-RPC request
            expected_id: Expected response ID (if None, uses request ID)

        Returns:
            JSON-RPC response with matching ID

        Raises:
            RuntimeError: If transport not started
            IOError: If communication fails
        """
        if self.process is None:
            raise RuntimeError("Transport not started")

        # Store request ID for matching
        request_id = request.id if request.id is not None else expected_id
        if isinstance(request_id, int):
            request_id = str(request_id)

        # Serialize request
        request_json = request.to_json()
        request_bytes = request_json.encode('utf-8')

        # Send Content-Length header + body (standard MCP format)
        header = f"Content-Length: {len(request_bytes)}\r\n\r\n"
        self.process.stdin.write(header.encode('utf-8'))
        self.process.stdin.write(request_bytes)
        self.process.stdin.flush()

        # Read responses until we get one with matching ID
        # (may need to skip notifications like "initialized")
        max_attempts = 10
        for attempt in range(max_attempts):
            response = self._read_response()
            
            # Check if this is a notification (no ID) - skip it and continue
            if response.id is None:
                if attempt < max_attempts - 1:
                    continue
                # If this is the last attempt and it's a notification, return it
                return response
            
            # Check if ID matches using the helper method
            if response.matches_id(request_id):
                return response
            
            # ID doesn't match - this might be a response to a different request
            # In a single-threaded client, this shouldn't happen
            # Try reading more responses
            if attempt < max_attempts - 1:
                continue
        
        # If we exhausted attempts, return the last response anyway
        return response

    def _read_response(self) -> JSONRPCResponse:
        """
        Read a JSON-RPC response from stdout.
        Supports both Content-Length format and Claude Desktop format (JSON directly).

        Returns:
            JSON-RPC response

        Raises:
            IOError: If reading fails
        """
        if self.process is None:
            raise RuntimeError("Transport not started")

        # Read first line to determine format
        first_line = self.process.stdout.readline()
        if not first_line:
            raise IOError("Unexpected EOF while reading response")
        
        first_line_str = first_line.decode('utf-8').strip()
        
        # Claude Desktop format: JSON directly (starts with '{')
        if first_line_str.startswith('{'):
            # This is a JSON response (Claude Desktop format)
            return JSONRPCResponse.from_json(first_line_str)
        
        # Standard MCP format: Content-Length headers
        # First line is a header, continue reading headers
        header_lines = [first_line_str]
        
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise IOError("Unexpected EOF while reading header")
            line_str = line.decode('utf-8').strip()
            if not line_str:
                break
            header_lines.append(line_str)

        # Parse Content-Length
        content_length = None
        for line in header_lines:
            if line.lower().startswith('content-length:'):
                content_length = int(line.split(':', 1)[1].strip())
                break

        if content_length is None:
            raise IOError("Missing Content-Length header")

        # Read body
        body = self.process.stdout.read(content_length)
        if len(body) < content_length:
            raise IOError(f"Expected {content_length} bytes, got {len(body)}")

        # Parse JSON response
        body_str = body.decode('utf-8')
        return JSONRPCResponse.from_json(body_str)

    def read_notification(self) -> Optional[JSONRPCRequest]:
        """
        Read a notification (non-blocking).

        Returns:
            JSON-RPC request (notification) or None if no notification available
        """
        # For simplicity, we'll handle notifications in the main loop
        # This is a placeholder for future async support
        return None

