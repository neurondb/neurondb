"""
Configuration loader for Claude Desktop format.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class MCPConfig:
    """MCP server configuration."""

    def __init__(
        self,
        command: str,
        env: Optional[Dict[str, str]] = None,
        args: Optional[list] = None
    ):
        self.command = command
        self.env = env or {}
        self.args = args or []

    def get_env(self) -> Dict[str, str]:
        """Get environment variables, merging with current environment."""
        merged = os.environ.copy()
        merged.update(self.env)
        return merged


def load_config(config_path: str, server_name: str = "neurondb") -> MCPConfig:
    """
    Load MCP configuration from Claude Desktop format.

    Args:
        config_path: Path to configuration file
        server_name: Name of the server in the config

    Returns:
        MCPConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If server not found or invalid config format
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, 'r') as f:
        config_data = json.load(f)

    # Claude Desktop format: { "mcpServers": { "server_name": { ... } } }
    if "mcpServers" not in config_data:
        raise ValueError("Invalid configuration format: missing 'mcpServers'")

    servers = config_data["mcpServers"]
    if server_name not in servers:
        available = ", ".join(servers.keys())
        raise ValueError(
            f"Server '{server_name}' not found in configuration. "
            f"Available servers: {available}"
        )

    server_config = servers[server_name]

    # Extract command
    if "command" not in server_config:
        raise ValueError(f"Server '{server_name}' missing 'command' field")

    command = server_config["command"]

    # Extract environment variables
    env = server_config.get("env", {})

    # Extract arguments (if any)
    args = server_config.get("args", [])

    return MCPConfig(command=command, env=env, args=args)

