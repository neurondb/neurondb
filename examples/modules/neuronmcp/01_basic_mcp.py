#!/usr/bin/env python3
"""
NeuronMCP Module: Basic MCP Usage
===================================
Learn how to use NeuronMCP (Model Context Protocol) server.

Run: python 01_basic_mcp.py

Requirements:
    - NeuronMCP server running
    - MCP client configured
"""

import os

print("=" * 60)
print("NeuronMCP Module: Basic MCP Usage")
print("=" * 60)

print("\nNeuronMCP provides:")
print("  - Model Context Protocol server")
print("  - PostgreSQL tools via MCP")
print("  - Claude Desktop integration")
print("  - Tool discovery and execution")

print("\n1. Starting NeuronMCP server:")
print("  cd NeuronMCP")
print("  ./start-server.sh")

print("\n2. Configuring Claude Desktop:")
print("  Edit ~/.config/Claude/claude_desktop_config.json")
print("  Add NeuronMCP server configuration")

print("\n3. Available MCP tools:")
print("  - SQL query execution")
print("  - Database schema inspection")
print("  - Vector search operations")
print("  - ML model operations")

print("\n4. For complete examples, see:")
print("  - NeuronMCP/README.md")
print("  - NeuronMCP/docs/")
print("  - examples/mcp-integration/")

print("\n5. Testing MCP connection:")
print("  python NeuronMCP/client/test_connection.py")

print("\nNote: MCP requires a compatible client (Claude Desktop, etc.)")
print("For standalone usage, use NeuronDB directly or NeuronAgent API.")

print("\n✓ Example complete!")







