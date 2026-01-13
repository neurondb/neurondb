# NeuronDB Modules - Examples by Module

Examples for each module in the NeuronDB ecosystem.

## NeuronDB

### `01_basic_usage.py`
Basic usage of the NeuronDB PostgreSQL extension.

## NeuronAgent

### `01_basic_agent.py`
Creating and using AI agents with NeuronAgent.

For more examples:
- `NeuronAgent/examples/` - Complete agent examples
- `../agent-tools/` - Agent tools integration

## NeuronMCP

### `01_basic_mcp.py`
Using NeuronMCP (Model Context Protocol) server.

For more examples:
- `NeuronMCP/docs/` - Complete MCP documentation
- `../mcp-integration/` - MCP integration examples

## NeuronDesktop

NeuronDesktop is a web interface. See:
- `NeuronDesktop/README.md` - Setup and usage
- `NeuronDesktop/docs/` - Documentation

## Quick Start

1. **NeuronDB (PostgreSQL extension):**
   ```bash
   python modules/neurondb/01_basic_usage.py
   ```

2. **NeuronAgent (Agent runtime):**
   ```bash
   # Start server first
   cd NeuronAgent && ./start_server.sh
   # Then run examples
   python modules/neuronagent/01_basic_agent.py
   ```

3. **NeuronMCP (MCP server):**
   ```bash
   # Start server first
   cd NeuronMCP && ./start-server.sh
   # Then configure MCP client
   python modules/neuronmcp/01_basic_mcp.py
   ```







