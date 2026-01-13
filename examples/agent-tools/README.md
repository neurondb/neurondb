# Agent Tool Execution

Example demonstrating NeuronAgent with multiple tools: SQL tool and HTTP tool.

## Overview

This example shows how to:
- Configure an agent with multiple tools
- Execute SQL queries via agent
- Call external HTTP APIs via agent
- Chain tool calls for complex tasks

## Quick Start

### Prerequisites

- NeuronAgent running on port 8080
- PostgreSQL 16+ with NeuronDB extension
- API key for NeuronAgent

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export AGENT_URL=http://localhost:8080
export AGENT_API_KEY=your_api_key_here
export DB_CONNECTION_STRING=postgresql://user:pass@localhost:5432/neurondb

# Create agent with tools
python create_agent.py

# Run example agent task
python run_agent.py "Find all documents about machine learning and fetch their summaries from the API"
```

## Files

- `create_agent.py` - Create agent with SQL and HTTP tools
- `run_agent.py` - Execute agent with example tasks
- `sql_tool.py` - SQL tool implementation
- `http_tool.py` - HTTP tool implementation
- `requirements.txt` - Python dependencies

## Tool Configuration

### SQL Tool

Allows agent to execute SQL queries on NeuronDB:

```python
{
    "name": "sql_query",
    "description": "Execute SQL queries on NeuronDB database",
    "parameters": {
        "query": "SQL query to execute"
    }
}
```

### HTTP Tool

Allows agent to call external APIs:

```python
{
    "name": "http_request",
    "description": "Make HTTP requests to external APIs",
    "parameters": {
        "url": "API endpoint URL",
        "method": "HTTP method (GET, POST, etc.)",
        "headers": "Request headers",
        "body": "Request body"
    }
}
```

## Example Tasks

1. **Data Retrieval:**
   ```
   "Find all documents with embeddings and return their IDs"
   ```

2. **API Integration:**
   ```
   "Fetch user information from the API for user ID 123"
   ```

3. **Chained Operations:**
   ```
   "Query the database for product IDs, then fetch details for each from the API"
   ```

## Related Documentation

- [NeuronAgent API](../../NeuronAgent/docs/API.md) - Complete API reference
- [NeuronAgent Architecture](../../NeuronAgent/docs/ARCHITECTURE.md) - System design
- [Tool Registry](../../NeuronAgent/readme.md#tool-registry) - Tool configuration







