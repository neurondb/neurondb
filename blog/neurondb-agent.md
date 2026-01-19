# NeuronAgent: Building Autonomous AI Agents with Persistent Memory

**[View on GitHub](https://github.com/neurondb-ai/neurondb)** | **[Download Latest Release](https://github.com/neurondb-ai/neurondb/releases)** | **[Documentation](https://neurondb.ai/docs)**

Building autonomous AI agents requires memory persistence, tool execution, and state management. Traditional approaches require separate systems for vector search, tool execution, and conversation state. Synchronizing these systems adds complexity. NeuronAgent provides a unified platform for building agents with persistent memory, tool execution, and multi-agent collaboration.

This guide explains NeuronAgent. We cover what it is, why you need it, its capabilities, and how to use it with Docker. We include a complete working example you copy and run.

## What is NeuronAgent

NeuronAgent is an AI agent runtime system. It provides REST API and WebSocket endpoints for building autonomous agent applications. Agents execute tasks using large language models. They persist conversations across sessions. They execute tools like SQL queries and HTTP requests. They remember context using vector search.

NeuronAgent integrates with NeuronDB PostgreSQL extension. This integration provides vector search, embedding generation, and machine learning functions. Agents store memories as vector embeddings. They retrieve relevant context using similarity search. They maintain state across conversations.

The system includes a complete agent runtime. It manages agent state machines. It handles session management. It provides tool execution. It includes a workflow engine for complex task orchestration. It supports multi-agent collaboration where agents communicate and delegate tasks.

NeuronAgent exposes a REST API on port 8080. You create agents through API calls. You manage sessions through API calls. You send messages through API calls. The system streams responses through WebSocket connections. This enables real-time interaction with agents.

## Why We Need NeuronAgent

Building agent systems requires solving several problems. Agents need persistent memory to remember past conversations. They need tool execution to interact with external systems. They need state management to track conversation context. They need workflow orchestration for complex tasks.

Traditional approaches use separate systems. Vector databases store embeddings. Tool execution frameworks handle external calls. Conversation systems manage state. These systems require synchronization. Data moves between systems. Queries span multiple databases. Complexity increases with scale.

NeuronAgent solves these problems with a unified platform. Memory stores in PostgreSQL. Vector search uses HNSW indexes in the same database. Tool execution runs in the agent runtime. Session state persists in PostgreSQL. All data lives in one place.

The platform includes built-in capabilities. Sixteen tools come pre-configured. SQL tool executes database queries. HTTP tool calls external APIs. Code tool runs Python scripts. Shell tool executes system commands. Browser tool automates web interactions. Memory tool stores and retrieves context. These tools integrate directly with the agent runtime.

The system supports production deployments. It includes authentication with API keys. It provides rate limiting. It logs all operations. It tracks costs per agent. It monitors performance with Prometheus metrics. It scales horizontally with multiple instances.

Use cases include research assistants that gather information from multiple sources. Data analysis agents query databases and generate reports. Customer support agents search knowledge bases and answer questions. Automation agents execute workflows with human approval gates. Multi-agent systems coordinate complex tasks across specialized agents.

## Capabilities

NeuronAgent provides comprehensive capabilities for building agent systems. These capabilities organize into core runtime features, tool execution, memory management, workflow orchestration, and multi-agent collaboration.

### Agent Runtime

The agent runtime manages agent execution. It implements a state machine for autonomous task execution. Agents move through states as they process tasks. The runtime loads context from memory. It selects appropriate tools. It executes tools and processes results. It updates memory with new information.

Session management handles conversation context. Each conversation uses a session. Sessions persist across restarts. The system loads conversation history when sessions resume. It manages multiple concurrent sessions. It isolates sessions for security.

Context management retrieves relevant information. The system searches memory using vector similarity. It loads recent conversation messages. It combines retrieved context with current conversation. This enables agents to maintain awareness of past interactions.

Prompt engineering constructs prompts for language models. The system includes conversation history. It includes retrieved context from memory. It includes available tools. It includes execution results. The runtime sends complete prompts to language models for generation.

### Tool System

NeuronAgent includes sixteen built-in tools. Tools execute external operations on behalf of agents. Agents call tools based on task requirements. The system validates tool parameters. It executes tools in sandboxed environments. It returns results to agents.

Core tools include SQL for database queries. The SQL tool executes read-only queries against PostgreSQL. It prevents destructive operations. It returns structured results. HTTP tool makes requests to external APIs. It supports GET, POST, PUT, and DELETE methods. It handles authentication. It processes JSON responses.

Code tool executes Python code in sandboxes. Sandboxes isolate execution. They prevent file system access. They limit network access. Shell tool executes whitelisted system commands. Browser tool automates web interactions using Playwright. Visualization tool generates charts and graphs from data.

Memory tool manages agent memory directly. Agents store facts in memory. They retrieve facts using semantic search. They update existing memories. They organize memories hierarchically. Filesystem tool provides isolated file storage per agent. Files persist across sessions.

NeuronDB integration tools access database functions. ML tool trains and runs machine learning models. Vector tool performs similarity search. RAG tool implements retrieval-augmented generation. Analytics tool performs statistical analysis. Hybrid search combines vector and text search. Reranking improves search results.

Multimodal tool processes images and media. It generates embeddings for images. It extracts text from images. It processes video frames. Collaboration tool enables agent-to-agent communication. Agents send messages to other agents. They delegate tasks. They coordinate in workspaces.

Custom tool registration extends capabilities. You define tools using JSON Schema. The system validates parameters. It executes tools through defined interfaces. This enables integration with proprietary systems.

### Memory Management

Memory management provides persistent storage for agent knowledge. The system uses HNSW indexes for fast vector similarity search. Memories store as vector embeddings. Agents retrieve relevant memories using semantic search. Memory retrieval improves response quality.

Hierarchical memory organizes knowledge at multiple levels. Working memory contains current session context. Episodic memory stores recent conversations. Semantic memory stores long-term knowledge. The system promotes important memories from working to episodic to semantic storage.

Memory promotion moves memories to long-term storage. Background workers analyze memory usage. Frequently accessed memories move to semantic storage. Important memories persist across sessions. This ensures agents retain useful knowledge.

Vector search finds similar memories quickly. HNSW indexes provide logarithmic query time. Searches return top-k similar memories. Agents use retrieved memories to inform responses. Memory summarization compresses conversation histories. Summaries preserve important information while reducing storage.

### Workflow Engine

The workflow engine orchestrates complex multi-step tasks. Workflows define as directed acyclic graphs. Steps connect through dependencies. The engine executes steps in parallel when possible. It waits for dependencies when required.

Step types include agent steps that execute agents. Tool steps execute specific tools. HTTP steps make API calls. SQL steps run database queries. Approval steps require human confirmation. Conditional steps branch based on results.

Dependency management ensures correct execution order. Steps declare dependencies on other steps. The engine builds an execution graph. It executes independent steps in parallel. It sequences dependent steps sequentially.

Input and output mapping transforms data between steps. Steps receive outputs from previous steps as inputs. The system transforms data formats. It extracts relevant fields. It combines multiple outputs.

Compensation steps roll back operations on failure. Failed workflows trigger compensation. Compensation steps undo completed operations. This maintains data consistency. Human-in-the-loop enables approval gates. Workflows pause for human review. Approval notifications send via email or webhook. Humans approve or reject workflows.

Idempotency prevents duplicate execution. Steps use execution keys. The system caches step results. Re-execution returns cached results. Retry logic handles transient failures. Steps retry with exponential backoff. Configurable retry policies control behavior.

Workflow scheduling executes workflows on schedules. Cron expressions define schedules. Workflows run automatically at specified times. The system tracks execution history. Monitoring provides real-time status updates.

### Multi-Agent Collaboration

Multi-agent collaboration enables agents to work together. Agents communicate through message passing. They delegate tasks to specialized agents. They coordinate in shared workspaces. They form hierarchical structures with parent and child agents.

Workspaces provide shared contexts for collaboration. Agents join workspaces. They share memory within workspaces. They coordinate task execution. Workspace isolation separates different projects. Participant management adds users and agents to workspaces.

Task delegation routes tasks to appropriate agents. Agents discover available agents. They select agents based on capabilities. They send tasks with required context. Delegated agents execute tasks and return results.

Hierarchical structures organize agents in trees. Parent agents coordinate child agents. Child agents execute subtasks. Results aggregate upward. This enables complex task decomposition.

### Integration with NeuronDB

NeuronAgent integrates directly with NeuronDB. The integration provides vector operations, embedding generation, and machine learning functions. Agents access NeuronDB functions through dedicated tools.

Vector operations perform similarity search. Agents find similar content in memory. They retrieve relevant context. They organize knowledge spatially. Embedding generation creates vectors from text. Agents embed user queries. They embed stored content. They compare embeddings for similarity.

Machine learning functions train and run models. Agents analyze data patterns. They make predictions. They classify content. RAG operations implement retrieval-augmented generation. Agents retrieve relevant documents. They generate answers using retrieved context.

## How to Use with Docker

Docker simplifies NeuronAgent deployment. The repository includes docker-compose configuration. You start services with single commands. This section covers setup from start to finish.

### Prerequisites

You need Docker and Docker Compose installed. Docker version 20.10 or later works. Docker Compose version 2.0 or later works. Verify installation with these commands:

```bash
docker --version
docker compose version
```

You need port 8080 available for NeuronAgent. You need port 5433 available for NeuronDB if connecting from the host. Check port availability:

```bash
# Check if ports are available (macOS/Linux)
lsof -i :8080
lsof -i :5433
```

### Step 1: Start NeuronDB

NeuronAgent requires NeuronDB running. NeuronDB is a PostgreSQL extension. Start it first using docker-compose:

```bash
# From repository root
docker compose up -d neurondb

# Wait for service to be healthy
docker compose ps neurondb
```

Wait until the service shows healthy status. This takes about thirty seconds. The service creates the database and enables the NeuronDB extension.

### Step 2: Start NeuronAgent

Start NeuronAgent using docker-compose:

```bash
# From repository root
docker compose up -d neuronagent

# Check status
docker compose ps neuronagent
```

NeuronAgent connects to NeuronDB automatically. It runs database migrations on startup. The service starts in about ten seconds. Verify it is running:

```bash
# Check health endpoint
curl http://localhost:8080/health
```

Expected response:

```json
{"status":"ok"}
```

### Step 3: Generate API Key

NeuronAgent requires API key authentication. Generate an API key:

```bash
# Navigate to NeuronAgent directory
cd NeuronAgent

# Generate API key
./scripts/neuronagent-generate-keys.sh

# Set environment variable
export NEURONAGENT_API_KEY=$(cat scripts/neuronagent_api_key.txt)
```

The script creates a file with the API key. Set the environment variable for use in commands. Alternatively, use the API key directly in Authorization headers.

### Step 4: Verify Installation

Test the API with authentication:

```bash
# List agents (should return empty array initially)
curl -H "Authorization: Bearer $NEURONAGENT_API_KEY" \
  http://localhost:8080/api/v1/agents
```

Expected response:

```json
[]
```

An empty array indicates successful authentication. The system is ready to create agents.

## Complete Use Case: Research Assistant Agent

This section implements a research assistant agent. The agent queries a database using SQL. It fetches external data using HTTP. It stores findings in memory. Follow these steps exactly to see it work.

### Step 1: Prepare Database

Create a sample table with research documents:

```bash
# Connect to database
docker exec -it neurondb-cpu psql -U neurondb -d neurondb

# Create table and insert sample data
CREATE TABLE research_documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    category TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO research_documents (title, content, category) VALUES
('Machine Learning Basics', 'Machine learning is a subset of artificial intelligence that enables systems to learn from data.', 'technology'),
('Database Design', 'Proper database design ensures data integrity and query performance.', 'technology'),
('API Development', 'RESTful APIs provide standardized interfaces for application integration.', 'development');

# Exit database
\q
```

### Step 2: Create Research Agent

Create an agent with SQL and HTTP tools:

```bash
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Authorization: Bearer $NEURONAGENT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "research-assistant",
    "system_prompt": "You are a research assistant. You help users find information by querying databases and fetching external data. Always provide accurate, well-sourced information.",
    "model_name": "gpt-4",
    "enabled_tools": ["sql", "http", "memory"],
    "config": {
      "temperature": 0.7,
      "max_tokens": 2000
    }
  }' | jq .
```

Expected response includes agent ID:

```json
{
  "id": "agent_abc123",
  "name": "research-assistant",
  "system_prompt": "You are a research assistant...",
  "model_name": "gpt-4",
  "enabled_tools": ["sql", "http", "memory"],
  "created_at": "2024-01-15T10:30:00Z"
}
```

Save the agent ID for next steps:

```bash
export AGENT_ID="agent_abc123"  # Use the actual ID from response
```

### Step 3: Create Session

Create a conversation session:

```bash
curl -X POST http://localhost:8080/api/v1/sessions \
  -H "Authorization: Bearer $NEURONAGENT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "'$AGENT_ID'",
    "metadata": {
      "user": "researcher",
      "project": "technology-research"
    }
  }' | jq .
```

Expected response:

```json
{
  "id": "session_xyz789",
  "agent_id": "agent_abc123",
  "created_at": "2024-01-15T10:35:00Z",
  "metadata": {
    "user": "researcher",
    "project": "technology-research"
  }
}
```

Save the session ID:

```bash
export SESSION_ID="session_xyz789"  # Use the actual ID from response
```

### Step 4: Send Research Query

Ask the agent to find information:

```bash
curl -X POST http://localhost:8080/api/v1/sessions/$SESSION_ID/messages \
  -H "Authorization: Bearer $NEURONAGENT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Find all documents about machine learning from the database and summarize them for me.",
    "metadata": {
      "query_type": "research"
    }
  }' | jq .
```

The agent processes the request. It uses the SQL tool to query the database. It retrieves matching documents. It generates a summary. Response includes the agent's answer and tool execution details.

Expected response structure:

```json
{
  "id": "msg_def456",
  "session_id": "session_xyz789",
  "role": "assistant",
  "content": "I found the following document about machine learning:\n\n**Machine Learning Basics**\n\nMachine learning is a subset of artificial intelligence that enables systems to learn from data. This technology allows computers to improve their performance on tasks through experience...",
  "created_at": "2024-01-15T10:40:00Z",
  "metadata": {
    "tool_calls": [
      {
        "tool": "sql",
        "query": "SELECT * FROM research_documents WHERE category = 'technology' AND content ILIKE '%machine learning%'",
        "result": "..."
      }
    ]
  }
}
```

### Step 5: Test HTTP Tool

Test external data fetching:

```bash
curl -X POST http://localhost:8080/api/v1/sessions/$SESSION_ID/messages \
  -H "Authorization: Bearer $NEURONAGENT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Fetch information about PostgreSQL from https://www.postgresql.org/about/ and summarize the key points."
  }' | jq .
```

The agent uses the HTTP tool to fetch the webpage. It extracts relevant information. It provides a summary. Note that HTTP tool access may require allowlist configuration in production.

### Step 6: Verify Memory Storage

Check that the agent stored information in memory:

```bash
curl -X GET "http://localhost:8080/api/v1/sessions/$SESSION_ID/memory?limit=10" \
  -H "Authorization: Bearer $NEURONAGENT_API_KEY" | jq .
```

This returns memories stored during the conversation. Memories include vector embeddings for semantic search. Future queries retrieve relevant memories automatically.

### Complete Python Script

For programmatic use, here is a complete Python script:

```python
#!/usr/bin/env python3
import os
import requests
import json

API_BASE = "http://localhost:8080/api/v1"
API_KEY = os.getenv("NEURONAGENT_API_KEY")
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Create agent
agent_data = {
    "name": "research-assistant",
    "system_prompt": "You are a research assistant. Query databases and fetch external data to answer questions.",
    "model_name": "gpt-4",
    "enabled_tools": ["sql", "http", "memory"]
}

response = requests.post(f"{API_BASE}/agents", headers=HEADERS, json=agent_data)
agent = response.json()
print(f"Created agent: {agent['id']}")

# Create session
session_data = {"agent_id": agent['id']}
response = requests.post(f"{API_BASE}/sessions", headers=HEADERS, json=session_data)
session = response.json()
print(f"Created session: {session['id']}")

# Send research query
message_data = {
    "content": "Find all documents about machine learning from the database and summarize them."
}

response = requests.post(
    f"{API_BASE}/sessions/{session['id']}/messages",
    headers=HEADERS,
    json=message_data
)
message = response.json()
print(f"Agent response: {message['content']}")

# Send HTTP query
message_data = {
    "content": "Fetch information about PostgreSQL and summarize key points."
}

response = requests.post(
    f"{API_BASE}/sessions/{session['id']}/messages",
    headers=HEADERS,
    json=message_data
)
message = response.json()
print(f"Agent response: {message['content']}")
```

Save this as `research_agent.py`. Run it:

```bash
export NEURONAGENT_API_KEY="your_api_key_here"
python3 research_agent.py
```

## Troubleshooting

Common issues and solutions:

### Database Connection Failed

NeuronAgent fails to connect to NeuronDB. Check that NeuronDB is running:

```bash
docker compose ps neurondb
```

Verify database connection from host:

```bash
docker exec -it neurondb-cpu psql -U neurondb -d neurondb -c "SELECT 1;"
```

Check NeuronAgent logs:

```bash
docker compose logs neuronagent
```

Ensure environment variables match:

```bash
docker compose exec neuronagent env | grep DB_
```

### API Authentication Error

Requests return 401 Unauthorized. Verify API key is set:

```bash
echo $NEURONAGENT_API_KEY
```

Test authentication:

```bash
curl -H "Authorization: Bearer $NEURONAGENT_API_KEY" \
  http://localhost:8080/api/v1/agents
```

Generate new API key if needed:

```bash
cd NeuronAgent
./scripts/neuronagent-generate-keys.sh
export NEURONAGENT_API_KEY=$(cat scripts/neuronagent_api_key.txt)
```

### Service Not Starting

NeuronAgent container exits immediately. Check logs:

```bash
docker compose logs neuronagent
```

Common causes include missing NeuronDB, incorrect database credentials, or port conflicts. Verify NeuronDB is healthy before starting NeuronAgent:

```bash
docker compose up -d neurondb
sleep 30
docker compose ps neurondb
docker compose up -d neuronagent
```

### Tool Execution Fails

Agents report tool execution errors. Check tool configuration in agent definition. Verify SQL tool has access to required tables. Verify HTTP tool allowlist includes required domains. Check tool execution logs in agent responses.

For SQL tool, ensure database user has SELECT permissions:

```sql
GRANT SELECT ON research_documents TO neurondb;
```

For HTTP tool, configure allowlist in agent config or system settings.

### Memory Not Persisting

Memories do not persist across sessions. Verify memory table exists. Check that agent has memory tool enabled. Verify vector search indexes are created. Check database for memory records:

```sql
SELECT COUNT(*) FROM neurondb_agent.memories;
```

## Next Steps

You have NeuronAgent running. You created a research assistant agent. Explore additional features:

- Create workflows for multi-step research tasks
- Set up multi-agent collaboration for complex projects
- Configure human-in-the-loop approval for sensitive operations
- Integrate custom tools for specialized use cases
- Monitor agent performance with Prometheus metrics
- Set up budget controls for cost management

Read the complete documentation at [neurondb.ai/docs/neuronagent](https://neurondb.ai/docs/neuronagent) for advanced features and production deployment guides.
