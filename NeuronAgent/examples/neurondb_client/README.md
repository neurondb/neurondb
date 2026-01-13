# NeuronAgent Python SDK

A comprehensive Python client library for NeuronAgent - AI agent runtime system with long-term memory, tool execution, and streaming capabilities.

## Features

- ✅ **Complete API Coverage** - All NeuronAgent endpoints supported
- ✅ **Type-Safe** - Full type hints throughout
- ✅ **Modular Design** - Organized by feature area
- ✅ **Error Handling** - Comprehensive exception handling
- ✅ **WebSocket Support** - Real-time streaming responses
- ✅ **Retry Logic** - Automatic retries with exponential backoff
- ✅ **Connection Pooling** - Efficient HTTP connection management
- ✅ **Metrics Collection** - Built-in metrics tracking

## Installation

### From Source

```bash
cd NeuronAgent/examples
pip install -e .
```

### Requirements

- Python 3.8+
- `requests` - HTTP client
- `websocket-client` - WebSocket support

```bash
pip install requests websocket-client
```

## Quick Start

### Basic Usage

```python
from neurondb_client import NeuronAgentClient, AgentManager, ConversationManager

# Initialize client
client = NeuronAgentClient(
    base_url="http://localhost:8080",
    api_key="your_api_key_here"
)

# Create an agent
agent_mgr = AgentManager(client)
agent = agent_mgr.create(
    name="my-assistant",
    system_prompt="You are a helpful assistant.",
    model_name="gpt-4",
    enabled_tools=['sql', 'http']
)

# Start a conversation
conversation = ConversationManager(client, agent_id=agent['id'])
conversation.start()

# Send a message
response = conversation.send("Hello! What can you help me with?")
print(response)
```

### Using Environment Variables

```bash
export NEURONAGENT_BASE_URL=http://localhost:8080
export NEURONAGENT_API_KEY=your_api_key_here
```

```python
from neurondb_client import NeuronAgentClient

# Client automatically reads from environment
client = NeuronAgentClient()
```

## Core Components

### NeuronAgentClient

Low-level HTTP client with retry logic and connection pooling.

```python
from neurondb_client import NeuronAgentClient

client = NeuronAgentClient(
    base_url="http://localhost:8080",
    api_key="your_api_key",
    max_retries=3,
    timeout=30
)

# Health check
if client.health_check():
    print("Server is healthy")

# Direct API calls
agents = client.get('/api/v1/agents')
agent = client.post('/api/v1/agents', json_data={...})
```

### Exception Handling

```python
from neurondb_client import (
    AuthenticationError,
    NotFoundError,
    ServerError,
    ConnectionError
)

try:
    agent = agent_mgr.get("invalid-id")
except AuthenticationError:
    print("Invalid API key")
except NotFoundError:
    print("Agent not found")
except ServerError as e:
    print(f"Server error: {e.status_code}")
except ConnectionError:
    print("Failed to connect to server")
```

## Module Reference

### Agent Management

Create, manage, and configure AI agents.

```python
from neurondb_client import AgentManager, AgentProfile

agent_mgr = AgentManager(client)

# Create agent
agent = agent_mgr.create(
    name="research-agent",
    system_prompt="You are a research assistant.",
    model_name="gpt-4",
    enabled_tools=['http', 'sql'],
    config={'temperature': 0.7}
)

# List agents
agents = agent_mgr.list()

# Find by name
agent = agent_mgr.find_by_name("research-agent")

# Update agent
updated = agent_mgr.update(
    agent['id'],
    description="Updated description",
    config={'temperature': 0.8}
)

# Delete agent
agent_mgr.delete(agent['id'])

# Use profiles
profile = AgentProfile(
    name="data-analyst",
    system_prompt="You are a data analyst.",
    model_name="gpt-4",
    enabled_tools=['sql'],
    config={'temperature': 0.2}
)
agent = agent_mgr.create_from_profile(profile)
```

### Session Management

Manage conversation sessions and messages.

```python
from neurondb_client import SessionManager

session_mgr = SessionManager(client)

# Create session
session = session_mgr.create(
    agent_id=agent['id'],
    external_user_id="user-123",
    metadata={'source': 'web'}
)

# Send message
response = session_mgr.send_message(
    session_id=session['id'],
    content="What is machine learning?",
    role="user"
)

# Get messages
messages = session_mgr.get_messages(session['id'], limit=100)

# List sessions for agent
sessions = session_mgr.list_for_agent(agent['id'])
```

### Conversation Management

High-level conversation handling with history tracking.

```python
from neurondb_client import ConversationManager

conversation = ConversationManager(
    client=client,
    agent_id=agent['id'],
    external_user_id="user-123"
)

# Start conversation
conversation.start()

# Send messages
response1 = conversation.send("Hello!")
response2 = conversation.send("What can you do?")

# Get history
history = conversation.get_history()
for exchange in history:
    print(f"User: {exchange['user']}")
    print(f"Assistant: {exchange['assistant']}")
    print(f"Tokens: {exchange.get('tokens', 0)}")

# Stream responses
def on_chunk(chunk):
    print(chunk, end='', flush=True)

full_response = conversation.stream(
    message="Explain neural networks in detail",
    on_chunk=on_chunk
)

# Get total tokens
total_tokens = conversation.get_total_tokens()

# Close conversation
conversation.close()
```

### Tools Management

List, create, and manage tools.

```python
from neurondb_client import ToolManager

tool_mgr = ToolManager(client)

# List all tools
tools = tool_mgr.list()

# Get tool details
sql_tool = tool_mgr.get("sql")

# Create custom tool
custom_tool = tool_mgr.create(
    name="custom-api",
    description="Custom API integration",
    tool_type="http",
    config={'base_url': 'https://api.example.com'}
)

# Update tool
updated = tool_mgr.update(
    "custom-api",
    description="Updated description"
)

# Delete tool
tool_mgr.delete("custom-api")
```

### Workflow Management

Manage workflow schedules and execution.

```python
from neurondb_client import WorkflowManager

workflow_mgr = WorkflowManager(client)

# Create workflow schedule
schedule = workflow_mgr.create_schedule(
    workflow_id="workflow-123",
    cron_expression="0 0 * * *",  # Daily at midnight
    timezone="UTC",
    enabled=True
)

# Get schedule
schedule = workflow_mgr.get_schedule("workflow-123")

# Update schedule
updated = workflow_mgr.update_schedule(
    "workflow-123",
    cron_expression="0 */6 * * *",  # Every 6 hours
    enabled=False
)

# List all schedules
schedules = workflow_mgr.list_schedules()

# Delete schedule
workflow_mgr.delete_schedule("workflow-123")
```

### Budget Management

Track and manage agent budgets.

```python
from neurondb_client import BudgetManager
from datetime import datetime

budget_mgr = BudgetManager(client)

# Get budget status
budget = budget_mgr.get_budget(
    agent_id=agent['id'],
    period_type="monthly"
)
print(f"Budget: ${budget.get('budget_amount', 0)}")
print(f"Spent: ${budget.get('spent', 0)}")
print(f"Remaining: ${budget.get('remaining', 0)}")

# Set budget
budget = budget_mgr.set_budget(
    agent_id=agent['id'],
    budget_amount=1000.0,
    period_type="monthly",
    start_date=datetime.now()
)

# Update budget
updated = budget_mgr.update_budget(
    agent_id=agent['id'],
    budget_amount=1500.0,
    period_type="monthly"
)
```

### Evaluation Framework

Create and execute evaluation tasks.

```python
from neurondb_client import EvaluationManager

eval_mgr = EvaluationManager(client)

# Create evaluation task
task = eval_mgr.create_task(
    task_type="end_to_end",
    input_text="What is 2+2?",
    expected_output="4",
    metadata={'category': 'math'}
)

# List tasks
tasks = eval_mgr.list_tasks(task_type="end_to_end")

# Create evaluation run
run = eval_mgr.create_run(
    dataset_version="v1.0",
    agent_id=agent['id'],
    total_tasks=100
)

# Execute run
result = eval_mgr.execute_run(run['id'])

# Get results
results = eval_mgr.get_results(run['id'])
for result in results:
    print(f"Task: {result['eval_task_id']}")
    print(f"Passed: {result['passed']}")
    print(f"Score: {result.get('score', 0)}")
```

### Replay & Snapshots

Create snapshots and replay executions.

```python
from neurondb_client import ReplayManager

replay_mgr = ReplayManager(client)

# Create snapshot
snapshot = replay_mgr.create_snapshot(
    session_id=session['id'],
    user_message="Analyze this data",
    deterministic_mode=False
)

# List snapshots
snapshots = replay_mgr.list_by_session(session['id'])
agent_snapshots = replay_mgr.list_by_agent(agent['id'])

# Get snapshot
snapshot = replay_mgr.get(snapshot['id'])

# Replay execution
replay_result = replay_mgr.replay(snapshot['id'])
print(f"Final answer: {replay_result['final_answer']}")
print(f"Tool calls: {len(replay_result['tool_calls'])}")
print(f"Tokens used: {replay_result['tokens_used']}")

# Delete snapshot
replay_mgr.delete(snapshot['id'])
```

### Memory Management

Search and manage agent memory.

```python
from neurondb_client import MemoryManager

memory_mgr = MemoryManager(client)

# List memory chunks
chunks = memory_mgr.list_chunks(
    agent_id=agent['id'],
    limit=50,
    offset=0
)

# Search memory
results = memory_mgr.search(
    agent_id=agent['id'],
    query="machine learning concepts",
    top_k=10
)

for result in results:
    print(f"Content: {result['content']}")
    print(f"Similarity: {result.get('similarity', 0)}")

# Get memory chunk
chunk = memory_mgr.get_chunk(chunk_id=123)

# Summarize memory
summary = memory_mgr.summarize(
    memory_id=123,
    max_length=200
)

# Delete memory chunk
memory_mgr.delete_chunk(chunk_id=123)
```

### Webhooks

Create and manage webhooks for event notifications.

```python
from neurondb_client import WebhookManager

webhook_mgr = WebhookManager(client)

# Create webhook
webhook = webhook_mgr.create(
    url="https://example.com/webhook",
    events=["message.sent", "agent.created", "session.created"],
    secret="webhook_secret",
    enabled=True,
    timeout_seconds=30,
    retry_count=3
)

# List webhooks
webhooks = webhook_mgr.list()

# Get webhook
webhook = webhook_mgr.get(webhook['id'])

# Update webhook
updated = webhook_mgr.update(
    webhook['id'],
    events=["message.sent", "agent.created"],
    enabled=False
)

# List deliveries
deliveries = webhook_mgr.list_deliveries(
    webhook_id=webhook['id'],
    limit=50
)

# Delete webhook
webhook_mgr.delete(webhook['id'])
```

### Virtual Filesystem

Manage files in agent's virtual filesystem.

```python
from neurondb_client import VFSManager

vfs_mgr = VFSManager(client)

# Create file
file = vfs_mgr.create_file(
    agent_id=agent['id'],
    path="/documents/readme.txt",
    content="Hello, World!",
    mime_type="text/plain"
)

# List files
listing = vfs_mgr.list_files(
    agent_id=agent['id'],
    path="/documents"
)

# Read file
file = vfs_mgr.read_file(
    agent_id=agent['id'],
    path="/documents/readme.txt"
)
print(file['content'])

# Write file
vfs_mgr.write_file(
    agent_id=agent['id'],
    path="/documents/readme.txt",
    content="Updated content"
)

# Copy file
copied = vfs_mgr.copy_file(
    agent_id=agent['id'],
    source_path="/documents/readme.txt",
    dest_path="/backup/readme.txt"
)

# Move file
moved = vfs_mgr.move_file(
    agent_id=agent['id'],
    source_path="/documents/readme.txt",
    dest_path="/archive/readme.txt"
)

# Delete file
vfs_mgr.delete_file(
    agent_id=agent['id'],
    path="/documents/readme.txt"
)
```

### Specializations

Manage agent specializations.

```python
from neurondb_client import SpecializationManager

spec_mgr = SpecializationManager(client)

# Create specialization
spec = spec_mgr.create(
    agent_id=agent['id'],
    specialization_type="coding",
    capabilities=["python", "javascript", "sql"],
    config={'max_complexity': 'high'}
)

# Get specialization
spec = spec_mgr.get(agent['id'])

# List specializations
specs = spec_mgr.list(specialization_type="coding")

# Update specialization
updated = spec_mgr.update(
    agent['id'],
    capabilities=["python", "javascript", "sql", "go"]
)

# Delete specialization
spec_mgr.delete(agent['id'])
```

### Plans & Reflections

Manage agent plans and reflections.

```python
from neurondb_client import PlanManager

plan_mgr = PlanManager(client)

# List plans
plans = plan_mgr.list(
    agent_id=agent['id'],
    session_id=session['id'],
    limit=50
)

# Get plan
plan = plan_mgr.get(plan_id="plan-123")

# Update plan status
updated = plan_mgr.update_status(
    plan_id="plan-123",
    status="completed",
    result={'tasks_completed': 5, 'success': True}
)
```

### Collaboration

Manage collaboration workspaces.

```python
from neurondb_client import CollaborationManager

collab_mgr = CollaborationManager(client)

# Create workspace
workspace = collab_mgr.create_workspace(
    name="Team Workspace",
    description="Shared workspace for team collaboration"
)

# Get workspace
workspace = collab_mgr.get_workspace(workspace['workspace_id'])

# Add participant
participant = collab_mgr.add_participant(
    workspace_id=workspace['workspace_id'],
    role="member",
    agent_id=agent['id']
)
```

## WebSocket Streaming

Stream agent responses in real-time.

```python
from neurondb_client.core.websocket import WebSocketClient

ws = WebSocketClient("http://localhost:8080", "your_api_key")

def on_message(data):
    if data.get('type') == 'chunk':
        print(data.get('content', ''), end='', flush=True)
    elif data.get('type') == 'response':
        print(f"\nComplete: {data.get('content', '')}")
    elif data.get('type') == 'error':
        print(f"\nError: {data.get('error', '')}")

ws.stream_message(
    session_id=session['id'],
    content="Explain quantum computing",
    on_message=on_message
)
```

## Utilities

### Configuration

```python
from neurondb_client import ConfigLoader

loader = ConfigLoader()

# Load from file
config = loader.load_from_file('config.json')

# Get environment variable
api_key = loader.get_env('NEURONAGENT_API_KEY')

# Load agent config
agent_config = loader.load_agent_config(
    'agent_configs.json',
    'research_assistant'
)
```

### Logging

```python
from neurondb_client import setup_logging, get_logger

# Setup logging
setup_logging(
    level="INFO",
    enable_file=True,
    filepath="app.log"
)

# Get logger
logger = get_logger(__name__)
logger.info("Application started")
```

### Metrics

```python
from neurondb_client import MetricsCollector

metrics = MetricsCollector()

# Record metrics
metrics.record("request_duration", 0.5)
metrics.increment("requests", 1)
metrics.timer("api_call", 0.3)

# Get summary
summary = metrics.get_summary()
print(summary['counters'])
print(summary['timers'])

# Reset
metrics.reset()
```

## Error Handling Best Practices

```python
from neurondb_client import (
    NeuronAgentError,
    AuthenticationError,
    NotFoundError,
    ServerError,
    ConnectionError,
    TimeoutError
)

try:
    response = session_mgr.send_message(...)
except AuthenticationError:
    # Handle auth failure - refresh API key
    print("Authentication failed. Check your API key.")
except NotFoundError:
    # Handle not found - create resource
    print("Resource not found.")
except ServerError as e:
    # Handle server errors - retry or log
    if e.status_code >= 500:
        # Retry logic
        print(f"Server error: {e.status_code}")
except ConnectionError:
    # Handle connection issues
    print("Failed to connect to server.")
except TimeoutError:
    # Handle timeouts
    print("Request timed out.")
except NeuronAgentError as e:
    # Catch-all for other errors
    print(f"Error: {e}")
```

## Advanced Usage

### Custom Retry Logic

```python
from neurondb_client import NeuronAgentClient
import time

client = NeuronAgentClient(max_retries=5)

def send_with_custom_retry(message, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            return session_mgr.send_message(...)
        except ServerError:
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
```

### Batch Operations

```python
def create_multiple_agents(profiles):
    agent_mgr = AgentManager(client)
    agents = []
    
    for profile in profiles:
        try:
            agent = agent_mgr.create_from_profile(profile)
            agents.append(agent)
        except Exception as e:
            logger.error(f"Failed to create {profile.name}: {e}")
    
    return agents
```

### Monitoring

```python
from neurondb_client import MetricsCollector

class MonitoredConversation:
    def __init__(self, conversation, metrics):
        self.conversation = conversation
        self.metrics = metrics
    
    def send(self, message):
        start = time.time()
        try:
            response = self.conversation.send(message)
            self.metrics.timer("send_duration", time.time() - start)
            self.metrics.increment("successful_sends")
            return response
        except Exception as e:
            self.metrics.increment("failed_sends")
            raise
```

## Complete Example

```python
from neurondb_client import (
    NeuronAgentClient,
    AgentManager,
    ConversationManager,
    BudgetManager,
    MemoryManager
)

# Initialize
client = NeuronAgentClient()
agent_mgr = AgentManager(client)
budget_mgr = BudgetManager(client)
memory_mgr = MemoryManager(client)

# Create agent with budget
agent = agent_mgr.create(
    name="research-assistant",
    system_prompt="You are a research assistant.",
    model_name="gpt-4",
    enabled_tools=['http', 'sql']
)

budget_mgr.set_budget(
    agent_id=agent['id'],
    budget_amount=500.0,
    period_type="monthly"
)

# Start conversation
conversation = ConversationManager(client, agent_id=agent['id'])
conversation.start()

# Search memory for context
memory_results = memory_mgr.search(
    agent_id=agent['id'],
    query="machine learning",
    top_k=5
)

# Send message with context
context = "\n".join([r['content'] for r in memory_results])
response = conversation.send(
    f"Based on this context:\n{context}\n\nWhat is machine learning?"
)

print(response)

# Check budget
budget = budget_mgr.get_budget(agent['id'], period_type="monthly")
print(f"Budget remaining: ${budget.get('remaining', 0)}")

# Close
conversation.close()
```

## API Reference

### Core Classes

- `NeuronAgentClient` - HTTP client
- `WebSocketClient` - WebSocket client
- `AgentManager` - Agent management
- `SessionManager` - Session management
- `ConversationManager` - Conversation handling
- `ToolManager` - Tool management
- `WorkflowManager` - Workflow management
- `BudgetManager` - Budget management
- `EvaluationManager` - Evaluation framework
- `ReplayManager` - Replay and snapshots
- `SpecializationManager` - Specialization management
- `PlanManager` - Plan management
- `WebhookManager` - Webhook management
- `MemoryManager` - Memory management
- `VFSManager` - Virtual filesystem
- `CollaborationManager` - Collaboration workspaces

### Exceptions

- `NeuronAgentError` - Base exception
- `AuthenticationError` - Authentication failed
- `NotFoundError` - Resource not found
- `ServerError` - Server error
- `ValidationError` - Validation error
- `ConnectionError` - Connection failed
- `TimeoutError` - Request timeout

## Requirements

- Python 3.8+
- requests >= 2.25.0
- websocket-client >= 1.0.0

## License

See [LICENSE](../../../LICENSE) for license information.

## Support

- **Documentation**: [NeuronAgent Docs](../../docs/)
- **API Reference**: [API Documentation](../../docs/API.md)
- **Examples**: See `examples_modular/` directory
- **Issues**: Report on GitHub
- **Email**: support@neurondb.ai

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](../../../CONTRIBUTING.md) for guidelines.


