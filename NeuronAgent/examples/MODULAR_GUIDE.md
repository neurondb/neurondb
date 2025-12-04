# Modular Client Library Guide

Complete guide to using the modular NeuronAgent Python client library.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Core Components](#core-components)
5. [Agent Management](#agent-management)
6. [Session Management](#session-management)
7. [Conversation Management](#conversation-management)
8. [Utilities](#utilities)
9. [Best Practices](#best-practices)
10. [Advanced Usage](#advanced-usage)

## Installation

The client library is included in the examples directory. To use it:

```python
import sys
sys.path.insert(0, '/path/to/NeuronAgent/examples')

from neurondb_client import NeuronAgentClient
```

Or install as a package:

```bash
cd NeuronAgent/examples
pip install -e .
```

## Quick Start

```python
from neurondb_client import (
    NeuronAgentClient,
    AgentManager,
    ConversationManager
)

# Initialize
client = NeuronAgentClient()
agent_mgr = AgentManager(client)

# Create agent
agent = agent_mgr.create(
    name="my-agent",
    system_prompt="You are helpful.",
    model_name="gpt-4"
)

# Start conversation
conversation = ConversationManager(client, agent_id=agent['id'])
conversation.start()

# Send message
response = conversation.send("Hello!")
print(response)
```

## Architecture

The library is organized into modular components:

```
neurondb_client/
├── core/           # Low-level HTTP/WebSocket clients
├── agents/         # Agent management
├── sessions/       # Session management
└── utils/          # Utilities (config, logging, metrics)
```

### Component Responsibilities

- **core**: HTTP client, WebSocket client, exceptions
- **agents**: Agent CRUD, profiles, configuration
- **sessions**: Session management, message sending
- **utils**: Configuration, logging, metrics

## Core Components

### NeuronAgentClient

Low-level HTTP client with retry logic and connection pooling.

```python
from neurondb_client import NeuronAgentClient

client = NeuronAgentClient(
    base_url="http://localhost:8080",
    api_key="your_key",
    max_retries=3,
    timeout=30
)

# Health check
if client.health_check():
    print("Server is healthy")

# Make requests
agents = client.get('/api/v1/agents')
agent = client.post('/api/v1/agents', json_data={...})

# Get metrics
metrics = client.get_metrics()
print(f"Requests: {metrics['requests']}")
print(f"Errors: {metrics['errors']}")
```

### WebSocketClient

WebSocket client for streaming responses.

```python
from neurondb_client.core.websocket import WebSocketClient

ws = WebSocketClient("http://localhost:8080", "api_key")

def on_message(data):
    print(data['content'], end='', flush=True)

ws.stream_message(
    session_id="...",
    content="Hello",
    on_message=on_message
)
```

### Exceptions

Custom exceptions for error handling.

```python
from neurondb_client import (
    AuthenticationError,
    NotFoundError,
    ServerError,
    ConnectionError
)

try:
    response = client.get('/api/v1/agents/123')
except AuthenticationError:
    print("Invalid API key")
except NotFoundError:
    print("Agent not found")
except ServerError as e:
    print(f"Server error: {e.status_code}")
```

## Agent Management

### AgentManager

High-level agent operations.

```python
from neurondb_client import AgentManager

agent_mgr = AgentManager(client)

# Create agent
agent = agent_mgr.create(
    name="my-agent",
    system_prompt="You are helpful.",
    model_name="gpt-4",
    enabled_tools=['sql', 'http'],
    config={'temperature': 0.7}
)

# Get agent
agent = agent_mgr.get(agent_id)

# List agents
agents = agent_mgr.list()

# Find by name
agent = agent_mgr.find_by_name("my-agent")

# Update agent
updated = agent_mgr.update(
    agent_id,
    description="New description",
    config={'temperature': 0.8}
)

# Delete agent
agent_mgr.delete(agent_id)
```

### Agent Profiles

Predefined or custom agent configurations.

```python
from neurondb_client.agents import (
    AgentProfile,
    get_default_profile,
    load_profiles_from_file
)

# Use default profile
profile = get_default_profile('research_assistant')
agent = agent_mgr.create_from_profile(profile)

# Load from file
profiles = load_profiles_from_file('agent_configs.json')
agent = agent_mgr.create_from_profile(profiles['data_analyst'])

# Create custom profile
custom = AgentProfile(
    name="custom-agent",
    system_prompt="Custom prompt",
    model_name="gpt-4",
    enabled_tools=['sql'],
    config={'temperature': 0.5}
)
agent = agent_mgr.create_from_profile(custom)
```

## Session Management

### SessionManager

Session and message operations.

```python
from neurondb_client import SessionManager

session_mgr = SessionManager(client)

# Create session
session = session_mgr.create(
    agent_id=agent['id'],
    external_user_id="user-123",
    metadata={'source': 'web'}
)

# Get session
session = session_mgr.get(session_id)

# List sessions for agent
sessions = session_mgr.list_for_agent(agent_id, limit=50)

# Send message
response = session_mgr.send_message(
    session_id=session['id'],
    content="Hello!",
    role="user",
    stream=False
)

# Get messages
messages = session_mgr.get_messages(session_id, limit=100)
```

## Conversation Management

### ConversationManager

High-level conversation handling with history tracking.

```python
from neurondb_client import ConversationManager

# Initialize
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

# Get total tokens
total = conversation.get_total_tokens()

# Refresh from server
conversation.refresh_history(limit=100)

# Stream responses
def on_chunk(chunk):
    print(chunk, end='', flush=True)

full_response = conversation.stream(
    message="Explain neural networks",
    on_chunk=on_chunk
)

# Cleanup
conversation.close()
```

## Utilities

### Configuration

```python
from neurondb_client.utils.config import ConfigLoader

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

# Find config file
filepath = loader.find_config_file('config.json')
```

### Logging

```python
from neurondb_client.utils.logging import setup_logging, get_logger

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
from neurondb_client.utils.metrics import MetricsCollector

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

## Best Practices

### 1. Error Handling

```python
from neurondb_client import (
    AuthenticationError,
    NotFoundError,
    ServerError
)

try:
    response = session_mgr.send_message(...)
except AuthenticationError:
    # Handle auth failure - refresh API key
    pass
except NotFoundError:
    # Handle not found - create resource
    pass
except ServerError as e:
    # Handle server errors - retry or log
    if e.status_code >= 500:
        # Retry logic
        pass
```

### 2. Resource Management

```python
client = NeuronAgentClient()
try:
    # Use client
    pass
finally:
    client.close()  # Always cleanup
```

### 3. Health Checks

```python
if not client.health_check():
    raise ConnectionError("Server is not healthy")
```

### 4. Metrics Collection

```python
from neurondb_client.utils.metrics import MetricsCollector

metrics = MetricsCollector()

start = time.time()
response = conversation.send("Hello")
duration = time.time() - start

metrics.timer("message_duration", duration)
metrics.increment("messages_sent")
```

### 5. Logging

```python
from neurondb_client.utils.logging import setup_logging

setup_logging(level="INFO", enable_file=True)
```

### 6. Configuration Management

```python
from neurondb_client.utils.config import ConfigLoader

loader = ConfigLoader()
api_key = loader.get_env('NEURONAGENT_API_KEY')
config = loader.load_from_file('config.json')
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
from neurondb_client.utils.metrics import MetricsCollector

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

### Async Operations

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def send_async(conversation, message):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        response = await loop.run_in_executor(
            executor,
            conversation.send,
            message
        )
    return response

# Usage
responses = await asyncio.gather(
    send_async(conversation, "Message 1"),
    send_async(conversation, "Message 2")
)
```

## Examples

See the `examples_modular/` directory for complete working examples:

- `01_basic_usage.py` - Basic operations
- `02_agent_profiles.py` - Using profiles
- `03_conversation_manager.py` - Conversation handling
- `04_streaming.py` - WebSocket streaming
- `05_production_patterns.py` - Production patterns
- `06_advanced_agent_management.py` - Advanced operations

## Support

For issues or questions:
- Check the [main README](../README.md)
- Review [API documentation](../docs/API.md)
- See example code in `examples_modular/`

