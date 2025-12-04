# Modular Examples

This directory contains detailed examples demonstrating the modular NeuronAgent client library.

## Structure

```
examples_modular/
├── 01_basic_usage.py          # Simplest usage example
├── 02_agent_profiles.py        # Using agent profiles
├── 03_conversation_manager.py # Conversation management
├── 04_streaming.py             # WebSocket streaming
├── 05_production_patterns.py   # Production-ready patterns
├── 06_advanced_agent_management.py  # Advanced agent operations
└── README.md                   # This file
```

## Client Library Structure

The modular client library is organized as follows:

```
neurondb_client/
├── __init__.py              # Main exports
├── core/                    # Core components
│   ├── client.py           # HTTP client
│   ├── websocket.py        # WebSocket client
│   └── exceptions.py       # Custom exceptions
├── agents/                  # Agent management
│   ├── manager.py          # Agent operations
│   └── profile.py          # Agent profiles
├── sessions/                # Session management
│   ├── manager.py          # Session operations
│   └── conversation.py    # Conversation handling
└── utils/                   # Utilities
    ├── config.py           # Configuration
    ├── logging.py          # Logging setup
    └── metrics.py          # Metrics collection
```

## Running Examples

### Prerequisites

```bash
# Install dependencies
pip install -r ../requirements.txt

# Set API key
export NEURONAGENT_API_KEY=your_api_key
```

### Run Examples

```bash
# Basic usage
python3 01_basic_usage.py

# Agent profiles
python3 02_agent_profiles.py

# Conversation management
python3 03_conversation_manager.py

# Streaming
python3 04_streaming.py

# Production patterns
python3 05_production_patterns.py

# Advanced agent management
python3 06_advanced_agent_management.py
```

## Example Details

### 01_basic_usage.py
Simplest example showing:
- Client initialization
- Agent creation
- Session creation
- Sending messages
- Basic metrics

### 02_agent_profiles.py
Demonstrates:
- Using default profiles
- Loading profiles from files
- Creating custom profiles
- Profile-based agent creation

### 03_conversation_manager.py
Shows:
- Conversation lifecycle
- Message history tracking
- Token usage tracking
- Context management

### 04_streaming.py
Demonstrates:
- WebSocket connection
- Streaming responses
- Chunk handling
- Completion callbacks

### 05_production_patterns.py
Production-ready patterns:
- Error handling
- Retry logic
- Metrics collection
- Logging
- Resource cleanup
- Graceful degradation

### 06_advanced_agent_management.py
Advanced operations:
- Finding agents
- Updating agents
- Listing agents
- Agent lifecycle
- Configuration management

## Usage Patterns

### Basic Pattern

```python
from neurondb_client import NeuronAgentClient, AgentManager, SessionManager

client = NeuronAgentClient()
agent_mgr = AgentManager(client)
session_mgr = SessionManager(client)

# Create agent
agent = agent_mgr.create(name="my-agent", system_prompt="...")

# Create session
session = session_mgr.create(agent_id=agent['id'])

# Send message
response = session_mgr.send_message(session_id=session['id'], content="Hello")
```

### Conversation Pattern

```python
from neurondb_client import ConversationManager

conversation = ConversationManager(client, agent_id=agent['id'])
conversation.start()

response = conversation.send("Hello!")
history = conversation.get_history()
```

### Profile Pattern

```python
from neurondb_client.agents import AgentProfile, get_default_profile

profile = get_default_profile('research_assistant')
agent = agent_mgr.create_from_profile(profile)
```

### Streaming Pattern

```python
def on_chunk(chunk):
    print(chunk, end='', flush=True)

conversation.stream(
    message="Explain neural networks",
    on_chunk=on_chunk
)
```

## Best Practices

1. **Always check server health** before operations
2. **Use ConversationManager** for multi-turn conversations
3. **Handle errors gracefully** with try/except
4. **Track metrics** for monitoring
5. **Clean up resources** in finally blocks
6. **Use profiles** for consistent agent configurations
7. **Enable logging** for debugging
8. **Set appropriate timeouts** for production

## Error Handling

```python
from neurondb_client import (
    AuthenticationError,
    NotFoundError,
    ServerError
)

try:
    response = session_mgr.send_message(...)
except AuthenticationError:
    # Handle auth failure
except NotFoundError:
    # Handle not found
except ServerError:
    # Handle server errors
```

## Metrics

```python
from neurondb_client.utils.metrics import MetricsCollector

metrics = MetricsCollector()
metrics.increment("requests")
metrics.timer("duration", 0.5)
summary = metrics.get_summary()
```

## Configuration

```python
from neurondb_client.utils.config import ConfigLoader

loader = ConfigLoader()
config = loader.load_from_file("config.json")
api_key = loader.get_env("NEURONAGENT_API_KEY")
```

## Next Steps

- Read the [main README](../README.md) for API documentation
- Check [QUICKSTART.md](../QUICKSTART.md) for quick setup
- Explore the client library source code
- Build your own custom agents!

