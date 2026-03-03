# NeuronDB SDKs

<div align="center">

**Official client libraries for NeuronDB ecosystem components**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg)](https://www.python.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6.svg)](https://www.typescriptlang.org/)
[![Status](https://img.shields.io/badge/status-maintained-brightgreen.svg)](.)

</div>

Official client libraries that provide type-safe, easy-to-use interfaces for interacting with NeuronDB ecosystem components.

---

## 📑 Table of Contents

<details>
<summary><strong>Expand full table of contents</strong></summary>

- [Overview](#overview)
- [Available SDKs](#available-sdks)
  - [Python SDK](#python-sdk)
  - [TypeScript/JavaScript SDK](#typescriptjavascript-sdk)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Generating SDKs](#generating-sdks)
- [Versioning](#versioning)
- [Contributing](#contributing)
- [Support](#support)

</details>

---

## Overview

The NeuronDB SDKs provide:

- ✅ **Type Safety** - Full TypeScript/Python type definitions
- ✅ **Auto-generated** - Generated from OpenAPI specifications
- ✅ **Complete Coverage** - All API endpoints supported
- ✅ **Easy Integration** - Simple, intuitive APIs
- ✅ **Documentation** - Comprehensive docs and examples
- ✅ **Active Maintenance** - Regularly updated with new features

---

## Available SDKs

### Python SDK

**Package:** `neuronagent` (or `@neurondb/neuronagent`)

**Installation:**
```bash
pip install neuronagent
```

**Features:**
- Full NeuronAgent API support
- Type hints for all models
- Async/await support
- Error handling
- Request/response validation

**Documentation:** [Python SDK Docs](python/README.md)

### TypeScript/JavaScript SDK

**Packages:**
- `@neurondb/neuronagent` - NeuronAgent client
- `@neurondb/neurondesktop` - NeuronDesktop client

**Installation:**
```bash
npm install @neurondb/neuronagent @neurondb/neurondesktop
# or
yarn add @neurondb/neuronagent @neurondb/neurondesktop
```

**Features:**
- Full TypeScript support
- Browser and Node.js compatible
- Promise-based API
- Type definitions included
- Tree-shakeable

**Documentation:** [TypeScript SDK Docs](typescript/README.md)

---

## Installation

### Python

```bash
# Install from PyPI
pip install neuronagent

# Install from source
git clone https://github.com/neurondb/neurondb.git
cd neurondb/sdks/python
pip install -e .
```

### TypeScript/JavaScript

```bash
# Install from npm
npm install @neurondb/neuronagent @neurondb/neurondesktop

# Install from source
git clone https://github.com/neurondb/neurondb.git
cd neurondb/sdks/typescript
npm install
npm run build
```

---

## Quick Start

### Python Example

```python
from neuronagent import NeuronAgentClient

# Initialize client
client = NeuronAgentClient(
    base_url="http://localhost:8080",
    api_key="your-api-key"
)

# Create an agent
agent = client.agents.create(
    name="my-agent",
    system_prompt="You are a helpful assistant",
    model_name="gpt-4",
    enabled_tools=["sql", "http"]
)

print(f"Created agent: {agent.id}")

# Create a session
session = client.sessions.create(agent_id=agent.id)

# Send a message
response = client.sessions.send_message(
    session_id=session.id,
    content="Hello, agent!"
)

print(f"Agent response: {response.content}")
```

### TypeScript Example

```typescript
import { NeuronAgentClient } from '@neurondb/neuronagent';

// Initialize client
const client = new NeuronAgentClient({
  baseURL: 'http://localhost:8080',
  apiKey: 'your-api-key'
});

// Create an agent
const agent = await client.agents.create({
  name: 'my-agent',
  systemPrompt: 'You are a helpful assistant',
  modelName: 'gpt-4',
  enabledTools: ['sql', 'http']
});

console.log(`Created agent: ${agent.id}`);

// Create a session
const session = await client.sessions.create({
  agentId: agent.id
});

// Send a message
const response = await client.sessions.sendMessage({
  sessionId: session.id,
  content: 'Hello, agent!'
});

console.log(`Agent response: ${response.content}`);
```

---

## Usage Examples

<details>
<summary><strong>📝 Complete Usage Examples</strong></summary>

### Python: Agent Management

```python
from neuronagent import NeuronAgentClient

client = NeuronAgentClient(
    base_url="http://localhost:8080",
    api_key="your-api-key"
)

# List all agents
agents = client.agents.list()
for agent in agents:
    print(f"Agent: {agent.name} ({agent.id})")

# Get agent details
agent = client.agents.get(agent_id="agent_123")
print(f"Model: {agent.model_name}")
print(f"Tools: {agent.enabled_tools}")

# Update agent
updated = client.agents.update(
    agent_id="agent_123",
    system_prompt="Updated prompt"
)

# Delete agent
client.agents.delete(agent_id="agent_123")
```

### Python: Session Management

```python
# Create session
session = client.sessions.create(
    agent_id="agent_123",
    metadata={"user_id": "user_456"}
)

# Send message
message = client.sessions.send_message(
    session_id=session.id,
    content="What is machine learning?",
    metadata={"priority": "high"}
)

# Get session history
history = client.sessions.get_history(session_id=session.id)
for msg in history:
    print(f"{msg.role}: {msg.content}")
```

### Python: WebSocket Streaming

```python
import asyncio
from neuronagent import NeuronAgentClient

async def stream_responses():
    client = NeuronAgentClient(
        base_url="http://localhost:8080",
        api_key="your-api-key"
    )
    
    async for event in client.sessions.stream(
        session_id="session_123",
        content="Tell me about AI"
    ):
        if event.type == "token":
            print(event.content, end="", flush=True)
        elif event.type == "complete":
            print(f"\nComplete: {event.response}")

asyncio.run(stream_responses())
```

### TypeScript: Agent Management

```typescript
import { NeuronAgentClient } from '@neurondb/neuronagent';

const client = new NeuronAgentClient({
  baseURL: 'http://localhost:8080',
  apiKey: 'your-api-key'
});

// List all agents
const agents = await client.agents.list();
agents.forEach(agent => {
  console.log(`Agent: ${agent.name} (${agent.id})`);
});

// Get agent details
const agent = await client.agents.get({ agentId: 'agent_123' });
console.log(`Model: ${agent.modelName}`);
console.log(`Tools: ${agent.enabledTools}`);

// Update agent
const updated = await client.agents.update({
  agentId: 'agent_123',
  systemPrompt: 'Updated prompt'
});

// Delete agent
await client.agents.delete({ agentId: 'agent_123' });
```

### TypeScript: WebSocket Streaming

```typescript
const stream = await client.sessions.stream({
  sessionId: 'session_123',
  content: 'Tell me about AI'
});

for await (const event of stream) {
  if (event.type === 'token') {
    process.stdout.write(event.content);
  } else if (event.type === 'complete') {
    console.log(`\nComplete: ${event.response}`);
  }
}
```

</details>

---

## API Reference

### Python SDK

| Module | Description | Documentation |
|:-------|:------------|:--------------|
| `neuronagent.client` | Main client class | [Client API](python/docs/client.md) |
| `neuronagent.agents` | Agent management | [Agents API](python/docs/agents.md) |
| `neuronagent.sessions` | Session management | [Sessions API](python/docs/sessions.md) |
| `neuronagent.messages` | Message handling | [Messages API](python/docs/messages.md) |
| `neuronagent.workflows` | Workflow management | [Workflows API](python/docs/workflows.md) |

### TypeScript SDK

| Module | Description | Documentation |
|:-------|:------------|:--------------|
| `@neurondb/neuronagent` | NeuronAgent client | [API Reference](typescript/docs/api.md) |
| `@neurondb/neurondesktop` | NeuronDesktop client | [API Reference](typescript/docs/desktop-api.md) |

---

## Generating SDKs

SDKs are generated from OpenAPI specifications using OpenAPI Generator.

### Prerequisites

```bash
# Install OpenAPI Generator CLI
npm install -g @openapitools/openapi-generator-cli

# Or using Docker
docker pull openapitools/openapi-generator-cli
```

### Generate All SDKs

```bash
# From repository root
./scripts/neurondb-pkgs.sh generate-sdk --language python
./scripts/neurondb-pkgs.sh generate-sdk --language typescript
```

### Generate Individual SDKs

#### Python SDK

```bash
openapi-generator-cli generate \
  -i /path/to/neuron-agent/openapi/openapi.yaml \
  -g python \
  -o sdks/python/neuronagent \
  --additional-properties=packageName=neuronagent,packageVersion=2.0.0
```

#### TypeScript SDK

```bash
openapi-generator-cli generate \
  -i /path/to/neuron-agent/openapi/openapi.yaml \
  -g typescript-axios \
  -o sdks/typescript/neuronagent \
  --additional-properties=npmName=@neurondb/neuronagent,npmVersion=2.0.0
```

### Post-Generation Steps

After generation, you may need to:

1. **Update package metadata** - Version, description, author
2. **Add custom code** - Helper functions, utilities
3. **Fix imports** - Adjust import paths if needed
4. **Add examples** - Include usage examples
5. **Run tests** - Verify SDK functionality

---

## Versioning

SDKs follow semantic versioning and are versioned independently:

| Version | Type | Description |
|:-------|:-----|:------------|
| **Major** | Breaking changes | API changes that break backward compatibility |
| **Minor** | New features | New features, backward compatible |
| **Patch** | Bug fixes | Bug fixes, backward compatible |

**Current Versions:**
- Python SDK: `2.0.0`
- TypeScript SDK: `2.0.0`

---

## Examples

See the `examples/` directories in each SDK for usage examples:

- **Basic agent usage** - Create agents, sessions, send messages
- **RAG pipeline** - Build RAG systems with vector search
- **Vector search** - Perform semantic search queries
- **Multi-agent collaboration** - Coordinate multiple agents
- **Workflow orchestration** - Create and execute workflows
- **WebSocket streaming** - Real-time agent responses

---

## Contributing

When updating SDKs:

<details>
<summary><strong>📝 Contribution Guidelines</strong></summary>

1. **Update OpenAPI specifications** - Modify API definitions in the **neuron-agent** repo (`openapi/`)
2. **Regenerate SDKs** - Use `neurondb-pkgs.sh generate-sdk`
3. **Update examples** - Add/update examples if API changes
4. **Run tests** - Verify compatibility with test suite
5. **Update documentation** - Keep docs in sync with changes
6. **Submit PR** - Create pull request with changes

</details>

---

## Support

<details>
<summary><strong>📞 Get Help</strong></summary>

| Resource | Link | Description |
|:---------|:-----|:------------|
| **GitHub Issues** | [Report Issues](https://github.com/neurondb/neurondb/issues) | Bug reports and feature requests |
| **Documentation** | [SDK Docs](https://www.neurondb.ai/docs/sdks) | Complete SDK documentation |
| **Email Support** | support@neurondb.ai | Direct email support |
| **Python SDK Docs** | [Python README](python/README.md) | Python-specific documentation |
| **TypeScript SDK Docs** | [TypeScript README](typescript/README.md) | TypeScript-specific documentation |

</details>

---

## License

See [LICENSE](../LICENSE) file for license information.

---

<div align="center">

**[Documentation](https://www.neurondb.ai/docs/sdks)** • 
**[Python SDK](python/README.md)** • 
**[TypeScript SDK](typescript/README.md)** • 
**[Support](mailto:support@neurondb.ai)**

[⬆ Back to Top](#neurondb-sdks)

</div>
