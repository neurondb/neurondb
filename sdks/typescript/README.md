# NeuronDB TypeScript SDK

TypeScript/JavaScript client libraries for NeuronDB ecosystem components.

## Installation

```bash
npm install @neurondb/neuronagent @neurondb/neurondesktop
# Or
yarn add @neurondb/neuronagent @neurondb/neurondesktop
```

## Usage

### NeuronAgent

```typescript
import { NeuronAgentClient } from '@neurondb/neuronagent'

// Initialize client
const client = new NeuronAgentClient({
  baseURL: 'http://localhost:8080',
  apiKey: 'your-api-key'
})

// Create an agent
const agent = await client.agents.createAgent({
  name: 'my-agent',
  systemPrompt: 'You are a helpful assistant',
  modelName: 'gpt-4',
  enabledTools: ['sql', 'http']
})

// Create a session
const session = await client.sessions.createSession({
  agentId: agent.id
})

// Send a message
const response = await client.sessions.sendMessage(session.id, {
  content: 'What is the weather today?'
})

console.log(response.content)
```

### NeuronDesktop

```typescript
import { NeuronDesktopClient } from '@neurondb/neurondesktop'

const client = new NeuronDesktopClient({
  baseURL: 'http://localhost:8081'
})

// List profiles
const profiles = await client.profiles.list()

// Search NeuronDB
const results = await client.neurondb.search(profiles[0].id, {
  query: 'vector search',
  limit: 10
})
```

## Browser Usage

The SDKs work in both Node.js and browser environments:

```typescript
// Browser
import { NeuronAgentClient } from '@neurondb/neuronagent'

const client = new NeuronAgentClient({
  baseURL: 'https://api.example.com',
  apiKey: 'your-api-key',
  // Cookies are used automatically for session-based auth
})
```

## Examples

See `examples/` directory for complete examples:
- `basic-agent.ts` - Basic agent usage
- `rag-pipeline.ts` - RAG pipeline example
- `vector-search.ts` - Vector search example

## API Reference

Full API documentation is available at:
- NeuronAgent: https://www.neurondb.ai/docs/neuronagent/api
- NeuronDesktop: https://www.neurondb.ai/docs/neurondesktop/api







