# How to Use NeuronAgent

NeuronAgent is an AI agent runtime system that integrates with NeuronDB to provide autonomous agent capabilities with long-term memory, tool execution, and streaming responses.

## Overview

NeuronAgent allows you to:
- Create AI agents with configurable models (GPT-4, Claude, Gemini, etc.)
- Manage agent sessions and conversations
- Execute tools (SQL, HTTP, Code, Shell)
- Use vector search for long-term memory
- Stream responses in real-time

## Prerequisites

1. **NeuronAgent Server Running**: NeuronAgent must be running on port 8080 (default)
2. **Profile Configuration**: Your NeuronDesktop profile must have:
   - `agent_endpoint`: URL to NeuronAgent (e.g., `http://localhost:8080`)
   - `agent_api_key`: API key for NeuronAgent authentication

## Using NeuronAgent in NeuronDesktop

### 1. Configure Profile

1. Go to **Settings** in NeuronDesktop
2. Select or create a profile
3. Set the **Agent Endpoint** (e.g., `http://localhost:8080`)
4. Set the **Agent API Key** (generate one from NeuronAgent)

### 2. Create an Agent

1. Navigate to **Agents** page in NeuronDesktop
2. Select your profile
3. Click **Create Agent**
4. Fill in the form:
   - **Name**: Unique identifier (e.g., `research-agent`)
   - **Description**: Brief description
   - **System Prompt**: Instructions for the agent
   - **Model**: Select from available models:
     - **Preset Models**: GPT-4, GPT-3.5 Turbo, Claude 3 Opus/Sonnet/Haiku, Gemini Pro, Llama 2
     - **Custom Model**: Enter any model name configured in NeuronDB
   - **Enabled Tools**: Select tools the agent can use (SQL, HTTP, Code, Shell)
5. Click **Create**

### 3. Model Selection

#### Preset Models

NeuronDesktop provides a list of common LLM models:

- **OpenAI**: `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`
- **Anthropic**: `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`
- **Google**: `gemini-pro`
- **Meta**: `llama-2-70b`

#### Custom Models

You can also use custom model names:
1. Select **Custom Model** option
2. Enter your model name (e.g., `my-custom-model`)
3. Ensure the model is configured in NeuronDB

**Note**: The model name must match what's configured in NeuronDB's LLM settings.

### 4. Create a Session

Sessions represent conversations with an agent:

```typescript
// Using the API
const session = await agentAPI.createSession(profileId, agentId)
```

### 5. Send Messages

Send messages to interact with the agent:

```typescript
// Using the API
const message = await agentAPI.sendMessage(
  profileId,
  sessionId,
  "What is machine learning?",
  false // stream: false for non-streaming
)
```

### 6. WebSocket Streaming

For real-time streaming responses:

```typescript
const ws = new WebSocket(
  `ws://localhost:8081/api/v1/profiles/${profileId}/agent/ws?session_id=${sessionId}`
)

ws.onmessage = (event) => {
  const data = JSON.parse(event.data)
  console.log('Agent response:', data.content)
}
```

## API Endpoints

### List Available Models

```bash
GET /api/v1/profiles/{profile_id}/agent/models
```

Returns list of available LLM models with descriptions.

### List Agents

```bash
GET /api/v1/profiles/{profile_id}/agent/agents
```

Returns all agents for the profile.

### Create Agent

```bash
POST /api/v1/profiles/{profile_id}/agent/agents
Content-Type: application/json

{
  "name": "my-agent",
  "description": "A helpful agent",
  "system_prompt": "You are a helpful assistant.",
  "model_name": "gpt-4",
  "enabled_tools": ["sql", "http"],
  "config": {
    "temperature": 0.7,
    "max_tokens": 1000
  }
}
```

### Create Session

```bash
POST /api/v1/profiles/{profile_id}/agent/sessions
Content-Type: application/json

{
  "agent_id": "uuid",
  "external_user_id": "user123",
  "metadata": {}
}
```

### Send Message

```bash
POST /api/v1/profiles/{profile_id}/agent/sessions/{session_id}/messages
Content-Type: application/json

{
  "role": "user",
  "content": "Hello, how are you?",
  "stream": false
}
```

## Model Configuration in NeuronDB

Models are configured in NeuronDB via environment variables or GUC settings. Common configuration:

```bash
# OpenAI
export OPENAI_API_KEY=your-key-here

# Anthropic
export ANTHROPIC_API_KEY=your-key-here

# Google
export GOOGLE_API_KEY=your-key-here
```

The model name you use in NeuronAgent must match the model identifier configured in NeuronDB.

## Example Workflow

1. **Start NeuronAgent**:
   ```bash
   cd NeuronAgent
   ./bin/agent-server
   ```

2. **Configure Profile** in NeuronDesktop:
   - Agent Endpoint: `http://localhost:8080`
   - Agent API Key: (from NeuronAgent)

3. **Create Agent**:
   - Name: `research-assistant`
   - Model: `gpt-4`
   - System Prompt: "You are a research assistant..."
   - Tools: `sql`, `http`

4. **Create Session** and start chatting!

## Troubleshooting

### "Agent endpoint not configured"
- Ensure your profile has `agent_endpoint` set
- Check that NeuronAgent is running

### "Model not found"
- Verify the model name matches NeuronDB configuration
- Check NeuronDB logs for model initialization errors

### "Authentication failed"
- Verify `agent_api_key` in profile matches NeuronAgent API key
- Generate a new API key from NeuronAgent if needed

## More Information

- [NeuronAgent README](../NeuronAgent/README.md)
- [NeuronAgent API Documentation](../NeuronAgent/docs/api.md)
- [NeuronAgent Architecture](../NeuronAgent/docs/architecture.md)

