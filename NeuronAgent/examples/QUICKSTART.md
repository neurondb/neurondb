# Quick Start Guide

Get up and running with NeuronAgent examples in 5 minutes!

## Step 1: Start NeuronAgent Server

```bash
# Option 1: Run from source
cd /path/to/NeuronAgent
go run cmd/agent-server/main.go

# Option 2: Use Docker
cd docker
docker compose up -d
```

Verify it's running:
```bash
curl http://localhost:8080/health
```

## Step 2: Get an API Key

```bash
# Generate an API key
cd /path/to/NeuronAgent
./scripts/generate_api_keys.sh

# Or set manually
export NEURONAGENT_API_KEY=your_api_key_here
```

## Step 3: Install Dependencies

```bash
cd examples
pip3 install -r requirements.txt
```

## Step 4: Run Examples

### Python Example
```bash
export NEURONAGENT_API_KEY=your_api_key
python3 python_client.py
```

### Production Example
```bash
export NEURONAGENT_API_KEY=your_api_key
python3 complete_example.py
```

### Go Example
```bash
export NEURONAGENT_API_KEY=your_api_key
go run go_client.go
```

## Step 5: Create Your First Agent

```python
from python_client import NeuronAgentClient

client = NeuronAgentClient()

# Create agent
agent = client.create_agent(
    name="my-first-agent",
    system_prompt="You are a helpful assistant.",
    model_name="gpt-4"
)

# Create session
session = client.create_session(agent_id=agent['id'])

# Send message
response = client.send_message(
    session_id=session['id'],
    content="Hello!"
)

print(response['response'])
```

## Troubleshooting

**Server not responding?**
```bash
# Check if running
curl http://localhost:8080/health

# Check logs
docker compose logs agent-server
```

**Authentication error?**
```bash
# Verify API key
echo $NEURONAGENT_API_KEY

# Test authentication
curl -H "Authorization: Bearer $NEURONAGENT_API_KEY" \
  http://localhost:8080/api/v1/agents
```

**Database connection issues?**
```bash
# Test connection
psql -h localhost -p 5432 -U postgres -d neurondb -c "SELECT 1;"
```

## Next Steps

- Read the [full README](README.md) for detailed examples
- Explore [API documentation](../docs/API.md)
- Check out [architecture guide](../docs/ARCHITECTURE.md)

Happy coding! 🚀

