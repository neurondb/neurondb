# NeuronAgent Testing Guide

Complete guide for testing NeuronAgent with psql and Python clients.

## Server Status

✅ **NeuronAgent server is running on http://localhost:8080**

## Quick Test Commands

### 1. Health Check

```bash
curl http://localhost:8080/health
```

### 2. Test with psql

In your psql session (neurondb database):

```sql
-- Check schema
\dt neurondb_agent.*

-- List agents
SELECT * FROM neurondb_agent.agents;

-- List sessions
SELECT * FROM neurondb_agent.sessions;

-- List messages
SELECT * FROM neurondb_agent.messages ORDER BY created_at DESC LIMIT 10;

-- Check API keys
SELECT key_prefix, organization_id, user_id FROM neurondb_agent.api_keys;

-- Run full test script
\i examples/test_with_psql.sql
```

### 3. Test with Python Client

```bash
cd NeuronAgent/examples

# Install dependencies
pip3 install -r requirements.txt

# Quick test
python3 quick_test.py

# Full examples
export NEURONAGENT_API_KEY=your_api_key
python3 examples_modular/01_basic_usage.py
```

### 4. Test API Endpoints

```bash
# Health check (no auth required)
curl http://localhost:8080/health

# List agents (requires API key)
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:8080/api/v1/agents

# Create agent
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-agent",
    "system_prompt": "You are helpful.",
    "model_name": "gpt-4",
    "enabled_tools": ["sql"]
  }'
```

## API Key Setup

Currently, there's a test API key in the database, but you need to generate a proper one:

```bash
cd NeuronAgent
./scripts/generate_api_keys.sh default default 60 user
```

Or create one manually in psql:

```sql
-- Generate a secure key (use a proper hash in production)
INSERT INTO neurondb_agent.api_keys 
  (key_hash, key_prefix, organization_id, user_id, rate_limit_per_minute, roles, metadata)
VALUES 
  ('$2a$10$...', 'mykey1', 'default', 'default', 60, ARRAY['user'], '{}'::jsonb)
RETURNING key_prefix;
```

## Server Management

### Start Server

```bash
cd NeuronAgent
DB_USER=pge DB_PASSWORD="" go run cmd/agent-server/main.go
```

Or use the startup script:

```bash
./start_server.sh
```

### Stop Server

```bash
pkill -f "agent-server"
# Or find PID and kill
ps aux | grep agent-server
kill <PID>
```

### Check Server Logs

```bash
tail -f /tmp/neurondb-agent.log
```

## Database Verification

### Check Tables

```sql
SELECT tablename 
FROM pg_tables 
WHERE schemaname = 'neurondb_agent';
```

Expected tables:
- agents
- sessions
- messages
- memory_chunks
- tools
- jobs
- api_keys

### Check Indexes

```sql
SELECT indexname, tablename
FROM pg_indexes
WHERE schemaname = 'neurondb_agent';
```

### Verify NeuronDB Extension

```sql
SELECT * FROM pg_extension WHERE extname = 'neurondb';
```

## Testing Workflow

### 1. Start Server

```bash
cd NeuronAgent
DB_USER=pge DB_PASSWORD="" go run cmd/agent-server/main.go &
```

### 2. Verify Health

```bash
curl http://localhost:8080/health
```

### 3. Generate API Key

```bash
./scripts/generate_api_keys.sh
```

### 4. Test with Python

```bash
cd examples
export NEURONAGENT_API_KEY=your_key
python3 examples_modular/01_basic_usage.py
```

### 5. Check in psql

```sql
-- See created agents
SELECT * FROM neurondb_agent.agents;

-- See sessions
SELECT * FROM neurondb_agent.sessions;

-- See messages
SELECT * FROM neurondb_agent.messages;
```

## Example Test Session

```bash
# Terminal 1: Start server
cd NeuronAgent
DB_USER=pge DB_PASSWORD="" go run cmd/agent-server/main.go

# Terminal 2: Test API
export API_KEY="your_key_here"

# Create agent
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-agent",
    "system_prompt": "You are helpful.",
    "model_name": "gpt-4",
    "enabled_tools": ["sql"]
  }' | jq .

# Terminal 3: psql
psql neurondb
SELECT * FROM neurondb_agent.agents WHERE name = 'test-agent';
```

## Troubleshooting

### Server won't start

1. Check database connection:
   ```bash
   psql -d neurondb -U pge -c "SELECT 1;"
   ```

2. Check environment variables:
   ```bash
   echo $DB_USER $DB_NAME
   ```

3. Check logs:
   ```bash
   tail -50 /tmp/neurondb-agent.log
   ```

### API returns 401

- Generate proper API key
- Check key is in database:
  ```sql
  SELECT key_prefix FROM neurondb_agent.api_keys;
  ```
- Use correct Authorization header format:
  ```
  Authorization: Bearer your_key_here
  ```

### Python client errors

1. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

2. Check API key:
   ```bash
   echo $NEURONAGENT_API_KEY
   ```

3. Test connection:
   ```python
   from neurondb_client import NeuronAgentClient
   client = NeuronAgentClient()
   print(client.health_check())
   ```

## Next Steps

1. ✅ Server is running
2. ✅ Database schema is set up
3. ⚠️  Generate proper API key
4. ✅ Test with psql
5. ✅ Test with Python client
6. ✅ Run full examples

See `examples/README.md` for more detailed examples.

