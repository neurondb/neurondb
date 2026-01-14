# NeuronAgent Troubleshooting Guide

Common issues and solutions for NeuronAgent deployment and usage.

## Server Issues

### Server Won't Start

**Symptoms:**
- Server fails to start
- Error messages about database connection
- Port already in use

**Solutions:**

1. **Check Database Connection**
   ```bash
   psql -h localhost -p 5432 -U neurondb -d neurondb -c "SELECT 1;"
   ```

2. **Verify Environment Variables**
   ```bash
   env | grep -E "DB_|SERVER_"
   ```

3. **Check Port Availability**
   ```bash
   lsof -i :8080
   # Or use different port
   export SERVER_PORT=8081
   ```

4. **Check Logs**
   ```bash
   tail -f /tmp/neurondb-agent.log
   ```

### Database Connection Failed

**Symptoms:**
- "Failed to connect to database" errors
- Connection timeout errors

**Solutions:**

1. **Verify NeuronDB Extension**
   ```sql
   SELECT * FROM pg_extension WHERE extname = 'neurondb';
   ```

2. **Check Database Permissions**
   ```sql
   GRANT ALL PRIVILEGES ON DATABASE neurondb TO neurondb;
   GRANT ALL ON SCHEMA neurondb_agent TO neurondb;
   ```

3. **Verify Connection String**
   ```bash
   # Test connection manually
   psql "host=localhost port=5432 user=neurondb dbname=neurondb"
   ```

4. **Check Firewall/Security Groups**
   - Ensure PostgreSQL port (5432) is accessible
   - Check firewall rules
   - Verify security group settings (AWS, GCP, etc.)

## API Issues

### 401 Unauthorized Errors

**Symptoms:**
- All API requests return 401
- "Invalid API key" errors

**Solutions:**

1. **Generate API Key**
   ```bash
   go run cmd/generate-key/main.go \
     -org my-org \
     -user my-user \
     -rate 1000 \
     -roles user,admin
   ```

2. **Verify API Key Format**
   ```bash
   # Should be: Authorization: Bearer <key>
   curl -H "Authorization: Bearer YOUR_KEY" http://localhost:8080/api/v1/agents
   ```

3. **Check Key in Database**
   ```sql
   SELECT key_prefix, organization_id, user_id FROM neurondb_agent.api_keys;
   ```

4. **Verify Key Hash**
   - Ensure key is hashed correctly
   - Check bcrypt hashing is working

### 429 Rate Limit Exceeded

**Symptoms:**
- Requests fail with 429 status
- "Rate limit exceeded" errors

**Solutions:**

1. **Check Rate Limit Settings**
   ```sql
   SELECT key_prefix, rate_limit_per_minute FROM neurondb_agent.api_keys;
   ```

2. **Increase Rate Limit**
   ```sql
   UPDATE neurondb_agent.api_keys 
   SET rate_limit_per_minute = 1000 
   WHERE key_prefix = 'your_prefix';
   ```

3. **Implement Backoff**
   - Use exponential backoff in client
   - Respect Retry-After headers

### 500 Internal Server Errors

**Symptoms:**
- Random 500 errors
- Server crashes

**Solutions:**

1. **Check Server Logs**
   ```bash
   tail -100 /tmp/neurondb-agent.log
   ```

2. **Verify Database Schema**
   ```sql
   SELECT tablename FROM pg_tables WHERE schemaname = 'neurondb_agent';
   ```

3. **Run Migrations**
   ```bash
   psql -d neurondb -f sql/001_initial_schema.sql
   psql -d neurondb -f sql/002_add_indexes.sql
   # ... run all migrations
   ```

4. **Check Resource Limits**
   - Memory usage
   - CPU usage
   - Database connections

## WebSocket Issues

### WebSocket Connection Fails

**Symptoms:**
- WebSocket upgrade fails
- Connection immediately closes

**Solutions:**

1. **Check Authentication**
   ```javascript
   // Use API key in query or header
   const ws = new WebSocket('ws://localhost:8080/ws?session_id=...&api_key=...');
   ```

2. **Verify Session ID**
   ```sql
   SELECT id FROM neurondb_agent.sessions WHERE id = 'session-id';
   ```

3. **Check CORS Settings**
   - Verify CORS middleware is configured
   - Check origin is allowed

4. **Test with curl**
   ```bash
   curl --include \
     --no-buffer \
     --header "Connection: Upgrade" \
     --header "Upgrade: websocket" \
     --header "Sec-WebSocket-Key: test" \
     --header "Sec-WebSocket-Version: 13" \
     http://localhost:8080/ws?session_id=...
   ```

### WebSocket Disconnects Frequently

**Symptoms:**
- Connection drops after short time
- Ping/pong not working

**Solutions:**

1. **Check Keepalive Settings**
   - Verify ping/pong is enabled
   - Check timeout settings

2. **Network Issues**
   - Check for proxy timeouts
   - Verify load balancer settings
   - Check firewall rules

3. **Server Load**
   - Monitor server resources
   - Check for memory leaks
   - Verify connection cleanup

## Connector Issues

### Slack Connector Errors

**Error: "Slack authentication failed"**

**Solutions:**
1. Verify bot token is valid
2. Check token has necessary scopes:
   - `channels:read`
   - `channels:history`
   - `chat:write`
   - `conversations.list`
3. Regenerate token if needed

**Error: "channel not found"**

**Solutions:**
1. Use channel ID instead of name
2. Ensure bot is member of channel
3. Check channel ID format (starts with C, D, or G)

### GitHub Connector Errors

**Error: "GitHub API error: 403"**

**Solutions:**
1. Check token permissions
2. Verify repository access
3. Check rate limit: `curl -H "Authorization: token TOKEN" https://api.github.com/rate_limit`

**Error: "invalid GitHub path format"**

**Solutions:**
1. Use format: `owner/repo/path/to/file`
2. URL-encode special characters
3. Verify repository exists and is accessible

### GitLab Connector Errors

**Error: "GitLab connection failed"**

**Solutions:**
1. Verify token is valid
2. Check project ID is correct
3. Ensure token has `api` scope

## Workflow Issues

### Workflow Execution Fails

**Symptoms:**
- Workflow status stuck at "running"
- Workflow fails immediately

**Solutions:**

1. **Check Workflow Definition**
   ```bash
   neuronagent-cli workflow validate workflow.yaml
   ```

2. **Verify Dependencies**
   - Check all step dependencies exist
   - Ensure no circular dependencies
   - Verify step IDs are unique

3. **Check Step Execution**
   ```sql
   SELECT * FROM neurondb_agent.workflow_step_executions 
   WHERE workflow_execution_id = 'execution-id';
   ```

4. **Review Error Messages**
   ```sql
   SELECT error_message FROM neurondb_agent.workflow_executions 
   WHERE id = 'execution-id';
   ```

### Workflow Timeout

**Symptoms:**
- Workflow never completes
- Status remains "running"

**Solutions:**

1. **Set Timeout**
   - Add timeout to workflow configuration
   - Set timeout per step

2. **Check Step Execution**
   - Verify steps are actually executing
   - Check for hanging operations

3. **Monitor Resources**
   - Check database connections
   - Verify no deadlocks
   - Monitor CPU and memory

## Sandbox Issues

### Docker Sandbox Not Working

**Symptoms:**
- Container isolation not working
- Commands fail in sandbox

**Solutions:**

1. **Verify Docker is Running**
   ```bash
   docker ps
   ```

2. **Check Docker Permissions**
   ```bash
   # Ensure user can access Docker socket
   sudo usermod -aG docker $USER
   ```

3. **Test Docker Client**
   ```bash
   docker run --rm alpine:latest echo "test"
   ```

4. **Check Resource Limits**
   - Verify memory limits are reasonable
   - Check CPU limits
   - Ensure disk space available

### Sandbox Timeout Issues

**Symptoms:**
- Commands timeout unexpectedly
- Timeout errors

**Solutions:**

1. **Increase Timeout**
   ```go
   config := tools.SandboxConfig{
       Timeout: 10 * time.Minute, // Increase timeout
   }
   ```

2. **Check Command Complexity**
   - Verify commands aren't too complex
   - Check for infinite loops
   - Monitor resource usage

## Memory and Performance Issues

### High Memory Usage

**Symptoms:**
- Server uses excessive memory
- Out of memory errors

**Solutions:**

1. **Check Connection Pool**
   ```go
   // Reduce connection pool size
   MaxOpenConns: 10,
   MaxIdleConns: 5,
   ```

2. **Monitor Memory Chunks**
   ```sql
   SELECT COUNT(*) FROM neurondb_agent.memory_chunks;
   -- Clean up old chunks if needed
   ```

3. **Check for Memory Leaks**
   - Use memory profiler
   - Monitor goroutine count
   - Check for unclosed resources

### Slow Performance

**Symptoms:**
- API responses are slow
- Database queries take long

**Solutions:**

1. **Check Database Indexes**
   ```sql
   SELECT indexname, tablename 
   FROM pg_indexes 
   WHERE schemaname = 'neurondb_agent';
   ```

2. **Analyze Queries**
   ```sql
   EXPLAIN ANALYZE SELECT * FROM neurondb_agent.agents WHERE name = 'test';
   ```

3. **Optimize Vector Search**
   - Check HNSW index exists
   - Verify index parameters
   - Monitor search performance

## Logging and Debugging

### Enable Debug Logging

```bash
export LOG_LEVEL=debug
go run cmd/agent-server/main.go
```

### View Logs

```bash
# Server logs
tail -f /tmp/neurondb-agent.log

# Docker logs (if using Docker)
docker compose logs -f agent-server

# Database logs
tail -f /var/log/postgresql/postgresql.log
```

### Debug WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8080/ws?session_id=...');
ws.onopen = () => console.log('Connected');
ws.onmessage = (e) => console.log('Message:', e.data);
ws.onerror = (e) => console.error('Error:', e);
ws.onclose = (e) => console.log('Closed:', e.code, e.reason);
```

## Getting Help

If you continue to experience issues:

1. **Check Documentation**: Review [API Documentation](API.md) and [Architecture Guide](architecture.md)
2. **Review Logs**: Collect relevant log entries
3. **Gather Information**:
   - Server version
   - Database version
   - Error messages
   - Steps to reproduce
4. **Contact Support**: support@neurondb.ai

