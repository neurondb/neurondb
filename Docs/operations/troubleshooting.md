# Troubleshooting Guide

<div align="center">

**Comprehensive troubleshooting guide for the NeuronDB ecosystem**

[![Troubleshooting](https://img.shields.io/badge/troubleshooting-complete-brightgreen)](.)
[![Status](https://img.shields.io/badge/status-stable-blue)](.)

</div>

---

## Table of Contents

1. [Common Issues](#common-issues)
2. [Database Issues](#database-issues)
3. [Service Issues](#service-issues)
4. [Authentication Issues](#authentication-issues)
5. [Performance Issues](#performance-issues)
6. [Deployment Issues](#deployment-issues)

---

## Common Issues

### Services Not Starting

**Symptoms**: Services fail to start or crash immediately

> [!TIP]
> Check logs first. Most startup issues show clear error messages in container logs.

**Diagnosis**:
```bash
# Check logs
docker logs neurondb-cpu
docker logs neuronagent
docker logs neurondesk-api

# Check health
./scripts/health-check.sh
```

**Solutions**:

> [!NOTE]
> Start with environment variables. Missing or incorrect values cause most startup failures.

1. **Check environment variables**:
   ```bash
   docker exec neurondb-cpu env | grep POSTGRES
   ```

2. **Check port conflicts**:
   ```bash
   # Check if ports are in use
   netstat -tuln | grep -E "5433|8080|8081|3000"
   
   # Or use lsof (macOS/Linux)
   lsof -i :5433 -i :8080 -i :8081 -i :3000
   
   # Or use ss (Linux)
   ss -tuln | grep -E "5433|8080|8081|3000"
   ```

3. **Check disk space**:
   ```bash
   df -h
   ```

4. **Check memory**:
   ```bash
   free -h
   ```

### Connection Refused

**Symptoms**: Cannot connect to services

> [!TIP]
> Verify the service is running before checking network configuration.

**Diagnosis**:
```bash
# Test connectivity
curl http://localhost:8080/health
curl http://localhost:8081/health

# Check firewall
sudo ufw status
```

**Solutions**:
1. **Check service is running**:
   ```bash
   docker ps | grep neuron
   ```

2. **Check port bindings**:
   ```bash
   docker port neurondb-cpu
   ```

3. **Check firewall rules**:
   ```bash
   sudo ufw allow 5433/tcp  # PostgreSQL (Docker Compose default)
   sudo ufw allow 8080/tcp  # NeuronAgent
   sudo ufw allow 8081/tcp  # NeuronDesktop API
   sudo ufw allow 3000/tcp  # NeuronDesktop UI
   ```

---

## Database Issues

### PostgreSQL Connection Failed

**Symptoms**: Cannot connect to PostgreSQL

> [!TIP]
> Test the connection with psql first. This shows the exact error message.

**Diagnosis**:
```bash
# Test connection (Docker Compose default)
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" -c "SELECT 1;"

# Or using docker compose exec
docker compose exec neurondb psql -U neurondb -d neurondb -c "SELECT 1;"

# Check PostgreSQL status
docker compose exec neurondb pg_isready -U neurondb -d neurondb

# Or check container directly
docker exec neurondb-cpu pg_isready -U neurondb -d neurondb
```

**Solutions**:

> [!NOTE]
> Default Docker Compose credentials are for development only. Change them in production.

1. **Check credentials**:
   ```bash
   # Default Docker Compose credentials (development only)
   # User: neurondb
   # Password: neurondb
   # Database: neurondb
   # Port: 5433 (host), 5432 (container)
   
   # Check environment variable
   echo $POSTGRES_PASSWORD
   
   # Test with default credentials
   psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" -c "SELECT current_user;"
   ```

2. **Check PostgreSQL is running**:
   ```bash
   # Using docker compose
   docker compose ps neurondb
   
   # Check container status
   docker exec neurondb-cpu pg_ctl status
   
   # Check health
   docker compose exec neurondb pg_isready -U neurondb -d neurondb
   ```

3. **Check connection limits**:
   ```sql
   SELECT count(*) FROM pg_stat_activity;
   SELECT max_connections FROM pg_settings WHERE name = 'max_connections';
   ```

### NeuronDB Extension Not Loading

**Symptoms**: `neurondb.version()` function not found

> [!TIP]
> The extension must be created in each database where you use it.

**Diagnosis**:
```sql
-- Check extension
SELECT * FROM pg_extension WHERE extname = 'neurondb';

-- Check extension files
\dx neurondb
```

**Solutions**:

> [!NOTE]
> Create the extension in the database where you need it. Extensions are database-specific.

1. **Install extension**:
   ```sql
   CREATE EXTENSION IF NOT EXISTS neurondb;
   ```

2. **Check extension files exist**:
   ```bash
   ls -la /usr/lib/postgresql/*/lib/neurondb.*
   ```

3. **Check PostgreSQL version compatibility**:
   ```sql
   SELECT version();
   ```

### Migration Failures

**Symptoms**: Database migrations fail

**Diagnosis**:
```bash
# Check migration status
psql -d neurondesk -c "SELECT * FROM schema_migrations ORDER BY version DESC LIMIT 5;"
```

**Solutions**:
1. **Backup before migration**:
   ```bash
   ./scripts/backup.sh
   ```

2. **Check for conflicts**:
   ```sql
   -- Check for orphaned records
   SELECT * FROM api_keys WHERE user_id NOT IN (SELECT id FROM users);
   ```

3. **Manual migration**:
   ```bash
   psql -d neurondesk -f NeuronDesktop/api/migrations/008_unified_identity_model.sql
   ```

---

## Service Issues

### NeuronAgent Not Responding

**Symptoms**: Agent API returns 500 or timeout

**Diagnosis**:
```bash
# Check logs
docker logs neuronagent --tail 100

# Check health
curl http://localhost:8080/health
```

**Solutions**:
1. **Check database connection**:
   ```bash
   docker exec neuronagent env | grep DATABASE
   ```

2. **Check API key**:
   ```bash
   docker exec neuronagent env | grep API_KEY
   ```

3. **Restart service**:
   ```bash
   docker restart neuronagent
   ```

### NeuronDesktop API Errors

**Symptoms**: API returns errors or 500

**Diagnosis**:
```bash
# Check logs
docker logs neurondesk-api --tail 100

# Check health
curl http://localhost:8081/health
```

**Solutions**:
1. **Check database connection**:
   ```bash
   docker exec neurondesk-api env | grep DATABASE
   ```

2. **Check OIDC configuration**:
   ```bash
   docker exec neurondesk-api env | grep OIDC
   ```

3. **Check session storage**:
   ```sql
   SELECT count(*) FROM sessions;
   SELECT count(*) FROM login_attempts;
   ```

### NeuronMCP Connection Issues

**Symptoms**: MCP tools not working

**Diagnosis**:
```bash
# Check MCP server
curl http://localhost:8082/health

# Check MCP config
cat NeuronMCP/mcp-config.json
```

**Solutions**:
1. **Check MCP command**:
   ```bash
   which npx
   npx --version
   ```

2. **Check MCP server logs**:
   ```bash
   docker logs neuronmcp --tail 100
   ```

3. **Test MCP config**:
   ```bash
   curl -X POST http://localhost:8081/api/v1/mcp/test \
     -H "Content-Type: application/json" \
     -d '{"command": "npx", "args": ["-y", "@modelcontextprotocol/server-pg"]}'
   ```

---

## Authentication Issues

### OIDC Login Not Working

**Symptoms**: OIDC login fails or redirects incorrectly

**Diagnosis**:
```sql
-- Check login attempts
SELECT * FROM login_attempts ORDER BY created_at DESC LIMIT 5;

-- Check sessions
SELECT * FROM sessions ORDER BY created_at DESC LIMIT 5;
```

**Solutions**:
1. **Check OIDC configuration**:
   ```bash
   docker exec neurondesk-api env | grep OIDC
   ```

2. **Check redirect URI**:
   - Ensure `redirect_uri` matches OIDC provider configuration
   - Check `login_attempts.redirect_uri` is set correctly

3. **Check cookies**:
   - Ensure cookies are enabled in browser
   - Check `SameSite` and `Secure` flags match environment

4. **Clear old login attempts**:
   ```sql
   DELETE FROM login_attempts WHERE expires_at < NOW();
   ```

### API Key Authentication Failing

**Symptoms**: API requests with API key return 401

**Diagnosis**:
```sql
-- Check API key exists
SELECT id, name, created_at FROM api_keys WHERE key_hash = encode(digest('nd_...', 'sha256'), 'hex');

-- Check API key is not expired
SELECT * FROM api_keys WHERE expires_at < NOW();
```

**Solutions**:
1. **Verify API key format**:
   - Should start with `nd_`
   - Should be 32+ characters

2. **Check API key in request**:
   ```bash
   curl -H "Authorization: Bearer nd_..." http://localhost:8080/api/v1/agents
   ```

3. **Check rate limits**:
   ```sql
   SELECT rate_limit_per_min FROM api_keys WHERE id = '...';
   ```

### Session Expired

**Symptoms**: User session expires unexpectedly

**Diagnosis**:
```sql
-- Check session expiration
SELECT id, created_at, expires_at FROM sessions WHERE user_id = '...';
```

**Solutions**:
1. **Check session TTL**:
   ```bash
   docker exec neurondesk-api env | grep SESSION
   ```

2. **Extend session TTL** (if needed):
   ```bash
   export SESSION_ACCESS_TTL=30m
   export SESSION_REFRESH_TTL=14d
   ```

3. **Check token rotation**:
   - Ensure refresh tokens are being used
   - Check for token reuse (should revoke session)

---

## Performance Issues

### Slow Queries

**Symptoms**: Database queries are slow

**Diagnosis**:
```sql
-- Check slow queries
SELECT query, mean_exec_time, calls 
FROM pg_stat_statements 
ORDER BY mean_exec_time DESC 
LIMIT 10;

-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan 
FROM pg_stat_user_indexes 
WHERE idx_scan = 0;
```

**Solutions**:
1. **Add indexes**:
   ```sql
   CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);
   CREATE INDEX idx_sessions_user_id ON sessions(user_id);
   ```

2. **Analyze tables**:
   ```sql
   ANALYZE api_keys;
   ANALYZE sessions;
   ```

3. **Check connection pooling**:
   - Use PgBouncer for connection pooling
   - Reduce connection count if needed

### High Memory Usage

**Symptoms**: Services using too much memory

**Diagnosis**:
```bash
# Check memory usage
docker stats --no-stream

# Check Go memory
go tool pprof http://localhost:8080/debug/pprof/heap
```

**Solutions**:
1. **Set memory limits**:
   ```yaml
   # docker-compose.yml
   services:
     neuronagent:
       mem_limit: 2g
   ```

2. **Enable GC tuning**:
   ```bash
   export GOGC=100
   ```

3. **Check for memory leaks**:
   - Review code for unclosed resources
   - Use memory profiler

### High CPU Usage

**Symptoms**: Services using too much CPU

**Diagnosis**:
```bash
# Check CPU usage
docker stats --no-stream

# Check Go CPU profile
go tool pprof http://localhost:8080/debug/pprof/profile
```

**Solutions**:
1. **Optimize queries**:
   - Add indexes
   - Use EXPLAIN ANALYZE

2. **Reduce worker concurrency**:
   ```bash
   export WORKER_CONCURRENCY=4
   ```

3. **Check for infinite loops**:
   - Review logs for repeated errors
   - Check for deadlocks

---

## Deployment Issues

### Docker Compose Not Starting

**Symptoms**: `docker compose up` fails

**Diagnosis**:
```bash
# Check logs
docker compose logs

# Check configuration
docker compose config
```

**Solutions**:
1. **Check environment file**:
   ```bash
   cp env.example .env
   # Edit .env with correct values
   ```

2. **Check Docker version**:
   ```bash
   docker --version
   docker compose version
   ```

3. **Check network**:
   ```bash
   docker network ls
   docker network create neurondb-network
   ```

### Helm Installation Fails

**Symptoms**: `helm install` fails

**Diagnosis**:
```bash
# Check chart
helm lint ./helm/neurondb

# Dry run
helm install neurondb ./helm/neurondb --dry-run --debug
```

**Solutions**:
1. **Check values**:
   ```bash
   helm show values ./helm/neurondb
   ```

2. **Check Kubernetes**:
   ```bash
   kubectl get nodes
   kubectl get pods
   ```

3. **Check resources**:
   ```bash
   kubectl describe pod <pod-name>
   ```

### Backup/Restore Fails

**Symptoms**: Backup or restore script fails

**Diagnosis**:
```bash
# Check disk space
df -h

# Check PostgreSQL access
psql -h localhost -U postgres -d postgres -c "SELECT 1;"
```

**Solutions**:
1. **Check permissions**:
   ```bash
   chmod +x scripts/backup.sh
   chmod +x scripts/restore.sh
   ```

2. **Check disk space**:
   ```bash
   df -h /backups
   ```

3. **Check PostgreSQL version**:
   ```bash
   psql --version
   pg_dump --version
   ```

---

## Getting Help

### Logs

**Location**:
- Docker: `docker logs <container-name>`
- Systemd: `journalctl -u <service-name>`
- Application: Check `logs/` directory

**Log Levels**:
- `DEBUG`: Detailed debugging information
- `INFO`: General information
- `WARN`: Warning messages
- `ERROR`: Error messages

### Debug Mode

**Enable debug logging**:
```bash
export LOG_LEVEL=debug
export DEBUG=true
```

### Support Resources

- **Documentation**: `Docs/` directory
- **Examples**: `examples/` directory
- **GitHub Issues**: Report bugs and issues
- **Email**: support@neurondb.ai

---

## Quick Reference

### Common Commands

```bash
# Health check
./scripts/health-check.sh

# Smoke test
./scripts/smoke-test.sh

# Backup
./scripts/backup.sh

# Restore
./scripts/restore.sh

# View logs
docker logs -f neurondb-cpu
docker logs -f neuronagent
docker logs -f neurondesk-api

# Restart services
docker compose restart

# Stop services
docker compose down

# Start services
docker compose up -d
```

### Useful SQL Queries

```sql
-- Check database size
SELECT pg_size_pretty(pg_database_size('neurondb'));

-- Check table sizes
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Check active connections
SELECT count(*) FROM pg_stat_activity;

-- Check locks
SELECT * FROM pg_locks WHERE NOT granted;
```

---

---

## 🔗 Related Documentation

| Document | Description |
|----------|-------------|
| **[Getting Started Troubleshooting](../getting-started/troubleshooting.md)** | Getting started issues |
| **[Operations Runbooks](runbooks/troubleshooting.md)** | Advanced troubleshooting |
| **[Observability Setup](observability-setup.md)** | Monitoring and diagnostics |
| **[Deployment Guide](../deployment/README.md)** | Deployment troubleshooting |

---

<div align="center">

[⬆ Back to Top](#troubleshooting-guide) · [📚 Operations Index](../README.md) · [📚 Main Documentation](../../README.md)

</div>






