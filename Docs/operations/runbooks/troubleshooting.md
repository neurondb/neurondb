# Troubleshooting Runbook

## Common Issues and Solutions

### Service Won't Start

**Symptoms**: Service container exits immediately or fails to start

**Diagnosis**:
```bash
# Check logs
docker compose logs <service-name>

# Check health endpoint
curl http://localhost:<port>/health
```

**Solutions**:
1. Verify database connection
2. Check environment variables
3. Review port conflicts
4. Check disk space

### High Error Rate

**Symptoms**: Error rate > 5% in metrics

**Diagnosis**:
```bash
# Check error logs
docker compose logs --tail=100 <service-name> | grep -i error

# Check metrics
curl http://localhost:<port>/metrics | grep errors_total
```

**Solutions**:
1. Review recent code changes
2. Check database connectivity
3. Verify API keys/credentials
4. Review rate limits

### High Latency

**Symptoms**: P95 latency > 1s

**Diagnosis**:
```bash
# Check request duration metrics
curl http://localhost:<port>/metrics | grep duration

# Check database query performance
psql -d neurondb -c "SELECT * FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;"
```

**Solutions**:
1. Optimize slow queries
2. Add database indexes
3. Increase connection pool size
4. Review vector index configuration

### Database Connection Issues

**Symptoms**: Connection timeouts or failures

**Diagnosis**:
```bash
# Test connection
psql -h localhost -p 5433 -U neurondb -d neurondb -c "SELECT 1;"

# Check connection pool
psql -d neurondb -c "SELECT count(*) FROM pg_stat_activity;"
```

**Solutions**:
1. Verify database is running
2. Check connection string
3. Review connection pool settings
4. Check firewall rules

### Vector Search Performance

**Symptoms**: Slow vector search queries

**Diagnosis**:
```sql
-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
WHERE indexname LIKE '%hnsw%' OR indexname LIKE '%ivf%';

-- Check index build status
SELECT * FROM neurondb.index_build_status;
```

**Solutions**:
1. Verify indexes are created
2. Tune HNSW parameters (m, ef_construction)
3. Increase ef_search for better recall
4. Consider rebuilding indexes

### Memory Issues

**Symptoms**: OOM kills or high memory usage

**Diagnosis**:
```bash
# Check memory usage
docker stats

# Check PostgreSQL memory
psql -d neurondb -c "SHOW shared_buffers;"
psql -d neurondb -c "SHOW work_mem;"
```

**Solutions**:
1. Reduce connection pool size
2. Lower work_mem
3. Increase container memory limits
4. Review index build memory settings

## Performance Tuning

### Database Tuning

```sql
-- Increase shared buffers
ALTER SYSTEM SET shared_buffers = '256MB';

-- Tune work memory
ALTER SYSTEM SET work_mem = '16MB';

-- Increase maintenance work memory for index builds
ALTER SYSTEM SET maintenance_work_mem = '512MB';
```

### Vector Index Tuning

```sql
-- Rebuild HNSW index with better parameters
DROP INDEX documents_embedding_idx;
CREATE INDEX documents_embedding_idx
ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 32, ef_construction = 128);
```

### Service Tuning

```bash
# Increase connection pool
export DB_MAX_OPEN_CONNS=50

# Tune timeouts
export SERVER_READ_TIMEOUT=60s
export SERVER_WRITE_TIMEOUT=60s
```

## Escalation

If issues persist:
1. Collect logs: `docker compose logs > logs.txt`
2. Collect metrics: Export Prometheus metrics
3. Review recent changes: Check git history
4. Contact support: support@neurondb.ai







