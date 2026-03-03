# NeuronDB Background Workers

## Overview

NeuronDB includes 4 sophisticated background workers that provide automatic management, tuning, and maintenance for production deployments.

## Workers

### 1. **neuranq** - Queue Worker (`worker_queue.c`)

**Purpose**: Process asynchronous jobs from the queue

**Responsibilities**:
- **Embedding Generation**: Process text → vector embedding jobs
- **Reranking**: Reorder search results based on relevance
- **Cache Refresh**: Update cached data periodically
- **HTTP Calls**: External API integration

**Configuration**:
```sql
neurondb.neuranq_naptime = 1000          -- Sleep between cycles (ms)
neurondb.neuranq_queue_depth = 10000     -- Max queue size
neurondb.neuranq_batch_size = 100        -- Jobs per cycle
neurondb.neuranq_timeout = 30000         -- Job timeout (ms)
neurondb.neuranq_max_retries = 3         -- Max retry attempts
neurondb.neuranq_enabled = true          -- Enable/disable worker
```

**Key Features**:
- SKIP LOCKED for concurrent processing
- Per-tenant rate limiting
- Automatic retry with exponential backoff
- Crash-safe job processing
- Dead letter queue for failed jobs

**Monitoring**:
```sql
-- Check queue status
SELECT status, COUNT(*) FROM neurondb.job_queue GROUP BY status;

-- Per-tenant metrics
SELECT tenant_id, COUNT(*), AVG(EXTRACT(EPOCH FROM (completed_at - created_at)))
FROM neurondb.job_queue WHERE status = 'completed' GROUP BY tenant_id;
```

---

### 2. **neuranmon** - Auto-Tuner Worker (`worker_tuner.c`)

**Purpose**: Automatic query performance tuning

**Responsibilities**:
- **ef_search Tuning**: Adjust based on latency/recall SLOs
- **Hybrid Weight Optimization**: Balance vector vs keyword search
- **Cache Rotation**: Keep embeddings and results fresh
- **Recall@k Tracking**: Monitor search quality
- **Prometheus Metrics**: Export performance data

**Configuration**:
```sql
neurondb.neuranmon_naptime = 60000       -- Tuning cycle interval (ms)
neurondb.neuranmon_sample_size = 1000    -- Queries to sample
neurondb.neuronmon_target_latency = 100.0  -- Target latency (ms)
neurondb.neuranmon_target_recall = 0.95    -- Target recall threshold
neurondb.neuranmon_enabled = true
```

**Tuning Strategy**:
1. Sample recent query metrics
2. Calculate avg latency and recall
3. If latency > target AND recall ≥ target → Decrease ef_search
4. If recall < target AND latency ≤ target → Increase ef_search
5. Adjust hybrid weights based on query patterns

**Key Features**:
- SLO-driven automatic tuning
- P50/P90/P95/P99 percentile tracking
- Latency vs recall tradeoff optimization
- Per-index tuning parameters
- Real-time adaptation to workload

**Monitoring**:
```sql
-- Current performance
SELECT ef_search, AVG(latency_ms), AVG(recall)
FROM neurondb.query_metrics
WHERE recorded_at > now() - interval '10 minutes'
GROUP BY ef_search;

-- Tuning recommendations
SELECT 
    AVG(latency_ms) as avg_latency,
    AVG(recall) as avg_recall,
    CASE 
        WHEN AVG(latency_ms) > 100 THEN 'Tune DOWN'
        WHEN AVG(recall) < 0.95 THEN 'Tune UP'
        ELSE 'Optimal'
    END as recommendation
FROM neurondb.query_metrics
WHERE recorded_at > now() - interval '10 minutes';
```

---

### 3. **neurandefrag** - Index Maintenance Worker (`worker_defrag.c`)

**Purpose**: Automatic index maintenance and defragmentation

**Responsibilities**:
- **HNSW Graph Compaction**: Optimize graph structure
- **Orphan Edge Cleanup**: Remove dangling connections
- **Level Rebalancing**: Maintain balanced hierarchy
- **Tombstone Pruning**: Clean up deleted vectors
- **Statistics Refresh**: Update index stats
- **Scheduled Rebuilds**: Full rebuilds during maintenance windows

**Configuration**:
```sql
neurondb.neurandefrag_naptime = 300000              -- Cycle interval (5 min)
neurondb.neurandefrag_compact_threshold = 10000     -- Min edges for compaction
neurondb.neurandefrag_fragmentation_threshold = 0.3  -- Rebuild trigger
neurondb.neurandefrag_maintenance_window = '02:00-04:00'  -- Rebuild window
neurondb.neurandefrag_enabled = true
```

**Maintenance Operations**:
1. **Compaction**: Reduce graph size by removing redundant edges
2. **Cleanup**: Delete orphaned nodes and edges
3. **Rebalancing**: Distribute nodes evenly across levels
4. **Pruning**: Remove tombstones from deleted vectors
5. **VACUUM**: PostgreSQL table maintenance
6. **ANALYZE**: Update statistics for query planner

**Key Features**:
- Maintenance window scheduling
- Fragmentation detection
- Non-blocking operations (where possible)
- Progress tracking
- SIGSEGV recovery with longjmp

**Monitoring**:
```sql
-- Fragmentation metrics
SELECT 
    schemaname, tablename,
    n_live_tup, n_dead_tup,
    (n_dead_tup::float / NULLIF(n_live_tup + n_dead_tup, 0)) as dead_ratio,
    pg_size_pretty(pg_relation_size(schemaname || '.' || tablename)) as size
FROM pg_stat_user_tables
WHERE schemaname = 'neurondb'
ORDER BY dead_ratio DESC NULLS LAST;

-- Maintenance candidates
SELECT tablename, last_vacuum, last_autovacuum, last_analyze
FROM pg_stat_user_tables
WHERE schemaname = 'neurondb'
AND (last_vacuum IS NULL OR last_vacuum < now() - interval '1 day');
```

---

### 4. **neuranllm** - LLM Job Processing Worker (`worker_llm.c`)

**Purpose**: Asynchronous LLM operations

**Responsibilities**:
- **Text Completion**: Generate text with LLMs
- **Embeddings**: Create vector embeddings
- **Reranking**: Semantic reranking of results
- **Cache Integration**: Cache frequently used results
- **Retry Logic**: Handle API failures gracefully

**Configuration**:
```sql
neurondb.neuranllm_naptime = 1000        -- Sleep between cycles (ms)
neurondb.neuranllm_batch_size = 10       -- Jobs per cycle
neurondb.neuranllm_timeout = 30000       -- Job timeout (ms)
neurondb.neuranllm_max_retries = 3       -- Max retry attempts
neurondb.neuranllm_enabled = true
```

**Supported Operations**:
- **completion**: GPT/Claude text generation
- **embedding**: OpenAI/Sentence Transformers
- **reranking**: Cross-encoder models

**Key Features**:
- SKIP LOCKED for concurrent processing
- Per-job retry logic
- Result caching
- Error handling and logging
- Crash-proof with full cleanup
- Memory context isolation

**Monitoring**:
```sql
-- LLM job status
SELECT operation, status, COUNT(*)
FROM neurondb.llm_jobs
GROUP BY operation, status
ORDER BY operation, status;

-- Processing times
SELECT 
    operation,
    AVG(EXTRACT(EPOCH FROM (completed_at - created_at))) as avg_seconds
FROM neurondb.llm_jobs
WHERE status = 'completed'
GROUP BY operation;

-- Failed jobs
SELECT job_id, operation, error_message, retry_count
FROM neurondb.llm_jobs
WHERE status = 'failed'
ORDER BY created_at DESC
LIMIT 10;
```

---

## Installation & Setup

### 1. Enable Workers

Add to `postgresql.conf`:
```
shared_preload_libraries = 'neurondb'
```

Restart PostgreSQL:
```bash
pg_ctl restart
```

### 2. Verify Workers

```sql
-- Check if workers are running
SELECT pid, backend_type, backend_start, state
FROM pg_stat_activity
WHERE backend_type = 'background worker'
AND query LIKE '%neuron%';

-- Check GUC settings
SELECT name, setting, unit, short_desc
FROM pg_settings
WHERE name LIKE 'neurondb.neuran%'
ORDER BY name;
```

### 3. Monitor Workers

```sql
-- Vector statistics
SELECT * FROM neurondb.vector_stats;

-- LLM job status
SELECT * FROM neurondb.llm_job_status;

-- Query performance
SELECT * FROM neurondb.query_performance;
```

---

## Testing

Run the complete test suite:

```bash
cd demo/workers
psql -U postgres -d your_database -f sql/000_run_all_tests.sql
```

Or run individual tests:

```bash
psql -U postgres -d your_database -f sql/001_worker_setup.sql
psql -U postgres -d your_database -f sql/002_queue_worker.sql
psql -U postgres -d your_database -f sql/003_tuner_worker.sql
psql -U postgres -d your_database -f sql/004_defrag_worker.sql
psql -U postgres -d your_database -f sql/005_llm_worker.sql
```

---

## Architecture

### Worker Lifecycle

1. **Initialization** (`_PG_init`)
   - Register with postmaster
   - Allocate shared memory
   - Set up signal handlers

2. **Main Loop** (`*_main`)
   - Wait on latch (interruptible sleep)
   - Process batch of work
   - Update shared memory stats
   - Check for signals (SIGTERM/SIGHUP)

3. **Shutdown** (SIGTERM)
   - Finish current transaction
   - Clean up resources
   - Exit gracefully

### Shared Memory

Each worker maintains shared state:
- Job counters
- Performance metrics
- Last heartbeat timestamp
- Worker PID
- Tenant-specific stats

### Error Handling

All workers use PostgreSQL's error handling:
- `PG_TRY/PG_CATCH` for transactions
- `PG_RE_THROW` for critical errors
- Memory context cleanup on error
- Transaction abort on failure

### Crash Safety

Workers are designed to be crash-proof:
- SIGSEGV handler (defrag worker)
- Memory context isolation
- Transaction boundaries
- Resource cleanup in error paths

---

## Performance Tuning

### Queue Worker
```sql
-- Increase throughput
ALTER SYSTEM SET neurondb.neuranq_batch_size = 200;

-- Reduce latency
ALTER SYSTEM SET neurondb.neuranq_naptime = 500;

-- Higher concurrency
-- Launch multiple queue workers (configure in code)
```

### Auto-Tuner
```sql
-- More aggressive tuning
ALTER SYSTEM SET neurondb.neuranmon_naptime = 30000;  -- 30 seconds

-- Tighter SLOs
ALTER SYSTEM SET neurondb.neuranmon_target_latency = 50.0;
ALTER SYSTEM SET neurondb.neuranmon_target_recall = 0.98;
```

### Defrag Worker
```sql
-- More frequent maintenance
ALTER SYSTEM SET neurondb.neurandefrag_naptime = 60000;  -- 1 minute

-- Lower fragmentation tolerance
ALTER SYSTEM SET neurondb.neurandefrag_fragmentation_threshold = 0.2;
```

### LLM Worker
```sql
-- Larger batches
ALTER SYSTEM SET neurondb.neuranllm_batch_size = 20;

-- Longer timeout for complex jobs
ALTER SYSTEM SET neurondb.neuranllm_timeout = 60000;  -- 60 seconds
```

Apply changes:
```sql
SELECT pg_reload_conf();
```

---

## Troubleshooting

### Workers Not Starting

**Check logs**:
```bash
tail -f /path/to/postgresql/log/postgresql-*.log | grep neuron
```

**Common issues**:
- `shared_preload_libraries` not set
- PostgreSQL not restarted after config change
- Insufficient shared memory

### High Resource Usage

**Check worker activity**:
```sql
SELECT pid, query_start, state, query
FROM pg_stat_activity
WHERE backend_type = 'background worker';
```

**Reduce load**:
```sql
-- Slow down workers
ALTER SYSTEM SET neurondb.neuranq_naptime = 2000;
ALTER SYSTEM SET neurondb.neuranmon_naptime = 120000;
SELECT pg_reload_conf();
```

### Failed Jobs

**Queue worker failures**:
```sql
SELECT job_id, job_type, error_message, retry_count
FROM neurondb.job_queue
WHERE status = 'failed'
ORDER BY created_at DESC
LIMIT 20;
```

**LLM worker failures**:
```sql
SELECT job_id, operation, model_name, error_message, retry_count
FROM neurondb.llm_jobs
WHERE status = 'failed'
ORDER BY created_at DESC
LIMIT 20;
```

---

## Production Best Practices

1. **Monitor worker health**: Check heartbeat timestamps
2. **Set appropriate timeouts**: Balance responsiveness vs completion
3. **Configure maintenance windows**: Run intensive tasks during low traffic
4. **Archive old jobs**: Keep queue tables manageable
5. **Alert on failures**: Monitor failed job counts
6. **Tune based on workload**: Adjust settings for your use case
7. **Review logs regularly**: Watch for errors and warnings

---

## Summary

| Worker | Purpose | Cycle Time | Key Metrics |
|--------|---------|-----------|-------------|
| neuranq | Job Queue | 1 second | Jobs/sec, Queue depth |
| neuranmon | Auto-Tuner | 1 minute | Latency, Recall, ef_search |
| neurandefrag | Maintenance | 5 minutes | Dead ratio, Last vacuum |
| neuranllm | LLM Jobs | 1 second | Completion rate, Errors |

All workers are crash-safe and fully integrated with PostgreSQL's background worker infrastructure.

