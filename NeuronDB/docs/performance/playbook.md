# Performance Playbook

This playbook provides practical guidance for optimizing NeuronDB performance, selecting appropriate indexes, and measuring query performance.

## Index Selection: HNSW vs. IVF

### When to Use HNSW

**Use HNSW when:**
- **Query latency is critical:** HNSW provides the fastest query times (typically <10ms for k=10 on datasets <1M vectors)
- **Dataset size:** Works well from 10K to 100M+ vectors
- **High recall requirements:** HNSW maintains high recall (95%+) even with low `ef_search` values
- **Frequent updates:** HNSW supports incremental updates better than IVF
- **Memory is available:** HNSW indexes are larger (typically 2-4x vector data size)

**HNSW Characteristics:**
- Build time: O(N log N) - slower for large datasets
- Query time: O(log N) - very fast
- Memory: Higher (stores graph structure)
- Update cost: Low (can add vectors incrementally)

**Example:**
```sql
-- HNSW for fast queries on medium datasets
CREATE INDEX documents_hnsw_idx ON documents 
USING hnsw (embedding vector_l2_ops) 
WITH (m = 16, ef_construction = 64);
```

### When to Use IVF

**Use IVF when:**
- **Large datasets:** IVF scales better to 100M+ vectors
- **Memory is constrained:** IVF indexes are smaller (typically 1.5-2x vector data size)
- **Batch queries:** IVF performs better when processing many queries at once
- **Build time matters:** IVF builds faster than HNSW for very large datasets
- **Lower recall acceptable:** IVF may require higher `nprobe` to achieve high recall

**IVF Characteristics:**
- Build time: O(N) - faster for very large datasets
- Query time: O(nprobe * N/k) - depends on nprobe setting
- Memory: Lower (stores centroids and inverted lists)
- Update cost: Higher (may require rebuilding)

**Example:**
```sql
-- IVF for large datasets with memory constraints
CREATE INDEX documents_ivf_idx ON documents 
USING ivfflat (embedding vector_l2_ops) 
WITH (lists = 1000);
```

### Decision Matrix

| Dataset Size | Query Latency | Memory Available | Recommended Index |
|--------------|---------------|------------------|-------------------|
| < 1M vectors | Critical | High | HNSW |
| < 1M vectors | Critical | Low | HNSW (with lower m) |
| 1M - 10M | Important | High | HNSW |
| 1M - 10M | Important | Low | IVF |
| 10M - 100M | Important | High | HNSW or IVF |
| 10M - 100M | Important | Low | IVF |
| > 100M | Acceptable | Any | IVF |

## Index Build Time and Memory Sizing

### HNSW Build Time Estimation

**Build time formula (approximate):**
```
Build time (seconds) ≈ (N * log(N) * ef_construction) / (CPU_cores * ops_per_second)
```

**Typical build times:**
- 100K vectors: 10-30 seconds
- 1M vectors: 2-5 minutes
- 10M vectors: 20-60 minutes
- 100M vectors: 3-10 hours

**Memory requirements:**
```
Memory (bytes) ≈ N * (dimension * 4 + m * 8 * 2)
```

**Example calculation:**
- 1M vectors, 768 dimensions, m=16
- Memory ≈ 1,000,000 * (768 * 4 + 16 * 8 * 2)
- Memory ≈ 1,000,000 * (3072 + 256) = ~3.3 GB

### IVF Build Time Estimation

**Build time formula (approximate):**
```
Build time (seconds) ≈ (N * dimension) / (CPU_cores * ops_per_second)
```

**Typical build times:**
- 100K vectors: 5-15 seconds
- 1M vectors: 1-3 minutes
- 10M vectors: 10-30 minutes
- 100M vectors: 2-6 hours

**Memory requirements:**
```
Memory (bytes) ≈ N * dimension * 4 + lists * (dimension * 4 + overhead)
```

### Optimizing Build Performance

1. **Parallel builds:** Use `CREATE INDEX CONCURRENTLY` when possible
2. **Batch inserts:** Insert vectors in batches of 1000-10000
3. **Tune construction parameters:**
   - HNSW: Lower `ef_construction` (default 64) for faster builds, higher for better quality
   - IVF: Lower `lists` (default 100) for faster builds, higher for better recall
4. **Increase work_mem:** Set `work_mem = '256MB'` or higher during index creation
5. **Disable autovacuum:** Temporarily disable during large index builds

## nprobe Tuning for IVF

### Understanding nprobe

`nprobe` controls how many inverted list clusters (centroids) are searched during query time. Higher values improve recall but increase query latency.

### Default nprobe

**Default:** `nprobe = 10` (or `lists / 10`, whichever is smaller)

### When to Adjust nprobe

**Increase nprobe when:**
- Recall is too low (< 90%)
- You can tolerate higher query latency
- Dataset has high variance (vectors spread across many clusters)

**Decrease nprobe when:**
- Query latency is too high
- Recall is acceptable (> 95%)
- Dataset has low variance (vectors clustered in few groups)

### nprobe Guidelines

| Dataset Size | Lists | Recommended nprobe | Expected Recall | Query Latency |
|--------------|-------|-------------------|-----------------|---------------|
| < 100K | 100 | 10 | 95%+ | < 5ms |
| 100K - 1M | 1000 | 10-50 | 90-95% | 5-20ms |
| 1M - 10M | 1000 | 50-100 | 90-95% | 20-50ms |
| 10M - 100M | 10000 | 100-200 | 85-95% | 50-200ms |
| > 100M | 10000 | 200-500 | 80-95% | 200ms-1s |

### Setting nprobe

```sql
-- Set nprobe for current session
SET ivfflat.nprobe = 50;

-- Set nprobe for specific query
SET LOCAL ivfflat.nprobe = 100;
SELECT * FROM documents 
ORDER BY embedding <-> query_vector 
LIMIT 10;
```

## Query Latency Targets

### Expected Latency by Dataset Size

**HNSW Index:**

| Dataset Size | k=10 | k=100 | k=1000 |
|--------------|------|-------|--------|
| 10K vectors | < 1ms | < 2ms | < 5ms |
| 100K vectors | < 2ms | < 5ms | < 10ms |
| 1M vectors | < 5ms | < 10ms | < 20ms |
| 10M vectors | < 10ms | < 20ms | < 50ms |
| 100M vectors | < 20ms | < 50ms | < 100ms |

**IVF Index (nprobe=10):**

| Dataset Size | k=10 | k=100 | k=1000 |
|--------------|------|-------|--------|
| 10K vectors | < 2ms | < 3ms | < 5ms |
| 100K vectors | < 5ms | < 10ms | < 15ms |
| 1M vectors | < 10ms | < 20ms | < 30ms |
| 10M vectors | < 20ms | < 50ms | < 100ms |
| 100M vectors | < 50ms | < 200ms | < 500ms |

### Latency Percentiles

For production systems, target:
- **P50 (median):** < 10ms for k=10 on datasets < 1M
- **P95:** < 50ms for k=10 on datasets < 1M
- **P99:** < 100ms for k=10 on datasets < 1M

## Measuring Query Performance

### Using EXPLAIN ANALYZE

```sql
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT id, content, embedding <-> query_vector AS distance
FROM documents
ORDER BY embedding <-> query_vector
LIMIT 10;
```

**Key metrics to check:**
- **Execution Time:** Total query time
- **Index Scan:** Should use your HNSW or IVF index
- **Buffers:** Shared blocks read (indicates cache efficiency)

### Using pg_stat_statements

Enable and query statistics:

```sql
-- Enable pg_stat_statements
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Find slowest vector queries
SELECT 
  query,
  calls,
  mean_exec_time,
  max_exec_time,
  (total_exec_time / 1000) as total_seconds
FROM pg_stat_statements
WHERE query LIKE '%<->%' OR query LIKE '%<=>%'
ORDER BY mean_exec_time DESC
LIMIT 10;
```

### Custom Performance Monitoring

```sql
-- Create performance tracking table
CREATE TABLE query_performance (
  id SERIAL PRIMARY KEY,
  query_type TEXT,
  dataset_size BIGINT,
  k_value INT,
  execution_time_ms FLOAT,
  recall FLOAT,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Log query performance
INSERT INTO query_performance (query_type, dataset_size, k_value, execution_time_ms)
SELECT 
  'knn_search',
  (SELECT COUNT(*) FROM documents),
  10,
  EXTRACT(EPOCH FROM (clock_timestamp() - statement_timestamp())) * 1000;
```

## Performance Optimization Checklist

- [ ] **Index selection:** Chosen HNSW or IVF based on requirements
- [ ] **Index parameters:** Tuned `m`, `ef_construction` (HNSW) or `lists` (IVF)
- [ ] **nprobe setting:** Adjusted for IVF to balance recall and latency
- [ ] **work_mem:** Increased for index builds and complex queries
- [ ] **Connection pooling:** Configured appropriate pool size
- [ ] **Query patterns:** Optimized for common query patterns
- [ ] **Monitoring:** Set up performance tracking and alerts
- [ ] **GPU acceleration:** Enabled if available and beneficial
- [ ] **Batch operations:** Used batch inserts and batch queries when possible

## Troubleshooting Slow Queries

1. **Check index usage:**
   ```sql
   EXPLAIN SELECT ...;
   -- Should show "Index Scan using ..._hnsw_idx" or "..._ivf_idx"
   ```

2. **Verify index exists:**
   ```sql
   SELECT indexname, indexdef 
   FROM pg_indexes 
   WHERE tablename = 'documents';
   ```

3. **Check index statistics:**
   ```sql
   ANALYZE documents;
   SELECT * FROM pg_stat_user_indexes WHERE indexrelname LIKE '%hnsw%';
   ```

4. **Monitor resource usage:**
   ```sql
   SELECT * FROM pg_stat_activity WHERE state = 'active';
   ```

5. **Check for table bloat:**
   ```sql
   SELECT schemaname, tablename, 
          pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
   FROM pg_tables 
   WHERE tablename = 'documents';
   ```

## Related Documentation

- [Indexing Guide](vector-search/indexing.md) - Detailed index configuration
- [Distance Metrics](vector-search/distance-metrics.md) - Distance function selection
- [GPU Acceleration](gpu/cuda-support.md) - GPU performance optimization
- [Monitoring](monitoring.md) - Performance monitoring setup








