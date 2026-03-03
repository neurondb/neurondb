# Troubleshooting

Solutions to common NeuronDB issues, error messages, and configuration problems.

## Learn More

For detailed troubleshooting guides, solutions, and support resources, visit:

**[Troubleshooting Documentation](https://neurondb.ai/docs/troubleshooting/)**

## GPU Acceleration Issues

### Error: "GPU function not available" or "CUDA initialization failed"

**Cause:** GPU support not compiled, CUDA/ROCm not installed, or GPU device unavailable.

**Solutions:**
- Verify NeuronDB was compiled with GPU support: Check build logs for `-DUSE_GPU`
- Ensure CUDA or ROCm drivers are installed: `nvidia-smi` or `rocm-smi`
- Check GPU device availability: `ls /dev/nvidia*` or `ls /dev/kfd`
- Set `neurondb.gpu_fail_open = true` to fallback to CPU on GPU errors

```sql
-- Enable fail-open mode (CPU fallback)
SET neurondb.gpu_fail_open = true;

-- Verify GPU status in logs
SET neurondb.log_level = 'debug';
```

### Warning: GPU operations slower than CPU

**Cause:** Small batch sizes, inefficient memory transfers, or insufficient GPU parallelism.

**Solutions:**
- Increase `neurondb.gpu_batch_size` (try 1000–10000 for large datasets)
- Use multiple streams: `SET neurondb.gpu_streams = 8;`
- Increase memory pool: `SET neurondb.gpu_memory_pool_mb = 1024;`
- GPU acceleration benefits large batch operations; small queries may be faster on CPU

```sql
-- Optimize for large batches
SET neurondb.gpu_batch_size = 5000;
SET neurondb.gpu_streams = 8;
SET neurondb.gpu_memory_pool_mb = 2048;
```

### Error: "GPU out of memory"

**Cause:** Batch size too large for available GPU memory.

**Solutions:**
- Reduce `neurondb.gpu_batch_size`
- Lower `neurondb.gpu_memory_pool_mb`
- Use smaller vector dimensions or quantization
- Close other GPU-using applications

```sql
-- Reduce memory usage
SET neurondb.gpu_batch_size = 500;
SET neurondb.gpu_memory_pool_mb = 256;

-- Use quantization to reduce memory
SELECT vector_to_int8_gpu(embedding) FROM documents;
```

## ML Analytics Issues

### Error: "K-Means did not converge"

**Cause:** Max iterations reached before clusters stabilized.

**Solutions:**
- Increase `max_iter` parameter (e.g., 200, 500)
- Adjust tolerance: lower `tol` for faster convergence
- Try different `k` values; too many clusters can slow convergence
- Check for outliers or scale features before clustering

```sql
-- Increase iterations and relax tolerance
SELECT * FROM cluster_kmeans(
  (SELECT embedding FROM documents),
  5,
  500,  -- increased max_iter
  0.001  -- relaxed tolerance
);
```

### Issue: Clustering produces poor quality results

**Cause:** Inappropriate `k`, non-normalized embeddings, or skewed data distribution.

**Solutions:**
- Normalize embeddings before clustering (especially for GMM)
- Use elbow method or silhouette analysis to find optimal `k`
- Try mini-batch K-Means for very large datasets
- Consider GMM for overlapping clusters

```sql
-- Normalize embeddings
WITH normalized AS (
  SELECT id, embedding / ||embedding|| AS norm_emb
  FROM documents
)
SELECT * FROM cluster_kmeans(
  (SELECT norm_emb FROM normalized),
  5, 100, 0.0001
);
```

### Error: "Outlier detection failed: insufficient data"

**Cause:** Not enough data points for statistical outlier detection.

**Solutions:**
- Ensure at least 30–50 data points for Z-score method
- Lower the Z-score threshold for more sensitive detection
- Combine with other outlier detection methods for robustness

```sql
-- Lower threshold for more outliers
SELECT * FROM detect_outliers_zscore(
  (SELECT embedding FROM documents),
  2.5  -- lower threshold (default 3.0)
);
```

## Indexing and Query Issues

### Issue: Index not being used (sequential scan instead of index scan)

**Cause:** Operator class mismatch, missing index, or planner cost estimation.

**Solutions:**
- Verify index operator class matches query: `vector_l2_ops` for `<->`
- Check index exists: `\d tablename`
- Use `EXPLAIN ANALYZE` to verify index usage
- Increase `effective_cache_size` if planner prefers sequential scan

```sql
-- Verify index usage
EXPLAIN ANALYZE
SELECT * FROM documents
ORDER BY embedding <-> '[1,2,3]'::vector
LIMIT 10;

-- Expected: Index Scan using hnsw_idx on documents

-- Force index usage (if needed)
SET enable_seqscan = off;
```

### Error: "Index build failed: out of memory"

**Cause:** HNSW index requires significant memory during construction.

**Solutions:**
- Increase `maintenance_work_mem` (e.g., 2GB, 4GB)
- Lower `m` or `ef_construction` parameters
- Switch to IVF index for very large datasets
- Build index in smaller batches or use parallel workers

```sql
-- Increase memory for index build
SET maintenance_work_mem = '4GB';

-- Lower HNSW parameters
CREATE INDEX ON documents USING hnsw (embedding vector_l2_ops)
WITH (m = 12, ef_construction = 32);

-- Or switch to IVF
CREATE INDEX ON documents USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);
```

### Issue: Low recall (relevant results not returned)

**Cause:** ANN index approximation or insufficient search width.

**Solutions:**
- Increase `hnsw.ef_search` (HNSW) or `ivfflat.probes` (IVF)
- Rebuild index with higher `ef_construction` (HNSW) or more `lists` (IVF)
- Use exact search (no index) for 100% recall if dataset is small
- Check that distance metric matches embedding model recommendations

```sql
-- Increase search width for HNSW
SET hnsw.ef_search = 200;

-- Increase probes for IVF
SET ivfflat.probes = 20;

-- Exact search (no index)
DROP INDEX documents_embedding_idx;
SELECT * FROM documents ORDER BY embedding <-> query LIMIT 10;
```

## Embedding and LLM Issues

### Error: "LLM API call failed: Unauthorized"

**Cause:** Invalid or missing API key.

**Solutions:**
- Verify `neurondb.llm_api_key` is set correctly
- Check API key has not expired or been revoked
- Ensure API key has embedding permissions (not just completion)
- Use environment variables or secrets manager for production

```sql
-- Set API key (session-level for testing)
SET neurondb.llm_api_key = 'sk-...';

-- Persistent configuration (postgresql.conf or ALTER DATABASE)
ALTER DATABASE mydb SET neurondb.llm_api_key = 'sk-...';
```

### Warning: "LLM API timeout"

**Cause:** Slow network, API rate limits, or large batch size.

**Solutions:**
- Increase `neurondb.llm_timeout_ms` (default 30000ms)
- Reduce batch size for `neurondb_embed_batch`
- Enable retries: `SET neurondb.llm_max_retries = 5;`
- Check network connectivity and API status

```sql
-- Increase timeout and retries
SET neurondb.llm_timeout_ms = 60000;  -- 60 seconds
SET neurondb.llm_max_retries = 5;
```

### Error: "Dimension mismatch"

**Cause:** Vector dimension doesn't match column definition or model output.

**Solutions:**
- Verify embedding model output dimensions (e.g., ada-002: 1536, ada-003: 3072)
- Match column definition: `ALTER TABLE ... ALTER COLUMN embedding TYPE vector(3072);`
- Use correct model name when calling `neurondb_embed`

```sql
-- Check current column dimension
SELECT attname, atttypmod 
FROM pg_attribute 
WHERE attrelid = 'documents'::regclass AND attname = 'embedding';

-- Update column dimension
ALTER TABLE documents ALTER COLUMN embedding TYPE vector(3072);
```

## Performance Issues

### Issue: Slow queries on large datasets

**Cause:** Missing indexes, suboptimal index parameters, or non-selective queries.

**Solutions:**
- Create appropriate vector indexes (HNSW or IVF)
- Tune runtime GUCs: `hnsw.ef_search`, `ivfflat.probes`
- Use LIMIT to reduce result set size
- Consider hybrid search to pre-filter with metadata

### Issue: High memory usage

**Cause:** Large HNSW indexes, high batch sizes, or insufficient memory configuration.

**Solutions:**
- Switch to IVF for memory-constrained environments
- Lower HNSW `m` parameter
- Use vector quantization (int8, fp16, binary) to reduce storage
- Increase `shared_buffers` and `effective_cache_size`

```sql
-- Use quantization to save memory
ALTER TABLE documents ADD COLUMN embedding_int8 vector;
UPDATE documents SET embedding_int8 = vector_to_int8_gpu(embedding);
CREATE INDEX ON documents USING hnsw (embedding_int8 vector_l2_ops);
```

## Diagnostic Tools

### Query Analysis

```sql
-- Detailed query plan
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT * FROM documents
ORDER BY embedding <-> '[1,2,3]'::vector
LIMIT 10;

-- Check index size and statistics
SELECT 
  schemaname, tablename, indexname, 
  pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE tablename = 'documents';
```

### Configuration Check

```sql
-- View current NeuronDB settings
SELECT name, setting, source 
FROM pg_settings 
WHERE name LIKE 'neurondb.%'
ORDER BY name;

-- Check GPU status in logs
SHOW neurondb.log_level;
SET neurondb.log_level = 'debug';
```

### Performance Monitoring

```sql
-- Enable timing
\timing on

-- Track execution statistics
SELECT 
  calls, total_exec_time, mean_exec_time, query
FROM pg_stat_statements
WHERE query LIKE '%embedding%'
ORDER BY mean_exec_time DESC
LIMIT 10;
```

## Getting Help

If you encounter issues not covered here:
- Check PostgreSQL logs for detailed error messages
- Enable `neurondb.log_level = 'debug'` for verbose output
- Review the [Configuration](configuration.md) guide for GUC details
- Consult the [SQL API Reference](sql-api.md) for function signatures
- Visit the NeuronDB GitHub repository for issue tracking and community support
