# Indexing

NeuronDB provides HNSW (Hierarchical Navigable Small World) and IVF (Inverted File Index) indexing for fast approximate nearest neighbor search.

## HNSW Index

HNSW is a graph-based index optimized for high-dimensional vectors.

### Create HNSW Index

**Standard SQL (pgvector-compatible):**
```sql
-- Create HNSW index with default parameters
CREATE INDEX ON documents USING hnsw (embedding vector_l2_ops);

-- With custom parameters
CREATE INDEX ON documents USING hnsw (embedding vector_l2_ops) 
WITH (m = 16, ef_construction = 200);

-- Cosine distance index
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 200);

-- Inner product index (for maximum inner product search)
CREATE INDEX ON documents USING hnsw (embedding vector_ip_ops) 
WITH (m = 16, ef_construction = 200);
```

**NeuronDB helper function (alternative):**
```sql
-- Create HNSW index with helper function
SELECT hnsw_create_index(
    'documents',      -- table name
    'embedding',      -- column name
    'doc_idx',        -- index name
    16,               -- m (connections per layer)
    200               -- ef_construction (build-time search width)
);
```

### Search with HNSW

```sql
-- K-nearest neighbor search (L2 distance)
SELECT id, content,
       embedding <-> embed_text('machine learning') AS distance
FROM documents
ORDER BY embedding <-> embed_text('machine learning')
LIMIT 10;

-- Cosine similarity search
SELECT id, content,
       embedding <=> embed_text('machine learning') AS cosine_distance
FROM documents
ORDER BY embedding <=> embed_text('machine learning')
LIMIT 10;

-- Maximum inner product search
SELECT id, content,
       embedding <#> embed_text('machine learning') AS ip_distance
FROM documents
ORDER BY embedding <#> embed_text('machine learning')
LIMIT 10;

-- Filtered search (index + WHERE clause)
SELECT id, content, embedding <-> '[0.1,0.2,0.3]'::vector AS distance
FROM documents
WHERE category = 'technology'
ORDER BY embedding <-> '[0.1,0.2,0.3]'::vector
LIMIT 10;
```

## IVF Index

IVF (Inverted File Index) partitions vectors into clusters for faster search. Best for large datasets with many vectors.

### Create IVF Index

**Standard SQL (pgvector-compatible):**
```sql
-- Create IVF index with default parameters (NeuronDB uses 'ivf', pgvector uses 'ivfflat')
CREATE INDEX ON documents USING ivf (embedding vector_l2_ops);

-- With custom parameters
CREATE INDEX ON documents USING ivf (embedding vector_l2_ops) 
WITH (lists = 100);

-- Cosine distance index
CREATE INDEX ON documents USING ivf (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Inner product index (now supported!)
CREATE INDEX ON documents USING ivf (embedding vector_ip_ops) 
WITH (lists = 100);

-- pgvector compatibility: 'ivfflat' also works (aliased to 'ivf')
CREATE INDEX ON documents USING ivfflat (embedding vector_l2_ops) 
WITH (lists = 100);
```

**NeuronDB helper function (alternative):**
```sql
-- Create IVF index with helper function
SELECT ivf_create_index(
    'documents',
    'embedding',
    'doc_ivf_idx',
    100  -- number of clusters
);
```

### Choosing Parameters

- **lists**: Number of clusters (centroids). Typical values:
  - Small datasets (< 1M vectors): `sqrt(row_count)`
  - Medium datasets (1M-100M): 100-1000
  - Large datasets (> 100M): 1000-10000

### Search with IVF

```sql
-- Basic K-NN search
SELECT id, content, embedding <-> '[0.1,0.2,0.3]'::vector AS distance
FROM documents
ORDER BY embedding <-> '[0.1,0.2,0.3]'::vector
LIMIT 10;

-- Query-time tuning (via GUC)
SET neurondb.ivf_probes = 20;  -- Number of clusters to probe
SELECT id, content FROM documents
ORDER BY embedding <-> '[0.1,0.2,0.3]'::vector
LIMIT 10;
RESET neurondb.ivf_probes;
```

## Index Maintenance

```sql
-- Check index health
SELECT * FROM neurondb.index_health;

-- Rebuild index
SELECT hnsw_rebuild_index('doc_idx');
```

## Learn More

For detailed documentation on indexing strategies, parameter tuning, automatic maintenance, and performance optimization, visit:

**[Indexing Documentation](https://neurondb.ai/docs/features/indexing/)**

## Choosing Between HNSW and IVF

- **HNSW**: Best for high recall requirements, dynamic datasets, and when query latency is critical
  - Higher memory usage
  - Better for smaller to medium datasets (< 100M vectors)
  - Excellent for real-time applications
  
- **IVF**: Best for very large datasets, lower memory usage, and when approximate results are acceptable
  - Lower memory usage
  - Better for large-scale deployments (100M+ vectors)
  - Faster index build time

**Recommendation**: Start with HNSW for most use cases. Consider IVF for very large datasets where memory is constrained.

## pgvector Compatibility

NeuronDB is fully compatible with pgvector syntax and behavior. See [pgvector Compatibility Guide](pgvector-compatibility.md) for details.

Example migration:
```sql
-- pgvector syntax (works identically in NeuronDB)
CREATE INDEX ON items USING hnsw (embedding vector_l2_ops);
CREATE INDEX ON items USING ivfflat (embedding vector_l2_ops);  -- Uses 'ivf' internally

SELECT * FROM items 
ORDER BY embedding <-> '[1,2,3]'::vector 
LIMIT 10;
```

## Related Topics

- [Vector Types](vector-types.md) - Understanding vector data types
- [Distance Metrics](distance-metrics.md) - Distance functions for similarity search
- [Quantization](quantization.md) - Compress vectors for faster search
- [pgvector Compatibility](pgvector-compatibility.md) - Migration guide and compatibility matrix

