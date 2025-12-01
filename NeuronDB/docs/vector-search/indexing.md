# Indexing

NeuronDB provides HNSW (Hierarchical Navigable Small World) and IVF (Inverted File Index) indexing for fast approximate nearest neighbor search.

## HNSW Index

HNSW is a graph-based index optimized for high-dimensional vectors.

### Create HNSW Index

```sql
-- Create HNSW index with default parameters
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
-- K-nearest neighbor search
SELECT id, content,
       embedding <-> embed_text('machine learning') AS distance
FROM documents
ORDER BY embedding <-> embed_text('machine learning')
LIMIT 10;
```

## IVF Index

IVF (Inverted File Index) partitions vectors into clusters for faster search.

### Create IVF Index

```sql
-- Create IVF index
SELECT ivf_create_index(
    'documents',
    'embedding',
    'doc_ivf_idx',
    100  -- number of clusters
);
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

## Related Topics

- [Vector Types](vector-types.md) - Understanding vector data types
- [Distance Metrics](distance-metrics.md) - Distance functions for similarity search
- [Quantization](quantization.md) - Compress vectors for faster search

