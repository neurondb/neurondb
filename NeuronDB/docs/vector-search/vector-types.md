# Vector Types

NeuronDB provides multiple vector types optimized for different use cases.

## vector (Standard)

32-bit floating-point vectors for general-purpose use.

```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(384)
);

INSERT INTO documents (content, embedding)
VALUES ('Hello world', '[1.0, 2.0, 3.0]'::vector);
```

## vectorp (Packed)

Optimized storage layout for faster memory access.

```sql
SELECT vectorp_in('[0.1, 0.2, 0.3, 0.4]')::text;
```

## vecmap (Sparse)

Stores only non-zero values, ideal for sparse data.

```sql
SELECT vecmap_in('{
    "dim": 10000,
    "nnz": 3,
    "indices": [0, 5, 9999],
    "values": [1.5, 2.3, 0.8]
}')::text;
```

## vgraph (Graph-Based)

For graph structures with connectivity information.

```sql
SELECT vgraph_in('{
    "nodes": 5,
    "edges": [[0, 1], [1, 2]]
}')::text;
```

## rtext (Retrieval Text)

Specialized type for retrieval-optimized text.

```sql
SELECT rtext_in('retrieval optimized text');
```

## Learn More

For detailed documentation on all vector types, when to use each, storage optimization, and performance characteristics, visit:

**[Vector Types Documentation](https://neurondb.ai/docs/features/vector-types/)**

## Related Topics

- [Distance Metrics](distance-metrics.md) - Measuring vector similarity
- [Indexing](indexing.md) - Creating indexes for fast search
