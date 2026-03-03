# Quantization

Quantization compresses vectors to reduce storage and increase search speed with minimal accuracy loss.

## Product Quantization (PQ)

PQ divides vectors into subvectors and quantizes each subvector separately.

### Create PQ Index

```sql
-- Create product quantization index
SELECT pq_create_index(
    'documents',
    'embedding',
    'doc_pq_idx',
    8,   -- number of subvectors
    256  -- codebook size per subvector
);
```

### Search with PQ

```sql
-- Search using quantized index
SELECT id, content
FROM documents
ORDER BY embedding <-> embed_text('query text')
LIMIT 10;
```

## Optimized Product Quantization (OPQ)

OPQ adds rotation before quantization for better compression.

```sql
-- Create OPQ index
SELECT opq_create_index(
    'documents',
    'embedding',
    'doc_opq_idx',
    8,   -- subvectors
    256  -- codebook size
);
```

## Compression Ratio

PQ and OPQ can achieve 2x-32x compression:
- 2x: High accuracy, minimal compression
- 32x: Maximum compression, slight accuracy trade-off

## Learn More

For detailed documentation on quantization techniques, choosing parameters, accuracy trade-offs, and hybrid quantization, visit:

**[Quantization Documentation](https://neurondb.ai/docs/features/quantization/)**

## Related Topics

- [Vector Types](vector-types.md) - Vector data types
- [Indexing](indexing.md) - Creating indexes
- [Distance Metrics](distance-metrics.md) - Similarity measures

