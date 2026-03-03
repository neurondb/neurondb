# Distance Metrics

NeuronDB supports multiple distance metrics for measuring vector similarity.

## L2 (Euclidean) Distance

Straight-line distance between two points. Lower values indicate more similar vectors.

```sql
-- L2 distance operator
SELECT embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
ORDER BY distance
LIMIT 10;
```

## Cosine Distance

Measures the angle between vectors. Ideal for normalized vectors where direction matters more than magnitude.

```sql
-- Cosine distance operator
SELECT embedding <=> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
ORDER BY distance
LIMIT 10;
```

## Inner Product

Dot product of two vectors. Higher values indicate greater similarity. Best used with normalized vectors.

```sql
-- Inner product (negative for ordering)
SELECT embedding <#> '[0.1, 0.2, 0.3]'::vector AS neg_inner_product
FROM documents
ORDER BY neg_inner_product
LIMIT 10;
```

## Other Metrics

NeuronDB also supports:
- Manhattan (L1) distance
- Hamming distance (for binary vectors)
- Jaccard distance (for sets)

## Learn More

For detailed documentation on all distance metrics, when to use each, GPU acceleration, and performance characteristics, visit:

**[Distance Metrics Documentation](https://neurondb.ai/docs/features/distance-metrics/)**

## Related Topics

- [Vector Types](vector-types.md) - Vector data types
- [Indexing](indexing.md) - Creating indexes for fast search

