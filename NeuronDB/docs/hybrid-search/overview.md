# Hybrid Search

Combine vector and full-text search with configurable weights.

## Basic Hybrid Search

Combine semantic and keyword search:

```sql
SELECT * FROM hybrid_search(
    'documents',
    embed_text('machine learning'),
    'neural networks',
    '{}',
    0.7,
    10
);
```

## Learn More

For detailed documentation on hybrid search strategies, weight tuning, filtering, and performance optimization, visit:

**[Hybrid Search Documentation](https://neurondb.ai/docs/hybrid/overview/)**

## Related Topics

- [Vector Search](../vector-search/indexing.md) - Vector similarity search
- [Multi-Vector](multi-vector.md) - Multiple embeddings per document
- [Faceted Search](faceted-search.md) - Category-aware search

