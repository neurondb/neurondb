# ColBERT Reranking

Late interaction models for efficient reranking.

## ColBERT Reranking

```sql
-- ColBERT reranking
SELECT idx, score FROM rerank_colbert(
    'query text',
    ARRAY['doc 1', 'doc 2', 'doc 3'],
    'colbert-base-msmarco',
    5
);
```

## Learn More

For detailed documentation on ColBERT models, late interaction, and efficiency optimization, visit:

**[ColBERT Documentation](https://neurondb.ai/docs/reranking/colbert/)**

## Related Topics

- [Cross-Encoder](cross-encoder.md) - Neural reranking
- [Ensemble](ensemble.md) - Combine strategies

