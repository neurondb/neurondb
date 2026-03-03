# Temporal Search

Time-decay relevance scoring for time-aware retrieval.

## Temporal Search Query

Apply time decay to relevance scores:

```sql
-- Temporal search with decay
SELECT * FROM temporal_search(
    'documents',
    embed_text('query'),
    'timestamp_column',
    INTERVAL '30 days',  -- decay period
    0.5,                 -- decay factor
    10                   -- top K
);
```

## Time-Weighted Ranking

Boost recent documents:

```sql
-- Time-weighted hybrid search
SELECT id, content,
       temporal_hybrid_score(
           embed_text('query'),
           embedding,
           created_at,
           INTERVAL '7 days',
           0.3  -- time weight
       ) AS score
FROM documents
ORDER BY score DESC
LIMIT 10;
```

## Learn More

For detailed documentation on temporal search, decay functions, time windowing, and freshness scoring, visit:

**[Temporal Search Documentation](https://neurondb.ai/docs/hybrid/temporal/)**

## Related Topics

- [Hybrid Search](overview.md) - Combined search
- [Faceted Search](faceted-search.md) - Category filtering

