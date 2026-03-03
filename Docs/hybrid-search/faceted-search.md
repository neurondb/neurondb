# Faceted Search

Category-aware retrieval with filtering.

## Faceted Search Query

```sql
-- Faceted search with category filtering
SELECT * FROM faceted_search(
    'documents',
    embed_text('query'),
    '{"category": "AI", "language": "en"}'::jsonb,
    10
);
```

## Multiple Facets

Filter by multiple categories:

```sql
-- Multi-facet search
SELECT * FROM faceted_search(
    'documents',
    embed_text('machine learning'),
    '{
        "category": ["AI", "ML"],
        "year": [2023, 2024],
        "status": "published"
    }'::jsonb,
    20
);
```

## Facet Aggregation

Get facet counts:

```sql
-- Get facet distribution
SELECT facet_counts(
    'documents',
    '{"category": "AI"}'::jsonb,
    ARRAY['year', 'language']
) AS facet_stats;
```

## Learn More

For detailed documentation on faceted search, facet hierarchies, filtering strategies, and performance optimization, visit:

**[Faceted Search Documentation](https://neurondb.ai/docs/hybrid/faceted/)**

## Related Topics

- [Hybrid Search](overview.md) - Combined search strategies
- [Temporal Search](temporal-search.md) - Time-based filtering

