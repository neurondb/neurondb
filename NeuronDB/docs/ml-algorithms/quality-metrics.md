# Quality Metrics

Evaluate model and search quality using various metrics.

## Recall@K

Fraction of relevant items in top K results:

```sql
-- Calculate Recall@K
SELECT recall_at_k(
    ARRAY[1, 2, 3],      -- retrieved items
    ARRAY[1, 2, 5, 6],   -- relevant items
    5                    -- K
);
```

## Precision@K

Fraction of retrieved items that are relevant:

```sql
-- Calculate Precision@K
SELECT precision_at_k(
    ARRAY[1, 2, 3],
    ARRAY[1, 2, 5],
    5
);
```

## F1@K

Harmonic mean of Precision@K and Recall@K:

```sql
-- Calculate F1@K
SELECT f1_at_k(
    ARRAY[1, 2, 3],
    ARRAY[1, 2, 5],
    5
);
```

## MRR (Mean Reciprocal Rank)

Average reciprocal rank of first relevant result:

```sql
-- Calculate MRR
SELECT mean_reciprocal_rank(
    ARRAY[
        ARRAY[1, 2, 3],
        ARRAY[5, 1, 2]
    ],
    ARRAY[1, 1]  -- relevant items per query
);
```

## Davies-Bouldin Index

Clustering quality metric (lower is better):

```sql
-- Calculate Davies-Bouldin Index
SELECT davies_bouldin_index(
    'data_table',
    'features',
    'cluster_label'
);
```

## Learn More

For detailed documentation on all quality metrics, choosing appropriate metrics, benchmarking, and interpretation, visit:

**[Quality Metrics Documentation](https://neurondb.ai/docs/analytics/quality/)**

## Related Topics

- [Clustering](clustering.md) - Evaluate clustering quality
- [Vector Search](../vector-search/indexing.md) - Evaluate search quality

