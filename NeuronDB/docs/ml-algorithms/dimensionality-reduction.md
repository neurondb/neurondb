# Dimensionality Reduction

Reduce vector dimensions while preserving important information using PCA and whitening.

## PCA (Principal Component Analysis)

Reduce dimensions while preserving variance:

```sql
-- PCA transformation
SELECT pca_transform(
    'data_table',
    'features',
    128,  -- target dimensions
    'pca_model'
);

-- Apply PCA to new data
SELECT pca_apply(features, 'pca_model') AS reduced_features
FROM test_table;
```

## PCA Whitening

Standardize variance across components:

```sql
-- PCA with whitening
SELECT pca_whiten(
    'data_table',
    'features',
    128,
    'pca_whitened_model'
);
```

## Benefits

- Reduce storage requirements
- Speed up training and inference
- Remove noise and redundant information
- Visualize high-dimensional data

## Learn More

For detailed documentation on PCA, whitening, choosing dimensions, and inverse transformation, visit:

**[Dimensionality Reduction Documentation](https://neurondb.ai/docs/analytics/dimensionality/)**

## Related Topics

- [Clustering](clustering.md) - Apply clustering after reduction
- [Quality Metrics](quality-metrics.md) - Evaluate reduction quality

