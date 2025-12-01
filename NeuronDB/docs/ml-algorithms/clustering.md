# Clustering

NeuronDB provides multiple clustering algorithms for unsupervised learning.

## K-Means

Partition data into k clusters:

```sql
CREATE TEMP TABLE kmeans_model AS
SELECT train_kmeans_model_id('data_table', 'features', 3, 100) AS model_id;
```

## Mini-batch K-Means

Faster version for large datasets:

```sql
SELECT train_minibatch_kmeans('data_table', 'features', 3, 100) AS model_id;
```

## DBSCAN

Density-based clustering:

```sql
SELECT train_dbscan('data_table', 'features', 0.5, 5) AS model_id;
```

## GMM (Gaussian Mixture Model)

Probabilistic clustering:

```sql
CREATE TEMP TABLE gmm_model AS
SELECT train_gmm_model_id('data_table', 'features', 3) AS model_id;
```

## Hierarchical Clustering

```sql
SELECT train_hierarchical_clustering('data_table', 'features', 3) AS model_id;
```

## Learn More

For detailed documentation on clustering algorithms, choosing parameters, evaluating clusters, and visualization, visit:

**[Clustering Documentation](https://neurondb.ai/docs/analytics/clustering/)**

## Related Topics

- [Dimensionality Reduction](dimensionality-reduction.md) - Reduce dimensions before clustering
- [Quality Metrics](quality-metrics.md) - Evaluate clustering quality
