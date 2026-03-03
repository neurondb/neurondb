# Outlier Detection

Detect anomalies and outliers in your data using statistical methods.

## Z-Score Method

Identify outliers using standard deviations:

```sql
-- Detect outliers using Z-score
SELECT id, features,
       z_score_outlier_detection(features, 3.0) AS is_outlier
FROM data_table;
-- is_outlier = true if |z-score| > 3.0
```

## Modified Z-Score

More robust to outliers than standard Z-score:

```sql
-- Modified Z-score detection
SELECT id,
       modified_zscore_outlier(features, 3.5) AS is_outlier
FROM data_table;
```

## IQR (Interquartile Range) Method

Detect outliers using quartiles:

```sql
-- IQR-based outlier detection
SELECT id,
       iqr_outlier(features) AS is_outlier
FROM data_table;
```

## Isolation Forest

Tree-based anomaly detection:

```sql
-- Isolation Forest
SELECT isolation_forest_outlier(
    'data_table',
    'features',
    0.1  -- contamination rate
);
```

## Learn More

For detailed documentation on outlier detection methods, choosing thresholds, visualization, and handling outliers, visit:

**[Outlier Detection Documentation](https://neurondb.ai/docs/analytics/outliers/)**

## Related Topics

- [Drift Detection](drift-detection.md) - Detect data distribution changes
- [Quality Metrics](quality-metrics.md) - Data quality assessment

