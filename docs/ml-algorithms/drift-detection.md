# Drift Detection

Monitor data distribution changes over time to detect concept drift.

## Centroid Drift

Detect changes in data center:

```sql
-- Detect centroid drift
SELECT centroid_drift_detection(
    'reference_table',
    'current_table',
    'features',
    0.1  -- threshold
);
```

## Distribution Divergence

Measure distribution differences:

```sql
-- KL divergence detection
SELECT distribution_divergence(
    'reference_table',
    'current_table',
    'features',
    'kl'  -- divergence type
);
```

## Temporal Monitoring

Track drift over time:

```sql
-- Monitor drift over time windows
SELECT temporal_drift_monitor(
    'time_series_table',
    'features',
    'timestamp_column',
    INTERVAL '1 day'  -- window size
);
```

## Learn More

For detailed documentation on drift detection methods, setting thresholds, alerting, and model retraining strategies, visit:

**[Drift Detection Documentation](https://neurondb.ai/docs/analytics/drift/)**

## Related Topics

- [Outlier Detection](outlier-detection.md) - Detect anomalies
- [Quality Metrics](quality-metrics.md) - Monitor data quality

