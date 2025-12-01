# Monitoring

7 built-in monitoring views and Prometheus metrics export.

## Monitoring Views

NeuronDB provides comprehensive monitoring views:

```sql
-- Vector statistics
SELECT * FROM neurondb.vector_stats;

-- Index health dashboard
SELECT * FROM neurondb.index_health;

-- Tenant quota usage
SELECT * FROM neurondb.tenant_quota_usage;

-- LLM job queue status
SELECT * FROM neurondb.llm_job_status;

-- Query performance metrics (last 24h)
SELECT * FROM neurondb.query_performance;

-- Index maintenance operations
SELECT * FROM neurondb.index_maintenance_status;

-- Prometheus metrics summary
SELECT * FROM neurondb.metrics_summary;
```

## Extension Statistics

```sql
-- Extension statistics
SELECT * FROM pg_stat_neurondb();

-- Worker status
SELECT * FROM neurondb_worker_status();
```

## Prometheus Export

Prometheus metrics are available for external monitoring.

## Learn More

For detailed documentation on monitoring, metrics interpretation, and alerting, visit:

**[Monitoring Documentation](https://neurondb.ai/docs/performance/monitoring/)**

## Related Topics

- [SIMD Optimization](simd-optimization.md) - Performance optimization
- [Configuration](../configuration.md) - Configuration options

