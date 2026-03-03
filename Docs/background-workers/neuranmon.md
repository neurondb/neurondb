# neuranmon - Live Query Auto-Tuner

Live query auto-tuner and performance optimization.

## Overview

neuranmon monitors query performance and automatically tunes parameters for optimal performance.

## Configuration

```conf
shared_preload_libraries = 'neurondb'
neurondb.neuranmon_enabled = true
neurondb.neuranmon_interval = 60  -- seconds
```

## Monitor Performance

```sql
-- View auto-tuning statistics
SELECT * FROM neurondb.query_performance;

-- Get tuning recommendations
SELECT * FROM neurondb.tuning_recommendations;
```

## Learn More

For detailed documentation on query auto-tuning, performance optimization, and monitoring, visit:

**[neuranmon Documentation](https://neurondb.ai/docs/background-workers/neuranmon/)**

## Related Topics

- [Background Workers](../background-workers.md) - Overview
- [Monitoring](../performance/monitoring.md) - Performance monitoring

