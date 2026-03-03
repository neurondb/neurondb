# neurandefrag - Index Maintenance

Automatic index maintenance and defragmentation.

## Overview

neurandefrag automatically maintains and optimizes vector indexes.

## Configuration

```conf
shared_preload_libraries = 'neurondb'
neurondb.neurandefrag_enabled = true
neurondb.neurandefrag_interval = 3600  -- seconds
```

## Index Maintenance

```sql
-- Check index health
SELECT * FROM neurondb.index_health;

-- View maintenance status
SELECT * FROM neurondb.index_maintenance_status;
```

## Learn More

For detailed documentation on index maintenance, defragmentation strategies, and performance optimization, visit:

**[neurandefrag Documentation](https://neurondb.ai/docs/background-workers/neurandefrag/)**

## Related Topics

- [Background Workers](../background-workers.md) - Overview
- [Indexing](../vector-search/indexing.md) - Index creation

