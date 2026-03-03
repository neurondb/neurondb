# neuranq - Async Job Queue Executor

Async job queue executor with batch processing support.

## Overview

neuranq processes async jobs from the job queue with efficient batch processing.

## Configuration

Enable in `postgresql.conf`:

```conf
shared_preload_libraries = 'neurondb'
neurondb.neuranq_enabled = true
neurondb.neuranq_queue_depth = 10000
neurondb.neuranq_naptime = 1000  -- milliseconds
```

## Queue Jobs

Add jobs to the queue:

```sql
-- Add job to queue
INSERT INTO neurondb.job_queue (job_type, job_data)
VALUES ('embedding', '{"text": "Hello world"}'::jsonb);
```

## Monitor Queue

```sql
-- Check queue status
SELECT * FROM neurondb.job_queue WHERE status = 'pending';

-- Queue statistics
SELECT status, COUNT(*) FROM neurondb.job_queue GROUP BY status;
```

## Learn More

For detailed documentation on job queue management, batch processing, error handling, and performance tuning, visit:

**[neuranq Documentation](https://neurondb.ai/docs/background-workers/neuranq/)**

## Related Topics

- [Background Workers](../background-workers.md) - Overview
- [neuranmon](neuranmon.md) - Auto-tuner worker

