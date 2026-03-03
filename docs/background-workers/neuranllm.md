# neuranllm - LLM Job Processor

LLM job processing with crash recovery.

## Overview

neuranllm processes LLM-related jobs with automatic crash recovery.

## Configuration

```conf
shared_preload_libraries = 'neurondb'
neurondb.neuranllm_enabled = true
neurondb_llm_provider = 'huggingface'
neurondb_llm_endpoint = 'https://api-inference.huggingface.co'
neurondb_llm_api_key = 'YOUR_KEY'
```

## LLM Jobs

```sql
-- Check LLM job status
SELECT * FROM neurondb.llm_job_status;

-- View job queue
SELECT * FROM neurondb.llm_jobs WHERE status = 'pending';
```

## Learn More

For detailed documentation on LLM job processing, crash recovery, and error handling, visit:

**[neuranllm Documentation](https://neurondb.ai/docs/background-workers/neuranllm/)**

## Related Topics

- [Background Workers](../background-workers.md) - Overview
- [LLM Integration](../rag/llm-integration.md) - LLM providers

