# Configuration

Key NeuronDB settings are exposed as PostgreSQL GUCs. Configure them in postgresql.conf or via ALTER SYSTEM, then reload or restart as required.

## Shared preload

```conf
shared_preload_libraries = 'neurondb'
```

Required for background workers and certain shared memory features.

## LLM/Embeddings

From `worker_llm.c` and `ml/embeddings.c`:

- neurondb_llm_provider   (text)
- neurondb_llm_endpoint   (text)
- neurondb_llm_model      (text)
- neurondb_llm_api_key    (text)
- neurondb_llm_timeout_ms (integer)

Example:

```conf
neurondb_llm_provider   = 'huggingface'
neurondb_llm_endpoint   = 'https://api-inference.huggingface.co'
neurondb_llm_model      = 'all-MiniLM-L6-v2'
neurondb_llm_api_key    = 'YOUR_KEY'
neurondb_llm_timeout_ms = 15000
```

## Queue/Workers

From `worker_queue.c`:

- neurondb.neuranq_queue_depth (integer)

```conf
neurondb.neuranq_queue_depth = 10000
```

## Observability

- Set server log level to DEBUG1 for additional planner and worker routing insights.
- Workers and subsystems emit elog(LOG/DEBUG) entries on lifecycle and significant events.

## Secrets

- Avoid placing API keys directly in postgresql.conf in production; use environment variables and include_dir, or a secrets manager with file permissions tightened.

## Learn More

For detailed documentation on all configuration options, settings, and optimization, visit:

**[Configuration Documentation](https://neurondb.ai/docs/configuration/)**
