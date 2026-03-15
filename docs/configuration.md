# Configuration

Key NeuronDB settings are exposed as PostgreSQL GUCs. Set them in `postgresql.conf` or via `ALTER SYSTEM`, then reload or restart as required. All GUCs below are defined in the extension source (`src/util/neurondb_guc.c`, `src/metrics/prometheus.c`).

## Shared preload

```conf
shared_preload_libraries = 'neurondb'
```

Required for background workers (neuranq, neuranmon, neurandefrag, neuranllm, Prometheus exporter) and certain shared memory features. Restart the postmaster after changing.

## Index and search

| GUC | Type | Default | Description |
|-----|------|---------|-------------|
| `neurondb.hnsw_ef_search` | int | 64 | HNSW ef_search (recall vs speed). 1–10000. |
| `neurondb.hnsw_k` | int | 10 | HNSW k (neighbors returned). 1–1000. |
| `neurondb.ivf_probes` | int | 10 | IVF probes. 1–1000. |
| `neurondb.ef_construction` | int | 200 | HNSW build quality. 4–2000. |
| `neurondb.hnsw_iterative_scan` | enum | off | off, strict_order, relaxed_order. |
| `neurondb.hnsw_max_scan_tuples` | int | 20000 | Max tuples for iterative scan. |
| `neurondb.hnsw_scan_mem_multiplier` | real | 1.0 | work_mem multiplier for iterative scan. |
| `neurondb.ivf_iterative_scan` | enum | off | off, relaxed_order. |
| `neurondb.ivf_max_probes` | int | 100 | Max IVF probes for iterative scan. |

## GPU

| GUC | Type | Default | Description |
|-----|------|---------|-------------|
| `neurondb.compute_mode` | int | 0 | 0=cpu, 1=gpu, 2=auto. |
| `neurondb.gpu_device` | int | 0 | GPU device ID (0-based). |
| `neurondb.gpu_batch_size` | int | 8192 | GPU batch size. |
| `neurondb.gpu_streams` | int | 2 | CUDA/HIP streams. |
| `neurondb.gpu_memory_pool_mb` | real | 512 | GPU memory pool (MB). |
| `neurondb.gpu_kernels` | string | l2,cosine,ip,... | Comma-separated kernel list. |
| `neurondb.gpu_timeout_ms` | int | 30000 | GPU kernel timeout (ms). |
| `neurondb.gpu_backend_type` | int | platform | 0=CUDA, 1=ROCm, 2=Metal. |

## LLM/Embeddings

| GUC | Type | Default | Description |
|-----|------|---------|-------------|
| `neurondb.llm_provider` | string | huggingface | LLM provider. |
| `neurondb.llm_model` | string | sentence-transformers/all-MiniLM-L6-v2 | Default model. |
| `neurondb.llm_endpoint` | string | (null) | Endpoint URL. |
| `neurondb.llm_api_key` | string | (null) | API key (use env or secrets in production). |
| `neurondb.llm_timeout_ms` | int | 30000 | Timeout (ms). |
| `neurondb.llm_cache_ttl` | int | 600 | Cache TTL (seconds). |
| `neurondb.llm_rate_limiter_qps` | int | 5 | Rate limit QPS. |
| `neurondb.llm_fail_open` | bool | true | If true, failures allow fallback. |

## Workers (require shared_preload_libraries)

Queue (neuranq): `neurondb.neuranq_naptime`, `neurondb.neuranq_queue_depth`, `neurondb.neuranq_batch_size`, `neurondb.neuranq_timeout`, `neurondb.neuranq_max_retries`, `neurondb.neuranq_enabled`.  
Monitor (neuranmon): `neurondb.neuranmon_*`.  
Defrag (neurandefrag): `neurondb.neurandefrag_*`.  

## ONNX

| GUC | Type | Default | Description |
|-----|------|---------|-------------|
| `neurondb.onnx_model_path` | string | (null) | Path to ONNX model. |
| `neurondb.onnx_use_gpu` | bool | true | Use GPU for ONNX. |
| `neurondb.onnx_threads` | int | 4 | Thread count. |
| `neurondb.onnx_cache_size` | int | 10 | Cache size. |

## Quotas and limits

| GUC | Type | Default | Description |
|-----|------|---------|-------------|
| `neurondb.default_max_vectors` | int64 | 1000000 | Default max vectors per index. |
| `neurondb.default_max_storage_mb` | int64 | 10240 | Default max storage (MB). |
| `neurondb.default_max_qps` | int | 1000 | Default max QPS. |
| `neurondb.enforce_quotas` | bool | true | Enforce quota limits. |

## ML and security

ML: `neurondb.ml_max_samples`, `neurondb.ml_max_feature_elements`, `neurondb.automl_use_gpu`, `neurondb.vector_capsule_enabled`.  
Security/audit: `neurondb.confidential_compute`, `neurondb.rls_embeddings_enabled`, `neurondb.encryption_enabled`, `neurondb.audit_ml_enabled`, `neurondb.audit_rag_enabled`, `neurondb.audit_retention_days`.  
Replication: `neurondb.enable_replication`.

## Prometheus (require shared_preload_libraries)

Defined in `src/metrics/prometheus.c`: `neurondb.prometheus_enabled`, `neurondb.prometheus_port` (default 9187), `neurondb.prometheus_host` (default 0.0.0.0). Restart postmaster after change.

## Secrets

Avoid storing API keys in `postgresql.conf` in production. Use environment variables, `include_dir`, or a secrets manager with restricted file permissions.
