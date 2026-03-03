# SQL API Reference

Complete reference for all NeuronDB SQL functions, operators, types, and configuration parameters.

## Table of Contents

- [Vector Types](#vector-types)
- [Distance Operators](#distance-operators)
- [Embedding Functions](#embedding-functions)
- [GPU Distance Functions](#gpu-distance-functions)
- [ML Analytics Functions](#ml-analytics-functions)
- [ML Project Management](#ml-project-management)

## Vector Types

### `vector(n)`

A fixed-dimension vector of floating-point numbers.

```sql
-- Create table with vector column
CREATE TABLE embeddings (
  id SERIAL PRIMARY KEY,
  embedding vector(1536)  -- 1536-dimensional vector
);

-- Insert vectors
INSERT INTO embeddings (embedding) VALUES ('[1, 2, 3]');
INSERT INTO embeddings (embedding) VALUES (ARRAY[1.0, 2.0, 3.0]::vector);

-- Cast from array
SELECT ARRAY[1.0, 2.0, 3.0]::vector(3);
```

## Distance Operators

| Operator | Description | Return Type | Stability | Example |
|----------|-------------|-------------|-----------|---------|
| `<->` | L2 (Euclidean) distance | `float` | **Stable** | `embedding <-> '[1,2,3]'` |
| `<=>` | Cosine distance | `float` | **Stable** | `embedding <=> '[1,2,3]'` |
| `<#>` | Inner product (negative dot product) | `float` | **Stable** | `embedding <#> '[1,2,3]'` |

## Embedding Functions

### `neurondb_embed(text, model)` {#stability-stable}

Generate embeddings using configured LLM providers.

**Stability:** **Stable**  
**Parameters:** text (TEXT), model (TEXT)  
**Returns:** vector

```sql
-- Configure OpenAI provider
SET neurondb.llm_provider = 'openai';
SET neurondb.llm_api_key = 'sk-...';

-- Generate embedding
SELECT neurondb_embed('Hello world', 'text-embedding-ada-002');

-- Batch embeddings
INSERT INTO documents (content, embedding)
SELECT content, neurondb_embed(content, 'text-embedding-ada-002')
FROM source_documents;
```

### `neurondb_embed_batch(texts, model)` {#stability-stable}

Generate embeddings for multiple texts in a single API call (more efficient).

**Stability:** **Stable**  
**Parameters:** texts (TEXT[]), model (TEXT)  
**Returns:** vector[]

```sql
-- Batch embed multiple texts
SELECT neurondb_embed_batch(
  ARRAY['text1', 'text2', 'text3'],
  'text-embedding-ada-002'
);
```

## GPU Distance Functions

GPU-accelerated distance computation functions. Require `neurondb.compute_mode = true`.

### `vector_l2_distance_gpu(a, b)` {#stability-experimental}

Compute L2 distance on GPU.

**Stability:** **Experimental**  
**Parameters:** a (vector), b (vector)  
**Returns:** float8

```sql
SET neurondb.compute_mode = true;
SELECT vector_l2_distance_gpu(embedding, '[1,2,3]'::vector) FROM documents;
```

### `vector_cosine_distance_gpu(a, b)` {#stability-experimental}

Compute cosine distance on GPU.

**Stability:** **Experimental**

### `vector_inner_product_gpu(a, b)` {#stability-experimental}

Compute inner product on GPU.

**Stability:** **Experimental**

### `vector_to_int8_gpu(v)`, `vector_to_fp16_gpu(v)`, `vector_to_binary_gpu(v)` {#stability-experimental}

Quantize vectors on GPU.

**Stability:** **Experimental**

## ML Analytics Functions

### `cluster_kmeans(data, k, max_iter, tol)` {#stability-stable}

K-Means clustering aggregate function.

**Stability:** **Stable**  
**Parameters:** data (vector), k (int), max_iter (int, default 100), tol (float8, default 0.0001)  
**Returns:** TABLE(cluster_id int, centroid vector, size bigint)

```sql
SELECT * FROM cluster_kmeans(
  (SELECT embedding FROM documents),
  5,  -- 5 clusters
  100,  -- max iterations
  0.0001  -- tolerance
);
```

### `cluster_minibatch_kmeans(data, k, batch_size, max_iter)` {#stability-stable}

Mini-batch K-Means for large datasets.

**Stability:** **Stable**  
**Parameters:** data (vector), k (int), batch_size (int, default 100), max_iter (int, default 100)  
**Returns:** TABLE(cluster_id int, centroid vector, size bigint)

### `cluster_gmm(data, k, max_iter, tol)` {#stability-experimental}

Gaussian Mixture Model clustering with soft assignments.

**Stability:** **Experimental**  
**Parameters:** data (vector), k (int), max_iter (int, default 100), tol (float8, default 0.0001)  
**Returns:** TABLE(cluster_id int, mean vector, covariance vector, weight float8)

```sql
-- Convert soft assignments to hard clusters with helper function
SELECT gmm_to_clusters(
  embedding,
  ARRAY(SELECT mean FROM cluster_gmm(...)),
  ARRAY(SELECT covariance FROM cluster_gmm(...)),
  ARRAY(SELECT weight FROM cluster_gmm(...))
) AS cluster_id
FROM documents;
```

### `detect_outliers_zscore(data, threshold)` {#stability-stable}

Detect outliers using Z-score method.

**Stability:** **Stable**  
**Parameters:** data (vector), threshold (float8, default 3.0)  
**Returns:** TABLE(vector_data vector, is_outlier boolean, z_score float8)

```sql
-- Detect outliers with Z-score > 3
SELECT 
  id,
  (stats).is_outlier,
  (stats).z_score
FROM (
  SELECT 
    id,
    detect_outliers_zscore(embedding, 3.0) AS stats
  FROM documents
) sub
WHERE (stats).is_outlier;
```

## ML Project Management

### `neurondb_create_ml_project(name, description)` {#stability-experimental}

Create a new ML project for model management.

**Stability:** **Experimental**

### `neurondb_train_kmeans_project(project_id, data, k, max_iter)` {#stability-experimental}

Train a K-Means model within a project.

**Stability:** **Experimental**

### `neurondb_list_project_models(project_id)` {#stability-experimental}

List all models in a project.

**Stability:** **Experimental**

### `neurondb_deploy_model(model_id, deployment_name)` {#stability-experimental}

Deploy a model for inference.

**Stability:** **Experimental**

### `neurondb_get_deployed_model(deployment_name)` {#stability-experimental}

Retrieve deployed model metadata.

**Stability:** **Experimental**

### `neurondb_get_project_info(project_id)` {#stability-experimental}

Get project information and statistics.

**Stability:** **Experimental**

## Configuration Parameters (GUCs)

### LLM Provider Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `neurondb.llm_provider` | string | 'openai' | LLM provider (openai, cohere, huggingface) |
| `neurondb.llm_api_key` | string | NULL | API key for LLM provider |
| `neurondb.llm_endpoint` | string | NULL | Custom API endpoint URL |
| `neurondb.llm_timeout_ms` | int | 30000 | API call timeout (milliseconds) |
| `neurondb.llm_max_retries` | int | 3 | Max retry attempts for failed API calls |

### GPU Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `neurondb.compute_mode` | bool | false | Enable GPU acceleration |
| `neurondb.gpu_device` | int | 0 | GPU device ID to use |
| `neurondb.gpu_batch_size` | int | 1000 | Batch size for GPU operations |
| `neurondb.gpu_streams` | int | 4 | Number of CUDA streams |
| `neurondb.gpu_memory_pool_mb` | int | 512 | GPU memory pool size (MB) |
| `neurondb.gpu_fail_open` | bool | true | Fallback to CPU on GPU error |
| `neurondb.gpu_kernels` | string | 'auto' | GPU kernel selection (auto, cuda, rocm) |
| `neurondb.gpu_backend` | string | 'cuda' | GPU backend (cuda, rocm) |
| `neurondb.gpu_timeout_ms` | int | 5000 | GPU operation timeout (ms) |

### Background Worker Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `neurondb.enable_background_workers` | bool | false | Enable background embedding workers |
| `neurondb.worker_batch_size` | int | 100 | Batch size for worker tasks |
| `neurondb.worker_interval_sec` | int | 60 | Worker polling interval (seconds) |
| `neurondb.worker_max_concurrent` | int | 4 | Max concurrent worker tasks |

### Monitoring and Metrics

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `neurondb.enable_metrics` | bool | true | Enable performance metrics collection |
| `neurondb.metrics_retention_days` | int | 7 | Metric retention period |
| `neurondb.log_level` | string | 'info' | Logging level (debug, info, warning, error) |

## Index Types and Operator Classes

### Index Types
- `hnsw` - Hierarchical Navigable Small World graph index
- `ivfflat` - Inverted File with Flat quantization

### Operator Classes
- `vector_l2_ops` - L2 (Euclidean) distance operations
- `vector_cosine_ops` - Cosine distance operations
- `vector_ip_ops` - Inner product operations

## API Contract and Policies

NeuronDB follows strict API contract policies to ensure stability and predictability:

- **[Function Stability Policy](function-stability.md)** - Classification of functions as stable, experimental, or internal
- **[Deprecation Policy](deprecation-policy.md)** - How functions are deprecated and removed
- **[API Snapshots](api-snapshots/readme.md)** - Versioned API reference snapshots for each release

All functions in this reference are classified according to their stability level. Functions marked as `stable` are safe for production use and maintain backward compatibility.

## Next Steps

- [Indexing and Distance Metrics](vector-search/indexing.md): Learn about index tuning and distance functions
- [GPU Acceleration](gpu/cuda-support.md): Configure and optimize GPU operations
- [ML Algorithms](ml-algorithms/clustering.md): Use clustering and outlier detection
- [Configuration Guide](configuration.md): Detailed GUC reference
- [Function Stability Policy](function-stability.md): Understand API stability classifications
- [Deprecation Policy](deprecation-policy.md): Learn how deprecations are handled

---

<div align="center">

[⬆ Back to Top](#sql-api-reference)

</div>
