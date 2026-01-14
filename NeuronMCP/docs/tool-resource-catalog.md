# NeuronMCP Tool and Resource Catalog

Complete catalog of all tools and resources available through NeuronMCP.

## Tools Catalog

NeuronMCP provides 74+ tools organized into the following categories:

### Vector Operations (8 tools)

| Tool | Description | Parameters |
|------|-------------|------------|
| `vector_search` | Vector similarity search with configurable distance metrics | table, vector_column, query_vector, limit, distance_metric, additional_columns |
| `vector_search_l2` | L2 (Euclidean) distance search | table, vector_column, query_vector, limit |
| `vector_search_cosine` | Cosine similarity search | table, vector_column, query_vector, limit |
| `vector_search_inner_product` | Inner product search | table, vector_column, query_vector, limit |
| `vector_similarity` | Calculate vector similarity | vector1, vector2, metric |
| `vector_arithmetic` | Vector arithmetic operations | operation, vector1, vector2, scalar |
| `vector_distance` | Compute distance between vectors | vector1, vector2, metric, p_value, covariance |
| `vector_similarity_unified` | Unified vector similarity with multiple metrics | vector1, vector2, metrics |

### Vector Quantization (7 tools)

| Tool | Description | Parameters |
|------|-------------|------------|
| `vector_quantize` | Quantize/dequantize vectors | operation, vector, data |
| `quantization_analyze` | Analyze quantization impact | table, vector_column, operation |

**Supported Quantization Types:**
- int8 (8-bit integer)
- fp16 (16-bit floating point)
- binary (1-bit)
- uint8 (unsigned 8-bit integer)
- ternary (2-bit)
- int4 (4-bit integer)

### Embeddings (8 tools)

| Tool | Description | Parameters |
|------|-------------|------------|
| `generate_embedding` | Generate text embedding | text, model |
| `batch_embedding` | Batch generate embeddings | texts[], model |
| `embed_image` | Generate image embedding | image_data (base64), model |
| `embed_multimodal` | Multimodal embedding (text + image) | text, image_data, model |
| `embed_cached` | Use cached embedding if available | text, model |
| `configure_embedding_model` | Configure embedding model | model_name, config_json |
| `get_embedding_model_config` | Get model configuration | model_name |
| `list_embedding_model_configs` | List all model configurations | - |
| `delete_embedding_model_config` | Delete model configuration | model_name |

### Hybrid Search (7 tools)

| Tool | Description | Parameters |
|------|-------------|------------|
| `hybrid_search` | Semantic + lexical search | table, query_vector, query_text, vector_column, text_column, vector_weight, limit, filters |
| `reciprocal_rank_fusion` | RRF on multiple rankings | rankings[], k |
| `semantic_keyword_search` | Semantic + keyword search | table, semantic_query, keyword_query, top_k |
| `multi_vector_search` | Multiple embeddings per document | table, query_vectors[], weights[], limit |
| `faceted_vector_search` | Category-aware retrieval | table, query_vector, facets, limit |
| `temporal_vector_search` | Time-decay relevance scoring | table, query_vector, time_column, decay_factor, limit |
| `diverse_vector_search` | Diverse result set | table, query_vector, diversity_factor, limit |

### Reranking (6 tools)

| Tool | Description | Parameters |
|------|-------------|------------|
| `rerank_cross_encoder` | Cross-encoder reranking | query, documents[], model, top_k |
| `rerank_llm` | LLM-powered reranking | query, documents[], model, top_k |
| `rerank_cohere` | Cohere reranking API | query, documents[], top_k |
| `rerank_colbert` | ColBERT reranking | query, documents[], top_k |
| `rerank_ltr` | Learning-to-rank reranking | query, documents[], features, top_k |
| `rerank_ensemble` | Ensemble reranking | query, documents[], methods[], weights[], top_k |

### Machine Learning (8 tools)

| Tool | Description | Parameters |
|------|-------------|------------|
| `train_model` | Train ML model | algorithm, table, feature_col, label_col, params, project |
| `predict` | Single prediction | model_id, features |
| `predict_batch` | Batch prediction | model_id, features[] |
| `evaluate_model` | Evaluate model | model_id, table, feature_col, label_col |
| `list_models` | List all models | project, algorithm |
| `get_model_info` | Get model details | model_id |
| `delete_model` | Delete model | model_id |
| `export_model` | Export model | model_id, format |

**Supported Algorithms:**
- Classification: logistic, random_forest, svm, knn, decision_tree, naive_bayes
- Regression: linear_regression, ridge, lasso
- Clustering: kmeans, gmm, dbscan, hierarchical

### Analytics (7 tools)

| Tool | Description | Parameters |
|------|-------------|------------|
| `analyze_data` | General data analysis | table, columns |
| `cluster_data` | Clustering analysis | algorithm, table, vector_column, k, eps |
| `reduce_dimensionality` | Dimensionality reduction (PCA) | table, vector_column, dimensions |
| `detect_outliers` | Outlier detection | method, table, vector_column, threshold |
| `quality_metrics` | Quality metrics (Recall@K, Precision@K, etc.) | metric, table, k, ground_truth_col, predicted_col |
| `detect_drift` | Data drift detection | method, table, vector_column, reference_table, threshold |
| `topic_discovery` | Topic modeling | table, text_column, num_topics |

### Time Series (1 tool)

| Tool | Description | Parameters |
|------|-------------|------------|
| `timeseries_analysis` | Time series analysis | table, time_column, value_column, method, params |

**Methods:** ARIMA, forecasting, seasonal_decomposition

### AutoML (1 tool)

| Tool | Description | Parameters |
|------|-------------|------------|
| `automl` | Automated ML pipeline | task_type, table, feature_col, label_col, constraints |

### ONNX (4 tools)

| Tool | Description | Parameters |
|------|-------------|------------|
| `onnx_model` | ONNX model operations | operation, model_path, input_data |

**Operations:** import, export, info, predict

### Index Management (6 tools)

| Tool | Description | Parameters |
|------|-------------|------------|
| `create_hnsw_index` | Create HNSW index | table, vector_column, index_name, m, ef_construction |
| `create_ivf_index` | Create IVF index | table, vector_column, index_name, num_lists, probes |
| `index_status` | Get index status | table, index_name |
| `drop_index` | Drop index | table, index_name |
| `tune_hnsw_index` | Auto-tune HNSW parameters | table, vector_column |
| `tune_ivf_index` | Auto-tune IVF parameters | table, vector_column |

### RAG Operations (4 tools)

| Tool | Description | Parameters |
|------|-------------|------------|
| `process_document` | Process document for RAG | document, chunk_size, overlap |
| `retrieve_context` | Retrieve context for query | query, table, limit, rerank |
| `generate_response` | Generate RAG response | query, context, model |
| `chunk_document` | Chunk document | document, strategy, size |

### Workers & GPU (2 tools)

| Tool | Description | Parameters |
|------|-------------|------------|
| `worker_management` | Manage background workers | operation, worker_type |
| `gpu_info` | Get GPU information | - |

### Vector Graph (1 tool)

| Tool | Description | Parameters |
|------|-------------|------------|
| `vector_graph` | Graph operations on vgraph | operation, graph, start_node, max_depth, damping_factor |

**Operations:** bfs, dfs, pagerank, community_detection

### Vecmap Operations (1 tool)

| Tool | Description | Parameters |
|------|-------------|------------|
| `vecmap_operations` | Sparse vector operations | operation, vecmap1, vecmap2, scalar |

**Operations:** l2_distance, cosine_distance, inner_product, l1_distance, add, subtract, multiply_scalar, norm

### Dataset Loading (1 tool)

| Tool | Description | Parameters |
|------|-------------|------------|
| `load_dataset` | Load datasets from various sources | source_type, source_path, format, auto_embed, create_indexes |

**Source Types:** huggingface, url, github, s3, local

### PostgreSQL (8 tools)

| Tool | Description | Parameters |
|------|-------------|------------|
| `postgresql_version` | Get PostgreSQL version | - |
| `postgresql_stats` | Get server statistics | include_database_stats, include_table_stats, include_connection_stats |
| `postgresql_databases` | List databases | - |
| `postgresql_connections` | Get connection info | - |
| `postgresql_locks` | Get lock information | - |
| `postgresql_replication` | Get replication status | - |
| `postgresql_settings` | Get configuration settings | - |
| `postgresql_extensions` | List extensions | - |

## Resources Catalog

NeuronMCP provides the following resources:

### Schema Resources

| Resource URI | Description | MIME Type |
|--------------|-------------|-----------|
| `neurondb://schema/tables` | List all tables with vector columns | `application/json` |
| `neurondb://schema/table/{table_name}` | Table schema details | `application/json` |
| `neurondb://schema/columns/{table_name}` | Column definitions for a table | `application/json` |
| `neurondb://schema/indexes` | List all indexes | `application/json` |
| `neurondb://schema/index/{index_name}` | Index details | `application/json` |

### Model Resources

| Resource URI | Description | MIME Type |
|--------------|-------------|-----------|
| `neurondb://models` | List all trained models | `application/json` |
| `neurondb://model/{model_id}` | Model metadata and information | `application/json` |
| `neurondb://model/{model_id}/metrics` | Model evaluation metrics | `application/json` |
| `neurondb://model/{model_id}/predictions` | Model prediction history | `application/json` |

### Index Resources

| Resource URI | Description | MIME Type |
|--------------|-------------|-----------|
| `neurondb://indexes` | List all vector indexes | `application/json` |
| `neurondb://index/{index_name}/stats` | Index statistics | `application/json` |
| `neurondb://index/{index_name}/status` | Index build status | `application/json` |

### Configuration Resources

| Resource URI | Description | MIME Type |
|--------------|-------------|-----------|
| `neurondb://config` | Current NeuronDB configuration | `application/json` |
| `neurondb://config/gpu` | GPU configuration | `application/json` |
| `neurondb://config/llm` | LLM provider configuration | `application/json` |

### Worker Resources

| Resource URI | Description | MIME Type |
|--------------|-------------|-----------|
| `neurondb://workers` | List all background workers | `application/json` |
| `neurondb://worker/{worker_name}/status` | Worker status | `application/json` |
| `neurondb://worker/{worker_name}/queue` | Worker queue status | `application/json` |

### Statistics Resources

| Resource URI | Description | MIME Type |
|--------------|-------------|-----------|
| `neurondb://stats/overview` | Overview statistics | `application/json` |
| `neurondb://stats/performance` | Performance metrics | `application/json` |
| `neurondb://stats/usage` | Usage statistics | `application/json` |

## Tool Discovery

To discover available tools programmatically:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}
```

Response:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [
      {
        "name": "vector_search",
        "description": "Perform vector similarity search",
        "inputSchema": {
          "type": "object",
          "properties": {
            "table": {"type": "string"},
            "vector_column": {"type": "string"},
            "query_vector": {"type": "array", "items": {"type": "number"}},
            "limit": {"type": "integer", "default": 10}
          },
          "required": ["table", "vector_column", "query_vector"]
        }
      }
    ]
  }
}
```

## Resource Discovery

To discover available resources:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "resources/list",
  "params": {}
}
```

Response:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "resources": [
      {
        "uri": "neurondb://schema/tables",
        "name": "Tables",
        "description": "List all tables with vector columns",
        "mimeType": "application/json"
      }
    ]
  }
}
```

## Related Documentation

- [Tools Reference](../REGISTERED_TOOLS.md) - Detailed tool documentation with examples
- [POSTGRESQL_TOOLS.md](../POSTGRESQL_TOOLS.md) - PostgreSQL-specific tools
- [Examples](./examples/) - Example client usage and transcripts
- [README](../README.md) - NeuronMCP overview and setup

