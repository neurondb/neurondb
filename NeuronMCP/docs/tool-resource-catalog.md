# NeuronMCP Tool and Resource Catalog

Complete catalog of all tools and resources available through NeuronMCP.

## Tools Catalog

NeuronMCP provides 600+ tools organized into the following categories. This catalog shows the main tool categories and representative tools. For a complete list of all tools, see the tool registration code or use the `tools/list` MCP method.

**Tool Categories:**
- Vector Operations: 100+ tools
- ML Tools: 50+ tools  
- RAG Operations: 15+ tools
- PostgreSQL Tools: 100+ tools
- Debugging Tools: 5+ tools
- Composition Tools: 4+ tools
- Workflow Tools: 4+ tools
- Plugin Tools: 6+ tools
- Developer Experience Tools: 10+ tools
- Enterprise Tools: 20+ tools
- Monitoring & Analytics Tools: 10+ tools
- AI Intelligence Tools: 10+ tools
- ... and more categories

### Vector Operations (12+ tools)

| Tool | Description | Parameters |
|------|-------------|------------|
| `vector_search` | Vector similarity search with configurable distance metrics | table, vector_column, query_vector, limit, distance_metric, additional_columns |
| `vector_search_l2` | L2 (Euclidean) distance search | table, vector_column, query_vector, limit |
| `vector_search_cosine` | Cosine similarity search | table, vector_column, query_vector, limit |
| `vector_search_inner_product` | Inner product search | table, vector_column, query_vector, limit |
| `vector_search_l1` | L1 (Manhattan) distance search | table, vector_column, query_vector, limit |
| `vector_search_hamming` | Hamming distance search | table, vector_column, query_vector, limit |
| `vector_search_chebyshev` | Chebyshev distance search | table, vector_column, query_vector, limit |
| `vector_search_minkowski` | Minkowski distance search | table, vector_column, query_vector, limit, p_value |
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

### RAG Operations (15+ tools)

| Tool | Description | Parameters |
|------|-------------|------------|
| `process_document` | Process document for RAG | document, chunk_size, overlap |
| `retrieve_context` | Retrieve context for query | query, table, limit, rerank |
| `generate_response` | Generate RAG response | query, context, model |
| `chunk_document` | Chunk document | document, strategy, size |
| `ingest_documents` | Ingest multiple documents | documents[], pipeline_config |
| `answer_with_citations` | Generate answer with citations | query, context, model |
| `rag_evaluate` | Evaluate RAG pipeline | pipeline_id, test_queries[] |
| `rag_chat` | RAG chat interface | query, session_id, pipeline_id |
| `rag_hybrid` | Hybrid RAG search | query, vector_weight, text_weight |
| `rag_rerank` | RAG with reranking | query, reranker, top_k |
| `rag_hyde` | Hypothetical Document Embeddings | query, generate_count |
| `rag_graph` | Graph-based RAG | query, graph_config |
| `rag_corrective` | Corrective RAG | query, feedback_loop |
| `rag_agentic` | Agentic RAG | query, agent_config |
| `rag_contextual` | Contextual RAG | query, context_window |
| `rag_modular` | Modular RAG pipeline | query, module_config |

### Workers & GPU (2+ tools)

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

### PostgreSQL (100+ tools)

**Note:** This is a comprehensive category with tools for complete database control. Categories include:

- **Server Information** (8 tools): version, stats, databases, connections, locks, replication, settings, extensions
- **Database Object Management** (8 tools): tables, indexes, schemas, views, sequences, functions, triggers, constraints
- **User and Role Management** (3 tools): users, roles, permissions
- **Performance and Statistics** (4 tools): table stats, index stats, active queries, wait events
- **Size and Storage** (4 tools): table size, index size, bloat, vacuum stats
- **Administration** (16 tools): explain, vacuum, analyze, reindex, transactions, query management, config, partitions, FDW
- **Query Execution & Management** (6 tools): execute query, query plan, cancel query, kill query, query history, optimization
- **Database & Schema Management** (6 tools): create/alter/drop database, create/alter/drop schema
- **User & Role Management** (6 tools): create/alter/drop user, create/alter/drop role
- **Permission Management** (4 tools): grant, revoke, grant role, revoke role
- **Backup & Recovery** (6 tools): backup database, restore database, backup table, list backups, verify backup, schedule backup
- **Schema Modification** (7 tools): create/alter/drop table, create index, create view, create function, create trigger
- **Object Management** (17 tools): alter/drop for indexes, views, functions, triggers, sequences, types, domains
- **Data Manipulation** (5 tools): INSERT, UPDATE, DELETE, TRUNCATE, COPY
- **Advanced DDL** (10 tools): materialized views, partitions, foreign tables
- **High Availability** (5 tools): replication lag, promote replica, sync status, cluster, failover
- **Security** (7 tools): audit log, security scan, compliance check, encryption status, SQL validation, permission checking, audit operations
- **Maintenance** (1 tool): maintenance windows

**Representative Tools:**

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
| `postgresql_execute_query` | Execute SQL query | query, params |
| `postgresql_query_plan` | Get query execution plan | query |
| `postgresql_cancel_query` | Cancel running query | pid |
| `postgresql_create_table` | Create table | table_name, columns, constraints |
| `postgresql_alter_table` | Alter table | table_name, changes |
| `postgresql_drop_table` | Drop table | table_name |
| `postgresql_create_index` | Create index | table_name, index_name, columns, index_type |
| `postgresql_backup_database` | Backup database | database_name, backup_path |
| `postgresql_restore_database` | Restore database | database_name, backup_path |
| `postgresql_grant` | Grant permissions | role, object, privileges |
| `postgresql_revoke` | Revoke permissions | role, object, privileges |
| `postgresql_create_user` | Create user | username, password, options |
| `postgresql_alter_user` | Alter user | username, changes |
| `postgresql_drop_user` | Drop user | username |
| `postgresql_vacuum` | Run VACUUM | table_name, options |
| `postgresql_analyze` | Run ANALYZE | table_name |
| `postgresql_explain` | Explain query | query |
| `postgresql_explain_analyze` | Explain and analyze query | query |
| ... and 70+ more PostgreSQL tools |

### Debugging Tools (5+ tools)

| Tool | Description | Parameters |
|------|-------------|------------|
| `debug_tool_call` | Debug tool call execution | tool_name, arguments, trace_level |
| `debug_query_plan` | Analyze and debug query plans | query, explain_options |
| `monitor_active_connections` | Monitor active database connections | filters |
| `monitor_query_performance` | Monitor query performance metrics | time_window, filters |
| `trace_request` | Trace MCP request execution | request_id, trace_level |

### Composition Tools (4+ tools)

| Tool | Description | Parameters |
|------|-------------|------------|
| `tool_chain` | Chain multiple tools in sequence | tools[], inputs[] |
| `tool_parallel` | Execute tools in parallel | tools[], inputs[] |
| `tool_conditional` | Conditional tool execution | condition, true_tool, false_tool, inputs |
| `tool_retry` | Retry tool execution with backoff | tool_name, max_retries, backoff_strategy, inputs |

### Workflow Tools (4+ tools)

| Tool | Description | Parameters |
|------|-------------|------------|
| `create_workflow` | Create a new workflow | name, steps[], config |
| `execute_workflow` | Execute a workflow | workflow_id, inputs |
| `workflow_status` | Get workflow execution status | workflow_id, execution_id |
| `list_workflows` | List all workflows | filters |

### Plugin Tools (6+ tools)

| Tool | Description | Parameters |
|------|-------------|------------|
| `plugin_marketplace` | Browse plugin marketplace | category, filters |
| `plugin_hot_reload` | Hot reload plugin | plugin_id |
| `plugin_versioning` | Manage plugin versions | plugin_id, version, operation |
| `plugin_sandbox` | Test plugin in sandbox | plugin_code, test_inputs |
| `plugin_testing` | Run plugin tests | plugin_id, test_suite |
| `plugin_builder` | Build custom plugin | plugin_config, code |

### Developer Experience Tools (10+ tools)

| Tool | Description | Parameters |
|------|-------------|------------|
| `nl_to_sql` | Convert natural language to SQL | query, schema_info |
| `sql_to_nl` | Convert SQL to natural language | sql_query |
| `query_builder` | Build SQL queries interactively | schema, conditions, operations |
| `code_generator` | Generate code from specifications | language, spec, templates |
| `test_data_generator` | Generate test data | schema, count, constraints |
| `schema_visualizer` | Visualize database schema | schema_name, format |
| `query_explainer` | Explain query in natural language | query |
| `schema_documentation` | Generate schema documentation | schema_name, format |
| `migration_generator` | Generate migration scripts | source_schema, target_schema |
| `sdk_generator` | Generate SDK code | language, api_spec |

### Enterprise Tools (20+ tools)

| Tool | Description | Parameters |
|------|-------------|------------|
| `multi_tenant_management` | Manage multi-tenant configurations | tenant_id, operation, config |
| `data_governance` | Data governance policies | policy_type, rules |
| `data_lineage` | Track data lineage | table_name, depth |
| `compliance_reporter` | Generate compliance reports | report_type, filters |
| `audit_analyzer` | Analyze audit logs | time_range, filters |
| `backup_automation` | Automate backup operations | schedule, retention |
| `query_result_cache` | Manage query result cache | operation, cache_key |
| `cache_optimizer` | Optimize cache settings | analysis_type |
| `performance_benchmark` | Run performance benchmarks | benchmark_type, config |
| `auto_scaling_advisor` | Get auto-scaling recommendations | metrics, thresholds |
| `slow_query_analyzer` | Analyze slow queries | time_window, threshold |
| `real_time_dashboard` | Create real-time dashboards | metrics[], refresh_rate |
| `anomaly_detection` | Detect anomalies | metric, time_window, threshold |
| `predictive_analytics` | Predictive analytics | model_type, data, horizon |
| `cost_forecasting` | Forecast costs | time_horizon, factors |
| `usage_analytics` | Usage analytics | time_range, dimensions |
| `alert_manager` | Manage alerts | alert_config, rules |

### AI Intelligence Tools (10+ tools)

| Tool | Description | Parameters |
|------|-------------|------------|
| `ai_model_orchestration` | Orchestrate AI models | models[], pipeline_config |
| `ai_cost_tracking` | Track AI costs | time_range, model_filters |
| `ai_embedding_quality` | Assess embedding quality | embeddings[], metrics |
| `ai_model_comparison` | Compare AI models | models[], test_data |
| `ai_rag_evaluation` | Evaluate RAG systems | pipeline_id, test_queries |
| `ai_embedding_drift_detection` | Detect embedding drift | baseline, current, threshold |
| `ai_model_finetuning` | Fine-tune AI models | base_model, training_data, config |
| `ai_prompt_versioning` | Version control prompts | prompt_id, version, changes |
| `ai_token_optimization` | Optimize token usage | text, model, target_reduction |
| `ai_multi_model_ensemble` | Create model ensembles | models[], weights[], strategy |

### PostgreSQL Advanced Tools (10+ tools)

| Tool | Description | Parameters |
|------|-------------|------------|
| `postgresql_query_optimizer` | Optimize SQL queries | query, options |
| `postgresql_performance_insights` | Get performance insights | time_range, metrics |
| `postgresql_index_advisor` | Get index recommendations | query, workload |
| `postgresql_query_plan_analyzer` | Analyze query plans | query, options |
| `postgresql_schema_evolution` | Manage schema evolution | source_schema, target_schema |
| `postgresql_migration` | Database migration tools | source, target, options |
| `postgresql_connection_pool_optimizer` | Optimize connection pools | current_config, workload |
| `postgresql_vacuum_analyzer` | Analyze vacuum needs | table_name, options |
| `postgresql_replication_lag_monitor` | Monitor replication lag | replica_name |
| `postgresql_wait_event_analyzer` | Analyze wait events | time_range, filters |

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

