# NeuronDB Features

Capability matrix and feature reference for the NeuronDB PostgreSQL extension.

| Status | Meaning |
|--------|--------|
| **Supported** | Feature is implemented and supported. |
| **Partial** | Feature is implemented with known limitations (see section). |
| **Not supported** | Feature is not available. |

---

## Scope

NeuronDB is a PostgreSQL extension. This document lists its capabilities and support level by area.

---

## Summary

| Area | Capability | Status |
|------|------------|--------|
| Vector types | `vector`, `vectorp`, `vecmap`, `vgraph`, `rtext`, `halfvec`, `binaryvec`, `sparsevec`; in some builds also `sparse_vector` | Supported |
| Index access methods | HNSW, IVF (two access methods for approximate nearest neighbor search) | Supported |
| Distance metrics | L2, cosine, inner product, L1/Manhattan, Hamming, Chebyshev, Minkowski, Jaccard, and others (20+ distance/similarity functions) | Supported |
| SQL functions and operators | ~650+ (CREATE FUNCTION/CREATE OPERATOR in extension SQL; exact count depends on version) | Supported |
| Embeddings | `embed_text`, `embed_text_batch`, `embed_image`, `embed_multimodal`, `embed_cached`; GPU acceleration when available | Supported |
| RAG | `neurondb.rag_query`, RAG pipeline tables and audit; retrieval + generation | Supported |
| Hybrid search | `hybrid_search`, `hybrid_search_fusion`, `reciprocal_rank_fusion` (vector + full-text) | Supported |
| Reranking | Cross-encoder, LLM, Cohere, ColBERT, LTR, ensemble, MMR, Borda, batch LLM; cache helpers | Supported |
| Vector aggregates | `vector_avg`, `vector_sum` (and batch variants) | Supported |
| Quantization | PQ codebook training (`train_pq_codebook`), OPQ rotation (`train_opq_rotation`); int8, fp16, binary, uint8, ternary conversions | Supported |
| ML algorithms | 25+ algorithm families: linear/ridge/lasso/elastic-net regression, logistic regression, KNN, random forest, decision tree, naive Bayes, SVM, K-Means, GMM, XGBoost, LightGBM, CatBoost, ARIMA, time series, collaborative filtering; train/predict/evaluate functions | Supported |
| Background workers | neuranq (job queue), neuranmon (tuner sampling), neurandefrag (index defrag), neuranllm (LLM jobs) — functions and tables present | Supported |
| GPU backends | CUDA, ROCm, Metal (distance, search, batch ops; see GPU matrix for limits) | Supported / Partial |
| Index build on GPU | HNSW/IVF index construction | Not supported (CPU only) |

---

## 1. Vector types (detailed)

**Eight main types** with full type I/O, storage, and where applicable distance support:

| Type | Description | Storage per dimension | Max dimensions | Index support (HNSW/IVF) |
|------|-------------|------------------------|----------------|---------------------------|
| `vector` | Float32 vector; primary type for embeddings and search | 4 bytes | 16,000 | Yes (vector_l2_ops, vector_cosine_ops, vector_ip_ops) |
| `vectorp` | Packed SIMD vector with fingerprint and version metadata | 4 bytes | same as vector | — |
| `vecmap` | Sparse map: only non-zero dimensions; for very high dimensions (>10K) | variable | >10K | — |
| `vgraph` | Graph-based structure for neighbor relations, BFS, PageRank, community detection | variable | — | — |
| `rtext` | Retrievable text with token offsets and section IDs for RAG | variable | — | — |
| `halfvec` | Float16; 2x compression | 2 bytes | 4,000 | Yes (halfvec_l2_ops, halfvec_cosine_ops, halfvec_ip_ops) |
| `binaryvec` | 1 bit per dimension; Hamming distance | 1/8 byte | variable | Yes (binaryvec_hamming_ops) |
| `sparsevec` | Sparse vector (token_ids + weights); model_type 0=BM25, 1=SPLADE, 2=ColBERTv2; max nnz 1,000, max dim 1e6 | variable | 1,000,000 | Yes (sparsevec_l2_ops, sparsevec_cosine_ops, sparsevec_ip_ops) |

**Casting and conversion:** `vector` ↔ `halfvec` (implicit/assignment); `vector_to_sparsevec` / `sparsevec_to_vector`; `vector_to_binary`; `vector_to_int8`, `vector_to_fp16`, `vector_to_uint8`, `vector_to_ternary`, `vector_to_int4`; `array_to_vector`, `vector_to_array`, `vector_cast_dimension`.

**Bit type:** `bit` has operator classes `bit_hamming_ops`, `bit_jaccard_ops` for Hamming/Jaccard on bit vectors.

---

## 2. Index access methods and operator classes

**Index access methods (2 only):**

- **`hnsw`** — Hierarchical Navigable Small World. Parameters: `m` (links, default 16, range 2–128), `ef_construction` (default 200, range 4–2000). Query tuning: `neurondb.hnsw_ef_search`, `neurondb.hnsw_k`.
- **`ivfflat`** — Inverted file with flat storage. Parameter: `lists` (number of clusters, default 100, range 1–1000). Query tuning: `neurondb.ivf_probes`.

**Operator classes (by type):**

- **vector:** `vector_l2_ops`, `vector_cosine_ops`, `vector_ip_ops` (for `<->`, `<=>`, `<#>`).
- **halfvec:** `halfvec_l2_ops`, `halfvec_cosine_ops`, `halfvec_ip_ops`.
- **sparsevec:** `sparsevec_l2_ops`, `sparsevec_cosine_ops`, `sparsevec_ip_ops`.
- **binaryvec:** `binaryvec_hamming_ops`.
- **bit:** `bit_hamming_ops`, `bit_jaccard_ops`.

**Not index types:** "Hybrid" and "temporal" are not separate access methods. Hybrid search is implemented as SQL functions (`hybrid_search`, `hybrid_search_fusion`) that combine vector and full-text search. Temporal-aware search is a query-level feature (e.g. time decay in application logic), not a third index AM.

---

## 3. Distance and similarity (exhaustive)

**Vector distance operators:**

| Operator | Name | Types | Description |
|----------|------|--------|-------------|
| `<->` | L2 (Euclidean) | vector, halfvec, sparsevec | Euclidean distance |
| `<=>` | Cosine distance | vector, halfvec, sparsevec | 1 - cosine_similarity |
| `<#>` | Inner product (negative for ordering) | vector, halfvec, sparsevec | Negative dot product for ANN ordering |
| `<+>` | L1 (Manhattan) | vector | Taxicab distance |
| `<~>` | Hamming | vector, binaryvec, bit | Hamming distance |
| `<*~*>` | Jaccard | vector | Jaccard distance |
| `<%>` | (binaryvec) | binaryvec | Binary inner product |

**Vector distance/similarity functions (vector type):**

- **L2 / Euclidean:** `vector_l2_distance`, `vector_squared_l2_distance`
- **Cosine:** `vector_cosine_distance`, `vector_cosine_similarity`, `vector_cosine_sim`, `vector_spherical_distance`
- **Inner product:** `vector_inner_product`, `vector_dot`
- **L1:** `vector_l1_distance`
- **Hamming:** `vector_hamming_distance`
- **Chebyshev:** `vector_chebyshev_distance`
- **Minkowski:** `vector_minkowski_distance(vector, vector, p)`
- **Jaccard / Dice:** `vector_jaccard_distance`, `vector_dice_distance`
- **Mahalanobis:** `vector_mahalanobis_distance(vector, vector, vector)` (covariance vector)
- **Batch:** `vector_l2_distance_batch`, `vector_cosine_distance_batch`, `vector_inner_product_batch`

**Halfvec:** `halfvec_l2_distance`, `halfvec_cosine_distance`, `halfvec_inner_product`; norms: `halfvec_l2_norm`, `halfvec_l2_normalize`.

**Sparsevec:** `sparsevec_l2_distance`, `sparsevec_cosine_distance`, `sparsevec_inner_product`; `sparsevec_l2_norm`, `sparsevec_l2_normalize`.

**Binaryvec:** `binaryvec_hamming_distance`.

**Norms and utilities:** `vector_dims`, `vector_norm`, `vector_normalize`, `vector_normalize_batch`; `vector_concat`, `vector_add`, `vector_sub`, `vector_mul`, `vector_div`, `vector_neg`; `vector_get`, `vector_set`, `vector_scale`, `vector_translate`, `vector_filter`, `vector_where`; `vector_cross_product`, `vector_percentile`, `vector_median`, `vector_quantile`; `subvector(halfvec, int, int)`.

---

## 4. Embedding functions

- **`embed_text(text, model DEFAULT NULL)`** — Single text embedding; model from GUC if omitted.
- **`embed_text_batch(text[], model DEFAULT NULL)`** — Batch text embeddings; GPU-accelerated when available.
- **`embed_image(bytea, model DEFAULT 'clip')`** — Image embedding (e.g. CLIP).
- **`embed_multimodal(text, bytea, model DEFAULT 'clip')`** — Text + image combined embedding.
- **`embed_cached(text, model DEFAULT 'all-MiniLM-L6-v2')`** — Cached text embedding.

**Unified / LLM layer:** `neurondb.embed(model, text, task)`; `ndb_llm_embed`; `neurondb_embed`, `neurondb_embed_batch` (legacy names). Configuration: `neurondb.llm_provider`, `neurondb.llm_model`, `neurondb.llm_endpoint`, `neurondb.llm_api_key`, `neurondb.llm_timeout_ms`; when `neurondb.compute_mode` is true, GPU is used when available.

---

## 5. RAG

- **`neurondb.rag_query(query, doc_table, vector_col, text_col, model, top_k)`** — Unified RAG: retrieve by vector + optional text, then generate.
- **Tables:** `neurondb.rag_pipelines` (pipeline configurations), `neurondb.rag_operation_audit_log` (audit log for retrieve/generate/chat).
- **Retrieval and generation** are implemented; document chunking and ingestion are typically done in application code or via extension helpers.

---

## 6. Hybrid search and fusion

- **`hybrid_search(table_name, query_vector, text_query, options DEFAULT '{}', vector_weight DEFAULT 0.7, limit DEFAULT 10, text_search_type DEFAULT 'plain')`** — Single-table hybrid: vector similarity + full-text search; combined ranking.
- **`reciprocal_rank_fusion(anyarray, k DEFAULT 60.0)`** — RRF over an array of ranked result arrays.
- **`hybrid_search_fusion(integer[], float8[], float8[], float8 DEFAULT 0.5, boolean DEFAULT true)`** — Fuse multiple rank arrays (e.g. from vector and text) with weights.
- **`mean_reciprocal_rank`** — MRR helper where applicable.

Hybrid search is **not** a third index type; it uses existing HNSW/IVF plus PostgreSQL full-text (GIN etc.) and fuses in SQL.

---

## 7. Reranking

- **Cross-encoder / local:** `rerank_cross_encoder(query, candidates[], model DEFAULT 'ms-marco-MiniLM-L-6-v2', top_k DEFAULT 10)`
- **LLM:** `rerank_llm(query, candidates[], model DEFAULT 'gpt-3.5-turbo', top_k DEFAULT 10)`; batch: `ndb_llm_rerank`, `ndb_llm_rerank_batch`
- **Cohere:** `rerank_cohere(query, candidates[], top_k DEFAULT 10)`
- **ColBERT:** `rerank_colbert(query, candidates[], model DEFAULT 'colbert-v2')`
- **LTR:** `rerank_ltr(query, candidates[], model, features)`
- **Ensemble:** `rerank_ensemble(query, candidates[], models[], weights[])`; `rerank_ensemble_weighted(integer[], float8[][], float8[], boolean)`; `rerank_ensemble_borda(integer[][])`
- **Long context / flash:** `rerank_flash`, `rerank_long_context` (when present in version)
- **Cache helpers:** `rerank_index_create`, `rerank_get_candidates`, `rerank_index_warm` — for building and warming rerank candidate indexes.

MMR and Borda-style fusion are available via ensemble/weighted helpers.

---

## 8. Vector aggregates and batch operations

- **Aggregates:** `vector_avg`, `vector_sum` (and batch variants: `vector_avg_batch`, `vector_sum_batch`).
- **Batch distance:** `vector_l2_distance_batch`, `vector_cosine_distance_batch`, `vector_inner_product_batch`.
- **Batch normalize:** `vector_normalize_batch`.

---

## 9. Quantization

- **PQ (Product Quantization):** `train_pq_codebook(table_name, column_name, sub_dim, n_centroids)` — train codebook; then `vector_quantize_int8(vector, min, max)`, `vector_dequantize_int8(bytea, min, max)` for int8 PQ.
- **OPQ (Orthogonal Procrustes):** `train_opq_rotation(table_name, column_name, n_subspace DEFAULT 8)` — train rotation for OPQ.
- **Conversions:** `vector_to_int8`, `vector_to_fp16`, `vector_to_binary`, `vector_to_bit`, `vector_to_halfvec`, `vector_to_sparsevec`, `vector_to_uint8`, `vector_to_ternary`, `vector_to_int4`; `vector_quantize_fp16`, `vector_dequantize_fp16`; `vector_quantize_binary`; FP16 distance: `vector_l2_distance_fp16`, `vector_cosine_distance_fp16`.

---

## 10. ML algorithms (exhaustive by family)

Each family has train/predict/evaluate (or equivalent) where applicable. Models are stored in the catalog and referenced by `model_id`.

**Regression:**

- **Linear:** `train_linear_regression`, `predict_linear_regression`, `predict_linear_regression_model_id`, `evaluate_linear_regression_by_model_id`
- **Ridge:** `train_ridge_regression`, `predict_ridge_regression_model_id`, `evaluate_ridge_regression_by_model_id`
- **Lasso:** `train_lasso_regression`, `predict_lasso_regression_model_id`, `evaluate_lasso_regression_by_model_id`

**Classification:**

- **Logistic:** `train_logistic_regression`, `predict_logistic_regression`, `predict_logistic_regression_model_id`, `evaluate_logistic_regression_by_model_id`
- **KNN:** `train_knn_model_id`, `predict_knn_model_id`, `predict_knn`, `evaluate_knn_by_model_id`
- **Random forest:** `train_random_forest_classifier`, `predict_random_forest`, `evaluate_random_forest`, `evaluate_random_forest_by_model_id`
- **Decision tree:** `train_decision_tree_classifier`, `predict_decision_tree_model_id`, `evaluate_decision_tree_by_model_id`
- **Naive Bayes:** `train_naive_bayes_classifier`, `train_naive_bayes_classifier_model_id`, `predict_naive_bayes`, `predict_naive_bayes_model_id`, `evaluate_naive_bayes_by_model_id`
- **SVM:** `train_svm_classifier`, `predict_svm_model_id`, `evaluate_svm_by_model_id`
- **XGBoost:** `train_xgboost_classifier`, `train_xgboost_regressor`, `predict_xgboost`, `evaluate_xgboost_by_model_id`
- **LightGBM:** `train_lightgbm_classifier`, `train_lightgbm_regressor`, `predict_lightgbm`, `evaluate_lightgbm_by_model_id`
- **CatBoost:** `train_catboost_classifier`, `train_catboost_regressor`, `predict_catboost`, `evaluate_catboost_by_model_id`

**Clustering:**

- **K-Means:** `train_kmeans_model_id`, `evaluate_kmeans_by_model_id`
- **GMM:** `train_gmm_model_id`, `predict_gmm_model_id`, `evaluate_gmm_by_model_id`

**Time series / forecasting:**

- **ARIMA:** `train_arima`, `evaluate_arima_by_model_id`
- **Time series (generic):** `train_timeseries_cpu`, `predict_timeseries_model_id`

**Other:**

- **Collaborative filtering:** `train_collaborative_filter`, `predict_collaborative_filter`, `evaluate_collaborative_filter_by_model_id`
- **Neural network (when present):** `predict_neural_network`

**Aggregate-style ML (no stored model):** `cluster_kmeans`, `cluster_minibatch_kmeans`, `cluster_gmm`, `gmm_to_clusters`, `detect_outliers_zscore`. ML project helpers: `neurondb_create_ml_project`, `neurondb_train_kmeans_project`, `neurondb_list_project_models`, `neurondb_deploy_model`, `neurondb_get_deployed_model`, `neurondb_get_project_info`.

---

## 11. Background workers

- **neuranq** — Job queue; table and depth config (e.g. `neurondb.neuranq_queue_depth`).
- **neuranmon** — Tuner/sampling for index parameter tuning; config e.g. `neurondb.neuranmon_enabled`, `neurondb.neuranmon_target_latency`, `neurondb.neuranmon_target_recall`.
- **neurandefrag** — Index defragmentation; config e.g. `neurondb.neurandefrag_enabled`, `neurandefrag_compact_threshold`, `neurandefrag_fragmentation_threshold`, `neurandefrag_maintenance_window`.
- **neuranllm** — LLM job processor; tables: `neurondb.llm_jobs`, `neurondb.llm_cache`, `neurondb.llm_config`, `neurondb.llm_stats`, `neurondb.llm_errors`; views: `neurondb.llm_job_status`, `neurondb.llm_model_stats`, `neurondb.llm_error_rates`, `neurondb.llm_latency_histograms`; function `neurondb.llm_gpu_utilization()`; `neurondb.llm_latency_percentile`. LLM completion: `ndb_llm_complete`, `ndb_llm_complete_batch`; vision: `ndb_llm_image_analyze`.

**Required:** `shared_preload_libraries = 'neurondb'` for workers and shared memory features.

---

## 12. GPU support (summary)

See [GPU feature matrix](docs/gpu/gpu-feature-matrix.md) for full detail.

**CUDA:** L2, cosine, inner product, Manhattan, Hamming; HNSW and IVF search; batch distance and batch embedding; memory pool and monitoring; multi-GPU. Index build is CPU only.

**ROCm:** L2, cosine, inner product full; Manhattan/Hamming partial; HNSW/IVF search; batch ops; memory monitoring partial. Index build CPU only.

**Metal (macOS):** L2, cosine, inner product; HNSW limited; IVF not supported; batch embedding limited; index build CPU only. Known limitations for large batches and concurrent queries.

**GUCs:** `neurondb.compute_mode`, `neurondb.gpu_device`, `neurondb.gpu_batch_size`, `neurondb.gpu_streams`, `neurondb.gpu_memory_pool_mb`, `neurondb.gpu_fail_open`, `neurondb.gpu_kernels`, `neurondb.gpu_backend`, `neurondb.gpu_timeout_ms`. Functions: `neurondb_gpu_info()`, `neurondb.gpu_enabled()`, `neurondb.gpu_device_count()`; GPU distance helpers e.g. `vector_l2_distance_gpu`, `vector_cosine_distance_gpu`, `vector_inner_product_gpu`; quantization on GPU: `vector_to_int8_gpu`, `vector_to_fp16_gpu`, `vector_to_binary_gpu`.

---

## 13. Configuration (GUCs)

**LLM/Embeddings:** `neurondb.llm_provider`, `neurondb.llm_api_key`, `neurondb.llm_endpoint`, `neurondb.llm_model`, `neurondb.llm_timeout_ms`, `neurondb.llm_max_retries`, `neurondb.llm_cache_ttl`, `neurondb.llm_rate_limiter_qps`, `neurondb.llm_fail_open`.

**GPU:** (see section 12.)

**Workers:** `neurondb.enable_background_workers`, `neurondb.worker_batch_size`, `neurondb.worker_interval_sec`, `neurondb.worker_max_concurrent`; queue: `neurondb.neuranq_queue_depth`; neuranmon, neurandefrag (see section 11).

**Monitoring:** `neurondb.enable_metrics`, `neurondb.metrics_retention_days`, `neurondb.log_level`.

**Index tuning:** `neurondb.hnsw_ef_search`, `neurondb.hnsw_k`, `neurondb.ivf_probes`.

---

## 14. Partial support and not supported

**Partial support**

- **GPU:** ROCm Manhattan/Hamming and memory monitoring partial; Metal has no IVF, limited batch and monitoring (see GPU matrix).
- **PostgreSQL versions:** Extension supports PG 16, 17, 18; on some platforms (e.g. macOS) PL/pgSQL fallbacks exist where C loaders differ.

**Not supported**

- **Index build on GPU:** HNSW and IVF index construction is CPU only.
- **Separate “hybrid” or “multi-vector” index AM:** Only two index access methods (HNSW, IVF). Hybrid and temporal are query-level features.
---

## 15. Counts (reference)

- **SQL objects:** The extension SQL defines on the order of **650+** functions and operators (exact number depends on the versioned SQL file, e.g. `neurondb--4.0.0-devel.sql` or `neurondb--2.1.0.sql`). See `docs/sql-api.md` and the SQL sources for the authoritative list.
- **Vector types:** 8 main types (`vector`, `vectorp`, `vecmap`, `vgraph`, `rtext`, `halfvec`, `binaryvec`, `sparsevec`).
- **Index access methods:** 2 (HNSW, IVF).
- **ML:** 25+ algorithm families with train/predict/evaluate (e.g. linear regression = one family; XGBoost classifier and regressor = two).

---

## Documentation

- [SQL API](docs/sql-api.md) — Functions, operators, and types
- [Data types](docs/reference/data-types.md) — Vector and related types, storage, casting
- [GPU feature matrix](docs/gpu/gpu-feature-matrix.md) — CUDA, ROCm, Metal support and gaps
- [Index methods](docs/internals/index-methods.md) — HNSW and IVF
- [Configuration](docs/configuration.md) — GUCs and deployment

---

[Back to top](#neurondb-features) · [README](README.md) · [Documentation](docs/readme.md)
