# NeuronDB Feature Summary

Short summary. For the full, scoped feature list (full / partial / not present) and exhaustive function lists, see **[FEATURES.md](FEATURES.md)**.

---

## What this extension provides

**Vector types (8):** `vector`, `vectorp`, `vecmap`, `vgraph`, `rtext`, `halfvec`, `binaryvec`, `sparsevec`. `vector` is float32, max 16,000 dimensions; `halfvec` is float16 (2x compression), max 4,000 dims; `sparsevec` supports BM25/SPLADE/ColBERTv2, max 1M dimensions and 1,000 non-zeros; `binaryvec` is Hamming-only.

**Index access methods (2 only):** `hnsw` (params: `m`, `ef_construction`) and `ivfflat` (param: `lists`). Operator classes: `vector_l2_ops`, `vector_cosine_ops`, `vector_ip_ops` for `vector`; same for `halfvec` and `sparsevec`; `binaryvec_hamming_ops`; `bit_hamming_ops`, `bit_jaccard_ops`. PQ and OPQ are quantization (e.g. `train_pq_codebook`, `train_opq_rotation`), not index types. Hybrid and temporal search are query-level SQL functions, not a third index AM.

**Distance operators:** `<->` L2, `<=>` cosine, `<#>` inner product (vector/halfvec/sparsevec); `<+>` L1, `<~>` Hamming, `<*~*>` Jaccard (vector). 20+ distance/similarity functions including L2, squared L2, cosine, inner product, L1, Hamming, Chebyshev, Minkowski, Jaccard, Dice, Mahalanobis; batch variants (`vector_*_distance_batch`).

**SQL surface:** ~650+ functions and operators (exact count in versioned SQL, e.g. `neurondb--3.0.0-devel.sql`). Includes: embedding (`embed_text`, `embed_text_batch`, `embed_image`, `embed_multimodal`, `embed_cached`); RAG (`neurondb.rag_query`, tables `neurondb.rag_pipelines`, `neurondb.rag_operation_audit_log`); hybrid (`hybrid_search`, `hybrid_search_fusion`, `reciprocal_rank_fusion`); reranking (cross-encoder, LLM, Cohere, ColBERT, LTR, ensemble, MMR/Borda, cache helpers); vector aggregates (`vector_avg`, `vector_sum`, batch); quantization conversions (int8, fp16, binary, uint8, ternary, int4).

**ML:** 25+ algorithm families with train/predict/evaluate and model_id catalog: linear, ridge, lasso regression; logistic regression; KNN; random forest; decision tree; naive Bayes; SVM; K-Means; GMM; XGBoost (classifier/regressor); LightGBM (classifier/regressor); CatBoost (classifier/regressor); ARIMA; time series; collaborative filtering. Aggregate-style: `cluster_kmeans`, `cluster_minibatch_kmeans`, `cluster_gmm`, `detect_outliers_zscore`.

**Workers:** neuranq (queue), neuranmon (index tuner), neurandefrag (defrag), neuranllm (LLM jobs). Require `shared_preload_libraries = 'neurondb'`. Tables/views: `neurondb.llm_jobs`, `neurondb.llm_cache`, `neurondb.llm_config`, `neurondb.llm_stats`, `neurondb.llm_errors`, `neurondb.llm_job_status`, etc.

**GPU:** CUDA (full distance + HNSW/IVF search + batch; index build CPU only), ROCm (Manhattan/Hamming partial, memory monitoring partial), Metal (HNSW limited, no IVF, batch limited). GUCs: `neurondb.compute_mode`, `neurondb.gpu_device`, `neurondb.gpu_batch_size`, `neurondb.gpu_fail_open`, etc. See [GPU feature matrix](docs/gpu/gpu-feature-matrix.md).

**Not in this extension:** Index build on GPU (CPU only).

---

## Documentation

- **Full feature list:** [FEATURES.md](FEATURES.md)
- **Docs index:** [docs/readme.md](docs/readme.md)
- **SQL API:** [docs/sql-api.md](docs/sql-api.md)
- **Data types:** [docs/reference/data-types.md](docs/reference/data-types.md)
- **Index methods:** [docs/internals/index-methods.md](docs/internals/index-methods.md)
- **Configuration:** [docs/configuration.md](docs/configuration.md)

[README](README.md) · [FEATURES.md](FEATURES.md) · [Documentation](docs/readme.md)
