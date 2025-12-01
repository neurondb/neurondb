# NeuronDB Documentation

NeuronDB is a PostgreSQL extension that brings native vector types, approximate nearest neighbor (ANN) search, GPU acceleration, ML analytics, hybrid semantic+lexical search, background workers, observability, and auto-tuning.

---

## Getting Started

- **[Installation](getting-started/installation.md)** - Install NeuronDB extension
- **[Quick Start](getting-started/quickstart.md)** - Get up and running quickly

---

## Vector Search & Indexing

- **[Vector Types](vector-search/vector-types.md)** - `vector`, `vectorp`, `vecmap`, `vgraph`, `rtext` types
- **[Indexing](vector-search/indexing.md)** - HNSW and IVF indexing with automatic tuning
- **[Distance Metrics](vector-search/distance-metrics.md)** - L2, Cosine, Inner Product, Manhattan, Hamming, Jaccard
- **[Quantization](vector-search/quantization.md)** - Product Quantization (PQ) and Optimized PQ (OPQ)

---

## ML Algorithms & Analytics

- **[Random Forest](ml-algorithms/random-forest.md)** - Classification and regression with GPU acceleration
- **[Gradient Boosting](ml-algorithms/gradient-boosting.md)** - XGBoost, LightGBM, CatBoost integration
- **[Clustering](ml-algorithms/clustering.md)** - K-Means, Mini-batch K-means, DBSCAN, GMM, Hierarchical clustering
- **[Dimensionality Reduction](ml-algorithms/dimensionality-reduction.md)** - PCA and PCA Whitening
- **[Classification](ml-algorithms/classification.md)** - SVM, Logistic Regression, Naive Bayes, Decision Trees, Neural Networks
- **[Regression](ml-algorithms/regression.md)** - Linear Regression, Ridge, Lasso, Deep Learning models
- **[Outlier Detection](ml-algorithms/outlier-detection.md)** - Z-score, Modified Z-score, IQR methods
- **[Quality Metrics](ml-algorithms/quality-metrics.md)** - Recall@K, Precision@K, F1@K, MRR, Davies-Bouldin Index
- **[Drift Detection](ml-algorithms/drift-detection.md)** - Centroid drift, Distribution divergence, Temporal monitoring
- **[Topic Discovery](ml-algorithms/topic-discovery.md)** - Topic modeling and analysis
- **[Time Series](ml-algorithms/time-series.md)** - Forecasting and analysis
- **[Recommendation Systems](ml-algorithms/recommendation-systems.md)** - Collaborative filtering and ranking

---

## ML & Embeddings

- **[Embedding Generation](ml-embeddings/embedding-generation.md)** - Text, image, multimodal embeddings with intelligent caching
- **[Model Inference](ml-embeddings/model-inference.md)** - ONNX runtime, batch processing, model management
- **[Model Management](ml-embeddings/model-management.md)** - Load, export, version, monitor models with catalog integration
- **[AutoML](ml-embeddings/automl.md)** - Automated hyperparameter tuning and model selection
- **[Feature Store](ml-embeddings/feature-store.md)** - Centralized feature management and versioning

---

## Hybrid Search & Retrieval

- **[Hybrid Search](hybrid-search/overview.md)** - Combine vector and full-text search with configurable weights
- **[Multi-Vector](hybrid-search/multi-vector.md)** - Multiple embeddings per document for enhanced retrieval
- **[Faceted Search](hybrid-search/faceted-search.md)** - Category-aware retrieval with filtering
- **[Temporal Search](hybrid-search/temporal-search.md)** - Time-decay relevance scoring

---

## Reranking

- **[Cross-Encoder](reranking/cross-encoder.md)** - Neural reranking models
- **[LLM Reranking](reranking/llm-reranking.md)** - GPT/Claude-powered scoring
- **[ColBERT](reranking/colbert.md)** - Late interaction models
- **[Ensemble](reranking/ensemble.md)** - Combine multiple reranking strategies

---

## RAG Pipeline

- **[Complete RAG Support](rag/overview.md)** - End-to-end Retrieval Augmented Generation
- **[LLM Integration](rag/llm-integration.md)** - Hugging Face and OpenAI integration
- **[Document Processing](rag/document-processing.md)** - Text processing and NLP capabilities

---

## Background Workers

- **[neuranq](background-workers/neuranq.md)** - Async job queue executor with batch processing
- **[neuranmon](background-workers/neuranmon.md)** - Live query auto-tuner and performance optimization
- **[neurandefrag](background-workers/neurandefrag.md)** - Automatic index maintenance and defragmentation
- **[neuranllm](background-workers/neuranllm.md)** - LLM job processing with crash recovery

---

## GPU Acceleration

- **[CUDA Support](gpu/cuda-support.md)** - NVIDIA GPU acceleration for vector operations and ML inference
- **[ROCm Support](gpu/rocm-support.md)** - AMD GPU acceleration
- **[Metal Support](gpu/metal-support.md)** - Apple Silicon GPU acceleration
- **[Auto-Detection](gpu/auto-detection.md)** - Automatic GPU detection and fallback to CPU

---

## Performance & Security

- **[SIMD Optimization](performance/simd-optimization.md)** - AVX2/AVX512 (x86_64), NEON (ARM64) with prefetching
- **[Security](security/overview.md)** - Encryption, differential privacy, Row-Level Security (RLS) integration
- **[Monitoring](performance/monitoring.md)** - 7 built-in monitoring views, Prometheus metrics export

---

## Configuration & Operations

- **[Configuration](configuration.md)** - Essential configuration options
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

---

## Requirements

- PostgreSQL 16, 17, or 18
- Build toolchain for PG extensions (server headers, `pg_config` on PATH)
- Optional: CUDA or ROCm for GPU acceleration

---

## Detailed Documentation

For detailed documentation, configuration, comprehensive API references, best practices, and performance tuning guides, visit:

**[www.neurondb.ai](https://neurondb.ai/docs)**

---

## Support & Community

- **GitHub Issues**: [Report bugs or request features](https://github.com/neurondb/NeurondB/issues)
- **Email**: [support@neurondb.ai](mailto:support@neurondb.ai)
- **Website**: [neurondb.ai](https://neurondb.ai/docs)

---

**Built by [neurondb, Inc.](https://neurondb.ai)**
