# NeuronDB - AI Database Extension for PostgreSQL

Vector search, machine learning, and hybrid search directly in PostgreSQL.

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/neurondb/NeurondB)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%2C17%2C18-blue.svg)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)

---

## Overview

NeuronDB extends PostgreSQL with vector search, ML model inference, hybrid retrieval, and RAG pipeline support.

## Documentation

### Getting Started
- **[Installation](docs/getting-started/installation.md)** - Install NeuronDB extension
- **[Quick Start](docs/getting-started/quickstart.md)** - Get up and running quickly

### Vector Search & Indexing
- **[Vector Types](docs/vector-search/vector-types.md)** - `vector`, `vectorp`, `vecmap`, `vgraph`, `rtext` types
- **[Indexing](docs/vector-search/indexing.md)** - HNSW and IVF indexing
- **[Distance Metrics](docs/vector-search/distance-metrics.md)** - L2, Cosine, Inner Product, and more
- **[Quantization](docs/vector-search/quantization.md)** - PQ and OPQ compression

### ML Algorithms & Analytics
- **[Random Forest](docs/ml-algorithms/random-forest.md)** - Classification and regression
- **[Gradient Boosting](docs/ml-algorithms/gradient-boosting.md)** - XGBoost, LightGBM, CatBoost
- **[Clustering](docs/ml-algorithms/clustering.md)** - K-Means, DBSCAN, GMM, Hierarchical
- **[Dimensionality Reduction](docs/ml-algorithms/dimensionality-reduction.md)** - PCA and PCA Whitening
- **[Classification](docs/ml-algorithms/classification.md)** - SVM, Logistic Regression, Naive Bayes, Decision Trees
- **[Regression](docs/ml-algorithms/regression.md)** - Linear, Ridge, Lasso, Deep Learning
- **[Outlier Detection](docs/ml-algorithms/outlier-detection.md)** - Z-score, Modified Z-score, IQR
- **[Quality Metrics](docs/ml-algorithms/quality-metrics.md)** - Recall@K, Precision@K, F1@K, MRR
- **[Drift Detection](docs/ml-algorithms/drift-detection.md)** - Centroid drift, Distribution divergence
- **[Topic Discovery](docs/ml-algorithms/topic-discovery.md)** - Topic modeling and analysis
- **[Time Series](docs/ml-algorithms/time-series.md)** - Forecasting and analysis
- **[Recommendation Systems](docs/ml-algorithms/recommendation-systems.md)** - Collaborative filtering

### ML & Embeddings
- **[Embedding Generation](docs/ml-embeddings/embedding-generation.md)** - Text, image, multimodal embeddings
- **[Model Inference](docs/ml-embeddings/model-inference.md)** - ONNX runtime, batch processing
- **[Model Management](docs/ml-embeddings/model-management.md)** - Load, export, version models
- **[AutoML](docs/ml-embeddings/automl.md)** - Automated hyperparameter tuning
- **[Feature Store](docs/ml-embeddings/feature-store.md)** - Feature management and versioning

### Hybrid Search & Retrieval
- **[Hybrid Search](docs/hybrid-search/overview.md)** - Combine vector and full-text search
- **[Multi-Vector](docs/hybrid-search/multi-vector.md)** - Multiple embeddings per document
- **[Faceted Search](docs/hybrid-search/faceted-search.md)** - Category-aware retrieval
- **[Temporal Search](docs/hybrid-search/temporal-search.md)** - Time-decay relevance scoring

### Reranking
- **[Cross-Encoder](docs/reranking/cross-encoder.md)** - Neural reranking models
- **[LLM Reranking](docs/reranking/llm-reranking.md)** - GPT/Claude-powered scoring
- **[ColBERT](docs/reranking/colbert.md)** - Late interaction models
- **[Ensemble](docs/reranking/ensemble.md)** - Combine multiple strategies

### RAG Pipeline
- **[Complete RAG Support](docs/rag/overview.md)** - End-to-end RAG
- **[LLM Integration](docs/rag/llm-integration.md)** - Hugging Face and OpenAI
- **[Document Processing](docs/rag/document-processing.md)** - Text processing and NLP

### Background Workers
- **[neuranq](docs/background-workers/neuranq.md)** - Async job queue executor
- **[neuranmon](docs/background-workers/neuranmon.md)** - Live query auto-tuner
- **[neurandefrag](docs/background-workers/neurandefrag.md)** - Index maintenance
- **[neuranllm](docs/background-workers/neuranllm.md)** - LLM job processor

### GPU Acceleration
- **[CUDA Support](docs/gpu/cuda-support.md)** - NVIDIA GPU acceleration
- **[ROCm Support](docs/gpu/rocm-support.md)** - AMD GPU acceleration
- **[Metal Support](docs/gpu/metal-support.md)** - Apple Silicon GPU acceleration
- **[Auto-Detection](docs/gpu/auto-detection.md)** - Automatic GPU detection

### Performance & Security
- **[SIMD Optimization](docs/performance/simd-optimization.md)** - AVX2/AVX512, NEON optimization
- **[Security](docs/security/overview.md)** - Encryption, privacy, RLS
- **[Monitoring](docs/performance/monitoring.md)** - Monitoring views and Prometheus

### Configuration & Operations
- **[Configuration](docs/configuration.md)** - Essential configuration options
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions

For detailed documentation, API references, best practices, and guides, visit [www.neurondb.ai](https://neurondb.ai/docs)

## Architecture

NeuronDB follows PostgreSQL's architectural patterns:

```
┌─────────────────────────────────────────────────────────┐
│                    SQL Interface                         │
│              (473 functions, types, operators)          │
├─────────────────────────────────────────────────────────┤
│  Vector Types  │  Distance Ops  │  Index Methods        │
│  (vector, vectorp, vecmap, vgraph, rtext)               │
├─────────────────────────────────────────────────────────┤
│  ML Inference  │  Embeddings    │  Model Management     │
│  (52 ML algorithms: RF, XGBoost, LightGBM, etc.)        │
├─────────────────────────────────────────────────────────┤
│  Hybrid Search │  Reranking     │  RAG Pipeline         │
├─────────────────────────────────────────────────────────┤
│  Background Workers (neuranq, neuranmon, neurandefrag)  │
├─────────────────────────────────────────────────────────┤
│  GPU Acceleration (CUDA/ROCm/Metal)                     │
├─────────────────────────────────────────────────────────┤
│  PostgreSQL Core (Storage, WAL, SPI, Shared Memory)     │
└─────────────────────────────────────────────────────────┘
```

## Compatibility

| PostgreSQL | Status | Platforms |
|------------|--------|-----------|
| 16.x | Supported | Ubuntu, Debian, Rocky Linux, macOS |
| 17.x | Supported | Ubuntu, Debian, Rocky Linux, macOS |
| 18.x | Supported | Ubuntu, Debian, Rocky Linux, macOS |

NeuronDB supports PostgreSQL 16, 17, and 18. The extension validates the PostgreSQL version at creation time.

## Support & Community

- **Issues**: [GitHub Issues](https://github.com/neurondb/NeurondB/issues)
- **Discussions**: [GitHub Discussions](https://github.com/neurondb/NeurondB/discussions)
- **Email**: support@neurondb.ai
- **Security**: Report vulnerabilities to support@neurondb.ai

## Contributing

We welcome contributions. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code style guidelines
- Development workflow
- Testing requirements
- Pull request process

## License

NeuronDB is released under the GNU Affero General Public License v3.0 (AGPL-3.0).
See [LICENSE](LICENSE) for details.

## Authors

**neurondb, Inc.**  
Email: support@neurondb.ai  
Website: https://neurondb.ai/docs

---

<div align="center">

**[Documentation](docs/README.md)** • 
**[Full Documentation](https://neurondb.ai/docs)** • 
**[GitHub](https://github.com/neurondb/NeurondB)** • 
**[Support](mailto:support@neurondb.ai)**

</div>
