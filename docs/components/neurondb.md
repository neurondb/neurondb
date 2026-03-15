# NeuronDB

NeuronDB is a PostgreSQL extension that adds vector search, machine learning, and AI capabilities directly to PostgreSQL.

## What it is

- A PostgreSQL extension that defines types (for example `vector`), operators, and index access methods
- ~650+ SQL functions for vector operations, ML, embeddings, and RAG
- 25+ machine learning algorithm families
- GPU acceleration for CUDA, ROCm, and Metal platforms
- Background workers for async operations, auto-tuning, and maintenance

## Key Modules & Features

### Vector Operations
- **Vector Types**: `vector`, `vectorp`, `vecmap`, `vgraph`, `rtext`, `sparse_vector` for dense and sparse vectors
- **Distance Metrics**: L2, cosine similarity, inner product, and more
- **Indexing**: HNSW and IVF indexes for fast similarity search
- **Quantization**: Product Quantization (PQ) and Optimized Product Quantization (OPQ)

### Machine Learning (25+ algorithm families)
- **Classification**: Random Forest, SVM, Logistic Regression, Naive Bayes, Decision Trees
- **Regression**: Linear, Ridge, Lasso, Neural Networks, Deep Learning
- **Clustering**: K-Means, Mini-batch K-Means, DBSCAN, Gaussian Mixture Model, Hierarchical
- **Gradient Boosting**: XGBoost, LightGBM, CatBoost
- **Dimensionality Reduction**: PCA, PCA Whitening
- **Outlier Detection**: Z-score, Modified Z-score, IQR
- **Time Series**: Forecasting and analysis
- **Recommendation Systems**: Collaborative filtering
- **Quality Metrics**: Recall@K, Precision@K, F1@K, MRR, Davies-Bouldin Index

### Embeddings & LLM Integration
- **Embedding Generation**: Text, image, and multimodal embeddings
- **ONNX Runtime**: Model inference and management
- **Hugging Face Integration**: Direct model loading and inference
- **OpenAI Integration**: API-based embeddings and completions
- **Model Management**: Model catalog, versioning, and deployment

### Hybrid Search & Retrieval
- **Hybrid Search**: Combine vector and full-text search
- **Temporal Search**: Time-decay relevance scoring
- **Sparse Search**: Sparse vector operations
- **Multi-Vector**: Multiple embeddings per document

### Reranking
- **Cross-Encoder**: Neural reranking models
- **LLM Reranking**: GPT/Claude-powered scoring
- **Ensemble Reranking**: Combine multiple strategies
- **Learning to Rank (LTR)**: Trainable reranking models

### RAG Pipeline
- **Document Processing**: Text extraction and chunking
- **Context Retrieval**: Semantic search for context
- **LLM Integration**: Complete RAG workflows

### Background Workers
- **neuranq**: Async job queue executor
- **neuranmon**: Live query auto-tuner for index optimization
- **neurandefrag**: Index maintenance and defragmentation
- **neuranllm**: LLM job processor for embeddings and completions

### GPU Acceleration
- **CUDA**: NVIDIA GPU acceleration
- **ROCm**: AMD GPU acceleration
- **Metal**: Apple Silicon GPU acceleration
- **Auto-Detection**: Automatic GPU backend selection

### Multi-Tenancy & Security
- **Tenant Management**: Per-tenant resource quotas and isolation
- **Row-Level Security**: RLS policies for vector data
- **Encryption**: Post-quantum encryption support
- **Access Control**: Fine-grained permissions

### Observability
- **Monitoring Views**: Vector stats, index health, query performance
- **Prometheus Integration**: Metrics export
- **Performance Tracking**: Query performance metrics

## Documentation

- **Main README**: [README.md](../../README.md) (repository root)
- **Installation**: [INSTALL.md](../../INSTALL.md) (repository root)
- **Complete Docs**: [docs/](../../docs/)
- **SQL API**: Functions defined in `sql/neurondb--*.sql` (repository root)
- **Official Docs**: [https://www.neurondb.ai/docs](https://www.neurondb.ai/docs)

## Docker

- Compose file: `docker/docker-compose.yml`. Services: `neurondb` (CPU) and GPU profiles (cuda, rocm, metal).
- See [docker/README.md](../../docker/README.md).

## Quick Start

After Postgres is running:

```sql
CREATE EXTENSION IF NOT EXISTS neurondb;
SELECT neurondb.version();
```

For detailed setup, see [INSTALL.md](../../INSTALL.md) or [Installation](../getting-started/installation.md).
