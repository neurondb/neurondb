# NeuronDB project overview

This page preserves the former long-form introduction from the repository README: capabilities, performance notes, documentation map, architecture diagrams, compatibility, support, license, and authors.

---

## Overview

**Vectors, embeddings, and ML—inside PostgreSQL.** NeuronDB keeps similarity search and models on **your live rows**, not in a separate database you have to sync and babysit.

**HNSW · IVFFlat · kNN · hybrid full-text + vector · RAG pieces · train & predict in SQL**—all first-class in the engine.

**One extension:** same Postgres **backups, HA, and security**. Start with **`CREATE EXTENSION neurondb;`**, then index and query from SQL.

### Key Capabilities

<details>
<summary><strong>Feature summary</strong></summary>

| Category | Details |
|:---------|:--------|
| **Vector types** | `vector`, `vectorp`, `vecmap`, `vgraph`, `rtext`, `halfvec`, `binaryvec`, `sparsevec` (8 types) |
| **Index access methods** | HNSW and IVF only. PQ and OPQ are quantization (codebook training), not separate index types. Hybrid and multi-vector search are query-level functions. |
| **Distance metrics** | L2, cosine, inner product, L1, Hamming, Jaccard, and others |
| **ML** | 25+ algorithm families (train/predict/evaluate): linear regression, XGBoost, LightGBM, CatBoost, K-Means, etc. |
| **SQL** | ~650+ functions and operators (vector, ML, embeddings, RAG, indexing). See [FEATURES.md](../FEATURES.md) and [SQL API](sql-api.md). |
| **GPU** | CUDA, ROCm, Metal (distance and search; index build is CPU only). See [GPU feature matrix](gpu/gpu-feature-matrix.md). |
| **Background workers** | neuranq, neuranmon, neurandefrag, neuranllm |

</details>

### Performance Metrics

NeuronDB provides significant performance improvements over standard PostgreSQL extensions:

**Index Build Performance:**

The index build time for HNSW follows the relationship:

$$T_{build} = O(N \cdot \log N \cdot m \cdot ef_{construction})$$

Where:
- $N$ = number of vectors
- $m$ = number of connections per node (typically 16-32)
- $ef_{construction}$ = size of candidate list during construction (typically 64-200)

**Query Performance:**

Query latency for HNSW search:

$$T_{query} = O(\log N + ef_{search} \cdot k)$$

Where:
- $ef_{search}$ = size of candidate list during search (typically 40-200)
- $k$ = number of results requested

**Throughput Calculation:**

$$QPS = \frac{1}{T_{query}} = \frac{1}{O(\log N + ef_{search} \cdot k)}$$

> [!TIP]
> For optimal performance, tune `ef_search` based on your recall requirements. Higher values improve recall but increase latency.

## Documentation map

### Getting Started
- **[Installation](getting-started/installation.md)** - Install NeuronDB extension
- **[Extension packaging](../EXTENSION.md)** - Control file, file layout, CREATE/UPDATE/DROP EXTENSION, dump/restore
- **[Quick Start](getting-started/quickstart.md)** - Get up and running quickly

### Vector Search & Indexing
- **[Vector Types](vector-search/vector-types.md)** — `vector`, `vectorp`, `vecmap`, `vgraph`, `rtext`, `halfvec`, `binaryvec`, `sparsevec`
- **[Indexing](vector-search/indexing.md)** — HNSW and IVF indexing
- **[Distance Metrics](vector-search/distance-metrics.md)** — L2, cosine, inner product, and more
- **[Quantization](vector-search/quantization.md)** — PQ and OPQ compression

### ML Algorithms & Analytics
- **[Random Forest](ml-algorithms/random-forest.md)** - Classification and regression
- **[Gradient Boosting](ml-algorithms/gradient-boosting.md)** - XGBoost, LightGBM, CatBoost
- **[Clustering](ml-algorithms/clustering.md)** - K-Means, DBSCAN, GMM, Hierarchical
- **[Dimensionality Reduction](ml-algorithms/dimensionality-reduction.md)** - PCA and PCA Whitening
- **[Classification](ml-algorithms/classification.md)** - SVM, Logistic Regression, Naive Bayes, Decision Trees
- **[Regression](ml-algorithms/regression.md)** - Linear, Ridge, Lasso, Deep Learning
- **[Outlier Detection](ml-algorithms/outlier-detection.md)** - Z-score, Modified Z-score, IQR
- **[Quality Metrics](ml-algorithms/quality-metrics.md)** - Recall@K, Precision@K, F1@K, MRR
- **[Drift Detection](ml-algorithms/drift-detection.md)** - Centroid drift, Distribution divergence
- **[Topic Discovery](ml-algorithms/topic-discovery.md)** - Topic modeling and analysis
- **[Time Series](ml-algorithms/time-series.md)** - Forecasting and analysis
- **[Recommendation Systems](ml-algorithms/recommendation-systems.md)** - Collaborative filtering

### ML & Embeddings
- **[Embedding Generation](ml-embeddings/embedding-generation.md)** - Text, image, multimodal embeddings
- **[Model Inference](ml-embeddings/model-inference.md)** - ONNX runtime, batch processing
- **[Model Management](ml-embeddings/model-management.md)** - Load, export, version models
- **[AutoML](ml-embeddings/automl.md)** - Automated hyperparameter tuning
- **[Feature Store](ml-embeddings/feature-store.md)** - Feature management and versioning

### Hybrid Search & Retrieval
- **[Hybrid Search](hybrid-search/overview.md)** - Combine vector and full-text search
- **[Multi-Vector](hybrid-search/multi-vector.md)** - Multiple embeddings per document
- **[Faceted Search](hybrid-search/faceted-search.md)** - Category-aware retrieval
- **[Temporal Search](hybrid-search/temporal-search.md)** - Time-decay relevance scoring

### Reranking
- **[Cross-Encoder](reranking/cross-encoder.md)** - Neural reranking models
- **[LLM Reranking](reranking/llm-reranking.md)** - GPT/Claude-powered scoring
- **[ColBERT](reranking/colbert.md)** - Late interaction models
- **[Ensemble](reranking/ensemble.md)** - Combine multiple strategies

### RAG Pipeline
- **[Complete RAG Support](rag/overview.md)** - End-to-end RAG
- **[LLM Integration](rag/llm-integration.md)** - Hugging Face and OpenAI
- **[Document Processing](rag/document-processing.md)** - Text processing and NLP

### Background Workers
- **[neuranq](background-workers/neuranq.md)** - Async job queue executor
- **[neuranmon](background-workers/neuranmon.md)** - Live query auto-tuner
- **[neurandefrag](background-workers/neurandefrag.md)** - Index maintenance
- **[neuranllm](background-workers/neuranllm.md)** - LLM job processor

### GPU Acceleration
- **[CUDA Support](gpu/cuda-support.md)** - NVIDIA GPU acceleration
- **[ROCm Support](gpu/rocm-support.md)** - AMD GPU acceleration
- **[Metal Support](gpu/metal-support.md)** - Apple Silicon GPU acceleration
- **[Auto-Detection](gpu/auto-detection.md)** - Automatic GPU detection

### Performance & Security
- **[SIMD Optimization](performance/simd-optimization.md)** - AVX2/AVX512, NEON optimization
- **[Security](security/overview.md)** - Encryption, privacy, RLS
- **[Monitoring](performance/monitoring.md)** - Monitoring views and Prometheus

### Configuration & Operations
- **[Configuration](configuration.md)** - Essential configuration options
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions (getting started and operations)

## Official Documentation

**[https://www.neurondb.ai/docs](https://www.neurondb.ai/docs)** — API reference (~650+ SQL functions), tutorials, deployment, and troubleshooting.

## Architecture

NeuronDB follows PostgreSQL's architectural patterns and extends the database with AI capabilities.

### System Architecture

```mermaid
graph TB
    subgraph SQL["SQL Interface Layer"]
        FUNC["~650+ SQL Functions"]
        TYPES["Vector Types: vector, vectorp, vecmap, vgraph, rtext, halfvec, binaryvec, sparsevec"]
        OPS["Distance Operators: <->, <=>, <#>"]
    end
    
    subgraph VECTOR["Vector Operations"]
        INDEX["HNSW/IVF Indexes"]
        DIST["Distance Metrics: L2, Cosine, Inner Product"]
        QUANT["Quantization: PQ, OPQ, int8, fp16"]
    end
    
    subgraph ML["Machine Learning"]
        ALGO["25+ ML algorithm families: RF, XGBoost, LightGBM, etc."]
        INFER["Model Inference: ONNX Runtime"]
        EMBED["Embedding Generation: Text, Image, Multimodal"]
    end
    
    subgraph SEARCH["Search & Retrieval"]
        HYBRID[Hybrid Search<br/>Vector + Full-text]
        RERANK[Reranking<br/>Cross-encoder, LLM, ColBERT]
        RAG[RAG Pipeline<br/>Document Processing]
    end
    
    subgraph WORKERS["Background Workers"]
        NEURANQ[neuranq<br/>Job Queue]
        NEURANMON[neuranmon<br/>Query Tuner]
        NEURANDEFRAG[neurandefrag<br/>Index Maintenance]
        NEURANLLM[neuranllm<br/>LLM Processor]
    end
    
    subgraph GPU["GPU Acceleration"]
        CUDA[CUDA<br/>NVIDIA]
        ROCM[ROCm<br/>AMD]
        METAL[Metal<br/>Apple Silicon]
    end
    
    subgraph PG["PostgreSQL Core"]
        STORAGE[Storage Engine]
        WAL[Write-Ahead Log]
        SPI[Server Programming Interface]
        SHMEM[Shared Memory]
    end
    
    SQL --> VECTOR
    SQL --> ML
    SQL --> SEARCH
    VECTOR --> INDEX
    ML --> INFER
    SEARCH --> RAG
    WORKERS --> PG
    GPU --> VECTOR
    GPU --> ML
    VECTOR --> PG
    ML --> PG
    SEARCH --> PG
```

### Vector Query Flow

```mermaid
sequenceDiagram
    participant Client
    participant PG as PostgreSQL
    participant ND as NeuronDB Extension
    participant Index as HNSW/IVF Index
    participant GPU as GPU Backend
    
    Client->>PG: SELECT ... ORDER BY embedding <=> query
    PG->>ND: Parse vector query
    ND->>ND: Optimize query plan
    
    alt Index Available
        ND->>Index: Search index (ef_search)
        Index->>ND: Return candidate vectors
        ND->>GPU: Compute distances (SIMD/GPU)
        GPU-->>ND: Distance scores
        ND->>ND: Sort and filter (LIMIT)
    else Sequential Scan
        ND->>PG: Scan table
        ND->>GPU: Compute all distances
        GPU-->>ND: Distance scores
        ND->>ND: Sort and filter
    end
    
    ND-->>PG: Return results
    PG-->>Client: Query results
```

### HNSW Index Structure

```mermaid
graph TD
    subgraph HNSW["HNSW (Hierarchical Navigable Small World)"]
        L2[Layer 2<br/>Few nodes, long edges]
        L1[Layer 1<br/>More nodes, medium edges]
        L0[Layer 0<br/>All nodes, short edges]
    end
    
    L2 -->|Entry Point| L1
    L1 -->|Entry Point| L0
```

> [!NOTE]
> HNSW creates a multi-layer graph where higher layers have fewer nodes and longer edges, enabling fast approximate nearest neighbor search. The search starts at the top layer and navigates down to find the closest neighbors.

## Compatibility

<details>
<summary><strong>Compatibility matrix</strong></summary>

| PostgreSQL | Status | Platforms | Architectures |
|:----------|:-------|:----------|:--------------|
| 16.x | ✅ Supported | Ubuntu 20.04/22.04, Debian 11/12, Rocky Linux 8/9, macOS 13+ | linux/amd64, linux/arm64, darwin/arm64 |
| 17.x | ✅ Supported | Ubuntu 20.04/22.04, Debian 11/12, Rocky Linux 8/9, macOS 13+ | linux/amd64, linux/arm64, darwin/arm64 |
| 18.x | ✅ Supported | Ubuntu 20.04/22.04, Debian 11/12, Rocky Linux 8/9, macOS 13+ | linux/amd64, linux/arm64, darwin/arm64 |

</details>

> [!NOTE]
> NeuronDB supports PostgreSQL 16, 17, and 18. The extension validates the PostgreSQL version at creation time. GPU acceleration requires platform-specific drivers (CUDA 12.2+, ROCm 5.7+, or macOS 13+ for Metal).

## Support & Community

<details>
<summary><strong>Get help</strong></summary>

| Resource | Link | Description |
|:---------|:-----|:------------|
| **GitHub Issues** | [Report Issues](https://github.com/neurondb/neurondb/issues) | Bug reports and feature requests |
| **GitHub Discussions** | [Join Discussion](https://github.com/neurondb/neurondb/discussions) | Community Q&A and discussions |
| **Email Support** | support@neurondb.ai | Direct email support |
| **Security Issues** | security@neurondb.ai | Report security vulnerabilities |
| **Documentation** | [neurondb.ai/docs](https://www.neurondb.ai/docs) | Complete documentation |

</details>

## Contributing

We welcome contributions. See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

NeuronDB is released under a proprietary license. See [LICENSE](../LICENSE) for details.

<details>
<summary><strong>License summary</strong></summary>

| Usage Type | Permitted | Notes |
|:-----------|:----------|:------|
| **Personal Use** | ✅ Yes | Binary code only |
| **Commercial Use** | ❌ No | Contact for licensing |
| **Source Modifications** | ❌ No | No derivatives allowed |
| **Redistribution** | ❌ No | Contact for distribution rights |

**For commercial licensing**, contact support@neurondb.ai

</details>

## Authors

**neurondb, Inc.**  
Email: support@neurondb.ai  
Website: https://neurondb.ai/docs
