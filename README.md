# NeuronDB - AI Database Extension for PostgreSQL

<div align="center">

**Vector search, machine learning, and hybrid search directly in PostgreSQL**

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/neurondb/NeurondB)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%2C17%2C18-blue.svg)](https://www.postgresql.org/)
[![Version](https://img.shields.io/badge/version-3.0.0--devel-blue.svg)](https://github.com/neurondb/neurondb)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-neurondb.ai-brightgreen.svg)](https://www.neurondb.ai/docs)

</div>

---

## 📑 Table of Contents

<details>
<summary><strong>Expand full table of contents</strong></summary>

- [Overview](#overview)
  - [Key Capabilities](#key-capabilities)
  - [Performance Metrics](#performance-metrics)
- [Documentation](#documentation)
  - [Getting Started](#getting-started)
  - [Vector Search & Indexing](#vector-search--indexing)
  - [ML Algorithms & Analytics](#ml-algorithms--analytics)
  - [ML & Embeddings](#ml--embeddings)
  - [Hybrid Search & Retrieval](#hybrid-search--retrieval)
  - [Reranking](#reranking)
  - [RAG Pipeline](#rag-pipeline)
  - [Background Workers](#background-workers)
  - [GPU Acceleration](#gpu-acceleration)
  - [Performance & Security](#performance--security)
  - [Configuration & Operations](#configuration--operations)
- [Official Documentation](#official-documentation)
- [Architecture](#architecture)
  - [System Architecture](#system-architecture)
  - [Vector Query Flow](#vector-query-flow)
  - [HNSW Index Structure](#hnsw-index-structure)
- [Compatibility](#compatibility)
- [Support & Community](#support--community)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)

</details>

---

## Overview

NeuronDB extends PostgreSQL with vector search, ML model inference, hybrid retrieval, and RAG pipeline support.

### Key Capabilities

<details>
<summary><strong>📊 Feature Summary</strong></summary>

| Category | Features | Count |
|:---------|:---------|:-----|
| **Vector Types** | `vector`, `vectorp`, `vecmap`, `vgraph`, `rtext`, `sparse_vector` | 6 types |
| **Index Types** | HNSW, IVF, PQ, OPQ, hybrid, multi-vector | 6+ types |
| **Distance Metrics** | L2, Cosine, Inner Product, Hamming, Jaccard, etc. | 7+ metrics |
| **ML Algorithms** | Random Forest, XGBoost, LightGBM, K-Means, PCA, etc. | 52+ algorithms |
| **SQL Functions** | Vector ops, ML inference, embeddings, RAG, etc. | 665+ functions |
| **GPU Backends** | CUDA, ROCm, Metal | 3 backends |
| **Background Workers** | neuranq, neuranmon, neurandefrag, neuranllm | 4 workers |

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

## Documentation

### Getting Started
- **[Installation](docs/getting-started/installation.md)** - Install NeuronDB extension
- **[Extension packaging](EXTENSION.md)** - Control file, file layout, CREATE/UPDATE/DROP EXTENSION, dump/restore
- **[Quick Start](docs/getting-started/quickstart.md)** - Get up and running quickly

### Vector Search & Indexing
- **[Vector Types](docs/vector-search/vector-types.md)** - `vector`, `vectorp`, `vecmap`, `vgraph`, `rtext`, `sparse_vector` types
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

## Official Documentation

**For comprehensive documentation, detailed tutorials, complete API references, best practices, and production guides, visit:**

🌐 **[https://www.neurondb.ai/docs](https://www.neurondb.ai/docs)**

The official documentation site provides:
- **Complete API Reference**: All 665+ SQL functions with examples
- **Detailed Tutorials**: Step-by-step guides for all features
- **Performance Guides**: Optimization strategies and benchmarks
- **Production Best Practices**: Deployment, scaling, and monitoring
- **Troubleshooting**: Common issues and solutions
- **Latest Updates**: Release notes and what's new

### Quick Links to Official Documentation

| Topic | Link |
|-------|------|
| Getting Started | [Quick Start Guide](https://www.neurondb.ai/docs/getting-started) |
| Vector Search | [Vector Search Documentation](https://www.neurondb.ai/docs/vector-search) |
| ML Algorithms | [ML Algorithms Guide](https://www.neurondb.ai/docs/ml-algorithms) |
| RAG Pipeline | [RAG Documentation](https://www.neurondb.ai/docs/rag) |
| GPU Acceleration | [GPU Support Guide](https://www.neurondb.ai/docs/gpu) |
| Hybrid Search | [Hybrid Search Guide](https://www.neurondb.ai/docs/hybrid-search) |
| Performance | [Performance Optimization](https://www.neurondb.ai/docs/performance) |
| Security | [Security Features](https://www.neurondb.ai/docs/security) |
| API Reference | [Complete API Reference](https://www.neurondb.ai/docs/api) |

## Architecture

NeuronDB follows PostgreSQL's architectural patterns and extends the database with AI capabilities.

### System Architecture

```mermaid
graph TB
    subgraph SQL["SQL Interface Layer"]
        FUNC[665+ SQL Functions]
        TYPES[Vector Types<br/>vector, vectorp, vecmap, vgraph, rtext, sparse_vector]
        OPS[Distance Operators<br/><=>, <->, <#>]
    end
    
    subgraph VECTOR["Vector Operations"]
        INDEX[HNSW/IVF Indexes]
        DIST[Distance Metrics<br/>L2, Cosine, Inner Product]
        QUANT[Quantization<br/>PQ, OPQ, int8, fp16]
    end
    
    subgraph ML["Machine Learning"]
        ALGO[52+ ML Algorithms<br/>RF, XGBoost, LightGBM, etc.]
        INFER[Model Inference<br/>ONNX Runtime]
        EMBED[Embedding Generation<br/>Text, Image, Multimodal]
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
    
    style SQL fill:#e3f2fd
    style VECTOR fill:#fff3e0
    style ML fill:#f3e5f5
    style SEARCH fill:#e8f5e9
    style WORKERS fill:#fce4ec
    style GPU fill:#fff9c4
    style PG fill:#e0f2f1
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
    
    style L2 fill:#ffebee
    style L1 fill:#fff3e0
    style L0 fill:#e8f5e9
```

> [!NOTE]
> HNSW creates a multi-layer graph where higher layers have fewer nodes and longer edges, enabling fast approximate nearest neighbor search. The search starts at the top layer and navigates down to find the closest neighbors.

## Compatibility

<details>
<summary><strong>📋 Compatibility Matrix</strong></summary>

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
<summary><strong>📞 Get Help</strong></summary>

| Resource | Link | Description |
|:---------|:-----|:------------|
| **GitHub Issues** | [Report Issues](https://github.com/neurondb/NeurondB/issues) | Bug reports and feature requests |
| **GitHub Discussions** | [Join Discussion](https://github.com/neurondb/NeurondB/discussions) | Community Q&A and discussions |
| **Email Support** | support@neurondb.ai | Direct email support |
| **Security Issues** | security@neurondb.ai | Report security vulnerabilities |
| **Documentation** | [neurondb.ai/docs](https://www.neurondb.ai/docs) | Complete documentation |

</details>

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for:

<details>
<summary><strong>📝 Contribution Guidelines</strong></summary>

- ✅ **Code style guidelines** - Follow PostgreSQL coding standards
- ✅ **Development workflow** - Fork, branch, test, submit PR
- ✅ **Testing requirements** - All changes must include tests
- ✅ **Pull request process** - Review and approval workflow
- ✅ **Documentation** - Update docs for new features

</details>

## License

NeuronDB is released under a proprietary license. See [LICENSE](../LICENSE) for details.

<details>
<summary><strong>📄 License Summary</strong></summary>

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

---

<div align="center">

**[Documentation](docs/)** • 
**[Full Documentation](https://neurondb.ai/docs)** • 
**[GitHub](https://github.com/neurondb/NeurondB)** • 
**[Support](mailto:support@neurondb.ai)**

[⬆ Back to Top](#neurondb---ai-database-extension-for-postgresql)

</div>
