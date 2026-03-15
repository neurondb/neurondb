# NeuronDB Documentation

Technical documentation for the **NeuronDB** PostgreSQL extension: vector search, machine learning, embeddings, and RAG directly in PostgreSQL.

---

## Quick links

| Goal | Document |
|------|----------|
| Run NeuronDB in under 5 minutes | [Simple Start](getting-started/simple-start.md) |
| Install (Docker or native) | [Installation](getting-started/installation.md) |
| First queries and sample data | [Quick Start](getting-started/quickstart.md) |
| All SQL functions and types | [SQL API](sql-api.md) · [Data types](reference/data-types.md) |
| Configuration (GUCs) | [Configuration](configuration.md) |
| Troubleshooting | [Troubleshooting](getting-started/troubleshooting.md) |

---

## Documentation map

<details open>
<summary><strong>Getting started</strong></summary>

| Document | Description |
|----------|-------------|
| [Simple Start](getting-started/simple-start.md) | Step-by-step setup (Docker or native) |
| [Quick Start](getting-started/quickstart.md) | Load data and run vector search |
| [Installation](getting-started/installation.md) | Docker, native build, and package install |
| [Architecture](getting-started/architecture.md) | Extension architecture and components |
| [Troubleshooting](getting-started/troubleshooting.md) | Common issues and fixes |

</details>

<details>
<summary><strong>Reference</strong></summary>

| Document | Description |
|----------|-------------|
| [SQL API](sql-api.md) | Functions, operators, and aggregates |
| [Data types](reference/data-types.md) | Vector and related types |
| [Configuration](configuration.md) | GUC variables and tuning |
| [API stability](reference/api-stability.md) | Stability and deprecation |

</details>

<details>
<summary><strong>Vector search & indexing</strong></summary>

| Document | Description |
|----------|-------------|
| [Vector types](vector-search/vector-types.md) | `vector`, `vectorp`, `vecmap`, `vgraph`, `rtext`, `halfvec`, `binaryvec`, `sparsevec` |
| [Indexing](vector-search/indexing.md) | HNSW, IVF (index access methods); PQ/OPQ are quantization |
| [Distance metrics](vector-search/distance-metrics.md) | L2, cosine, inner product |
| [Quantization](vector-search/quantization.md) | PQ, OPQ, and compression |

</details>

<details>
<summary><strong>ML algorithms</strong></summary>

| Document | Description |
|----------|-------------|
| [Gradient boosting](ml-algorithms/gradient-boosting.md) | XGBoost, LightGBM, CatBoost |
| [Random forest](ml-algorithms/random-forest.md) | Classification and regression |
| [Clustering](ml-algorithms/clustering.md) | K-Means, DBSCAN, GMM |
| [Classification](ml-algorithms/classification.md) | SVM, logistic regression, Naive Bayes |
| [Regression](ml-algorithms/regression.md) | Linear, Ridge, Lasso |
| [Outlier detection](ml-algorithms/outlier-detection.md) | Z-score, IQR |
| [Time series](ml-algorithms/time-series.md) | Forecasting |
| [Recommendation systems](ml-algorithms/recommendation-systems.md) | Collaborative filtering |

</details>

<details>
<summary><strong>Embeddings & RAG</strong></summary>

| Document | Description |
|----------|-------------|
| [Embedding generation](ml-embeddings/embedding-generation.md) | Text, image, and batch embeddings |
| [Model inference](ml-embeddings/model-inference.md) | ONNX runtime and models |
| [RAG overview](rag/overview.md) | End-to-end RAG pipeline |
| [LLM integration](rag/llm-integration.md) | Providers and configuration |
| [Document processing](rag/document-processing.md) | Chunking and ingestion |

</details>

<details>
<summary><strong>GPU & performance</strong></summary>

| Document | Description |
|----------|-------------|
| [GPU feature matrix](gpu/gpu-feature-matrix.md) | CUDA, ROCm, Metal support |
| [CUDA support](gpu/cuda-support.md) | NVIDIA GPU setup |
| [ROCm support](gpu/rocm-support.md) | AMD GPU setup |
| [Metal support](gpu/metal-support.md) | Apple Silicon setup |
| [SIMD optimization](performance/simd-optimization.md) | CPU vectorization |

</details>

<details>
<summary><strong>Deployment & operations</strong></summary>

| Document | Description |
|----------|-------------|
| [Docker](deployment/docker-unified.md) | Build and run with Docker Compose |
| [Container images](deployment/container-images.md) | Image names and tags |
| [Production install](deployment/production-install.md) | Production and Kubernetes |
| [Backup and restore](deployment/backup-restore.md) | Data backup and recovery |
| [Observability](operations/observability-setup.md) | Monitoring and metrics |
| [Troubleshooting](operations/troubleshooting.md) | Operations runbooks |

</details>

<details>
<summary><strong>Development</strong></summary>

| Document | Description |
|----------|-------------|
| [Development guide](development/development-guide.md) | Code layout and adding features |
| [Build system](development/build-system.md) | Makefiles and build targets |
| [Testing with Docker](readme-docker.md) | Running tests against Docker Postgres |

</details>

<details>
<summary><strong>Internals</strong></summary>

| Document | Description |
|----------|-------------|
| [Index methods](internals/index-methods.md) | HNSW, IVF implementation |
| [Vector contract](vector/vector-contract.md) | Vector type contracts and invariants |

</details>

---

## Full index

For a flat index of all documents, see [Documentation index](documentation-index.md).

---

<div align="center">

[Back to top](#neurondb-documentation) · [README](../../README.md) · [QUICKSTART](../../QUICKSTART.md)

</div>
