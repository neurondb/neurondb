# NeuronDB Complete Documentation Index

<div align="center">

**Complete index of all documentation in the NeuronDB ecosystem.**

[![Version](https://img.shields.io/badge/version-2.0-blue)](.)
[![Last Updated](https://img.shields.io/badge/updated-2026--02--26-lightgrey)](.)

</div>

---

## Quick Navigation

- [Getting Started](#getting-started)
- [Reference Documentation](#reference-documentation)
- [Internals Documentation](#internals-documentation)
- [Advanced Features](#advanced-features)
- [Development](#development)
- [Deployment](#deployment)

---

## Getting Started

### Quick Start Guides

| Guide | Description | Time | Difficulty |
|-------|-------------|------|------------|
| **[QUICKSTART.md](../../QUICKSTART.md)** | Get NeuronDB running quickly | 5-10 min | Easy |
| **[Simple Start Guide](getting-started/simple-start.md)** | Beginner-friendly setup | 10 min | Easy |
| **[Architecture Overview](getting-started/architecture.md)** | Understand the architecture | 15 min | Easy |
| **[Troubleshooting](getting-started/troubleshooting.md)** | Common issues and solutions | - | Easy |

---

## Reference Documentation

### SQL API

<details>
<summary><strong>Complete SQL API Reference</strong></summary>

- **[SQL API Reference](sql-api.md)** - ~650+ SQL functions, types, operators, and aggregates
  - Vector operations
  - Distance metrics
  - Quantization functions
  - Indexing functions
  - Embedding generation
  - Hybrid search
  - Reranking
  - Machine learning
  - RAG functions
  - LLM functions
  - Utility functions

</details>

### Data Types

<details>
<summary><strong>Data Types Reference</strong></summary>

- **[Data Types Complete Reference](reference/data-types.md)** - All data types with C structures
  - Vector types (vector, halfvec, sparsevec, binaryvec, etc.)
  - Internal C structures
  - Type storage formats
  - Type casting rules
  - Memory layout
  - Quantization formats

</details>

### Configuration

<details>
<summary><strong>Configuration Reference</strong></summary>

- **[Configuration Reference](configuration.md)** - All GUC variables
  - Core/index settings
  - GPU settings
  - LLM settings
  - Worker settings
  - ONNX Runtime settings
  - Quota settings
  - AutoML settings

</details>

---

## Internals Documentation

### Architecture

<details>
<summary><strong>Architecture Documentation</strong></summary>

| Document | Description | Status |
|----------|-------------|--------|
| **[Architecture Overview](getting-started/architecture.md)** | Extension architecture overview | Complete |
| **[Index Methods](internals/index-methods.md)** | HNSW, IVF, and index implementation | Complete |
| **[Vector contract](vector/vector-contract.md)** | Vector type contracts and invariants | Complete |

</details>

### Index Methods

<details>
<summary><strong>Index Methods Reference</strong></summary>

- **[Index Methods Reference](internals/index-methods.md)** - Index access methods and implementation
  - HNSW index
  - IVF index
  - Index tuning
  - Index maintenance

Hybrid search and temporal search are implemented as query-level functions (e.g. `hybrid_search`), not as separate index types. The only index access methods are HNSW and IVF.

</details>

---

## Advanced Features

### GPU Acceleration

<details>
<summary><strong>GPU Acceleration Documentation</strong></summary>

| Platform | Documentation | Status |
|----------|---------------|--------|
| **GPU Feature Matrix** | [gpu-feature-matrix.md](gpu/gpu-feature-matrix.md) | Complete |
| **CUDA Support** | [cuda-support.md](gpu/cuda-support.md) | Complete |
| **ROCm Support** | [rocm-support.md](gpu/rocm-support.md) | Complete |
| **Metal Support** | [metal-support.md](gpu/metal-support.md) | Complete |
| **Auto-Detection** | [auto-detection.md](gpu/auto-detection.md) | Complete |

</details>

### Machine Learning

<details>
<summary><strong>ML Algorithms Documentation</strong></summary>

| Category | Documentation | Algorithms |
|----------|---------------|------------|
| **Clustering** | [clustering.md](ml-algorithms/clustering.md) | K-Means, DBSCAN, GMM, Hierarchical |
| **Classification** | [classification.md](ml-algorithms/classification.md) | Random Forest, Logistic Regression, SVM, etc. |
| **Regression** | [regression.md](ml-algorithms/regression.md) | Linear, Ridge, Lasso |
| **Random Forest** | [random-forest.md](ml-algorithms/random-forest.md) | Classification and regression |
| **Gradient Boosting** | [gradient-boosting.md](ml-algorithms/gradient-boosting.md) | XGBoost, LightGBM, CatBoost |
| **Outlier Detection** | [outlier-detection.md](ml-algorithms/outlier-detection.md) | Z-score, Modified Z-score, IQR |
| **Time Series** | [time-series.md](ml-algorithms/time-series.md) | ARIMA |
| **Recommendation Systems** | [recommendation-systems.md](ml-algorithms/recommendation-systems.md) | Recommendation algorithms |

</details>

### RAG Pipeline

<details>
<summary><strong>RAG Pipeline Documentation</strong></summary>

| Topic | Documentation |
|-------|---------------|
| **RAG Overview** | [overview.md](rag/overview.md) |
| **Document Processing** | [document-processing.md](rag/document-processing.md) |
| **LLM Integration** | [llm-integration.md](rag/llm-integration.md) |
| **Vector Search** | [vector-search/](vector-search/) |
| **Hybrid Search** | [hybrid-search/](hybrid-search/) |
| **Reranking** | [reranking/](reranking/) |

</details>

---

## Development

### Build System

<details>
<summary><strong>Build System Documentation</strong></summary>

- **[Build System Documentation](development/build-system.md)** - Complete build system
  - Makefile structure
  - Build targets
  - Platform-specific builds
  - GPU backend compilation
  - Dependency management
  - Testing infrastructure

</details>

### Development Guide

<details>
<summary><strong>Development Procedures</strong></summary>

- **[Development Guide](development/development-guide.md)** - Development procedures
  - Code organization
  - Adding new SQL functions
  - Adding new ML algorithms
  - Testing procedures
  - Debugging guides

</details>

---

## Deployment

<details>
<summary><strong>Deployment Documentation</strong></summary>

| Document | Description | Difficulty |
|----------|-------------|------------|
| **[Deployment overview](deployment/docker.md)** | Docker and deployment guide | Medium |
| **[Production Installation](deployment/production-install.md)** | Production setup | Medium |
| **[Docker Deployment](deployment/docker.md)** | Docker deployment (all profiles) | Easy |
| **[Kubernetes/Helm](deployment/kubernetes-helm.md)** | Kubernetes deployment (CNPG, Pooler, backup, local test) | Advanced |
| **[Container Images](deployment/container-images.md)** | Container image information | Easy |
| **[Backup and Restore](deployment/backup-restore.md)** | Backup and recovery (CNPG + legacy) | Easy |
| **[Upgrade and Rollback](deployment/upgrade-rollback.md)** | Upgrade procedures | Medium |
| **[Sizing Guide](deployment/sizing-guide.md)** | Resource sizing recommendations | Easy |
| **[HA Architecture](deployment/ha-architecture.md)** | High availability (CNPG + Patroni) | Advanced |

</details>

---

## Related documentation

| Document | Description |
|----------|-------------|
| **[Main entry](readme.md)** | Documentation map and quick links |
| **[Contributing](../../CONTRIBUTING.md)** | Contribution guidelines |
| **[README](../../README.md)** | Project overview |

---

<div align="center">

[Back to top](#neurondb-complete-documentation-index) · [Main Documentation](readme.md)

</div>
