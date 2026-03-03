# 📚 NeuronDB Complete Documentation Index

<div align="center">

**Complete index of all documentation in the NeuronDB ecosystem.**

[![Version](https://img.shields.io/badge/version-2.0-blue)](.)
[![Last Updated](https://img.shields.io/badge/updated-2026--02--26-lightgrey)](.)

</div>

---

## 🧭 Quick Navigation

- [Getting Started](#-getting-started)
- [Reference Documentation](#-reference-documentation)
- [Internals Documentation](#-internals-documentation)
- [Advanced Features](#-advanced-features)
- [Development](#-development)
- [Deployment](#-deployment)
- [Ecosystem Integration](#-ecosystem-integration)

---

## 🚀 Getting Started

### Quick Start Guides

| Guide | Description | Time | Difficulty |
|-------|-------------|------|------------|
| **[QUICKSTART.md](../../QUICKSTART.md)** | Get all services running in minutes | 5-10 min | ⭐ Easy |
| **[Simple Start Guide](getting-started/simple-start.md)** | Beginner-friendly setup | 10 min | ⭐ Easy |
| **[Architecture Overview](getting-started/architecture.md)** | Understand the architecture | 15 min | ⭐ Easy |
| **[Troubleshooting](getting-started/troubleshooting.md)** | Common issues and solutions | - | ⭐ Easy |

---

## 📚 Reference Documentation

### SQL API

<details>
<summary><strong>📊 Complete SQL API Reference</strong></summary>

- **[SQL API Reference](../../NeuronDB/docs/sql-api.md)** - All 665+ SQL functions, types, operators, and aggregates
  - ✅ Vector operations
  - ✅ Distance metrics
  - ✅ Quantization functions
  - ✅ Indexing functions
  - ✅ Embedding generation
  - ✅ Hybrid search
  - ✅ Reranking
  - ✅ Machine learning
  - ✅ RAG functions
  - ✅ LLM functions
  - ✅ Utility functions

</details>

### Data Types

<details>
<summary><strong>🔢 Data Types Reference</strong></summary>

- **[Data Types Complete Reference](reference/data-types.md)** - All data types with C structures
  - ✅ Vector types (vector, halfvec, sparsevec, binaryvec, etc.)
  - ✅ Internal C structures
  - ✅ Type storage formats
  - ✅ Type casting rules
  - ✅ Memory layout
  - ✅ Quantization formats

</details>

### Configuration

<details>
<summary><strong>⚙️ Configuration Reference</strong></summary>

- **[Configuration Reference](../../NeuronDB/docs/configuration.md)** - All GUC variables
  - ✅ Core/index settings
  - ✅ GPU settings
  - ✅ LLM settings
  - ✅ Worker settings
  - ✅ ONNX Runtime settings
  - ✅ Quota settings
  - ✅ AutoML settings

</details>

### Component APIs

<details>
<summary><strong>🔌 Component API References</strong></summary>

| Component | Documentation | Description |
|-----------|---------------|-------------|
| **NeuronAgent** | [API Reference](reference/neuronagent-api.md) | REST and WebSocket API |
| **NeuronMCP** | [Tools Reference](https://github.com/neurondb/neuron-mcp) | All 100+ MCP tools |
| **NeuronDesktop** | [API Reference](reference/api-reference.md#neurondesktop-api) | REST and WebSocket API |

</details>

---

## 🔍 Internals Documentation

### Architecture

<details>
<summary><strong>🏗️ Architecture Documentation</strong></summary>

| Document | Description | Status |
|----------|-------------|--------|
| **[Architecture Overview](getting-started/architecture.md)** | System architecture overview | ✅ Complete |
| **[NeuronDB Documentation](../../NeuronDB/docs/)** | Complete NeuronDB extension documentation | ✅ Complete |
| **[NeuronAgent Architecture](internals/neuronagent-architecture.md)** | Agent runtime architecture | ✅ Complete |
| **[NeuronDesktop Frontend](internals/neurondesktop-frontend.md)** | Frontend architecture | ✅ Complete |

</details>

### Index Methods

<details>
<summary><strong>📇 Index Methods Reference</strong></summary>

- **[Index Methods Complete Reference](internals/index-methods.md)** - All index types
  - ✅ HNSW index
  - ✅ IVF index
  - ✅ Hybrid index
  - ✅ Temporal index
  - ✅ Sparse index
  - ✅ Index tuning
  - ✅ Index maintenance

</details>

---

## ⚡ Advanced Features

### GPU Acceleration

<details>
<summary><strong>🎮 GPU Acceleration Documentation</strong></summary>

| Platform | Documentation | Status |
|----------|---------------|--------|
| **GPU Feature Matrix** | [gpu_feature_matrix.md](gpu/gpu-feature-matrix.md) | ✅ Complete |
| **CUDA Support** | [CUDA Support](../../NeuronDB/docs/gpu/cuda-support.md) | ✅ Complete |
| **ROCm Support** | [ROCm Support](../../NeuronDB/docs/gpu/rocm-support.md) | ✅ Complete |
| **Metal Support** | [Metal Support](../../NeuronDB/docs/gpu/metal-support.md) | ✅ Complete |
| **Auto-Detection** | [Auto-Detection](../../NeuronDB/docs/gpu/auto-detection.md) | ✅ Complete |

</details>

### Machine Learning

<details>
<summary><strong>🤖 ML Algorithms Documentation</strong></summary>

| Category | Documentation | Algorithms |
|----------|---------------|------------|
| **Clustering** | [Clustering](../../NeuronDB/docs/ml-algorithms/clustering.md) | K-Means, DBSCAN, GMM, Hierarchical |
| **Classification** | [Classification](../../NeuronDB/docs/ml-algorithms/classification.md) | Random Forest, Logistic Regression, SVM, etc. |
| **Regression** | [Regression](../../NeuronDB/docs/ml-algorithms/regression.md) | Linear, Ridge, Lasso |
| **Random Forest** | [Random Forest](../../NeuronDB/docs/ml-algorithms/random-forest.md) | Classification and regression |
| **Gradient Boosting** | [Gradient Boosting](../../NeuronDB/docs/ml-algorithms/gradient-boosting.md) | XGBoost, LightGBM, CatBoost |
| **Outlier Detection** | [Outlier Detection](../../NeuronDB/docs/ml-algorithms/outlier-detection.md) | Z-score, Modified Z-score, IQR |
| **Time Series** | [Time Series](../../NeuronDB/docs/ml-algorithms/time-series.md) | ARIMA |
| **Recommendation Systems** | [Recommendation Systems](../../NeuronDB/docs/ml-algorithms/recommendation-systems.md) | Recommendation algorithms |

</details>

### RAG Pipeline

<details>
<summary><strong>📄 RAG Pipeline Documentation</strong></summary>

| Topic | Documentation |
|-------|---------------|
| **RAG Overview** | [RAG Overview](../../NeuronDB/docs/rag/overview.md) |
| **Document Processing** | [Document Processing](../../NeuronDB/docs/rag/document-processing.md) |
| **LLM Integration** | [LLM Integration](../../NeuronDB/docs/rag/llm-integration.md) |
| **Vector Search** | [Vector Search](../../NeuronDB/docs/vector-search/) |
| **Hybrid Search** | [Hybrid Search](../../NeuronDB/docs/hybrid-search/) |
| **Reranking** | [Reranking](../../NeuronDB/docs/reranking/) |

</details>

---

## 💻 Development

### Build System

<details>
<summary><strong>🔨 Build System Documentation</strong></summary>

- **[Build System Documentation](development/build-system.md)** - Complete build system
  - ✅ Makefile structure
  - ✅ Build targets
  - ✅ Platform-specific builds
  - ✅ GPU backend compilation
  - ✅ Dependency management
  - ✅ Testing infrastructure

</details>

### Development Guide

<details>
<summary><strong>📝 Development Procedures</strong></summary>

- **[Development Guide](development/development-guide.md)** - Development procedures
  - ✅ Code organization
  - ✅ Adding new SQL functions
  - ✅ Adding new ML algorithms
  - ✅ Adding new tools
  - ✅ Testing procedures
  - ✅ Debugging guides

</details>

---

## 🚢 Deployment

<details>
<summary><strong>📦 Deployment Documentation</strong></summary>

| Document | Description | Difficulty |
|----------|-------------|------------|
| **[Deployment Documentation](deployment/README.md)** | Complete deployment guide | ⭐⭐ Medium |
| **[Production Installation](deployment/production-install.md)** | Production setup | ⭐⭐ Medium |
| **[Docker Deployment](deployment/docker.md)** | Docker deployment (all profiles) | ⭐ Easy |
| **[Kubernetes/Helm](deployment/kubernetes-helm.md)** | Kubernetes deployment (CNPG, Pooler, backup, local test) | ⭐⭐⭐ Advanced |
| **[Container Images](deployment/container-images.md)** | Container image information | ⭐ Easy |
| **[Backup and Restore](deployment/backup-restore.md)** | Backup and recovery (CNPG + legacy) | ⭐ Easy |
| **[Upgrade and Rollback](deployment/upgrade-rollback.md)** | Upgrade procedures | ⭐⭐ Medium |
| **[Sizing Guide](deployment/sizing-guide.md)** | Resource sizing recommendations | ⭐ Easy |
| **[HA Architecture](deployment/ha-architecture.md)** | High availability (CNPG + Patroni) | ⭐⭐⭐ Advanced |

</details>

---

## 🌐 Ecosystem Integration

<details>
<summary><strong>🔗 Integration Documentation</strong></summary>

| Document | Description |
|----------|-------------|
| **[Ecosystem Integration Guide](ecosystem/integration.md)** | Integration guide |
| **[Ecosystem Overview](ecosystem/readme.md)** | How components work together |
| **[Directory structure](directory-structure.md)** | Repo layout (pge/: neurondb, neuron-agent, neuron-desktop, neuron-mcp) |

**Topics covered:**
- ✅ Component communication
- ✅ Data flow
- ✅ Authentication
- ✅ Configuration sharing
- ✅ Deployment coordination
- ✅ Integration examples

</details>

---

## 📊 Documentation Statistics

### Coverage

| Category | Count | Status |
|----------|-------|--------|
| **SQL Functions** | 665+ | ✅ Documented |
| **Data Types** | 8+ | ✅ Documented |
| **Configuration Options** | 30+ | ✅ Documented |
| **API Endpoints** | 50+ | ✅ Documented |
| **MCP Tools** | 100+ | ✅ Documented |
| **ML Algorithms** | 19 | ✅ Documented |
| **Index Methods** | 5 | ✅ Documented |

### Documentation Files

| Category | Count | Location |
|----------|-------|----------|
| **Reference** | 6 files | `Docs/reference/` |
| **Internals** | 4 files | `Docs/internals/` |
| **Advanced** | 3 files | `Docs/advanced/` |
| **Development** | 2 files | `Docs/development/` |
| **Deployment** | 1 file | `Docs/deployment/` |
| **Ecosystem** | 1 file | `Docs/ecosystem/` |

**Total:** 17 comprehensive documentation files

---

## 🔗 Related Documentation

| Document | Description |
|----------|-------------|
| **[Main Documentation Index](documentation.md)** | Original documentation index |
| **[Contributing Guide](../../CONTRIBUTING.md)** | Contribution guidelines |
| **[README](../../README.md)** | Project overview |

---

<div align="center">

**Last Updated:** 2026-02-26  
**Documentation Version:** 2.0.0

[⬆ Back to Top](#-neurondb-complete-documentation-index) · [📚 Main Documentation](documentation.md)

</div>
