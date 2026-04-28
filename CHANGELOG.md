# Changelog

All notable changes to NeuronDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [4.0.0-devel] - 2026-04-28

### Changed
- Main branch uses `default_version` **4.0.0-devel** in `neurondb.control`
- Added `neurondb--4.0.0-devel.sql` (generated from 3.1.0 body) and upgrade path `neurondb--3.1.0--4.0.0-devel.sql`
- Helm chart, Docker image defaults, and documentation versions aligned to **4.0.0-devel**

## [3.1.0] - 2026-04-28

### Added
- LLM model registry schema and upgrade path from 3.0.0-devel via `neurondb--3.0.0-devel--3.1.0.sql`
- Consolidated install script `neurondb--3.1.0.sql` (core SQL plus LLM schema)

### Changed
- `default_version` set to **3.1.0** in `neurondb.control`
- Helm chart, Docker image defaults, and documentation versions aligned to **3.1.0**

## [3.0.0-devel] - Development (historical)

### Changed
- Version updated to 3.0.0-devel for main branch development
- Extension default version updated to 3.0
- All package versions, Docker images, and Helm charts updated to 3.0.0-devel

## [1.0.0] - TBD

### Added
- PostgreSQL extension for vector search (HNSW, IVF indexes)
- ~650+ SQL functions
- 25+ ML algorithm families (classification, regression, clustering)
- GPU acceleration (CUDA, ROCm, Metal)
- Embedding generation and RAG pipelines
- Hybrid search (vector + full-text search)
- NeuronAgent: REST/WebSocket API for agent runtime
- NeuronMCP: MCP protocol server with 650+ tools
- NeuronDesktop: Web UI for ecosystem management
- Benchmark suite (Vector, Hybrid, RAG)
- Comprehensive documentation

### Supported Platforms
- PostgreSQL: 16, 17, 18
- GPU Backends: CPU, CUDA, ROCm, Metal
- Operating Systems: Ubuntu 20.04+, RHEL/CentOS 8+, macOS 13+
- Architectures: linux/amd64, linux/arm64

---

## Release Notes Format

Each release includes:

- **Version**: Semantic version (MAJOR.MINOR.PATCH)
- **Date**: Release date
- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security fixes

See [RELEASE.md](RELEASE.md) for release process documentation.
