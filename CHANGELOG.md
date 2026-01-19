# Changelog

All notable changes to NeuronDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release preparation
- GitHub Container Registry (GHCR) image publishing
- DEB and RPM package builds
- Comprehensive documentation

## [2.1.0] - 2026-01-19

### Added
- Prometheus alerting configuration for monitoring
- TypeScript SDK for NeuronAgent
- Unified build script for copying module files to bin directory
- Enhanced embedding generation and RAG demo queries

### Changed
- Updated version identifiers across all components to 2.1.0
- Enhanced MCP server module with improved capabilities
- Improved frontend components and API integration in NeuronDesktop
- Expanded agent capabilities and reliability features in NeuronAgent
- Enhanced tool system and transport capabilities in NeuronMCP
- Updated workflow configurations and chart metadata
- Updated SDK documentation and code examples
- Updated installation and utility scripts
- Updated examples and Docker configuration

### Fixed
- Fixed test documentation file references
- Fixed documentation inconsistencies and configuration errors
- Fixed references to use uppercase README.md

### Improved
- Standardized file formatting across all components
- Enhanced documentation with improved formatting and style
- Updated component guides and quickstart instructions
- Improved documentation structure and organization
- Removed duplicate uppercase documentation files
- Removed generated test and benchmark output artifacts
- Removed obsolete files

## [1.0.0] - TBD

### Added
- PostgreSQL extension for vector search (HNSW, IVF indexes)
- 473+ SQL functions
- 52+ ML algorithms (classification, regression, clustering)
- GPU acceleration (CUDA, ROCm, Metal)
- Embedding generation and RAG pipelines
- Hybrid search (vector + full-text search)
- NeuronAgent: REST/WebSocket API for agent runtime
- NeuronMCP: MCP protocol server with 100+ tools
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

