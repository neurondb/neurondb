# Versioning Policy

## Overview

This document defines the versioning policy for NeuronDB ecosystem components.

## Semantic Versioning

All components follow [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR** (X.0.0): Incompatible API changes
- **MINOR** (0.X.0): New functionality, backward compatible
- **PATCH** (0.0.X): Bug fixes, backward compatible

## Component Versioning

### NeuronDB Extension

- Versioned in `neurondb.control`
- Format: `MAJOR.MINOR` (e.g., 1.0)
- SQL migration files: `neurondb--FROM--TO.sql`

### NeuronAgent

- Versioned in `go.mod`
- Format: `vMAJOR.MINOR.PATCH` (e.g., v1.0.0)
- API versioning: `/api/v1/`

### NeuronMCP

- Versioned in `go.mod`
- Format: `vMAJOR.MINOR.PATCH` (e.g., v1.0.0)
- MCP protocol version: Independent

### NeuronDesktop

- Frontend: `package.json` version
- API: Go module version
- Format: `MAJOR.MINOR.PATCH` (e.g., 2.0.0)

## Version Synchronization

While components are versioned independently, major releases are typically synchronized:

- **Major Release**: All components bump major version
- **Minor Release**: Components bump independently
- **Patch Release**: Components bump independently

## Compatibility Matrix

Each release includes a compatibility matrix:

| Component | PostgreSQL | Go | Node.js |
|-----------|-----------|-----|---------|
| NeuronDB | 16, 17, 18 | - | - |
| NeuronAgent | 16+ | 1.21+ | - |
| NeuronMCP | 16+ | 1.21+ | - |
| NeuronDesktop | 16+ | 1.21+ | 18+ |

## Deprecation Policy

### Deprecation Timeline

1. **Announcement**: Feature marked as deprecated in release notes
2. **Warning Period**: 2 minor versions (e.g., 1.0 → 1.2)
3. **Removal**: Removed in next major version

### Example

- v1.0.0: Feature deprecated
- v1.1.0: Still available, warnings
- v1.2.0: Still available, warnings
- v2.0.0: Feature removed

## Version Tags

Git tags follow the format:
- `v<version>` (e.g., `v1.0.0`)
- Component-specific: `neurondb-v1.0.0`, `neuronagent-v1.0.0`

## Docker Image Tags

Images are tagged with:
- Version: `ghcr.io/neurondb/component:1.0.0`
- Latest: `ghcr.io/neurondb/component:latest`
- Major: `ghcr.io/neurondb/component:1`
- Minor: `ghcr.io/neurondb/component:1.0`

## Package Versions

### Python SDK

- PyPI: `neurondb==1.0.0`
- Version in `setup.py` or `pyproject.toml`

### TypeScript SDK

- npm: `@neurondb/neuronagent@1.0.0`
- Version in `package.json`

## Release Cadence

- **Major**: As needed (breaking changes)
- **Minor**: Quarterly (new features)
- **Patch**: Monthly (bug fixes)
- **Hotfix**: As needed (critical fixes)

## Version Bumping

### Automatic

- CI/CD bumps patch version on merge to main
- Release script bumps versions during release

### Manual

- Update version files before release
- Run release script to sync versions

## Examples

### Major Release (Breaking Changes)

```
NeuronDB: 1.0.0 → 2.0.0
NeuronAgent: 1.0.0 → 2.0.0
NeuronMCP: 1.0.0 → 2.0.0
NeuronDesktop: 2.0.0 → 3.0.0
```

### Minor Release (New Features)

```
NeuronDB: 1.0.0 → 1.1.0
NeuronAgent: 1.0.0 → 1.1.0
NeuronMCP: 1.0.0 → 1.0.0 (no changes)
NeuronDesktop: 2.0.0 → 2.1.0
```

### Patch Release (Bug Fixes)

```
NeuronDB: 1.0.0 → 1.0.1
NeuronAgent: 1.0.0 → 1.0.1
NeuronMCP: 1.0.0 → 1.0.0 (no changes)
NeuronDesktop: 2.0.0 → 2.0.1
```








