# Release Process

## Overview

This document describes the release process for NeuronDB ecosystem components.

## Versioning

All components follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Component Versions

Components are versioned independently but typically released together:
- NeuronDB: Extension version (e.g., 1.0.0)
- NeuronAgent: Go module version (e.g., 1.0.0)
- NeuronMCP: Go module version (e.g., 1.0.0)
- NeuronDesktop: Frontend package version (e.g., 2.0.0)

## Release Checklist

### Pre-Release

- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version numbers updated
- [ ] Security scan completed
- [ ] Performance benchmarks run

### Release

- [ ] Create release branch: `release/v1.0.0`
- [ ] Run release script: `./scripts/release.sh 1.0.0`
- [ ] Build and tag Docker images
- [ ] Generate SBOMs for all images
- [ ] Sign images with cosign
- [ ] Push images to registry
- [ ] Create GitHub release
- [ ] Publish SDKs to package registries

### Post-Release

- [ ] Update documentation site
- [ ] Announce release
- [ ] Monitor for issues
- [ ] Update compatibility matrix

## Release Script

The release script automates the release process:

```bash
# Dry run
./scripts/release.sh 1.0.0 --dry-run

# Actual release
./scripts/release.sh 1.0.0
```

### What the Script Does

1. **Version Updates**: Updates version in all component files
2. **Image Building**: Builds Docker images for all components
3. **SBOM Generation**: Generates Software Bill of Materials
4. **Image Signing**: Signs images with cosign
5. **Manifest Creation**: Creates release manifest

## Artifacts

### Docker Images

All images are published to GitHub Container Registry:
- `ghcr.io/neurondb/neurondb-postgres:<version>`
- `ghcr.io/neurondb/neuronagent:<version>`
- `ghcr.io/neurondb/neuronmcp:<version>`
- `ghcr.io/neurondb/neurondesktop-api:<version>`
- `ghcr.io/neurondb/neurondesktop-frontend:<version>`

### SBOMs

Software Bill of Materials in SPDX format:
- `releases/<version>/neurondb.sbom.json`
- `releases/<version>/neuronagent.sbom.json`
- `releases/<version>/neuronmcp.sbom.json`

### Release Manifest

JSON manifest with version information:
- `releases/<version>/manifest.json`

## Image Signing

Images are signed using [cosign](https://github.com/sigstore/cosign):

```bash
# Generate key pair
cosign generate-key-pair

# Sign image
cosign sign --key cosign.key ghcr.io/neurondb/neurondb-postgres:1.0.0

# Verify signature
cosign verify --key cosign.pub ghcr.io/neurondb/neurondb-postgres:1.0.0
```

## CI/CD Integration

### GitHub Actions

Release workflow triggers on:
- Tag push: `v*`
- Manual workflow dispatch

### Steps

1. **Build**: Build all Docker images
2. **Test**: Run integration tests
3. **SBOM**: Generate SBOMs
4. **Sign**: Sign images
5. **Push**: Push to registry
6. **Release**: Create GitHub release

## Compatibility Matrix

Each release includes a compatibility matrix:

```json
{
  "compatibility": {
    "postgresql": ["16", "17", "18"],
    "go": ["1.21", "1.22", "1.23", "1.24"],
    "node": ["18", "20", "22"]
  }
}
```

## Rollback Procedure

If a release has critical issues:

1. **Stop Promotion**: Don't promote to `latest` tag
2. **Identify Issue**: Document the problem
3. **Create Hotfix**: Create patch release if needed
4. **Communicate**: Notify users of issue and fix

## Release Notes Template

```markdown
# NeuronDB v1.0.0 Release Notes

## What's New

- Feature 1
- Feature 2

## Improvements

- Improvement 1
- Improvement 2

## Bug Fixes

- Fix 1
- Fix 2

## Breaking Changes

- Change 1
- Change 2

## Upgrade Guide

[Upgrade instructions]

## Compatibility

- PostgreSQL: 16, 17, 18
- Go: 1.21+
- Node: 18+

## Downloads

- Docker Images: [links]
- SDKs: [links]
```







