# Release Process

Release process for the NeuronDB extension and related artifacts in this repository.

---

## Versioning

NeuronDB follows [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

Extension version is defined in `neurondb.control`. SQL upgrades use migration files `neurondb--FROM--TO.sql`.

## Release checklist

### Pre-release

- [ ] All tests passing (`make installcheck`)
- [ ] Documentation updated
- [ ] CHANGELOG updated
- [ ] Version updated in control and build
- [ ] Security scan and benchmarks (as applicable)

### Release

- [ ] Create release branch (e.g. `release/v1.0.0`)
- [ ] Run release script if present (e.g. `./scripts/release.sh 1.0.0`)
- [ ] Build and tag Docker image(s) for NeuronDB
- [ ] Push image to registry (e.g. GHCR)
- [ ] Create GitHub release and tag

### Post-release

- [ ] Update docs site if applicable
- [ ] Announce and monitor for issues

## Artifacts

### Docker image

Published to **Docker Hub** and **GHCR** (see [`.github/workflows/docker-publish.yml`](../../.github/workflows/docker-publish.yml)):

- `neurondb/neurondb-cuda` and `ghcr.io/neurondb/neurondb-cuda` — CUDA, PostgreSQL **16 / 17 / 18**, `linux/amd64`

Optional / manual workflow (see `neurondb-docker.yml`) may add extra tag styles; the **default** tag set is documented in [Container images](../deployment/container-images.md).

### SBOM

Software Bill of Materials (if generated):
- `releases/<version>/neurondb.sbom.json`

---

[Release](README.md) · [Documentation](../readme.md)
