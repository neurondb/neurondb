# Package-Based Docker Builds

Use pre-built DEB (or RPM) packages to build NeuronDB Docker images instead of building from source.

---

## Overview

**Advantages:**
- **Faster builds** — No compilation inside Docker
- **Reproducible** — Same package across environments
- **Smaller images** — No build dependencies in the final image

## NeuronDB package-based image

**Dockerfile:** `docker/neurondb/Dockerfile.package` (if present in the repository).

### Build from repository root

```bash
# Build with PostgreSQL 17 (or set PG_MAJOR=16, 18)
docker build \
  -f docker/neurondb/Dockerfile.package \
  --build-arg PG_MAJOR=17 \
  --build-arg PACKAGE_VERSION=1.0.0.beta \
  -t neurondb:package-pg17 \
  .
```

### Use in Compose

Point the Neurondb service build to the package Dockerfile:

```yaml
services:
  neurondb:
    build:
      context: ..
      dockerfile: docker/neurondb/Dockerfile.package
      args:
        PG_MAJOR: 17
        PACKAGE_VERSION: 1.0.0.beta
```

## Prerequisites

Package build requires the `packaging/` directory (if used) and PostgreSQL development packages; the Dockerfile typically installs build dependencies.

## Related documentation

- [Docker deployment](docker.md)
- [Container images](container-images.md)

---

[Deployment](README.md) · [Documentation](../readme.md)
