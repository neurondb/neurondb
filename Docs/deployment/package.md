# Package-Based Docker Builds

<div align="center">

**Using RPM/Debian packages for building Docker images**

[![Packages](https://img.shields.io/badge/packages-DEB%20%7C%20RPM-blue)](.)
[![Status](https://img.shields.io/badge/status-stable-brightgreen)](.)

</div>

---

> [!TIP]
> Package-based builds are faster than source builds. They produce smaller images and are more reproducible.

---

## Overview

Use pre-built DEB packages instead of building from source in Docker containers.

**Advantages:**

| Advantage | Description |
|-----------|-------------|
| **Faster builds** | No compilation in Docker |
| **More reproducible** | Same packages across environments |
| **Better versioning** | Packages are versioned and distributed |
| **Smaller images** | No build dependencies in final image |

## Available Package-Based Dockerfiles

1. **NeuronDB**: `dockers/neurondb/Dockerfile.package`
2. **NeuronAgent**: `dockers/neuronagent/Dockerfile.package`
3. **NeuronMCP**: `dockers/neuronmcp/Dockerfile.package`

## Building Package-Based Images

### NeuronDB

```bash
# Build from repository root
cd /path/to/neurondb

# Build with PostgreSQL 18
docker build \
  -f dockers/neurondb/Dockerfile.package \
  --build-arg PG_MAJOR=18 \
  --build-arg PACKAGE_VERSION=1.0.0.beta \
  -t neurondb:package-pg18 \
  .
```

### NeuronAgent

```bash
docker build \
  -f dockers/neuronagent/Dockerfile.package \
  --build-arg PACKAGE_VERSION=1.0.0.beta \
  -t neuronagent:package \
  .
```

### NeuronMCP

```bash
docker build \
  -f dockers/neuronmcp/Dockerfile.package \
  --build-arg PACKAGE_VERSION=1.0.0.beta \
  -t neuronmcp:package \
  .
```

## How It Works

1. The Dockerfile copies the `packaging/` directory into the container
2. It runs the appropriate `build.sh` script to create the DEB package
3. The DEB package is installed using `dpkg`
4. The packaging directory is cleaned up

## Prerequisites

The package build scripts require:
- **NeuronDB**: PostgreSQL development packages, build tools
- **NeuronAgent**: Go 1.23+, build tools
- **NeuronMCP**: Go 1.23+, build tools

These are installed automatically in the Dockerfile.

## Integration with docker-compose

To use package-based builds in docker-compose, update the `build.dockerfile` field:

```yaml
services:
  neurondb:
    build:
      context: .
      dockerfile: dockers/neurondb/Dockerfile.package
      args:
        PG_MAJOR: 18
        PACKAGE_VERSION: 1.0.0.beta
```

## Building Packages Separately

You can also build packages outside Docker and then copy them in:

```bash
# Build NeuronDB package
cd packaging/deb/neurondb
VERSION=1.0.0.beta ./build.sh

# Then use a simpler Dockerfile that just installs the .deb file
```

## Comparison: Source vs Package Builds

| Aspect | Source Build | Package Build |
|--------|-------------|----------------|
| Build Time | Slower (compiles in Docker) | Faster (pre-built) |
| Image Size | Larger (includes build deps) | Smaller (runtime only) |
| Reproducibility | Depends on build environment | High (same package) |
| Flexibility | Can modify source | Uses fixed package version |

## Notes

> [!NOTE]
> Package-based builds require the `packaging/` directory to be present. Build scripts automatically detect PostgreSQL versions.

- Package-based builds require the `packaging/` directory
- Build scripts automatically detect PostgreSQL versions
- All four components have complete DEB packaging
- RPM packaging is available for NeuronDB and NeuronAgent

---

## 🔗 Related Documentation

| Document | Description |
|----------|-------------|
| **[Docker Deployment](docker.md)** | Docker deployment guide |
| **[Container Images](container-images.md)** | Container image information |
| **[Packaging Guide](../../packaging/README.md)** | Packaging documentation |

---

<div align="center">

[⬆ Back to Top](#package-based-docker-builds) · [📚 Deployment Index](README.md) · [📚 Main Documentation](../../README.md)

</div>

