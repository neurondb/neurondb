# Container Images

NeuronDB publishes official container images to GitHub Container Registry (GHCR).

## Image Names and Tags

### NeuronDB (PostgreSQL Extension)

Base images with NeuronDB extension pre-installed:

- `ghcr.io/neurondb/neurondb-postgres:{version}-pg{16|17|18}-{cpu|cuda|rocm|metal}`

**Examples:**
- `ghcr.io/neurondb/neurondb-postgres:v1.0.0-pg17-cpu`
- `ghcr.io/neurondb/neurondb-postgres:v1.0.0-pg17-cuda`
- `ghcr.io/neurondb/neurondb-postgres:v1.0.0-pg16-rocm`
- `ghcr.io/neurondb/neurondb-postgres:v1.0.0-pg18-metal`

**Tag format:**
- `{version}` - Release version (e.g., `v1.0.0`, `v1.1.0`)
- `{pg_version}` - PostgreSQL major version (16, 17, 18)
- `{backend}` - GPU backend (cpu, cuda, rocm, metal)

**Latest tags:**
- `ghcr.io/neurondb/neurondb-postgres:latest-pg17-cpu` (points to latest stable release)

## Using Published Images

### Pull and Use in docker-compose.yml

Update `docker-compose.yml` to use published images:

```yaml
services:
  neurondb:
    image: ghcr.io/neurondb/neurondb-postgres:v1.0.0-pg17-cpu
    # ... other configuration (see docker/docker-compose.yml)
```

### Pull Manually

```bash
# Pull specific version
docker pull ghcr.io/neurondb/neurondb-postgres:v1.0.0-pg17-cpu

# Pull latest
docker pull ghcr.io/neurondb/neurondb-postgres:latest-pg17-cpu
```

### Authentication

GHCR images are public for releases. To pull:

1. **No authentication required** for public releases
2. **For private/nightly images**: Authenticate with GitHub:

```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
```

## Image Digests

For production deployments, use image digests for reproducibility:

```yaml
services:
  neurondb:
    image: ghcr.io/neurondb/neurondb-postgres:v1.0.0-pg17-cpu@sha256:abc123...
```

Find digests in:
- GitHub Releases (release notes)
- GHCR package pages: `https://github.com/neurondb/neurondb/pkgs/container/neurondb-postgres`
- `docker inspect ghcr.io/neurondb/neurondb-postgres:v1.0.0-pg17-cpu | jq '.[0].RepoDigests'`

## Nightly Builds

Nightly builds are available for the `main` branch:

- `ghcr.io/neurondb/neurondb-postgres:nightly-pg17-cpu`

> [!WARNING]
> Nightly builds are for testing only and may be unstable.

## Multi-Architecture Support

Images are built for:
- `linux/amd64` (default)
- `linux/arm64` (when available)

Use platform-specific tags or let Docker auto-select:

```bash
docker pull --platform linux/arm64 ghcr.io/neurondb/neurondb-postgres:v1.0.0-pg17-cpu
```

## Building Locally

If you need to build images locally, see:
- [`docker/README.md`](../docker/README.md)
- [`Docs/deployment/package.md`](package.md)

## Related Documentation

- [Docker Deployment Guide](docker.md)
- [Quick Start Guide](../../QUICKSTART.md)
- [Release Notes](../../CHANGELOG.md)


