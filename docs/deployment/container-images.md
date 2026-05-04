# Container images

NeuronDB ships **two** complementary image lines:

| Image | Role | Typical use |
|-------|------|-------------|
| **`neurondb/neurondb`** | CPU PostgreSQL + NeuronDB (Bookworm-based build) | Quickstart, laptops, CI — default for [`scripts/install-docker.sh`](../../scripts/install-docker.sh) |
| **`neurondb/neurondb-cuda`** | CUDA + ONNX + NeuronDB (**linux/amd64**) | GPU inference on NVIDIA hosts |

**CUDA** images are published to **Docker Hub** and **GHCR** on release tags (`v*`). Same NeuronDB version as the git tag (e.g. **`v3.1.0`** → **`3.1.0`** image tags).

## Naming convention (CUDA)

Use the same **namespace / image name** everywhere; only the **registry hostname** changes:

| Part | Value | Notes |
|------|--------|--------|
| Organization | `neurondb` | Docker Hub org / GHCR owner |
| Image name | `neurondb-cuda` | CUDA PostgreSQL + NeuronDB (not `neurondb-postgres`) |
| Full path (no tag) | `neurondb/neurondb-cuda` | Matches `docker pull neurondb/neurondb-cuda` |

Resolved references:

| Registry host | Full image (example tag) |
|---------------|-------------------------|
| Docker Hub (CUDA) | `docker.io/neurondb/neurondb-cuda:latest` |
| GHCR (CUDA) | `ghcr.io/neurondb/neurondb-cuda:latest` |

**Helm** splits this into `registry` + `repository` + `tag` (see [values.yaml](../../src/helm/neurondb/values.yaml)): `registry: ghcr.io`, `repository: neurondb/neurondb-cuda`.

### CPU image (`neurondb/neurondb`)

- **Docker Hub:** `neurondb/neurondb:latest` (and other tags as published).
- **Build locally:** [`docker/neurondb/Dockerfile`](../../docker/neurondb/Dockerfile) — same Dockerfile used for Hub CPU builds when published.

Legacy **`neurondb-postgres`** or older names may appear in historical docs; prefer **`neurondb/neurondb`** (CPU) or **`neurondb/neurondb-cuda`** (GPU) for new deployments.

## Tags (CUDA examples)

PostgreSQL majors **16**, **17**, and **18** each get **`pg<major>`** and **`pg<major>-<NeuronDB version>`** tags. **`latest`** and the bare semver tag (for example **`3.1.0`**) refer to the **PostgreSQL 17** line only.

| Tag | Meaning |
|-----|---------|
| `latest` | PG **17** + latest NeuronDB release from CI |
| `pg16`, `pg17`, `pg18` | Rolling tag for that Postgres major |
| `pg16-3.1.0`, `pg17-3.1.0`, … | Pinned Postgres major + NeuronDB version |
| `3.1.0` | PG **17** + NeuronDB **3.1.0** only (use **`pg16-3.1.0`** / **`pg18-3.1.0`** for other majors) |

**Platform:** published CUDA images are **`linux/amd64`** only.

### Pull (CUDA)

```bash
docker pull neurondb/neurondb-cuda:latest
docker pull ghcr.io/neurondb/neurondb-cuda:latest
```

### Pull (CPU quickstart)

```bash
docker pull neurondb/neurondb:latest
```

Run CUDA with GPU passthrough:

```bash
docker run --gpus all -p 5433:5432 \
  -e POSTGRES_PASSWORD=neurondb \
  neurondb/neurondb-cuda:latest
```

Or use **[`scripts/install-docker.sh`](../../scripts/install-docker.sh)** (defaults to **`neurondb/neurondb:latest`**; pass **`--image neurondb/neurondb-cuda:latest`** for CUDA).

### docker-compose

Point `image:` at `neurondb/neurondb` or `neurondb/neurondb-cuda` plus your tag. For CUDA, configure GPU access per Docker’s Compose + GPU docs for your Compose version.

```yaml
services:
  neurondb:
    image: neurondb/neurondb-cuda:latest
```

## Workflow dispatch builds (GHCR)

The optional workflow **`.github/workflows/neurondb-docker.yml`** can push additional tags (for example `...-pg17-cuda` / `...-pg18-cuda` with a version or `nightly` prefix) when run manually—see that workflow for the exact naming.

## Authentication

Public pulls usually need no login. For private packages or higher rate limits:

```bash
echo "$GITHUB_TOKEN" | docker login ghcr.io -u USERNAME --password-stdin
```

## Digests

Pin by digest in production:

```yaml
image: neurondb/neurondb-cuda:3.1.0@sha256:...
```

Inspect: `docker inspect neurondb/neurondb-cuda:3.1.0 | jq '.[0].RepoDigests'`

## Build locally

- CUDA image: [`docker/neurondb/Dockerfile.gpu.cuda`](../../docker/neurondb/Dockerfile.gpu.cuda)
- CPU image: [`docker/neurondb/Dockerfile`](../../docker/neurondb/Dockerfile)

See [`docker/neurondb/README.md`](../../docker/neurondb/README.md) and [docker.md](../docker.md).

## Related

- [Docker deployment](docker.md)
- [Maintainer release](../release-docker.md)
- [Root README](../../README.md)
