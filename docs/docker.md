# NeuronDB and Docker

## Official images

### CPU quickstart (default for the install script)

- **Docker Hub:** `neurondb/neurondb` — default image for the one-line [install script](../scripts/install-docker.sh) (`neurondb/neurondb:latest`). Suitable for laptops and CI without a GPU. Built from [`docker/neurondb/Dockerfile`](../docker/neurondb/Dockerfile) (PostgreSQL on Debian Bookworm; extension bundled). The install script does **not** pass `--gpus` for this image name.

### CUDA GPU (production ML inference)

- **Docker Hub:** `neurondb/neurondb-cuda` — CUDA + ONNX + NeuronDB for **NVIDIA** hosts. Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). The install script adds `docker run --gpus …` when the image reference contains **`neurondb-cuda`**, unless `NEURONDB_DOCKER_GPUS=no` (or `0`).
- **GHCR:** `ghcr.io/neurondb/neurondb-cuda` — same tags as Docker Hub on release builds.

Published **CUDA** images target **PostgreSQL 16, 17, and 18**, **linux/amd64** only. Tags and pinning are described in [deployment/container-images.md](deployment/container-images.md).

### Tags (`neurondb/neurondb-cuda`)

| Tag pattern | Use |
|-------------|-----|
| `latest` | **PG 17** + current NeuronDB release (common default when using `--image neurondb/neurondb-cuda:latest`) |
| `pg16` / `pg17` / `pg18` | Latest CUDA build for that PostgreSQL major |
| `pg16-<version>` … `pg18-<version>` | Pinned to major + release (e.g. `pg16-3.1.0`) |
| `<version>` | **PG 17** + that release (e.g. `3.1.0`); use **`pg16-`** / **`pg18-`** for other majors |

## End-user install

From the repo root README:

```bash
curl -fsSL https://raw.githubusercontent.com/neurondb/neurondb/main/scripts/install-docker.sh | bash
```

Options: `--name`, `--port`, `--image`, `--password`, `--quiet`, `--reset` (see script `--help`).  
Destructive reset requires `NEURONDB_CONFIRM_RESET=yes`.

**CUDA example:**

```bash
curl -fsSL https://raw.githubusercontent.com/neurondb/neurondb/main/scripts/install-docker.sh | env NEURONDB_IMAGE=neurondb/neurondb-cuda:latest bash
```

Or: `bash scripts/install-docker.sh --image neurondb/neurondb-cuda:latest`

## Environment variables

The image follows the standard **PostgreSQL** Docker variables:

| Variable | Purpose |
|----------|---------|
| `POSTGRES_USER` | Database superuser name |
| `POSTGRES_PASSWORD` | Password for that user |
| `POSTGRES_DB` | Default database created on first start |

The install script defaults to user/database/password `neurondb`. For production, set a strong password and restrict network access.

## Persistent data

The install script creates a **named volume** per container name, e.g. `neurondb-neurondb-pgdata` for the default container `neurondb`. Data survives container removal **unless** you use `--reset` with confirmation.

## Ports

- Container listens on **5432** internally.
- The install script maps **host 5433 → 5432** by default to avoid clashing with a local PostgreSQL.

Override with `--port`.

## Docker Compose (repository)

The repo includes [`docker/docker-compose.yml`](../docker/docker-compose.yml). By default it **builds** a local image for development. To use a published Hub image instead, override the service `image` and **remove or comment out** the `build:` section for that service, for example:

```yaml
services:
  neurondb:
    image: neurondb/neurondb:latest
    pull_policy: always
```

For CUDA, use `neurondb/neurondb-cuda:latest` and configure GPU access per your Compose version (DeviceRequests / `runtime: nvidia`, etc.).

## CPU vs GPU images

- **CPU (default install):** [`docker/neurondb/Dockerfile`](../docker/neurondb/Dockerfile) — **`neurondb/neurondb`** on Docker Hub for the quickstart path.
- **CUDA (registry):** [`docker/neurondb/Dockerfile.gpu.cuda`](../docker/neurondb/Dockerfile.gpu.cuda) — **`neurondb/neurondb-cuda`** on Docker Hub / GHCR.

## Resetting local data

1. Stop the container: `docker stop neurondb`
2. Run the install script with `--reset` **only after** setting `NEURONDB_CONFIRM_RESET=yes`, or remove the container and volume manually:

```bash
docker rm -f neurondb
docker volume rm neurondb-neurondb-pgdata
```

Use the exact volume name from `docker volume ls` if you changed `--name`.

## Troubleshooting

- **Port in use:** pick another `--port` or stop the conflicting service.
- **Cannot pull image:** log in to Docker Hub if the repo is private; for GHCR, `docker login ghcr.io`.
- **Extension errors:** see [getting-started/troubleshooting.md](getting-started/troubleshooting.md) and [troubleshooting.md](troubleshooting.md).
