# Releasing Docker images

Maintainers publish **CUDA** images (**`linux/amd64`**) to Docker Hub and GHCR using GitHub Actions.

## One-time setup

1. **Docker Hub:** **`neurondb/neurondb-cuda`** — [view on Hub](https://hub.docker.com/r/neurondb/neurondb-cuda). This is where CI pushes CUDA images; create the repo under the **neurondb** org if it is still missing. If you also maintain **`neurondb/neurondb`**, keep its readme in sync separately (see below).
2. **GitHub secrets** (repository settings → Secrets and variables → Actions):
   - `DOCKERHUB_USERNAME`
   - `DOCKERHUB_TOKEN` (access token with push scope)
3. **GHCR:** Uses `GITHUB_TOKEN` from the workflow (`packages: write`). Ensure workflow permissions allow package publish.

### Docker Hub page (readme + summary)

Use **[`scripts/dockerhub-update-repo.sh`](../scripts/dockerhub-update-repo.sh)** (`DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` with Read/Write/Delete scope). Pick the readme file that matches the **repository name** so pull commands on Hub match the page URL:

| Hub repo | Default readme used by script |
|:---------|:------------------------------|
| **`neurondb/neurondb-cuda`** (default `DOCKERHUB_REPOSITORY`) | [`docker/README.DockerHub.md`](../docker/README.DockerHub.md) |
| **`neurondb/neurondb`** (`DOCKERHUB_REPOSITORY=neurondb`) | [`docker/README.DockerHub.neurondb.md`](../docker/README.DockerHub.neurondb.md) |

```bash
./scripts/dockerhub-update-repo.sh
DOCKERHUB_REPOSITORY=neurondb ./scripts/dockerhub-update-repo.sh
```

Optional: `DOCKERHUB_README_PATH` to override; `DOCKERHUB_DESCRIPTION` for the one-line summary.

## Automated publish

Workflow: [`.github/workflows/docker-publish.yml`](../.github/workflows/docker-publish.yml)

- **On push** of tag `v*` (e.g. `v3.1.0`): builds **`Dockerfile.gpu.cuda`** (matrix **PostgreSQL 16, 17, and 18**, **linux/amd64**), smoke-tests each, pushes **`neurondb/neurondb-cuda`** tags **`pg16`**, **`pg17`**, **`pg18`**, matching **`pg<major>-<version>`**, and (for **PG 17 only**) **`latest`** and **`<version>`** to Docker Hub + GHCR.
- **workflow_dispatch:** optional toggle to skip GHCR (default pushes to GHCR when enabled).

## Manual release with `scripts/docker-release.sh`

From repository root (requires Docker):

```bash
export DOCKERHUB_USERNAME=...
export DOCKERHUB_TOKEN=...
./scripts/docker-release.sh 3.1.0
# Other Postgres majors (do not overwrite :latest / bare semver — those stay on PG 17):
# PG_MAJOR=16 ./scripts/docker-release.sh 3.1.0
# PG_MAJOR=18 ./scripts/docker-release.sh 3.1.0
# Optional: PUSH_GHCR=1 after docker login ghcr.io
```

## Verify after release

1. **Docker Hub:** `docker pull neurondb/neurondb-cuda:latest` and run with `--gpus all`; smoke SQL (`CREATE EXTENSION`, `SELECT neurondb.version();`).
2. **GHCR:** `docker pull ghcr.io/neurondb/neurondb-cuda:latest` (after `docker login ghcr.io`).
3. **End-user script:** run the [install-docker.sh](../scripts/install-docker.sh) one-liner from `main` on a clean machine or VM.

## Version tags

Semantic versions should match Git tags (`v3.1.0` → image tags `3.1.0`, `pg17-3.1.0`, etc.).
