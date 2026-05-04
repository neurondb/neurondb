# NeuronDB — CUDA

Official **CUDA** container for **NeuronDB**: vector search, embeddings, hybrid retrieval, and ML primitives in **PostgreSQL**.

This Docker Hub repository is **`neurondb/neurondb`**. The same release line may also appear as [`neurondb/neurondb-cuda`](https://hub.docker.com/r/neurondb/neurondb-cuda); use **one** Hub path consistently in your pulls and deploy configs.

| | |
|:---|:---|
| **Docker Hub (this repo)** | [`neurondb/neurondb`](https://hub.docker.com/r/neurondb/neurondb) |
| **Source** | [`github.com/neurondb/neurondb`](https://github.com/neurondb/neurondb) |
| **Documentation** | [neurondb.ai/docs](https://www.neurondb.ai/docs) |

---

## Requirements

- **Platform:** `linux/amd64`
- **GPU:** NVIDIA + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **PostgreSQL (in image):** 16, 17, or 18 — choose via image tag (see below)

---

## Pull

```bash
docker pull neurondb/neurondb:latest
```

Equivalent on GitHub Container Registry: `ghcr.io/neurondb/neurondb-cuda`

---

## Run

```bash
docker run -d --gpus all --name neurondb -p 5433:5432 \
  -e POSTGRES_USER=neurondb \
  -e POSTGRES_PASSWORD=neurondb \
  -e POSTGRES_DB=neurondb \
  neurondb/neurondb:latest
```

**Installer** (defaults to the **`neurondb/neurondb-cuda`** image name — override if you standardize on this repo):

```bash
export NEURONDB_IMAGE=neurondb/neurondb:latest
curl -fsSL https://raw.githubusercontent.com/neurondb/neurondb/main/scripts/install-docker.sh | bash
```

---

## Tags

| Tag | Meaning |
|:----|:--------|
| `latest` | PostgreSQL **17** + latest NeuronDB release |
| `pg16` · `pg17` · `pg18` | Latest build for that PostgreSQL major |
| `pg16-<version>` … `pg18-<version>` | Pin PostgreSQL major + NeuronDB version (e.g. `pg17-3.1.0`) |
| `<version>` | PostgreSQL **17** + NeuronDB version only (e.g. `3.1.0`) |

Use **`pg16-`** / **`pg18-`** prefixes when pinning majors other than 17.

---

## Verify

```bash
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" \
  -c "CREATE EXTENSION IF NOT EXISTS neurondb;" \
  -c "SELECT neurondb.version();"
```

---

## Naming

| Component | Value |
|:----------|:------|
| Organization | `neurondb` |
| Image | `neurondb` |
| Full name | `neurondb/neurondb` |

---

## License

Proprietary. Commercial use requires a separate agreement — [LICENSE](https://github.com/neurondb/neurondb/blob/main/LICENSE).

Support: **support@neurondb.ai**
