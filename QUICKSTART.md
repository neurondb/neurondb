# 🚀 Quick Start Guide

<div align="center">

**Get NeuronDB (PostgreSQL extension) up and running in minutes**

[![Quick Start](https://img.shields.io/badge/quick--start-5_min-green)](.)
[![Difficulty](https://img.shields.io/badge/difficulty-easy-brightgreen)](.)

</div>

---

> [!TIP]
> **New here?** Start with **[Simple Start Guide](docs/getting-started/simple-start.md)** for a plain-English walkthrough.

> [!NOTE]
> This repo runs the **NeuronDB extension only**. For the full stack (NeuronAgent, NeuronMCP, NeuronDesktop), use the **neuron-deploy** repo or run each component from **neuron-agent**, **neuron-mcp**, **neuron-desktop**.

---

## 📋 Prerequisites

- [ ] **Docker** 20.10+ and **Docker Compose** 2.0+
- [ ] **5–10 minutes**
- [ ] **4GB RAM** minimum (8GB recommended)
- [ ] Port **5433** available

| Requirement   | Minimum | Recommended |
|---------------|---------|-------------|
| **Docker**    | 20.10+  | Latest      |
| **Docker Compose** | 2.0+ | Latest      |
| **RAM**       | 4GB     | 8GB+        |
| **Disk Space**| 5GB     | 10GB+       |

<details>
<summary><strong>Verify Docker</strong></summary>

```bash
docker --version
docker compose version
```

</details>

## 🚀 Step 1: Start NeuronDB

From the repository root:

```bash
# Default: CPU image on port 5433
docker compose -f docker/docker-compose.yml up -d

# Or from the docker directory
cd docker && docker compose up -d

# Or build from source
docker compose -f docker/docker-compose.yml up -d --build
```

Optional profiles (see [docker/README.md](docker/README.md)):

- `docker compose -f docker/docker-compose.yml --profile cuda up -d` — CUDA GPU
- `docker compose -f docker/docker-compose.yml --profile rocm up -d` — ROCm GPU
- `docker compose -f docker/docker-compose.yml --profile metal up -d` — Apple Metal (ARM)

**Check status:**

```bash
docker compose -f docker/docker-compose.yml ps
```

You should see the `neurondb` (or `neurondb-cpu`) service healthy.

## ✅ Step 2: Verify

```bash
# Extension version (from repo root)
docker compose -f docker/docker-compose.yml exec neurondb psql -U neurondb -d neurondb -c "SELECT neurondb.version();"

# Or from host
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" -c "SELECT neurondb.version();"
```

Expected: JSON with version, capabilities, and API flags.

Create extension and a small vector table:

```bash
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" <<EOF
CREATE EXTENSION IF NOT EXISTS neurondb;
CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  content TEXT,
  embedding vector(3)
);
INSERT INTO documents (content, embedding)
VALUES ('Hello, world!', '[0.1, 0.2, 0.3]'::vector);
SELECT id, content FROM documents;
EOF
```

## 🔧 Troubleshooting

- **Port in use:** set `POSTGRES_PORT` in `.env` or change the port in `docker/docker-compose.yml`.
- **Build failures:** ensure enough disk (e.g. 10GB+) and try `docker compose -f docker/docker-compose.yml build --no-cache`.
- **Logs:** `docker compose -f docker/docker-compose.yml logs neurondb` or `docker compose -f docker/docker-compose.yml logs -f neurondb`.

## 🗑️ Cleanup

```bash
docker compose -f docker/docker-compose.yml down
docker compose -f docker/docker-compose.yml down -v   # also remove volumes (data)
```

## 📦 Quickstart with sample data

```bash
./scripts/neurondb-quickstart-data.sh
```

See [src/examples/quickstart-data/README.md](src/examples/quickstart-data/README.md).

## Next steps

- [Full documentation](README.md)
- [NeuronDB examples](src/examples/)
- **Full stack:** [neuron-deploy](https://github.com/neurondb/neuron-deploy) or **neuron-agent**, **neuron-mcp**, **neuron-desktop** repos for Agent, MCP, and Web UI.

---

<div align="center">[⬆ Back to Top](#-quick-start-guide)</div>
