# NeuronDB

**PostgreSQL with NeuronDB built in:** vector search, embeddings, hybrid retrieval, and ML primitives in SQL—running in one database you already operate.

![Terminal demo: install with Docker, connect with psql, run NeuronDB smoke SQL](docs/assets/neurondb-demo.gif)

[![Release](https://img.shields.io/github/v/release/neurondb/neurondb?label=release)](https://github.com/neurondb/neurondb/releases)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20|%2017%20|%2018-blue)](https://www.postgresql.org/)
[![Docs](https://img.shields.io/badge/docs-neurondb.ai-green)](https://www.neurondb.ai/docs)
[![License](https://img.shields.io/badge/License-Proprietary-lightgrey)](LICENSE)

## Try NeuronDB in one command

```bash
curl -fsSL https://raw.githubusercontent.com/neurondb/neurondb/main/scripts/install-docker.sh | bash
```

Connect:

```bash
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb"
```

Smoke test in `psql`:

```sql
CREATE EXTENSION IF NOT EXISTS neurondb;
SELECT neurondb.version();
```

You get PostgreSQL on **localhost:5433**, database and user **neurondb**, a persistent Docker volume, and the install script checks the extension before it finishes. For GPU-backed images, ports, Compose, and resets, see [docs/docker.md](docs/docker.md).

## Learn more

| Topic | Document |
|--------|-----------|
| Install (Docker, packages, source) | [docs/install.md](docs/install.md) |
| Docker details (images, GPU, volumes, Compose) | [docs/docker.md](docs/docker.md) |
| SQL functions and types | [docs/sql-api.md](docs/sql-api.md) |
| Development and contributing | [docs/development.md](docs/development.md) |
| Troubleshooting | [docs/troubleshooting.md](docs/troubleshooting.md) |
| Full documentation index | [docs/README.md](docs/README.md) |

Official site: **[neurondb.ai/docs](https://www.neurondb.ai/docs)**

## License

See [LICENSE](LICENSE). **Commercial use** requires a separate agreement; contact support@neurondb.ai.
