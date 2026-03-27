# NeuronDB Component

NeuronDB is the PostgreSQL extension in this project. It adds vector search, machine learning, embeddings, and RAG capabilities to PostgreSQL.

## Documentation

| Topic | Document |
|-------|----------|
| **Overview** | [neurondb.md](neurondb.md) |
| **Installation** | [Getting started / Installation](../getting-started/installation.md) |
| **Configuration** | [Configuration](../configuration.md) |
| **SQL API** | [SQL API](../sql-api.md) |
| **Docker** | [Docker deployment](../deployment/docker-unified.md) |

## Quick start

```sql
CREATE EXTENSION neurondb;
SELECT neurondb.version();
```

See [Simple Start](../getting-started/simple-start.md) and [Quick Start](../getting-started/quickstart.md) for full setup.
