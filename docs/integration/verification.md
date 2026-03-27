# Verification

How to verify that the NeuronDB extension is installed and working.

## Extension

- **Load:** Connect to the database and run:
  ```sql
  SELECT neurondb.version();
  ```
- **Vector:** Create a table with a `vector` column, build an HNSW index, run a k-NN query.
- **RAG:** Run `neurondb.rag_query` or `neurondb.rag_ingest_document` (ensure GUCs such as `neurondb.llm_provider` are set if required).
- **Embedding:** Call `embed_text(...)` (or equivalent); ensure required GUCs are set.

## Unit tests

From the repository root (PostgreSQL with the extension installed):

```bash
make installcheck
```

## Docker

After starting NeuronDB with Docker Compose:

```bash
docker compose -f docker/docker-compose.yml up -d neurondb
./scripts/verify-docker-ecosystem.sh
```

Optional: `./scripts/integration/verify-hub-detailed.sh` runs additional checks when extra services are deployed; see the script for usage.

## See also

- [Troubleshooting](../getting-started/troubleshooting.md)
- [Documentation](../readme.md)
