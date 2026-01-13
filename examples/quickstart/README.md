# NeuronDB Quickstart Data Pack

**Get started with NeuronDB in under a minute!**

This quickstart data pack provides a minimal sample dataset (~500 documents with vector embeddings) that you can load with a single command to start experimenting with NeuronDB immediately.

## What's Included

- **Schema**: `quickstart_documents` table with id, title, content, and embedding columns
- **Sample Data**: ~500 documents with technology/AI-related content
- **Pre-built Index**: HNSW index on embeddings for fast similarity search
- **Ready to Query**: Data is ready for vector search queries immediately

## Quick Start

### Option 1: Using the Loader Script (Recommended)

```bash
# From repository root
./examples/quickstart/load_quickstart.sh
```

The script automatically detects your setup (Docker Compose or native PostgreSQL) and loads the data.

### Option 2: Using psql Directly

**With Docker Compose:**
```bash
docker compose exec neurondb psql -U neurondb -d neurondb -f examples/quickstart/quickstart_data.sql
```

**With Native PostgreSQL:**
```bash
psql -h localhost -p 5432 -U postgres -d neurondb -f examples/quickstart/quickstart_data.sql
```

### Option 3: Using Custom Connection

```bash
./examples/quickstart/load_quickstart.sh -h localhost -p 5432 -d mydb -U postgres -W password
```

## What Gets Created

### Table: `quickstart_documents`

```sql
CREATE TABLE quickstart_documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384)
);
```

### Index: `quickstart_documents_embedding_idx`

```sql
CREATE INDEX quickstart_documents_embedding_idx
ON quickstart_documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

## Quick Examples

After loading the data pack, try these queries:

### 1. View Sample Documents

```sql
SELECT id, title, LEFT(content, 50) AS preview 
FROM quickstart_documents 
LIMIT 5;
```

### 2. Basic Similarity Search

```sql
SELECT id, title,
       embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1) AS distance
FROM quickstart_documents
WHERE id != 1
ORDER BY embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)
LIMIT 10;
```

### 3. Text Search

```sql
SELECT id, title
FROM quickstart_documents
WHERE title LIKE '%Machine Learning%'
LIMIT 5;
```

### 4. Get Statistics

```sql
SELECT 
    COUNT(*) AS total_documents,
    COUNT(embedding) AS documents_with_embeddings,
    pg_size_pretty(pg_total_relation_size('quickstart_documents')) AS table_size
FROM quickstart_documents;
```

## More Examples

For more advanced examples and recipes, see:

- **[SQL Recipe Library](../../Docs/getting-started/recipes/)** - Ready-to-run queries for vector search, hybrid search, and more
- **[Basics Examples](../basics/)** - Python examples for getting started
- **[NeuronDB Quickstart Guide](../../Docs/getting-started/quickstart.md)** - Complete quickstart guide

## Loader Script Options

The `load_quickstart.sh` script supports the following options:

```bash
./load_quickstart.sh [OPTIONS]

Options:
  -h, --host HOST       Database host (default: auto-detect)
  -p, --port PORT       Database port (default: auto-detect)
  -d, --database DB     Database name (default: neurondb)
  -U, --user USER       Database user (default: neurondb)
  -W, --password PASS   Database password (default: neurondb)
  -f, --file FILE       SQL file path (default: quickstart_data.sql)
  --help                Show help message
```

### Auto-Detection

If you don't specify host/port, the script will:

1. Check if Docker Compose is available
2. Check if the `neurondb` service is running
3. Use `docker compose exec` if detected
4. Otherwise, use `psql` with defaults (localhost:5433)

## Requirements

- **NeuronDB Extension**: Must be installed and enabled
- **PostgreSQL 16+**: Compatible with PostgreSQL 16, 17, or 18
- **psql**: PostgreSQL client (or Docker Compose for Docker setup)

## Troubleshooting

### "Extension neurondb does not exist"

Make sure NeuronDB is installed and the extension is created:

```sql
CREATE EXTENSION IF NOT EXISTS neurondb;
```

### "Connection refused"

**Docker Compose setup:**
```bash
# Check if service is running
docker compose ps neurondb

# Start if needed
docker compose up -d neurondb
```

**Native PostgreSQL:**
```bash
# Check if PostgreSQL is running
psql -h localhost -p 5432 -U postgres -c "SELECT 1;"
```

### "Permission denied" when running script

Make sure the script is executable:

```bash
chmod +x examples/quickstart/load_quickstart.sh
```

### Script detects wrong setup

You can override auto-detection by explicitly specifying connection parameters:

```bash
./load_quickstart.sh -h localhost -p 5432 -d mydb -U postgres
```

## Next Steps

1. **Explore the Data**: Try the example queries above
2. **Learn SQL Recipes**: Check out the [SQL Recipe Library](../../Docs/getting-started/recipes/)
3. **Try Python Examples**: See the [Basics Examples](../basics/)
4. **Build Your Own**: Create your own tables and indexes using the CLI helpers (see [scripts/neurondb-cli.sh](../../scripts/neurondb-cli.sh))

## Cleaning Up

To remove the quickstart data:

```sql
DROP TABLE IF EXISTS quickstart_documents CASCADE;
```

Or reload fresh data:

```bash
# The SQL script is idempotent - it drops and recreates the table
./examples/quickstart/load_quickstart.sh
```

## File Structure

```
examples/quickstart/
├── README.md              # This file
├── quickstart_data.sql    # SQL script with schema and data
└── load_quickstart.sh     # Loader script (bash)
```

## Support

- **Documentation**: [neurondb.ai/docs](https://neurondb.ai/docs)
- **Issues**: [GitHub Issues](https://github.com/neurondb/neurondb/issues)
- **Quickstart Guide**: [Docs/getting-started/quickstart.md](../../Docs/getting-started/quickstart.md)

---

**Happy Learning! 🚀**



