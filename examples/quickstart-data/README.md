# Quickstart Data Pack

A ready-to-use sample dataset for getting started with NeuronDB quickly. This pack includes pre-generated sample documents with embeddings, so you can start querying immediately without waiting for data generation.

## Quick Start

```bash
# Generate sample data (takes 1-2 minutes)
python generate_sample_data.py

# Load into your database
psql -h localhost -p 5433 -U neurondb -d neurondb -f sample_data/sample_data.sql
```

## What's Included

- **200 sample documents** covering topics like databases, machine learning, and AI
- **Pre-computed embeddings** using `all-MiniLM-L6-v2` (384 dimensions)
- **SQL file** ready to load into PostgreSQL
- **HNSW index** already configured for fast similarity search

## Requirements

- Python 3.8+
- sentence-transformers: `pip install sentence-transformers numpy`

## Usage

### Generate Sample Data

```bash
# Basic usage (generates 200 documents)
python generate_sample_data.py

# Custom count
python generate_sample_data.py --count 300

# Custom output directory
python generate_sample_data.py --output-dir ./my_data

# Different embedding model
python generate_sample_data.py --model all-mpnet-base-v2
```

### Load into Database

After generating, load the SQL file into your database:

```bash
# Using psql
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" -f sample_data/sample_data.sql

# Or using docker compose
docker compose exec neurondb psql -U neurondb -d neurondb -f /path/to/sample_data.sql
```

### Verify Setup

```sql
-- Check document count
SELECT COUNT(*) FROM quickstart_documents;

-- Check embeddings
SELECT COUNT(*) FROM quickstart_documents WHERE embedding IS NOT NULL;

-- Test similarity search
WITH q AS (SELECT embed_text('vector databases') AS query_vec)
SELECT title, content, embedding <=> q.query_vec AS distance
FROM quickstart_documents, q
ORDER BY distance
LIMIT 5;
```

## What Gets Created

The script creates:

1. **`quickstart_documents` table** with columns:
   - `id` - Serial primary key
   - `title` - Document title
   - `content` - Document content
   - `category` - Document category
   - `tags` - Array of tags
   - `embedding` - Vector(384) embedding
   - `created_at` - Timestamp

2. **HNSW index** (`quickstart_documents_embedding_idx`) for fast similarity search

## Sample Data

The dataset includes documents about:
- Vector databases and embeddings
- PostgreSQL optimization
- Machine learning concepts
- RAG (Retrieval-Augmented Generation)
- Search algorithms (HNSW, ANN)
- Database indexing strategies

## Next Steps

After loading the data:

1. **Try the SQL recipes**: See `examples/sql-recipes/` for ready-to-run queries
2. **Explore vector search**: Run similarity queries with different metrics
3. **Experiment with hybrid search**: Combine vector and text search
4. **Build your own data**: Modify the generator for your use case

## Troubleshooting

**"ModuleNotFoundError: No module named 'sentence_transformers'"**
```bash
pip install sentence-transformers numpy
```

**"Connection refused" when loading**
- Ensure PostgreSQL is running
- Check connection string (host, port, credentials)
- For Docker: `docker compose ps neurondb`

**Slow embedding generation**
- First run downloads the model (~80MB)
- Subsequent runs are faster
- Use `--count` to generate fewer documents for testing

## Integration with Setup Script

This data pack is designed to work with the `neurondb-quickstart-data.sh` script:

```bash
./scripts/neurondb-quickstart-data.sh
```

The setup script will automatically generate and load the sample data.




