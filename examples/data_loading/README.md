# Data Loading Examples

This directory contains examples for loading datasets into NeuronDB, including Hugging Face datasets with automatic embedding generation.

## Examples

### `load_hf_dataset.py`

Simple example that loads a Hugging Face dataset and generates embeddings using NeuronDB's built-in `embed_text` function.

**Features:**
- Loads datasets from Hugging Face Hub
- Uses NeuronDB's `embed_text` function (no local model required)
- Automatically creates tables with vector columns
- Tests semantic search on loaded data

**Usage:**

```bash
# Basic usage - load SQuAD dataset (100 rows)
python3 load_hf_dataset.py --dataset squad --limit 100

# Load IMDB reviews (50 rows)
python3 load_hf_dataset.py --dataset imdb --limit 50

# Load AG News with custom embedding model
python3 load_hf_dataset.py --dataset ag_news --limit 200 --model sentence-transformers/all-MiniLM-L6-v2

# Skip semantic search test
python3 load_hf_dataset.py --dataset squad --limit 100 --skip-search
```

**Options:**

- `--dataset`: Dataset name (default: `squad`)
- `--split`: Dataset split - `train`, `test`, or `validation` (default: `train`)
- `--text-column`: Text column name (auto-detected if not specified)
- `--limit`: Maximum rows to load (default: 100)
- `--schema`: PostgreSQL schema name (default: `datasets`)
- `--table`: Table name (auto-generated from dataset name if not specified)
- `--model`: Embedding model name (default: `default`, uses PostgreSQL GUC config)
- `--test-query`: Custom query for semantic search test
- `--skip-search`: Skip semantic search test

**Simple Datasets:**

- `squad`: Stanford Question Answering Dataset - questions
- `imdb`: IMDB movie reviews
- `ag_news`: AG News articles

**Environment Variables:**

The script uses standard PostgreSQL environment variables:

```bash
export PGHOST=localhost
export PGPORT=5432
export PGDATABASE=neurondb
export PGUSER=postgres
export PGPASSWORD=your_password
```

**Prerequisites:**

```bash
pip install datasets psycopg2-binary
```

**How It Works:**

1. Loads dataset from Hugging Face Hub (streaming or non-streaming)
2. Creates a table in PostgreSQL with a vector column for embeddings
3. For each row, uses NeuronDB's `embed_text()` function to generate embeddings
4. Stores text, embeddings, and metadata in the database
5. Tests semantic search with a sample query

**Example Output:**

```
╔══════════════════════════════════════════════════════════════╗
║   Hugging Face Dataset Loader for NeuronDB                   ║
║   Using NeuronDB embed_text function for embeddings          ║
╚══════════════════════════════════════════════════════════════╝

Loading dataset: squad (split: train)
Loaded 100 rows using streaming
Created table: datasets.squad

Inserting 100 rows with embeddings...
  Inserted 10/100 rows...
  Inserted 20/100 rows...
  ...
✅ Inserted 100 rows (errors: 0)

🔍 Testing semantic search with query: 'What is the capital of France?'

Top 5 results:
1. Similarity: 0.8234 | ID: 42
   Text: What is the capital city of France?...
```

### `load_huggingface_dataset.py`

Advanced example using local SentenceTransformer models for embedding generation.

**Features:**
- Uses local SentenceTransformer models
- More control over embedding generation
- Better for offline use or custom models

**Usage:**

```bash
python3 load_huggingface_dataset.py --dataset ag_news --max-rows 100
```

**Prerequisites:**

```bash
pip install datasets sentence-transformers psycopg2-binary numpy
```

## Using with NeuronMCP

The `load_hf_dataset.py` example is compatible with NeuronMCP's dataset loading tool. When using the MCP server, you can load datasets directly through the `load_dataset` tool without running Python scripts manually.

**Example MCP Request:**

```json
{
  "name": "load_dataset",
  "arguments": {
    "source_type": "huggingface",
    "source_path": "squad",
    "split": "train",
    "limit": 100,
    "auto_embed": true,
    "create_indexes": true,
    "table_name": "squad_dataset",
    "schema_name": "datasets"
  }
}
```

## Troubleshooting

### "datasets library not available"

Install the datasets library:

```bash
pip install datasets huggingface-hub
```

### "Embeddings are all zeros"

This usually means the embedding model isn't configured in PostgreSQL. Set the API key:

```sql
-- For Hugging Face API
ALTER SYSTEM SET neurondb.llm_api_key = 'your-api-key';
SELECT pg_reload_conf();

-- Or use GUC variable for current session
SET neurondb.llm_api_key = 'your-api-key';
```

### "Connection refused"

Check your database connection settings:

```bash
# Test connection
psql -h localhost -p 5432 -U postgres -d neurondb -c "SELECT 1;"
```

### "No module named 'psycopg2'"

Install psycopg2:

```bash
pip install psycopg2-binary
```

## Related Documentation

- [NeuronMCP Dataset Loading](../../NeuronMCP/README.md#dataset-loading-examples)
- [NeuronDB Embeddings Documentation](../../NeuronDB/docs/ml-embeddings/)
- [Hugging Face Datasets](https://huggingface.co/datasets)


