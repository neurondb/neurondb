# Documentation Loading Scripts

This directory contains scripts for loading documentation files into NeurondB and creating vector embeddings for semantic search.

## Overview

These scripts provide a complete workflow for:
1. Loading documentation files from a directory
2. Chunking documents into smaller segments
3. Generating vector embeddings using NeurondB
4. Creating indexes for fast similarity search

## Scripts

### `load_docs.py` (Main Entry Point)
Main Python script that orchestrates the complete workflow. Uses modular components for maximum flexibility.

**Features:**
- Supports HTML, Markdown, Text, RST, and XML files
- Automatic text extraction and cleaning
- Configurable chunking with overlap
- Batch embedding generation
- HNSW vector index creation

**Usage:**
```bash
python3 load_docs.py -d /path/to/docs -D mydb -U myuser
```

### Modular Components

The script is built from highly modular components:

- **`doc_processor.py`**: File processing and text extraction
  - `FileProcessor`: Processes individual files
  - `TextCleaner`: Cleans HTML, Markdown, and plain text
  - `TitleExtractor`: Extracts titles from various formats
  - `HTMLTextExtractor`: HTML parser for text extraction

- **`chunking.py`**: Text chunking utilities
  - `TextChunker`: Python-based chunking with sentence boundary detection
  - `SQLChunker`: SQL-based chunking using PostgreSQL functions

- **`db_operations.py`**: Database schema and operations
  - `DatabaseSchema`: Schema creation and management
  - `DocumentLoader`: Document insertion and batch loading
  - `StatisticsManager`: Statistics and reporting

- **`embeddings.py`**: Embedding generation and vector indexes
  - `EmbeddingGenerator`: Batch embedding generation
  - `VectorIndexManager`: HNSW index creation and management

### `load_documentation.sh`
Bash wrapper script for convenience (uses Python internally).

**Usage:**
```bash
chmod +x load_documentation.sh
./load_documentation.sh -d /path/to/docs -D mydb -U myuser
```

## Prerequisites

1. **PostgreSQL** with NeurondB extension installed
2. **Python 3.6+** with `psycopg2`:
   ```bash
   pip install psycopg2-binary
   ```
3. **NeurondB configured** with embedding model access:
   - Hugging Face API key (if using API)
   - Or GPU-enabled local models

## Quick Start

### 1. Set up database connection

```bash
export PGPASSWORD=your_password
```

Or use `.pgpass` file for password-less access.

### 2. Run the script

```bash
# Using bash script
./load_documentation.sh \
  -d /path/to/documentation \
  -D my_database \
  -U my_user \
  -H localhost \
  -p 5432

# Or using Python script
python3 load_documentation.py \
  -d /path/to/documentation \
  -D my_database \
  -U my_user
```

### 3. Verify results

```sql
-- Check statistics
SELECT * FROM docs_stats;

-- View documents
SELECT doc_id, filename, title, file_type 
FROM docs_documents 
LIMIT 10;

-- View chunks with embeddings
SELECT chunk_id, LEFT(chunk_text, 100) AS preview, 
       embedding IS NOT NULL AS has_embedding
FROM docs_chunks 
LIMIT 10;
```

## Configuration Options

### Common Options

- `-d, --directory DIR`: Directory containing documentation files (required)
- `-D, --database DB`: Database name (default: postgres)
- `-U, --user USER`: Database user (default: postgres)
- `-H, --host HOST`: Database host (default: localhost)
- `-p, --port PORT`: Database port (default: 5432)
- `-m, --model MODEL`: Embedding model (default: all-MiniLM-L6-v2)
- `-c, --chunk-size SIZE`: Chunk size in characters (default: 1000)
- `-o, --overlap SIZE`: Chunk overlap in characters (default: 200)
- `-t, --table-prefix PFX`: Table name prefix (default: docs)
- `--skip-embeddings`: Skip embedding generation
- `--skip-index`: Skip index creation
- `-v, --verbose`: Verbose output
- `-h, --help`: Show help message

### Embedding Models

Available models (depending on your NeurondB configuration):

- `all-MiniLM-L6-v2`: Fast, 384 dimensions (default)
- `all-mpnet-base-v2`: Higher quality, 768 dimensions
- `sentence-transformers/all-MiniLM-L6-v2`: Full model path
- Custom ONNX models: Use model name as configured

### Chunking Strategy

The scripts use intelligent chunking:
- Splits documents into overlapping segments
- Attempts to break at sentence boundaries
- Skips very short chunks (< 50 characters)
- Configurable size and overlap

**Recommended settings:**
- **Small documents (< 5KB)**: chunk_size=500, overlap=100
- **Medium documents (5-50KB)**: chunk_size=1000, overlap=200 (default)
- **Large documents (> 50KB)**: chunk_size=2000, overlap=400

## Database Schema

The scripts create the following tables:

### `{prefix}_documents`
Stores the original documents:
- `doc_id`: Primary key
- `filepath`: Full path to source file
- `filename`: Just the filename
- `title`: Extracted or generated title
- `content`: Cleaned text content
- `file_size`: File size in bytes
- `file_type`: Type (html, markdown, text)
- `metadata`: JSONB with additional metadata
- `created_at`, `updated_at`: Timestamps

### `{prefix}_chunks`
Stores document chunks with embeddings:
- `chunk_id`: Primary key
- `doc_id`: Foreign key to documents
- `chunk_index`: Index within document
- `chunk_text`: Chunk text content
- `chunk_tokens`: Approximate token count
- `embedding`: Vector embedding (384 dimensions)
- `metadata`: JSONB metadata
- `created_at`: Timestamp

### `{prefix}_stats`
View with statistics:
- Total documents and chunks
- Embedding coverage
- Average document length
- Total content size

## Querying Your Documentation

### Basic Queries

```sql
-- Find all documents
SELECT * FROM docs_documents;

-- Find chunks for a document
SELECT chunk_index, LEFT(chunk_text, 200) AS preview
FROM docs_chunks
WHERE doc_id = 1
ORDER BY chunk_index;
```

### Semantic Search

```sql
-- Search for similar content
WITH query_embedding AS (
    SELECT embed_text('how to optimize queries', 'all-MiniLM-L6-v2') AS emb
)
SELECT 
    d.title,
    d.filename,
    c.chunk_index,
    LEFT(c.chunk_text, 200) AS preview,
    c.embedding <-> qe.emb AS distance,
    1 - (c.embedding <-> qe.emb) AS similarity
FROM docs_chunks c
JOIN docs_documents d ON c.doc_id = d.doc_id
CROSS JOIN query_embedding qe
WHERE c.embedding IS NOT NULL
ORDER BY distance
LIMIT 10;
```

### Hybrid Search (Vector + Full-Text)

```sql
-- Combine vector similarity with full-text search
WITH query_embedding AS (
    SELECT embed_text('performance tuning', 'all-MiniLM-L6-v2') AS emb
),
vector_results AS (
    SELECT 
        c.chunk_id,
        c.chunk_text,
        c.embedding <-> qe.emb AS vector_distance
    FROM docs_chunks c
    CROSS JOIN query_embedding qe
    WHERE c.embedding IS NOT NULL
    ORDER BY vector_distance
    LIMIT 50
),
text_results AS (
    SELECT 
        c.chunk_id,
        c.chunk_text,
        ts_rank(to_tsvector('english', c.chunk_text), 
                to_tsquery('english', 'performance & tuning')) AS text_rank
    FROM docs_chunks c
    WHERE to_tsvector('english', c.chunk_text) @@ 
          to_tsquery('english', 'performance & tuning')
    ORDER BY text_rank DESC
    LIMIT 50
)
SELECT 
    COALESCE(v.chunk_id, t.chunk_id) AS chunk_id,
    COALESCE(v.chunk_text, t.chunk_text) AS chunk_text,
    COALESCE(1 - v.vector_distance, 0) AS vector_score,
    COALESCE(t.text_rank, 0) AS text_score,
    (COALESCE(1 - v.vector_distance, 0) * 0.7 + 
     COALESCE(t.text_rank, 0) * 0.3) AS combined_score
FROM vector_results v
FULL OUTER JOIN text_results t ON v.chunk_id = t.chunk_id
ORDER BY combined_score DESC
LIMIT 10;
```

## Performance Tips

1. **Batch Processing**: Embeddings are generated in batches for efficiency
2. **Index Creation**: HNSW index significantly speeds up similarity search
3. **Chunk Size**: Larger chunks = fewer embeddings but less granular search
4. **Model Selection**: Faster models (384-dim) for large datasets, higher quality (768-dim) for accuracy

## Troubleshooting

### Connection Errors
- Verify PostgreSQL is running
- Check credentials and host/port
- Ensure NeurondB extension is installed

### Embedding Generation Fails
- Check NeurondB configuration (GUCs)
- Verify API key if using Hugging Face API
- Check GPU availability if using local models
- Review logs for specific error messages

### Slow Performance
- Use batch processing (default)
- Create vector index after embeddings
- Consider using faster embedding model
- Process in smaller batches if memory constrained

### Missing Embeddings
- Check NeurondB logs
- Verify model name is correct
- Ensure sufficient API quota (if using API)
- Check for timeout issues

## Examples

### Example 1: Load PostgreSQL Documentation

```bash
./load_documentation.sh \
  -d ~/postgresql-docs/html \
  -D postgres \
  -U postgres \
  -m all-MiniLM-L6-v2 \
  -c 1500 \
  -o 300 \
  -t pg_docs
```

### Example 2: Load API Documentation (Smaller Chunks)

```bash
python3 load_documentation.py \
  -d ~/api-docs \
  -D mydb \
  -c 500 \
  -o 100 \
  -t api_docs \
  -v
```

### Example 3: Load and Chunk Only (Skip Embeddings)

```bash
./load_documentation.sh \
  -d ~/docs \
  -D mydb \
  --skip-embeddings
```

## License

Copyright (c) 2024-2025, neurondb, Inc.

