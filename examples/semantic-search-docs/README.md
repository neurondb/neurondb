# Semantic Search Over Documents

Complete example demonstrating semantic search over document collections using NeuronDB.

## Overview

This example shows how to:
- Ingest documents (markdown files)
- Generate embeddings
- Store vectors in NeuronDB
- Perform semantic search queries
- Evaluate search quality

## Quick Start (5 minutes)

### Prerequisites

- PostgreSQL 16+ with NeuronDB extension
- Python 3.8+
- Sample documents (included)

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set database connection (Docker Compose defaults)
export DB_HOST=localhost
export DB_PORT=5433        # Docker Compose default port
export DB_NAME=neurondb
export DB_USER=neurondb   # Docker Compose default user
export DB_PASSWORD=neurondb  # Docker Compose default password

# Run ingestion
python ingest_documents.py

# Run search interface
python search.py "What is machine learning?"
```

## Files

- `ingest_documents.py` - Document ingestion and embedding generation
- `search.py` - Semantic search interface
- `evaluate.py` - Evaluation metrics script
- `requirements.txt` - Python dependencies
- `sample_docs/` - Sample markdown documents

## Usage

### Ingest Documents

```bash
python ingest_documents.py --input-dir sample_docs --chunk-size 512
```

### Search

```bash
python search.py "your query here" --limit 10
```

### Evaluate

```bash
python evaluate.py --queries queries.json --expected expected_results.json
```

## Next Steps

- Modify chunking strategy in `ingest_documents.py`
- Try different embedding models
- Adjust HNSW index parameters
- Add hybrid search (vector + full-text)

## Related Documentation

- [RAG Playbook](../../NeuronDB/docs/rag/playbook.md) - Chunking and embedding guidance
- [Vector Search](../../NeuronDB/docs/vector-search/indexing.md) - Index configuration
- [Performance Playbook](../../NeuronDB/docs/performance/playbook.md) - Optimization tips









