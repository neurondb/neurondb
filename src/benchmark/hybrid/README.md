# NeuronDB Hybrid Search Benchmarks

This directory contains benchmarks for evaluating hybrid search (lexical + semantic) using NeuronDB's built-in hybrid search functions.

## Overview

The hybrid benchmark suite provides:

- **Standardized Evaluation**: Uses BEIR framework for consistent comparison
- **Multiple Retrieval Modes**: Compares lexical-only (FTS), vector-only, and hybrid search
- **In-Database Embeddings**: Uses NeuronDB's `neurondb_embed` functions for embedding generation
- **Performance Metrics**: Measures both accuracy (NDCG, MAP, Recall, MRR) and latency (QPS, p50/p95)

## Why Hybrid Search?

Hybrid search combines:
- **Lexical search** (full-text search): Captures exact keyword matches, term frequency, and document structure
- **Semantic search** (vector embeddings): Captures meaning, synonyms, and conceptual similarity

NeuronDB's `hybrid_search` function combines both approaches with configurable weights.

## Prerequisites

1. **PostgreSQL with NeuronDB extension** installed and running
2. **Python 3.8+** with pip
3. **NeuronDB embedding configuration**: Set `neurondb.llm_provider` and optionally `neurondb.llm_api_key` if using remote embedding APIs
4. **BEIR datasets** (downloaded automatically on first use)

## Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Ensure PostgreSQL is running with NeuronDB extension
psql -d neurondb -c "CREATE EXTENSION IF NOT EXISTS neurondb;"

# Configure embedding provider (if needed)
psql -d neurondb -c "SET neurondb.llm_provider = 'huggingface';"
# Or for OpenAI:
# psql -d neurondb -c "SET neurondb.llm_provider = 'openai';"
# psql -d neurondb -c "SET neurondb.llm_api_key = 'your-key-here';"
```

## Quick Start

### Single Benchmark

Run hybrid search benchmark on a single dataset:

```bash
python run_bm.py \
    --datasets msmarco \
    --model all-MiniLM-L6-v2 \
    --vector-weights 0.7 \
    --top-k 100
```

### Benchmark Suite

Run multiple benchmarks with different configurations:

```bash
# Multiple datasets, vector weights, and top-k values
python run_bm.py \
    --datasets msmarco,nq \
    --vector-weights 0.0,0.5,0.7,1.0 \
    --top-k 10,100
```

### Full Benchmark Suite

The `run_bm.py` script provides a comprehensive benchmark runner:

```bash
# Full benchmark suite with build/restart
python run_bm.py \
    --datasets msmarco \
    --vector-weights 0.7 \
    --top-k 100

# Skip build/restart (if already set up)
python run_bm.py \
    --skip-build \
    --skip-restart \
    --datasets msmarco,nq
```

## Configuration

### Database Connection

Set via command-line arguments or environment variables:

```bash
# Command-line
python beir_hybrid_benchmark.py --host localhost --port 5432 --database neurondb --user pge

# Environment variables
export PGHOST=localhost
export PGPORT=5432
export PGDATABASE=neurondb
export PGUSER=pge
export PGPASSWORD=your_password
```

### Hybrid Search Parameters

- **`--vector-weight`**: Weight for vector search (0.0 = pure FTS, 1.0 = pure vector, 0.7 = balanced hybrid)
- **`--query-type`**: FTS query type (`plain`, `to`, `phrase`)
- **`--top-k`**: Number of results to retrieve

### Standard Datasets

Available BEIR datasets:

- `msmarco`: MS MARCO passage ranking (8.8M passages)
- `nq`: Natural Questions (268K passages)
- `scifact`: SciFact (5K passages)
- `fiqa`: Financial Question Answering (57K passages)
- `arguana`: ArguAna (8.7K passages)
- `climate-fever`: Climate-FEVER (5.4K passages)
- `dbpedia-entity`: DBpedia Entity (4.6M passages)
- `fever`: FEVER (5.4K passages)
- `hotpotqa`: HotpotQA (5.2M passages)
- `nfcorpus`: NFCorpus (3.6K passages)
- `quora`: Quora (523K passages)
- `scidocs`: SCIDOCS (25K passages)
- `trec-covid`: TREC-COVID (171K passages)
- `webis-touche2020`: Webis-Touche2020 (382K passages)

## Understanding Results

### Metrics

- **NDCG@k**: Normalized Discounted Cumulative Gain at k (higher is better, max 1.0)
- **MAP@k**: Mean Average Precision at k (higher is better, max 1.0)
- **Recall@k**: Recall at k (higher is better, max 1.0)
- **MRR@k**: Mean Reciprocal Rank at k (higher is better, max 1.0)
- **QPS**: Queries per second (higher is better)
- **Latency (p50/p95)**: Median and 95th percentile query latency (lower is better)

### Output Files

Results are saved in the output directory:

- `beir_hybrid_<dataset>_<timestamp>.json`: Detailed benchmark results
- `benchmark_summary.txt`: Consolidated summary report (from `run_bm.py`)

### Example Output

```json
{
  "dataset": "msmarco",
  "model": "all-MiniLM-L6-v2",
  "retrieval_modes": {
    "lexical": {
      "ndcg@10": 0.2345,
      "map@10": 0.1234,
      "recall@10": 0.3456,
      "qps": 1234.56
    },
    "hybrid": {
      "vector_weight": 0.7,
      "ndcg@10": 0.3456,
      "map@10": 0.2345,
      "recall@10": 0.4567,
      "qps": 987.65
    }
  }
}
```

## Advanced Usage

### Custom Vector Weights

Test different hybrid search configurations:

```bash
python beir_hybrid_benchmark.py \
    --dataset msmarco \
    --vector-weight 0.5 \
    --query-type plain
```

### Multiple k Values

```bash
python beir_hybrid_benchmark.py \
    --dataset msmarco \
    --top-k 10 50 100
```

### Custom Index Parameters

The benchmark automatically creates:
- HNSW index on `embedding` column (for vector search)
- GIN index on `fts_vector` column (for lexical search)

Index parameters can be tuned in the code if needed.

## Troubleshooting

### Database Connection Errors

```
Error: Cannot connect to database
```

- Ensure PostgreSQL is running: `pg_ctl status`
- Check connection parameters: `--host`, `--port`, `--database`, `--user`
- Verify NeuronDB extension: `psql -d neurondb -c "\dx neurondb"`

### Embedding Errors

```
Error: Embedding generation failed
```

- Check embedding provider configuration: `psql -d neurondb -c "SHOW neurondb.llm_provider;"`
- Verify API key (if using remote provider): `psql -d neurondb -c "SHOW neurondb.llm_api_key;"`
- Test embedding function: `psql -d neurondb -c "SELECT neurondb_embed('test', 'all-MiniLM-L6-v2');"`

### Missing Datasets

```
Error: Dataset not found
```

- BEIR will download datasets automatically on first use
- Large datasets may take time to download
- Check disk space (some datasets are several GB)

### Low Performance

If QPS is lower than expected:

- Check database connection pooling
- Ensure indexes are built (check with `\d+ beir_hybrid_documents`)
- Consider using faster index configurations
- Check system resources (CPU, memory, disk I/O)

## Performance Tips

1. **Index Building**: Build indexes once and reuse for multiple queries
2. **Connection Pooling**: Use connection pooling for better performance
3. **Batch Embeddings**: Use `neurondb_embed_batch` for bulk embedding generation
4. **Weight Tuning**: Experiment with different `vector_weight` values (0.5-0.8 often works well)
5. **Hardware**: Use SSD storage and sufficient RAM for large datasets

## Contributing

When adding new benchmarks:

1. Follow the BEIR evaluation framework
2. Include both lexical and hybrid search comparisons
3. Document embedding model and configuration
4. Add results to the benchmark suite

## References

- [BEIR](https://github.com/beir-cellar/beir): Benchmarking IR framework
- [NeuronDB Hybrid Search Documentation](../../docs/hybrid-search/overview.md)
- [NeuronDB Documentation](../../docs/)

## License

Same as NeuronDB project license.

