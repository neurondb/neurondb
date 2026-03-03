# NeuronDB RAG Benchmarks

This directory contains benchmarks for evaluating RAG (Retrieval Augmented Generation) systems using NeuronDB.

## Overview

Three benchmark suites are provided:

1. **MTEB** (Massive Text Embedding Benchmark) - Tests embedding quality
2. **BEIR** (Benchmarking IR) - Tests retrieval quality
3. **RAGAS** (Retrieval Augmented Generation Assessment) - Tests answer quality

## Prerequisites

### Python Dependencies

```bash
# Install required packages
pip install mteb beir ragas datasets numpy psycopg2-binary
```

### Database Setup

Ensure NeuronDB extension is installed:

```sql
CREATE EXTENSION IF NOT EXISTS neurondb;
```

## Benchmarks

### 1. MTEB Benchmark (`mteb_benchmark.py`)

Tests embedding quality across multiple tasks:
- Classification
- Clustering
- Pair Classification
- Reranking
- Retrieval
- STS (Semantic Text Similarity)
- Summarization

#### Usage

```bash
# Run all MTEB tasks
python mteb_benchmark.py --model all-MiniLM-L6-v2

# Run specific task types
python mteb_benchmark.py --model all-MiniLM-L6-v2 --task-types Classification Retrieval

# Run specific tasks
python mteb_benchmark.py --model all-MiniLM-L6-v2 --tasks STSBenchmark

# Custom database connection
python mteb_benchmark.py \
    --model all-MiniLM-L6-v2 \
    --host localhost \
    --port 5432 \
    --database neurondb \
    --user pge \
    --output-dir ./results
```

#### Options

- `--model`: Embedding model name (default: `all-MiniLM-L6-v2`)
- `--tasks`: Specific tasks to run (default: all)
- `--task-types`: Task types to run (default: all)
- `--output-dir`: Output directory (default: `./results`)
- `--host`, `--port`, `--database`, `--user`, `--password`: Database connection

### 2. BEIR Benchmark (`beir_benchmark.py`)

Tests retrieval quality across diverse domains and tasks.

#### Usage

```bash
# Run on MS MARCO dataset
python beir_benchmark.py --dataset msmarco --model all-MiniLM-L6-v2

# Run on Natural Questions
python beir_benchmark.py --dataset nq --model all-MiniLM-L6-v2

# Run on SciFact
python beir_benchmark.py --dataset scifact --model all-MiniLM-L6-v2

# Custom settings
python beir_benchmark.py \
    --dataset msmarco \
    --model all-MiniLM-L6-v2 \
    --index hnsw \
    --top-k 100 \
    --data-path ./beir_data \
    --output-dir ./results
```

#### Available Datasets

- `msmarco`: MS MARCO passage ranking
- `nq`: Natural Questions
- `scifact`: SciFact
- `fiqa`: Financial Question Answering
- `arguana`: ArguAna
- `climate-fever`: Climate-FEVER
- `dbpedia-entity`: DBpedia Entity
- `fever`: FEVER
- `hotpotqa`: HotpotQA
- `nfcorpus`: NFCorpus
- `quora`: Quora
- `scidocs`: SCIDOCS
- `scifact`: SciFact
- `trec-covid`: TREC-COVID
- `webis-touche2020`: Webis-Touche2020

#### Options

- `--dataset`: BEIR dataset name (required)
- `--model`: Embedding model name (default: `all-MiniLM-L6-v2`)
- `--index`: Index type - `hnsw` or `ivfflat` (default: `hnsw`)
- `--top-k`: Number of top results to retrieve (default: 100)
- `--data-path`: Path for BEIR datasets (default: `./beir_data`)
- `--output-dir`: Output directory (default: `./results`)

#### Metrics

BEIR reports:
- **NDCG@k**: Normalized Discounted Cumulative Gain at k
- **MAP@k**: Mean Average Precision at k
- **Recall@k**: Recall at k
- **MRR@k**: Mean Reciprocal Rank at k

### 3. RAGAS Benchmark (`ragas_benchmark.py`)

Tests answer quality for RAG systems.

#### Usage

```bash
# Run with dataset file
python ragas_benchmark.py \
    --dataset ./data/rag_dataset.json \
    --model all-MiniLM-L6-v2

# Custom settings
python ragas_benchmark.py \
    --dataset ./data/rag_dataset.json \
    --model all-MiniLM-L6-v2 \
    --index hnsw \
    --top-k 5 \
    --output-dir ./results
```

#### Dataset Format

The dataset JSON file should have the following structure:

```json
{
  "documents": [
    {
      "id": "doc1",
      "text": "Document text content..."
    }
  ],
  "examples": [
    {
      "question": "What is the question?",
      "answer": "The generated answer",
      "contexts": ["Retrieved context 1", "Retrieved context 2"],
      "ground_truth": "The correct answer (optional)"
    }
  ]
}
```

If `contexts` are not provided, the benchmark will retrieve them using NeuronDB.

#### Options

- `--dataset`: Path to dataset JSON file (required)
- `--model`: Embedding model name (default: `all-MiniLM-L6-v2`)
- `--index`: Index type - `hnsw` or `ivfflat` (default: `hnsw`)
- `--top-k`: Number of documents to retrieve (default: 5)
- `--output-dir`: Output directory (default: `./results`)

#### Metrics

RAGAS evaluates:
- **Context Precision**: Precision of retrieved contexts
- **Context Recall**: Recall of retrieved contexts
- **Faithfulness**: How grounded the answer is in the contexts
- **Answer Relevancy**: How relevant the answer is to the question
- **Answer Correctness**: Accuracy compared to ground truth (if provided)
- **Answer Semantic Similarity**: Semantic similarity to ground truth (if provided)

## Running All Benchmarks

### Automated Runner (`run_bm.py`)

The recommended way to run all benchmarks is using the automated runner script, which handles:
- Core dump setup and analysis
- Database restart (`make install` and `pg_ctl restart`)
- Error recovery and crash detection
- Unified results output (JSON and Markdown)

#### Quick Start

```bash
# Run all benchmarks
python run_bm.py \
    --beir-dataset msmarco \
    --ragas-dataset ./data/rag_dataset.json

# Run specific benchmarks
python run_bm.py \
    --benchmarks mteb,beir \
    --beir-dataset msmarco

# Custom database connection
python run_bm.py \
    --beir-dataset msmarco \
    --ragas-dataset ./data/rag_dataset.json \
    --db-host localhost \
    --db-port 5432 \
    --db-name neurondb \
    --db-user pge
```

#### Options

- `--benchmarks`: Comma-separated list (mteb,beir,ragas) or "all" (default: all)
- `--mteb-model`: Model name for MTEB (default: all-MiniLM-L6-v2)
- `--beir-dataset`: BEIR dataset name (required if beir selected)
- `--ragas-dataset`: Path to RAGAS dataset JSON (required if ragas selected)
- `--skip-install`: Skip `make install` step
- `--skip-restart`: Skip database restart
- `--output-dir`: Output directory for results (default: ./results)
- `--db-host`, `--db-port`, `--db-name`, `--db-user`, `--db-password`: Database connection
- `--verbose`: Verbose output

#### Features

1. **Core Dump Management**:
   - Automatically sets up `/tmp/cores` directory
   - Configures `ulimit -c unlimited` for core dumps
   - Detects crashes during benchmark execution
   - Analyzes core dumps using `gdb` to extract stack traces

2. **Database Restart**:
   - Runs `make install` in NeuronDB directory
   - Restarts PostgreSQL using `pg_ctl restart`
   - Verifies database connection before running benchmarks

3. **Error Handling**:
   - Continues with remaining benchmarks if one fails
   - Retries database connections with exponential backoff
   - Reports all errors and crashes in results

4. **Unified Results**:
   - Generates JSON file with complete results
   - Generates Markdown report for human-readable output
   - Includes crash reports with stack traces

#### Example Output

The runner generates two output files:
- `rag_benchmark_results_YYYYMMDD_HHMMSS.json`: Complete structured results
- `rag_benchmark_results_YYYYMMDD_HHMMSS.md`: Human-readable report

### Manual Execution

You can also run benchmarks individually:

```bash
# 1. MTEB
python mteb_benchmark.py --model all-MiniLM-L6-v2 --output-dir ./results/mteb

# 2. BEIR
python beir_benchmark.py --dataset msmarco --model all-MiniLM-L6-v2 --output-dir ./results/beir

# 3. RAGAS
python ragas_benchmark.py --dataset ./data/rag_dataset.json --model all-MiniLM-L6-v2 --output-dir ./results/ragas
```

## Results

All benchmarks save results to JSON files in the specified output directory:

- **MTEB**: `mteb_summary_YYYYMMDD_HHMMSS.json`
- **BEIR**: `beir_<dataset>_YYYYMMDD_HHMMSS.json`
- **RAGAS**: `ragas_YYYYMMDD_HHMMSS.json`

Results include:
- Model and configuration information
- Timing metrics
- Evaluation scores
- Detailed metrics per task/query

## Troubleshooting

### Connection Errors

Ensure PostgreSQL is running and accessible:

```bash
psql -h localhost -p 5432 -U pge -d neurondb -c "SELECT 1;"
```

### Extension Errors

Verify NeuronDB extension is installed:

```sql
\dx neurondb
```

If not installed:

```sql
CREATE EXTENSION neurondb;
```

### Embedding Errors

Check that the embedding model is available:

```sql
SELECT neurondb_embed('test', 'all-MiniLM-L6-v2');
```

### Memory Issues

For large datasets, consider:
- Reducing batch sizes
- Using smaller datasets
- Increasing database memory settings

## See Also

- [MTEB Documentation](https://github.com/embeddings-benchmark/mteb)
- [BEIR Documentation](https://github.com/beir-cellar/beir)
- [RAGAS Documentation](https://github.com/explodinggradients/ragas)
- [NeuronDB Documentation](../../docs/)

