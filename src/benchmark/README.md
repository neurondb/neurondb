# NeuronDB pgbench Benchmark Suite

This directory contains custom SQL benchmark files for testing NeuronDB performance using `pgbench`.

# NeuronDB Benchmark Results

**Last Updated:** 2026-01-06 16:17:17 UTC

## Overview

This report consolidates benchmark results from Vector Search, Hybrid Search, and RAG Pipeline benchmarks.

## Vector Search Benchmarks

**Status:** completed

### Performance Metrics

| Metric | Value |
|--------|-------|
| Dataset | sift-128-euclidean |
| Index Type | hnsw |
| Recall@10 | 1.000 |
| Recall@100 | 0.000 |
| QPS | 1.90 |
| Avg Latency | 525.62 ms |
| p50 Latency | 524.68 ms |
| p95 Latency | 546.62 ms |
| p99 Latency | 555.52 ms |

## Hybrid Search Benchmarks

**Status:** not_run

No results available. Run the hybrid benchmark to generate results.

## RAG Pipeline Benchmarks

**Status:** completed

### Configuration

| Setting | Value |
|---------|-------|
| Model | unknown |
| Timestamp | 20260104_143411 |

See `rag/results/` for detailed RAG benchmark results.

## Running Benchmarks

To regenerate these results, run:

```bash
cd NeuronDB/benchmark
./run_bm.sh
```

## Detailed Reports

For more detailed reports, see:
- Vector: `vector/results/`
- Hybrid: `hybrid/benchmark_summary.txt`
- RAG: `rag/results/`


## Available Benchmarks

### `vector.sql`
Benchmarks vector operations including:
- Distance calculations (L2, cosine, inner product, L1)
- Vector arithmetic (addition, subtraction, scalar multiplication)
- Vector normalization and norms
- Similarity calculations

### `embedding.sql`
Benchmarks embedding generation functions:
- Single text embedding (`embed_text`, `neurondb_embed`)
- Batch embedding generation (`embed_text_batch`, `neurondb_embed_batch`)
- Cached embeddings (`embed_cached`)
- Embedding dimension checks
- Consistency tests

### `ml.sql`
Benchmarks machine learning operations:
- Model prediction
- Model information lookup
- Algorithm listing
- Feature vector operations
- Model metrics retrieval

### `multimodal.sql`
Benchmarks multimodal embedding operations:
- CLIP embeddings (text and image)
- ImageBind embeddings (text, audio, video)
- Cross-modal distance calculations
- Multimodal embedding generation

### `gpu.sql`
Benchmarks GPU-accelerated operations:
- GPU distance calculations (L2, cosine, inner product)
- GPU batch operations
- GPU availability checks
- GPU vs CPU performance comparison

## Usage

### Prerequisites

1. Ensure NeuronDB extension is installed and enabled:
```sql
CREATE EXTENSION IF NOT EXISTS neurondb;
```

2. For embedding benchmarks, configure your embedding provider (optional):
```sql
SET neurondb.llm_provider = 'openai';
SET neurondb.llm_api_key = 'your-api-key';
```

3. For GPU benchmarks, enable GPU mode:
```sql
SET neurondb.compute_mode = true;
```

### Running Benchmarks

#### Quick Start: Run All Benchmarks

The easiest way to run all benchmarks is using the provided script:

```bash
# Run all benchmarks with default settings
./benchmark/run_all_benchmarks.sh

# Run with custom database and settings
./benchmark/run_all_benchmarks.sh -d mydb -c 20 -t 2000

# Run each benchmark for a specific duration
./benchmark/run_all_benchmarks.sh -T 60

# Run only specific benchmark
./benchmark/run_all_benchmarks.sh --only vector

# Skip specific benchmarks
./benchmark/run_all_benchmarks.sh --skip embedding --skip ml

# See all options
./benchmark/run_all_benchmarks.sh --help
```

The script will:
- Test database connectivity
- Verify NeuronDB extension is installed
- Run all benchmarks in sequence
- Save results to timestamped log files in `./results/`
- Generate a summary report

#### Manual Usage: Individual Benchmarks

```bash
# Initialize pgbench (creates test tables)
pgbench -i -s 10 -d your_database

# Run vector benchmark
pgbench -f benchmark/vector.sql -c 10 -j 2 -t 1000 -d your_database

# Run embedding benchmark
pgbench -f benchmark/embedding.sql -c 5 -j 2 -t 500 -d your_database

# Run ML benchmark
pgbench -f benchmark/ml.sql -c 10 -j 2 -t 1000 -d your_database

# Run multimodal benchmark
pgbench -f benchmark/multimodal.sql -c 5 -j 2 -t 500 -d your_database

# Run GPU benchmark
pgbench -f benchmark/gpu.sql -c 10 -j 2 -t 1000 -d your_database
```

#### Advanced Usage

```bash
# Run with custom duration (60 seconds)
pgbench -f benchmark/vector.sql -c 10 -j 2 -T 60 -d your_database

# Run with progress reporting
pgbench -f benchmark/vector.sql -c 10 -j 2 -t 1000 -P 5 -d your_database

# Run with logging
pgbench -f benchmark/vector.sql -c 10 -j 2 -t 1000 -l -d your_database

# Run multiple benchmarks in sequence
for bench in vector embedding ml multimodal gpu; do
    echo "Running $bench benchmark..."
    pgbench -f benchmark/${bench}.sql -c 10 -j 2 -t 1000 -d your_database
done
```

#### Using the Automated Script

The `run_all_benchmarks.sh` script provides additional features:

- **Automatic Setup**: Checks database connection and NeuronDB extension
- **Comprehensive Logging**: Saves detailed logs for each benchmark run
- **Progress Tracking**: Shows real-time progress for each benchmark
- **Flexible Configuration**: Supports command-line options and environment variables
- **Error Handling**: Continues running even if individual benchmarks fail
- **Summary Report**: Generates a summary of all benchmark results
- **Selective Execution**: Run only specific benchmarks or skip unwanted ones

**Environment Variables** (can be used instead of command-line options):
- `PGDATABASE`: Database name
- `PGHOST`: Database host
- `PGPORT`: Database port
- `PGUSER`: Database user
- `BENCH_CLIENTS`: Number of concurrent clients
- `BENCH_JOBS`: Number of worker threads
- `BENCH_TRANSACTIONS`: Number of transactions per client
- `BENCH_DURATION`: Duration in seconds (overrides transactions)
- `BENCH_PROGRESS`: Progress report interval
- `BENCH_LOG_DIR`: Directory for log files

### Command Line Options

- `-c, --clients=N`: Number of concurrent database clients (default: 1)
- `-j, --jobs=N`: Number of worker threads (default: 1)
- `-t, --transactions=N`: Number of transactions each client runs (default: 10)
- `-T, --time=N`: Duration of benchmark test in seconds
- `-f, --file=FILENAME`: Read transaction script from file
- `-P, --progress=N`: Show progress report every N seconds
- `-l, --log`: Write transaction times to log file
- `-d, --dbname=DBNAME`: Database name to connect to

### Interpreting Results

pgbench outputs several metrics:

- **TPS (Transactions Per Second)**: Higher is better
- **Latency**: Average, minimum, maximum transaction time
- **Stddev**: Standard deviation of transaction times

Example output:
```
transaction type: <builtin: TPC-B (sort of)>
scaling factor: 10
query mode: simple
number of clients: 10
number of threads: 2
number of transactions per client: 1000
number of transactions actually processed: 10000/10000
latency average = 2.345 ms
latency stddev = 0.567 ms
tps = 4263.456789 (including connections establishing)
tps = 4265.123456 (excluding connections establishing)
```

## Notes

1. **Embedding Benchmarks**: These may be slower as they involve actual embedding generation. Adjust `-t` (transactions) accordingly.

2. **ML Benchmarks**: The `ml.sql` benchmark assumes models exist. For full ML training benchmarks, use separate long-running tests.

3. **GPU Benchmarks**: Requires GPU support and `neurondb.compute_mode = true`. Results will vary based on GPU availability.

4. **Variable Randomization**: All benchmarks use `\set` with `random()` to generate varied test data, ensuring realistic performance measurements.

5. **Error Handling**: Some benchmarks may fail if required models or configurations are missing. This is expected behavior for benchmarking.

## Customization

You can customize the benchmarks by:

1. Adjusting the random value ranges in `\set` statements
2. Modifying the test queries to match your use case
3. Adding additional benchmark scenarios
4. Changing the model names or parameters

## Troubleshooting

- **Connection errors**: Ensure PostgreSQL is running and accessible
- **Extension errors**: Verify NeuronDB extension is installed: `\dx neurondb`
- **GPU errors**: Check GPU availability: `SELECT * FROM neurondb_gpu_info();`
- **Embedding errors**: Verify embedding provider configuration
- **ML errors**: Ensure models exist before running ML benchmarks

## See Also

- [PostgreSQL pgbench documentation](https://www.postgresql.org/docs/current/pgbench.html)
- [NeuronDB documentation](../docs/)
- [NeuronDB SQL API reference](../docs/sql-api.md)

