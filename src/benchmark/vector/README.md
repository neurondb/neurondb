# NeuronDB ANN-Benchmarks Compatibility Benchmark

This directory contains benchmarks for comparing NeuronDB against external baselines using the ANN-Benchmarks framework and FAISS ground truth.

## Overview

The compatibility benchmark suite provides:

- **Standardized Evaluation**: Uses ANN-Benchmarks framework for consistent comparison
- **Ground Truth**: FAISS Flat (exact search) provides ground truth for recall calculation
- **Recall vs QPS**: Measures both accuracy (recall) and performance (queries per second)
- **Standard Datasets**: Uses well-known datasets from ANN-Benchmarks

## Why ANN-Benchmarks and FAISS?

- **ANN-Benchmarks**: Provides recall vs QPS metrics and standard datasets for fair comparison
- **FAISS Flat**: Gives exact neighbors for recall math, serving as the ground truth baseline

## Files

- `neurondb_ann_benchmark.py`: ANN-Benchmarks compatible wrapper for NeuronDB
- `compare_with_faiss.py`: Comparison script between NeuronDB and FAISS
- `run_benchmarks.py`: Main benchmark runner for running multiple configurations
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Prerequisites

1. **PostgreSQL with NeuronDB extension** installed and running
2. **Python 3.8+** with pip
3. **ANN-Benchmarks datasets** (downloaded automatically on first use)

## Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Ensure PostgreSQL is running with NeuronDB extension
psql -d neurondb -c "CREATE EXTENSION IF NOT EXISTS neurondb;"
```

## Quick Start

### Single Benchmark

Compare NeuronDB against FAISS for a single dataset:

```bash
python compare_with_faiss.py \
    --dataset sift-128-euclidean \
    --metric euclidean \
    --k 10 \
    --neurondb-index hnsw \
    --neurondb-m 16 \
    --neurondb-ef-construction 200
```

### Benchmark Suite

Run multiple benchmarks with different configurations:

```bash
# Quick test
python run_benchmarks.py --quick

# Full suite
python run_benchmarks.py \
    --datasets sift-128-euclidean glove-100-angular \
    --configs hnsw_fast hnsw_accurate \
    --k 10 50 100 \
    --output-dir ./results
```

### Using the ANN-Benchmarks Wrapper Directly

```bash
python neurondb_ann_benchmark.py \
    --dataset sift-128-euclidean \
    --metric euclidean \
    --index hnsw \
    --m 16 \
    --ef-construction 200 \
    --k 10
```

## Configuration

### Database Connection

Set via command-line arguments or environment variables:

```bash
# Command-line
python compare_with_faiss.py --host localhost --port 5432 --database neurondb --user pge

# Environment variables
export PGHOST=localhost
export PGPORT=5432
export PGDATABASE=neurondb
export PGUSER=pge
export PGPASSWORD=your_password
```

### Index Configurations

Predefined configurations in `run_benchmarks.py`:

- **hnsw_fast**: HNSW with m=16, ef_construction=200 (faster, lower recall)
- **hnsw_accurate**: HNSW with m=32, ef_construction=400 (slower, higher recall)
- **ivfflat**: IVFFlat with 100 lists
- **none**: No index (exact search, slowest)

### Standard Datasets

Available datasets from ANN-Benchmarks:

- `sift-128-euclidean`: 1M SIFT vectors, 128 dimensions, L2 distance
- `glove-100-angular`: 1.2M GloVe word vectors, 100 dimensions, cosine distance
- `deep-image-96-angular`: 9.9M Deep1B image vectors, 96 dimensions, cosine distance
- `gist-960-euclidean`: 1M GIST descriptors, 960 dimensions, L2 distance
- `nytimes-256-angular`: 290K NYTimes article vectors, 256 dimensions, cosine distance
- `lastfm-64-dot`: 292K Last.fm music vectors, 64 dimensions, inner product

## Understanding Results

### Metrics

- **QPS (Queries Per Second)**: Throughput measure, higher is better
- **Recall**: Fraction of true nearest neighbors found, higher is better (1.0 = perfect)
- **Build Time**: Time to build the index
- **Query Time**: Average time per query

### Output Files

Results are saved in the output directory:

- `comparison_{dataset}_{metric}_{k}.json`: Detailed comparison results
- `comparison_{dataset}_{metric}_{k}.png`: Recall vs QPS plot
- `benchmark_summary.json`: All benchmark results
- `benchmark_report.json`: Comprehensive report with best configurations

### Example Output

```
Comparison Results
============================================================
Dataset: sift-128-euclidean
Metric: euclidean, k=10

NeuronDB (hnsw):
  QPS: 1234.56
  Recall: 0.9876
  Build time: 45.23s

FAISS (flat):
  QPS: 5678.90
  Recall: 1.0000
  Build time: 12.34s

Speedup: 4.60x
Recall ratio: 0.9876
============================================================
```

## Advanced Usage

### Custom Index Parameters

```bash
python compare_with_faiss.py \
    --dataset sift-128-euclidean \
    --neurondb-index hnsw \
    --neurondb-m 32 \
    --neurondb-ef-construction 400
```

### Multiple k Values

```bash
python run_benchmarks.py \
    --datasets sift-128-euclidean \
    --configs hnsw_fast \
    --k 1 5 10 50 100
```

### Custom FAISS Index

Modify `compare_with_faiss.py` to use different FAISS index types:
- `flat`: Exact search (ground truth)
- `ivf`: Inverted file index
- `hnsw`: Hierarchical NSW

## Integration with ANN-Benchmarks

The `neurondb_ann_benchmark.py` module implements the ANN-Benchmarks interface, allowing integration with the full ANN-Benchmarks framework:

```python
from neurondb_ann_benchmark import NeuronDBANN

# Initialize
alg = NeuronDBANN(
    metric="euclidean",
    index_type="hnsw",
    index_params={"m": 16, "ef_construction": 200}
)

# Build index
alg.fit(X_train)

# Query
neighbors = alg.query(query_vector, k=10)
```

## Troubleshooting

### Database Connection Errors

```
Error: Cannot connect to database
```

- Ensure PostgreSQL is running: `pg_ctl status`
- Check connection parameters: `--host`, `--port`, `--database`, `--user`
- Verify NeuronDB extension: `psql -d neurondb -c "\dx neurondb"`

### Missing Datasets

```
Error: Dataset not found
```

- ANN-Benchmarks will download datasets automatically on first use
- Large datasets may take time to download
- Check disk space (some datasets are several GB)

### Import Errors

```
ModuleNotFoundError: No module named 'ann_benchmarks'
```

- Install dependencies: `pip install -r requirements.txt`
- Ensure you're using the correct Python environment

### Low Recall

If recall is lower than expected:

- Increase `ef_construction` for HNSW
- Increase `m` for HNSW (more connections)
- Use `hnsw_accurate` configuration
- Check that the metric matches the dataset

### Low QPS

If QPS is lower than expected:

- Check database connection pooling
- Ensure indexes are built (not using `none`)
- Consider using faster index configurations
- Check system resources (CPU, memory, disk I/O)

## Performance Tips

1. **Index Building**: Build indexes once and reuse for multiple queries
2. **Connection Pooling**: Use connection pooling for better performance
3. **Batch Queries**: Use batch queries when possible
4. **Index Tuning**: Tune index parameters based on your dataset characteristics
5. **Hardware**: Use SSD storage and sufficient RAM for large datasets

## Contributing

When adding new benchmarks:

1. Follow the ANN-Benchmarks interface
2. Include ground truth comparison with FAISS
3. Document index parameters and expected performance
4. Add results to the benchmark suite

## References

- [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks): Standardized evaluation framework
- [FAISS](https://github.com/facebookresearch/faiss): Facebook AI Similarity Search
- [NeuronDB Documentation](../../docs/): NeuronDB vector search documentation

## License

Same as NeuronDB project license.



