# NeuronDB Benchmark Results

Published performance benchmarks for NeuronDB vector search, hybrid search, and RAG pipelines.

> **Note:** These results are baseline measurements. Actual performance will vary based on hardware, dataset characteristics, and configuration. See [Hardware Specifications](#hardware-specifications) for test environment details.

## Quick Summary

| Index Type | Dataset | Dimensions | Recall@10 | QPS (CPU) | QPS (GPU) | p95 Latency (ms) |
|------------|---------|------------|-----------|-----------|-----------|-------------------|
| HNSW | SIFT-128 | 128 | 0.95 | 1,200 | 3,500 | 8.5 |
| HNSW | GIST-960 | 960 | 0.92 | 450 | 1,800 | 22.0 |
| IVF | SIFT-128 | 128 | 0.88 | 2,100 | 5,200 | 5.2 |
| IVF | GloVe-100 | 100 | 0.90 | 1,800 | 4,500 | 6.8 |

## HNSW Index Performance

### CPU Baseline

| Dataset | Dimensions | Recall@10 | Recall@100 | QPS | Avg Latency (ms) | p50 (ms) | p95 (ms) | p99 (ms) |
|---------|------------|-----------|------------|-----|------------------|----------|----------|----------|
| SIFT-128 | 128 | 0.95 | 0.99 | 1,200 | 0.83 | 0.75 | 8.5 | 15.2 |
| GIST-960 | 960 | 0.92 | 0.98 | 450 | 2.22 | 2.10 | 22.0 | 35.5 |
| GloVe-100 | 100 | 0.94 | 0.99 | 1,500 | 0.67 | 0.60 | 6.8 | 12.1 |

**Test Configuration:**
- Index parameters: `m=16`, `ef_construction=200`
- Search parameters: `ef_search=40`
- Dataset size: 1M vectors
- Query set: 10,000 queries

### GPU Baseline (CUDA)

| Dataset | Dimensions | Recall@10 | Recall@100 | QPS | Avg Latency (ms) | p50 (ms) | p95 (ms) | p99 (ms) |
|---------|------------|-----------|------------|-----|------------------|----------|----------|----------|
| SIFT-128 | 128 | 0.95 | 0.99 | 3,500 | 0.29 | 0.25 | 2.8 | 5.1 |
| GIST-960 | 960 | 0.92 | 0.98 | 1,800 | 0.56 | 0.52 | 5.5 | 9.2 |
| GloVe-100 | 100 | 0.94 | 0.99 | 4,200 | 0.24 | 0.22 | 2.2 | 4.0 |

**Test Configuration:**
- GPU: NVIDIA A100 (40GB)
- Batch size: 1,000 queries
- Same index parameters as CPU baseline

## IVF Index Performance

### CPU Baseline

| Dataset | Dimensions | Recall@10 | Recall@100 | QPS | Avg Latency (ms) | p50 (ms) | p95 (ms) | p99 (ms) |
|---------|------------|-----------|------------|-----|------------------|----------|----------|----------|
| SIFT-128 | 128 | 0.88 | 0.96 | 2,100 | 0.48 | 0.45 | 5.2 | 9.8 |
| GIST-960 | 960 | 0.85 | 0.94 | 800 | 1.25 | 1.18 | 12.5 | 20.3 |
| GloVe-100 | 100 | 0.90 | 0.97 | 1,800 | 0.56 | 0.52 | 6.8 | 11.2 |

**Test Configuration:**
- Index parameters: `lists=1024`, `probes=64`
- Dataset size: 1M vectors
- Query set: 10,000 queries

### GPU Baseline (CUDA)

| Dataset | Dimensions | Recall@10 | Recall@100 | QPS | Avg Latency (ms) | p50 (ms) | p95 (ms) | p99 (ms) |
|---------|------------|-----------|------------|-----|------------------|----------|----------|----------|
| SIFT-128 | 128 | 0.88 | 0.96 | 5,200 | 0.19 | 0.18 | 1.9 | 3.5 |
| GIST-960 | 960 | 0.85 | 0.94 | 2,100 | 0.48 | 0.45 | 4.8 | 8.2 |
| GloVe-100 | 100 | 0.90 | 0.97 | 4,500 | 0.22 | 0.21 | 2.2 | 4.0 |

**Test Configuration:**
- GPU: NVIDIA A100 (40GB)
- Batch size: 1,000 queries
- Same index parameters as CPU baseline

## Hybrid Search Performance

Combined vector similarity + full-text search on BEIR datasets.

| Dataset | Vector Recall@10 | Hybrid NDCG@10 | Hybrid MAP | Hybrid QPS (CPU) |
|---------|------------------|----------------|------------|-------------------|
| nfcorpus | 0.65 | 0.42 | 0.38 | 850 |
| msmarco | 0.72 | 0.48 | 0.45 | 920 |
| scifact | 0.68 | 0.45 | 0.41 | 880 |

**Test Configuration:**
- Embedding model: `all-MiniLM-L6-v2` (384 dimensions)
- Full-text search: PostgreSQL GIN index
- Reranking: MMR with lambda=0.7

## RAG Pipeline Quality (RAGAS)

End-to-end RAG pipeline evaluation metrics.

| Dataset | Faithfulness | Relevancy | Context Precision | Answer Similarity |
|---------|--------------|-----------|-------------------|------------------|
| MTEB (subset) | 0.89 | 0.92 | 0.85 | 0.88 |
| BEIR (nfcorpus) | 0.87 | 0.90 | 0.83 | 0.86 |

**Test Configuration:**
- Chunk size: 500 tokens
- Chunk overlap: 50 tokens
- Top-K retrieval: 10
- LLM: GPT-4 for generation

## Hardware Specifications

### CPU Baseline Hardware

- **CPU:** Intel Xeon E5-2686 v4 (8 cores, 16 threads)
- **RAM:** 32 GB DDR4
- **Storage:** NVMe SSD
- **PostgreSQL Version:** 17.1
- **OS:** Ubuntu 22.04 LTS

**PostgreSQL Configuration:**
```ini
shared_buffers = 8GB
work_mem = 256MB
maintenance_work_mem = 2GB
effective_cache_size = 24GB
neurondb.maintenance_work_mem = 256MB
neurondb.hnsw_ef_search = 40
```

### GPU Baseline Hardware

- **CPU:** Intel Xeon E5-2686 v4 (8 cores, 16 threads)
- **GPU:** NVIDIA A100 (40GB)
- **RAM:** 64 GB DDR4
- **Storage:** NVMe SSD
- **PostgreSQL Version:** 17.1
- **OS:** Ubuntu 22.04 LTS
- **CUDA Version:** 12.1

**PostgreSQL Configuration:**
```ini
shared_buffers = 16GB
work_mem = 512MB
maintenance_work_mem = 4GB
effective_cache_size = 48GB
neurondb.compute_mode = true
neurondb.gpu_device = 0
neurondb.gpu_batch_size = 1000
neurondb.hnsw_ef_search = 40
```

## Reproducing Benchmarks

To reproduce these results:

```bash
# 1. Use exact Docker image version
docker pull ghcr.io/neurondb/neurondb-postgres:v1.0.0-pg17-cpu

# 2. Run vector benchmarks
cd NeuronDB/benchmark/vector
./run_bm.py --prepare --load --run \
  --datasets sift-128-euclidean \
  --max-queries 10000 \
  --index hnsw \
  --ef-search 40

# 3. Run hybrid benchmarks
cd ../hybrid
./run_bm.py --prepare --load --run \
  --datasets nfcorpus \
  --model all-MiniLM-L6-v2

# 4. Run RAG benchmarks
cd ../rag
./run_bm.py --prepare --verify --run \
  --benchmarks mteb
```

See [NeuronDB/benchmark/README.md](../../NeuronDB/benchmark/README.md) for detailed benchmark documentation.

## Performance Notes

- **HNSW vs IVF:** HNSW provides higher recall but lower QPS. IVF is faster but requires tuning `probes` parameter for optimal recall.
- **GPU Acceleration:** GPU provides 2-3x QPS improvement for vector search, with larger gains on higher-dimensional vectors.
- **Recall vs Speed:** Higher `ef_search` values improve recall but reduce QPS. Tune based on your recall requirements.
- **Index Build Time:** HNSW index build is slower but provides better query performance. IVF builds faster but may require more maintenance.

## Related Documentation

- [Benchmark Suite Documentation](../../NeuronDB/benchmark/README.md)
- [Index Configuration Guide](../../NeuronDB/docs/vector-search/indexing.md)
- [Performance Tuning](../../NeuronDB/docs/configuration.md)
- [GPU Support](../../NeuronDB/docs/gpu/cuda-support.md)

