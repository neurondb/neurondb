# NeuronDB Index Methods Complete Reference

**Complete reference for all index methods in NeuronDB (HNSW, IVF, Hybrid, Temporal, Sparse).**

> **Version:** 1.0  
> **Last Updated:** 2025-01-01

## Table of Contents

- [HNSW Index](#hnsw-index)
- [IVF Index](#ivf-index)
- [Hybrid Index](#hybrid-index)
- [Temporal Index](#temporal-index)
- [Sparse Index](#sparse-index)
- [Index Tuning](#index-tuning)
- [Index Maintenance](#index-maintenance)

---

## HNSW Index

### Overview

**Access Method:** `hnsw`  
**Algorithm:** Hierarchical Navigable Small World  
**Best For:** High-dimensional vectors, high recall requirements

### Structure

HNSW builds a multi-layer graph where:
- **Bottom Layer:** Contains all vectors
- **Upper Layers:** Contain progressively fewer vectors
- **Navigation:** Starts at top layer, navigates down to bottom

### Creation

```sql
CREATE INDEX ON documents USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 200);
```

**Parameters:**
- `m`: Number of bi-directional links (default: 16, range: 2-128)
- `ef_construction`: Size of candidate list during construction (default: 200, range: 4-2000)

### Query

```sql
SELECT id, content, embedding <-> query_vector AS distance
FROM documents
ORDER BY embedding <-> query_vector
LIMIT 10;
```

**Configuration:**
```sql
SET neurondb.hnsw_ef_search = 128;  -- Higher recall
SET neurondb.hnsw_k = 10;           -- Number of results
```

### Algorithm

**Build Process:**
1. Initialize with random entry point
2. For each vector:
   - Start at top layer
   - Navigate to nearest neighbor
   - Move down layers
   - Insert at bottom layer
   - Create links to nearest neighbors

**Query Process:**
1. Start at top layer entry point
2. Navigate to nearest neighbor
3. Move down layers
4. Search bottom layer with ef_search candidates
5. Return top-k results

### Performance

**Build Time:** O(n * log(n))  
**Query Time:** O(log(n))  
**Memory:** O(n * m)

**Tuning:**

> [!TIP]
> Start with default values. Increase `m` and `ef_construction` for better quality. Increase `ef_search` for better recall.

- Higher `m`: Better recall, slower build, more memory
- Higher `ef_construction`: Better quality, slower build
- Higher `ef_search`: Better recall, slower queries

---

## IVF Index

### Overview

**Access Method:** `ivfflat`  
**Algorithm:** Inverted File Index  
**Best For:** Very large datasets, lower recall acceptable

### Structure

IVF organizes vectors into clusters (lists):
- **Clusters:** K-means clustering of vectors
- **Lists:** Each cluster is a list of vectors
- **Centroids:** Cluster centers for routing

### Creation

```sql
CREATE INDEX ON documents USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);
```

**Parameters:**
- `lists`: Number of clusters (default: 100, range: 1-1000)

### Query

```sql
SELECT id, content
FROM documents
ORDER BY embedding <-> query_vector
LIMIT 10;
```

**Configuration:**
```sql
SET neurondb.ivf_probes = 20;  -- Number of lists to search
```

### Algorithm

**Build Process:**
1. K-means clustering of vectors
2. Assign vectors to nearest cluster
3. Store vectors in cluster lists

**Query Process:**
1. Find nearest cluster centroids
2. Search vectors in selected clusters
3. Return top-k results

### Performance

**Build Time:** O(n * k)  
**Query Time:** O(probes * n/k)  
**Memory:** O(n)

**Tuning:**

> [!TIP]
> Use more lists for better recall. Use fewer lists for faster builds on large datasets.

- More `lists`: Better recall, slower build
- More `probes`: Better recall, slower queries

---

## Hybrid Index

### Overview

**Access Method:** `hnsw` with full-text search  
**Algorithm:** Fused HNSW + GIN  
**Best For:** Combined vector and text search

### Structure

Hybrid index combines:
- **HNSW:** Vector similarity search
- **GIN:** Full-text search index
- **Unified Scan:** Single access method

### Creation

```sql
CREATE INDEX ON documents USING hnsw (embedding vector_l2_ops, content gin_trgm_ops);
```

### Query

```sql
SELECT * FROM hybrid_search(
    'documents',
    query_vector,
    'machine learning',
    '{}',
    0.7,  -- vector_weight
    10,   -- limit
    'plain'
);
```

**Features:**
- Combines vector and text scores
- Configurable weight for each component
- Single unified result set

---

## Temporal Index

### Overview

**Access Method:** `hnsw` with time decay  
**Algorithm:** HNSW + temporal decay  
**Best For:** Time-sensitive search, recency bias

### Structure

Temporal index extends HNSW with:
- **Timestamp Storage:** Per-vector timestamps
- **Decay Function:** Exponential decay
- **Time-Aware Scoring:** Combines distance and time

### Creation

```sql
CREATE INDEX ON documents USING hnsw (embedding vector_l2_ops);
-- Temporal search uses timestamp column
```

### Query

```sql
SELECT * FROM temporal_vector_search(
    'documents',
    query_vector,
    'created_at',
    0.01,  -- decay_rate per day
    10     -- top_k
);
```

**Decay Function:**
```
score = distance * exp(-decay_rate * days_since_created)
```

---

## Sparse Index

### Overview

**Access Method:** `sparse`  
**Algorithm:** Optimized for sparse vectors  
**Best For:** Sparse vectors (BM25, SPLADE, ColBERT)

### Structure

Sparse index optimizes for:
- **Token IDs:** Vocabulary indices
- **Weights:** Learned weights
- **Efficient Storage:** Only non-zero values

### Creation

```sql
CREATE INDEX ON documents USING sparse (embedding sparsevec_ops);
```

### Query

```sql
SELECT id, content
FROM documents
WHERE embedding @@ sparse_query_vector
ORDER BY embedding <=> sparse_query_vector
LIMIT 10;
```

---

## Index Tuning

### Auto-Tuning

**Background Worker:** `neuranmon`

**Configuration:**
```sql
SET neurondb.neuranmon_enabled = true;
SET neurondb.neuranmon_target_latency = 100.0;  -- ms
SET neurondb.neuranmon_target_recall = 0.95;
```

**Process:**
1. Samples queries from workload
2. Tests different parameter combinations
3. Selects optimal parameters
4. Updates index configuration

### Manual Tuning

**HNSW Tuning:**
```sql
-- Higher quality index
CREATE INDEX ON documents USING hnsw (embedding vector_l2_ops)
WITH (m = 32, ef_construction = 400);

-- Faster queries
SET neurondb.hnsw_ef_search = 32;
```

**IVF Tuning:**
```sql
-- More clusters for better recall
CREATE INDEX ON documents USING ivfflat (embedding vector_l2_ops)
WITH (lists = 200);

-- More probes for better recall
SET neurondb.ivf_probes = 20;
```

---

## Index Maintenance

### Defragmentation

**Background Worker:** `neurandefrag`

**Configuration:**
```sql
SET neurondb.neurandefrag_enabled = true;
SET neurandefrag_compact_threshold = 10000;
SET neurandefrag_fragmentation_threshold = 0.3;
SET neurandefrag_maintenance_window = '02:00-04:00';
```

**Process:**
1. Monitors index fragmentation
2. Triggers compaction when threshold reached
3. Rebuilds index during maintenance window

### Manual Maintenance

**Rebuild Index:**
```sql
REINDEX INDEX documents_hnsw_idx;
```

**Analyze Index:**
```sql
ANALYZE documents;
```

**Check Index Status:**
```sql
SELECT * FROM pg_stat_neurondb_indexes;
```

---

---

## 🔗 Related Documentation

| Document | Description |
|----------|-------------|
| **[SQL API Reference](../../NeuronDB/docs/sql-api.md)** | Complete SQL API |
| **[Configuration Reference](../../NeuronDB/docs/configuration.md)** | Configuration options |
| **[Data Types Reference](../reference/data-types.md)** | Vector data types |
| **[Vector Search Guide](../../NeuronDB/docs/vector-search/indexing.md)** | Indexing guide |

---

<div align="center">

**Last Updated:** 2025-01-01  
**Documentation Version:** 1.0.0

[⬆ Back to Top](#neurondb-index-methods-complete-reference) · [📚 Internals Index](README.md) · [📚 Main Documentation](../../README.md)

</div>

