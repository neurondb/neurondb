# Embedding Compatibility Guide

<div align="center">

**Vector dimensions, storage layout, memory behavior, and performance characteristics**

[![Reference](https://img.shields.io/badge/reference-complete-brightgreen)](.)
[![Dimensions](https://img.shields.io/badge/dimensions-1--16K-blue)](.)

</div>

---

> [!NOTE]
> This guide covers embedding compatibility, storage, and performance. Use it to plan your vector dimensions and storage requirements.

## Supported Vector Dimensions

### Standard Dimensions

NeuronDB supports vector dimensions from **1 to 16,000** dimensions.

**Common embedding model dimensions:**

| Model | Dimensions | Use Case |
|-------|------------|----------|
| `all-MiniLM-L6-v2` | 384 | Fast, general-purpose |
| `all-mpnet-base-v2` | 768 | Higher quality, general-purpose |
| `text-embedding-ada-002` | 1536 | OpenAI embeddings |
| `text-embedding-3-small` | 1536 | OpenAI (small) |
| `text-embedding-3-large` | 3072 | OpenAI (large) |
| `multilingual-e5-base` | 768 | Multilingual |
| `paraphrase-multilingual-mpnet-base-v2` | 768 | Multilingual |

### Dimension Limits

- **Minimum:** 1 dimension
- **Maximum:** 16,000 dimensions
- **Recommended:** 128-4096 dimensions for optimal performance
- **Performance impact:** Higher dimensions = slower queries, more memory

## Storage Layout

### Vector Type Storage

**`vector(n)` type:**
- **Storage:** 4 bytes per dimension (float32)
- **Overhead:** 8 bytes header (dimension count + padding)
- **Total size:** `8 + (n * 4)` bytes per vector

**Example sizes:**
- 128 dimensions: 520 bytes
- 384 dimensions: 1,544 bytes
- 768 dimensions: 3,080 bytes
- 1536 dimensions: 6,152 bytes

### TOAST Behavior

PostgreSQL automatically uses TOAST (The Oversized-Attribute Storage Technique) for large values.

**TOAST thresholds:**
- **Inline storage:** Vectors < 2KB (512 dimensions)
- **Extended storage:** Vectors >= 2KB (512+ dimensions)
- **Compression:** Enabled by default for extended storage

**Impact:**
- Smaller vectors (< 512 dims): Stored inline, faster access
- Larger vectors (>= 512 dims): TOAST storage, slightly slower but compressed

### Memory Layout

**In-memory representation:**
- Vectors stored as contiguous float32 arrays
- Aligned to 8-byte boundaries for SIMD operations
- GPU transfers use same layout (zero-copy when possible)

## Memory Behavior

### Per-Vector Memory

**Storage size:**
- On-disk: `8 + (n * 4)` bytes
- In-memory: `8 + (n * 4)` bytes (plus PostgreSQL tuple overhead)

**Example for 1M vectors (768 dims):**
- On-disk: ~3.08 GB
- In-memory: ~3.08 GB (plus ~10% overhead) = ~3.4 GB

### Index Memory

**HNSW index:**
- Memory: ~3-4x vector data size
- Example: 3.4 GB vectors → ~10-14 GB index memory

**IVF index:**
- Memory: ~1.5-2x vector data size
- Example: 3.4 GB vectors → ~5-7 GB index memory

### Batch Operations

**Batch embedding generation:**
- Memory scales linearly with batch size
- Recommended batch size: 32-128 texts
- GPU batches: 100-1000 texts (depending on GPU memory)

## Limits and Performance Cliffs

### Hard Limits

| Limit | Value | Notes |
|-------|-------|-------|
| **Max dimensions** | 16,000 | Hard limit, cannot exceed |
| **Max vectors per table** | Unlimited | Limited by PostgreSQL (practically billions) |
| **Max index size** | ~2TB | PostgreSQL limit |
| **Batch size** | 10,000 | Recommended max for batch operations |

### Performance Cliffs

**Dimension thresholds:**

| Dimensions | Performance | Notes |
|------------|-------------|-------|
| < 128 | Very fast | Optimal for high-QPS scenarios |
| 128-512 | Fast | Good balance of quality and speed |
| 512-1024 | Moderate | TOAST storage kicks in |
| 1024-2048 | Slower | Higher memory, slower queries |
| > 2048 | Slow | Consider dimensionality reduction |

**Query performance (approximate QPS on CPU):**
- 128 dims: 1,200-1,500 QPS
- 384 dims: 800-1,000 QPS
- 768 dims: 500-700 QPS
- 1536 dims: 300-400 QPS

### Memory Cliffs

**TOAST threshold (512 dimensions):**
- Below: Inline storage, faster
- Above: Extended storage, compressed, slightly slower

**Index build memory:**
- HNSW: Requires 4-5x data size during build
- Large datasets may require significant RAM

## Migration Between Embedding Models

### Changing Dimensions

**Scenario:** Migrating from 384-dim to 768-dim embeddings.

**Process:**
1. Create new column with target dimension
2. Generate new embeddings
3. Build new index
4. Update application to use new column
5. Drop old column when ready

```sql
-- Step 1: Add new column
ALTER TABLE documents 
ADD COLUMN embedding_new vector(768);

-- Step 2: Generate new embeddings
UPDATE documents 
SET embedding_new = embed_text(content, 'all-mpnet-base-v2')
WHERE embedding_new IS NULL;

-- Step 3: Build new index
CREATE INDEX documents_embedding_new_idx 
ON documents 
USING hnsw (embedding_new vector_cosine_ops);

-- Step 4: Update application code
-- Step 5: Drop old column (after verification)
-- ALTER TABLE documents DROP COLUMN embedding;
```

### Dimension Compatibility

**Important:** Vectors of different dimensions are **not compatible** for distance calculations.

```sql
-- This will ERROR:
SELECT vector_384 <=> vector_768;  -- ERROR: dimension mismatch

-- Must use same dimensions:
SELECT vector_384_a <=> vector_384_b;  -- OK
```

### Embedding Provider Migration

**Scenario:** Switching from OpenAI to HuggingFace embeddings.

**Process:**
1. Configure new provider
2. Regenerate embeddings (if dimensions differ)
3. Update queries to use new embeddings

```sql
-- Step 1: Configure new provider
SELECT neurondb.set_llm_config(
  'huggingface',
  NULL,  -- no API key needed for local
  NULL
);

-- Step 2: Regenerate embeddings (if needed)
UPDATE documents 
SET embedding = embed_text(content, 'all-mpnet-base-v2')
WHERE embedding IS NULL;

-- Step 3: Rebuild index if dimensions changed
REINDEX INDEX documents_embedding_idx;
```

## Best Practices

### Dimension Selection

1. **Start with 384 dimensions:** Good balance of quality and performance
2. **Upgrade to 768 if needed:** Better quality for complex queries
3. **Use 1536+ sparingly:** Only for highest quality requirements
4. **Consider model quality:** Higher dimensions don't always mean better

### Storage Optimization

1. **Use appropriate dimensions:** Don't use more than needed
2. **Monitor TOAST usage:** Vectors > 512 dims use TOAST
3. **Consider compression:** TOAST compression helps with large vectors
4. **Partition large tables:** Split by dimension if mixing models

### Performance Optimization

1. **Batch embedding generation:** Use `embed_text_batch` for efficiency
2. **Cache embeddings:** Use `embed_cached` for repeated texts
3. **Index appropriately:** HNSW for high recall, IVF for speed
4. **Monitor memory:** Watch index memory usage

## Compatibility Matrix

| Feature | 128 dims | 384 dims | 768 dims | 1536 dims |
|--------|----------|----------|----------|-----------|
| **Storage (inline)** | ✅ | ✅ | ⚠️ TOAST | ⚠️ TOAST |
| **Query QPS (CPU)** | 1,200+ | 800+ | 500+ | 300+ |
| **Query QPS (GPU)** | 3,500+ | 2,000+ | 1,200+ | 700+ |
| **Index build time** | Fast | Moderate | Slow | Very slow |
| **Memory per 1M vectors** | ~520 MB | ~1.5 GB | ~3.1 GB | ~6.2 GB |
| **Recommended use** | High QPS | General | Quality | Highest quality |

---

## 🔗 Related Documentation

| Document | Description |
|----------|-------------|
| **[Vector Types](../../NeuronDB/docs/vector-search/vector-types.md)** | Vector type details |
| **[Data Types Reference](data-types.md)** | Complete data types reference |
| **[Embedding Generation](../../NeuronDB/docs/ml-embeddings/embedding-generation.md)** | How to generate embeddings |
| **[Indexing Guide](../../NeuronDB/docs/vector-search/indexing.md)** | Index configuration |
| **[Performance Tuning](../../NeuronDB/docs/configuration.md)** | Performance optimization |

---

<div align="center">

[⬆ Back to Top](#embedding-compatibility-guide) · [📚 Reference Index](README.md) · [📚 Main Documentation](../../README.md)

</div>

