# pgvector Compatibility Matrix

This document tracks NeuronDB's compatibility with pgvector extension features, enabling smooth migration and ensuring feature parity for vector database operations.

## Status Legend

- **✅ Full**: Complete implementation matching pgvector behavior
- **✅ Enhanced**: Implementation exceeds pgvector (e.g., GPU acceleration, additional features)
- **⚠️ Partial**: Partially implemented, some limitations exist
- **❌ Not Yet**: Not yet implemented
- **🔵 Different**: Different implementation approach (documented differences)

## Types

| Feature | pgvector | NeuronDB Status | Notes |
|---------|----------|-----------------|-------|
| `vector` type | ✅ | ✅ Full | Full compatibility |
| `vector(n)` typmod | ✅ | ✅ Full | Dimension enforcement supported |
| Input format `[1,2,3]` | ✅ | ✅ Full | Array-style input |
| Input format `'[1,2,3]'::vector` | ✅ | ✅ Full | String casting |
| Output format | ✅ | ✅ Full | `[1,2,3]` format |
| Binary I/O | ✅ | ✅ Full | `vector_recv`/`vector_send` |
| Additional types | ❌ | ✅ Enhanced | `halfvec`, `sparsevec`, `binaryvec` beyond pgvector |

## Operators

| Operator | pgvector | NeuronDB Status | Notes |
|----------|----------|-----------------|-------|
| `<->` (L2 distance) | ✅ | ✅ Full | Euclidean distance |
| `<=>` (cosine distance) | ✅ | ✅ Full | Cosine distance (1 - similarity) |
| `<#>` (negative inner product) | ✅ | ✅ Full | For maximum inner product search |
| `=` (equality) | ✅ | ✅ Full | Exact equality check |
| `<`, `<=`, `>`, `>=` | ✅ | ✅ Full | Lexicographic comparison |
| `!=` or `<>` | ✅ | ✅ Full | Inequality |

## Functions

### Core Functions

| Function | pgvector | NeuronDB Status | Notes |
|----------|----------|-----------------|-------|
| `vector_dims(vector)` | ✅ | ✅ Full | Returns dimension count |
| `l2_norm(vector)` | ✅ | ✅ Full | L2 (Euclidean) norm |
| `vector_norm(vector)` | ❌ | ✅ Enhanced | Alias for `l2_norm` |
| `normalize_l2(vector)` | ✅ | ✅ Full | Normalize to unit length (via `vector_normalize`) |
| `l2_normalize(vector)` | ❌ | ✅ Enhanced | Compatibility alias |

### Distance Functions

| Function | pgvector | NeuronDB Status | Notes |
|----------|----------|-----------------|-------|
| `l2_distance(vector, vector)` | ✅ | ✅ Full | Compatibility alias |
| `cosine_distance(vector, vector)` | ✅ | ✅ Full | Compatibility alias |
| `inner_product(vector, vector)` | ✅ | ✅ Full | Compatibility alias |
| `vector_l2_distance(vector, vector)` | ❌ | ✅ Enhanced | Canonical name |
| `vector_cosine_distance(vector, vector)` | ❌ | ✅ Enhanced | Canonical name |
| `vector_inner_product(vector, vector)` | ❌ | ✅ Enhanced | Canonical name |

### Array Conversions

| Function | pgvector | NeuronDB Status | Notes |
|----------|----------|-----------------|-------|
| `vector_to_array(vector)` | ✅ | ✅ Full | Convert to `real[]` |
| `array_to_vector(real[])` | ✅ | ✅ Full | Convert from `real[]` |
| `array_to_vector(double precision[])` | ❌ | ✅ Enhanced | Additional cast support |
| `array_to_vector(integer[])` | ❌ | ✅ Enhanced | Additional cast support |
| `array_to_vector(numeric[])` | ❌ | ✅ Enhanced | Additional cast support |

### Subvector Operations

| Function | pgvector | NeuronDB Status | Notes |
|----------|----------|-----------------|-------|
| `subvector(vector, start, count)` | ✅ | ✅ Full | 1-based start, count (compatibility) |
| `vector_slice(vector, start, end)` | ❌ | ✅ Enhanced | 0-based start, exclusive end (canonical) |

**Note**: pgvector uses 1-based indexing with count: `subvector(vec, 1, 3)` extracts first 3 elements.  
NeuronDB also supports 0-based indexing: `vector_slice(vec, 0, 3)` extracts elements 0-2.

## Aggregates

| Aggregate | pgvector | NeuronDB Status | Notes |
|-----------|----------|-----------------|-------|
| `avg(vector)` | ✅ | ✅ Full | Element-wise average |
| `sum(vector)` | ✅ | ✅ Full | Element-wise sum |
| `vector_avg(vector)` | ❌ | ✅ Enhanced | Canonical name |
| `vector_sum(vector)` | ❌ | ✅ Enhanced | Canonical name |

## Indexes

### HNSW Index

| Feature | pgvector | NeuronDB Status | Notes |
|---------|----------|-----------------|-------|
| Access method `hnsw` | ✅ | ✅ Full | `CREATE INDEX USING hnsw` |
| Operator class `vector_l2_ops` | ✅ | ✅ Full | L2 distance indexing |
| Operator class `vector_cosine_ops` | ✅ | ✅ Full | Cosine distance indexing |
| Operator class `vector_ip_ops` | ✅ | ✅ Full | Inner product indexing |
| Index option `m` | ✅ | ✅ Full | Number of bi-directional links (default: 16) |
| Index option `ef_construction` | ✅ | ✅ Full | Search width during construction (default: 64) |
| Query parameter `ef_search` | ✅ | ⚠️ Partial | Via GUC or function parameter, not index option |

### IVF Index

| Feature | pgvector | NeuronDB Status | Notes |
|---------|----------|-----------------|-------|
| Access method `ivfflat` | ✅ | ✅ Full | NeuronDB uses `ivf` (same functionality) |
| Access method `ivf` | ❌ | ✅ Enhanced | Canonical name in NeuronDB |
| Operator class `vector_l2_ops` | ✅ | ✅ Full | L2 distance indexing |
| Operator class `vector_cosine_ops` | ✅ | ✅ Full | Cosine distance indexing |
| Operator class `vector_ip_ops` | ✅ | ✅ Full | Inner product indexing (now supported) |
| Index option `lists` | ✅ | ✅ Full | Number of clusters (default: 100) |
| Query parameter `probes` | ✅ | ⚠️ Partial | Via GUC or function parameter, not index option |

**Note**: IVF index now fully supports all three operator classes (L2, cosine, and inner product) matching pgvector parity.

### Index Creation Examples

**pgvector style:**
```sql
CREATE INDEX ON items USING hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX ON items USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
```

**NeuronDB (fully compatible):**
```sql
CREATE INDEX ON items USING hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX ON items USING ivf (embedding vector_l2_ops) WITH (lists = 100);
```

**NeuronDB enhanced (ivfflat alias supported via compatibility):**
```sql
CREATE INDEX ON items USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
```

## Query Patterns

### Basic K-NN Search

**pgvector:**
```sql
SELECT * FROM items 
ORDER BY embedding <-> '[1,2,3]'::vector 
LIMIT 10;
```

**NeuronDB:** ✅ **Identical**

### Cosine Similarity Search

**pgvector:**
```sql
SELECT * FROM items 
ORDER BY embedding <=> '[1,2,3]'::vector 
LIMIT 10;
```

**NeuronDB:** ✅ **Identical**

### Inner Product Search

**pgvector:**
```sql
SELECT * FROM items 
ORDER BY embedding <#> '[1,2,3]'::vector 
LIMIT 10;
```

**NeuronDB:** ✅ **Identical** (requires `vector_ip_ops` for IVF)

### Filtered Search

**pgvector:**
```sql
SELECT * FROM items 
WHERE category = 'electronics'
ORDER BY embedding <-> '[1,2,3]'::vector 
LIMIT 10;
```

**NeuronDB:** ✅ **Identical** (with enhanced planner support)

### Distance in SELECT

**pgvector:**
```sql
SELECT id, embedding <-> '[1,2,3]'::vector AS distance
FROM items
ORDER BY distance
LIMIT 10;
```

**NeuronDB:** ✅ **Identical**

## Casts

| Cast | pgvector | NeuronDB Status | Notes |
|------|----------|-----------------|-------|
| `real[]` → `vector` | ✅ | ✅ Full | Assignment cast |
| `vector` → `real[]` | ✅ | ✅ Full | Assignment cast |
| `double precision[]` → `vector` | ❌ | ✅ Enhanced | Additional cast |
| `integer[]` → `vector` | ❌ | ✅ Enhanced | Additional cast |
| `numeric[]` → `vector` | ❌ | ✅ Enhanced | Additional cast |

## Enhanced Features (Beyond pgvector)

NeuronDB provides additional features not in pgvector:

1. **GPU Acceleration**: GPU-accelerated distance computation and index search
2. **Additional Vector Types**: `halfvec` (FP16), `sparsevec`, `binaryvec`
3. **Quantization**: INT8, FP16, binary, ternary quantization support
4. **Hybrid Search**: Dense + sparse vector hybrid search
5. **Index Tuning**: Automated index parameter tuning
6. **Advanced Analytics**: ML functions, drift detection, clustering
7. **Operational Features**: RLS, tenant quotas, metrics, monitoring

## Known Limitations / Differences

1. **Index Option vs GUC**: Some query-time parameters (`ef_search`, `probes`) are controlled via GUCs or function parameters rather than index options (may affect planner behavior)
2. **Access Method Name**: NeuronDB uses `ivf` while pgvector uses `ivfflat` (compatibility alias exists via `CREATE INDEX ... USING ivfflat` which maps to `ivf`)

## Migration Guide

### From pgvector to NeuronDB

1. **Drop pgvector extension:**
   ```sql
   DROP EXTENSION vector;
   ```

2. **Install NeuronDB extension:**
   ```sql
   CREATE EXTENSION neurondb;
   ```

3. **Recreate indexes** (recommended, but existing data works):
   ```sql
   -- HNSW (same syntax)
   CREATE INDEX ON items USING hnsw (embedding vector_l2_ops);
   
   -- IVF (note: use 'ivf' instead of 'ivfflat', but 'ivfflat' alias works too)
   CREATE INDEX ON items USING ivf (embedding vector_l2_ops);
   ```

4. **Query syntax remains identical** - no code changes needed for basic operations

5. **Optional: Enable GPU acceleration:**
   ```sql
   SELECT neurondb_gpu_enable();
   ```

### Testing Compatibility

Run this query to verify basic operations:
```sql
-- Create test table
CREATE TABLE test_vectors (id int, embedding vector(3));

-- Insert test data
INSERT INTO test_vectors VALUES 
  (1, '[1,2,3]'),
  (2, '[4,5,6]'),
  (3, '[7,8,9]');

-- Test operators
SELECT id, embedding <-> '[1,2,3]' AS l2_dist,
       embedding <=> '[1,2,3]' AS cosine_dist,
       embedding <#> '[1,2,3]' AS ip_dist
FROM test_vectors
ORDER BY embedding <-> '[1,2,3]'
LIMIT 10;

-- Test functions
SELECT vector_dims(embedding), l2_norm(embedding), vector_normalize(embedding)
FROM test_vectors LIMIT 1;

-- Test aggregates
SELECT avg(embedding), sum(embedding) FROM test_vectors;
```

## Version History

- **2024-12**: Initial compatibility matrix created
- **2024-12**: Added IVF inner product (`vector_ip_ops`) support - full parity achieved
- **Status**: Full pgvector parity + enhancements for NeuronDB 2.0+

## References

- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [NeuronDB Vector Search Documentation](../vector-search/)
- [NeuronDB Indexing Guide](../vector-search/indexing.md)

