# NeuronDB Data Types Complete Reference

<div align="center">

**Complete reference for all data types, internal structures, and type system in NeuronDB**

[![Reference](https://img.shields.io/badge/reference-complete-brightgreen)](.)
[![Version](https://img.shields.io/badge/version-1.0-blue)](.)

</div>

---

> **Version:** 1.0  
> **PostgreSQL Compatibility:** 16, 17, 18  
> **Last Updated:** 2025-01-01

## Table of Contents

- [Vector Types](#vector-types)
- [Internal C Structures](#internal-c-structures)
- [Type Storage Formats](#type-storage-formats)
- [Type Casting Rules](#type-casting-rules)
- [Memory Layout](#memory-layout)
- [Quantization Formats](#quantization-formats)

---

## Vector Types

### `vector`

**PostgreSQL Type:** `vector`  
**C Structure:** `Vector`  
**Storage:** Extended (varlena)  
**Base Type:** Float32 (4 bytes per dimension)

#### Description

The main vector type in NeuronDB. It uses float32 precision. This is the primary type for storing embeddings and performing vector operations.

> [!NOTE]
> Use the `vector` type for most vector operations. It provides the best balance of precision and performance.

#### Structure

```c
typedef struct Vector {
    int32  vl_len_;     /* varlena header (required) */
    int16  dim;         /* number of dimensions */
    int16  unused;      /* padding for alignment */
    float4 data[FLEXIBLE_ARRAY_MEMBER];  /* vector data */
} Vector;
```

#### Memory Layout

```
Offset  Size    Field
------  ----    -----
0       4       vl_len_ (varlena header)
4       2       dim (dimension count)
6       2       unused (padding)
8       4*dim   data[] (float32 array)
```

**Total Size:** `offsetof(Vector, data) + sizeof(float4) * dim`

#### Limits

- **Maximum Dimensions:** 16,000
- **Minimum Dimensions:** 1
- **Storage Overhead:** 8 bytes (header + dimension)

#### Example

```sql
-- Create a vector
SELECT '[1.0, 2.0, 3.0]'::vector;

-- Create with dimension constraint
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    embedding vector(384)  -- Fixed 384 dimensions
);

-- Insert vector
INSERT INTO embeddings (embedding) VALUES ('[0.1, 0.2, 0.3]'::vector);
```

#### Macros

```c
#define VECTOR_SIZE(dim) (offsetof(Vector, data) + sizeof(float4) * (dim))
#define VECTOR_DIM(v) ((v)->dim)
#define VECTOR_DATA(v) ((v)->data)
#define DatumGetVector(x) ((Vector *)PG_DETOAST_DATUM(x))
#define PG_GETARG_VECTOR_P(x) DatumGetVector(PG_GETARG_DATUM(x))
#define PG_RETURN_VECTOR_P(x) PG_RETURN_POINTER(x)
```

---

### `halfvec`

**PostgreSQL Type:** `halfvec`  
**C Structure:** `VectorF16`  
**Storage:** Extended (varlena)  
**Base Type:** Float16 (2 bytes per dimension)

#### Description

Half-precision vector type. It provides 2x compression. It uses IEEE 754 half-precision floating point format.

> [!TIP]
> Use `halfvec` when you need to reduce storage size. It works well for large datasets where slight precision loss is acceptable.

#### Structure

```c
typedef struct VectorF16 {
    int32   vl_len_;    /* varlena header */
    int16   dim;        /* number of dimensions */
    uint16  data[FLEXIBLE_ARRAY_MEMBER];  /* FP16 data */
} VectorF16;
```

#### Memory Layout

```
Offset  Size      Field
------  ----      -----
0       4         vl_len_ (varlena header)
4       2         dim (dimension count)
6       2         unused (padding)
8       2*dim     data[] (FP16 array)
```

**Total Size:** `offsetof(VectorF16, data) + sizeof(uint16) * dim`

#### Limits

- **Maximum Dimensions:** 4,000
- **Compression Ratio:** 2x (compared to vector)
- **Precision:** ~3 decimal digits

#### Example

```sql
-- Convert vector to halfvec
SELECT vector_to_halfvec('[1.0, 2.0, 3.0]'::vector);

-- Cast between types
SELECT '[1.0, 2.0, 3.0]'::vector::halfvec;

-- Create table with halfvec
CREATE TABLE embeddings_fp16 (
    id SERIAL PRIMARY KEY,
    embedding halfvec(384)
);
```

#### Compatibility

- Implicit cast from `vector` to `halfvec`
- Assignment cast from `halfvec` to `vector`

---

### `sparsevec`

**PostgreSQL Type:** `sparsevec`  
**C Structure:** `SparseVector`  
**Storage:** Extended (varlena)  
**Base Type:** Sparse representation

#### Description

Sparse vector type storing only non-zero values. Optimized for high-dimensional vectors with many zeros.

#### Structure

```c
typedef struct SparseVector {
    int32   vl_len_;        /* varlena header */
    int32   vocab_size;     /* Vocabulary size */
    int32   nnz;            /* Number of non-zero entries */
    uint16  model_type;     /* 0=BM25, 1=SPLADE, 2=ColBERTv2 */
    uint16  flags;          /* Reserved */
    /* Followed by: int32 token_ids[nnz], float4 weights[nnz] */
} SparseVector;
```

#### Memory Layout

```
Offset          Size        Field
------          ----        -----
0               4           vl_len_ (varlena header)
4               4           vocab_size (vocabulary size)
8               4           nnz (non-zero count)
12              2           model_type
14              2           flags
16              4*nnz       token_ids[] (indices)
16+4*nnz        4*nnz       weights[] (values)
```

**Total Size:** `SPARSE_VEC_SIZE(nnz) = offsetof(SparseVector, flags) + sizeof(uint16) + sizeof(int32) * nnz + sizeof(float4) * nnz`

#### Limits

- **Maximum Non-Zero Entries:** 1,000
- **Maximum Dimensions:** 1,000,000
- **Model Types:**
  - `0`: BM25-style sparse retrieval
  - `1`: SPLADE
  - `2`: ColBERTv2

#### Example

```sql
-- Convert to sparse vector
SELECT vector_to_sparsevec('[0, 0, 1.5, 0, 2.3, 0]'::vector);

-- Create table with sparsevec
CREATE TABLE sparse_embeddings (
    id SERIAL PRIMARY KEY,
    embedding sparsevec
);
```

#### Macros

```c
#define SPARSE_VEC_TOKEN_IDS(sv) \
    ((int32 *)(((char *)(sv)) + sizeof(SparseVector)))
#define SPARSE_VEC_WEIGHTS(sv) \
    ((float4 *)(SPARSE_VEC_TOKEN_IDS(sv) + (sv)->nnz))
```

---

### `binaryvec`

**PostgreSQL Type:** `binaryvec`  
**C Structure:** `VectorBinary`  
**Storage:** Extended (varlena)  
**Base Type:** Binary (1 bit per dimension)

#### Description

Binary vector type for 32x compression using Hamming distance.

#### Structure

```c
typedef struct VectorBinary {
    int32  vl_len_;     /* varlena header */
    int16  dim;         /* number of bits */
    uint8  data[FLEXIBLE_ARRAY_MEMBER];  /* packed bits */
} VectorBinary;
```

#### Memory Layout

```
Offset  Size        Field
------  ----        -----
0       4           vl_len_ (varlena header)
4       2           dim (bit count)
6       2           unused (padding)
8       ceil(dim/8) data[] (packed bits)
```

**Total Size:** `offsetof(VectorBinary, data) + ceil(dim / 8)`

#### Limits

- **Compression Ratio:** 32x (compared to vector)
- **Distance Metric:** Hamming distance only

#### Example

```sql
-- Quantize to binary
SELECT vector_to_binary('[1.0, -1.0, 0.5, -0.5]'::vector);

-- Hamming distance
SELECT binaryvec_hamming_distance(v1, v2) FROM vectors;
```

---

### `vectorp`

**PostgreSQL Type:** `vectorp`  
**C Structure:** `VectorPacked`  
**Storage:** Extended (varlena)  
**Base Type:** Float32 with SIMD optimization

#### Description

Packed SIMD vector with optimized storage layout. Includes metadata for validation and portability.

#### Structure

```c
typedef struct VectorPacked {
    int32   vl_len_;        /* varlena header */
    uint32  fingerprint;    /* CRC32 of dimensions for validation */
    uint16  version;        /* Schema version */
    uint16  dim;            /* Number of dimensions */
    uint8   endian_guard;   /* 0x01 for little, 0x10 for big */
    uint8   flags;          /* Reserved for future use */
    uint16  unused;         /* Alignment padding */
    float4  data[FLEXIBLE_ARRAY_MEMBER];
} VectorPacked;
```

#### Memory Layout

```
Offset  Size    Field
------  ----    -----
0       4       vl_len_ (varlena header)
4       4       fingerprint (CRC32)
8       2       version (schema version)
10      2       dim (dimension count)
12      1       endian_guard (byte order)
13      1       flags (reserved)
14      2       unused (padding)
16      4*dim   data[] (float32 array)
```

#### Features

- **Dimension Fingerprint:** CRC32 checksum for validation
- **Endianness Guard:** Detects byte order for portability
- **Version Tag:** Supports schema evolution
- **SIMD Optimized:** Aligned for SIMD operations

#### Example

```sql
-- Create vectorp (typically used internally)
-- Direct creation not typically exposed to users
```

---

### `vecmap`

**PostgreSQL Type:** `vecmap`  
**C Structure:** `VectorMap`  
**Storage:** Extended (varlena)  
**Base Type:** Sparse map representation

#### Description

Sparse high-dimensional map storing only non-zero values. Optimized for very high dimensions (>10K).

#### Structure

```c
typedef struct VectorMap {
    int32  vl_len_;     /* varlena header */
    int32  total_dim;   /* Total dimensionality */
    int32  nnz;         /* Number of non-zero entries */
    /* Followed by parallel arrays: int32 indices[], float4 values[] */
} VectorMap;
```

#### Memory Layout

```
Offset          Size        Field
------          ----        -----
0               4           vl_len_ (varlena header)
4               4           total_dim (total dimensions)
8               4           nnz (non-zero count)
12              4*nnz       indices[] (dimension indices)
12+4*nnz        4*nnz       values[] (float32 values)
```

**Total Size:** `offsetof(VectorMap, nnz) + sizeof(int32) + sizeof(int32) * nnz + sizeof(float4) * nnz`

#### Use Case

Very high-dimensional vectors (>10K dimensions) with sparse data.

#### Example

```sql
-- Create vecmap (typically used internally)
-- Direct creation not typically exposed to users
```

---

### `vgraph`

**PostgreSQL Type:** `vgraph`  
**Storage:** Extended (varlena)  
**Description:** Graph-based vector structure for neighbor relations and clustering

#### Use Case

- Graph algorithms
- Community detection
- PageRank computation
- Neighbor relations

#### Example

```sql
-- Graph operations
SELECT * FROM vgraph_bfs(graph_data, 0, -1);
SELECT * FROM vgraph_pagerank(graph_data, 0.85, 100, 1e-6);
SELECT * FROM vgraph_community_detection(graph_data, 10);
```

---

### `rtext`

**PostgreSQL Type:** `rtext`  
**Storage:** Extended (varlena)  
**Description:** Retrievable text with token metadata for RAG pipelines

#### Features

- Token offsets for highlighting
- Section IDs for structured documents
- Language tag for multilingual support

#### Use Case

RAG pipelines requiring token-level metadata and highlighting.

---

## Internal C Structures

### Quantization Structures

#### `VectorI8`

**Description:** INT8 quantized vector (8x compression)

```c
typedef struct VectorI8 {
    int32  vl_len_;
    int16  dim;
    int8   data[FLEXIBLE_ARRAY_MEMBER];
} VectorI8;
```

#### `VectorU8`

**Description:** UINT8 quantized vector (8x compression, unsigned)

```c
typedef struct VectorU8 {
    int32  vl_len_;
    int16  dim;
    uint8  data[FLEXIBLE_ARRAY_MEMBER];
} VectorU8;
```

#### `VectorTernary`

**Description:** Ternary vector (16x compression, 2 bits per dimension)

```c
typedef struct VectorTernary {
    int32  vl_len_;
    int16  dim;         /* number of dimensions */
    uint8  data[FLEXIBLE_ARRAY_MEMBER];  /* Packed: 4 values per byte */
} VectorTernary;
```

#### `VectorI4`

**Description:** INT4 quantized vector (16x compression, 4 bits per dimension)

```c
typedef struct VectorI4 {
    int32  vl_len_;
    int16  dim;         /* number of dimensions */
    uint8  data[FLEXIBLE_ARRAY_MEMBER];  /* Packed: 2 values per byte */
} VectorI4;
```

---

## Type Storage Formats

### Varlena Format

All NeuronDB types use PostgreSQL's varlena format:

```
Byte 0:     Length bits (high bit = 1 for 4-byte length, 0 for 1-byte)
Bytes 1-4:  Length (if 4-byte format)
Bytes 5+:   Actual data
```

### Storage Classes

- **Extended:** Stored out-of-line (TOAST)
- **External:** Stored externally
- **Main:** Stored inline (if small enough)

All NeuronDB vector types use **Extended** storage.

---

## Type Casting Rules

### Implicit Casts

- `vector` → `halfvec` (implicit)
- `halfvec` → `vector` (assignment)

### Explicit Casts

```sql
-- Vector to halfvec
SELECT '[1.0, 2.0, 3.0]'::vector::halfvec;

-- Halfvec to vector
SELECT halfvec_value::vector;

-- Vector to sparsevec
SELECT vector_to_sparsevec('[1.0, 2.0, 3.0]'::vector);

-- Sparsevec to vector
SELECT sparsevec_to_vector(sparsevec_value);

-- Array to vector
SELECT array_to_vector(ARRAY[1.0, 2.0, 3.0]::real[]);

-- Vector to array
SELECT vector_to_array('[1.0, 2.0, 3.0]'::vector);
```

### Cast Functions

| Function | Description |
|----------|-------------|
| `vector_to_halfvec(vector)` | Convert vector to halfvec |
| `halfvec_to_vector(halfvec)` | Convert halfvec to vector |
| `vector_to_sparsevec(vector)` | Convert vector to sparsevec |
| `sparsevec_to_vector(sparsevec)` | Convert sparsevec to vector |
| `array_to_vector(real[])` | Convert array to vector |
| `vector_to_array(vector)` | Convert vector to array |
| `vector_cast_dimension(vector, integer)` | Change vector dimension |

---

## Memory Layout

### Vector Memory Layout

```
┌─────────────────────────────────────┐
│ vl_len_ (4 bytes)                   │
├─────────────────────────────────────┤
│ dim (2 bytes)                        │
├─────────────────────────────────────┤
│ unused (2 bytes)                     │
├─────────────────────────────────────┤
│ data[0] (4 bytes)                    │
│ data[1] (4 bytes)                    │
│ ...                                  │
│ data[dim-1] (4 bytes)                │
└─────────────────────────────────────┘
```

### Halfvec Memory Layout

```
┌─────────────────────────────────────┐
│ vl_len_ (4 bytes)                   │
├─────────────────────────────────────┤
│ dim (2 bytes)                        │
├─────────────────────────────────────┤
│ unused (2 bytes)                     │
├─────────────────────────────────────┤
│ data[0] (2 bytes, FP16)              │
│ data[1] (2 bytes, FP16)              │
│ ...                                  │
│ data[dim-1] (2 bytes, FP16)          │
└─────────────────────────────────────┘
```

### Sparse Vector Memory Layout

```
┌─────────────────────────────────────┐
│ vl_len_ (4 bytes)                    │
├─────────────────────────────────────┤
│ vocab_size (4 bytes)                 │
├─────────────────────────────────────┤
│ nnz (4 bytes)                        │
├─────────────────────────────────────┤
│ model_type (2 bytes)                 │
├─────────────────────────────────────┤
│ flags (2 bytes)                      │
├─────────────────────────────────────┤
│ token_ids[0] (4 bytes)                │
│ token_ids[1] (4 bytes)               │
│ ...                                  │
│ token_ids[nnz-1] (4 bytes)           │
├─────────────────────────────────────┤
│ weights[0] (4 bytes)                  │
│ weights[1] (4 bytes)                 │
│ ...                                  │
│ weights[nnz-1] (4 bytes)             │
└─────────────────────────────────────┘
```

---

## Quantization Formats

### INT8 Quantization

**Compression:** 8x  
**Range:** -128 to 127  
**Precision:** ~2-3 decimal digits

**Format:**
```c
typedef struct {
    int8 data[dim];
} VectorI8;
```

**Storage:** `dim` bytes

### FP16 Quantization

**Compression:** 2x  
**Format:** IEEE 754 half-precision  
**Precision:** ~3 decimal digits

**Format:**
```c
typedef struct {
    uint16 data[dim];  // FP16 values
} VectorF16;
```

**Storage:** `2 * dim` bytes

### Binary Quantization

**Compression:** 32x  
**Format:** 1 bit per dimension (0 or 1)  
**Distance:** Hamming distance only

**Format:**
```c
typedef struct {
    uint8 data[ceil(dim/8)];  // Packed bits
} VectorBinary;
```

**Storage:** `ceil(dim / 8)` bytes

### UINT8 Quantization

**Compression:** 4x  
**Range:** 0 to 255  
**Precision:** ~2 decimal digits

**Format:**
```c
typedef struct {
    uint8 data[dim];
} VectorU8;
```

**Storage:** `dim` bytes

### Ternary Quantization

**Compression:** 16x  
**Values:** -1, 0, +1  
**Format:** 2 bits per dimension

**Format:**
```c
typedef struct {
    uint8 data[ceil(dim/4)];  // Packed: 4 values per byte
} VectorTernary;
```

**Storage:** `ceil(dim / 4)` bytes

### INT4 Quantization

**Compression:** 8x  
**Range:** -8 to 7  
**Format:** 4 bits per dimension

**Format:**
```c
typedef struct {
    uint8 data[ceil(dim/2)];  // Packed: 2 values per byte
} VectorI4;
```

**Storage:** `ceil(dim / 2)` bytes

---

## Type Comparison

| Type | Dimensions | Storage/Dim | Compression | Use Case |
|------|-----------|-------------|-------------|----------|
| `vector` | Up to 16K | 4 bytes | 1x | General purpose |
| `halfvec` | Up to 4K | 2 bytes | 2x | Memory-constrained |
| `sparsevec` | Up to 1M | Variable | Variable | Sparse data |
| `binaryvec` | Variable | 1/8 byte | 32x | Binary features |
| `vectorp` | Variable | 4 bytes | 1x | SIMD-optimized |
| `vecmap` | >10K | Variable | Variable | Very high-dim sparse |

---

## Performance Considerations

### Memory Usage

- **vector:** `8 + 4 * dim` bytes
- **halfvec:** `8 + 2 * dim` bytes
- **sparsevec:** `16 + 8 * nnz` bytes (where nnz << dim)
- **binaryvec:** `8 + ceil(dim / 8)` bytes

### Type Selection Guide

1. **General Purpose:** Use `vector` for most use cases
2. **Memory Constrained:** Use `halfvec` for 2x compression
3. **Sparse Data:** Use `sparsevec` when >90% zeros
4. **Binary Features:** Use `binaryvec` for binary/categorical features
5. **Very High Dimensions:** Use `vecmap` for >10K dimensions

### Conversion Overhead

- **vector ↔ halfvec:** Fast (direct conversion)
- **vector ↔ sparsevec:** Moderate (requires scanning)
- **vector ↔ binaryvec:** Fast (threshold-based)

---

---

## 🔗 Related Documentation

| Document | Description |
|----------|-------------|
| **[SQL API Reference](../../NeuronDB/docs/sql-api.md)** | Complete SQL API |
| **[Configuration Reference](../../NeuronDB/docs/configuration.md)** | Configuration options |
| **[Index Methods](../internals/index-methods.md)** | Index implementation |
| **[Vector Search Documentation](../../NeuronDB/docs/vector-search/)** | Vector search guide |

---

<div align="center">

**Last Updated:** 2025-01-01  
**Documentation Version:** 1.0.0

[⬆ Back to Top](#neurondb-data-types-complete-reference) · [📚 Reference Index](README.md) · [📚 Main Documentation](../../README.md)

</div>

