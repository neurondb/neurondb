# NeuronDB Vector Type Contract

## Overview
This document defines the invariants, contracts, and behavioral guarantees for all vector types in NeuronDB.

## Vector Families

### 1. Dense Vector (`Vector`)
- **Type**: `vector` (PostgreSQL type)
- **Header**: `[NeuronDB/include/neurondb.h](/home/pge/pge/neurondb/NeuronDB/include/neurondb.h)`
- **Implementation**: `[NeuronDB/src/core/neurondb.c](/home/pge/pge/neurondb/NeuronDB/src/core/neurondb.c)`
- **Structure**:
  ```c
  typedef struct Vector {
      int32  vl_len_;  // varlena header
      int16  dim;      // number of dimensions
      int16  unused;  // padding for alignment
      float4 data[FLEXIBLE_ARRAY_MEMBER];
  } Vector;
  ```
- **Max Dimensions**: `VECTOR_MAX_DIM` (16000)
- **Element Type**: `float4` (IEEE 754 single precision)
- **Memory Layout**: Varlena with `VARSIZE(vec) == offsetof(Vector, data) + sizeof(float4) * dim`
- **Alignment**: 4-byte aligned (float4)

### 2. Packed Vector (`VectorPacked`)
- **Type**: `vectorp` (PostgreSQL type)
- **Header**: `[NeuronDB/include/neurondb_types.h](/home/pge/pge/neurondb/NeuronDB/include/neurondb_types.h)`
- **Implementation**: `[NeuronDB/src/vector/vector_types.c](/home/pge/pge/neurondb/NeuronDB/src/vector/vector_types.c)`
- **Features**: CRC32 fingerprint, version tag, endianness guard
- **Max Dimensions**: Same as `Vector` (16000)

### 3. Sparse Vector (`VectorMap` / `sparsevec`)
- **Type**: `sparse_vector` (PostgreSQL type)
- **Header**: `[NeuronDB/include/neurondb_types.h](/home/pge/pge/neurondb/NeuronDB/include/neurondb_types.h)`
- **Implementation**: `[NeuronDB/src/vector/vector_types.c](/home/pge/pge/neurondb/NeuronDB/src/vector/vector_types.c)`
- **Structure**: Stores only non-zero values with indices
- **Max Dimensions**: 1,000,000 (sparsevec), 1000 non-zero entries
- **Index Format**: 1-based in SQL I/O, 0-based internally

### 4. Quantized Vectors
- **FP16** (`VectorF16`): 2x compression, max 4000 dims
- **INT8** (`VectorI8`): 4x compression, requires min/max vectors
- **Binary** (`VectorBinary`): 32x compression, bit-packed
- **UINT8** (`VectorU8`): 8x compression, unsigned
- **Ternary** (`VectorTernary`): 16x compression, 2 bits per dim
- **INT4** (`VectorI4`): 16x compression, 4 bits per dim

## Invariants

### Dimension Limits
- **Dense/Packed**: `1 <= dim <= VECTOR_MAX_DIM` (16000)
- **Sparse**: `1 <= total_dim <= 1000000`, `0 <= nnz <= 1000`
- **Halfvec**: `1 <= dim <= 4000`
- All dimension checks must use `VECTOR_MAX_DIM` constant, not hardcoded values

### Memory Safety
- All vectors are varlena types; must validate `VARSIZE_ANY(vec)` matches expected size
- Size validation: `VARSIZE_ANY(vec) >= offsetof(Vector, data) + sizeof(float4) * dim`
- Alignment: Vectors must be properly aligned for SIMD operations (16-byte minimum for AVX2)

### Numeric Values
- **NaN Policy**: NaN values are rejected during input parsing (`vector_in_internal` with `check=true`)
- **Infinity Policy**: Infinity values are rejected during input parsing
- **Zero Vectors**: Allowed; normalization returns zero vector or errors (implementation-dependent)
- **Comparison Tolerance**: `vector_eq` uses `1e-6` tolerance for float comparison

### NULL Semantics
- **SQL Level**: Functions marked `STRICT` handle NULL automatically
- **C Level**: Functions must check `PG_ARGISNULL(n)` before `PG_GETARG_*` if not STRICT
- **Validation**: `NDB_CHECK_VECTOR_VALID` checks for NULL pointer; SQL STRICT handles NULL Datum

### Serialization
- **Binary Format**: `vector_recv`/`vector_send` use PostgreSQL binary protocol
  - Format: `int16 dim` followed by `dim * float4` values
- **Text Format**: `[val1,val2,...,valN]` with optional whitespace
- **Endianness**: Little-endian for binary (with guard in `VectorPacked`)

### Type Modifiers
- PostgreSQL type modifiers can specify expected dimension: `vector(768)`
- Input functions validate dimension matches typmod if provided
- Dimension mismatch raises `ERRCODE_DATA_EXCEPTION`

## Validation Requirements

### `NDB_CHECK_VECTOR_VALID` Contract
Must validate:
1. Pointer is not NULL
2. `dim > 0 && dim <= VECTOR_MAX_DIM`
3. (Optional but recommended) `VARSIZE_ANY(vec) >= offsetof(Vector, data) + sizeof(float4) * dim`

### Function-Level Validation
- All vector input functions must validate dimensions before allocation
- All distance/operation functions must validate dimensions match
- All index access must validate bounds: `0 <= idx < dim`

## Portability Requirements

### SIMD
- AVX2: Requires `__AVX2__` compile-time flag
- AVX-512: Requires `__AVX512F__` compile-time flag
- FMA: Requires `__FMA__` or runtime detection (not just AVX2)
- Fallback: All SIMD functions must have scalar fallback

### Strict Aliasing
- FP16 conversions must use `memcpy`, not pointer casts
- All type punning must go through union or memcpy

### Endianness
- Binary serialization assumes little-endian
- `VectorPacked` includes endianness guard for validation

## Performance Contracts

### Distance Functions
- SIMD-optimized versions used when available and dimension >= threshold (8 for AVX2, 16 for AVX-512)
- Scalar fallback for small vectors or unsupported CPUs
- Kahan summation for L2 distance to maintain precision

### Quantization
- FP16: Lossy, preserves ~3-4 decimal digits
- INT8: Requires per-dimension min/max; quantization error depends on range
- Binary: Threshold-based (positive = 1, non-positive = 0)

## Error Codes
- `ERRCODE_INVALID_PARAMETER_VALUE`: Invalid dimension, out of range
- `ERRCODE_NULL_VALUE_NOT_ALLOWED`: NULL vector where not allowed
- `ERRCODE_DATA_EXCEPTION`: Dimension mismatch, corrupted data
- `ERRCODE_ARRAY_SUBSCRIPT_ERROR`: Index out of bounds
- `ERRCODE_INVALID_TEXT_REPRESENTATION`: Parse errors
- `ERRCODE_DATA_CORRUPTED`: Invalid varlena size, fingerprint mismatch

## Testing Requirements
- All vector types must have I/O roundtrip tests (text/binary)
- Edge cases: zero vectors, max dimension, NaN/Inf rejection
- Fuzz testing for parsers
- NULL handling tests (STRICT vs non-STRICT)
- Dimension mismatch tests
- Corrupted varlena detection tests



