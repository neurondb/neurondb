# Critical HNSW Index Fixes Applied

## Date: 2025-01-27

This document summarizes the critical fixes applied to address on-disk layout and parameter consistency issues in the HNSW indexing code.

## 1. m Parameter Consistency (CRITICAL FIX)

### Problem
The code used `HNSW_DEFAULT_M` (16) hardcoded in layout calculations (`HnswNodeSize`, `HnswGetNeighbors`) but validated against `meta->m` which could be different. This created a dangerous mismatch where:
- If user set `m != HNSW_DEFAULT_M`, node layout on disk would be wrong
- Array accesses would be out of bounds
- Index corruption would occur

### Solution
- **Created `HnswNodeSizeWithM(dim, level, m)`** - Uses m parameter explicitly
- **Created `HnswGetNeighborsSafe(node, level, m)`** - Uses m parameter explicitly
- **Updated `hnswComputeNodeSizeSafe`** - Now takes m as parameter
- **Updated all call sites** to pass `meta->m` from meta page:
  - `hnswbuild` - Uses options->m
  - `hnswinsert` - Uses metaPage->m
  - `hnswbulkdelete` - Uses meta->m
  - `hnswdelete` - Uses meta->m
  - `hnswRemoveNodeFromNeighbor` - Reads meta->m
  - `hnswSearch` - Uses metaPage->m
  - `hnswSearchLayerGreedy` (in hnsw_scan.c) - Now takes m parameter
  - `hnswSearchLayer0` (in hnsw_scan.c) - Now takes m parameter
- **Added reloptions validation** - Enforces m in range [HNSW_MIN_M, HNSW_MAX_M] at CREATE INDEX time

### Files Modified
- `NeuronDB/src/index/hnsw_am.c`
- `NeuronDB/src/scan/hnsw_scan.c`

## 2. Sparse Vector Zero Initialization (CRITICAL FIX)

### Problem
`hnswExtractVectorData` for sparsevec allocated result buffer but did not zero-initialize it. Only non-zero entries were populated, leaving uninitialized memory in other positions. Distance computations would read garbage values.

### Solution
- Added `memset(result, 0, sv->total_dim * sizeof(float4))` before populating non-zero entries
- Added detailed comment explaining why zero-initialization is critical

### Files Modified
- `NeuronDB/src/index/hnsw_am.c` (line ~1337)

## 3. OID Caching (PERFORMANCE FIX)

### Problem
`hnswExtractVectorData` called `LookupTypeNameOid` for vector, halfvec, sparsevec types on every single call. This was expensive in hot paths (insert, scan).

### Solution
- Added static cached OIDs: `cached_vectorOid`, `cached_halfvecOid`, `cached_sparsevecOid`, `cached_bitOid`
- Added `hnswCacheTypeOids()` function that initializes once
- Cache is checked on first call and reused thereafter

### Files Modified
- `NeuronDB/src/index/hnsw_am.c`

## 4. Validation with Held Locks (SAFETY FIX)

### Problem
`hnswValidateLevel` used `elog(ERROR)` which could be called while holding buffer locks. PostgreSQL will clean up, but mixing patterns is unsafe.

### Solution
- Created `hnswValidateLevelSafe()` - Returns false instead of raising ERROR
- Kept `hnswValidateLevel()` for non-lock contexts
- Updated all validation calls that occur while holding locks to use `hnswValidateLevelSafe()`

### Files Modified
- `NeuronDB/src/index/hnsw_am.c` (multiple locations)

## 5. Reloptions Validation (SAFETY FIX)

### Problem
No validation of m, ef_construction, ef_search parameter ranges. Users could set invalid values that would cause errors during first insert rather than at CREATE INDEX time.

### Solution
- Added range validation in `hnswoptions()` when `validate == true`:
  - `m`: [HNSW_MIN_M (2), HNSW_MAX_M (128)]
  - `ef_construction`: [HNSW_MIN_EF_CONSTRUCTION (4), HNSW_MAX_EF_CONSTRUCTION (10000)]
  - `ef_search`: [HNSW_MIN_EF_SEARCH (4), HNSW_MAX_EF_SEARCH (10000)]
  - `ef_construction >= m` and `ef_search >= m` checks
- Added constants: `HNSW_MIN_M`, `HNSW_MAX_M`, `HNSW_MIN_EF_CONSTRUCTION`, etc.

### Files Modified
- `NeuronDB/src/index/hnsw_am.c`

## 6. Documentation Improvements

### One-Node-Per-Page Assumption
Added prominent documentation in file header explaining that HNSW index uses ONE NODE PER PAGE as a fundamental design constraint. This is used throughout for:
- Page layout (PageIsEmpty checks)
- Node access (always FirstOffsetNumber)
- Neighbor removal
- Bulk delete

### Files Modified
- `NeuronDB/src/index/hnsw_am.c` (header comment)

## Known Limitations (Not Fixed - Design Decisions)

### Result Representation
Currently stores only `BlockNumber` in search results, then uses `FirstOffsetNumber` in `hnswgettuple`. This works with one-node-per-page design but is fragile if:
- Future optimization wants multiple nodes per page
- Vacuum compacts items

**Recommendation**: Consider storing `{BlockNumber, OffsetNumber}` struct in results for future-proofing.

### Cost Estimation
`hnswcostestimate` uses hardcoded values. This is acceptable for initial version but should be improved to use actual index statistics.

### Concurrency
Insertion logic uses multiple buffer locks but no global index lock. This is similar to other custom AMs but concurrency semantics are complex. Current design assumes single writer or careful coordination.

### Memory Scaling
`visitedSet` in `hnswSearch` scales linearly with index size. This is normal for HNSW but could be optimized with temporary contexts or bailout for very large indexes.

## Testing Recommendations

1. **Test with different m values**: Create indexes with m=8, m=16, m=32, m=64 and verify:
   - Index creation succeeds
   - Insertions work correctly
   - Searches return correct results
   - No crashes or corruption

2. **Test sparsevec**: Verify that sparse vectors with many zero entries compute distances correctly

3. **Test parameter validation**: Try creating indexes with invalid m, ef_construction, ef_search values and verify errors are caught at CREATE INDEX time

4. **Test concurrent operations**: If applicable, test concurrent inserts and searches

## Summary

All critical issues related to m parameter mismatch, sparsevec initialization, OID caching, validation safety, and reloptions validation have been fixed. The code now consistently uses `meta->m` throughout, ensuring on-disk layout matches validation logic.

