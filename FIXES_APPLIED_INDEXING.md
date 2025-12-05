# Indexing Code Crash Fixes - Applied

## Summary
All critical and high-priority crash potential issues identified in the indexing code have been fixed. This document summarizes all fixes applied.

---

## Fixes Applied

### 1. ✅ Array Bounds Validation for neighborCount
**Files**: `hnsw_am.c`, `hnsw_scan.c`

**Fix**: Added `hnswValidateNeighborCount()` helper function that validates and clamps `neighborCount[level]` values to prevent array bounds violations.

**Locations Fixed**:
- `hnswSearch()` - Lines 1183, 1243
- `hnswInsertNode()` - Lines 1500, 1763, 2212
- `hnswbulkdelete()` - Lines 553, 583
- `hnswdelete()` - Lines 2545, 2572
- `hnswRemoveNodeFromNeighbor()` - Line 2460
- `hnswSearchLayerGreedy()` - Line 508
- `hnswSearchLayer0()` - Line 685

**Implementation**:
```c
static int16
hnswValidateNeighborCount(int16 neighborCount, int m, int level)
{
    int16 maxNeighbors = m * 2;
    if (neighborCount < 0) return 0;
    if (neighborCount > maxNeighbors) return maxNeighbors;
    return neighborCount;
}
```

---

### 2. ✅ visitedSet Array Bounds Protection
**File**: `hnsw_am.c`

**Fix**: Added comprehensive bounds checking before all `visitedSet` array accesses. All accesses now validate block numbers against `RelationGetNumberOfBlocks(index)`.

**Locations Fixed**:
- Line 1229-1230: Entry point visitedSet update
- Line 1256-1257: Neighbor visitedSet check
- Line 1269-1270: Neighbor visitedSet update

**Implementation**: All `visitedSet[block]` accesses now check:
```c
if (block < RelationGetNumberOfBlocks(index))
    visitedSet[block] = true;
```

---

### 3. ✅ Integer Overflow Protection in Node Size Calculation
**File**: `hnsw_am.c`

**Fix**: Replaced `HnswNodeSize()` macro usage with `hnswComputeNodeSizeSafe()` function that performs overflow checking.

**Location Fixed**: `hnswInsertNode()` - Line 1457

**Implementation**:
```c
static Size
hnswComputeNodeSizeSafe(int dim, int level, bool *overflow)
{
    // Validates vectorSize, neighborSize, and totalSize for overflow
    // Returns 0 and sets *overflow=true if overflow detected
}
```

---

### 4. ✅ Level Value Validation
**File**: `hnsw_am.c`, `hnsw_scan.c`

**Fix**: Added `hnswValidateLevel()` helper function that validates all level values against `HNSW_MAX_LEVEL`.

**Locations Fixed**:
- `hnswSearch()` - Entry level validation, node level checks
- `hnswInsertNode()` - Level validation after random generation
- `hnswbulkdelete()` - Node level validation
- `hnswdelete()` - Node level validation
- `hnswRemoveNodeFromNeighbor()` - Level parameter validation
- `hnswSearchLayerGreedy()` - Node level validation
- `hnswSearchLayer0()` - Node level validation

**Implementation**:
```c
static bool
hnswValidateLevel(int level)
{
    if (level < 0 || level >= HNSW_MAX_LEVEL) {
        elog(ERROR, ...);
        return false;
    }
    return true;
}
```

---

### 5. ✅ NULL Pointer Checks After PageGetItem
**Files**: `hnsw_am.c`, `hnsw_scan.c`

**Fix**: Added NULL checks after all `PageGetItem()` calls before dereferencing node pointers.

**Locations Fixed** (20+ locations):
- All `PageGetItem()` calls in `hnswSearch()`
- All `PageGetItem()` calls in `hnswInsertNode()`
- All `PageGetItem()` calls in `hnswbulkdelete()`
- All `PageGetItem()` calls in `hnswdelete()`
- All `PageGetItem()` calls in `hnswRemoveNodeFromNeighbor()`
- All `PageGetItem()` calls in `hnswgettuple()`
- All `PageGetItem()` calls in `hnswSearchLayerGreedy()`
- All `PageGetItem()` calls in `hnswSearchLayer0()`

**Pattern Applied**:
```c
node = (HnswNode) PageGetItem(...);
if (node == NULL) {
    UnlockReleaseBuffer(buf);
    continue;  // or return/break as appropriate
}
```

---

### 6. ✅ Block Number Validation Before ReadBuffer
**Files**: `hnsw_am.c`, `hnsw_scan.c`

**Fix**: Added `hnswValidateBlockNumber()` helper function and applied validation before all `ReadBuffer()` calls.

**Locations Fixed** (15+ locations):
- All neighbor block accesses in `hnswSearch()`
- All neighbor block accesses in `hnswInsertNode()`
- All neighbor block accesses in `hnswbulkdelete()`
- All neighbor block accesses in `hnswdelete()`
- All neighbor block accesses in `hnswSearchLayerGreedy()`
- All neighbor block accesses in `hnswSearchLayer0()`

**Implementation**:
```c
static bool
hnswValidateBlockNumber(BlockNumber blkno, Relation index)
{
    if (blkno == InvalidBlockNumber) return false;
    if (blkno >= RelationGetNumberOfBlocks(index)) return false;
    return true;
}
```

---

### 7. ✅ candidateCount Validation Against efSearch
**File**: `hnsw_am.c`

**Fix**: Added explicit check to ensure `candidateCount` never exceeds `efSearch` in the candidate selection loop.

**Location Fixed**: `hnswSearch()` - Line 1233, 1279

**Implementation**: Loop condition ensures `candidateCount < efSearch` before adding candidates, and worst candidate replacement logic validates bounds.

---

### 8. ✅ visited Array Reallocation Limits
**File**: `hnsw_am.c`

**Fix**: Added maximum capacity limit (`HNSW_MAX_VISITED_CAPACITY = 1M`) to prevent excessive memory allocation.

**Location Fixed**: `hnswSearch()` - Line 1272-1277

**Implementation**:
```c
#define HNSW_MAX_VISITED_CAPACITY (1024 * 1024)  /* 1M entries max */

if (visitedCount >= visitedCapacity) {
    if (visitedCapacity >= HNSW_MAX_VISITED_CAPACITY) {
        // Stop expanding, continue without adding more
    } else {
        int newCapacity = Min(visitedCapacity * 2, HNSW_MAX_VISITED_CAPACITY);
        // Reallocate
    }
}
```

---

### 9. ✅ Integer Overflow in visitedSet Allocation
**File**: `hnsw_am.c`

**Fix**: Added overflow checking when calculating `visitedSet` allocation size.

**Location Fixed**: `hnswSearch()` - Line 1157

**Implementation**:
```c
BlockNumber numBlocks = RelationGetNumberOfBlocks(index);
size_t visitedSetSize = (size_t) numBlocks * sizeof(bool);
if (visitedSetSize / sizeof(bool) != (size_t) numBlocks) {
    ereport(ERROR, ...);  // Overflow detected
}
NDB_ALLOC(visitedSet, bool, numBlocks);
```

---

### 10. ✅ Entry Point Validation
**File**: `hnsw_am.c`

**Fix**: Added validation of block number before setting as entry point.

**Location Fixed**: `hnswInsertNode()` - Line 2343

**Implementation**:
```c
if (hnswValidateBlockNumber(blkno, index)) {
    metaPage->entryPoint = blkno;
    metaPage->entryLevel = level;
} else {
    ereport(ERROR, ...);
}
```

---

### 11. ✅ Vector Pointer Validation
**Files**: `hnsw_am.c`, `hnsw_scan.c`

**Fix**: Added NULL checks after `HnswGetVector()` calls.

**Locations Fixed**: Multiple locations where `HnswGetVector()` is called.

**Pattern**:
```c
nodeVector = HnswGetVector(node);
if (nodeVector == NULL) {
    // Handle error
    continue;
}
```

---

### 12. ✅ Pruning Count Validation
**File**: `hnsw_am.c`

**Fix**: Added bounds checking for `pruneCount` in neighbor pruning logic.

**Location Fixed**: `hnswInsertNode()` - Line 2239

**Implementation**:
```c
int16 pruneCount = neighborNode->neighborCount[currentLevel];
if (pruneCount > m * 2) pruneCount = m * 2;
if (pruneCount < 0) pruneCount = 0;
```

---

## New Helper Functions Added

1. **`hnswValidateNeighborCount()`** - Validates and clamps neighborCount values
2. **`hnswValidateLevel()`** - Validates level values against HNSW_MAX_LEVEL
3. **`hnswValidateBlockNumber()`** - Validates block numbers are within index bounds
4. **`hnswComputeNodeSizeSafe()`** - Overflow-safe node size calculation

---

## Constants Added

- `HNSW_MAX_VISITED_CAPACITY` - Maximum visited array size (1M entries)

---

## Testing Recommendations

1. **Fuzzing**: Test with corrupted index data, extreme values
2. **Stress Testing**: Test with very large dimensions, high levels, maximum efSearch
3. **Memory Testing**: Use valgrind/AddressSanitizer
4. **Bounds Checking**: Enable compiler bounds checking in debug builds
5. **Corruption Testing**: Test with intentionally corrupted neighborCount values

---

## Files Modified

1. `/home/pge/pge/neurondb/NeuronDB/src/index/hnsw_am.c` - Primary fixes
2. `/home/pge/pge/neurondb/NeuronDB/src/scan/hnsw_scan.c` - Scan validation fixes

---

## Verification

- ✅ All linter checks pass
- ✅ No compilation errors
- ✅ All critical issues addressed
- ✅ All high-priority issues addressed
- ✅ Defensive programming patterns applied throughout

---

## Impact

These fixes significantly improve the robustness of the HNSW indexing code by:
- Preventing array bounds violations
- Preventing integer overflows
- Preventing NULL pointer dereferences
- Preventing invalid memory accesses
- Adding graceful error handling for corrupted data
- Limiting memory allocation to prevent OOM conditions

The code is now much more resilient to corrupted index data and edge cases.

