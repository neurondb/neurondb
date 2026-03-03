# Crash-Proof Patterns for NeuronDB

## Overview

This document describes the crash-proof patterns implemented across NeuronDB to prevent the 6 identified crash categories. All code must follow these patterns to ensure system stability.

## Crash Categories Addressed

1. **Memory Context Issues** - Wrong context switches, invalid context usage
2. **Invalid pfree Calls** - Freeing NULL, already freed, or invalid pointers
3. **Wrong Context pfree** - Freeing memory allocated in different memory context
4. **NULL Return Values** - Functions returning NULL (models, allocations) not checked
5. **NULL Input Parameters** - Functions dereferencing NULL parameters without validation
6. **SPI Context Crashes** - SPI_execute failures, context cleanup issues, accessing freed SPI data

## Critical Pattern: Safe pfree with NULL Assignment

### MANDATORY PATTERN

**ALWAYS** use this pattern for ALL memory deallocation:

```c
ndb_safe_pfree(ptr);
ptr = NULL;
```

**OR** use the convenience macro:

```c
NDB_SAFE_PFREE_AND_NULL(ptr);
```

### Why This Pattern is Critical

- `ndb_safe_pfree()` checks for NULL internally (returns safely if NULL)
- Setting to NULL after freeing prevents:
  - **Double-free attempts**: If freed again, it's NULL and safe_pfree returns safely
  - **Use-after-free bugs**: Code can check `if (ptr == NULL)` before use
  - **Accidental reuse**: Prevents code from using stale pointers

### Where to Apply

- All cleanup paths (error handlers, function exits)
- All PG_CATCH blocks
- All memory context cleanup sections
- All resource cleanup functions
- After every single pfree/ndb_safe_pfree call

### Examples

**Error Handler Cleanup:**
```c
PG_CATCH();
{
	EmitErrorReport();
	FlushErrorState();
	
	// Cleanup all pointers with NULL assignment
	ndb_safe_pfree(ptr1);
	ptr1 = NULL;
	ndb_safe_pfree(ptr2);
	ptr2 = NULL;
	
	if (IsTransactionState())
		AbortCurrentTransaction();
}
PG_END_TRY();
```

**Function Cleanup:**
```c
cleanup:
	if (payload != NULL)
	{
		ndb_safe_pfree(payload);
		payload = NULL;
	}
	if (metrics != NULL)
	{
		ndb_safe_pfree(metrics);
		metrics = NULL;
	}
	return success;
```

## Pattern 2: HTTP/Network Operations

### Pattern: PG_TRY/PG_CATCH around all curl operations

All HTTP operations using libcurl MUST be wrapped in PG_TRY/PG_CATCH blocks with proper cleanup:

```c
PG_TRY();
{
	curl = curl_easy_init();
	if (!curl) {
		ereport(ERROR, ...);
	}
	// Setup curl options
	res = curl_easy_perform(curl);
	// Validate response
}
PG_CATCH();
{
	if (curl) {
		curl_easy_cleanup(curl);
		curl = NULL;
	}
	if (headers) {
		curl_slist_free_all(headers);
		headers = NULL;
	}
	// Cleanup any allocated strings with NULL assignment
	ndb_safe_pfree(url_str);
	url_str = NULL;
	ndb_safe_pfree(body_str);
	body_str = NULL;
	FlushErrorState();
	ereport(ERROR, ...);
}
PG_END_TRY();
```

### Files Requiring This Pattern

- `src/llm/hf_http.c` - ✅ Partially implemented
- `src/llm/openai_http.c` - Needs implementation
- `src/ml/model_runtime.c` - Needs implementation

## Pattern 3: SPI Operations

### Pattern: Safe SPI execution with validation

**ALWAYS** use `ndb_spi_execute_safe()` and validate results:

```c
ret = ndb_spi_execute_safe(query, true, 0);
if (ret != SPI_OK_SELECT || SPI_processed == 0) {
	ereport(ERROR, ...);
}
NDB_CHECK_SPI_TUPTABLE();
// Copy data before SPI_finish()
result = SPI_getbinval(...);
copied_result = DatumCopy(result, ...);
SPI_finish();
// Use copied_result after SPI_finish()
```

### Critical Rules

1. **NEVER** access `SPI_tuptable` after `SPI_finish()`
2. **ALWAYS** copy data out of SPI context before `SPI_finish()`
3. **ALWAYS** validate `SPI_tuptable` before access using `NDB_CHECK_SPI_TUPTABLE()`
4. **ALWAYS** validate return codes (`SPI_OK_SELECT`, `SPI_OK_INSERT`, etc.)

### Helper Functions Available

- `ndb_spi_execute_safe()` - Safe SPI execution with error handling
- `ndb_spi_get_result_safe()` - Safe result extraction with NULL checks
- `ndb_spi_get_jsonb_safe()` - Safe JSONB extraction with copying
- `ndb_spi_get_text_safe()` - Safe text extraction with copying

## Pattern 4: Input Parameter Validation

### Pattern: NULL checks for all function parameters

**ALWAYS** validate input parameters before use:

```c
// For PG function arguments
if (PG_ARGISNULL(0))
	ereport(ERROR, (errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
		errmsg("neurondb: features parameter cannot be NULL")));
features = PG_GETARG_FLOAT4ARRAY(0);
if (features == NULL || ARR_SIZE(features) == 0)
	ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
		errmsg("features array is empty")));
```

### Validation Macros Available

- `NDB_CHECK_NULL(param, name)` - Standardized NULL parameter checks
- `NDB_CHECK_NULL_ARG(argnum, name)` - Check PG function argument is NULL
- `NDB_CHECK_ALLOC(ptr, name)` - Allocation validation
- `NDB_CHECK_ARRAY_BOUNDS(idx, size, name)` - Array bounds checking
- `NDB_CHECK_VECTOR_VALID(vec)` - Vector structure validation

## Pattern 5: Model Return Validation

### Pattern: Validate model structures before use

**ALWAYS** check model returns for NULL:

```c
ModelHandle *model = load_model(id);
if (model == NULL || model->data == NULL) {
	ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR),
		errmsg("Model %d not found or invalid", id)));
}
NDB_VALIDATE_MODEL(model);
use_model(model->data);
```

### Validation Macros Available

- `NDB_VALIDATE_MODEL(model)` - Validate model structure is not NULL
- `NDB_VALIDATE_MODEL_DATA(model)` - Validate model and model->data

## Pattern 6: Memory Context Management

### Pattern: Comprehensive PG_TRY/PG_CATCH with context cleanup

Reference implementation: `src/worker/worker_llm.c` (lines 67-525)

```c
PG_TRY();
{
	MemoryContext oldcxt = MemoryContextSwitchTo(worker_ctx);
	// Operations
	MemoryContextSwitchTo(oldcxt);
	MemoryContextReset(worker_ctx);
}
PG_CATCH();
{
	EmitErrorReport();
	FlushErrorState();
	if (IsTransactionState())
		AbortCurrentTransaction();
	// Cleanup all pointers with NULL assignment
	ndb_safe_pfree(ptr1);
	ptr1 = NULL;
	ndb_safe_pfree(ptr2);
	ptr2 = NULL;
	MemoryContextSwitchTo(TopMemoryContext);
	MemoryContextReset(worker_ctx);
}
PG_END_TRY();
```

### Helper Functions Available

- `ndb_memory_context_validate(context)` - Validates context before operations
- `ndb_ensure_memory_context(context)` - Ensures we're in specified context
- `ndb_safe_context_cleanup(context, oldcontext)` - Safely clean up memory context

## Pattern 7: GPU Error Handling

### Pattern: PG_TRY wrapper around GPU calls with CPU fallback

Reference: `src/ml/ml_random_forest.c` GPU fallback pattern

```c
PG_TRY();
{
	result = gpu_operation(...);
}
PG_CATCH();
{
	FlushErrorState();
	elog(WARNING, "GPU operation failed, falling back to CPU");
	// Cleanup GPU resources with NULL assignment
	ndb_safe_pfree(gpu_buffer);
	gpu_buffer = NULL;
	result = cpu_fallback_operation(...);
}
PG_END_TRY();
```

## Common Anti-Patterns to Avoid

### ❌ WRONG: Direct pfree without check
```c
pfree(ptr);
```

### ❌ WRONG: Safe pfree but missing NULL assignment
```c
ndb_safe_pfree(ptr);
```

### ✅ CORRECT: Safe pfree with NULL assignment
```c
ndb_safe_pfree(ptr);
ptr = NULL;
```

### ❌ WRONG: Accessing SPI data after SPI_finish()
```c
SPI_execute(query, true, 0);
result = SPI_tuptable->vals[0];
SPI_finish();
use_result(result);  // CRASH! result is invalid
```

### ✅ CORRECT: Copy data before SPI_finish()
```c
ret = ndb_spi_execute_safe(query, true, 0);
NDB_CHECK_SPI_TUPTABLE();
result = SPI_getbinval(...);
copied_result = DatumCopy(result, ...);
SPI_finish();
use_result(copied_result);  // Safe - data is copied
```

### ❌ WRONG: No NULL check on input
```c
features = PG_GETARG_FLOAT4ARRAY(0);
process_features(features);  // CRASH if NULL!
```

### ✅ CORRECT: Validate input first
```c
if (PG_ARGISNULL(0))
	ereport(ERROR, ...);
features = PG_GETARG_FLOAT4ARRAY(0);
if (features == NULL || ARR_SIZE(features) == 0)
	ereport(ERROR, ...);
process_features(features);
```

## Implementation Status

### Phase 1: Infrastructure ✅ COMPLETE
- ✅ `ndb_safe_pfree()` exists and checks for NULL
- ✅ `ndb_safe_pfree_multi()` exists for batch cleanup
- ✅ `ndb_memory_context_validate()` exists
- ✅ `NDB_SAFE_PFREE_AND_NULL` macro created
- ✅ Validation macros created/enhanced
- ✅ SPI safe wrapper functions exist

### Phase 2: Code Audits & Fixes IN PROGRESS
- pfree() calls audit (4301 instances across 118 files)
  - ✅ worker_llm.c - Complete
  - ml_ridge_lasso.c - Partial
  - ml_random_forest.c - Partial
  - hf_http.c - Partial
  - Remaining files - Pending
- SPI operations audit
  - ✅ ml_unified_api.c - Good patterns already
  - Other files - Pending
- Model return validation - Pending
- Input parameter validation - Pending
- Memory context management - Pending

### Phase 3: Pattern Enforcement IN PROGRESS
- HTTP operations - hf_http.c partially done
- GPU error handling - Pending
- Model evaluation functions - Pending

### Phase 4: Testing & Validation PENDING
- Crash test suite - Pending
- Defensive assertions - Pending

## Files Requiring Immediate Attention

### Critical Priority
1. `src/ml/ml_unified_api.c` - ✅ Good patterns, minor fixes needed
2. `src/ml/ml_inference.c` - Needs review
3. `src/gpu/common/gpu_model_bridge.c` - Needs review
4. All `evaluate_*_by_model_id()` functions - Needs review
5. `src/llm/hf_http.c` - Partially done
6. `src/llm/openai_http.c` - Needs implementation

### High Priority
1. `src/ml/ml_ridge_lasso.c` (446 pfree calls) - Partial
2. `src/ml/ml_linear_regression.c` (168 pfree calls) - Pending
3. `src/ml/ml_knn.c` (254 pfree calls) - Pending
4. `src/ml/ml_random_forest.c` (268 pfree calls) - Partial
5. `src/gpu/metal/gpu_backend_metal.c` (134 pfree calls) - Pending

## Success Criteria

- ✅ Zero crashes from the 6 identified categories
- ✅ 100% NULL checks on all input parameters
- 100% of pfree calls replaced with `ndb_safe_pfree(ptr); ptr = NULL;` pattern
- 100% error handling on all SPI operations
- Consistent patterns across all similar functions
- Memory context tracking for all critical paths
- Comprehensive test coverage for crash scenarios
- ✅ All freed pointers set to NULL immediately after freeing

## References

- `src/worker/worker_llm.c` - Reference implementation for worker pattern
- `src/ml/ml_random_forest.c` - Reference for GPU fallback pattern
- `src/ml/ml_unified_api.c` - Reference for SPI and error handling
- `include/neurondb_validation.h` - All validation macros
- `src/util/neurondb_safe_memory.c` - Safe memory utilities
- `src/util/neurondb_spi_safe.c` - Safe SPI utilities

