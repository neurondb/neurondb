/*-------------------------------------------------------------------------
 *
 * neurondb_validation.h
 *    Validation macros and utilities for NeuronDB crash prevention
 *
 * Provides standardized validation macros for common crash patterns:
 * - NULL parameter checks
 * - Allocation validation
 * - SPI return code validation
 * - Model structure validation
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/neurondb_validation.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_VALIDATION_H
#define NEURONDB_VALIDATION_H

#include "postgres.h"
#include "fmgr.h"
#include "executor/spi.h"
#include "utils/elog.h"
#include "utils/memutils.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_constants.h"

/*-------------------------------------------------------------------------
 * NULL Parameter Validation Macros
 *-------------------------------------------------------------------------
 */

/*
 * NDB_CHECK_NULL - Check if parameter is NULL and error if so
 *
 * Usage:
 *   NDB_CHECK_NULL(features, "features");
 */
#define NDB_CHECK_NULL(param, name) \
	do { \
		if ((param) == NULL) \
			ereport(ERROR, \
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED), \
				 errmsg(NDB_ERR_MSG("%s cannot be NULL"), (name)))); \
	} while (0)

/*
 * NDB_CHECK_NULL_ARG - Check if PG function argument is NULL
 *
 * Usage:
 *   NDB_CHECK_NULL_ARG(0, "model_id");
 */
#define NDB_CHECK_NULL_ARG(argnum, name) \
	do { \
		if (PG_ARGISNULL(argnum)) \
			ereport(ERROR, \
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED), \
				 errmsg(NDB_ERR_MSG("%s (argument %d) cannot be NULL"), \
					(name), (argnum)))); \
	} while (0)

/*
 * NDB_CHECK_NULL_OR_ERROR - Check if parameter is NULL and return error code
 *
 * Returns -1 if NULL, does not raise ERROR
 * Usage:
 *   if (NDB_CHECK_NULL_OR_ERROR(features, "features", errstr) < 0)
 *       return -1;
 */
#define NDB_CHECK_NULL_OR_ERROR(param, name, errstr) \
	((param) == NULL ? \
		((errstr) ? (*(errstr) = psprintf(NDB_ERR_MSG("%s cannot be NULL"), (name)), -1) : -1) : \
		0)

/*-------------------------------------------------------------------------
 * Allocation Validation Macros
 *-------------------------------------------------------------------------
 */

/*
 * NDB_CHECK_ALLOC - Check if allocation succeeded
 *
 * Usage:
 *   ptr = palloc(size);
 *   NDB_CHECK_ALLOC(ptr, "feature_matrix");
 */
#define NDB_CHECK_ALLOC(ptr, name) \
	do { \
		if ((ptr) == NULL) \
			ereport(ERROR, \
				(errcode(ERRCODE_OUT_OF_MEMORY), \
				 errmsg(NDB_ERR_MSG("failed to allocate memory for %s"), \
					(name)))); \
	} while (0)

/*
 * NDB_CHECK_ALLOC_OR_ERROR - Check if allocation succeeded, return error code
 *
 * Returns -1 if NULL, does not raise ERROR
 * Usage:
 *   ptr = palloc(size);
 *   if (NDB_CHECK_ALLOC_OR_ERROR(ptr, "features", errstr) < 0)
 *       return -1;
 */
#define NDB_CHECK_ALLOC_OR_ERROR(ptr, name, errstr) \
	((ptr) == NULL ? \
		((errstr) ? (*(errstr) = psprintf(NDB_ERR_MSG("allocation failed for %s"), (name)), -1) : -1) : \
		0)

/*
 * NDB_CHECK_ALLOC_SIZE - Validate allocation size before allocating
 *
 * Usage:
 *   NDB_CHECK_ALLOC_SIZE(alloc_size, "feature_matrix");
 */
#define NDB_CHECK_ALLOC_SIZE(size, name) \
	do { \
		if ((size) > MaxAllocSize) \
			ereport(ERROR, \
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED), \
				 errmsg(NDB_ERR_MSG("allocation size for %s (%zu bytes) exceeds MaxAllocSize (%zu bytes)"), \
					(name), (size_t)(size), (size_t)MaxAllocSize), \
				 errhint("Reduce dataset size or use batch processing"))); \
		if ((size) == 0) \
			ereport(ERROR, \
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE), \
				 errmsg(NDB_ERR_MSG("allocation size for %s cannot be zero"), \
					(name)))); \
	} while (0)

/*-------------------------------------------------------------------------
 * SPI Return Code Validation Macros
 *-------------------------------------------------------------------------
 */

/*
 * NDB_CHECK_SPI_RET - Validate SPI_execute return code
 *
 * Usage:
 *   ret = SPI_execute(query, true, 0);
 *   NDB_CHECK_SPI_RET(ret, SPI_OK_SELECT, query);
 */
#define NDB_CHECK_SPI_RET(ret, expected, query) \
	do { \
		if ((ret) != (expected)) { \
			const char *err_msg; \
			switch (ret) { \
				case SPI_ERROR_UNCONNECTED: \
					err_msg = "SPI not connected"; \
					break; \
				case SPI_ERROR_COPY: \
					err_msg = "COPY command in progress"; \
					break; \
				case SPI_ERROR_TRANSACTION: \
					err_msg = "transaction state error"; \
					break; \
				case SPI_ERROR_ARGUMENT: \
					err_msg = "invalid argument"; \
					break; \
				case SPI_ERROR_OPUNKNOWN: \
					err_msg = "unknown operation"; \
					break; \
				default: \
					err_msg = "unknown SPI error"; \
					break; \
			} \
			ereport(ERROR, \
				(errcode(ERRCODE_INTERNAL_ERROR), \
				 errmsg(NDB_ERR_MSG("SPI operation failed: %s (code %d)"), \
					err_msg, (ret)), \
				 errhint("Query: %s", (query) ? (query) : "unknown"))); \
		} \
	} while (0)

/*
 * NDB_CHECK_SPI_PROCESSED - Validate SPI_processed count
 *
 * Usage:
 *   NDB_CHECK_SPI_PROCESSED(min_rows, "evaluation query");
 */
#define NDB_CHECK_SPI_PROCESSED(min_rows, query_name) \
	do { \
		if (SPI_processed < (min_rows)) \
			ereport(ERROR, \
				(errcode(ERRCODE_DATA_EXCEPTION), \
				 errmsg(NDB_ERR_MSG("%s returned %ld rows, expected at least %ld"), \
					(query_name), (long)SPI_processed, (long)(min_rows)))); \
	} while (0)

/*
 * NDB_CHECK_SPI_OK - Validate SPI return code AND SPI_processed count
 *
 * Checks ret == expected, and optionally validates SPI_processed count.
 * For generic wrappers, can validate set of allowed codes.
 *
 * Usage:
 *   NDB_CHECK_SPI_OK(ret, SPI_OK_SELECT, 1);  // Expect exactly 1 row
 *   NDB_CHECK_SPI_OK(ret, SPI_OK_SELECT, -1); // Don't check row count
 */
#define NDB_CHECK_SPI_OK(ret, expected, min_rows) \
	do { \
		if ((ret) != (expected)) { \
			const char *err_msg; \
			switch (ret) { \
				case SPI_ERROR_UNCONNECTED: \
					err_msg = "SPI not connected"; \
					break; \
				case SPI_ERROR_COPY: \
					err_msg = "COPY command in progress"; \
					break; \
				case SPI_ERROR_TRANSACTION: \
					err_msg = "transaction state error"; \
					break; \
				case SPI_ERROR_ARGUMENT: \
					err_msg = "invalid argument"; \
					break; \
				case SPI_ERROR_OPUNKNOWN: \
					err_msg = "unknown operation"; \
					break; \
				default: \
					err_msg = "unknown SPI error"; \
					break; \
			} \
			ereport(ERROR, \
				(errcode(ERRCODE_INTERNAL_ERROR), \
				 errmsg(NDB_ERR_MSG("SPI operation failed: %s (got %d, expected %d)"), \
					err_msg, (ret), (expected)))); \
		} \
		if ((min_rows) >= 0 && SPI_processed < (min_rows)) { \
			ereport(ERROR, \
				(errcode(ERRCODE_DATA_EXCEPTION), \
				 errmsg(NDB_ERR_MSG("SPI query returned %ld rows, expected at least %ld"), \
					(long)SPI_processed, (long)(min_rows)))); \
		} \
	} while (0)

/*
 * NDB_CHECK_SPI_TUPTABLE - Validate SPI_tuptable is valid
 *
 * Only checks SPI_tuptable for queries that return result sets (SELECT, etc.)
 * For non-SELECT queries (INSERT, UPDATE, DELETE without RETURNING),
 * SPI_tuptable is NULL and that's expected.
 *
 * Usage:
 *   NDB_CHECK_SPI_TUPTABLE();  // Checks unconditionally (use with caution)
 *   NDB_CHECK_SPI_TUPTABLE_IF_SELECT(ret);  // Only checks if ret indicates result set
 */
#define NDB_CHECK_SPI_TUPTABLE() \
	do { \
		if (SPI_tuptable == NULL || SPI_tuptable->tupdesc == NULL) \
			ereport(ERROR, \
				(errcode(ERRCODE_INTERNAL_ERROR), \
				 errmsg(NDB_ERR_MSG("SPI_tuptable is NULL or invalid")))); \
	} while (0)

/*
 * NDB_CHECK_SPI_TUPTABLE_IF_SELECT - Conditionally validate SPI_tuptable
 *
 * Only checks SPI_tuptable if the return code indicates a query that
 * returns a result set. Safe to use after any SPI_execute call.
 *
 * Usage:
 *   ret = ndb_spi_execute_safe(sql.data, false, 0);
 *   NDB_CHECK_SPI_TUPTABLE_IF_SELECT(ret);
 */
#define NDB_CHECK_SPI_TUPTABLE_IF_SELECT(ret_code) \
	do { \
		if ((ret_code) == SPI_OK_SELECT || \
			(ret_code) == SPI_OK_SELINTO || \
			(ret_code) == SPI_OK_INSERT_RETURNING || \
			(ret_code) == SPI_OK_UPDATE_RETURNING || \
			(ret_code) == SPI_OK_DELETE_RETURNING) \
		{ \
			if (SPI_tuptable == NULL || SPI_tuptable->tupdesc == NULL) \
				ereport(ERROR, \
					(errcode(ERRCODE_INTERNAL_ERROR), \
					 errmsg(NDB_ERR_MSG("SPI_tuptable is NULL or invalid for result-set query"))); \
		} \
	} while (0)

/*-------------------------------------------------------------------------
 * Model Validation Macros
 *-------------------------------------------------------------------------
 */

/*
 * NDB_VALIDATE_MODEL - Validate model structure is not NULL
 *
 * Usage:
 *   NDB_VALIDATE_MODEL(model);
 */
#define NDB_VALIDATE_MODEL(model) \
	do { \
		if ((model) == NULL) \
			ereport(ERROR, \
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE), \
				 errmsg(NDB_ERR_MSG("model is NULL"))); \
	} while (0)

/*
 * NDB_VALIDATE_MODEL_DATA - Validate model data is not NULL
 *
 * Usage:
 *   NDB_VALIDATE_MODEL_DATA(model);
 */
#define NDB_VALIDATE_MODEL_DATA(model) \
	do { \
		NDB_VALIDATE_MODEL(model); \
		if ((model)->model_data == NULL) \
			ereport(ERROR, \
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE), \
				 errmsg(NDB_ERR_MSG("model data is NULL"))); \
	} while (0)

/*-------------------------------------------------------------------------
 * Array/Vector Validation Macros
 *-------------------------------------------------------------------------
 */

/*
 * NDB_CHECK_ARRAY_BOUNDS - Validate array index is within bounds
 *
 * Usage:
 *   NDB_CHECK_ARRAY_BOUNDS(idx, size, "features");
 */
#define NDB_CHECK_ARRAY_BOUNDS(idx, size, name) \
	do { \
		if ((idx) < 0 || (idx) >= (size)) \
			ereport(ERROR, \
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR), \
				 errmsg(NDB_ERR_MSG("array index %d out of bounds for %s (size %d)"), \
					(idx), (name), (size)))); \
	} while (0)

/*
 * NDB_CHECK_VECTOR_VALID - Validate vector structure
 *
 * Usage:
 *   NDB_CHECK_VECTOR_VALID(vec);
 */
#define NDB_CHECK_VECTOR_VALID(vec) \
	do { \
		if ((vec) == NULL) \
			ereport(ERROR, \
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED), \
				 errmsg(NDB_ERR_MSG("vector is NULL")))); \
		if ((vec)->dim <= 0 || (vec)->dim > 32767) \
			ereport(ERROR, \
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE), \
				 errmsg(NDB_ERR_MSG("invalid vector dimension %d"), \
					(vec)->dim))); \
	} while (0)

/*-------------------------------------------------------------------------
 * Integer Overflow Checks
 *-------------------------------------------------------------------------
 */

/*
 * NDB_CHECK_SIZE_OVERFLOW - Check for integer overflow in size calculation
 *
 * Usage:
 *   NDB_CHECK_SIZE_OVERFLOW(n_samples, feature_dim, sizeof(float));
 */
#define NDB_CHECK_SIZE_OVERFLOW(count1, count2, element_size) \
	do { \
		if ((size_t)(count1) > SIZE_MAX / (size_t)(count2) || \
			(size_t)(count1) * (size_t)(count2) > SIZE_MAX / (size_t)(element_size)) \
			ereport(ERROR, \
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED), \
				 errmsg(NDB_ERR_MSG("allocation size would overflow: %d * %d * %zu"), \
					(count1), (count2), (size_t)(element_size)))); \
	} while (0)

/*-------------------------------------------------------------------------
 * Safe Memory Free with NULL Assignment
 *-------------------------------------------------------------------------
 */

/*
 * NDB_SAFE_PFREE_AND_NULL - Safely free pointer and set to NULL
 *
 * CRITICAL PATTERN: Always use this for all pfree operations to prevent
 * double-free and use-after-free bugs.
 *
 * Usage:
 *   NDB_SAFE_PFREE_AND_NULL(ptr);
 *
 * This macro is an alias for NDB_FREE from neurondb_macros.h.
 * It checks for NULL internally and sets ptr = NULL to prevent
 * double-free and use-after-free.
 *
 * Must be applied to ALL pfree operations per crash-proof plan.
 */
#define NDB_SAFE_PFREE_AND_NULL(ptr) NDB_FREE(ptr)

/*-------------------------------------------------------------------------
 * SPI Context Safety Macros
 *-------------------------------------------------------------------------
 */

/*
 * NDB_SPI_COPY_BEFORE_FINISH - Reminder macro for SPI data copying pattern
 *
 * This is a documentation/reminder macro. Always copy SPI data to caller's
 * context before calling SPI_finish().
 *
 * Pattern:
 *   Datum temp = SPI_getbinval(...);  // Still in SPI context
 *   MemoryContextSwitchTo(caller_context);
 *   Datum copied = DatumCopy(temp, ...);  // Copy to caller context
 *   SPI_finish();  // Now safe - data is copied
 *
 * See ndb_spi_get_jsonb_safe() and ndb_spi_get_text_safe() for examples.
 */
#define NDB_SPI_COPY_BEFORE_FINISH \
	/* CRITICAL: Copy all data from SPI context before SPI_finish() */ \
								/* SPI_finish() deletes SPI memory context -
								 * any pointers become invalid */

#endif							/* NEURONDB_VALIDATION_H */
