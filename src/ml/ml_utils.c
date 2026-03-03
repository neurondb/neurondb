/*-------------------------------------------------------------------------
 *
 * ml_utils.c
 *    Common utility functions for ML operations.
 *
 * This module provides shared utility functions used across ML algorithms.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_utils.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/memutils.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"
#include "neurondb_spi_safe.h"

/*
 * neurondb_fetch_vectors_from_table
 *    Extract all vectors from a table column using SPI for training operations.
 *
 * This function executes a SELECT query through PostgreSQL's SPI interface
 * to retrieve all vector values from a specified table and column. It
 * creates a dedicated memory context for the SPI session to isolate
 * allocations and prevent memory leaks, then switches back to the caller's
 * context before returning results to ensure data persists after the SPI
 * session ends. The function enforces a maximum limit on the number of
 * vectors retrieved to prevent excessive memory allocation that could
 * exhaust system resources. It validates the first row to determine vector
 * dimensionality and ensures all subsequent rows match this dimension. The
 * returned array of float pointers is allocated in the caller's memory
 * context, so the caller must free it using nfree. This pattern is
 * critical for safe memory management when integrating SPI operations with
 * machine learning algorithms that process large datasets.
 */
float	  **
neurondb_fetch_vectors_from_table(const char *table,
								  const char *col,
								  int *out_count,
								  int *out_dim)
{
	bool		isnull;
	Datum		first_datum;
	Datum		vec_datum;
	float	   **result = NULL;
	int			d;
	int			i;
	int			j_local;
	int			max_vectors_limit;
	int			ret;
	MemoryContext caller_context;
	MemoryContext oldcontext;
	MemoryContext oldcontext_spi;
	NdbSpiSession *spi_session = NULL;
	size_t		result_array_size;
	StringInfoData sql;
	Vector	   *first_vec = NULL;
	Vector	   *vec = NULL;

	caller_context = CurrentMemoryContext;

	max_vectors_limit = 500000;
	initStringInfo(&sql);
	/* Note: No ORDER BY clause - views don't have ctid, and ordering isn't required for training.
	 * Use quote_identifier to prevent SQL injection. */
	appendStringInfo(&sql, "SELECT %s FROM %s LIMIT %d",
					 quote_identifier(col),
					 quote_identifier(table),
					 max_vectors_limit);
	oldcontext_spi = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext_spi);

	ret = ndb_spi_execute(spi_session, sql.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		char	   *query_str = sql.data;
		nfree(sql.data);
		NDB_SPI_SESSION_END(spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: SPI_execute failed: %s", query_str)));
	}

	*out_count = SPI_processed;
	if (*out_count == 0)
	{
		nfree(sql.data);
		NDB_SPI_SESSION_END(spi_session);
		*out_dim = 0;
		return NULL;
	}

	if (*out_count >= 500000)
	{
	}

	if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
		SPI_processed == 0 || SPI_tuptable->vals[0] == NULL || SPI_tuptable->tupdesc == NULL)
	{
		nfree(sql.data);
		NDB_SPI_SESSION_END(spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: NULL vector in first row")));
	}
	first_datum = SPI_getbinval(SPI_tuptable->vals[0],
								SPI_tuptable->tupdesc,
								1,
								&isnull);
	if (isnull)
	{
		nfree(sql.data);
		NDB_SPI_SESSION_END(spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: NULL vector in first row")));
	}

	first_vec = DatumGetVector(first_datum);
	*out_dim = first_vec->dim;

	/*
	 * Switch to caller's context to allocate result that survives
	 * SPI_finish()
	 */
	oldcontext = MemoryContextSwitchTo(caller_context);

	/* Check memory allocation size before palloc */
	result_array_size = sizeof(float *) * (size_t) (*out_count);

	if (result_array_size > MaxAllocSize)
	{
		/*
		 * sql.data is allocated before SPI session, so it's in caller's
		 * context
		 */
		/* We must free it explicitly before SPI session end */
		nfree(sql.data);
		NDB_SPI_SESSION_END(spi_session);
		MemoryContextSwitchTo(oldcontext);
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("neurondb_fetch_vectors_from_table: result array size (%zu bytes) exceeds MaxAllocSize (%zu bytes)",
						result_array_size, (size_t) MaxAllocSize),
				 errhint("Reduce dataset size or use a different algorithm")));
	}

	nalloc(result, float *, *out_count);

	for (i = 0; i < *out_count; i++)
	{
		/* Safe access to SPI_tuptable - validate before access */
		if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
			i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
		{
			/* Free already allocated vectors */
			for (j_local = 0; j_local < i; j_local++)
				nfree(result[j_local]);
			nfree(result);
			nfree(sql.data);
			NDB_SPI_SESSION_END(spi_session);
			MemoryContextSwitchTo(oldcontext);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb_fetch_vectors_from_table: invalid row %d", i)));
		}
		vec_datum = SPI_getbinval(SPI_tuptable->vals[i],
								  SPI_tuptable->tupdesc,
								  1,
								  &isnull);
		if (isnull)
		{
			/* Free already allocated vectors */
			for (j_local = 0; j_local < i; j_local++)
				nfree(result[j_local]);
			nfree(result);

			/*
			 * sql.data is allocated before SPI session, so it's in caller's
			 * context
			 */
			/* We must free it explicitly before SPI session end */
			nfree(sql.data);
			NDB_SPI_SESSION_END(spi_session);
			MemoryContextSwitchTo(oldcontext);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: NULL vector at row %d", i)));
		}

		vec = DatumGetVector(vec_datum);

		/* Verify dimension consistency */
		if (vec->dim != *out_dim)
		{
			int			j_free;

			/* Free already allocated vectors */
			for (j_free = 0; j_free < i; j_free++)
				nfree(result[j_free]);
			nfree(result);

			/*
			 * sql.data is allocated before SPI session, so it's in caller's
			 * context
			 */
			/* We must free it explicitly before SPI session end */
			nfree(sql.data);
			NDB_SPI_SESSION_END(spi_session);
			MemoryContextSwitchTo(oldcontext);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: inconsistent vector dimension at row %d: expected %d, got %d",
							i,
							*out_dim,
							vec->dim)));
		}

		/* Check individual vector allocation size */
		{
			int			j_free2;
			size_t		vector_size = sizeof(float) * (size_t) (*out_dim);

			if (vector_size > MaxAllocSize)
			{
				/* Free already allocated vectors */
				for (j_free2 = 0; j_free2 < i; j_free2++)
					nfree(result[j_free2]);
				nfree(result);

				/*
				 * sql.data is allocated before SPI session, so it's in
				 * caller's context
				 */
				/* We must free it explicitly before SPI session end */
				nfree(sql.data);
				NDB_SPI_SESSION_END(spi_session);
				MemoryContextSwitchTo(oldcontext);
				ereport(ERROR,
						(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
						 errmsg("neurondb_fetch_vectors_from_table: vector size (%zu bytes) exceeds MaxAllocSize (%zu bytes)",
								vector_size, (size_t) MaxAllocSize),
						 errhint("Vector dimension too large")));
			}
		}

		/* Copy vector data */
		{
			float	   *vec_data = NULL;

			nalloc(vec_data, float, *out_dim);
			result[i] = vec_data;
			for (d = 0; d < *out_dim; d++)
				result[i][d] = vec->data[d];
		}
	}

	/* Switch back to SPI context before finishing */
	MemoryContextSwitchTo(oldcontext);
	nfree(sql.data);
	NDB_SPI_SESSION_END(spi_session);

	return result;
}

/*
 * neurondb_free_vectors
 *    Safely free an array of vectors allocated by neurondb_fetch_vectors_from_table
 *
 * This function safely frees all vectors in the array, handling cases where
 * some pointers might be NULL or invalid.
 */
void
neurondb_free_vectors(float **data, int nvec)
{
	if (data == NULL)
		return;

	for (int i = 0; i < nvec; i++)
	{
		/* Only free non-NULL pointers that look valid */
		if (data[i] != NULL)
		{
			/* Basic sanity check: pointer should be reasonably aligned (at least 4-byte aligned for float) */
			/* This helps catch obviously corrupted pointers like 0x300000000 */
			/* Cast to intptr_t (available in postgres.h) to check alignment */
			if (((intptr_t) data[i] & 0x3) == 0)
			{
				nfree(data[i]);
			}
		}
	}
	nfree(data);
}
