/*-------------------------------------------------------------------------
 *
 * ml_utils.c
 *    Common utility functions for ML operations.
 *
 * This module provides shared utility functions used across ML algorithms.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
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
 * fetch_vectors_from_table
 *    Extract all vectors from a table column as float arrays via SPI.
 *    Returns a pointer to an allocated array of float pointers (each is a vector),
 *    sets *out_count to the number returned and *out_dim to inferred vector dimension.
 */
float	  **
neurondb_fetch_vectors_from_table(const char *table,
								  const char *col,
								  int *out_count,
								  int *out_dim)
{
	StringInfoData sql;
	int			ret;
	int			i,
				d,
				j;
	NDB_DECLARE(float **, result);
	MemoryContext oldcontext;
	MemoryContext caller_context;
	MemoryContext oldcontext_spi;
	NDB_DECLARE(NdbSpiSession *, spi_session);

	/* Save the caller's context before SPI operations */
	caller_context = CurrentMemoryContext;

	/* Limit query to prevent memory allocation errors for very large tables */
	/* Use a conservative limit: 500k vectors max to avoid MaxAllocSize errors */
	{
		int			max_vectors_limit = 500000;

		/* sql is allocated in caller's context, not SPI context */
		initStringInfo(&sql);
		appendStringInfo(&sql, "SELECT %s FROM %s LIMIT %d", col, table, max_vectors_limit);
	}
	oldcontext_spi = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext_spi);

	ret = ndb_spi_execute(spi_session, sql.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		/*
		 * sql.data is allocated before SPI session, so it's in caller's
		 * context
		 */
		/* We must free it explicitly before SPI session end */
		char	   *query_str = sql.data;	/* Save pointer before freeing */

		NDB_FREE(sql.data);
		NDB_SPI_SESSION_END(spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: SPI_execute failed: %s", query_str)));
	}

	*out_count = SPI_processed;
	if (*out_count == 0)
	{
		/*
		 * sql.data is allocated before SPI session, so it's in caller's
		 * context
		 */
		/* We must free it explicitly before SPI session end */
		NDB_FREE(sql.data);
		NDB_SPI_SESSION_END(spi_session);
		*out_dim = 0;
		return NULL;
	}

	/* Warn if we hit the limit */
	if (*out_count >= 500000)
	{
		elog(DEBUG1,
			 "neurondb_fetch_vectors_from_table: table has more than %d vectors, "
			 "limiting to %d to avoid memory allocation errors",
			 500000, *out_count);
	}

	/* Get dimension from first vector */
	{
		bool		isnull;
		Datum		first_datum;
		Vector	   *first_vec;

		/* Safe access for complex types - validate before access */
		if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || 
			SPI_processed == 0 || SPI_tuptable->vals[0] == NULL || SPI_tuptable->tupdesc == NULL)
		{
			NDB_FREE(sql.data);
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
			/*
			 * sql.data is allocated before SPI session, so it's in caller's
			 * context
			*/
			/* We must free it explicitly before SPI session end */
			NDB_FREE(sql.data);
			NDB_SPI_SESSION_END(spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: NULL vector in first row")));
		}

		first_vec = DatumGetVector(first_datum);
		*out_dim = first_vec->dim;
	}

	/*
	 * Switch to caller's context to allocate result that survives
	 * SPI_finish()
	 */
	oldcontext = MemoryContextSwitchTo(caller_context);

	/* Check memory allocation size before palloc */
	{
		size_t		result_array_size = sizeof(float *) * (size_t) (*out_count);

		if (result_array_size > MaxAllocSize)
		{
			/*
			 * sql.data is allocated before SPI session, so it's in caller's
			 * context
			 */
			/* We must free it explicitly before SPI session end */
			NDB_FREE(sql.data);
			NDB_SPI_SESSION_END(spi_session);
			MemoryContextSwitchTo(oldcontext);
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("neurondb_fetch_vectors_from_table: result array size (%zu bytes) exceeds MaxAllocSize (%zu bytes)",
							result_array_size, (size_t) MaxAllocSize),
					 errhint("Reduce dataset size or use a different algorithm")));
		}
	}

	NDB_ALLOC(result, float *, *out_count);

	for (i = 0; i < *out_count; i++)
	{
		bool		isnull;
		Datum		vec_datum;
		Vector	   *vec;
		NDB_DECLARE(float *, vec_data);

		/* Safe access to SPI_tuptable - validate before access */
		if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || 
			i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
		{
			/* Free already allocated vectors */
			for (j = 0; j < i; j++)
				NDB_FREE(result[j]);
			NDB_FREE(result);
			NDB_FREE(sql.data);
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
			for (int j = 0; j < i; j++)
				NDB_FREE(result[j]);
			NDB_FREE(result);

			/*
			 * sql.data is allocated before SPI session, so it's in caller's
			 * context
			 */
			/* We must free it explicitly before SPI session end */
			NDB_FREE(sql.data);
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
			/* Free already allocated vectors */
			for (int j = 0; j < i; j++)
				NDB_FREE(result[j]);
			NDB_FREE(result);

			/*
			 * sql.data is allocated before SPI session, so it's in caller's
			 * context
			 */
			/* We must free it explicitly before SPI session end */
			NDB_FREE(sql.data);
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
			size_t		vector_size = sizeof(float) * (size_t) (*out_dim);

			if (vector_size > MaxAllocSize)
			{
				/* Free already allocated vectors */
				for (int j = 0; j < i; j++)
					NDB_FREE(result[j]);
				NDB_FREE(result);

				/*
				 * sql.data is allocated before SPI session, so it's in
				 * caller's context
				 */
				/* We must free it explicitly before SPI session end */
				NDB_FREE(sql.data);
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
		NDB_ALLOC(vec_data, float, *out_dim);
		result[i] = vec_data;
		for (d = 0; d < *out_dim; d++)
			result[i][d] = vec->data[d];
	}

	/* Switch back to SPI context before finishing */
	MemoryContextSwitchTo(oldcontext);
	NDB_FREE(sql.data);
	NDB_SPI_SESSION_END(spi_session);

	return result;
}
