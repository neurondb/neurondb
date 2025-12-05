/*-------------------------------------------------------------------------
 *
 * analytics.c
 *    Vector analytics and machine learning analysis.
 *
 * This module implements comprehensive vector analytics including clustering,
 * dimensionality reduction, outlier detection, and quality metrics.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/analytics.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"
#include "utils/lsyscache.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_simd.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"

/*
 * feedback_loop_integrate
 *    Feedback loop integration: records feedback in a dedicated table,
 *    updating aggregations. Table: neurondb_feedback (query TEXT, result TEXT,
 *    rating REAL, ts TIMESTAMPTZ DEFAULT now()). If the table does not exist, creates it.
 */
PG_FUNCTION_INFO_V1(feedback_loop_integrate);

Datum
feedback_loop_integrate(PG_FUNCTION_ARGS)
{
	text	   *query = PG_GETARG_TEXT_PP(0);
	text	   *result = PG_GETARG_TEXT_PP(1);
	float4		user_rating = PG_GETARG_FLOAT4(2);
	char	   *query_str;
	char	   *result_str;
	StringInfoData sql;
	const char *tbl_def;
	int			ret;

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext;

	query_str = text_to_cstring(query);
	result_str = text_to_cstring(result);

	oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	tbl_def = "CREATE TABLE IF NOT EXISTS neurondb_feedback ("
		"id SERIAL PRIMARY KEY, "
		"query TEXT NOT NULL, "
		"result TEXT NOT NULL, "
		"rating REAL NOT NULL, "
		"ts TIMESTAMPTZ NOT NULL DEFAULT now()"
		")";
	ret = ndb_spi_execute(spi_session, tbl_def, false, 0);
	if (ret != SPI_OK_UTILITY)
	{
		NDB_SPI_SESSION_END(spi_session);
		NDB_FREE(query_str);
		NDB_FREE(result_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to create neurondb_feedback table")));
	}

	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql,
					 "INSERT INTO neurondb_feedback (query, result, rating) VALUES "
					 "($$%s$$, $$%s$$, %g)",
					 query_str,
					 result_str,
					 user_rating);
	ret = ndb_spi_execute(spi_session, sql.data, false, 0);
	if (ret != SPI_OK_INSERT)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		NDB_FREE(query_str);
		NDB_FREE(result_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to insert feedback row")));
	}

	ndb_spi_stringinfo_free(spi_session, &sql);
	NDB_SPI_SESSION_END(spi_session);
	NDB_FREE(query_str);
	NDB_FREE(result_str);

	PG_RETURN_BOOL(true);
}

/* DBSCAN moved to ml_dbscan.c */

/*
 * =============================================================================
 * PCA - Principal Component Analysis
 * =============================================================================
 * Dimensionality reduction via singular value decomposition (SVD)
 * - n_components: Target dimension (must be <= original dimension)
 * - Returns projected vectors in lower dimensional space
 */

static void
pca_power_iteration(float **data,
					int nvec,
					int dim,
					float *eigvec,
					int max_iter)
{
	NDB_DECLARE(float *, y);
	int			iter,
				i,
				j;
	double		norm;

	NDB_ALLOC(y, float, dim);

	for (i = 0; i < dim; i++)
		eigvec[i] = (float) (rand() % 1000) / 1000.0f;

	norm = 0.0;
	for (i = 0; i < dim; i++)
		norm += eigvec[i] * eigvec[i];
	norm = sqrt(norm);
	for (i = 0; i < dim; i++)
		eigvec[i] /= norm;

	/* Power iteration - SIMD optimized */
	for (iter = 0; iter < max_iter; iter++)
	{
		/* y = X^T * X * eigvec */
		memset(y, 0, sizeof(float) * dim);

		for (j = 0; j < nvec; j++)
		{
			/* Use SIMD-optimized dot product */
			double		dot = neurondb_dot_product(data[j], eigvec, dim);

			for (i = 0; i < dim; i++)
				y[i] += data[j][i] * dot;
		}

		/* Normalize y */
		norm = 0.0;
		for (i = 0; i < dim; i++)
			norm += y[i] * y[i];
		norm = sqrt(norm);

		if (norm < 1e-10)
			break;

		for (i = 0; i < dim; i++)
			eigvec[i] = y[i] / norm;
	}

	NDB_FREE(y);
}

/* Deflate matrix by removing component of eigenvector */
static void
pca_deflate(float **data, int nvec, int dim, const float *eigvec)
{
	int			i,
				j;

	for (j = 0; j < nvec; j++)
	{
		double		dot = 0.0;

		for (i = 0; i < dim; i++)
			dot += data[j][i] * eigvec[i];

		for (i = 0; i < dim; i++)
			data[j][i] -= dot * eigvec[i];
	}
}

PG_FUNCTION_INFO_V1(reduce_pca);

Datum
reduce_pca(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *column_name;
	int			n_components;
	char	   *tbl_str;
	char	   *col_str;
	float	  **data;
	float	  **components = NULL;
	float	  **projected = NULL;
	int			nvec,
				dim;
	int			i,
				j,
				c;
	ArrayType  *result_array;

	NDB_DECLARE(Datum *, result_datums);
	NDB_DECLARE(float *, mean);

	table_name = PG_GETARG_TEXT_PP(0);
	column_name = PG_GETARG_TEXT_PP(1);
	n_components = PG_GETARG_INT32(2);

	if (n_components < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("n_components must be at least 1")));

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(column_name);

	elog(DEBUG1,
		 "neurondb: PCA dimensionality reduction on %s.%s "
		 "(n_components=%d)",
		 tbl_str,
		 col_str,
		 n_components);

	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);
	if (data == NULL || nvec == 0)
	{
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found")));
	}

	if (dim <= 0)
	{
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		/* Free data array and rows if data is not NULL */
		if (data != NULL)
		{
			for (j = 0; j < nvec; j++)
			{
				if (data[j] != NULL)
					NDB_FREE(data[j]);
			}
			NDB_FREE(data);
		}
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Invalid vector dimension: %d", dim)));
	}

	if (n_components > dim)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("n_components (%d) cannot exceed "
						"dimension (%d)",
						n_components,
						dim)));

	NDB_ALLOC(mean, float, dim);
	for (j = 0; j < nvec; j++)
		for (i = 0; i < dim; i++)
			mean[i] += data[j][i];
	for (i = 0; i < dim; i++)
		mean[i] /= nvec;
	for (j = 0; j < nvec; j++)
		for (i = 0; i < dim; i++)
			data[j][i] -= mean[i];

	NDB_ALLOC(components, float *, n_components);
	for (c = 0; c < n_components; c++)
	{
		NDB_DECLARE(float *, component_row);
		NDB_ALLOC(component_row, float, dim);
		components[c] = component_row;
		pca_power_iteration(data, nvec, dim, components[c], 100);
		pca_deflate(data, nvec, dim, components[c]);
	}

	for (j = 0; j < nvec; j++)
		for (i = 0; i < dim; i++)
			data[j][i] += mean[i];
	for (j = 0; j < nvec; j++)
		for (i = 0; i < dim; i++)
			data[j][i] -= mean[i];

	NDB_ALLOC(projected, float *, nvec);
	for (j = 0; j < nvec; j++)
	{
		NDB_DECLARE(float *, projected_row);
		NDB_ALLOC(projected_row, float, n_components);
		projected[j] = projected_row;
		for (c = 0; c < n_components; c++)
		{
			double		dot = 0.0;

			for (i = 0; i < dim; i++)
				dot += data[j][i] * components[c][i];
			projected[j][c] = dot;
		}
	}

	/*
	 * Validate inputs before array construction - nvec should already be
	 * validated above
	 */
	/* n_components is validated at function entry, but double-check */
	if (n_components <= 0)
	{
		for (j = 0; j < nvec; j++)
		{
			if (data[j] != NULL)
				NDB_FREE(data[j]);
			if (projected[j] != NULL)
				NDB_FREE(projected[j]);
		}
		for (c = 0; c < n_components; c++)
		{
			if (components[c] != NULL)
				NDB_FREE(components[c]);
		}
		NDB_FREE(data);
		NDB_FREE(projected);
		NDB_FREE(components);
		NDB_FREE(mean);
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("reduce_pca: invalid n_components: %d", n_components)));
	}

	NDB_ALLOC(result_datums, Datum, nvec);
	if (result_datums == NULL)
	{
		for (j = 0; j < nvec; j++)
		{
			if (data[j] != NULL)
				NDB_FREE(data[j]);
			if (projected[j] != NULL)
				NDB_FREE(projected[j]);
		}
		for (c = 0; c < n_components; c++)
		{
			if (components[c] != NULL)
				NDB_FREE(components[c]);
		}
		NDB_FREE(data);
		NDB_FREE(projected);
		NDB_FREE(components);
		NDB_FREE(mean);
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("reduce_pca: failed to allocate result_datums")));
	}

	for (j = 0; j < nvec; j++)
	{
		ArrayType  *vec_array;

		NDB_DECLARE(Datum *, vec_datums);
		int16		typlen;
		bool		typbyval;
		char		typalign;

		/* Validate projected data */
		if (projected[j] == NULL)
		{
			/* Clean up and error */
			for (i = 0; i < j; i++)
			{
				if (result_datums[i] != 0)
				{
					ArrayType  *arr = DatumGetArrayTypeP(result_datums[i]);

					if (arr != NULL)
						NDB_FREE(arr);
				}
			}
			NDB_FREE(result_datums);
			for (i = 0; i < nvec; i++)
			{
				if (data[i] != NULL)
					NDB_FREE(data[i]);
				if (projected[i] != NULL)
					NDB_FREE(projected[i]);
			}
			for (c = 0; c < n_components; c++)
			{
				if (components[c] != NULL)
					NDB_FREE(components[c]);
			}
			NDB_FREE(data);
			NDB_FREE(projected);
			NDB_FREE(components);
			NDB_FREE(mean);
			NDB_FREE(tbl_str);
			NDB_FREE(col_str);
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("reduce_pca: projected[%d] is NULL", j)));
		}

		NDB_ALLOC(vec_datums, Datum, n_components);
		if (vec_datums == NULL)
		{
			/* Clean up and error */
			for (i = 0; i < j; i++)
			{
				if (result_datums[i] != 0)
				{
					ArrayType  *arr = DatumGetArrayTypeP(result_datums[i]);

					if (arr != NULL)
						NDB_FREE(arr);
				}
			}
			NDB_FREE(result_datums);
			for (i = 0; i < nvec; i++)
			{
				if (data[i] != NULL)
					NDB_FREE(data[i]);
				if (projected[i] != NULL)
					NDB_FREE(projected[i]);
			}
			for (c = 0; c < n_components; c++)
			{
				if (components[c] != NULL)
					NDB_FREE(components[c]);
			}
			NDB_FREE(data);
			NDB_FREE(projected);
			NDB_FREE(components);
			NDB_FREE(mean);
			NDB_FREE(tbl_str);
			NDB_FREE(col_str);
			ereport(ERROR,
					(errcode(ERRCODE_OUT_OF_MEMORY),
					 errmsg("reduce_pca: failed to allocate vec_datums for row %d", j)));
		}

		for (c = 0; c < n_components; c++)
		{
			/* Validate projected value is finite */
			if (!isfinite(projected[j][c]))
			{
				NDB_FREE(vec_datums);
				/* Clean up and error */
				for (i = 0; i < j; i++)
				{
					if (result_datums[i] != 0)
					{
						ArrayType  *arr = DatumGetArrayTypeP(result_datums[i]);

						NDB_FREE(arr);
					}
				}
				NDB_FREE(result_datums);
				for (i = 0; i < nvec; i++)
				{
					if (data[i] != NULL)
						NDB_FREE(data[i]);
					if (projected[i] != NULL)
						NDB_FREE(projected[i]);
				}
				for (c = 0; c < n_components; c++)
				{
					if (components[c] != NULL)
						NDB_FREE(components[c]);
				}
				NDB_FREE(data);
				NDB_FREE(projected);
				NDB_FREE(components);
				NDB_FREE(mean);
				NDB_FREE(tbl_str);
				NDB_FREE(col_str);
				ereport(ERROR,
						(errcode(ERRCODE_DATA_EXCEPTION),
						 errmsg("reduce_pca: non-finite value in projected[%d][%d]", j, c)));
			}
			vec_datums[c] = Float4GetDatum(projected[j][c]);
		}

		get_typlenbyvalalign(FLOAT4OID, &typlen, &typbyval, &typalign);

		/* Validate type information */
		if (typlen <= 0)
		{
			NDB_FREE(vec_datums);
			/* Clean up and error */
			for (i = 0; i < j; i++)
			{
				if (result_datums[i] != 0)
				{
					ArrayType  *arr = DatumGetArrayTypeP(result_datums[i]);

					if (arr != NULL)
						NDB_FREE(arr);
				}
			}
			NDB_FREE(result_datums);
			for (i = 0; i < nvec; i++)
			{
				if (data[i] != NULL)
					NDB_FREE(data[i]);
				if (projected[i] != NULL)
					NDB_FREE(projected[i]);
			}
			for (c = 0; c < n_components; c++)
			{
				if (components[c] != NULL)
					NDB_FREE(components[c]);
			}
			NDB_FREE(data);
			NDB_FREE(projected);
			NDB_FREE(components);
			NDB_FREE(mean);
			NDB_FREE(tbl_str);
			NDB_FREE(col_str);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("reduce_pca: invalid type length for FLOAT4OID: %d", typlen)));
		}

		vec_array = construct_array(vec_datums,
									n_components,
									FLOAT4OID,
									typlen,
									typbyval,
									typalign);

		/* Validate constructed array */
		if (vec_array == NULL)
		{
			NDB_FREE(vec_datums);
			/* Clean up and error */
			for (i = 0; i < j; i++)
			{
				if (result_datums[i] != 0)
				{
					ArrayType  *arr = DatumGetArrayTypeP(result_datums[i]);

					if (arr != NULL)
						NDB_FREE(arr);
				}
			}
			NDB_FREE(result_datums);
			for (i = 0; i < nvec; i++)
			{
				if (data[i] != NULL)
					NDB_FREE(data[i]);
				if (projected[i] != NULL)
					NDB_FREE(projected[i]);
			}
			for (c = 0; c < n_components; c++)
			{
				if (components[c] != NULL)
					NDB_FREE(components[c]);
			}
			NDB_FREE(data);
			NDB_FREE(projected);
			NDB_FREE(components);
			NDB_FREE(mean);
			NDB_FREE(tbl_str);
			NDB_FREE(col_str);
			ereport(ERROR,
					(errcode(ERRCODE_OUT_OF_MEMORY),
					 errmsg("reduce_pca: failed to construct array for row %d", j)));
		}

		/* Validate array structure */
		if (ARR_NDIM(vec_array) != 1 || ARR_DIMS(vec_array)[0] != n_components)
		{
			NDB_FREE(vec_array);
			NDB_FREE(vec_datums);
			/* Clean up and error */
			for (i = 0; i < j; i++)
			{
				if (result_datums[i] != 0)
				{
					ArrayType  *arr = DatumGetArrayTypeP(result_datums[i]);

					if (arr != NULL)
						NDB_FREE(arr);
				}
			}
			NDB_FREE(result_datums);
			for (i = 0; i < nvec; i++)
			{
				if (data[i] != NULL)
					NDB_FREE(data[i]);
				if (projected[i] != NULL)
					NDB_FREE(projected[i]);
			}
			for (c = 0; c < n_components; c++)
			{
				if (components[c] != NULL)
					NDB_FREE(components[c]);
			}
			NDB_FREE(data);
			NDB_FREE(projected);
			NDB_FREE(components);
			NDB_FREE(mean);
			NDB_FREE(tbl_str);
			NDB_FREE(col_str);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("reduce_pca: constructed array has invalid dimensions for row %d", j)));
		}

		result_datums[j] = PointerGetDatum(vec_array);
		NDB_FREE(vec_datums);
	}

	{
		int16		typlen;
		bool		typbyval;
		char		typalign;

		get_typlenbyvalalign(FLOAT4ARRAYOID, &typlen, &typbyval, &typalign);

		/* Validate type information for array of arrays */
		/* Note: typlen = -1 is valid for variable-length types (arrays) */
		if (typlen < -1)
		{
			/* Clean up */
			for (j = 0; j < nvec; j++)
			{
				if (result_datums[j] != 0)
				{
					ArrayType  *arr = DatumGetArrayTypeP(result_datums[j]);

					NDB_FREE(arr);
				}
			}
			NDB_FREE(result_datums);
			for (j = 0; j < nvec; j++)
			{
				if (data[j] != NULL)
					NDB_FREE(data[j]);
				if (projected[j] != NULL)
					NDB_FREE(projected[j]);
			}
			for (c = 0; c < n_components; c++)
			{
				if (components[c] != NULL)
					NDB_FREE(components[c]);
			}
			NDB_FREE(data);
			NDB_FREE(projected);
			NDB_FREE(components);
			NDB_FREE(mean);
			NDB_FREE(tbl_str);
			NDB_FREE(col_str);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("reduce_pca: invalid type length for FLOAT4ARRAYOID: %d", typlen)));
		}

		result_array = construct_array(result_datums,
									   nvec,
									   FLOAT4ARRAYOID,
									   typlen,
									   typbyval,
									   typalign);

		/* Validate final array */
		if (result_array == NULL)
		{
			/* Clean up */
			for (j = 0; j < nvec; j++)
			{
				if (result_datums[j] != 0)
				{
					ArrayType  *arr = DatumGetArrayTypeP(result_datums[j]);

					NDB_FREE(arr);
				}
			}
			NDB_FREE(result_datums);
			for (j = 0; j < nvec; j++)
			{
				if (data[j] != NULL)
					NDB_FREE(data[j]);
				if (projected[j] != NULL)
					NDB_FREE(projected[j]);
			}
			for (c = 0; c < n_components; c++)
			{
				if (components[c] != NULL)
					NDB_FREE(components[c]);
			}
			NDB_FREE(data);
			NDB_FREE(projected);
			NDB_FREE(components);
			NDB_FREE(mean);
			NDB_FREE(tbl_str);
			NDB_FREE(col_str);
			ereport(ERROR,
					(errcode(ERRCODE_OUT_OF_MEMORY),
					 errmsg("reduce_pca: failed to construct result array")));
		}

		/* Validate final array structure */
		if (ARR_NDIM(result_array) != 1 || ARR_DIMS(result_array)[0] != nvec)
		{
			NDB_FREE(result_array);
			/* Clean up */
			for (j = 0; j < nvec; j++)
			{
				if (result_datums[j] != 0)
				{
					ArrayType  *arr = DatumGetArrayTypeP(result_datums[j]);

					NDB_FREE(arr);
				}
			}
			NDB_FREE(result_datums);
			for (j = 0; j < nvec; j++)
			{
				if (data[j] != NULL)
					NDB_FREE(data[j]);
				if (projected[j] != NULL)
					NDB_FREE(projected[j]);
			}
			for (c = 0; c < n_components; c++)
			{
				if (components[c] != NULL)
					NDB_FREE(components[c]);
			}
			NDB_FREE(data);
			NDB_FREE(projected);
			NDB_FREE(components);
			NDB_FREE(mean);
			NDB_FREE(tbl_str);
			NDB_FREE(col_str);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("reduce_pca: result array has invalid dimensions")));
		}

		/* Validate element type matches expected FLOAT4ARRAYOID */
		if (ARR_ELEMTYPE(result_array) != FLOAT4ARRAYOID)
		{
			NDB_FREE(result_array);
			/* Clean up */
			for (j = 0; j < nvec; j++)
			{
				if (result_datums[j] != 0)
				{
					ArrayType  *arr = DatumGetArrayTypeP(result_datums[j]);

					NDB_FREE(arr);
				}
			}
			NDB_FREE(result_datums);
			for (j = 0; j < nvec; j++)
			{
				if (data[j] != NULL)
					NDB_FREE(data[j]);
				if (projected[j] != NULL)
					NDB_FREE(projected[j]);
			}
			for (c = 0; c < n_components; c++)
			{
				if (components[c] != NULL)
					NDB_FREE(components[c]);
			}
			NDB_FREE(data);
			NDB_FREE(projected);
			NDB_FREE(components);
			NDB_FREE(mean);
			NDB_FREE(tbl_str);
			NDB_FREE(col_str);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("reduce_pca: result array has incorrect element type (expected %u, got %u)",
							FLOAT4ARRAYOID, ARR_ELEMTYPE(result_array))));
		}

		/*
		 * Validate nested array structure - check directly from result_datums
		 * before construction
		 */

		/*
		 * This validation happens after construction but validates the source
		 * arrays
		 */

		/*
		 * We validate a sample to ensure structure is correct without risking
		 * crashes
		 */
		for (j = 0; j < nvec && j < 10; j++)
		{
			ArrayType  *nested_array;

			if (result_datums[j] == 0)
			{
				NDB_FREE(result_array);
				/* Clean up */
				for (i = 0; i < nvec; i++)
				{
					if (result_datums[i] != 0)
					{
						ArrayType  *arr = DatumGetArrayTypeP(result_datums[i]);

						NDB_FREE(arr);
					}
				}
				NDB_FREE(result_datums);
				for (i = 0; i < nvec; i++)
				{
					if (data[i] != NULL)
						NDB_FREE(data[i]);
					if (projected[i] != NULL)
						NDB_FREE(projected[i]);
				}
				for (c = 0; c < n_components; c++)
				{
					if (components[c] != NULL)
						NDB_FREE(components[c]);
				}
				NDB_FREE(data);
				NDB_FREE(projected);
				NDB_FREE(components);
				NDB_FREE(mean);
				NDB_FREE(tbl_str);
				NDB_FREE(col_str);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("reduce_pca: nested array element %d datum is NULL", j)));
			}

			nested_array = DatumGetArrayTypeP(result_datums[j]);
			if (nested_array == NULL)
			{
				NDB_FREE(result_array);
				/* Clean up */
				for (i = 0; i < nvec; i++)
				{
					if (result_datums[i] != 0)
					{
						ArrayType  *arr = DatumGetArrayTypeP(result_datums[i]);

						NDB_FREE(arr);
					}
				}
				NDB_FREE(result_datums);
				for (i = 0; i < nvec; i++)
				{
					if (data[i] != NULL)
						NDB_FREE(data[i]);
					if (projected[i] != NULL)
						NDB_FREE(projected[i]);
				}
				for (c = 0; c < n_components; c++)
				{
					if (components[c] != NULL)
						NDB_FREE(components[c]);
				}
				NDB_FREE(data);
				NDB_FREE(projected);
				NDB_FREE(components);
				NDB_FREE(mean);
				NDB_FREE(tbl_str);
				NDB_FREE(col_str);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("reduce_pca: nested array element %d is invalid", j)));
			}

			/* Validate nested array is one-dimensional with correct size */
			if (ARR_NDIM(nested_array) != 1 || ARR_DIMS(nested_array)[0] != n_components)
			{
				NDB_FREE(result_array);
				/* Clean up */
				for (i = 0; i < nvec; i++)
				{
					if (result_datums[i] != 0)
					{
						ArrayType  *arr = DatumGetArrayTypeP(result_datums[i]);

						NDB_FREE(arr);
					}
				}
				NDB_FREE(result_datums);
				for (i = 0; i < nvec; i++)
				{
					if (data[i] != NULL)
						NDB_FREE(data[i]);
					if (projected[i] != NULL)
						NDB_FREE(projected[i]);
				}
				for (c = 0; c < n_components; c++)
				{
					if (components[c] != NULL)
						NDB_FREE(components[c]);
				}
				NDB_FREE(data);
				NDB_FREE(projected);
				NDB_FREE(components);
				NDB_FREE(mean);
				NDB_FREE(tbl_str);
				NDB_FREE(col_str);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("reduce_pca: nested array element %d has invalid dimensions (expected 1 dim with %d elements, got %d dims)",
								j, n_components, ARR_NDIM(nested_array))));
			}

			/* Validate nested array element type is FLOAT4OID */
			if (ARR_ELEMTYPE(nested_array) != FLOAT4OID)
			{
				NDB_FREE(result_array);
				/* Clean up */
				for (i = 0; i < nvec; i++)
				{
					if (result_datums[i] != 0)
					{
						ArrayType  *arr = DatumGetArrayTypeP(result_datums[i]);

						NDB_FREE(arr);
					}
				}
				NDB_FREE(result_datums);
				for (i = 0; i < nvec; i++)
				{
					if (data[i] != NULL)
						NDB_FREE(data[i]);
					if (projected[i] != NULL)
						NDB_FREE(projected[i]);
				}
				for (c = 0; c < n_components; c++)
				{
					if (components[c] != NULL)
						NDB_FREE(components[c]);
				}
				NDB_FREE(data);
				NDB_FREE(projected);
				NDB_FREE(components);
				NDB_FREE(mean);
				NDB_FREE(tbl_str);
				NDB_FREE(col_str);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("reduce_pca: nested array element %d has incorrect element type (expected %u, got %u)",
								j, FLOAT4OID, ARR_ELEMTYPE(nested_array))));
			}
		}
	}

	for (j = 0; j < nvec; j++)
	{
		NDB_FREE(data[j]);
		NDB_FREE(projected[j]);
	}
	for (c = 0; c < n_components; c++)
		NDB_FREE(components[c]);
	NDB_FREE(data);
	NDB_FREE(projected);
	NDB_FREE(components);
	NDB_FREE(mean);
	NDB_FREE(result_datums);
	NDB_FREE(tbl_str);
	NDB_FREE(col_str);

	/* Final validation: ensure array is properly constructed and typed */
	if (result_array == NULL)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("reduce_pca: result array is NULL")));
	}

	/* Ensure array is in current memory context for proper return */
	if (GetMemoryChunkContext((void *) result_array) != CurrentMemoryContext)
	{
		/* This shouldn't happen with construct_array, but check anyway */
		elog(WARNING, "reduce_pca: result array is not in current memory context");
	}

	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * =============================================================================
 * Isolation Forest - Outlier Detection
 * =============================================================================
 * Anomaly detection using ensemble of isolation trees
 * - n_trees: Number of trees in the forest (default 100)
 * - contamination: Expected proportion of outliers (0.0-0.5)
 * - Returns anomaly scores (higher = more anomalous)
 */

typedef struct IsoTreeNode
{
	int			split_dim;		/* Dimension to split on (-1 = leaf) */
	float		split_val;		/* Value to split at */
	struct IsoTreeNode *left;
	struct IsoTreeNode *right;
	int			size;			/* Number of points in this node */
}			IsoTreeNode;

static IsoTreeNode *
build_iso_tree(float **data,
			   int *indices,
			   int n,
			   int dim,
			   int depth,
			   int max_depth)
{
	NDB_DECLARE(IsoTreeNode *, node);
	int			i,
				split_dim;
	float		split_val,
				min_val,
				max_val;
	int			left_count,
				right_count;
	int		   *left_indices = NULL,
			   *right_indices = NULL;

	NDB_ALLOC(node, IsoTreeNode, 1);
	node->size = n;

	if (n <= 1 || depth >= max_depth)
	{
		node->split_dim = -1;	/* Leaf node */
		return node;
	}

	split_dim = rand() % dim;
	node->split_dim = split_dim;

	min_val = max_val = data[indices[0]][split_dim];
	for (i = 1; i < n; i++)
	{
		float		val = data[indices[i]][split_dim];

		if (val < min_val)
			min_val = val;
		if (val > max_val)
			max_val = val;
	}

	if (max_val - min_val < 1e-6)
	{
		node->split_dim = -1;	/* Can't split */
		return node;
	}
	split_val = min_val + (float) (((double) rand() / (double) RAND_MAX)) * (max_val - min_val);
	node->split_val = split_val;

	NDB_ALLOC(left_indices, int, n);
	NDB_ALLOC(right_indices, int, n);
	left_count = right_count = 0;

	for (i = 0; i < n; i++)
	{
		if (data[indices[i]][split_dim] < split_val)
			left_indices[left_count++] = indices[i];
		else
			right_indices[right_count++] = indices[i];
	}

	if (left_count > 0)
		node->left = build_iso_tree(data,
									left_indices,
									left_count,
									dim,
									depth + 1,
									max_depth);
	if (right_count > 0)
		node->right = build_iso_tree(data,
									 right_indices,
									 right_count,
									 dim,
									 depth + 1,
									 max_depth);

	NDB_FREE(left_indices);
	NDB_FREE(right_indices);

	return node;
}

static double
iso_tree_path_length(IsoTreeNode * node, const float *point, int depth)
{
	double		h;

	if (node->split_dim == -1)
	{
		if (node->size <= 1)
			return depth;
		h = log(node->size) + 0.5772156649; /* Euler's constant */
		return depth + h;
	}

	if (point[node->split_dim] < node->split_val && node->left)
		return iso_tree_path_length(node->left, point, depth + 1);
	else if (node->right)
		return iso_tree_path_length(node->right, point, depth + 1);
	else
		return depth;
}

static void
free_iso_tree(IsoTreeNode * node)
{
	if (node == NULL)
		return;
	free_iso_tree(node->left);
	free_iso_tree(node->right);
	NDB_FREE(node);
}

PG_FUNCTION_INFO_V1(detect_outliers);

Datum
detect_outliers(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *column_name;
	int			n_trees;
	float		contamination;
	char	   *tbl_str;
	char	   *col_str;
	float	  **data;
	int			nvec,
				dim;

	NDB_DECLARE(IsoTreeNode * *, forest);
	NDB_DECLARE(double *, scores);
	int			i,
				t;

	NDB_DECLARE(int *, indices);
	int			max_depth;
	double		avg_path_length_full;
	ArrayType  *result_array;

	NDB_DECLARE(Datum *, result_datums);
	int16		typlen;
	bool		typbyval;
	char		typalign;

	table_name = PG_GETARG_TEXT_PP(0);
	column_name = PG_GETARG_TEXT_PP(1);
	n_trees = PG_GETARG_INT32(2);
	contamination = PG_GETARG_FLOAT4(3);

	if (n_trees < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("n_trees must be at least 1")));

	if (contamination < 0.0 || contamination > 0.5)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("contamination must be between 0.0 and "
						"0.5")));

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(column_name);

	elog(DEBUG1,
		 "neurondb: Isolation Forest on %s.%s (n_trees=%d, contamination=%.3f)",
		 tbl_str,
		 col_str,
		 n_trees,
		 contamination);

	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);
	if (nvec == 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found")));

	max_depth = (int) ceil(log2(nvec));
	NDB_ALLOC(forest, IsoTreeNode *, n_trees);
	NDB_ALLOC(indices, int, nvec);

	for (t = 0; t < n_trees; t++)
	{
		int			sample_size = (nvec < 256) ? nvec : 256;

		for (i = 0; i < sample_size; i++)
			indices[i] = rand() % nvec;

		forest[t] = build_iso_tree(
								   data, indices, sample_size, dim, 0, max_depth);
	}

	avg_path_length_full = (nvec > 1) ? 2.0 * (log(nvec - 1) + 0.5772156649)
		- 2.0 * (nvec - 1.0) / nvec
		: 0.0;
	NDB_ALLOC(scores, double, nvec);

	for (i = 0; i < nvec; i++)
	{
		double		avg_path = 0.0;

		for (t = 0; t < n_trees; t++)
			avg_path += iso_tree_path_length(forest[t], data[i], 0);
		avg_path /= n_trees;

		if (avg_path_length_full > 0)
			scores[i] = pow(2.0, -avg_path / avg_path_length_full);
		else
			scores[i] = 0.0;
	}

	NDB_ALLOC(result_datums, Datum, nvec);
	for (i = 0; i < nvec; i++)
		result_datums[i] = Float4GetDatum((float) scores[i]);

	get_typlenbyvalalign(FLOAT4OID, &typlen, &typbyval, &typalign);
	result_array = construct_array(
								   result_datums, nvec, FLOAT4OID, typlen, typbyval, typalign);

	for (t = 0; t < n_trees; t++)
		free_iso_tree(forest[t]);
	for (i = 0; i < nvec; i++)
		NDB_FREE(data[i]);
	NDB_FREE(data);
	NDB_FREE(forest);
	NDB_FREE(scores);
	NDB_FREE(indices);
	NDB_FREE(result_datums);
	NDB_FREE(tbl_str);
	NDB_FREE(col_str);

	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * =============================================================================
 * KNN Graph Construction
 * =============================================================================
 * Build k-nearest neighbor graph for vectors
 * - k: Number of neighbors per point
 * - Returns edge list as array of (source, target, distance) tuples
 */

typedef struct KNNEdge
{
	int			target;
	float		distance;
}			KNNEdge;

static int
knn_edge_compare(const void *a, const void *b)
{
	const		KNNEdge *ea = (const KNNEdge *) a;
	const		KNNEdge *eb = (const KNNEdge *) b;

	if (ea->distance < eb->distance)
		return -1;
	if (ea->distance > eb->distance)
		return 1;
	return 0;
}

PG_FUNCTION_INFO_V1(build_knn_graph);

Datum
build_knn_graph(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *column_name;
	int			k;
	char	   *tbl_str;
	char	   *col_str;
	float	  **data;
	int			nvec,
				dim;
	int			i,
				j,
				n;

	NDB_DECLARE(KNNEdge *, edges);
	ArrayType  *result_array;

	NDB_DECLARE(Datum *, result_datums);
	int			result_count;
	int16		typlen;
	bool		typbyval;
	char		typalign;

	table_name = PG_GETARG_TEXT_PP(0);
	column_name = PG_GETARG_TEXT_PP(1);
	k = PG_GETARG_INT32(2);

	if (k < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("k must be at least 1")));

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(column_name);

	elog(DEBUG1,
		 "neurondb: Building KNN graph on %s.%s (k=%d)",
		 tbl_str,
		 col_str,
		 k);

	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);
	if (data == NULL || nvec == 0)
	{
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found")));
	}

	if (dim <= 0)
	{
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		/* Free data array and rows if data is not NULL */
		if (data != NULL)
		{
			for (i = 0; i < nvec; i++)
			{
				if (data[i] != NULL)
					NDB_FREE(data[i]);
			}
			NDB_FREE(data);
		}
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Invalid vector dimension: %d", dim)));
	}

	if (k >= nvec)
		k = nvec - 1;

	NDB_ALLOC(edges, KNNEdge, nvec);
	result_count = 0;
	NDB_ALLOC(result_datums, Datum, nvec * k * 3);

	for (i = 0; i < nvec; i++)
	{
		double		dist_sq;
		double		diff;

		for (j = 0; j < nvec; j++)
		{
			if (i == j)
				continue;

			dist_sq = 0.0;
			for (n = 0; n < dim; n++)
			{
				diff = (double) data[i][n] - (double) data[j][n];
				dist_sq += diff * diff;
			}
			edges[j].target = j;
			edges[j].distance = sqrt(dist_sq);
		}

		qsort(edges, nvec, sizeof(KNNEdge), knn_edge_compare);

		for (j = 0; j < k && j < nvec - 1; j++)
		{
			result_datums[result_count++] = Int32GetDatum(i);
			result_datums[result_count++] =
				Int32GetDatum(edges[j].target);
			result_datums[result_count++] =
				Float4GetDatum(edges[j].distance);
		}
	}

	get_typlenbyvalalign(FLOAT4OID, &typlen, &typbyval, &typalign);
	result_array = construct_array(result_datums,
								   result_count,
								   FLOAT4OID,
								   typlen,
								   typbyval,
								   typalign);

	for (i = 0; i < nvec; i++)
		NDB_FREE(data[i]);
	NDB_FREE(data);
	NDB_FREE(edges);
	NDB_FREE(result_datums);
	NDB_FREE(tbl_str);
	NDB_FREE(col_str);

	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * =============================================================================
 * Embedding Quality Metrics
 * =============================================================================
 * Compute quality metrics for embeddings (silhouette score, etc.)
 * - Returns quality score between -1 and 1 (higher = better)
 */

PG_FUNCTION_INFO_V1(compute_embedding_quality);

Datum
compute_embedding_quality(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *column_name;
	text	   *cluster_column;
	char	   *tbl_str;
	char	   *col_str;
	char	   *cluster_col_str;
	float	  **data;

	NDB_DECLARE(int *, clusters);
	int			nvec,
				dim;
	int			i,
				j;

	NDB_DECLARE(double *, a_scores);	/* Average distance to same cluster */
	NDB_DECLARE(double *, b_scores);	/* Average distance to nearest other
										 * cluster */
	double		silhouette;
	StringInfoData sql;
	int			ret;

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext;

	table_name = PG_GETARG_TEXT_PP(0);
	column_name = PG_GETARG_TEXT_PP(1);
	cluster_column = PG_GETARG_TEXT_PP(2);

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(column_name);
	cluster_col_str = text_to_cstring(cluster_column);

	elog(DEBUG1,
		 "neurondb: Computing embedding quality for %s.%s (clusters=%s)",
		 tbl_str,
		 col_str,
		 cluster_col_str);

	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);
	if (data == NULL || nvec == 0)
	{
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		NDB_FREE(cluster_col_str);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found")));
	}

	if (dim <= 0)
	{
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		NDB_FREE(cluster_col_str);
		/* Free data array and rows if data is not NULL */
		if (data != NULL)
		{
			for (i = 0; i < nvec; i++)
			{
				if (data[i] != NULL)
					NDB_FREE(data[i]);
			}
			NDB_FREE(data);
		}
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Invalid vector dimension: %d", dim)));
	}

	oldcontext = CurrentMemoryContext;
	NDB_ALLOC(clusters, int, nvec);

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql, "SELECT %s FROM %s", cluster_col_str, tbl_str);
	ret = ndb_spi_execute(spi_session, sql.data, true, 0);

	if (ret != SPI_OK_SELECT || (int) SPI_processed != nvec)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		NDB_FREE(clusters);
		NDB_FREE(tbl_str);
		NDB_FREE(cluster_col_str);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Failed to fetch cluster assignments")));
	}

	for (i = 0; i < nvec; i++)
	{
		int32		val;

		if (ndb_spi_get_int32(spi_session, i, 1, &val))
		{
			clusters[i] = val;
		}
		else
		{
			clusters[i] = -1;
		}
	}

	ndb_spi_stringinfo_free(spi_session, &sql);
	NDB_SPI_SESSION_END(spi_session);

	NDB_ALLOC(a_scores, double, nvec);
	NDB_ALLOC(b_scores, double, nvec);

	for (i = 0; i < nvec; i++)
	{
		int			my_cluster = clusters[i];
		int			same_count = 0;
		double		same_dist = 0.0;
		double		min_other_dist = DBL_MAX;
		double		dist;
		int			d;
		double		diff;

		if (my_cluster == -1)	/* Noise point */
			continue;

		for (j = 0; j < nvec; j++)
		{
			if (i == j)
				continue;

			dist = 0.0;
			for (d = 0; d < dim; d++)
			{
				diff = (double) data[i][d] - (double) data[j][d];
				dist += diff * diff;
			}
			dist = sqrt(dist);

			if (clusters[j] == my_cluster)
			{
				same_dist += dist;
				same_count++;
			}
			else if (clusters[j] != -1)
			{
				if (dist < min_other_dist)
					min_other_dist = dist;
			}
		}

		if (same_count > 0)
			a_scores[i] = same_dist / same_count;
		b_scores[i] = min_other_dist;
	}

	{
		int			valid_count = 0;
		double		s;

		silhouette = 0.0;
		for (i = 0; i < nvec; i++)
		{
			if (clusters[i] == -1)
				continue;

			if (a_scores[i] < b_scores[i])
				s = 1.0 - a_scores[i] / b_scores[i];
			else if (a_scores[i] > b_scores[i])
				s = b_scores[i] / a_scores[i] - 1.0;
			else
				s = 0.0;

			silhouette += s;
			valid_count++;
		}

		if (valid_count > 0)
			silhouette /= valid_count;
	}

	for (i = 0; i < nvec; i++)
		NDB_FREE(data[i]);
	NDB_FREE(data);
	NDB_FREE(clusters);
	NDB_FREE(a_scores);
	NDB_FREE(b_scores);
	NDB_FREE(tbl_str);
	NDB_FREE(col_str);
	NDB_FREE(cluster_col_str);

	PG_RETURN_FLOAT8(silhouette);
}
