/*-------------------------------------------------------------------------
 *
 * vector_batch.c
 *		Batch operations for bulk vector processing
 *
 * Implements efficient batch processing of multiple vectors for
 * distance calculations, normalization, and aggregation.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/vector/vector_batch.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/syscache.h"
#include "utils/lsyscache.h"
#include "catalog/pg_type.h"
#include "catalog/namespace.h"
#include "access/htup_details.h"
#include <math.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/* Forward declarations */
extern float4 l2_distance(Vector *a, Vector *b);
extern float4 cosine_distance(Vector *a, Vector *b);
extern float4 inner_product_distance(Vector *a, Vector *b);
extern Vector *normalize_vector_new(Vector *v);

/*
 * vector_l2_distance_batch
 *
 * Compute L2 distance between query vector and array of vectors.
 * Returns array of distances.
 */
PG_FUNCTION_INFO_V1(vector_l2_distance_batch);
Datum
vector_l2_distance_batch(PG_FUNCTION_ARGS)
{
	ArrayType  *vec_array;
	Vector	   *query;
	ArrayType  *result;
	Datum	   *elems;
	bool	   *nulls;
	int			nvec;
	int			i;
	Oid			vector_oid;
	int16		typlen;
	bool		typbyval;
	char		typalign;
	bool		isnull;
	Datum		vec_datum;
	Vector	   *vec;
	int			lb;
	int			idx;

	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_l2_distance_batch requires 2 arguments, got %d",
						PG_NARGS())));

	vec_array = PG_GETARG_ARRAYTYPE_P(0);
	query = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(query);

	if (vec_array == NULL || query == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("vector array and query vector must not be NULL")));

	if (query->dim <= 0 || query->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid query vector dimension: %d",
						query->dim)));

	/* Get vector type OID and type information */
	vector_oid = ARR_ELEMTYPE(vec_array);
	if (!OidIsValid(vector_oid))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid array element type")));

	get_typlenbyvalalign(vector_oid, &typlen, &typbyval, &typalign);

	if (ARR_NDIM(vec_array) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("vector array must be one-dimensional")));

	nvec = ARR_DIMS(vec_array)[0];
	if (nvec <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector array must not be empty")));

	elems = NULL;
	nulls = NULL;
	NDB_ALLOC(elems, Datum, nvec);
	NDB_ALLOC(nulls, bool, nvec);

	/* Get the lower bound of the input array (typically 1 for PostgreSQL arrays) */
	lb = ARR_LBOUND(vec_array)[0];

	for (i = 0; i < nvec; i++)
	{
		idx = lb + i;	/* Convert 0-based loop index to array index */

		vec_datum = array_ref(vec_array,
							  1,
							  &idx,
							  -1,
							  -1,
							  false,
							  'i',
							  &isnull);

		if (isnull)
		{
			nulls[i] = true;
			elems[i] = (Datum) 0;
			continue;
		}

		vec = DatumGetVector(vec_datum);
		if (vec == NULL)
		{
			nulls[i] = true;
			elems[i] = (Datum) 0;
			continue;
		}
		NDB_CHECK_VECTOR_VALID(vec);
		if (vec->dim <= 0 || vec->dim > VECTOR_MAX_DIM)
		{
			nulls[i] = true;
			elems[i] = (Datum) 0;
			continue;
		}

		if (vec->dim != query->dim)
		{
			nulls[i] = true;
			elems[i] = (Datum) 0;
			continue;
		}

		elems[i] = Float4GetDatum(l2_distance(vec, query));
		nulls[i] = false;
	}

	result = construct_array(elems, nvec, FLOAT4OID, sizeof(float4), true, 'i');
	NDB_FREE(elems);
	NDB_FREE(nulls);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * vector_cosine_distance_batch
 *
 * Compute cosine distance between query vector and array of vectors.
 */
PG_FUNCTION_INFO_V1(vector_cosine_distance_batch);
Datum
vector_cosine_distance_batch(PG_FUNCTION_ARGS)
{
	ArrayType  *vec_array;
	Vector	   *query;
	ArrayType  *result;
	Datum	   *elems;
	bool	   *nulls;
	int			nvec;
	int			i;
	Oid			vector_oid;
	int16		typlen;
	bool		typbyval;
	char		typalign;
	bool		isnull;
	Datum		vec_datum;
	Vector	   *vec;
	int			lb;
	int			idx;

	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_cosine_distance_batch requires 2 arguments, got %d",
						PG_NARGS())));

	vec_array = PG_GETARG_ARRAYTYPE_P(0);
	query = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(query);

	if (vec_array == NULL || query == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("vector array and query vector must not be NULL")));

	if (query->dim <= 0 || query->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid query vector dimension: %d",
						query->dim)));

	/* Get vector type OID and type information */
	vector_oid = ARR_ELEMTYPE(vec_array);
	if (!OidIsValid(vector_oid))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid array element type")));

	get_typlenbyvalalign(vector_oid, &typlen, &typbyval, &typalign);

	if (ARR_NDIM(vec_array) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("vector array must be one-dimensional")));

	nvec = ARR_DIMS(vec_array)[0];
	if (nvec <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector array must not be empty")));

	elems = NULL;
	nulls = NULL;
	NDB_ALLOC(elems, Datum, nvec);
	NDB_ALLOC(nulls, bool, nvec);

	/* Get the lower bound of the input array (typically 1 for PostgreSQL arrays) */
	lb = ARR_LBOUND(vec_array)[0];

	for (i = 0; i < nvec; i++)
	{
		idx = lb + i;	/* Convert 0-based loop index to array index */

		vec_datum = array_ref(vec_array,
							  1,
							  &idx,
							  -1,
							  -1,
							  false,
							  'i',
							  &isnull);

		if (isnull)
		{
			nulls[i] = true;
			elems[i] = (Datum) 0;
			continue;
		}

		vec = DatumGetVector(vec_datum);
		if (vec == NULL)
		{
			nulls[i] = true;
			elems[i] = (Datum) 0;
			continue;
		}
		NDB_CHECK_VECTOR_VALID(vec);
		if (vec->dim <= 0 || vec->dim > VECTOR_MAX_DIM)
		{
			nulls[i] = true;
			elems[i] = (Datum) 0;
			continue;
		}

		if (vec->dim != query->dim)
		{
			nulls[i] = true;
			elems[i] = (Datum) 0;
			continue;
		}

		elems[i] = Float4GetDatum(cosine_distance(vec, query));
		nulls[i] = false;
	}

	result = construct_array(elems, nvec, FLOAT4OID, sizeof(float4), true, 'i');
	NDB_FREE(elems);
	NDB_FREE(nulls);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * vector_inner_product_batch
 *
 * Compute inner product between query vector and array of vectors.
 */
PG_FUNCTION_INFO_V1(vector_inner_product_batch);
Datum
vector_inner_product_batch(PG_FUNCTION_ARGS)
{
	ArrayType  *vec_array;
	Vector	   *query;
	ArrayType  *result;
	Datum	   *elems;
	bool	   *nulls;
	int			nvec;
	int			i;
	Oid			vector_oid;
	int16		typlen;
	bool		typbyval;
	char		typalign;
	bool		isnull;
	Datum		vec_datum;
	Vector	   *vec;
	int			lb;
	int			idx;

	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_inner_product_batch requires 2 arguments, got %d",
						PG_NARGS())));

	vec_array = PG_GETARG_ARRAYTYPE_P(0);
	query = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(query);

	if (vec_array == NULL || query == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("vector array and query vector must not be NULL")));

	if (query->dim <= 0 || query->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid query vector dimension: %d",
						query->dim)));

	/* Get vector type OID and type information */
	vector_oid = ARR_ELEMTYPE(vec_array);
	if (!OidIsValid(vector_oid))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid array element type")));

	get_typlenbyvalalign(vector_oid, &typlen, &typbyval, &typalign);

	if (ARR_NDIM(vec_array) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("vector array must be one-dimensional")));

	nvec = ARR_DIMS(vec_array)[0];
	if (nvec <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector array must not be empty")));

	elems = NULL;
	nulls = NULL;
	NDB_ALLOC(elems, Datum, nvec);
	NDB_ALLOC(nulls, bool, nvec);

	/* Get the lower bound of the input array (typically 1 for PostgreSQL arrays) */
	lb = ARR_LBOUND(vec_array)[0];

	for (i = 0; i < nvec; i++)
	{
		idx = lb + i;	/* Convert 0-based loop index to array index */

		vec_datum = array_ref(vec_array,
							  1,
							  &idx,
							  -1,
							  -1,
							  false,
							  'i',
							  &isnull);

		if (isnull)
		{
			nulls[i] = true;
			elems[i] = (Datum) 0;
			continue;
		}

		vec = DatumGetVector(vec_datum);
		if (vec == NULL)
		{
			nulls[i] = true;
			elems[i] = (Datum) 0;
			continue;
		}
		NDB_CHECK_VECTOR_VALID(vec);
		if (vec->dim <= 0 || vec->dim > VECTOR_MAX_DIM)
		{
			nulls[i] = true;
			elems[i] = (Datum) 0;
			continue;
		}

		if (vec->dim != query->dim)
		{
			nulls[i] = true;
			elems[i] = (Datum) 0;
			continue;
		}

		elems[i] = Float4GetDatum(-inner_product_distance(vec, query));
		nulls[i] = false;
	}

	result = construct_array(elems, nvec, FLOAT4OID, sizeof(float4), true, 'i');
	NDB_FREE(elems);
	NDB_FREE(nulls);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * vector_normalize_batch
 *
 * Normalize array of vectors (L2 normalization).
 */
PG_FUNCTION_INFO_V1(vector_normalize_batch);
Datum
vector_normalize_batch(PG_FUNCTION_ARGS)
{
	ArrayType  *vec_array;
	ArrayType  *result;
	Datum	   *elems;
	bool	   *nulls;
	int			nvec;
	int			i;
	Oid			vector_oid;
	int16		typlen;
	bool		typbyval;
	char		typalign;
	bool		isnull;
	Datum		vec_datum;
	Vector	   *vec;
	Vector	   *normalized;
	int			lb;
	int			idx;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_normalize_batch requires 1 argument, got %d",
						PG_NARGS())));

	vec_array = PG_GETARG_ARRAYTYPE_P(0);

	if (vec_array == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("vector array must not be NULL")));

	if (ARR_NDIM(vec_array) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("vector array must be one-dimensional")));

	nvec = ARR_DIMS(vec_array)[0];
	if (nvec <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector array must not be empty")));

	/* Get vector type OID and type information */
	vector_oid = ARR_ELEMTYPE(vec_array);
	if (!OidIsValid(vector_oid))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid array element type")));

	get_typlenbyvalalign(vector_oid, &typlen, &typbyval, &typalign);

	elems = NULL;
	nulls = NULL;
	NDB_ALLOC(elems, Datum, nvec);
	NDB_ALLOC(nulls, bool, nvec);

	/* Get the lower bound of the input array (typically 1 for PostgreSQL arrays) */
	lb = ARR_LBOUND(vec_array)[0];

	for (i = 0; i < nvec; i++)
	{
		idx = lb + i;	/* Convert 0-based loop index to array index */

		vec_datum = array_ref(vec_array,
							  1,
							  &idx,
							  -1,
							  -1,
							  false,
							  'i',
							  &isnull);

		if (isnull)
		{
			nulls[i] = true;
			elems[i] = (Datum) 0;
			continue;
		}

		vec = DatumGetVector(vec_datum);
		if (vec == NULL)
		{
			nulls[i] = true;
			elems[i] = (Datum) 0;
			continue;
		}
		NDB_CHECK_VECTOR_VALID(vec);
		if (vec->dim <= 0 || vec->dim > VECTOR_MAX_DIM)
		{
			nulls[i] = true;
			elems[i] = (Datum) 0;
			continue;
		}

		normalized = normalize_vector_new(vec);
		if (normalized == NULL)
		{
			nulls[i] = true;
			elems[i] = (Datum) 0;
			continue;
		}

		elems[i] = PointerGetDatum(normalized);
		nulls[i] = false;
	}

	{
		int			dims[1];
		int			lbs[1];

		dims[0] = nvec;
		lbs[0] = 1;
		result = construct_md_array(elems, nulls, 1, dims, lbs, vector_oid, typlen, typbyval, typalign);
	}
	NDB_FREE(elems);
	NDB_FREE(nulls);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * vector_sum_batch
 *
 * Sum array of vectors element-wise.
 */
PG_FUNCTION_INFO_V1(vector_sum_batch);
Datum
vector_sum_batch(PG_FUNCTION_ARGS)
{
	ArrayType  *vec_array;
	Vector	   *result = NULL;
	int			nvec;
	int			i;
	int			dim = 0;
	int			j;
	bool		isnull;
	Datum		vec_datum;
	Vector	   *vec;
	Oid			vector_oid;
	int16		typlen;
	bool		typbyval;
	char		typalign;
	int			lb;
	int			idx;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_sum_batch requires 1 argument, got %d",
						PG_NARGS())));

	vec_array = PG_GETARG_ARRAYTYPE_P(0);

	if (vec_array == NULL)
		PG_RETURN_NULL();

	if (ARR_NDIM(vec_array) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("vector array must be one-dimensional")));

	nvec = ARR_DIMS(vec_array)[0];
	if (nvec <= 0)
		PG_RETURN_NULL();

	/* Get vector type OID and type information */
	vector_oid = ARR_ELEMTYPE(vec_array);
	if (!OidIsValid(vector_oid))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid array element type")));

	get_typlenbyvalalign(vector_oid, &typlen, &typbyval, &typalign);

	/* Get the lower bound of the input array (typically 1 for PostgreSQL arrays) */
	lb = ARR_LBOUND(vec_array)[0];

	for (i = 0; i < nvec; i++)
	{
		idx = lb + i;	/* Convert 0-based loop index to array index */

		vec_datum = array_ref(vec_array,
							  1,
							  &idx,
							  -1,
							  -1,
							  false,
							  'i',
							  &isnull);

		if (isnull)
			continue;

		vec = DatumGetVector(vec_datum);
		if (vec == NULL)
			continue;

		if (vec->dim <= 0 || vec->dim > VECTOR_MAX_DIM)
			continue;

		if (result == NULL)
		{
			dim = vec->dim;
			result = new_vector(dim);
			if (result == NULL)
				ereport(ERROR,
						(errcode(ERRCODE_OUT_OF_MEMORY),
						 errmsg("out of memory")));
			for (j = 0; j < dim; j++)
				result->data[j] = vec->data[j];
		}
		else
		{
			if (vec->dim != dim)
				continue;
			for (j = 0; j < dim; j++)
				result->data[j] += vec->data[j];
		}
	}

	if (result == NULL)
		PG_RETURN_NULL();

	PG_RETURN_VECTOR_P(result);
}

/*
 * vector_avg_batch
 *
 * Average array of vectors element-wise.
 */
PG_FUNCTION_INFO_V1(vector_avg_batch);
Datum
vector_avg_batch(PG_FUNCTION_ARGS)
{
	ArrayType  *vec_array;
	Vector	   *result = NULL;
	int			nvec;
	int			i;
	int			dim = 0;
	int			count = 0;
	int			j;
	bool		isnull;
	Datum		vec_datum;
	Vector	   *vec;
	Oid			vector_oid;
	int16		typlen;
	bool		typbyval;
	char		typalign;
	int			lb;
	int			idx;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_avg_batch requires 1 argument, got %d",
						PG_NARGS())));

	vec_array = PG_GETARG_ARRAYTYPE_P(0);

	if (vec_array == NULL)
		PG_RETURN_NULL();

	if (ARR_NDIM(vec_array) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("vector array must be one-dimensional")));

	nvec = ARR_DIMS(vec_array)[0];
	if (nvec <= 0)
		PG_RETURN_NULL();

	/* Get vector type OID and type information */
	vector_oid = ARR_ELEMTYPE(vec_array);
	if (!OidIsValid(vector_oid))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid array element type")));

	get_typlenbyvalalign(vector_oid, &typlen, &typbyval, &typalign);

	/* Get the lower bound of the input array (typically 1 for PostgreSQL arrays) */
	lb = ARR_LBOUND(vec_array)[0];

	for (i = 0; i < nvec; i++)
	{
		idx = lb + i;	/* Convert 0-based loop index to array index */

		vec_datum = array_ref(vec_array,
							  1,
							  &idx,
							  -1,
							  -1,
							  false,
							  'i',
							  &isnull);

		if (isnull)
			continue;

		vec = DatumGetVector(vec_datum);
		if (vec == NULL)
			continue;

		if (vec->dim <= 0 || vec->dim > VECTOR_MAX_DIM)
			continue;

		if (result == NULL)
		{
			dim = vec->dim;
			result = new_vector(dim);
			if (result == NULL)
				ereport(ERROR,
						(errcode(ERRCODE_OUT_OF_MEMORY),
						 errmsg("out of memory")));
			for (j = 0; j < dim; j++)
				result->data[j] = vec->data[j];
			count = 1;
		}
		else
		{
			if (vec->dim != dim)
				continue;
			for (j = 0; j < dim; j++)
				result->data[j] += vec->data[j];
			count++;
		}
	}

	if (result == NULL || count == 0)
		PG_RETURN_NULL();

	/* Divide by count */
	if (count > 0)
	{
		for (i = 0; i < dim; i++)
			result->data[i] /= (float4) count;
	}

	PG_RETURN_VECTOR_P(result);
}
