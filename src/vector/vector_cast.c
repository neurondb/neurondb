/*-------------------------------------------------------------------------
 *
 * vector_cast.c
 *		Vector type casting and conversion functions
 *
 * Implements comprehensive type casting between vectors, arrays, and
 * quantized formats (FP16, INT8, sparse).
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/vector/vector_cast.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_types.h"
#include "fmgr.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/numeric.h"
#include "catalog/pg_type.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_macros.h"
#include "neurondb_validation.h"
#include <math.h>
#include <string.h>
#include <stdint.h>

PGDLLEXPORT float fp16_to_float(uint16 h);

static inline uint16
float_to_fp16_local(float f)
{
	uint32		u;
	uint16		sign;
	uint32		mantissa;
	int16		exp;

	memcpy(&u, &f, sizeof(uint32));
	sign = (u >> 16) & 0x8000;
	mantissa = u & 0x7fffff;
	exp = ((u >> 23) & 0xff) - 127 + 15;

	if (exp <= 0)
		return sign;
	else if (exp >= 31)
		return sign | 0x7c00;
	else
		return sign | (exp << 10) | (mantissa >> 13);
}

PG_FUNCTION_INFO_V1(array_to_vector_float4);
Datum
array_to_vector_float4(PG_FUNCTION_ARGS)
{
	ArrayType *arr = NULL;
	Vector *result = NULL;
	float4 *data = NULL;
	int			dim;
	int			i;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("array_to_vector_float4 requires 1 argument, got %d",
						PG_NARGS())));

	arr = PG_GETARG_ARRAYTYPE_P(0);

	if (arr == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("array must not be NULL")));

	if (ARR_NDIM(arr) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("array must be one-dimensional")));

	dim = ARR_DIMS(arr)[0];
	if (dim <= 0 || dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid array dimension: %d (max: %d)",
						dim,
						VECTOR_MAX_DIM)));

	if (ARR_HASNULL(arr))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("array must not contain NULL values")));

	data = (float4 *) ARR_DATA_PTR(arr);
	if (data == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid array data")));

	result = new_vector(dim);
	if (result == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("out of memory")));
	for (i = 0; i < dim; i++)
		result->data[i] = data[i];

	PG_RETURN_VECTOR_P(result);
}

PG_FUNCTION_INFO_V1(array_to_vector_float8);
Datum
array_to_vector_float8(PG_FUNCTION_ARGS)
{
	ArrayType *arr = NULL;
	Vector *result = NULL;
	float8 *data = NULL;
	int			dim;
	int			i;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("array_to_vector_float8 requires 1 argument, got %d",
						PG_NARGS())));

	arr = PG_GETARG_ARRAYTYPE_P(0);

	if (arr == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("array must not be NULL")));

	if (ARR_NDIM(arr) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("array must be one-dimensional")));

	dim = ARR_DIMS(arr)[0];
	if (dim <= 0 || dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid array dimension: %d (max: %d)",
						dim,
						VECTOR_MAX_DIM)));

	if (ARR_HASNULL(arr))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("array must not contain NULL values")));

	data = (float8 *) ARR_DATA_PTR(arr);
	if (data == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid array data")));

	result = new_vector(dim);
	if (result == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("out of memory")));
	for (i = 0; i < dim; i++)
		result->data[i] = (float4) data[i];

	PG_RETURN_VECTOR_P(result);
}

PG_FUNCTION_INFO_V1(array_to_vector_integer);
Datum
array_to_vector_integer(PG_FUNCTION_ARGS)
{
	ArrayType *arr = NULL;
	Vector *result = NULL;
	int32 *data = NULL;
	int			dim;
	int			i;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("array_to_vector_integer requires 1 argument, got %d",
						PG_NARGS())));

	arr = PG_GETARG_ARRAYTYPE_P(0);

	if (arr == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("array must not be NULL")));

	if (ARR_NDIM(arr) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("array must be one-dimensional")));

	dim = ARR_DIMS(arr)[0];
	if (dim <= 0 || dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid array dimension: %d (max: %d)",
						dim,
						VECTOR_MAX_DIM)));

	if (ARR_HASNULL(arr))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("array must not contain NULL values")));

	data = (int32 *) ARR_DATA_PTR(arr);
	if (data == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid array data")));

	result = new_vector(dim);
	if (result == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("out of memory")));
	for (i = 0; i < dim; i++)
		result->data[i] = (float4) data[i];

	PG_RETURN_VECTOR_P(result);
}

PG_FUNCTION_INFO_V1(array_to_vector_numeric);
Datum
array_to_vector_numeric(PG_FUNCTION_ARGS)
{
	ArrayType *arr = NULL;
	Vector *result = NULL;
	Datum *elems = NULL;
	bool *nulls = NULL;
	int			dim;
	int			i;
	Numeric		num;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("array_to_vector_numeric requires 1 argument, got %d",
						PG_NARGS())));

	arr = PG_GETARG_ARRAYTYPE_P(0);

	if (arr == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("array must not be NULL")));

	if (ARR_NDIM(arr) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("array must be one-dimensional")));

	dim = ARR_DIMS(arr)[0];
	if (dim <= 0 || dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid array dimension: %d (max: %d)",
						dim,
						VECTOR_MAX_DIM)));

	deconstruct_array(arr, NUMERICOID, -1, false, 'i', &elems, &nulls, &dim);

	if (ARR_HASNULL(arr) || nulls != NULL)
	{
		for (i = 0; i < dim; i++)
		{
			if (nulls != NULL && nulls[i])
				ereport(ERROR,
						(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
						 errmsg("array must not contain NULL values")));
		}
	}

	result = new_vector(dim);
	if (result == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("out of memory")));

	for (i = 0; i < dim; i++)
	{
		num = DatumGetNumeric(elems[i]);
		result->data[i] = (float4) DatumGetFloat8(DirectFunctionCall1(numeric_float8, NumericGetDatum(num)));
	}

	if (elems)
		pfree(elems);
	if (nulls)
		pfree(nulls);

	PG_RETURN_VECTOR_P(result);
}

/*
 * vector_to_array_float4
 *
 * Convert vector to PostgreSQL float4 array.
 */
PG_FUNCTION_INFO_V1(vector_to_array_float4);
Datum
vector_to_array_float4(PG_FUNCTION_ARGS)
{
	Vector *vec = NULL;
	ArrayType *result = NULL;
	Datum *elems = NULL;
	bool *nulls = NULL;
	int			i;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_to_array_float4 requires 1 argument, got %d",
						PG_NARGS())));

	vec = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(vec);

	if (vec == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("vector must not be NULL")));

	if (vec->dim <= 0 || vec->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension: %d",
						vec->dim)));

	elems = NULL;
	nulls = NULL;
	nalloc(elems, Datum, vec->dim);
	nalloc(nulls, bool, vec->dim);
	if (elems == NULL || nulls == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("out of memory")));

	for (i = 0; i < vec->dim; i++)
	{
		elems[i] = Float4GetDatum(vec->data[i]);
		nulls[i] = false;
	}

	result = construct_array(elems,
							 vec->dim,
							 FLOAT4OID,
							 sizeof(float4),
							 true,
							 'i');
	pfree(elems);
	pfree(nulls);

	PG_RETURN_ARRAYTYPE_P(result);
}

PG_FUNCTION_INFO_V1(vector_to_array_float8);
Datum
vector_to_array_float8(PG_FUNCTION_ARGS)
{
	Vector *vec = NULL;
	ArrayType *result = NULL;
	Datum *elems = NULL;
	bool *nulls = NULL;
	int			i;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_to_array_float8 requires 1 argument, got %d",
						PG_NARGS())));

	vec = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(vec);

	if (vec == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("vector must not be NULL")));

	if (vec->dim <= 0 || vec->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension: %d",
						vec->dim)));

	elems = NULL;
	nulls = NULL;
	nalloc(elems, Datum, vec->dim);
	nalloc(nulls, bool, vec->dim);
	if (elems == NULL || nulls == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("out of memory")));

	for (i = 0; i < vec->dim; i++)
	{
		elems[i] = Float8GetDatum((float8) vec->data[i]);
		nulls[i] = false;
	}

	result = construct_array(elems,
							 vec->dim,
							 FLOAT8OID,
							 sizeof(float8),
							 true,
							 'd');
	pfree(elems);
	pfree(nulls);

	PG_RETURN_ARRAYTYPE_P(result);
}

PG_FUNCTION_INFO_V1(vector_cast_dimension);
Datum
vector_cast_dimension(PG_FUNCTION_ARGS)
{
	Vector *vec = NULL;
	int32		new_dim;
	Vector *result = NULL;
	int			i;

	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_cast_dimension requires 2 arguments, got %d",
						PG_NARGS())));

	vec = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(vec);
	new_dim = PG_GETARG_INT32(1);

	if (vec == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("vector must not be NULL")));

	if (vec->dim <= 0 || vec->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension: %d",
						vec->dim)));

	if (new_dim <= 0 || new_dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid dimension: %d (max: %d)",
						new_dim,
						VECTOR_MAX_DIM)));

	/* Allow dimension changes - truncate if new_dim < vec->dim, pad with zeros if new_dim > vec->dim */
	result = new_vector(new_dim);
	if (result == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("out of memory")));

	if (new_dim <= vec->dim)
	{
		for (i = 0; i < new_dim; i++)
			result->data[i] = vec->data[i];
	}
	else
	{
		for (i = 0; i < vec->dim; i++)
			result->data[i] = vec->data[i];
		for (i = vec->dim; i < new_dim; i++)
			result->data[i] = 0.0f;
	}

	PG_RETURN_VECTOR_P(result);
}

PG_FUNCTION_INFO_V1(vector_to_halfvec);
Datum
vector_to_halfvec(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	VectorF16 *result = NULL;
	int			size;
	int			i;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	if (v == NULL)
		PG_RETURN_NULL();

	if (v->dim > 4000)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("halfvec dimension %d exceeds maximum of 4000",
						v->dim)));

	size = offsetof(VectorF16, data) + sizeof(uint16) * v->dim;

	/*
	 * Variable-length type requires custom size - use palloc0 but ensure
	 * proper cleanup
	 */
	/* Note: For variable-length PostgreSQL types, palloc0 is standard */
	result = (VectorF16 *) palloc0(size);
	SET_VARSIZE(result, size);
	result->dim = v->dim;

	for (i = 0; i < v->dim; i++)
		result->data[i] = float_to_fp16_local(v->data[i]);

	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(halfvec_to_vector);
Datum
halfvec_to_vector(PG_FUNCTION_ARGS)
{
	VectorF16  *vf16 = (VectorF16 *) PG_GETARG_POINTER(0);
	Vector *result = NULL;
	int			i;

	if (vf16 == NULL)
		PG_RETURN_NULL();

	result = new_vector(vf16->dim);

	for (i = 0; i < vf16->dim; i++)
		result->data[i] = fp16_to_float(vf16->data[i]);

	PG_RETURN_VECTOR_P(result);
}

PG_FUNCTION_INFO_V1(vector_to_sparsevec);
Datum
vector_to_sparsevec(PG_FUNCTION_ARGS)
{
	Vector *v = NULL;
	VectorMap *result = NULL;
	int32 *indices = NULL;
	float4 *values = NULL;
	int32		nnz = 0;
	int			capacity;
	int			i;
	int			size;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	if (v == NULL)
		PG_RETURN_NULL();

	capacity = 16;
	indices = NULL;
	values = NULL;
	nalloc(indices, int32, capacity);
	nalloc(values, float4, capacity);

	for (i = 0; i < v->dim; i++)
	{
		if (fabs(v->data[i]) > 1e-10f)
		{
			if (nnz >= capacity)
			{
				capacity *= 2;
				indices = (int32 *) repalloc(indices, sizeof(int32) * capacity);
				values = (float4 *) repalloc(values, sizeof(float4) * capacity);
			}
			indices[nnz] = i;
			values[nnz] = v->data[i];
			nnz++;
		}
	}

	if (nnz > 1000)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("sparsevec cannot have more than 1000 nonzero entries")));

	size = sizeof(VectorMap) + sizeof(int32) * nnz + sizeof(float4) * nnz;
	result = (VectorMap *) palloc0(size);
	SET_VARSIZE(result, size);
	result->total_dim = v->dim;
	result->nnz = nnz;

	/* Suppress nonnull warning - result is guaranteed non-null after palloc0 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnonnull"
	memcpy(VECMAP_INDICES(result), indices, sizeof(int32) * nnz);
	memcpy(VECMAP_VALUES(result), values, sizeof(float4) * nnz);
#pragma GCC diagnostic pop

	pfree(indices);
	pfree(values);

	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(sparsevec_to_vector);
Datum
sparsevec_to_vector(PG_FUNCTION_ARGS)
{
	VectorMap  *sv = (VectorMap *) PG_GETARG_POINTER(0);
	Vector *result = NULL;
	int32 *indices = NULL;
	float4 *values = NULL;
	int			i;

	if (sv == NULL)
		PG_RETURN_NULL();

	result = new_vector(sv->total_dim);
	memset(result->data, 0, sizeof(float4) * sv->total_dim);

	indices = VECMAP_INDICES(sv);
	values = VECMAP_VALUES(sv);

	for (i = 0; i < sv->nnz; i++)
	{
		if (indices[i] < 0 || indices[i] >= result->dim)
		{
			ereport(ERROR,
					(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
					 errmsg("neurondb: array index %d out of bounds (dim=%d)",
							indices[i], result->dim)));
		}
		result->data[indices[i]] = values[i];
	}

	PG_RETURN_VECTOR_P(result);
}
