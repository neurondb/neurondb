/*-------------------------------------------------------------------------
 *
 * vector_advanced.c
 *		Advanced vector operations beyond basic arithmetic
 *
 * Implements linear algebra operations, statistics, transformations,
 * and filtering operations for comprehensive vector manipulation.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/vector/vector_advanced.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include <math.h>
#include <float.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_macros.h"

PG_FUNCTION_INFO_V1(vector_cross_product);
Datum
vector_cross_product(PG_FUNCTION_ARGS)
{
	Vector *a = NULL;
	Vector *b = NULL;
	Vector *result = NULL;

	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_cross_product requires 2 arguments, got %d",
						PG_NARGS())));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);

	if (a == NULL || b == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("vectors must not be NULL")));

	if (a->dim <= 0 || a->dim > VECTOR_MAX_DIM ||
		b->dim <= 0 || b->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension")));

	if (a->dim != 3 || b->dim != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("cross product requires 3D vectors")));

	result = new_vector(3);
	if (result == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("out of memory")));

	result->data[0] = a->data[1] * b->data[2] - a->data[2] * b->data[1];
	result->data[1] = a->data[2] * b->data[0] - a->data[0] * b->data[2];
	result->data[2] = a->data[0] * b->data[1] - a->data[1] * b->data[0];

	PG_RETURN_VECTOR_P(result);
}

PG_FUNCTION_INFO_V1(vector_percentile);
Datum
vector_percentile(PG_FUNCTION_ARGS)
{
	Vector *vec = NULL;
	float8		p;
	float4 *sorted = NULL;
	float4		result;
	int			i;
	int			j;
	int			idx;

	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_percentile requires 2 arguments, got %d",
						PG_NARGS())));

	vec = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(vec);
	p = PG_GETARG_FLOAT8(1);

	if (vec == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("vector must not be NULL")));

	if (vec->dim <= 0 || vec->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension: %d",
						vec->dim)));

	if (p < 0.0 || p > 1.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("percentile must be between 0 and 1")));

	/* Copy and sort */
	sorted = NULL;
	nalloc(sorted, float4, vec->dim);
	if (sorted == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("out of memory")));
	for (i = 0; i < vec->dim; i++)
		sorted[i] = vec->data[i];

	for (i = 0; i < vec->dim - 1; i++)
	{
		for (j = 0; j < vec->dim - i - 1; j++)
		{
			if (sorted[j] > sorted[j + 1])
			{
				float4		tmp = sorted[j];

				sorted[j] = sorted[j + 1];
				sorted[j + 1] = tmp;
			}
		}
	}

	idx = (int) (p * (vec->dim - 1));
	if (idx < 0)
		idx = 0;
	if (idx >= vec->dim)
		idx = vec->dim - 1;
	result = sorted[idx];
	pfree(sorted);

	PG_RETURN_FLOAT4(result);
}

PG_FUNCTION_INFO_V1(vector_median);
Datum
vector_median(PG_FUNCTION_ARGS)
{
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_median requires 1 argument, got %d",
						PG_NARGS())));

	return DirectFunctionCall2(vector_percentile,
							   PG_GETARG_DATUM(0),
							   Float8GetDatum(0.5));
}

/*
 * vector_quantile
 *
 * Compute multiple quantiles of vector elements.
 */
PG_FUNCTION_INFO_V1(vector_quantile);
Datum
vector_quantile(PG_FUNCTION_ARGS)
{
	Vector *vec = NULL;
	ArrayType *quantiles_arr = NULL;
	ArrayType *result = NULL;
	Datum *elems = NULL;
	bool *nulls = NULL;
	float8 *quantiles = NULL;
	int			n_quantiles;
	int			i;

	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_quantile requires 2 arguments, got %d",
						PG_NARGS())));

	vec = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(vec);
	quantiles_arr = PG_GETARG_ARRAYTYPE_P(1);

	if (vec == NULL || quantiles_arr == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("vector and quantiles array must not be NULL")));

	if (vec->dim <= 0 || vec->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension: %d",
						vec->dim)));

	if (ARR_NDIM(quantiles_arr) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("quantiles array must be one-dimensional")));

	n_quantiles = ARR_DIMS(quantiles_arr)[0];
	if (n_quantiles <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("quantiles array must not be empty")));

	if (ARR_HASNULL(quantiles_arr))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("quantiles array must not contain NULL values")));

	quantiles = (float8 *) ARR_DATA_PTR(quantiles_arr);
	if (quantiles == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid quantiles array data")));

	elems = NULL;
	nulls = NULL;
	nalloc(elems, Datum, n_quantiles);
	nalloc(nulls, bool, n_quantiles);
	if (elems == NULL || nulls == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("out of memory")));

	for (i = 0; i < n_quantiles; i++)
	{
		if (quantiles[i] < 0.0 || quantiles[i] > 1.0)
		{
			nulls[i] = true;
			elems[i] = (Datum) 0;
			continue;
		}

		elems[i] = Float4GetDatum(DatumGetFloat4(DirectFunctionCall2(
																	 vector_percentile,
																	 PointerGetDatum(vec),
																	 Float8GetDatum(quantiles[i]))));
		nulls[i] = false;
	}

	result = construct_array(elems, n_quantiles, FLOAT4OID, sizeof(float4), true, 'i');
	pfree(elems);
	pfree(nulls);

	PG_RETURN_ARRAYTYPE_P(result);
}

PG_FUNCTION_INFO_V1(vector_scale);
Datum
vector_scale(PG_FUNCTION_ARGS)
{
	Vector *vec = NULL;
	ArrayType *scale_arr = NULL;
	Vector *result = NULL;
	float4 *scales = NULL;
	int			i;

	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_scale requires 2 arguments, got %d",
						PG_NARGS())));

	vec = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(vec);
	scale_arr = PG_GETARG_ARRAYTYPE_P(1);

	if (vec == NULL || scale_arr == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("vector and scale array must not be NULL")));

	if (vec->dim <= 0 || vec->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension: %d",
						vec->dim)));

	if (ARR_NDIM(scale_arr) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("scale array must be one-dimensional")));

	if (ARR_DIMS(scale_arr)[0] != vec->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("scale array dimension must match vector dimension")));

	if (ARR_HASNULL(scale_arr))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("scale array must not contain NULL values")));

	scales = (float4 *) ARR_DATA_PTR(scale_arr);
	if (scales == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid scale array data")));

	result = new_vector(vec->dim);
	if (result == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("out of memory")));

	for (i = 0; i < vec->dim; i++)
		result->data[i] = vec->data[i] * scales[i];

	PG_RETURN_VECTOR_P(result);
}

PG_FUNCTION_INFO_V1(vector_translate);
Datum
vector_translate(PG_FUNCTION_ARGS)
{
	Vector *vec = NULL;
	Vector *offset = NULL;
	Vector *result = NULL;
	int			i;

	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_translate requires 2 arguments, got %d",
						PG_NARGS())));

	vec = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(vec);
	offset = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(offset);

	if (vec == NULL || offset == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("vectors must not be NULL")));

	if (vec->dim <= 0 || vec->dim > VECTOR_MAX_DIM ||
		offset->dim <= 0 || offset->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension")));

	if (vec->dim != offset->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector dimensions must match")));

	result = new_vector(vec->dim);
	if (result == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("out of memory")));
	for (i = 0; i < vec->dim; i++)
		result->data[i] = vec->data[i] + offset->data[i];

	PG_RETURN_VECTOR_P(result);
}

PG_FUNCTION_INFO_V1(vector_filter);
Datum
vector_filter(PG_FUNCTION_ARGS)
{
	Vector *vec = NULL;
	ArrayType *mask_arr = NULL;
	Vector *result = NULL;
	bool *mask = NULL;
	int			dim;
	int			i;
	int			j;

	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_filter requires 2 arguments, got %d",
						PG_NARGS())));

	vec = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(vec);
	mask_arr = PG_GETARG_ARRAYTYPE_P(1);

	if (vec == NULL || mask_arr == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("vector and mask array must not be NULL")));

	if (vec->dim <= 0 || vec->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension: %d",
						vec->dim)));

	if (ARR_NDIM(mask_arr) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("mask array must be one-dimensional")));

	if (ARR_DIMS(mask_arr)[0] != vec->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("mask array dimension must match vector dimension")));

	if (ARR_HASNULL(mask_arr))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("mask array must not contain NULL values")));

	mask = (bool *) ARR_DATA_PTR(mask_arr);

	/* Count true values */
	dim = 0;
	for (i = 0; i < vec->dim; i++)
		if (mask[i])
			dim++;

	if (dim == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("filter mask must have at least one true value")));

	result = new_vector(dim);
	j = 0;
	for (i = 0; i < vec->dim; i++)
	{
		if (mask[i])
		{
			result->data[j] = vec->data[i];
			j++;
		}
	}

	PG_RETURN_VECTOR_P(result);
}

PG_FUNCTION_INFO_V1(vector_where);
Datum
vector_where(PG_FUNCTION_ARGS)
{
	Vector *condition = NULL;
	Vector *value = NULL;
	float4		else_value;
	Vector *result = NULL;
	int			i;

	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_where requires 3 arguments, got %d",
						PG_NARGS())));

	condition = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(condition);
	value = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(value);
	else_value = PG_GETARG_FLOAT4(2);

	if (condition == NULL || value == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("vectors must not be NULL")));

	if (condition->dim <= 0 || condition->dim > VECTOR_MAX_DIM ||
		value->dim <= 0 || value->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension")));

	if (condition->dim != value->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("condition and value vectors must have same dimension")));

	result = new_vector(condition->dim);
	if (result == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("out of memory")));
	for (i = 0; i < condition->dim; i++)
		result->data[i] = (condition->data[i] != 0.0f) ? value->data[i] : else_value;

	PG_RETURN_VECTOR_P(result);
}

/*
 * ============================================================================
 * Advanced Geometric Transformations
 * ============================================================================
 */

/*
 * Project vector onto another vector
 */
PG_FUNCTION_INFO_V1(vector_project);
Datum
vector_project(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	Vector	   *onto = NULL;
	Vector	   *result = NULL;
	float4		dot_product = 0.0;
	float4		onto_norm_sq = 0.0;
	int			i;

	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_project requires 2 arguments")));

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);
	onto = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(onto);

	if (v->dim != onto->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector dimensions must match")));

	/* Compute dot product and norm squared */
	for (i = 0; i < v->dim; i++)
	{
		dot_product += v->data[i] * onto->data[i];
		onto_norm_sq += onto->data[i] * onto->data[i];
	}

	if (onto_norm_sq == 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("cannot project onto zero vector")));

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
		result->data[i] = (dot_product / onto_norm_sq) * onto->data[i];

	PG_RETURN_VECTOR_P(result);
}

/*
 * Reject component (orthogonal projection)
 */
PG_FUNCTION_INFO_V1(vector_reject);
Datum
vector_reject(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	Vector	   *from = NULL;
	Vector	   *projected = NULL;
	Vector	   *result = NULL;
	int			i;

	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_reject requires 2 arguments")));

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);
	from = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(from);

	/* Compute projection first */
	projected = DatumGetVector(DirectFunctionCall2(vector_project,
													PointerGetDatum(v),
													PointerGetDatum(from)));

	/* Reject = v - projection */
	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
		result->data[i] = v->data[i] - projected->data[i];

	PG_RETURN_VECTOR_P(result);
}

/*
 * Reflect vector across a plane (defined by normal vector)
 */
PG_FUNCTION_INFO_V1(vector_reflect);
Datum
vector_reflect(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	Vector	   *normal = NULL;
	Vector	   *result = NULL;
	float4		dot_product = 0.0;
	float4		normal_norm_sq = 0.0;
	int			i;

	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_reflect requires 2 arguments")));

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);
	normal = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(normal);

	if (v->dim != normal->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector dimensions must match")));

	/* Compute dot product and norm squared */
	for (i = 0; i < v->dim; i++)
	{
		dot_product += v->data[i] * normal->data[i];
		normal_norm_sq += normal->data[i] * normal->data[i];
	}

	if (normal_norm_sq == 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("cannot reflect across zero vector")));

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
		result->data[i] = v->data[i] - 2.0 * (dot_product / normal_norm_sq) * normal->data[i];

	PG_RETURN_VECTOR_P(result);
}

/*
 * Rotate 2D vector by angle (in radians)
 * For higher dimensions, rotates in the plane defined by first two dimensions
 */
PG_FUNCTION_INFO_V1(vector_rotate);
Datum
vector_rotate(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	float8		angle;
	Vector	   *result = NULL;
	double		cos_a, sin_a;
	int			i;

	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_rotate requires 2 arguments")));

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);
	angle = PG_GETARG_FLOAT8(1);

	if (v->dim < 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_rotate requires at least 2 dimensions")));

	cos_a = cos(angle);
	sin_a = sin(angle);

	result = new_vector(v->dim);
	if (result == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("out of memory")));

	/* Rotate first two dimensions */
	result->data[0] = v->data[0] * cos_a - v->data[1] * sin_a;
	result->data[1] = v->data[0] * sin_a + v->data[1] * cos_a;

	/* Copy remaining dimensions unchanged */
	for (i = 2; i < v->dim; i++)
		result->data[i] = v->data[i];

	PG_RETURN_VECTOR_P(result);
}
