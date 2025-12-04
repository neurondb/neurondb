/*-------------------------------------------------------------------------
 *
 * vector_ops.c
 *		Extended vector operations and transformations
 *
 * This file implements advanced vector manipulation functions including
 * element access, slicing, concatenation, element-wise operations
 * (absolute value, square, square root, power), Hadamard product,
 * statistical functions (mean, variance, stddev, min, max), vector
 * comparison, clipping, standardization, and normalization methods.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/vector/vector_ops.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "utils/array.h"
#include "catalog/pg_type.h"
#include "neurondb_safe_memory.h"
#include "neurondb_validation.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"
#include <math.h>
#include <stdint.h>

/*
 * Vector element access
 */
PG_FUNCTION_INFO_V1(vector_get);
Datum
vector_get(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	int32		idx;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);
	idx = PG_GETARG_INT32(1);

	if (idx < 0 || idx >= v->dim)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("index %d out of bounds for vector of "
						"dimension %d",
						idx,
						v->dim)));

	PG_RETURN_FLOAT4(v->data[idx]);
}

/*
 * Vector element update
 */
PG_FUNCTION_INFO_V1(vector_set);
Datum
vector_set(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	int32		idx;
	float4		val;
	Vector	   *result;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);
	idx = PG_GETARG_INT32(1);
	val = PG_GETARG_FLOAT4(2);

	if (idx < 0 || idx >= v->dim)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("index %d out of bounds", idx)));

	result = copy_vector(v);
	result->data[idx] = val;

	PG_RETURN_VECTOR_P(result);
}

/*
 * Vector slice (extract subvector)
 */
PG_FUNCTION_INFO_V1(vector_slice);
Datum
vector_slice(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	int32		start;
	int32		end;
	Vector	   *result;
	int			new_dim;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);
	start = PG_GETARG_INT32(1);
	end = PG_GETARG_INT32(2);

	if (start < 0 || start >= v->dim || end < start || end > v->dim)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("invalid slice bounds")));

	new_dim = end - start;
	result = new_vector(new_dim);
	memcpy(result->data, v->data + start, sizeof(float4) * new_dim);

	PG_RETURN_VECTOR_P(result);
}

/*
 * Vector append
 */
PG_FUNCTION_INFO_V1(vector_append);
Datum
vector_append(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	float4		val;

	Vector	   *result;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);
	val = PG_GETARG_FLOAT4(1);

	result = new_vector(v->dim + 1);
	if (v->dim > 0)
		memcpy(result->data, v->data, sizeof(float4) * v->dim);
	result->data[v->dim] = val;

	PG_RETURN_VECTOR_P(result);
}

/*
 * Vector prepend
 */
PG_FUNCTION_INFO_V1(vector_prepend);
Datum
vector_prepend(PG_FUNCTION_ARGS)
{
	float4		val;
	Vector	   *v;
	Vector	   *result;

	val = PG_GETARG_FLOAT4(0);
	v = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(v);

	result = new_vector(v->dim + 1);
	result->data[0] = val;
	memcpy(result->data + 1, v->data, sizeof(float4) * v->dim);

	PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise absolute value
 */
PG_FUNCTION_INFO_V1(vector_abs);
Datum
vector_abs(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	Vector	   *result;

	int			i;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
		result->data[i] = fabs(v->data[i]);

	PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise square
 */
PG_FUNCTION_INFO_V1(vector_square);
Datum
vector_square(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	Vector	   *result;

	int			i;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
		result->data[i] = v->data[i] * v->data[i];

	PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise square root
 */
PG_FUNCTION_INFO_V1(vector_sqrt);
Datum
vector_sqrt(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	Vector	   *result;

	int			i;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
	{
		if (v->data[i] < 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("cannot take square root of "
							"negative number")));
		result->data[i] = sqrt(v->data[i]);
	}

	PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise power
 */
PG_FUNCTION_INFO_V1(vector_pow);
Datum
vector_pow(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	float8		exp;

	Vector	   *result;
	int			i;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);
	exp = PG_GETARG_FLOAT8(1);

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
		result->data[i] = pow(v->data[i], exp);

	PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise multiplication (Hadamard product)
 */
PG_FUNCTION_INFO_V1(vector_hadamard);
Datum
vector_hadamard(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;
	Vector	   *result;
	int			i;

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);

	if (a->dim != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector dimensions must match")));

	result = new_vector(a->dim);
	for (i = 0; i < a->dim; i++)
		result->data[i] = a->data[i] * b->data[i];

	PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise division
 */
PG_FUNCTION_INFO_V1(vector_divide);
Datum
vector_divide(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;
	Vector	   *result;
	int			i;

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);

	if (a->dim != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector dimensions must match")));

	result = new_vector(a->dim);
	for (i = 0; i < a->dim; i++)
	{
		if (b->data[i] == 0.0)
			ereport(ERROR,
					(errcode(ERRCODE_DIVISION_BY_ZERO),
					 errmsg("division by zero")));
		result->data[i] = a->data[i] / b->data[i];
	}

	PG_RETURN_VECTOR_P(result);
}

/*
 * Vector statistics
 */
PG_FUNCTION_INFO_V1(vector_mean);
Datum
vector_mean(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	double		sum = 0.0;
	int			i;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot compute mean of NULL vector")));

	if (v->dim <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("cannot compute mean of vector with dimension %d",
						v->dim)));

	for (i = 0; i < v->dim; i++)
		sum += v->data[i];

	PG_RETURN_FLOAT8(sum / v->dim);
}

PG_FUNCTION_INFO_V1(vector_variance);
Datum
vector_variance(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	double		mean = 0.0;
	double		variance = 0.0;
	int			i;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot compute variance of NULL vector")));

	if (v->dim <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("cannot compute variance of vector with dimension %d",
						v->dim)));

	for (i = 0; i < v->dim; i++)
		mean += v->data[i];
	mean /= v->dim;

	for (i = 0; i < v->dim; i++)
	{
		double		diff = v->data[i] - mean;

		variance += diff * diff;
	}

	PG_RETURN_FLOAT8(variance / v->dim);
}

PG_FUNCTION_INFO_V1(vector_stddev);
Datum
vector_stddev(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	double		mean = 0.0;
	double		variance = 0.0;
	int			i;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot compute standard deviation of NULL vector")));

	if (v->dim <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("cannot compute standard deviation of vector with dimension %d",
						v->dim)));

	for (i = 0; i < v->dim; i++)
		mean += v->data[i];
	mean /= v->dim;

	for (i = 0; i < v->dim; i++)
	{
		double		diff = v->data[i] - mean;

		variance += diff * diff;
	}

	PG_RETURN_FLOAT8(sqrt(variance / v->dim));
}

PG_FUNCTION_INFO_V1(vector_min);
Datum
vector_min(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	float4		min_val;
	int			i;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot find minimum of NULL vector")));

	if (v->dim <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("cannot find minimum of empty vector")));

	min_val = v->data[0];
	for (i = 1; i < v->dim; i++)
		if (v->data[i] < min_val)
			min_val = v->data[i];

	PG_RETURN_FLOAT4(min_val);
}

PG_FUNCTION_INFO_V1(vector_max);
Datum
vector_max(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	float4		max_val;
	int			i;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot find maximum of NULL vector")));

	if (v->dim <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("cannot find maximum of empty vector")));

	max_val = v->data[0];
	for (i = 1; i < v->dim; i++)
		if (v->data[i] > max_val)
			max_val = v->data[i];

	PG_RETURN_FLOAT4(max_val);
}

PG_FUNCTION_INFO_V1(vector_sum);
Datum
vector_sum(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	double		sum = 0.0;

	int			i;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	for (i = 0; i < v->dim; i++)
		sum += v->data[i];

	PG_RETURN_FLOAT8(sum);
}

/*
 * Vector comparison
 */
PG_FUNCTION_INFO_V1(vector_eq);
Datum
vector_eq(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;
	int			i;

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);

	/* Handle NULL vectors */
	if (a == NULL && b == NULL)
		PG_RETURN_BOOL(true);
	if (a == NULL || b == NULL)
		PG_RETURN_BOOL(false);

	if (a->dim != b->dim)
		PG_RETURN_BOOL(false);

	/* Use epsilon comparison for float equality */
	for (i = 0; i < a->dim; i++)
	{
		if (isnan(a->data[i]) || isnan(b->data[i]))
		{
			/* NaN != NaN */
			if (isnan(a->data[i]) != isnan(b->data[i]))
				PG_RETURN_BOOL(false);
		}
		else if (fabs(a->data[i] - b->data[i]) > 1e-6)
			PG_RETURN_BOOL(false);
	}

	PG_RETURN_BOOL(true);
}

PG_FUNCTION_INFO_V1(vector_ne);
Datum
vector_ne(PG_FUNCTION_ARGS)
{
	return DirectFunctionCall2(
							   vector_eq, PG_GETARG_DATUM(0), PG_GETARG_DATUM(1))
		? BoolGetDatum(false)
		: BoolGetDatum(true);
}

/*
 * vector_hash(vector) -> uint32
 * Hash function for vector type to support hash joins and hash-based operations
 */
PG_FUNCTION_INFO_V1(vector_hash);
Datum
vector_hash(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	uint32		hash = 5381;	/* DJB hash seed */
	int			i;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	if (v == NULL)
		PG_RETURN_UINT32(0);

	/* Hash dimension first */
	hash = ((hash << 5) + hash) + (uint32) v->dim;

	/* Hash vector data (use first 16 elements for performance) */
	for (i = 0; i < v->dim && i < 16; i++)
	{
		/* Convert float to int for hashing (multiply by large number) */
		int32		tmp = (int32) (v->data[i] * 1000000.0f);

		hash = ((hash << 5) + hash) + (uint32) tmp;
	}

	/* If vector is longer, hash remaining elements with stride */
	if (v->dim > 16)
	{
		int			stride = v->dim / 16;

		for (i = 16; i < v->dim; i += stride)
		{
			int32		tmp = (int32) (v->data[i] * 1000000.0f);

			hash = ((hash << 5) + hash) + (uint32) tmp;
		}
	}

	PG_RETURN_UINT32(hash);
}

/*
 * Vector clipping (element-wise min/max)
 */
PG_FUNCTION_INFO_V1(vector_clip);
Datum
vector_clip(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	float4		min_val;
	float4		max_val;
	Vector	   *result;
	int			i;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);
	min_val = PG_GETARG_FLOAT4(1);
	max_val = PG_GETARG_FLOAT4(2);

	if (min_val > max_val)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("min_val must be <= max_val")));

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
	{
		if (v->data[i] < min_val)
			result->data[i] = min_val;
		else if (v->data[i] > max_val)
			result->data[i] = max_val;
		else
			result->data[i] = v->data[i];
	}

	PG_RETURN_VECTOR_P(result);
}

/*
 * Vector standardization (z-score normalization)
 */
PG_FUNCTION_INFO_V1(vector_standardize);
Datum
vector_standardize(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	Vector	   *result;
	double		mean = 0.0;
	double		stddev = 0.0;
	int			i;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot standardize NULL vector")));

	if (v->dim <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("cannot standardize vector with dimension %d",
						v->dim)));

	/* Calculate mean */
	for (i = 0; i < v->dim; i++)
		mean += v->data[i];
	mean /= v->dim;

	/* Calculate standard deviation */
	for (i = 0; i < v->dim; i++)
	{
		double		diff = v->data[i] - mean;

		stddev += diff * diff;
	}
	stddev = sqrt(stddev / v->dim);

	/* Standardize */
	result = new_vector(v->dim);
	if (stddev > 0.0)
	{
		for (i = 0; i < v->dim; i++)
			result->data[i] = (v->data[i] - mean) / stddev;
	}
	else
	{
		/* All values are the same - set to zero */
		memset(result->data, 0, sizeof(float4) * v->dim);
	}

	PG_RETURN_VECTOR_P(result);
}

/*
 * Vector min-max normalization
 */
PG_FUNCTION_INFO_V1(vector_minmax_normalize);
Datum
vector_minmax_normalize(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	Vector	   *result;
	float4		min_val;
	float4		max_val;
	float4		range;
	int			i;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot normalize NULL vector")));

	if (v->dim <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("cannot normalize empty vector")));

	/* Find min and max */
	min_val = max_val = v->data[0];
	for (i = 1; i < v->dim; i++)
	{
		if (v->data[i] < min_val)
			min_val = v->data[i];
		if (v->data[i] > max_val)
			max_val = v->data[i];
	}

	range = max_val - min_val;
	result = new_vector(v->dim);

	if (range > 0.0)
	{
		for (i = 0; i < v->dim; i++)
			result->data[i] = (v->data[i] - min_val) / range;
	}
	else
	{
		/* All values are the same */
		for (i = 0; i < v->dim; i++)
			result->data[i] = 0.5;	/* Middle of [0,1] range */
	}

	PG_RETURN_VECTOR_P(result);
}
