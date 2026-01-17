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
 * Copyright (c) 2024-2026, neurondb, Inc.
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

PG_FUNCTION_INFO_V1(vector_get);
Datum
vector_get(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	int32		idx;

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_get requires 2 arguments")));

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
	Vector	   *v = NULL;
	int32		idx;
	float4		val;
	Vector *result = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_set requires 3 arguments")));

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

PG_FUNCTION_INFO_V1(vector_slice);
Datum
vector_slice(PG_FUNCTION_ARGS)
{
	Vector *v = NULL;
	int32		start;
	int32		end;
	Vector *result = NULL;
	int			new_dim;

	/* Validate argument count */
	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_slice requires 3 arguments")));

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

PG_FUNCTION_INFO_V1(vector_append);
Datum
vector_append(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	float4		val;

	Vector *result = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_append requires 2 arguments")));

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);
	val = PG_GETARG_FLOAT4(1);

	result = new_vector(v->dim + 1);
	if (v->dim > 0)
		memcpy(result->data, v->data, sizeof(float4) * v->dim);
	result->data[v->dim] = val;

	PG_RETURN_VECTOR_P(result);
}

PG_FUNCTION_INFO_V1(vector_prepend);
Datum
vector_prepend(PG_FUNCTION_ARGS)
{
	float4		val;
	Vector	   *v = NULL;
	Vector *result = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_prepend requires 2 arguments")));

	val = PG_GETARG_FLOAT4(0);
	v = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(v);

	result = new_vector(v->dim + 1);
	result->data[0] = val;
	memcpy(result->data + 1, v->data, sizeof(float4) * v->dim);

	PG_RETURN_VECTOR_P(result);
}

PG_FUNCTION_INFO_V1(vector_abs);
Datum
vector_abs(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	Vector *result = NULL;

	int			i;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_abs requires 1 argument")));

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
		result->data[i] = fabs(v->data[i]);

	PG_RETURN_VECTOR_P(result);
}

PG_FUNCTION_INFO_V1(vector_square);
Datum
vector_square(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	Vector *result = NULL;

	int			i;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_square requires 1 argument")));

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
		result->data[i] = v->data[i] * v->data[i];

	PG_RETURN_VECTOR_P(result);
}

PG_FUNCTION_INFO_V1(vector_sqrt);
Datum
vector_sqrt(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	Vector *result = NULL;

	int			i;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_sqrt requires 1 argument")));

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

PG_FUNCTION_INFO_V1(vector_pow);
Datum
vector_pow(PG_FUNCTION_ARGS)
{
	Vector *v = NULL;
	float8		exp;

	Vector *result = NULL;
	int			i;

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_pow requires 2 arguments")));

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
	Vector	   *a = NULL;
	Vector *b = NULL;
	Vector *result = NULL;
	int			i;

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_hadamard requires 2 arguments")));

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

PG_FUNCTION_INFO_V1(vector_divide);
Datum
vector_divide(PG_FUNCTION_ARGS)
{
	Vector	   *a = NULL;
	Vector *b = NULL;
	Vector *result = NULL;
	int			i;

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_divide requires 2 arguments")));

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

PG_FUNCTION_INFO_V1(vector_mean);
Datum
vector_mean(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	double		sum = 0.0;
	int			i;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_mean requires 1 argument")));

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
	Vector	   *v = NULL;
	double		mean = 0.0;
	double		variance = 0.0;
	int			i;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_variance requires 1 argument")));

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
	Vector	   *v = NULL;
	double		mean = 0.0;
	double		variance = 0.0;
	int			i;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_stddev requires 1 argument")));

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

/*
 * Skewness of vector elements
 */
PG_FUNCTION_INFO_V1(vector_skewness);
Datum
vector_skewness(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	double		mean = 0.0;
	double		stddev = 0.0;
	double		skewness = 0.0;
	int			i;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_skewness requires 1 argument")));

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	if (v->dim <= 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("skewness requires at least 3 elements")));

	/* Compute mean */
	for (i = 0; i < v->dim; i++)
		mean += v->data[i];
	mean /= v->dim;

	/* Compute standard deviation */
	for (i = 0; i < v->dim; i++)
	{
		double		diff = v->data[i] - mean;
		stddev += diff * diff;
	}
	stddev = sqrt(stddev / v->dim);

	if (stddev == 0.0)
		PG_RETURN_FLOAT8(0.0);

	/* Compute skewness */
	for (i = 0; i < v->dim; i++)
	{
		double		diff = v->data[i] - mean;
		skewness += (diff / stddev) * (diff / stddev) * (diff / stddev);
	}
	skewness /= v->dim;

	PG_RETURN_FLOAT8(skewness);
}

/*
 * Kurtosis of vector elements
 */
PG_FUNCTION_INFO_V1(vector_kurtosis);
Datum
vector_kurtosis(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	double		mean = 0.0;
	double		stddev = 0.0;
	double		kurtosis = 0.0;
	int			i;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_kurtosis requires 1 argument")));

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	if (v->dim <= 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("kurtosis requires at least 3 elements")));

	/* Compute mean */
	for (i = 0; i < v->dim; i++)
		mean += v->data[i];
	mean /= v->dim;

	/* Compute standard deviation */
	for (i = 0; i < v->dim; i++)
	{
		double		diff = v->data[i] - mean;
		stddev += diff * diff;
	}
	stddev = sqrt(stddev / v->dim);

	if (stddev == 0.0)
		PG_RETURN_FLOAT8(0.0);

	/* Compute kurtosis (excess kurtosis, so subtract 3) */
	for (i = 0; i < v->dim; i++)
	{
		double		diff = v->data[i] - mean;
		double		normalized = diff / stddev;
		kurtosis += normalized * normalized * normalized * normalized;
	}
	kurtosis = (kurtosis / v->dim) - 3.0;

	PG_RETURN_FLOAT8(kurtosis);
}

/*
 * Shannon entropy of vector elements (treating as probability distribution)
 */
PG_FUNCTION_INFO_V1(vector_entropy);
Datum
vector_entropy(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	double		sum = 0.0;
	double		entropy = 0.0;
	int			i;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_entropy requires 1 argument")));

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	/* Normalize to probability distribution */
	for (i = 0; i < v->dim; i++)
	{
		if (v->data[i] < 0.0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("entropy requires non-negative values")));
		sum += v->data[i];
	}

	if (sum == 0.0)
		PG_RETURN_FLOAT8(0.0);

	/* Compute entropy */
	for (i = 0; i < v->dim; i++)
	{
		double		p = v->data[i] / sum;
		if (p > 0.0)
			entropy -= p * log(p);
	}

	PG_RETURN_FLOAT8(entropy);
}

PG_FUNCTION_INFO_V1(vector_min);
Datum
vector_min(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	float4		min_val;
	int			i;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_min requires 1 argument")));

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
	Vector	   *v = NULL;
	float4		max_val;
	int			i;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_max requires 1 argument")));

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

/*
 * Element-wise minimum of two vectors (overloaded version)
 */
PG_FUNCTION_INFO_V1(vector_min_2arg);
Datum
vector_min_2arg(PG_FUNCTION_ARGS)
{
	Vector	   *a = NULL;
	Vector	   *b = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_min_2arg requires 2 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);

	if (a->dim != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector dimensions must match: %d vs %d",
						a->dim,
						b->dim)));

	{
		Vector *result = NULL;
		int			i;

		result = new_vector(a->dim);
		for (i = 0; i < a->dim; i++)
			result->data[i] = (a->data[i] < b->data[i]) ? a->data[i] : b->data[i];

		PG_RETURN_VECTOR_P(result);
	}
}

/*
 * Element-wise maximum of two vectors (overloaded version)
 */
PG_FUNCTION_INFO_V1(vector_max_2arg);
Datum
vector_max_2arg(PG_FUNCTION_ARGS)
{
	Vector	   *a = NULL;
	Vector	   *b = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_max_2arg requires 2 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);

	if (a->dim != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector dimensions must match: %d vs %d",
						a->dim,
						b->dim)));

	{
		Vector *result = NULL;
		int			i;

		result = new_vector(a->dim);
		for (i = 0; i < a->dim; i++)
			result->data[i] = (a->data[i] > b->data[i]) ? a->data[i] : b->data[i];

		PG_RETURN_VECTOR_P(result);
	}
}

PG_FUNCTION_INFO_V1(vector_sum);
Datum
vector_sum(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	double		sum = 0.0;

	int			i;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_sum requires 1 argument")));

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	for (i = 0; i < v->dim; i++)
		sum += v->data[i];

	PG_RETURN_FLOAT8(sum);
}

PG_FUNCTION_INFO_V1(vector_eq);
Datum
vector_eq(PG_FUNCTION_ARGS)
{
	Vector	   *a = NULL;
	Vector	   *b = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_eq requires 2 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);

	/* Handle NULL vectors */
	if (a == NULL && b == NULL)
		PG_RETURN_BOOL(true);
	if (a == NULL || b == NULL)
		PG_RETURN_BOOL(false);

	/* Use lexicographic comparison supporting different dimensions */
	{
		int			dim = Min(a->dim, b->dim);
		int			j;

		/* Check values before dimensions to be consistent with Postgres arrays */
		for (j = 0; j < dim; j++)
		{
			if (isnan(a->data[j]) || isnan(b->data[j]))
			{
				/* NaN comparison: both must be NaN for equality */
				if (isnan(a->data[j]) != isnan(b->data[j]))
					PG_RETURN_BOOL(false);
			}
			else if (fabs(a->data[j] - b->data[j]) > 1e-6)
			{
				PG_RETURN_BOOL(false);
			}
		}

		/* If common elements are equal, compare by dimension */
		if (a->dim != b->dim)
			PG_RETURN_BOOL(false);
	}

	PG_RETURN_BOOL(true);
}

PG_FUNCTION_INFO_V1(vector_ne);
Datum
vector_ne(PG_FUNCTION_ARGS)
{
	/* Validate argument count */
	if (PG_NARGS() < 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_ne requires at least 2 arguments")));

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
	Vector	   *v = NULL;
	uint32		hash = 5381;	/* DJB hash seed */
	int			i;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_hash requires 1 argument")));

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	if (v == NULL)
		PG_RETURN_UINT32(0);

	hash = ((hash << 5) + hash) + (uint32) v->dim;

	for (i = 0; i < v->dim && i < 16; i++)
	{
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

PG_FUNCTION_INFO_V1(vector_clip);
Datum
vector_clip(PG_FUNCTION_ARGS)
{
	Vector *v = NULL;
	float4		min_val;
	float4		max_val;
	Vector *result = NULL;
	int			i;

	/* Validate argument count */
	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_clip requires 3 arguments")));

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
	Vector *v = NULL;
	Vector *result = NULL;
	double		mean = 0.0;
	double		stddev = 0.0;
	int			i;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_standardize requires 1 argument")));

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

	result = new_vector(v->dim);
	if (stddev > 0.0)
	{
		for (i = 0; i < v->dim; i++)
			result->data[i] = (v->data[i] - mean) / stddev;
		}
	else
		{
			memset(result->data, 0, sizeof(float4) * v->dim);
	}

	PG_RETURN_VECTOR_P(result);
}

PG_FUNCTION_INFO_V1(vector_minmax_normalize);
Datum
vector_minmax_normalize(PG_FUNCTION_ARGS)
{
	Vector *v = NULL;
	Vector *result = NULL;
	float4		min_val;
	float4		max_val;
	float4		range;
	int			i;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_minmax_normalize requires 1 argument")));

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
			for (i = 0; i < v->dim; i++)
				result->data[i] = 0.5;
	}

	PG_RETURN_VECTOR_P(result);
}

/*
 * ============================================================================
 * Advanced Mathematical Operations
 * ============================================================================
 */

/*
 * Element-wise exponential
 */
PG_FUNCTION_INFO_V1(vector_exp);
Datum
vector_exp(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	Vector	   *result = NULL;
	int			i;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_exp requires 1 argument")));

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
		result->data[i] = exp(v->data[i]);

	PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise natural logarithm
 */
PG_FUNCTION_INFO_V1(vector_log);
Datum
vector_log(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	Vector	   *result = NULL;
	int			i;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_log requires 1 argument")));

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
	{
		if (v->data[i] <= 0.0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("cannot take logarithm of non-positive number")));
		result->data[i] = log(v->data[i]);
	}

	PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise base-10 logarithm
 */
PG_FUNCTION_INFO_V1(vector_log10);
Datum
vector_log10(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	Vector	   *result = NULL;
	int			i;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_log10 requires 1 argument")));

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
	{
		if (v->data[i] <= 0.0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("cannot take logarithm of non-positive number")));
		result->data[i] = log10(v->data[i]);
	}

	PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise sine
 */
PG_FUNCTION_INFO_V1(vector_sin);
Datum
vector_sin(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	Vector	   *result = NULL;
	int			i;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_sin requires 1 argument")));

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
		result->data[i] = sin(v->data[i]);

	PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise cosine
 */
PG_FUNCTION_INFO_V1(vector_cos);
Datum
vector_cos(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	Vector	   *result = NULL;
	int			i;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_cos requires 1 argument")));

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
		result->data[i] = cos(v->data[i]);

	PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise tangent
 */
PG_FUNCTION_INFO_V1(vector_tan);
Datum
vector_tan(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	Vector	   *result = NULL;
	int			i;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_tan requires 1 argument")));

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
		result->data[i] = tan(v->data[i]);

	PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise arcsine
 */
PG_FUNCTION_INFO_V1(vector_asin);
Datum
vector_asin(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	Vector	   *result = NULL;
	int			i;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_asin requires 1 argument")));

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
	{
		if (v->data[i] < -1.0 || v->data[i] > 1.0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("arcsine argument must be in range [-1, 1]")));
		result->data[i] = asin(v->data[i]);
	}

	PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise arccosine
 */
PG_FUNCTION_INFO_V1(vector_acos);
Datum
vector_acos(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	Vector	   *result = NULL;
	int			i;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_acos requires 1 argument")));

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
	{
		if (v->data[i] < -1.0 || v->data[i] > 1.0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("arccosine argument must be in range [-1, 1]")));
		result->data[i] = acos(v->data[i]);
	}

	PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise arctangent
 */
PG_FUNCTION_INFO_V1(vector_atan);
Datum
vector_atan(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	Vector	   *result = NULL;
	int			i;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_atan requires 1 argument")));

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
		result->data[i] = atan(v->data[i]);

	PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise hyperbolic sine
 */
PG_FUNCTION_INFO_V1(vector_sinh);
Datum
vector_sinh(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	Vector	   *result = NULL;
	int			i;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_sinh requires 1 argument")));

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
		result->data[i] = sinh(v->data[i]);

	PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise hyperbolic cosine
 */
PG_FUNCTION_INFO_V1(vector_cosh);
Datum
vector_cosh(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	Vector	   *result = NULL;
	int			i;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_cosh requires 1 argument")));

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
		result->data[i] = cosh(v->data[i]);

	PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise hyperbolic tangent
 */
PG_FUNCTION_INFO_V1(vector_tanh);
Datum
vector_tanh(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	Vector	   *result = NULL;
	int			i;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_tanh requires 1 argument")));

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
		result->data[i] = tanh(v->data[i]);

	PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise error function
 */
PG_FUNCTION_INFO_V1(vector_erf);
Datum
vector_erf(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	Vector	   *result = NULL;
	int			i;
	double		x, t, y;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_erf requires 1 argument")));

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
	{
		x = v->data[i];
		/* Approximation of error function */
		t = 1.0 / (1.0 + 0.47047 * fabs(x));
		y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
		result->data[i] = (x >= 0.0) ? y : -y;
	}

	PG_RETURN_VECTOR_P(result);
}

/*
 * Element-wise complementary error function
 */
PG_FUNCTION_INFO_V1(vector_erfc);
Datum
vector_erfc(PG_FUNCTION_ARGS)
{
	Vector	   *v = NULL;
	Vector	   *result = NULL;
	int			i;
	double		x, t, y;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_erfc requires 1 argument")));

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
	{
		x = v->data[i];
		/* Approximation of complementary error function */
		t = 1.0 / (1.0 + 0.47047 * fabs(x));
		y = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
		result->data[i] = (x >= 0.0) ? y : 2.0 - y;
	}

	PG_RETURN_VECTOR_P(result);
}
