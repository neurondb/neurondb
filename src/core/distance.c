/*-------------------------------------------------------------------------
 *
 * distance.c
 *		Distance and similarity metric functions for vectors
 *
 * This file implements comprehensive distance metrics including L2
 * (Euclidean), cosine, inner product, L1 (Manhattan), Hamming,
 * Chebyshev, and Minkowski distances. All functions use Kahan
 * summation for numerical stability where appropriate.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/core/distance.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include <math.h>
#include <float.h>
#include "neurondb_safe_memory.h"
#include "neurondb_validation.h"

static inline void
check_dimensions(const Vector *a, const Vector *b)
{
	if (a->dim != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector dimensions must match: %d vs %d",
						a->dim,
						b->dim)));
}

/* L2 (Euclidean) distance, uses Kahan summation for numerical stability */
float4
l2_distance(Vector *a, Vector *b)
{
	double		c = 0.0;
	double		diff;
	double		sum = 0.0;
	double		t;
	double		y;
	int			i;

	check_dimensions(a, b);

	/*
	 * Kahan summation algorithm for numerical stability. When summing
	 * many floating point values, rounding errors accumulate. Kahan
	 * summation tracks a correction term 'c' that captures the lost
	 * precision from each addition. The correction is computed as the
	 * difference between the exact sum and the rounded sum, then added
	 * back in the next iteration. This reduces accumulated error from
	 * O(n) to O(1) for n additions.
	 */
	for (i = 0; i < a->dim; i++)
	{
		diff = (double) a->data[i] - (double) b->data[i];
		y = (diff * diff) - c;
		t = sum + y;

		c = (t - sum) - y;
		sum = t;
	}

	return (float4) sqrt(sum);
}

/* L2 squared distance (without sqrt) - used for index optimization */
float4
l2_squared_distance(Vector *a, Vector *b)
{
	double		c = 0.0;
	double		diff;
	double		sum = 0.0;
	double		t;
	double		y;
	int			i;

	check_dimensions(a, b);

	/* Kahan summation for numerical stability */
	for (i = 0; i < a->dim; i++)
	{
		diff = (double) a->data[i] - (double) b->data[i];
		y = (diff * diff) - c;
		t = sum + y;

		c = (t - sum) - y;
		sum = t;
	}

	return (float4) sum;
}

PG_FUNCTION_INFO_V1(vector_l2_distance);
Datum
vector_l2_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a = NULL;
	Vector	   *b = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_l2_distance requires 2 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);

	NDB_CHECK_VECTOR_VALID(b);
	PG_RETURN_FLOAT4(l2_distance(a, b));
}

/* vector_l2_squared_distance: Optimized L2 distance without sqrt for index optimization */
PG_FUNCTION_INFO_V1(vector_l2_squared_distance);
Datum
vector_l2_squared_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a = NULL;
	Vector	   *b = NULL;

	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_l2_squared_distance requires 2 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);

	PG_RETURN_FLOAT8((double) l2_squared_distance(a, b));
}

float4
inner_product_distance(Vector *a, Vector *b)
{
	double		sum = 0.0;
	int			i;

	check_dimensions(a, b);

	for (i = 0; i < a->dim; i++)
		sum += (double) a->data[i] * (double) b->data[i];

	return (float4) (-sum);
}

PG_FUNCTION_INFO_V1(vector_inner_product);
Datum
vector_inner_product(PG_FUNCTION_ARGS)
{
	Vector	   *a = NULL;
	Vector	   *b = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_inner_product requires 2 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);

	NDB_CHECK_VECTOR_VALID(b);
	PG_RETURN_FLOAT4(inner_product_distance(a, b));
}

/* vector_negative_inner_product: Negative inner product for index optimization */
PG_FUNCTION_INFO_V1(vector_negative_inner_product);
Datum
vector_negative_inner_product(PG_FUNCTION_ARGS)
{
	Vector	   *a = NULL;
	Vector	   *b = NULL;

	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_negative_inner_product requires 2 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);

	/* inner_product_distance already returns negative inner product */
	PG_RETURN_FLOAT8((double) inner_product_distance(a, b));
}

/* Cosine distance: 1.0 - (dot(a,b) / (||a||*||b||)), returns 1.0 if norm is zero */
float4
cosine_distance(Vector *a, Vector *b)
{
	double		dot = 0.0;
	double		norm_a = 0.0;
	double		norm_b = 0.0;
	double		va;
	double		vb;
	int			i;

	check_dimensions(a, b);

	/* Compute dot(a, b), ||a||^2, ||b||^2 */
	for (i = 0; i < a->dim; i++)
	{
		va = (double) a->data[i];
		vb = (double) b->data[i];

		dot += va * vb;
		norm_a += va * va;
		norm_b += vb * vb;
	}

	/* Handle zero norms to prevent divide by zero */
	if (norm_a == 0.0 || norm_b == 0.0)
		return 1.0;

	return (float4) (1.0 - (dot / (sqrt(norm_a) * sqrt(norm_b))));
}

PG_FUNCTION_INFO_V1(vector_cosine_distance);
Datum
vector_cosine_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a = NULL;
	Vector	   *b = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_cosine_distance requires 2 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);

	NDB_CHECK_VECTOR_VALID(b);
	PG_RETURN_FLOAT4(cosine_distance(a, b));
}

/* L1 (Manhattan) distance, standard implementation */
float4
l1_distance(Vector *a, Vector *b)
{
	double		sum = 0.0;
	int			i;

	check_dimensions(a, b);

	for (i = 0; i < a->dim; i++)
		sum += fabs((double) a->data[i] - (double) b->data[i]);

	return (float4) sum;
}

PG_FUNCTION_INFO_V1(vector_l1_distance);
Datum
vector_l1_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a = NULL;
	Vector	   *b = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_l1_distance requires 2 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);

	NDB_CHECK_VECTOR_VALID(b);
	PG_RETURN_FLOAT4(l1_distance(a, b));
}

/* Hamming distance: counts differing coordinates */
PG_FUNCTION_INFO_V1(vector_hamming_distance);
Datum
vector_hamming_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a = NULL;
	Vector	   *b = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_hamming_distance requires 2 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);

	NDB_CHECK_VECTOR_VALID(b);
	int			count = 0,
				i;

	check_dimensions(a, b);

	for (i = 0; i < a->dim; i++)
	{
		/*
		 * Using "!=" direct numerical for float equality; optionally consider
		 * tolerances
		 */
		if (a->data[i] != b->data[i])
			count++;
	}

	PG_RETURN_INT32(count);
}

/* Chebyshev distance: maximum coordinate-wise difference */
PG_FUNCTION_INFO_V1(vector_chebyshev_distance);
Datum
vector_chebyshev_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a = NULL;
	Vector	   *b = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_chebyshev_distance requires 2 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);

	NDB_CHECK_VECTOR_VALID(b);
	double		max_diff = 0.0;
	int			i;

	check_dimensions(a, b);

	for (i = 0; i < a->dim; i++)
	{
		double		diff = fabs((double) a->data[i] - (double) b->data[i]);

		if (diff > max_diff)
			max_diff = diff;
	}

	PG_RETURN_FLOAT8(max_diff);
}

/* Minkowski distance, p-norm (generalized distance): sum(|a_i-b_i|^p)^(1/p) */
PG_FUNCTION_INFO_V1(vector_minkowski_distance);
Datum
vector_minkowski_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a = NULL;
	Vector	   *b = NULL;
	float8		p;

	/* Validate argument count */
	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_minkowski_distance requires 3 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);
	p = PG_GETARG_FLOAT8(2);
	double		sum = 0.0;
	int			i;

	check_dimensions(a, b);

	if (p <= 0 || isnan(p) || isinf(p))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("p must be positive and finite")));

	if (p == 1.0)
	{
		/* Shortcut: L1 distance (Manhattan) */
		for (i = 0; i < a->dim; i++)
			sum += fabs((double) a->data[i] - (double) b->data[i]);
		PG_RETURN_FLOAT8(sum);
	}
	else if (p == 2.0)
	{
		/* Shortcut: L2 distance */
		double		partial = 0.0,
					c = 0.0;

		for (i = 0; i < a->dim; i++)
		{
			double		diff = (double) a->data[i] - (double) b->data[i];
			double		y = (diff * diff) - c;
			double		t = partial + y;

			c = (t - partial) - y;
			partial = t;
		}
		PG_RETURN_FLOAT8(sqrt(partial));
	}
	else if (p == INFINITY || p > 1e10) /* Large p treated as Chebyshev */
	{
		double		max_diff = 0.0;

		for (i = 0; i < a->dim; i++)
		{
			double		this_diff =
				fabs((double) a->data[i] - (double) b->data[i]);

			if (this_diff > max_diff)
				max_diff = this_diff;
		}
		PG_RETURN_FLOAT8(max_diff);
	}
	else
	{
		/* General Minkowski sum */
		for (i = 0; i < a->dim; i++)
		{
			double		diff =
				fabs((double) a->data[i] - (double) b->data[i]);

			sum += pow(diff, p);
		}
		PG_RETURN_FLOAT8(pow(sum, 1.0 / p));
	}
}
