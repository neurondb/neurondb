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
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/vector/vector_distance.c
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
#include "neurondb_macros.h"
#include "neurondb_spi.h"

/* SIMD-optimized distance functions */
extern float4 l2_distance_simd(Vector *a, Vector *b);
extern float4 inner_product_simd(Vector *a, Vector *b);
extern float4 cosine_distance_simd(Vector *a, Vector *b);
extern float4 l1_distance_simd(Vector *a, Vector *b);

static inline void
check_dimensions(const Vector *a, const Vector *b)
{
	if (a == NULL || b == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot compute distance with NULL vectors")));

	if (a->dim != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector dimensions must match: %d vs %d",
						a->dim,
						b->dim)));

	if (a->dim <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("cannot compute distance for vector with dimension %d",
						a->dim)));

	/* Check for NaN/Inf in vectors */
	{
		int			i;

		for (i = 0; i < a->dim; i++)
		{
			if (isnan(a->data[i]) || isinf(a->data[i]))
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("vector contains NaN or Infinity at index %d", i)));
		}
		for (i = 0; i < b->dim; i++)
		{
			if (isnan(b->data[i]) || isinf(b->data[i]))
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("vector contains NaN or Infinity at index %d", i)));
		}
	}
}

/* L2 (Euclidean) distance, uses Kahan summation for numerical stability */
float4
l2_distance(Vector *a, Vector *b)
{
	double		sum = 0.0,
				c = 0.0;
	int			i;
	float4		result;

	check_dimensions(a, b);

	for (i = 0; i < a->dim; i++)
	{
		double		diff = (double) a->data[i] - (double) b->data[i];
		double		y = (diff * diff) - c;
		double		t = sum + y;

		c = (t - sum) - y;
		sum = t;
	}

	result = (float4) sqrt(sum);

	/* Check for overflow/underflow */
	if (isnan(result) || isinf(result))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("L2 distance calculation resulted in NaN or Infinity")));

	return result;
}

PG_FUNCTION_INFO_V1(vector_l2_distance);
Datum
vector_l2_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);
	/* Use SIMD-optimized version if available */
	PG_RETURN_FLOAT4(l2_distance_simd(a, b));
}

/* Inner product distance, negative for correct ordering for similarity */
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
	Vector	   *a;
	Vector	   *b;

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);
	/* Use SIMD-optimized version if available */
	PG_RETURN_FLOAT4(inner_product_simd(a, b));
}

/* Cosine distance: 1.0 - (dot(a,b) / (||a||*||b||)), returns 1.0 if norm is zero */
float4
cosine_distance(Vector *a, Vector *b)
{
	double		dot = 0.0,
				norm_a = 0.0,
				norm_b = 0.0;
	int			i;
	float4		result;

	check_dimensions(a, b);

	/* Compute dot(a, b), ||a||^2, ||b||^2 */
	for (i = 0; i < a->dim; i++)
	{
		double		va = (double) a->data[i];
		double		vb = (double) b->data[i];

		dot += va * vb;
		norm_a += va * va;
		norm_b += vb * vb;
	}

	/* Handle zero norms to prevent divide by zero */
	if (norm_a == 0.0 || norm_b == 0.0)
		return 1.0;

	result = (float4) (1.0 - (dot / (sqrt(norm_a) * sqrt(norm_b))));

	/* Validate result */
	if (isnan(result) || isinf(result))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("cosine distance calculation resulted in NaN or Infinity")));

	return result;
}

PG_FUNCTION_INFO_V1(vector_cosine_distance);
Datum
vector_cosine_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);
	/* Use SIMD-optimized version if available */
	PG_RETURN_FLOAT4(cosine_distance_simd(a, b));
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
	Vector	   *a;
	Vector	   *b;

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);
	/* Use SIMD-optimized version if available */
	PG_RETURN_FLOAT4(l1_distance_simd(a, b));
}

/* Hamming distance: counts differing coordinates */
PG_FUNCTION_INFO_V1(vector_hamming_distance);
Datum
vector_hamming_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;
	int			count = 0;
	int			i;

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);

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
	Vector	   *a;
	Vector	   *b;
	double		max_diff = 0.0;
	int			i;

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);

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
	Vector	   *a;
	Vector	   *b;
	float8		p;
	double		sum = 0.0;
	int			i;

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);
	p = PG_GETARG_FLOAT8(2);

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

/*
 * Squared Euclidean distance: L2^2 (faster, no sqrt)
 * Useful when only relative distances matter
 */
PG_FUNCTION_INFO_V1(vector_squared_l2_distance);
Datum
vector_squared_l2_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;
	double		sum = 0.0;
	double		c = 0.0;
	int			i;

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);

	check_dimensions(a, b);

	for (i = 0; i < a->dim; i++)
	{
		double		diff = (double) a->data[i] - (double) b->data[i];
		double		y = (diff * diff) - c;
		double		t = sum + y;

		c = (t - sum) - y;
		sum = t;
	}

	PG_RETURN_FLOAT8(sum);
}

/*
 * Jaccard distance: 1 - Jaccard similarity
 * Jaccard similarity = |A ∩ B| / |A ∪ B|
 * For vectors, treats non-zero values as set membership
 */
PG_FUNCTION_INFO_V1(vector_jaccard_distance);
Datum
vector_jaccard_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;
	int			intersection = 0;
	int			union_count = 0;
	int			i;
	double		jaccard_sim;

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);

	check_dimensions(a, b);

	for (i = 0; i < a->dim; i++)
	{
		bool		a_nonzero = (fabs((double) a->data[i]) > 1e-10);
		bool		b_nonzero = (fabs((double) b->data[i]) > 1e-10);

		if (a_nonzero && b_nonzero)
			intersection++;
		if (a_nonzero || b_nonzero)
			union_count++;
	}

	if (union_count == 0)
	{
		PG_RETURN_FLOAT8(0.0);	/* Both vectors are zero */
	}

	jaccard_sim = (double) intersection / (double) union_count;
	PG_RETURN_FLOAT8(1.0 - jaccard_sim);
}

/*
 * Dice distance: 1 - Dice coefficient
 * Dice coefficient = 2|A ∩ B| / (|A| + |B|)
 * For vectors, treats non-zero values as set membership
 */
PG_FUNCTION_INFO_V1(vector_dice_distance);
Datum
vector_dice_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;
	int			intersection = 0;
	int			a_count = 0;
	int			b_count = 0;
	int			i;
	double		dice_coeff;

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);

	check_dimensions(a, b);

	for (i = 0; i < a->dim; i++)
	{
		bool		a_nonzero = (fabs((double) a->data[i]) > 1e-10);
		bool		b_nonzero = (fabs((double) b->data[i]) > 1e-10);

		if (a_nonzero && b_nonzero)
			intersection++;
		if (a_nonzero)
			a_count++;
		if (b_nonzero)
			b_count++;
	}

	if (a_count == 0 && b_count == 0)
		return 0.0;				/* Both vectors are zero */

	if (a_count == 0 || b_count == 0)
		return 1.0;				/* No overlap */

	dice_coeff = (2.0 * (double) intersection) / ((double) a_count + (double) b_count);
	PG_RETURN_FLOAT8(1.0 - dice_coeff);
}

/*
 * Mahalanobis distance: sqrt((a-b)^T * S^(-1) * (a-b))
 * Where S is the covariance matrix
 * Requires pre-computed covariance matrix (passed as vector of inverse diagonal)
 * Simplified version: assumes diagonal covariance matrix
 */
PG_FUNCTION_INFO_V1(vector_mahalanobis_distance);
Datum
vector_mahalanobis_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;
	Vector	   *covariance_inv;
	double		sum = 0.0;
	int			i;

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);
	covariance_inv = PG_ARGISNULL(2) ? NULL : PG_GETARG_VECTOR_P(2);

	check_dimensions(a, b);

	if (covariance_inv == NULL)
	{
		/* No covariance matrix: fall back to L2 distance */
		PG_RETURN_FLOAT4(l2_distance(a, b));
	}

	if (covariance_inv->dim != a->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("covariance matrix dimension must match vector dimension: %d vs %d",
						covariance_inv->dim,
						a->dim)));

	/* Compute Mahalanobis distance with diagonal covariance assumption */
	for (i = 0; i < a->dim; i++)
	{
		double		diff = (double) a->data[i] - (double) b->data[i];
		double		inv_var = (double) covariance_inv->data[i];

		if (inv_var <= 0.0 || isnan(inv_var) || isinf(inv_var))
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("covariance inverse must be positive and finite")));

		sum += diff * diff * inv_var;
	}

	PG_RETURN_FLOAT8(sqrt(sum));
}
