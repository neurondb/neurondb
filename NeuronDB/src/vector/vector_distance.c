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

extern float4 l2_distance_simd(Vector *a, Vector *b);
extern float4 inner_product_simd(Vector *a, Vector *b);

/* Internal distance functions */
static float4 l2_squared_distance(Vector *a, Vector *b);
extern float4 cosine_distance_simd(Vector *a, Vector *b);
extern float4 l1_distance_simd(Vector *a, Vector *b);

/* Forward declaration */
float4 spherical_distance(Vector *a, Vector *b);

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

/*
 * l2_distance
 *    Compute Euclidean distance between two vectors using Kahan summation.
 *
 * This function calculates the L2 norm distance between two vectors by
 * summing the squared differences of corresponding elements and taking the
 * square root. The implementation uses Kahan summation algorithm to maintain
 * numerical precision when accumulating squared differences across many
 * dimensions. Standard floating-point addition can lose precision when adding
 * numbers of vastly different magnitudes, which is common in high-dimensional
 * vector spaces. The Kahan algorithm compensates for rounding errors by
 * tracking the accumulated error in a correction term and incorporating it
 * into subsequent additions. This ensures that the final result maintains
 * accuracy even when summing thousands of small squared differences, which
 * is critical for reliable distance calculations in vector similarity search
 * and machine learning applications where small errors can compound.
 */
float4
l2_distance(Vector *a, Vector *b)
{
	double		c = 0.0;
	double		sum = 0.0;
	float4		result;
	int			i;

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
	/* Use SIMD-optimized version if available */
	PG_RETURN_FLOAT4(l2_distance_simd(a, b));
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
	/* Use SIMD-optimized version if available */
	PG_RETURN_FLOAT4(inner_product_simd(a, b));
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

float4
cosine_distance(Vector *a, Vector *b)
{
	double		dot = 0.0,
				norm_a = 0.0,
				norm_b = 0.0;
	int			i;
	float4		result;

	check_dimensions(a, b);

	for (i = 0; i < a->dim; i++)
	{
		double		va = (double) a->data[i];
		double		vb = (double) b->data[i];

		dot += va * vb;
		norm_a += va * va;
		norm_b += vb * vb;
	}

	if (norm_a == 0.0 || norm_b == 0.0)
		return 1.0;

	result = (float4) (1.0 - (dot / (sqrt(norm_a) * sqrt(norm_b))));

	/* Clamp to non-negative to handle floating-point rounding errors */
	if (result < 0.0f)
		result = 0.0f;

	/* Validate result */
	if (isnan(result) || isinf(result))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("cosine distance calculation resulted in NaN or Infinity")));

	return result;
}

/*
 * spherical_distance - Internal function for index optimization
 * Computes spherical distance: 1 - (dot / (norm_a * norm_b))
 * This is similar to cosine distance but used internally by indexes
 * for optimization purposes.
 */
float4
spherical_distance(Vector *a, Vector *b)
{
	double		dot = 0.0,
				norm_a = 0.0,
				norm_b = 0.0;
	int			i;
	float4		result;

	check_dimensions(a, b);

	for (i = 0; i < a->dim; i++)
	{
		double		va = (double) a->data[i];
		double		vb = (double) b->data[i];

		dot += va * vb;
		norm_a += va * va;
		norm_b += vb * vb;
	}

	if (norm_a == 0.0 || norm_b == 0.0)
		return 1.0;

	result = (float4) (1.0 - (dot / (sqrt(norm_a) * sqrt(norm_b))));

	/* Validate result */
	if (isnan(result) || isinf(result))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("spherical distance calculation resulted in NaN or Infinity")));

	return result;
}

PG_FUNCTION_INFO_V1(vector_spherical_distance);
Datum
vector_spherical_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a = NULL;
	Vector	   *b = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_spherical_distance requires 2 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);

	PG_RETURN_FLOAT4(spherical_distance(a, b));
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
	/* Use SIMD-optimized version if available */
	PG_RETURN_FLOAT4(cosine_distance_simd(a, b));
}

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
	/* Use SIMD-optimized version if available */
	PG_RETURN_FLOAT4(l1_distance_simd(a, b));
}

PG_FUNCTION_INFO_V1(vector_hamming_distance);
Datum
vector_hamming_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a = NULL;
	Vector *b = NULL;
	int			count = 0;
	int			i;

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_hamming_distance requires 2 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);

	check_dimensions(a, b);

	for (i = 0; i < a->dim; i++)
	{
		if (a->data[i] != b->data[i])
			count++;
	}

	PG_RETURN_INT32(count);
}

PG_FUNCTION_INFO_V1(vector_chebyshev_distance);
Datum
vector_chebyshev_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a = NULL;
	Vector *b = NULL;
	double		max_diff = 0.0;
	int			i;

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_chebyshev_distance requires 2 arguments")));

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

PG_FUNCTION_INFO_V1(vector_minkowski_distance);
Datum
vector_minkowski_distance(PG_FUNCTION_ARGS)
{
	Vector *a = NULL;
	Vector *b = NULL;
	float8		p;
	double		sum = 0.0;
	int			i;

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

	check_dimensions(a, b);

	if (p <= 0 || isnan(p) || isinf(p))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("p must be positive and finite")));

	if (p == 1.0)
	{
		for (i = 0; i < a->dim; i++)
			sum += fabs((double) a->data[i] - (double) b->data[i]);
		PG_RETURN_FLOAT8(sum);
	}
	else if (p == 2.0)
	{
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
	else if (p == INFINITY || p > 1e10)
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
	Vector *a = NULL;
	Vector *b = NULL;
	double		sum = 0.0;
	double		c = 0.0;
	int			i;

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_squared_l2_distance requires 2 arguments")));

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

PG_FUNCTION_INFO_V1(vector_jaccard_distance);
Datum
vector_jaccard_distance(PG_FUNCTION_ARGS)
{
	Vector *a = NULL;
	Vector *b = NULL;
	int			intersection = 0;
	int			union_count = 0;
	int			i;
	double		jaccard_sim;

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_jaccard_distance requires 2 arguments")));

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
		PG_RETURN_FLOAT8(0.0);
	}

	jaccard_sim = (double) intersection / (double) union_count;
	PG_RETURN_FLOAT8(1.0 - jaccard_sim);
}

PG_FUNCTION_INFO_V1(vector_dice_distance);
Datum
vector_dice_distance(PG_FUNCTION_ARGS)
{
	Vector *a = NULL;
	Vector *b = NULL;
	int			intersection = 0;
	int			a_count = 0;
	int			b_count = 0;
	int			i;
	double		dice_coeff;

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_dice_distance requires 2 arguments")));

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
		return 0.0;

	if (a_count == 0 || b_count == 0)
		return 1.0;

	dice_coeff = (2.0 * (double) intersection) / ((double) a_count + (double) b_count);
	PG_RETURN_FLOAT8(1.0 - dice_coeff);
}

PG_FUNCTION_INFO_V1(vector_mahalanobis_distance);
Datum
vector_mahalanobis_distance(PG_FUNCTION_ARGS)
{
	Vector *a = NULL;
	Vector *b = NULL;
	Vector *covariance_inv = NULL;
	double		sum = 0.0;
	int			i;

	/* Validate minimum argument count */
	if (PG_NARGS() < 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_mahalanobis_distance requires at least 2 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);
	covariance_inv = PG_ARGISNULL(2) ? NULL : PG_GETARG_VECTOR_P(2);

	check_dimensions(a, b);

	if (covariance_inv == NULL)
	{
		PG_RETURN_FLOAT4(l2_distance(a, b));
	}

	if (covariance_inv->dim != a->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("covariance matrix dimension must match vector dimension: %d vs %d",
						covariance_inv->dim,
						a->dim)));

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

/*
 * ============================================================================
 * Correlation-Based Distance Metrics
 * ============================================================================
 */

/*
 * Pearson correlation coefficient
 */
PG_FUNCTION_INFO_V1(vector_pearson_correlation);
Datum
vector_pearson_correlation(PG_FUNCTION_ARGS)
{
	Vector	   *a = NULL;
	Vector	   *b = NULL;
	double		mean_a = 0.0;
	double		mean_b = 0.0;
	double		cov = 0.0;
	double		var_a = 0.0;
	double		var_b = 0.0;
	int			i;

	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_pearson_correlation requires 2 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);

	check_dimensions(a, b);

	/* Compute means */
	for (i = 0; i < a->dim; i++)
	{
		mean_a += a->data[i];
		mean_b += b->data[i];
	}
	mean_a /= a->dim;
	mean_b /= a->dim;

	/* Compute covariance and variances */
	for (i = 0; i < a->dim; i++)
	{
		double		diff_a = a->data[i] - mean_a;
		double		diff_b = b->data[i] - mean_b;

		cov += diff_a * diff_b;
		var_a += diff_a * diff_a;
		var_b += diff_b * diff_b;
	}

	if (var_a == 0.0 || var_b == 0.0)
		PG_RETURN_FLOAT8(0.0);

	PG_RETURN_FLOAT8(cov / sqrt(var_a * var_b));
}

/*
 * Weighted distance with per-dimension weights
 */
PG_FUNCTION_INFO_V1(vector_weighted_distance);
Datum
vector_weighted_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a = NULL;
	Vector	   *b = NULL;
	Vector	   *weights = NULL;
	double		sum = 0.0;
	int			i;

	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_weighted_distance requires 3 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);
	weights = PG_GETARG_VECTOR_P(2);
	NDB_CHECK_VECTOR_VALID(weights);

	check_dimensions(a, b);

	if (weights->dim != a->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("weights vector dimension must match input vectors")));

	for (i = 0; i < a->dim; i++)
	{
		double		diff = a->data[i] - b->data[i];
		double		weight = weights->data[i];

		if (weight < 0.0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("weights must be non-negative")));

		sum += weight * diff * diff;
	}

	PG_RETURN_FLOAT8(sqrt(sum));
}

/*
 * ============================================================================
 * Information-Theoretic Distance Metrics
 * ============================================================================
 */

/*
 * Kullback-Leibler divergence (requires probability distributions)
 */
PG_FUNCTION_INFO_V1(vector_kl_divergence);
Datum
vector_kl_divergence(PG_FUNCTION_ARGS)
{
	Vector	   *p = NULL;
	Vector	   *q = NULL;
	double		kl = 0.0;
	int			i;

	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_kl_divergence requires 2 arguments")));

	p = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(p);
	q = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(q);

	check_dimensions(p, q);

	for (i = 0; i < p->dim; i++)
	{
		if (p->data[i] < 0.0 || q->data[i] < 0.0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("KL divergence requires non-negative probability distributions")));

		if (q->data[i] == 0.0 && p->data[i] > 0.0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("KL divergence undefined when q[i]=0 and p[i]>0")));

		if (p->data[i] > 0.0 && q->data[i] > 0.0)
			kl += p->data[i] * log(p->data[i] / q->data[i]);
	}

	PG_RETURN_FLOAT8(kl);
}

/*
 * Jensen-Shannon divergence
 */
PG_FUNCTION_INFO_V1(vector_js_divergence);
Datum
vector_js_divergence(PG_FUNCTION_ARGS)
{
	Vector	   *p = NULL;
	Vector	   *q = NULL;
	Vector	   *m = NULL;
	double		js = 0.0;
	int			i;

	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_js_divergence requires 2 arguments")));

	p = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(p);
	q = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(q);

	check_dimensions(p, q);

	/* Create mixture distribution m = (p + q) / 2 */
	m = new_vector(p->dim);
	for (i = 0; i < p->dim; i++)
		m->data[i] = (p->data[i] + q->data[i]) / 2.0;

	/* JS = (KL(p||m) + KL(q||m)) / 2 */
	js = (DatumGetFloat8(DirectFunctionCall2(vector_kl_divergence,
											 PointerGetDatum(p),
											 PointerGetDatum(m))) +
		  DatumGetFloat8(DirectFunctionCall2(vector_kl_divergence,
											 PointerGetDatum(q),
											 PointerGetDatum(m)))) / 2.0;

	pfree(m);

	PG_RETURN_FLOAT8(js);
}
