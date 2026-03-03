/*-------------------------------------------------------------------------
 *
 * vector_sparse.c
 *	  Sparse vector operations optimized for sparse data
 *
 * This file implements distance metrics and arithmetic operations for
 * sparse vectors (vecmap type). All operations are optimized to only
 * iterate over non-zero entries, providing O(nnz) complexity instead
 * of O(dim) for dense vectors.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/vector/vector_sparse.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_types.h"
#include "fmgr.h"
#include "lib/stringinfo.h"
#include <math.h>
#include <float.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

static inline void
check_vecmap_dimensions(const VectorMap *a, const VectorMap *b)
{
	if (a->total_dim != b->total_dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("sparse vector dimensions must match: %d vs %d",
						a->total_dim,
						b->total_dim)));
}

/*
 * Sparse L2 distance: optimized for sparse vectors
 * Uses merge-like algorithm to only process non-zero entries
 */
PG_FUNCTION_INFO_V1(vecmap_l2_distance);
Datum
vecmap_l2_distance(PG_FUNCTION_ARGS)
{
	VectorMap  *a = (VectorMap *) PG_GETARG_POINTER(0);
	VectorMap  *b = (VectorMap *) PG_GETARG_POINTER(1);
	int32 *a_indices = NULL;
	float4 *a_values = NULL;
	int32 *b_indices = NULL;
	float4 *b_values = NULL;
	double		sum = 0.0;
	double		c = 0.0;		/* Kahan summation correction */
	int			i;
	int			j;
	int32		idx_a;
	int32		idx_b;
	double		diff;
	double		y;
	double		t;

	check_vecmap_dimensions(a, b);

	a_indices = VECMAP_INDICES(a);
	a_values = VECMAP_VALUES(a);
	b_indices = VECMAP_INDICES(b);
	b_values = VECMAP_VALUES(b);

	i = 0;
	j = 0;

	while (i < a->nnz || j < b->nnz)
	{
		if (i >= a->nnz)
		{
			idx_b = b_indices[j];
			diff = 0.0 - (double) b_values[j];
			y = (diff * diff) - c;
			t = sum + y;
			c = (t - sum) - y;
			sum = t;
			j++;
		}
		else if (j >= b->nnz)
		{
			idx_a = a_indices[i];
			diff = (double) a_values[i] - 0.0;
			y = (diff * diff) - c;
			t = sum + y;
			c = (t - sum) - y;
			sum = t;
			i++;
		}
		else
		{
			idx_a = a_indices[i];
			idx_b = b_indices[j];

			if (idx_a < idx_b)
			{
				diff = (double) a_values[i] - 0.0;
				y = (diff * diff) - c;
				t = sum + y;
				c = (t - sum) - y;
				sum = t;
				i++;
			}
			else if (idx_a > idx_b)
			{
				diff = 0.0 - (double) b_values[j];
				y = (diff * diff) - c;
				t = sum + y;
				c = (t - sum) - y;
				sum = t;
				j++;
			}
			else
			{
				diff = (double) a_values[i] - (double) b_values[j];
				y = (diff * diff) - c;
				t = sum + y;
				c = (t - sum) - y;
				sum = t;
				i++;
				j++;
			}
		}
	}

	PG_RETURN_FLOAT4((float4) sqrt(sum));
}

/*
 * Sparse inner product: optimized for sparse vectors
 */
PG_FUNCTION_INFO_V1(vecmap_inner_product);
Datum
vecmap_inner_product(PG_FUNCTION_ARGS)
{
	VectorMap  *a = (VectorMap *) PG_GETARG_POINTER(0);
	VectorMap  *b = (VectorMap *) PG_GETARG_POINTER(1);
	int32 *a_indices = NULL;
	float4 *a_values = NULL;
	int32 *b_indices = NULL;
	float4 *b_values = NULL;
	double		sum = 0.0;
	int			i;
	int			j;

	check_vecmap_dimensions(a, b);

	a_indices = VECMAP_INDICES(a);
	a_values = VECMAP_VALUES(a);
	b_indices = VECMAP_INDICES(b);
	b_values = VECMAP_VALUES(b);

	i = 0;
	j = 0;

	while (i < a->nnz && j < b->nnz)
	{
		if (a_indices[i] < b_indices[j])
		{
			i++;
		}
		else if (a_indices[i] > b_indices[j])
		{
			j++;
		}
		else
		{
			sum += (double) a_values[i] * (double) b_values[j];
			i++;
			j++;
		}
	}

	PG_RETURN_FLOAT4((float4) sum);
}

PG_FUNCTION_INFO_V1(vecmap_cosine_distance);
Datum
vecmap_cosine_distance(PG_FUNCTION_ARGS)
{
	VectorMap  *a = (VectorMap *) PG_GETARG_POINTER(0);
	VectorMap  *b = (VectorMap *) PG_GETARG_POINTER(1);
	int32 *a_indices = NULL;
	float4 *a_values = NULL;
	int32 *b_indices = NULL;
	float4 *b_values = NULL;
	double		dot = 0.0;
	double		norm_a = 0.0;
	double		norm_b = 0.0;
	int			i;
	int			j;
	int32		idx_a;
	int32		idx_b;

	check_vecmap_dimensions(a, b);

	a_indices = VECMAP_INDICES(a);
	a_values = VECMAP_VALUES(a);
	b_indices = VECMAP_INDICES(b);
	b_values = VECMAP_VALUES(b);

	i = 0;
	j = 0;

	while (i < a->nnz || j < b->nnz)
	{
		if (i >= a->nnz)
		{
			idx_b = b_indices[j];
			norm_b += (double) b_values[j] * (double) b_values[j];
			j++;
		}
		else if (j >= b->nnz)
		{
			idx_a = a_indices[i];
			norm_a += (double) a_values[i] * (double) a_values[i];
			i++;
		}
		else
		{
			idx_a = a_indices[i];
			idx_b = b_indices[j];

			if (idx_a < idx_b)
			{
				norm_a += (double) a_values[i] * (double) a_values[i];
				i++;
			}
			else if (idx_a > idx_b)
			{
				norm_b += (double) b_values[j] * (double) b_values[j];
				j++;
			}
			else
			{
				double		va = (double) a_values[i];
				double		vb = (double) b_values[j];

				dot += va * vb;
				norm_a += va * va;
				norm_b += vb * vb;
				i++;
				j++;
			}
		}
	}

	if (norm_a == 0.0 || norm_b == 0.0)
		PG_RETURN_FLOAT4(1.0f);

	PG_RETURN_FLOAT4((float4) (1.0 - (dot / (sqrt(norm_a) * sqrt(norm_b)))));
}

PG_FUNCTION_INFO_V1(vecmap_l1_distance);
Datum
vecmap_l1_distance(PG_FUNCTION_ARGS)
{
	VectorMap  *a = (VectorMap *) PG_GETARG_POINTER(0);
	VectorMap  *b = (VectorMap *) PG_GETARG_POINTER(1);
	int32 *a_indices = NULL;
	float4 *a_values = NULL;
	int32 *b_indices = NULL;
	float4 *b_values = NULL;
	double		sum = 0.0;
	int			i;
	int			j;

	check_vecmap_dimensions(a, b);

	a_indices = VECMAP_INDICES(a);
	a_values = VECMAP_VALUES(a);
	b_indices = VECMAP_INDICES(b);
	b_values = VECMAP_VALUES(b);

	i = 0;
	j = 0;

	while (i < a->nnz || j < b->nnz)
	{
		if (i >= a->nnz)
		{
			sum += fabs((double) b_values[j]);
			j++;
		}
		else if (j >= b->nnz)
		{
			sum += fabs((double) a_values[i]);
			i++;
		}
		else
		{
			if (a_indices[i] < b_indices[j])
			{
				sum += fabs((double) a_values[i]);
				i++;
			}
			else if (a_indices[i] > b_indices[j])
			{
				sum += fabs((double) b_values[j]);
				j++;
			}
			else
			{
				sum += fabs((double) a_values[i] - (double) b_values[j]);
				i++;
				j++;
			}
		}
	}

	PG_RETURN_FLOAT4((float4) sum);
}

PG_FUNCTION_INFO_V1(vecmap_add);
Datum
vecmap_add(PG_FUNCTION_ARGS)
{
	VectorMap  *a = (VectorMap *) PG_GETARG_POINTER(0);
	VectorMap  *b = (VectorMap *) PG_GETARG_POINTER(1);
	int32 *a_indices = NULL;
	float4 *a_values = NULL;
	int32 *b_indices = NULL;
	float4 *b_values = NULL;
	VectorMap *result = NULL;
	int32 *result_indices = NULL;
	float4 *result_values = NULL;
	int32		max_nnz;
	int32		result_nnz;
	int			i;
	int			j;
	int			k;
	int			size;

	check_vecmap_dimensions(a, b);

	a_indices = VECMAP_INDICES(a);
	a_values = VECMAP_VALUES(a);
	b_indices = VECMAP_INDICES(b);
	b_values = VECMAP_VALUES(b);
	
	/* Validate macro results are not null (macros now check, but double-check for safety) */
	if (a_indices == NULL || a_values == NULL || b_indices == NULL || b_values == NULL)
	{
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: VECMAP macro returned NULL pointer")));
	}

	/* Maximum possible nnz is sum of both (if no overlap) */
	max_nnz = a->nnz + b->nnz;
	result_indices = NULL;
	result_values = NULL;
	nalloc(result_indices, int32, max_nnz);
	nalloc(result_values, float4, max_nnz);

	i = 0;
	j = 0;
	k = 0;

	while (i < a->nnz || j < b->nnz)
	{
		if (i >= a->nnz)
		{
			result_indices[k] = b_indices[j];
			result_values[k] = b_values[j];
			j++;
			k++;
		}
		else if (j >= b->nnz)
		{
			result_indices[k] = a_indices[i];
			result_values[k] = a_values[i];
			i++;
			k++;
		}
		else
		{
			if (a_indices[i] < b_indices[j])
			{
				result_indices[k] = a_indices[i];
				result_values[k] = a_values[i];
				i++;
				k++;
			}
			else if (a_indices[i] > b_indices[j])
			{
				result_indices[k] = b_indices[j];
				result_values[k] = b_values[j];
				j++;
				k++;
			}
			else
			{
				float4		sum_val = a_values[i] + b_values[j];

				if (fabs(sum_val) > 1e-10f) /* Skip near-zero results */
				{
					result_indices[k] = a_indices[i];
					result_values[k] = sum_val;
					k++;
				}
				i++;
				j++;
			}
		}
	}

	result_nnz = k;

	size = sizeof(VectorMap) + sizeof(int32) * result_nnz +
		sizeof(float4) * result_nnz;

	result = (VectorMap *) palloc0(size);
	SET_VARSIZE(result, size);

	result->total_dim = a->total_dim;
	result->nnz = result_nnz;

	/* Suppress nonnull warning - result is guaranteed non-null after palloc0 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnonnull"
	memcpy(VECMAP_INDICES(result), result_indices, sizeof(int32) * result_nnz);
	memcpy(VECMAP_VALUES(result), result_values, sizeof(float4) * result_nnz);
#pragma GCC diagnostic pop

	pfree(result_indices);
	pfree(result_values);

	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(vecmap_sub);
Datum
vecmap_sub(PG_FUNCTION_ARGS)
{
	VectorMap  *a = (VectorMap *) PG_GETARG_POINTER(0);
	VectorMap  *b = (VectorMap *) PG_GETARG_POINTER(1);
	int32 *a_indices = NULL;
	float4 *a_values = NULL;
	int32 *b_indices = NULL;
	float4 *b_values = NULL;
	VectorMap *result = NULL;
	int32 *result_indices = NULL;
	float4 *result_values = NULL;
	int32		max_nnz;
	int32		result_nnz;
	int			i;
	int			j;
	int			k;
	int			size;

	check_vecmap_dimensions(a, b);

	a_indices = VECMAP_INDICES(a);
	a_values = VECMAP_VALUES(a);
	b_indices = VECMAP_INDICES(b);
	b_values = VECMAP_VALUES(b);

	max_nnz = a->nnz + b->nnz;
	result_indices = NULL;
	result_values = NULL;
	nalloc(result_indices, int32, max_nnz);
	nalloc(result_values, float4, max_nnz);

	i = 0;
	j = 0;
	k = 0;

	while (i < a->nnz || j < b->nnz)
	{
		if (i >= a->nnz)
		{
			result_indices[k] = b_indices[j];
			result_values[k] = -b_values[j];
			j++;
			k++;
		}
		else if (j >= b->nnz)
		{
			result_indices[k] = a_indices[i];
			result_values[k] = a_values[i];
			i++;
			k++;
		}
		else
		{
			if (a_indices[i] < b_indices[j])
			{
				result_indices[k] = a_indices[i];
				result_values[k] = a_values[i];
				i++;
				k++;
			}
			else if (a_indices[i] > b_indices[j])
			{
				result_indices[k] = b_indices[j];
				result_values[k] = -b_values[j];
				j++;
				k++;
			}
			else
			{
				float4		diff_val = a_values[i] - b_values[j];

				if (fabs(diff_val) > 1e-10f)
				{
					result_indices[k] = a_indices[i];
					result_values[k] = diff_val;
					k++;
				}
				i++;
				j++;
			}
		}
	}

	result_nnz = k;

	size = sizeof(VectorMap) + sizeof(int32) * result_nnz +
		sizeof(float4) * result_nnz;
	result = (VectorMap *) palloc0(size);
	SET_VARSIZE(result, size);

	result->total_dim = a->total_dim;
	result->nnz = result_nnz;

	/* Suppress nonnull warning - result is guaranteed non-null after palloc0 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnonnull"
	memcpy(VECMAP_INDICES(result), result_indices, sizeof(int32) * result_nnz);
	memcpy(VECMAP_VALUES(result), result_values, sizeof(float4) * result_nnz);
#pragma GCC diagnostic pop

	pfree(result_indices);
	pfree(result_values);

	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(vecmap_mul_scalar);
Datum
vecmap_mul_scalar(PG_FUNCTION_ARGS)
{
	VectorMap  *a = (VectorMap *) PG_GETARG_POINTER(0);
	float4		scalar = PG_GETARG_FLOAT4(1);
	VectorMap *result = NULL;
	int32 *a_indices = NULL;
	float4 *a_values = NULL;
	int32 *result_indices = NULL;
	float4 *result_values = NULL;
	int32		result_nnz;
	int			i;
	int			size;

	a_indices = VECMAP_INDICES(a);
	a_values = VECMAP_VALUES(a);

	result_nnz = 0;
	for (i = 0; i < a->nnz; i++)
	{
		if (fabs(a_values[i] * scalar) > 1e-10f)
			result_nnz++;
	}

	size = sizeof(VectorMap) + sizeof(int32) * result_nnz +
		sizeof(float4) * result_nnz;
	result = (VectorMap *) palloc0(size);
	SET_VARSIZE(result, size);

	result->total_dim = a->total_dim;
	result->nnz = result_nnz;

	result_indices = VECMAP_INDICES(result);
	result_values = VECMAP_VALUES(result);

	{
		int			k = 0;

		for (i = 0; i < a->nnz; i++)
		{
			float4		scaled = a_values[i] * scalar;

			if (fabs(scaled) > 1e-10f)
			{
				result_indices[k] = a_indices[i];
				result_values[k] = scaled;
				k++;
			}
		}
	}

	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(vecmap_norm);
Datum
vecmap_norm(PG_FUNCTION_ARGS)
{
	VectorMap  *a = (VectorMap *) PG_GETARG_POINTER(0);
	float4	   *values = NULL;
	double		sum = 0.0;
	int			i;

	values = VECMAP_VALUES(a);

	for (i = 0; i < a->nnz; i++)
		sum += (double) values[i] * (double) values[i];

	PG_RETURN_FLOAT4((float4) sqrt(sum));
}

/*
 * sparsevec_add: Add two sparsevec vectors
 * Wrapper around vecmap_add for sparsevec type
 */
PG_FUNCTION_INFO_V1(sparsevec_add);
Datum
sparsevec_add(PG_FUNCTION_ARGS)
{
	return vecmap_add(fcinfo);
}

/*
 * sparsevec_sub: Subtract two sparsevec vectors
 * Wrapper around vecmap_sub for sparsevec type
 */
PG_FUNCTION_INFO_V1(sparsevec_sub);
Datum
sparsevec_sub(PG_FUNCTION_ARGS)
{
	return vecmap_sub(fcinfo);
}

/*
 * sparsevec_mul: Multiply sparsevec by scalar
 * Wrapper around vecmap_mul_scalar for sparsevec type
 */
PG_FUNCTION_INFO_V1(sparsevec_mul);
Datum
sparsevec_mul(PG_FUNCTION_ARGS)
{
	VectorMap  *a = (VectorMap *) PG_GETARG_POINTER(0);
	float8		scalar = PG_GETARG_FLOAT8(1);
	VectorMap  *result = NULL;
	int32	   *a_indices = NULL;
	float4	   *a_values = NULL;
	int32	   *result_indices = NULL;
	float4	   *result_values = NULL;
	int32		result_nnz;
	int			i;
	int			size;

	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: sparsevec_mul requires 2 arguments")));

	if (a == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot multiply NULL sparsevec")));

	if (isnan(scalar) || isinf(scalar))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("scalar multiplier cannot be NaN or Infinity")));

	a_indices = VECMAP_INDICES(a);
	a_values = VECMAP_VALUES(a);

	/* Count non-zero results */
	result_nnz = 0;
	for (i = 0; i < a->nnz; i++)
	{
		if (fabs(a_values[i] * scalar) > 1e-10f)
			result_nnz++;
	}

	size = sizeof(VectorMap) + sizeof(int32) * result_nnz +
		sizeof(float4) * result_nnz;
	result = (VectorMap *) palloc0(size);
	SET_VARSIZE(result, size);
	result->total_dim = a->total_dim;
	result->nnz = result_nnz;

	result_indices = VECMAP_INDICES(result);
	result_values = VECMAP_VALUES(result);

	{
		int			k = 0;

		for (i = 0; i < a->nnz; i++)
		{
			float4		scaled = a_values[i] * scalar;

			if (fabs(scaled) > 1e-10f)
			{
				result_indices[k] = a_indices[i];
				result_values[k] = scaled;
				k++;
			}
		}
	}

	PG_RETURN_POINTER(result);
}
