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
 * Copyright (c) 2024-2025, neurondb, Inc.
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

/* Helper: Check dimensions match */
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
	int32	   *a_indices;
	float4	   *a_values;
	int32	   *b_indices;
	float4	   *b_values;
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

	/*
	 * Compute L2 distance using a merge algorithm that processes both sparse
	 * vectors simultaneously. Since sparse vectors store only non-zero
	 * elements with their indices, we cannot simply iterate through all
	 * dimensions. Instead, we maintain pointers into both index arrays and
	 * process elements in sorted index order. When indices match, we compute
	 * the difference between both values. When indices differ, we treat the
	 * missing value as zero and compute the difference accordingly. The
	 * algorithm uses Kahan summation to maintain numerical precision when
	 * accumulating squared differences, which is critical for
	 * high-dimensional sparse vectors where cancellation errors can
	 * accumulate. This approach achieves O(nnz) time complexity where nnz is
	 * the number of non-zero elements, compared to O(dim) for dense vectors.
	 */
	i = 0;
	j = 0;

	while (i < a->nnz || j < b->nnz)
	{
		if (i >= a->nnz)
		{
			/* Only b has value at this index */
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
			/* Only a has value at this index */
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
				/* Only a has value at this index */
				diff = (double) a_values[i] - 0.0;
				y = (diff * diff) - c;
				t = sum + y;
				c = (t - sum) - y;
				sum = t;
				i++;
			}
			else if (idx_a > idx_b)
			{
				/* Only b has value at this index */
				diff = 0.0 - (double) b_values[j];
				y = (diff * diff) - c;
				t = sum + y;
				c = (t - sum) - y;
				sum = t;
				j++;
			}
			else
			{
				/* Both have values at this index */
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
	int32	   *a_indices;
	float4	   *a_values;
	int32	   *b_indices;
	float4	   *b_values;
	double		sum = 0.0;
	int			i;
	int			j;

	check_vecmap_dimensions(a, b);

	a_indices = VECMAP_INDICES(a);
	a_values = VECMAP_VALUES(a);
	b_indices = VECMAP_INDICES(b);
	b_values = VECMAP_VALUES(b);

	/* Merge-like algorithm: only multiply when both have non-zero values */
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
			/* Both have values at this index */
			sum += (double) a_values[i] * (double) b_values[j];
			i++;
			j++;
		}
	}

	PG_RETURN_FLOAT4((float4) sum);
}

/*
 * Sparse cosine distance: optimized for sparse vectors
 */
PG_FUNCTION_INFO_V1(vecmap_cosine_distance);
Datum
vecmap_cosine_distance(PG_FUNCTION_ARGS)
{
	VectorMap  *a = (VectorMap *) PG_GETARG_POINTER(0);
	VectorMap  *b = (VectorMap *) PG_GETARG_POINTER(1);
	int32	   *a_indices;
	float4	   *a_values;
	int32	   *b_indices;
	float4	   *b_values;
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

	/* Compute dot product and norms simultaneously */
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
				/* Both have values at this index */
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

	/* Handle zero norms */
	if (norm_a == 0.0 || norm_b == 0.0)
		PG_RETURN_FLOAT4(1.0f);

	PG_RETURN_FLOAT4((float4) (1.0 - (dot / (sqrt(norm_a) * sqrt(norm_b)))));
}

/*
 * Sparse L1 (Manhattan) distance: optimized for sparse vectors
 */
PG_FUNCTION_INFO_V1(vecmap_l1_distance);
Datum
vecmap_l1_distance(PG_FUNCTION_ARGS)
{
	VectorMap  *a = (VectorMap *) PG_GETARG_POINTER(0);
	VectorMap  *b = (VectorMap *) PG_GETARG_POINTER(1);
	int32	   *a_indices;
	float4	   *a_values;
	int32	   *b_indices;
	float4	   *b_values;
	double		sum = 0.0;
	int			i;
	int			j;

	check_vecmap_dimensions(a, b);

	a_indices = VECMAP_INDICES(a);
	a_values = VECMAP_VALUES(a);
	b_indices = VECMAP_INDICES(b);
	b_values = VECMAP_VALUES(b);

	/* Merge-like algorithm */
	i = 0;
	j = 0;

	while (i < a->nnz || j < b->nnz)
	{
		if (i >= a->nnz)
		{
			/* Only b has value */
			sum += fabs((double) b_values[j]);
			j++;
		}
		else if (j >= b->nnz)
		{
			/* Only a has value */
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
				/* Both have values */
				sum += fabs((double) a_values[i] - (double) b_values[j]);
				i++;
				j++;
			}
		}
	}

	PG_RETURN_FLOAT4((float4) sum);
}

/*
 * Sparse vector addition: creates new sparse vector
 */
PG_FUNCTION_INFO_V1(vecmap_add);
Datum
vecmap_add(PG_FUNCTION_ARGS)
{
	VectorMap  *a = (VectorMap *) PG_GETARG_POINTER(0);
	VectorMap  *b = (VectorMap *) PG_GETARG_POINTER(1);
	int32	   *a_indices;
	float4	   *a_values;
	int32	   *b_indices;
	float4	   *b_values;
	VectorMap  *result;
	int32	   *result_indices;
	float4	   *result_values;
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

	/* Maximum possible nnz is sum of both (if no overlap) */
	max_nnz = a->nnz + b->nnz;
	result_indices = NULL;
	result_values = NULL;
	NDB_ALLOC(result_indices, int32, max_nnz);
	NDB_ALLOC(result_values, float4, max_nnz);

	/* Merge-like addition */
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
				/* Both have values - add them */
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

	/* Allocate result - variable-length type requires custom size */
	size = sizeof(VectorMap) + sizeof(int32) * result_nnz +
		sizeof(float4) * result_nnz;

	/*
	 * Variable-length type requires custom size - use palloc0 but ensure
	 * proper cleanup
	 */
	/* Note: For variable-length PostgreSQL types, palloc0 is standard */
	result = (VectorMap *) palloc0(size);
	SET_VARSIZE(result, size);

	result->total_dim = a->total_dim;
	result->nnz = result_nnz;

	memcpy(VECMAP_INDICES(result), result_indices, sizeof(int32) * result_nnz);
	memcpy(VECMAP_VALUES(result), result_values, sizeof(float4) * result_nnz);

	NDB_FREE(result_indices);
	NDB_FREE(result_values);

	PG_RETURN_POINTER(result);
}

/*
 * Sparse vector subtraction: creates new sparse vector
 */
PG_FUNCTION_INFO_V1(vecmap_sub);
Datum
vecmap_sub(PG_FUNCTION_ARGS)
{
	VectorMap  *a = (VectorMap *) PG_GETARG_POINTER(0);
	VectorMap  *b = (VectorMap *) PG_GETARG_POINTER(1);
	int32	   *a_indices;
	float4	   *a_values;
	int32	   *b_indices;
	float4	   *b_values;
	VectorMap  *result;
	int32	   *result_indices;
	float4	   *result_values;
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
	NDB_ALLOC(result_indices, int32, max_nnz);
	NDB_ALLOC(result_values, float4, max_nnz);

	/* Merge-like subtraction */
	i = 0;
	j = 0;
	k = 0;

	while (i < a->nnz || j < b->nnz)
	{
		if (i >= a->nnz)
		{
			result_indices[k] = b_indices[j];
			result_values[k] = -b_values[j];	/* Negate b */
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
				/* Both have values - subtract */
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

	memcpy(VECMAP_INDICES(result), result_indices, sizeof(int32) * result_nnz);
	memcpy(VECMAP_VALUES(result), result_values, sizeof(float4) * result_nnz);

	NDB_FREE(result_indices);
	NDB_FREE(result_values);

	PG_RETURN_POINTER(result);
}

/*
 * Sparse vector scalar multiplication: creates new sparse vector
 */
PG_FUNCTION_INFO_V1(vecmap_mul_scalar);
Datum
vecmap_mul_scalar(PG_FUNCTION_ARGS)
{
	VectorMap  *a = (VectorMap *) PG_GETARG_POINTER(0);
	float4		scalar = PG_GETARG_FLOAT4(1);
	VectorMap  *result;
	int32	   *a_indices;
	float4	   *a_values;
	int32	   *result_indices;
	float4	   *result_values;
	int32		result_nnz;
	int			i;
	int			size;

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

/*
 * Sparse vector norm (L2): compute ||v||
 */
PG_FUNCTION_INFO_V1(vecmap_norm);
Datum
vecmap_norm(PG_FUNCTION_ARGS)
{
	VectorMap  *a = (VectorMap *) PG_GETARG_POINTER(0);
	float4	   *values;
	double		sum = 0.0;
	int			i;

	values = VECMAP_VALUES(a);

	for (i = 0; i < a->nnz; i++)
		sum += (double) values[i] * (double) values[i];

	PG_RETURN_FLOAT4((float4) sqrt(sum));
}
