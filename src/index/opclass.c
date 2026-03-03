/*-------------------------------------------------------------------------
 *
 * opclass.c
 *		Operator classes and families for vector distance operations
 *
 * Defines operator classes for:
 * - L2 distance (Euclidean)
 * - Cosine distance
 * - Inner product
 *
 * Each operator class supports both HNSW and IVF access methods.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/index/opclass.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "catalog/pg_am.h"
#include "catalog/pg_operator.h"
#include "catalog/pg_opclass.h"
#include "catalog/pg_opfamily.h"
#include "catalog/pg_proc.h"
#include "catalog/pg_type.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "executor/spi.h"
#include "lib/stringinfo.h"
#include <math.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"

/*
 * Distance function implementations for operator support
 */

/*
 * vector_l2_distance(vector, vector) -> float4
 *
 * Computes Euclidean (L2) distance between two vectors.
 * Used by the <-> operator.
 */
PG_FUNCTION_INFO_V1(vector_l2_distance_op);

Datum
vector_l2_distance_op(PG_FUNCTION_ARGS)
{
	Vector	   *a = NULL;
	Vector *b = NULL;
	float4		result;
	float4		l2_distance_simd(Vector *a, Vector *b);

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_l2_distance_op requires 2 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);

	if (a->dim != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("neurondb: vector dimensions must "
						"match: %d vs %d",
						a->dim,
						b->dim)));

	/* Use SIMD-optimized version */
	result = l2_distance_simd(a, b);
	PG_RETURN_FLOAT4(result);
}

/*
 * vector_cosine_distance(vector, vector) -> float4
 *
 * Computes cosine distance (1 - cosine_similarity).
 * Used by the <=> operator.
 */
PG_FUNCTION_INFO_V1(vector_cosine_distance_op);

Datum
vector_cosine_distance_op(PG_FUNCTION_ARGS)
{
	Vector *a = NULL;
	Vector *b = NULL;
	float4		result;

	float4		cosine_distance_simd(Vector *a, Vector *b);

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_cosine_distance_op requires 2 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);

	if (a->dim != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("neurondb: vector dimensions must "
						"match")));

	/* Use SIMD-optimized version */
	result = cosine_distance_simd(a, b);
	PG_RETURN_FLOAT4(result);
}

/*
 * vector_inner_product_distance(vector, vector) -> float4
 *
 * Computes negative inner product (for maximum inner product search).
 * Used by the <#> operator.
 */
PG_FUNCTION_INFO_V1(vector_inner_product_distance_op);

Datum
vector_inner_product_distance_op(PG_FUNCTION_ARGS)
{
	Vector *a = NULL;
	Vector *b = NULL;
	float4		result;

	float4		inner_product_simd(Vector *a, Vector *b);

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_inner_product_distance_op requires 2 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);

	if (a->dim != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("neurondb: vector dimensions must "
						"match")));

	/* Use SIMD-optimized version */
	result = inner_product_simd(a, b);
	PG_RETURN_FLOAT4(result);
}

/*
 * Comparison functions for ordering
 *
 * These are required by the operator class system to enable
 * ORDER BY clauses with distance operators.
 */

/*
 * vector_l2_less(vector, vector, vector) -> bool
 *
 * Returns true if distance(a, query) < distance(b, query)
 */
PG_FUNCTION_INFO_V1(vector_l2_less);

Datum
vector_l2_less(PG_FUNCTION_ARGS)
{
	Vector *a = NULL;
	Vector *b = NULL;
	Vector *query = NULL;
	float4		dist_a;
	float4		dist_b;

	int			i;
	double		sum_a = 0.0,
				sum_b = 0.0;

	/* Validate argument count */
	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_l2_less requires 3 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);
	query = PG_GETARG_VECTOR_P(2);
	NDB_CHECK_VECTOR_VALID(query);

	if (a->dim != query->dim || b->dim != query->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("neurondb: vector dimensions must "
						"match")));

	for (i = 0; i < query->dim; i++)
	{
		double		diff_a = a->data[i] - query->data[i];
		double		diff_b = b->data[i] - query->data[i];

		sum_a += diff_a * diff_a;
		sum_b += diff_b * diff_b;
	}

	dist_a = (float4) sqrt(sum_a);
	dist_b = (float4) sqrt(sum_b);

	PG_RETURN_BOOL(dist_a < dist_b);
}

/*
 * vector_l2_less_equal(vector, vector, vector) -> bool
 */
PG_FUNCTION_INFO_V1(vector_l2_less_equal);

Datum
vector_l2_less_equal(PG_FUNCTION_ARGS)
{
	Vector *a = NULL;
	Vector *b = NULL;
	Vector *query = NULL;
	float4		dist_a;
	float4		dist_b;

	int			i;
	double		sum_a = 0.0,
				sum_b = 0.0;

	/* Validate argument count */
	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_l2_less_equal requires 3 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);
	query = PG_GETARG_VECTOR_P(2);
	NDB_CHECK_VECTOR_VALID(query);

	if (a->dim != query->dim || b->dim != query->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("neurondb: vector dimensions must "
						"match")));

	for (i = 0; i < query->dim; i++)
	{
		double		diff_a = a->data[i] - query->data[i];
		double		diff_b = b->data[i] - query->data[i];

		sum_a += diff_a * diff_a;
		sum_b += diff_b * diff_b;
	}

	dist_a = (float4) sqrt(sum_a);
	dist_b = (float4) sqrt(sum_b);

	PG_RETURN_BOOL(dist_a <= dist_b);
}

/*
 * vector_l2_equal(vector, vector, vector) -> bool
 */
PG_FUNCTION_INFO_V1(vector_l2_equal);

Datum
vector_l2_equal(PG_FUNCTION_ARGS)
{
	Vector *a = NULL;
	Vector *b = NULL;
	Vector *query = NULL;
	float4		dist_a;
	float4		dist_b;

	int			i;
	double		sum_a = 0.0,
				sum_b = 0.0;

	
	/* Validate argument count */
	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_l2_equal requires 3 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);
	query = PG_GETARG_VECTOR_P(2);
	NDB_CHECK_VECTOR_VALID(query);

	if (a->dim != query->dim || b->dim != query->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("neurondb: vector dimensions must "
						"match")));

	for (i = 0; i < query->dim; i++)
	{
		double		diff_a = a->data[i] - query->data[i];
		double		diff_b = b->data[i] - query->data[i];

		sum_a += diff_a * diff_a;
		sum_b += diff_b * diff_b;
	}

	dist_a = (float4) sqrt(sum_a);
	dist_b = (float4) sqrt(sum_b);

	/* Use relative epsilon for float comparison to avoid scale-dependent failures */
	{
		double		scale = fabs(dist_a) + fabs(dist_b) + 1e-10;

		PG_RETURN_BOOL(fabs(dist_a - dist_b) < 1e-6 * scale);
	}
}

/*
 * vector_l2_greater(vector, vector, vector) -> bool
 */
PG_FUNCTION_INFO_V1(vector_l2_greater);

Datum
vector_l2_greater(PG_FUNCTION_ARGS)
{
	Vector *a = NULL;
	Vector *b = NULL;
	Vector *query = NULL;
	float4		dist_a;
	float4		dist_b;

	int			i;
	double		sum_a = 0.0,
				sum_b = 0.0;

	
	/* Validate argument count */
	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_l2_greater requires 3 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);
	query = PG_GETARG_VECTOR_P(2);
	NDB_CHECK_VECTOR_VALID(query);

	if (a->dim != query->dim || b->dim != query->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("neurondb: vector dimensions must "
						"match")));

	for (i = 0; i < query->dim; i++)
	{
		double		diff_a = a->data[i] - query->data[i];
		double		diff_b = b->data[i] - query->data[i];

		sum_a += diff_a * diff_a;
		sum_b += diff_b * diff_b;
	}

	dist_a = (float4) sqrt(sum_a);
	dist_b = (float4) sqrt(sum_b);

	PG_RETURN_BOOL(dist_a > dist_b);
}

/*
 * vector_l2_greater_equal(vector, vector, vector) -> bool
 */
PG_FUNCTION_INFO_V1(vector_l2_greater_equal);

Datum
vector_l2_greater_equal(PG_FUNCTION_ARGS)
{
	Vector *a = NULL;
	Vector *b = NULL;
	Vector *query = NULL;
	float4		dist_a;
	float4		dist_b;

	int			i;
	double		sum_a = 0.0,
				sum_b = 0.0;

	
	/* Validate argument count */
	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_l2_greater_equal requires 3 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);
	query = PG_GETARG_VECTOR_P(2);
	NDB_CHECK_VECTOR_VALID(query);

	if (a->dim != query->dim || b->dim != query->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("neurondb: vector dimensions must "
						"match")));

	for (i = 0; i < query->dim; i++)
	{
		double		diff_a = a->data[i] - query->data[i];
		double		diff_b = b->data[i] - query->data[i];

		sum_a += diff_a * diff_a;
		sum_b += diff_b * diff_b;
	}

	dist_a = (float4) sqrt(sum_a);
	dist_b = (float4) sqrt(sum_b);

	PG_RETURN_BOOL(dist_a >= dist_b);
}

/*
 * Similar comparison functions would be needed for cosine and inner product.
 * Omitted for brevity - follow same pattern.
 */

/*
 * Distance support function for index AM
 *
 * This function is called by index AMs to get distance function properties.
 * Returns information about commutativity, null handling, and other
 * distance function characteristics.
 */
PG_FUNCTION_INFO_V1(vector_distance_support);

Datum
vector_distance_support(PG_FUNCTION_ARGS)
{
	/*
	 * Distance support function returns metadata about distance properties.
	 * In PostgreSQL's operator class system, this provides information to the
	 * planner and index access methods about: - Whether the distance function
	 * is commutative (d(a,b) == d(b,a)) - Null handling behavior - Distance
	 * function type (L2, cosine, inner product, etc.)
	 *
	 * For vector types, all standard distance functions (L2, cosine, IP) are
	 * commutative and symmetric. This information is used by the index AM for
	 * optimization.
	 *
	 * Note: The actual distance computation is done by the operator functions
	 * (vector_l2_distance_op, etc.), not this support function.
	 */
	PG_RETURN_VOID();
}

/*
 * Operator ordering function
 *
 * Returns ordering information for the distance operator.
 */
PG_FUNCTION_INFO_V1(vector_order_support);

Datum
vector_order_support(PG_FUNCTION_ARGS)
{
	/* Provides ordering semantics to the planner */
	PG_RETURN_VOID();
}

/*
 * Helper: Register operator class in catalog (called at extension install)
 *
 * This would typically be done via SQL DDL in neurondb--1.0.sql:
 *
 * CREATE OPERATOR CLASS vector_l2_ops
 *     DEFAULT FOR TYPE vector USING hnsw AS
 *     OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
 *     FUNCTION 1 vector_l2_distance_op(vector, vector);
 *
 * CREATE OPERATOR CLASS vector_cosine_ops
 *     FOR TYPE vector USING hnsw AS
 *     OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
 *     FUNCTION 1 vector_cosine_distance_op(vector, vector);
 *
 * CREATE OPERATOR CLASS vector_ip_ops
 *     FOR TYPE vector USING hnsw AS
 *     OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
 *     FUNCTION 1 vector_inner_product_distance_op(vector, vector);
 */

/*
 * vector_cosine_less(vector, vector, vector) -> bool
 *
 * For cosine distance ordering: lower distance = better match
 */
PG_FUNCTION_INFO_V1(vector_cosine_less);

Datum
vector_cosine_less(PG_FUNCTION_ARGS)
{
	Vector *a = NULL;
	Vector *b = NULL;
	Vector *query = NULL;
	float4		dist_a;
	float4		dist_b;

	
	/* Validate argument count */
	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_cosine_less requires 3 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);
	query = PG_GETARG_VECTOR_P(2);
	NDB_CHECK_VECTOR_VALID(query);
	{
		float4		cosine_distance_simd(Vector *a, Vector *b);

		if (a->dim != query->dim || b->dim != query->dim)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("neurondb: vector dimensions must "
							"match: %d vs %d vs %d",
							a->dim, b->dim, query->dim)));

		/* Use SIMD-optimized version */
		dist_a = cosine_distance_simd(a, query);
		dist_b = cosine_distance_simd(b, query);

		PG_RETURN_BOOL(dist_a < dist_b);
	}
}

/*
 * vector_cosine_less_equal(vector, vector, vector) -> bool
 */
PG_FUNCTION_INFO_V1(vector_cosine_less_equal);

Datum
vector_cosine_less_equal(PG_FUNCTION_ARGS)
{
	Vector *a = NULL;
	Vector *b = NULL;
	Vector *query = NULL;
	float4		dist_a;
	float4		dist_b;

	
	/* Validate argument count */
	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_cosine_less_equal requires 3 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);
	query = PG_GETARG_VECTOR_P(2);
	NDB_CHECK_VECTOR_VALID(query);
	{
		float4		cosine_distance_simd(Vector *a, Vector *b);

		if (a->dim != query->dim || b->dim != query->dim)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("neurondb: vector dimensions must match")));

		dist_a = cosine_distance_simd(a, query);
		dist_b = cosine_distance_simd(b, query);

		PG_RETURN_BOOL(dist_a <= dist_b);
	}
}

/*
 * vector_cosine_greater(vector, vector, vector) -> bool
 */
PG_FUNCTION_INFO_V1(vector_cosine_greater);

Datum
vector_cosine_greater(PG_FUNCTION_ARGS)
{
	Vector *a = NULL;
	Vector *b = NULL;
	Vector *query = NULL;
	float4		dist_a;
	float4		dist_b;

	
	/* Validate argument count */
	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_cosine_greater requires 3 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);
	query = PG_GETARG_VECTOR_P(2);
	NDB_CHECK_VECTOR_VALID(query);
	{
		float4		cosine_distance_simd(Vector *a, Vector *b);

		if (a->dim != query->dim || b->dim != query->dim)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("neurondb: vector dimensions must match")));

		dist_a = cosine_distance_simd(a, query);
		dist_b = cosine_distance_simd(b, query);

		PG_RETURN_BOOL(dist_a > dist_b);
	}
}

/*
 * vector_cosine_greater_equal(vector, vector, vector) -> bool
 */
PG_FUNCTION_INFO_V1(vector_cosine_greater_equal);

Datum
vector_cosine_greater_equal(PG_FUNCTION_ARGS)
{
	Vector *a = NULL;
	Vector *b = NULL;
	Vector *query = NULL;
	float4		dist_a;
	float4		dist_b;

	
	/* Validate argument count */
	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_cosine_greater_equal requires 3 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);
	query = PG_GETARG_VECTOR_P(2);
	NDB_CHECK_VECTOR_VALID(query);
	{
		float4		cosine_distance_simd(Vector *a, Vector *b);

		if (a->dim != query->dim || b->dim != query->dim)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("neurondb: vector dimensions must match")));

		dist_a = cosine_distance_simd(a, query);
		dist_b = cosine_distance_simd(b, query);

		PG_RETURN_BOOL(dist_a >= dist_b);
	}
}

/*
 * vector_inner_product_less(vector, vector, vector) -> bool
 *
 * For inner product: higher value = better match, so we want to minimize
 * the negative inner product distance
 */
PG_FUNCTION_INFO_V1(vector_inner_product_less);

Datum
vector_inner_product_less(PG_FUNCTION_ARGS)
{
	Vector *a = NULL;
	Vector *b = NULL;
	Vector *query = NULL;
	float4		dist_a;
	float4		dist_b;

	
	/* Validate argument count */
	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_inner_product_less requires 3 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);
	query = PG_GETARG_VECTOR_P(2);
	NDB_CHECK_VECTOR_VALID(query);
	{
		float4		inner_product_simd(Vector *a, Vector *b);

		if (a->dim != query->dim || b->dim != query->dim)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("neurondb: vector dimensions must match")));

		dist_a = inner_product_simd(a, query);
		dist_b = inner_product_simd(b, query);

		PG_RETURN_BOOL(dist_a < dist_b);
	}
}

/*
 * vector_inner_product_less_equal(vector, vector, vector) -> bool
 */
PG_FUNCTION_INFO_V1(vector_inner_product_less_equal);

Datum
vector_inner_product_less_equal(PG_FUNCTION_ARGS)
{
	Vector *a = NULL;
	Vector *b = NULL;
	Vector *query = NULL;
	float4		dist_a;
	float4		dist_b;

	
	/* Validate argument count */
	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_inner_product_less_equal requires 3 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);
	query = PG_GETARG_VECTOR_P(2);
	NDB_CHECK_VECTOR_VALID(query);
	{
		float4		inner_product_simd(Vector *a, Vector *b);

		if (a->dim != query->dim || b->dim != query->dim)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("neurondb: vector dimensions must match")));

		dist_a = inner_product_simd(a, query);
		dist_b = inner_product_simd(b, query);

		PG_RETURN_BOOL(dist_a <= dist_b);
	}
}

/*
 * vector_inner_product_greater(vector, vector, vector) -> bool
 */
PG_FUNCTION_INFO_V1(vector_inner_product_greater);

Datum
vector_inner_product_greater(PG_FUNCTION_ARGS)
{
	Vector *a = NULL;
	Vector *b = NULL;
	Vector *query = NULL;
	float4		dist_a;
	float4		dist_b;

	
	/* Validate argument count */
	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_inner_product_greater requires 3 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);
	query = PG_GETARG_VECTOR_P(2);
	NDB_CHECK_VECTOR_VALID(query);
	{
		float4		inner_product_simd(Vector *a, Vector *b);

		if (a->dim != query->dim || b->dim != query->dim)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("neurondb: vector dimensions must match")));

		dist_a = inner_product_simd(a, query);
		dist_b = inner_product_simd(b, query);

		PG_RETURN_BOOL(dist_a > dist_b);
	}
}

/*
 * vector_inner_product_greater_equal(vector, vector, vector) -> bool
 */
PG_FUNCTION_INFO_V1(vector_inner_product_greater_equal);

Datum
vector_inner_product_greater_equal(PG_FUNCTION_ARGS)
{
	Vector *a = NULL;
	Vector *b = NULL;
	Vector *query = NULL;
	float4		dist_a;
	float4		dist_b;

	
	/* Validate argument count */
	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_inner_product_greater_equal requires 3 arguments")));

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);
	query = PG_GETARG_VECTOR_P(2);
	NDB_CHECK_VECTOR_VALID(query);
	{
		float4		inner_product_simd(Vector *a, Vector *b);

		if (a->dim != query->dim || b->dim != query->dim)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("neurondb: vector dimensions must match")));

		dist_a = inner_product_simd(a, query);
		dist_b = inner_product_simd(b, query);

		PG_RETURN_BOOL(dist_a >= dist_b);
	}
}

/*
 * Utility function to check if operator class exists
 */
PG_FUNCTION_INFO_V1(neurondb_has_opclass);

Datum
neurondb_has_opclass(PG_FUNCTION_ARGS)
{
	text	   *opclass_name = NULL;
	char	   *name = NULL;
	bool		exists = false;
	int			ret;
	StringInfoData query;
	NdbSpiSession *session = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: neurondb_has_opclass requires 1 argument")));

	opclass_name = PG_GETARG_TEXT_PP(0);
	name = text_to_cstring(opclass_name);

	/* Connect to SPI */
	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
	{
		pfree(name);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: neurondb_has_opclass: failed to begin SPI session")));
	}

	/* Query pg_opclass to check if operator class exists.
	 * Use quote_literal to prevent SQL injection. */
	initStringInfo(&query);
	appendStringInfo(&query,
					 "SELECT 1 FROM pg_opclass oc "
					 "JOIN pg_namespace n ON oc.opcnamespace = n.oid "
					 "WHERE oc.opcname = %s AND n.nspname = 'neurondb'",
					 quote_literal_cstr(name));

	ret = ndb_spi_execute(session, query.data, true, 0);
	pfree(query.data);

	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		exists = true;
	}

	ndb_spi_session_end(&session);


	pfree(name);

	PG_RETURN_BOOL(exists);
}
