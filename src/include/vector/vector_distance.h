/*-------------------------------------------------------------------------
 *
 * vector_distance.h
 *	  Vector distance metric implementations
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 * SPDX-License-Identifier: PostgreSQL
 *
 *-------------------------------------------------------------------------
 */
#ifndef VECTOR_DISTANCE_H
#define VECTOR_DISTANCE_H

#include "postgres.h"
#include "fmgr.h"

/* Distance metric functions */
extern Datum vector_l2_distance(PG_FUNCTION_ARGS);
extern Datum vector_l2_distance_op(PG_FUNCTION_ARGS);
extern Datum vector_l2_squared_distance(PG_FUNCTION_ARGS);
extern Datum vector_inner_product(PG_FUNCTION_ARGS);
extern Datum vector_inner_product_distance_op(PG_FUNCTION_ARGS);
extern Datum vector_negative_inner_product(PG_FUNCTION_ARGS);
extern Datum vector_cosine_distance(PG_FUNCTION_ARGS);
extern Datum vector_cosine_distance_op(PG_FUNCTION_ARGS);
extern Datum vector_l1_distance(PG_FUNCTION_ARGS);
extern Datum vector_hamming_distance(PG_FUNCTION_ARGS);
extern Datum vector_chebyshev_distance(PG_FUNCTION_ARGS);
extern Datum vector_minkowski_distance(PG_FUNCTION_ARGS);
extern Datum vector_spherical_distance(PG_FUNCTION_ARGS);

/* Internal distance functions for index optimization */
extern float4 spherical_distance(Vector *a, Vector *b);

/* GPU-accelerated distance functions */
extern Datum vector_l2_distance_gpu(PG_FUNCTION_ARGS);
extern Datum vector_cosine_distance_gpu(PG_FUNCTION_ARGS);
extern Datum vector_inner_product_gpu(PG_FUNCTION_ARGS);

/* Correlation-based distance functions */
extern Datum vector_pearson_correlation(PG_FUNCTION_ARGS);
extern Datum vector_weighted_distance(PG_FUNCTION_ARGS);

/* Information-theoretic distance functions */
extern Datum vector_kl_divergence(PG_FUNCTION_ARGS);
extern Datum vector_js_divergence(PG_FUNCTION_ARGS);

/* Operator class comparison functions */
extern Datum vector_l2_less(PG_FUNCTION_ARGS);
extern Datum vector_l2_less_equal(PG_FUNCTION_ARGS);
extern Datum vector_l2_greater(PG_FUNCTION_ARGS);
extern Datum vector_l2_greater_equal(PG_FUNCTION_ARGS);
extern Datum vector_l2_equal(PG_FUNCTION_ARGS);

#endif							/* VECTOR_DISTANCE_H */
