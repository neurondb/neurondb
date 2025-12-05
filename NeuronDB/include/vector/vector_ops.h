/*-------------------------------------------------------------------------
 *
 * vector_ops.h
 *	  Vector operations (math, normalization, conversion)
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 * SPDX-License-Identifier: PostgreSQL
 *
 *-------------------------------------------------------------------------
 */
#ifndef VECTOR_OPS_H
#define VECTOR_OPS_H

#include "postgres.h"
#include "fmgr.h"

/* Vector utility functions */
extern Datum vector_dims(PG_FUNCTION_ARGS);
extern Datum vector_norm(PG_FUNCTION_ARGS);
extern Datum vector_normalize(PG_FUNCTION_ARGS);

/* Vector arithmetic operations */
extern Datum vector_add(PG_FUNCTION_ARGS);
extern Datum vector_sub(PG_FUNCTION_ARGS);
extern Datum vector_mul(PG_FUNCTION_ARGS);
extern Datum vector_concat(PG_FUNCTION_ARGS);

/* Vector conversion functions */
extern Datum array_to_vector(PG_FUNCTION_ARGS);
extern Datum vector_to_array(PG_FUNCTION_ARGS);
extern Datum vector_to_int8(PG_FUNCTION_ARGS);
extern Datum int8_to_vector(PG_FUNCTION_ARGS);
extern Datum vector_to_binary(PG_FUNCTION_ARGS);

/* GPU-accelerated conversions */
extern Datum vector_to_int8_gpu(PG_FUNCTION_ARGS);
extern Datum vector_to_fp16_gpu(PG_FUNCTION_ARGS);
extern Datum vector_to_binary_gpu(PG_FUNCTION_ARGS);

#endif							/* VECTOR_OPS_H */
