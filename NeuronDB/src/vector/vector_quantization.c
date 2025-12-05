/*-------------------------------------------------------------------------
 *
 * vector_quantization.c
 *		Vector quantization functions (FP16, INT8)
 *
 * Implements quantization and dequantization for storage efficiency.
 * FP16 provides 2x compression, INT8 provides 4x compression.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/vector/vector_quantization.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/memutils.h"
#include <math.h>
#include <stdint.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/* FP16 conversion utilities */
static inline uint16_t
float_to_fp16(float f)
{
	uint32_t	bits = *(uint32_t *) & f;
	uint16_t	sign = (bits >> 16) & 0x8000;
	uint32_t	exp = (bits >> 23) & 0xFF;
	uint32_t	mantissa = bits & 0x7FFFFF;
	int			new_exp;

	if (exp == 0xFF)
	{
		/* Infinity or NaN */
		return sign | 0x7C00 | (mantissa ? 0x200 : 0);
	}

	if (exp == 0)
	{
		/* Denormalized */
		return sign;
	}

	/* Normalized */
	new_exp = exp - 127 + 15;
	if (new_exp <= 0)
		return sign;
	if (new_exp >= 31)
		return sign | 0x7C00;	/* Infinity */

	return sign | (new_exp << 10) | (mantissa >> 13);
}

static inline float
fp16_to_float(uint16_t fp16)
{
	uint16_t	sign = fp16 & 0x8000;
	uint16_t	exp = (fp16 >> 10) & 0x1F;
	uint16_t	mantissa = fp16 & 0x3FF;
	uint32_t	bits;

	if (exp == 0)
	{
		if (mantissa == 0)
			bits = sign << 16;
		else
		{
			/* Denormalized */
			bits = (sign << 16) | ((mantissa << 13) >> 23);
		}
	}
	else if (exp == 31)
	{
		/* Infinity or NaN */
		bits = (sign << 16) | 0x7F800000 | (mantissa << 13);
	}
	else
	{
		/* Normalized */
		int			new_exp = exp - 15 + 127;

		bits = (sign << 16) | (new_exp << 23) | (mantissa << 13);
	}

	return *(float *) &bits;
}

/*
 * vector_quantize_fp16
 *
 * Quantize vector to FP16 format (2x compression).
 */
PG_FUNCTION_INFO_V1(vector_quantize_fp16);
Datum
vector_quantize_fp16(PG_FUNCTION_ARGS)
{
	Vector	   *vec;
	bytea	   *result;
	uint16_t   *fp16_data;
	int			i;
	size_t		size;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_quantize_fp16 requires 1 argument, got %d",
						PG_NARGS())));

	vec = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(vec);

	if (vec == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("vector must not be NULL")));

	if (vec->dim <= 0 || vec->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension: %d",
						vec->dim)));

	size = sizeof(int16) + sizeof(uint16_t) * vec->dim;
	if (size > MaxAllocSize)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("vector size exceeds maximum allocation")));

	result = (bytea *) palloc(VARHDRSZ + size);
	if (result == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("out of memory")));
	SET_VARSIZE(result, VARHDRSZ + size);

	/* Store dimension */
	*(int16 *) VARDATA(result) = vec->dim;

	/* Convert to FP16 */
	fp16_data = (uint16_t *) (VARDATA(result) + sizeof(int16));
	for (i = 0; i < vec->dim; i++)
		fp16_data[i] = float_to_fp16(vec->data[i]);

	PG_RETURN_BYTEA_P(result);
}

/*
 * vector_dequantize_fp16
 *
 * Dequantize FP16 vector back to FP32.
 */
PG_FUNCTION_INFO_V1(vector_dequantize_fp16);
Datum
vector_dequantize_fp16(PG_FUNCTION_ARGS)
{
	bytea	   *fp16_vec;
	Vector	   *result;
	uint16_t   *fp16_data;
	int			dim;
	int			i;
	size_t		expected_size;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_dequantize_fp16 requires 1 argument, got %d",
						PG_NARGS())));

	fp16_vec = PG_GETARG_BYTEA_P(0);

	if (fp16_vec == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("FP16 vector must not be NULL")));

	if (VARSIZE(fp16_vec) < VARHDRSZ + sizeof(int16))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_BINARY_REPRESENTATION),
				 errmsg("invalid FP16 vector size")));

	dim = *(int16 *) VARDATA(fp16_vec);
	if (dim <= 0 || dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_BINARY_REPRESENTATION),
				 errmsg("invalid FP16 vector dimension: %d",
						dim)));

	expected_size = VARHDRSZ + sizeof(int16) + sizeof(uint16_t) * dim;
	if (VARSIZE(fp16_vec) < expected_size)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_BINARY_REPRESENTATION),
				 errmsg("FP16 vector size mismatch: expected %zu, got %d",
						expected_size,
						VARSIZE(fp16_vec))));

	fp16_data = (uint16_t *) (VARDATA(fp16_vec) + sizeof(int16));
	if (fp16_data == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_BINARY_REPRESENTATION),
				 errmsg("invalid FP16 vector data")));

	result = new_vector(dim);
	if (result == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("out of memory")));

	for (i = 0; i < dim; i++)
		result->data[i] = fp16_to_float(fp16_data[i]);

	PG_RETURN_VECTOR_P(result);
}

/*
 * vector_quantize_int8
 *
 * Quantize vector to INT8 format (4x compression).
 * Requires min, max, and scale vectors for calibration.
 */
PG_FUNCTION_INFO_V1(vector_quantize_int8);
Datum
vector_quantize_int8(PG_FUNCTION_ARGS)
{
	Vector	   *vec;
	Vector	   *min_vec;
	Vector	   *max_vec;
	bytea	   *result;
	int8	   *int8_data;
	int			i;
	size_t		size;
	float		scale;
	float		normalized;
	float		range;

	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_quantize_int8 requires 3 arguments, got %d",
						PG_NARGS())));

	vec = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(vec);
	min_vec = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(min_vec);
	max_vec = PG_GETARG_VECTOR_P(2);
	NDB_CHECK_VECTOR_VALID(max_vec);

	if (vec == NULL || min_vec == NULL || max_vec == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("vectors must not be NULL")));

	if (vec->dim <= 0 || vec->dim > VECTOR_MAX_DIM ||
		min_vec->dim <= 0 || min_vec->dim > VECTOR_MAX_DIM ||
		max_vec->dim <= 0 || max_vec->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension")));

	if (vec->dim != min_vec->dim || vec->dim != max_vec->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector dimensions must match")));

	size = sizeof(int16) + sizeof(int8) * vec->dim;
	if (size > MaxAllocSize)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("vector size exceeds maximum allocation")));

	result = (bytea *) palloc(VARHDRSZ + size);
	if (result == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("out of memory")));
	SET_VARSIZE(result, VARHDRSZ + size);

	*(int16 *) VARDATA(result) = vec->dim;
	int8_data = (int8 *) (VARDATA(result) + sizeof(int16));

	for (i = 0; i < vec->dim; i++)
	{
		range = max_vec->data[i] - min_vec->data[i];
		if (range <= 0.0f)
		{
			int8_data[i] = 0;
			continue;
		}

		scale = 127.0f / range;
		normalized = (vec->data[i] - min_vec->data[i]) * scale;
		if (normalized > 127.0f)
			int8_data[i] = 127;
		else if (normalized < -128.0f)
			int8_data[i] = -128;
		else
			int8_data[i] = (int8) roundf(normalized);
	}

	PG_RETURN_BYTEA_P(result);
}

/*
 * vector_dequantize_int8
 *
 * Dequantize INT8 vector back to FP32.
 */
PG_FUNCTION_INFO_V1(vector_dequantize_int8);
Datum
vector_dequantize_int8(PG_FUNCTION_ARGS)
{
	bytea	   *int8_vec;
	Vector	   *min_vec;
	Vector	   *max_vec;
	Vector	   *result;
	int8	   *int8_data;
	int			dim;
	int			i;
	size_t		expected_size;
	float		range;
	float		scale;

	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_dequantize_int8 requires 3 arguments, got %d",
						PG_NARGS())));

	int8_vec = PG_GETARG_BYTEA_P(0);
	min_vec = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(min_vec);
	max_vec = PG_GETARG_VECTOR_P(2);
	NDB_CHECK_VECTOR_VALID(max_vec);

	if (int8_vec == NULL || min_vec == NULL || max_vec == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("vectors must not be NULL")));

	if (VARSIZE(int8_vec) < VARHDRSZ + sizeof(int16))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_BINARY_REPRESENTATION),
				 errmsg("invalid INT8 vector size")));

	dim = *(int16 *) VARDATA(int8_vec);
	if (dim <= 0 || dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_BINARY_REPRESENTATION),
				 errmsg("invalid INT8 vector dimension: %d",
						dim)));

	if (min_vec->dim <= 0 || min_vec->dim > VECTOR_MAX_DIM ||
		max_vec->dim <= 0 || max_vec->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid min/max vector dimension")));

	if (dim != min_vec->dim || dim != max_vec->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector dimensions must match")));

	expected_size = VARHDRSZ + sizeof(int16) + sizeof(int8) * dim;
	if (VARSIZE(int8_vec) < expected_size)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_BINARY_REPRESENTATION),
				 errmsg("INT8 vector size mismatch")));

	int8_data = (int8 *) (VARDATA(int8_vec) + sizeof(int16));
	if (int8_data == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_BINARY_REPRESENTATION),
				 errmsg("invalid INT8 vector data")));

	result = new_vector(dim);
	if (result == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("out of memory")));

	for (i = 0; i < dim; i++)
	{
		range = max_vec->data[i] - min_vec->data[i];
		if (range <= 0.0f)
		{
			result->data[i] = min_vec->data[i];
		}
		else
		{
			scale = range / 127.0f;
			result->data[i] = min_vec->data[i] + (float) int8_data[i] * scale;
		}
	}

	PG_RETURN_VECTOR_P(result);
}

/*
 * vector_l2_distance_fp16
 *
 * Compute L2 distance between two FP16 quantized vectors.
 */
PG_FUNCTION_INFO_V1(vector_l2_distance_fp16);
Datum
vector_l2_distance_fp16(PG_FUNCTION_ARGS)
{
	bytea	   *fp16_a;
	bytea	   *fp16_b;
	Vector	   *a;
	Vector	   *b;
	float4		result;
	extern float4 l2_distance(Vector *a, Vector *b);

	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_l2_distance_fp16 requires 2 arguments, got %d",
						PG_NARGS())));

	fp16_a = PG_GETARG_BYTEA_P(0);
	fp16_b = PG_GETARG_BYTEA_P(1);

	if (fp16_a == NULL || fp16_b == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("FP16 vectors must not be NULL")));

	/* Dequantize both vectors */
	a = DatumGetVector(DirectFunctionCall1(vector_dequantize_fp16,
										   PointerGetDatum(fp16_a)));
	b = DatumGetVector(DirectFunctionCall1(vector_dequantize_fp16,
										   PointerGetDatum(fp16_b)));

	if (a == NULL || b == NULL)
	{
		if (a != NULL)
			NDB_FREE(a);
		if (b != NULL)
			NDB_FREE(b);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_BINARY_REPRESENTATION),
				 errmsg("failed to dequantize FP16 vectors")));
	}

	/* Compute distance */
	result = l2_distance(a, b);

	/* Free temporary vectors */
	NDB_FREE(a);
	NDB_FREE(b);

	PG_RETURN_FLOAT4(result);
}

/*
 * vector_cosine_distance_fp16
 *
 * Compute cosine distance between two FP16 quantized vectors.
 */
PG_FUNCTION_INFO_V1(vector_cosine_distance_fp16);
Datum
vector_cosine_distance_fp16(PG_FUNCTION_ARGS)
{
	bytea	   *fp16_a;
	bytea	   *fp16_b;
	Vector	   *a;
	Vector	   *b;
	float4		result;
	extern float4 cosine_distance(Vector *a, Vector *b);

	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_cosine_distance_fp16 requires 2 arguments, got %d",
						PG_NARGS())));

	fp16_a = PG_GETARG_BYTEA_P(0);
	fp16_b = PG_GETARG_BYTEA_P(1);

	if (fp16_a == NULL || fp16_b == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("FP16 vectors must not be NULL")));

	/* Dequantize both vectors */
	a = DatumGetVector(DirectFunctionCall1(vector_dequantize_fp16,
										   PointerGetDatum(fp16_a)));
	b = DatumGetVector(DirectFunctionCall1(vector_dequantize_fp16,
										   PointerGetDatum(fp16_b)));

	/* Compute distance */
	result = cosine_distance(a, b);

	/* Free temporary vectors */
	NDB_FREE(a);
	NDB_FREE(b);

	PG_RETURN_FLOAT4(result);
}

/*
 * vector_quantize_binary
 *
 * Quantize vector to binary format (32x compression).
 * Each float value is converted to a single bit: positive = 1, zero/negative = 0.
 * Returns a BinaryVec (packed bits).
 */
PG_FUNCTION_INFO_V1(vector_quantize_binary);
Datum
vector_quantize_binary(PG_FUNCTION_ARGS)
{
	Vector	   *vec;
	typedef struct BinaryVec
	{
		int32		vl_len_;
		int32		dim;
		uint8		data[FLEXIBLE_ARRAY_MEMBER];
	}			BinaryVec;
	BinaryVec  *result;
	int			i;
	int			byte_index;
	int			bit_index;
	size_t		size;

	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_quantize_binary requires 1 argument, got %d",
						PG_NARGS())));

	vec = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(vec);

	if (vec == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("vector must not be NULL")));

	if (vec->dim <= 0 || vec->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension: %d",
						vec->dim)));

	/* Calculate size: header + dimension + packed bits */
	size = offsetof(BinaryVec, data) + ((vec->dim + 7) / 8);
	if (size > MaxAllocSize)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("vector size exceeds maximum allocation")));

	/*
	 * Variable-length type requires custom size - use palloc0 but ensure
	 * proper cleanup
	 */
	/* Note: For variable-length PostgreSQL types, palloc0 is standard */
	result = (BinaryVec *) palloc0(size);
	if (result == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("out of memory")));
	SET_VARSIZE(result, size);
	result->dim = vec->dim;

	/* Convert each float to a bit: positive = 1, zero/negative = 0 */
	for (i = 0; i < vec->dim; i++)
	{
		if (vec->data[i] > 0.0f)
		{
			byte_index = i / 8;
			bit_index = i % 8;
			result->data[byte_index] |= (1 << bit_index);
		}
	}

	PG_RETURN_POINTER(result);
}
