/*-------------------------------------------------------------------------
 *
 * ml_quantization_fp8.c
 *    FP8 quantization implementation.
 *
 * This module implements INT4 and FP8 quantization formats for efficient
 * vector storage and computation.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_quantization_fp8.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "neurondb.h"
#include "neurondb_types.h"
#include "neurondb_gpu.h"
#include <math.h>
#include <string.h>
#include <stdint.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/* Forward declaration */
extern VectorI4 * quantize_vector_int4(Vector *v);

/*
 * FP8 E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits
 * Range: approximately [-448, 448]
 */
typedef uint8_t fp8_e4m3;

/*
 * FP8 E5M2 format: 1 sign bit, 5 exponent bits, 2 mantissa bits
 * Range: approximately [-57344, 57344]
 */
typedef uint8_t fp8_e5m2;

/*
 * FP8Vector: FP8 quantized vector
 */
typedef struct FP8Vector
{
	int32		vl_len_;
	int16		dim;
	uint8		format;			/* 0 = E4M3, 1 = E5M2 */
	uint8		unused;
	uint8		data[FLEXIBLE_ARRAY_MEMBER];
}			FP8Vector;

#define FP8_VEC_SIZE(dim) \
	(offsetof(FP8Vector, data) + sizeof(uint8) * (dim))

/*
 * float_to_fp8_e4m3: Convert float32 to FP8 E4M3
 */
static uint8_t
float_to_fp8_e4m3(float val)
{
	uint32_t	bits;
	uint8_t		sign,
				mant;
	int			exp;
	uint8_t		result;

	if (val == 0.0f)
		return 0;

	memcpy(&bits, &val, sizeof(float));
	sign = (bits >> 31) & 0x1;
	exp = ((bits >> 23) & 0xFF) - 127;	/* FP32 exponent bias */
	mant = (bits >> 20) & 0x7;	/* Top 3 mantissa bits */

	/* E4M3: 4 exponent bits (bias 7), 3 mantissa bits */
	if (exp > 7)
	{
		/* Overflow: return max value */
		result = (sign << 7) | 0x7F;
	}
	else if (exp < -6)
	{
		/* Underflow: return zero */
		result = 0;
	}
	else
	{
		exp = exp + 7;			/* E4M3 bias */
		result = (sign << 7) | ((exp & 0xF) << 3) | (mant & 0x7);
	}

	return result;
}

/*
 * fp8_e4m3_to_float: Convert FP8 E4M3 to float32
 */
static float
fp8_e4m3_to_float(uint8_t fp8)
{
	uint8_t		sign = (fp8 >> 7) & 0x1;
	uint8_t		exp = (fp8 >> 3) & 0xF;
	uint8_t		mant = fp8 & 0x7;
	uint32_t	bits;
	float		result;

	if (exp == 0)
	{
		/* Zero or denormal */
		result = 0.0f;
	}
	else
	{
		exp = exp - 7;			/* Remove E4M3 bias */
		bits = ((uint32_t) sign << 31) | ((uint32_t) (exp + 127) << 23) |
			((uint32_t) mant << 20);
		memcpy(&result, &bits, sizeof(float));
	}

	return result;
}

/*
 * float_to_fp8_e5m2: Convert float32 to FP8 E5M2
 */
static uint8_t
float_to_fp8_e5m2(float val)
{
	uint32_t	bits;
	uint8_t		sign,
				mant;
	int			exp;
	uint8_t		result;

	if (val == 0.0f)
		return 0;

	memcpy(&bits, &val, sizeof(float));
	sign = (bits >> 31) & 0x1;
	exp = ((bits >> 23) & 0xFF) - 127;	/* FP32 exponent bias */
	mant = (bits >> 21) & 0x3;	/* Top 2 mantissa bits */

	/* E5M2: 5 exponent bits (bias 15), 2 mantissa bits */
	if (exp > 15)
	{
		/* Overflow: return max value */
		result = (sign << 7) | 0x7F;
	}
	else if (exp < -14)
	{
		/* Underflow: return zero */
		result = 0;
	}
	else
	{
		exp = exp + 15;			/* E5M2 bias */
		result = (sign << 7) | ((exp & 0x1F) << 2) | (mant & 0x3);
	}

	return result;
}

/*
 * fp8_e5m2_to_float: Convert FP8 E5M2 to float32
 */
static float
fp8_e5m2_to_float(uint8_t fp8)
{
	uint8_t		sign = (fp8 >> 7) & 0x1;
	uint8_t		exp = (fp8 >> 2) & 0x1F;
	uint8_t		mant = fp8 & 0x3;
	uint32_t	bits;
	float		result;

	if (exp == 0)
	{
		/* Zero or denormal */
		result = 0.0f;
	}
	else
	{
		exp = exp - 15;			/* Remove E5M2 bias */
		bits = ((uint32_t) sign << 31) | ((uint32_t) (exp + 127) << 23) |
			((uint32_t) mant << 21);
		memcpy(&result, &bits, sizeof(float));
	}

	return result;
}

/*
 * quantize_vector_fp8: Quantize vector to FP8 (with GPU support)
 */
static FP8Vector *
quantize_vector_fp8(Vector *v, uint8_t format)
{
	FP8Vector  *result;
	int			size;
	int			i;

	size = FP8_VEC_SIZE(v->dim);
	result = (FP8Vector *) palloc0(size);
	SET_VARSIZE(result, size);
	result->dim = v->dim;
	result->format = format;

	/* Try GPU first if available */
	if (neurondb_gpu_is_available())
	{
		if (format == 0)
			neurondb_gpu_quantize_fp8_e4m3(v->data, result->data, v->dim);
		else
			neurondb_gpu_quantize_fp8_e5m2(v->data, result->data, v->dim);
		return result;
	}

	/* CPU fallback */
	for (i = 0; i < v->dim; i++)
	{
		if (format == 0)
			result->data[i] = float_to_fp8_e4m3(v->data[i]);
		else
			result->data[i] = float_to_fp8_e5m2(v->data[i]);
	}

	return result;
}

/*
 * dequantize_fp8_vector: Dequantize FP8 vector back to float32
 */
static Vector *
dequantize_fp8_vector(FP8Vector * fp8_vec)
{
	Vector	   *result;
	int			i;

	result = new_vector(fp8_vec->dim);

	for (i = 0; i < fp8_vec->dim; i++)
	{
		if (fp8_vec->format == 0)
			result->data[i] = fp8_e4m3_to_float(fp8_vec->data[i]);
		else
			result->data[i] = fp8_e5m2_to_float(fp8_vec->data[i]);
	}

	return result;
}

/*
 * quantize_fp8_e4m3: SQL function to quantize to FP8 E4M3
 */
PG_FUNCTION_INFO_V1(quantize_fp8_e4m3);
Datum
quantize_fp8_e4m3(PG_FUNCTION_ARGS)
{
	FP8Vector  *result;
	Vector	   *v = PG_GETARG_VECTOR_P(0);

	NDB_CHECK_VECTOR_VALID(v);

	result = quantize_vector_fp8(v, 0); /* E4M3 format */
	PG_RETURN_POINTER(result);
}

/*
 * quantize_fp8_e5m2: SQL function to quantize to FP8 E5M2
 */
PG_FUNCTION_INFO_V1(quantize_fp8_e5m2);
Datum
quantize_fp8_e5m2(PG_FUNCTION_ARGS)
{
	FP8Vector  *result;
	Vector	   *v = PG_GETARG_VECTOR_P(0);

	NDB_CHECK_VECTOR_VALID(v);

	result = quantize_vector_fp8(v, 1); /* E5M2 format */
	PG_RETURN_POINTER(result);
}

/*
 * dequantize_fp8: SQL function to dequantize FP8 vector
 */
PG_FUNCTION_INFO_V1(dequantize_fp8);
Datum
dequantize_fp8(PG_FUNCTION_ARGS)
{
	FP8Vector  *fp8_vec = (FP8Vector *) PG_GETARG_POINTER(0);
	Vector	   *result;

	result = dequantize_fp8_vector(fp8_vec);
	PG_RETURN_POINTER(result);
}

/*
 * auto_quantize: Automatically select best quantization based on accuracy/size tradeoff
 */
PG_FUNCTION_INFO_V1(auto_quantize);
Datum
auto_quantize(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	text	   *target_compression;

	char	   *compression_str;
	VectorI4   *result_int4;
	FP8Vector  *result_fp8;
	Datum		result;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);
	target_compression = PG_GETARG_TEXT_P(1);
	compression_str = text_to_cstring(target_compression);

	/* Select quantization based on target compression ratio */
	if (pg_strcasecmp(compression_str, "int4") == 0 ||
		pg_strcasecmp(compression_str, "4bit") == 0)
	{
		result_int4 = quantize_vector_int4(v);
		result = PointerGetDatum(result_int4);
	}
	else if (pg_strcasecmp(compression_str, "fp8_e4m3") == 0)
	{
		result_fp8 = quantize_vector_fp8(v, 0);
		result = PointerGetDatum(result_fp8);
	}
	else if (pg_strcasecmp(compression_str, "fp8_e5m2") == 0)
	{
		result_fp8 = quantize_vector_fp8(v, 1);
		result = PointerGetDatum(result_fp8);
	}
	else
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("auto_quantize: unsupported compression type: %s",
						compression_str),
				 errhint("Supported: int4, fp8_e4m3, fp8_e5m2")));
		result = (Datum) 0;		/* Not reached */
	}

	NDB_FREE(compression_str);
	PG_RETURN_DATUM(result);
}
