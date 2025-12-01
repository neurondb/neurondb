/*-------------------------------------------------------------------------
 *
 * quantization.c
 *	  Vector quantization for memory efficiency and performance
 *
 * This file implements multiple quantization schemes including INT8
 * quantization (8x compression), float16 quantization (2x compression),
 * and binary quantization (32x compression). Includes optimized
 * Hamming distance for binary vectors.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/types/quantization.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_gpu.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "utils/varbit.h"
#include "lib/stringinfo.h"
#include "libpq/pqformat.h"
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/* Forward declaration */
PGDLLEXPORT float fp16_to_float(uint16 h);

/* PG_MODULE_MAGIC already defined in neurondb.c */

/* Helper function: float32 -> int8 quantization (max-abs scaling) */
VectorI8 *
quantize_vector_i8(Vector *v)
{
	int			i;
	VectorI8   *result = NULL;
	float4		max_abs;
	float4		scale;
	int			size;

	/* Find maximum absolute value */
	max_abs = 0.0f;
	for (i = 0; i < v->dim; i++)
	{
		float4		abs_val = fabsf(v->data[i]);

		if (abs_val > max_abs)
			max_abs = abs_val;
	}

	/* Allocate result */
	size = offsetof(VectorI8, data) + sizeof(int8) * v->dim;
	result = (VectorI8 *) palloc0(size);
	SET_VARSIZE(result, size);
	result->dim = v->dim;

	/* Compute scale */
	if (max_abs == 0.0f)
		return result;			/* All zeros - return zero vector */

	scale = 127.0f / max_abs;

	/* Quantize */
	for (i = 0; i < v->dim; i++)
	{
		float4		val = v->data[i] * scale;

		if (val > 127.0f)
			val = 127.0f;
		if (val < -128.0f)
			val = -128.0f;
		result->data[i] = (int8) rintf(val);	/* round to nearest int8 */
	}

	return result;
}

/*
 * SQL interface: float32 vector -> int8 quantized vector
 */
PG_FUNCTION_INFO_V1(vector_to_int8);
Datum
vector_to_int8(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	VectorI8   *result;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	result = quantize_vector_i8(v);
	PG_RETURN_POINTER(result);
}

/*
 * SQL interface: int8 quantized vector -> float32 vector (dequantize)
 * NOTE: This is a simplified dequantization that doesn't restore exact values.
 * For accurate dequantization, use quantize_analyze_int8 which stores the scale.
 */
PG_FUNCTION_INFO_V1(int8_to_vector);
Datum
int8_to_vector(PG_FUNCTION_ARGS)
{
	VectorI8   *v8;
	Vector	   *result;
	int			i;

	v8 = (VectorI8 *) PG_GETARG_POINTER(0);
	result = new_vector(v8->dim);
	for (i = 0; i < v8->dim; i++)
		result->data[i] = (float4) v8->data[i] / 127.0f;

	PG_RETURN_VECTOR_P(result);
}

/*
 * Helper: Convert float32 to float16 (FP16)
 */
static uint16
float4_to_fp16(float4 f)
{
	uint32		u;
	uint16		sign;
	uint32		mantissa;
	int16		exp;

	memcpy(&u, &f, sizeof(uint32));
	sign = (u >> 16) & 0x8000;
	mantissa = u & 0x7fffff;
	exp = ((u >> 23) & 0xff) - 127 + 15;	/* bias change */

	if (exp <= 0)
	{
		/* flush to zero */
		return sign;
	}
	else if (exp >= 31)
	{
		/* inf/NaN */
		return sign | 0x7c00;
	}
	else
	{
		return sign | (exp << 10) | (mantissa >> 13);
	}
}

PGDLLEXPORT float
fp16_to_float(uint16 h)
{
	uint32		sign = (h & 0x8000) << 16;
	uint32		exp = (h & 0x7c00) >> 10;
	uint32		mantissa = h & 0x03ff;
	uint32		f;

	if (exp == 0)
	{
		if (mantissa == 0)
			f = sign;
		else
		{
			/* subnormal */
			uint32		m = mantissa;
			uint32		exponent;

			exp = 1;

			while ((m & 0x0400) == 0)
			{
				m <<= 1;
				exp--;
			}
			m &= 0x03ff;
			exponent = 127 - 15 - (10 - exp);
			f = sign | (exponent << 23) | (m << 13);
		}
	}
	else if (exp == 0x1f)
	{
		/* inf/NaN */
		f = sign | 0x7f800000 | (mantissa << 13);
	}
	else
	{
		uint32		exponent = exp + 127 - 15;

		f = sign | (exponent << 23) | (mantissa << 13);
	}

	{
		float		ret;

		memcpy(&ret, &f, 4);
		return ret;
	}
}

VectorF16 *
quantize_vector_f16(Vector *v)
{
	VectorF16  *result;
	int			size;
	int			i;

	size = offsetof(VectorF16, data) + sizeof(uint16) * v->dim;
	result = (VectorF16 *) palloc0(size);
	SET_VARSIZE(result, size);
	result->dim = v->dim;

	for (i = 0; i < v->dim; i++)
		result->data[i] = float4_to_fp16(v->data[i]);

	return result;
}

PG_FUNCTION_INFO_V1(vector_to_float16);
Datum
vector_to_float16(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	VectorF16  *result;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	result = quantize_vector_f16(v);
	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(float16_to_vector);
Datum
float16_to_vector(PG_FUNCTION_ARGS)
{
	VectorF16  *vf16 = (VectorF16 *) PG_GETARG_POINTER(0);
	Vector	   *result;
	int			i;

	result = new_vector(vf16->dim);
	for (i = 0; i < vf16->dim; i++)
		result->data[i] = fp16_to_float(vf16->data[i]);

	PG_RETURN_VECTOR_P(result);
}

/*
 * Helper function: float32 -> binary quantization
 */
VectorBinary *
quantize_vector_binary(Vector *v)
{
	VectorBinary *result;
	int			nbytes;
	int			size;
	int			i;
	int			byte_idx;
	int			bit_idx;

	nbytes = (v->dim + 7) / 8;
	size = offsetof(VectorBinary, data) + nbytes;
	result = (VectorBinary *) palloc0(size);
	SET_VARSIZE(result, size);
	result->dim = v->dim;
	memset(result->data, 0, nbytes);

	for (i = 0; i < v->dim; i++)
	{
		if (v->data[i] > 0.0f)
		{
			byte_idx = i / 8;
			bit_idx = i % 8;
			result->data[byte_idx] |= (1 << bit_idx);
		}
	}

	return result;
}

/*
 * Binary quantization: maps each component to a bit (sign > 0)
 */
PG_FUNCTION_INFO_V1(vector_to_binary);
Datum
vector_to_binary(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	VectorBinary *result;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	result = quantize_vector_binary(v);
	PG_RETURN_POINTER(result);
}

/*
 * binary_quantize: Alias for vector_to_binary (pgvector compatibility)
 */
PG_FUNCTION_INFO_V1(binary_quantize);
Datum
binary_quantize(PG_FUNCTION_ARGS)
{
	return vector_to_binary(fcinfo);
}

/*
 * Binary to float32: sign decoding (+1.0 or -1.0)
 */
PG_FUNCTION_INFO_V1(binary_to_vector);
Datum
binary_to_vector(PG_FUNCTION_ARGS)
{
	VectorBinary *vb = (VectorBinary *) PG_GETARG_POINTER(0);
	Vector	   *result;
	int			i;
	int			byte_idx;
	int			bit_idx;

	result = new_vector(vb->dim);

	for (i = 0; i < vb->dim; i++)
	{
		byte_idx = i / 8;
		bit_idx = i % 8;

		result->data[i] =
			(vb->data[byte_idx] & (1 << bit_idx)) ? 1.0f : -1.0f;
	}

	PG_RETURN_VECTOR_P(result);
}

/*
 * Hamming distance between two binary vectors
 */
PG_FUNCTION_INFO_V1(binary_hamming_distance);
Datum
binary_hamming_distance(PG_FUNCTION_ARGS)
{
	VectorBinary *a = (VectorBinary *) PG_GETARG_POINTER(0);
	VectorBinary *b = (VectorBinary *) PG_GETARG_POINTER(1);
	int			count = 0;
	int			nbytes;
	int			i;

	if (a->dim != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("binary vector dimensions must match")));

	nbytes = (a->dim + 7) / 8;

	for (i = 0; i < nbytes; i++)
	{
		uint8		xor_val = a->data[i] ^ b->data[i];
#if defined(__GNUC__) && (__GNUC__ >= 4)
		count += __builtin_popcount(xor_val);
#else
		/* fallback: Kernighan’s algorithm */
		while (xor_val)
		{
			xor_val &= xor_val - 1;
			count++;
		}
#endif
	}

	PG_RETURN_INT32(count);
}

/*
 * Utility: Dynamic quantization selector
 */
PG_FUNCTION_INFO_V1(dynamic_quantize_vector);
Datum
dynamic_quantize_vector(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	float8		memory_pressure;
	float8		recall_target;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);
	memory_pressure = PG_GETARG_FLOAT8(1);
	recall_target = PG_GETARG_FLOAT8(2);

	if ((memory_pressure > 0.8) || (recall_target < 0.85))
		PG_RETURN_POINTER(quantize_vector_i8(v));
	else if ((memory_pressure > 0.6) || (recall_target < 0.90))
		PG_RETURN_POINTER(quantize_vector_f16(v));
	else
		PG_RETURN_POINTER(v);
}

/*
 * Quantization accuracy analysis: Compute error metrics for INT8 quantization
 * Returns JSONB with: mse, mae, max_error, compression_ratio, relative_error
 */
PG_FUNCTION_INFO_V1(quantize_analyze_int8);
Datum
quantize_analyze_int8(PG_FUNCTION_ARGS)
{
	Vector	   *original;
	VectorI8   *quantized;
	Vector	   *dequantized;
	float4		max_abs;
	float4		scale;
	double		mse;
	double		mae;
	double		max_error;

	double		sum_abs_original;
	double		relative_error;
	int			i;
	int			original_bytes;
	int			quantized_bytes;
	double		compression_ratio;
	StringInfoData json_buf;
	Jsonb	   *result_jsonb;

	original = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(original);
	max_abs = 0.0f;
	scale = 0.0f;
	mse = 0.0;
	mae = 0.0;
	max_error = 0.0;
	sum_abs_original = 0.0;
	relative_error = 0.0;

	/* Quantize the vector */
	quantized = quantize_vector_i8(original);

	/* Find max_abs for proper dequantization */
	for (i = 0; i < original->dim; i++)
	{
		float4		abs_val = fabsf(original->data[i]);

		if (abs_val > max_abs)
			max_abs = abs_val;
	}

	if (max_abs == 0.0f)
	{
		/* All zeros - perfect quantization */
		initStringInfo(&json_buf);
		appendStringInfo(&json_buf,
						 "{\"mse\":0.0,\"mae\":0.0,\"max_error\":0.0,"
						 "\"compression_ratio\":%.2f,\"relative_error\":0.0}",
						 8.0);
		result_jsonb = DatumGetJsonbP(DirectFunctionCall1(
														  jsonb_in, CStringGetTextDatum(json_buf.data)));
		NDB_FREE(json_buf.data);
		NDB_FREE(quantized);
		PG_RETURN_POINTER(result_jsonb);
	}

	scale = 127.0f / max_abs;

	/* Dequantize properly */
	dequantized = new_vector(original->dim);
	for (i = 0; i < original->dim; i++)
		dequantized->data[i] = ((float4) quantized->data[i]) / scale;

	/* Compute error metrics */
	for (i = 0; i < original->dim; i++)
	{
		double		error = (double) original->data[i] - (double) dequantized->data[i];
		double		abs_error = fabs(error);
		double		abs_original = fabs((double) original->data[i]);

		mse += error * error;
		mae += abs_error;
		sum_abs_original += abs_original;

		if (abs_error > max_error)
			max_error = abs_error;
	}

	mse /= (double) original->dim;
	mae /= (double) original->dim;
	relative_error = (sum_abs_original > 0.0) ? (mae / sum_abs_original) : 0.0;

	/* Compute compression ratio */
	original_bytes = VECTOR_SIZE(original->dim);
	quantized_bytes = offsetof(VectorI8, data) + sizeof(int8) * original->dim;
	compression_ratio = (double) original_bytes / (double) quantized_bytes;

	/* Build JSONB result */
	initStringInfo(&json_buf);
	appendStringInfo(&json_buf,
					 "{\"mse\":%.10f,\"mae\":%.10f,\"max_error\":%.10f,"
					 "\"compression_ratio\":%.2f,\"relative_error\":%.10f,"
					 "\"original_bytes\":%d,\"quantized_bytes\":%d}",
					 mse,
					 mae,
					 max_error,
					 compression_ratio,
					 relative_error,
					 original_bytes,
					 quantized_bytes);

	result_jsonb = DatumGetJsonbP(DirectFunctionCall1(
													  jsonb_in, CStringGetTextDatum(json_buf.data)));

	NDB_FREE(json_buf.data);
	NDB_FREE(quantized);
	NDB_FREE(dequantized);

	PG_RETURN_POINTER(result_jsonb);
}

/*
 * Quantization accuracy analysis: Compute error metrics for FP16 quantization
 */
PG_FUNCTION_INFO_V1(quantize_analyze_fp16);
Datum
quantize_analyze_fp16(PG_FUNCTION_ARGS)
{
	Vector	   *original;
	VectorF16  *quantized;
	Vector	   *dequantized;

	double		mse;
	double		mae;
	double		max_error;
	double		sum_abs_original;
	double		relative_error;

	int			i;
	int			original_bytes;
	int			quantized_bytes;
	double		compression_ratio;
	StringInfoData json_buf;
	Jsonb	   *result_jsonb;

	original = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(original);

	mse = 0.0;
	mae = 0.0;
	max_error = 0.0;
	sum_abs_original = 0.0;
	relative_error = 0.0;

	/* Quantize the vector */
	quantized = quantize_vector_f16(original);

	/* Dequantize */
	dequantized = new_vector(original->dim);
	for (i = 0; i < original->dim; i++)
		dequantized->data[i] = fp16_to_float(quantized->data[i]);

	/* Compute error metrics */
	for (i = 0; i < original->dim; i++)
	{
		double		error = (double) original->data[i] - (double) dequantized->data[i];
		double		abs_error = fabs(error);
		double		abs_original = fabs((double) original->data[i]);

		mse += error * error;
		mae += abs_error;
		sum_abs_original += abs_original;

		if (abs_error > max_error)
			max_error = abs_error;
	}

	mse /= (double) original->dim;
	mae /= (double) original->dim;
	relative_error = (sum_abs_original > 0.0) ? (mae / sum_abs_original) : 0.0;

	/* Compute compression ratio */
	original_bytes = VECTOR_SIZE(original->dim);
	quantized_bytes = offsetof(VectorF16, data) + sizeof(uint16) * original->dim;
	compression_ratio = (double) original_bytes / (double) quantized_bytes;

	/* Build JSONB result */
	initStringInfo(&json_buf);
	appendStringInfo(&json_buf,
					 "{\"mse\":%.10f,\"mae\":%.10f,\"max_error\":%.10f,"
					 "\"compression_ratio\":%.2f,\"relative_error\":%.10f,"
					 "\"original_bytes\":%d,\"quantized_bytes\":%d}",
					 mse,
					 mae,
					 max_error,
					 compression_ratio,
					 relative_error,
					 original_bytes,
					 quantized_bytes);

	result_jsonb = DatumGetJsonbP(DirectFunctionCall1(
													  jsonb_in, CStringGetTextDatum(json_buf.data)));

	NDB_FREE(json_buf.data);
	NDB_FREE(quantized);
	NDB_FREE(dequantized);

	PG_RETURN_POINTER(result_jsonb);
}

/*
 * Quantization accuracy analysis: Compute error metrics for binary quantization
 */
PG_FUNCTION_INFO_V1(quantize_analyze_binary);
Datum
quantize_analyze_binary(PG_FUNCTION_ARGS)
{
	Vector	   *original;
	VectorBinary *quantized;
	Vector	   *dequantized;

	double		mse;
	double		mae;
	double		max_error;
	double		sum_abs_original;
	double		relative_error;

	int			i;
	int			original_bytes;
	int			quantized_bytes;
	double		compression_ratio;
	StringInfoData json_buf;
	Jsonb	   *result_jsonb;

	int			byte_idx;
	int			bit_idx;

	original = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(original);

	mse = 0.0;
	mae = 0.0;
	max_error = 0.0;
	sum_abs_original = 0.0;
	relative_error = 0.0;

	/* Quantize the vector */
	quantized = quantize_vector_binary(original);

	/* Dequantize */
	dequantized = new_vector(original->dim);
	for (i = 0; i < original->dim; i++)
	{
		byte_idx = i / 8;
		bit_idx = i % 8;
		dequantized->data[i] =
			(quantized->data[byte_idx] & (1 << bit_idx)) ? 1.0f : -1.0f;
	}

	/* Compute error metrics */
	for (i = 0; i < original->dim; i++)
	{
		double		error = (double) original->data[i] - (double) dequantized->data[i];
		double		abs_error = fabs(error);
		double		abs_original = fabs((double) original->data[i]);

		mse += error * error;
		mae += abs_error;
		sum_abs_original += abs_original;

		if (abs_error > max_error)
			max_error = abs_error;
	}

	mse /= (double) original->dim;
	mae /= (double) original->dim;
	relative_error = (sum_abs_original > 0.0) ? (mae / sum_abs_original) : 0.0;

	/* Compute compression ratio */
	original_bytes = VECTOR_SIZE(original->dim);
	quantized_bytes = offsetof(VectorBinary, data) + ((original->dim + 7) / 8);
	compression_ratio = (double) original_bytes / (double) quantized_bytes;

	/* Build JSONB result */
	initStringInfo(&json_buf);
	appendStringInfo(&json_buf,
					 "{\"mse\":%.10f,\"mae\":%.10f,\"max_error\":%.10f,"
					 "\"compression_ratio\":%.2f,\"relative_error\":%.10f,"
					 "\"original_bytes\":%d,\"quantized_bytes\":%d}",
					 mse,
					 mae,
					 max_error,
					 compression_ratio,
					 relative_error,
					 original_bytes,
					 quantized_bytes);

	result_jsonb = DatumGetJsonbP(DirectFunctionCall1(
													  jsonb_in, CStringGetTextDatum(json_buf.data)));

	NDB_FREE(json_buf.data);
	NDB_FREE(quantized);
	NDB_FREE(dequantized);

	PG_RETURN_POINTER(result_jsonb);
}

/*
 * Compare distance preservation: Compute distances before and after quantization
 * Returns JSONB with: original_distance, quantized_distance, distance_error, distance_preservation
 */
PG_FUNCTION_INFO_V1(quantize_compare_distances);
Datum
quantize_compare_distances(PG_FUNCTION_ARGS)
{
	Vector	   *a_original;
	Vector	   *b_original;
	text	   *quant_type;
	char	   *qtype;
	Vector	   *a_dequantized;
	Vector	   *b_dequantized;
	float4		original_dist;
	float4		quantized_dist;
	double		distance_error;
	double		distance_preservation;
	StringInfoData json_buf;
	Jsonb	   *result_jsonb;

	a_original = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a_original);
	b_original = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b_original);
	quant_type = PG_GETARG_TEXT_P(2);
	qtype = text_to_cstring(quant_type);

	if (strcmp(qtype, "int8") == 0)
	{
		VectorI8   *aq,
				   *bq;
		float4		max_abs_a = 0.0f,
					max_abs_b = 0.0f;
		float4		scale_a,
					scale_b;
		int			i;

		aq = quantize_vector_i8(a_original);
		bq = quantize_vector_i8(b_original);

		/* Find scales */
		for (i = 0; i < a_original->dim; i++)
		{
			float4		abs_val = fabsf(a_original->data[i]);

			if (abs_val > max_abs_a)
				max_abs_a = abs_val;
		}
		for (i = 0; i < b_original->dim; i++)
		{
			float4		abs_val = fabsf(b_original->data[i]);

			if (abs_val > max_abs_b)
				max_abs_b = abs_val;
		}

		scale_a = (max_abs_a > 0.0f) ? (127.0f / max_abs_a) : 1.0f;
		scale_b = (max_abs_b > 0.0f) ? (127.0f / max_abs_b) : 1.0f;

		a_dequantized = new_vector(a_original->dim);
		b_dequantized = new_vector(b_original->dim);

		for (i = 0; i < a_original->dim; i++)
			a_dequantized->data[i] = ((float4) aq->data[i]) / scale_a;
		for (i = 0; i < b_original->dim; i++)
			b_dequantized->data[i] = ((float4) bq->data[i]) / scale_b;

		NDB_FREE(aq);
		NDB_FREE(bq);
	}
	else if (strcmp(qtype, "fp16") == 0)
	{
		VectorF16  *aq,
				   *bq;
		int			i;

		aq = quantize_vector_f16(a_original);
		bq = quantize_vector_f16(b_original);

		a_dequantized = new_vector(a_original->dim);
		b_dequantized = new_vector(b_original->dim);

		for (i = 0; i < a_original->dim; i++)
			a_dequantized->data[i] = fp16_to_float(aq->data[i]);
		for (i = 0; i < b_original->dim; i++)
			b_dequantized->data[i] = fp16_to_float(bq->data[i]);

		NDB_FREE(aq);
		NDB_FREE(bq);
	}
	else if (strcmp(qtype, "binary") == 0)
	{
		VectorBinary *aq,
				   *bq;
		int			i;
		int			byte_idx;
		int			bit_idx;

		aq = quantize_vector_binary(a_original);
		bq = quantize_vector_binary(b_original);

		a_dequantized = new_vector(a_original->dim);
		b_dequantized = new_vector(b_original->dim);

		for (i = 0; i < a_original->dim; i++)
		{
			byte_idx = i / 8;
			bit_idx = i % 8;
			a_dequantized->data[i] =
				(aq->data[byte_idx] & (1 << bit_idx)) ? 1.0f : -1.0f;
		}
		for (i = 0; i < b_original->dim; i++)
		{
			byte_idx = i / 8;
			bit_idx = i % 8;
			b_dequantized->data[i] =
				(bq->data[byte_idx] & (1 << bit_idx)) ? 1.0f : -1.0f;
		}

		NDB_FREE(aq);
		NDB_FREE(bq);
	}
	else
	{
		NDB_FREE(qtype);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("quantization type must be 'int8', 'fp16', or 'binary'")));
		PG_RETURN_NULL();
	}

	/* Compute original distance */
	original_dist = l2_distance(a_original, b_original);

	/* Compute quantized distance */
	quantized_dist = l2_distance(a_dequantized, b_dequantized);

	/* Compute error metrics */
	distance_error = fabs((double) original_dist - (double) quantized_dist);
	distance_preservation = (original_dist > 0.0f)
		? (1.0 - (distance_error / (double) original_dist))
		: 1.0;

	/* Build JSONB result */
	initStringInfo(&json_buf);
	appendStringInfo(&json_buf,
					 "{\"original_distance\":%.10f,\"quantized_distance\":%.10f,"
					 "\"distance_error\":%.10f,\"distance_preservation\":%.10f,"
					 "\"quantization_type\":\"%s\"}",
					 original_dist,
					 quantized_dist,
					 distance_error,
					 distance_preservation,
					 qtype);

	result_jsonb = DatumGetJsonbP(DirectFunctionCall1(
													  jsonb_in, CStringGetTextDatum(json_buf.data)));

	NDB_FREE(json_buf.data);
	NDB_FREE(qtype);
	NDB_FREE(a_dequantized);
	NDB_FREE(b_dequantized);

	PG_RETURN_POINTER(result_jsonb);
}

/*
 * Quantization accuracy analysis: Compute error metrics for UINT8 quantization
 */
PG_FUNCTION_INFO_V1(quantize_analyze_uint8);
Datum
quantize_analyze_uint8(PG_FUNCTION_ARGS)
{
	Vector	   *original;
	VectorU8   *quantized;
	Vector	   *dequantized;

	float4		min_val;
	float4		max_val;
	float4		scale;
	double		mse;
	double		mae;

	double		max_error;
	double		sum_abs_original;
	double		relative_error;
	int			i;
	int			original_bytes;
	int			quantized_bytes;
	double		compression_ratio;
	StringInfoData json_buf;
	Jsonb	   *result_jsonb;
	bool		first;

	original = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(original);

	min_val = 0.0f;
	max_val = 0.0f;
	mse = 0.0;
	mae = 0.0;
	max_error = 0.0;
	sum_abs_original = 0.0;
	relative_error = 0.0;
	first = true;

	/* Find min and max for proper dequantization */
	for (i = 0; i < original->dim; i++)
	{
		if (first)
		{
			min_val = max_val = original->data[i];
			first = false;
		}
		else
		{
			if (original->data[i] < min_val)
				min_val = original->data[i];
			if (original->data[i] > max_val)
				max_val = original->data[i];
		}
	}

	/* Quantize the vector */
	quantized = quantize_vector_uint8(original);

	if (max_val == min_val)
	{
		/* All values are the same - perfect quantization */
		initStringInfo(&json_buf);
		appendStringInfo(&json_buf,
						 "{\"mse\":0.0,\"mae\":0.0,\"max_error\":0.0,"
						 "\"compression_ratio\":%.2f,\"relative_error\":0.0}",
						 4.0);
		result_jsonb = DatumGetJsonbP(DirectFunctionCall1(
														  jsonb_in, CStringGetTextDatum(json_buf.data)));
		NDB_FREE(json_buf.data);
		NDB_FREE(quantized);
		PG_RETURN_POINTER(result_jsonb);
	}

	scale = 255.0f / (max_val - min_val);

	/* Dequantize properly */
	dequantized = new_vector(original->dim);
	for (i = 0; i < original->dim; i++)
		dequantized->data[i] = min_val + ((float4) quantized->data[i]) / scale;

	/* Compute error metrics */
	for (i = 0; i < original->dim; i++)
	{
		double		error = (double) original->data[i] - (double) dequantized->data[i];
		double		abs_error = fabs(error);
		double		abs_original = fabs((double) original->data[i]);

		mse += error * error;
		mae += abs_error;
		sum_abs_original += abs_original;

		if (abs_error > max_error)
			max_error = abs_error;
	}

	mse /= (double) original->dim;
	mae /= (double) original->dim;
	relative_error = (sum_abs_original > 0.0) ? (mae / sum_abs_original) : 0.0;

	/* Compute compression ratio */
	original_bytes = VECTOR_SIZE(original->dim);
	quantized_bytes = offsetof(VectorU8, data) + sizeof(uint8) * original->dim;
	compression_ratio = (double) original_bytes / (double) quantized_bytes;

	/* Build JSONB result */
	initStringInfo(&json_buf);
	appendStringInfo(&json_buf,
					 "{\"mse\":%.10f,\"mae\":%.10f,\"max_error\":%.10f,"
					 "\"compression_ratio\":%.2f,\"relative_error\":%.10f,"
					 "\"original_bytes\":%d,\"quantized_bytes\":%d}",
					 mse,
					 mae,
					 max_error,
					 compression_ratio,
					 relative_error,
					 original_bytes,
					 quantized_bytes);

	result_jsonb = DatumGetJsonbP(DirectFunctionCall1(
													  jsonb_in, CStringGetTextDatum(json_buf.data)));

	NDB_FREE(json_buf.data);
	NDB_FREE(quantized);
	NDB_FREE(dequantized);

	PG_RETURN_POINTER(result_jsonb);
}

/*
 * Quantization accuracy analysis: Compute error metrics for ternary quantization
 */
PG_FUNCTION_INFO_V1(quantize_analyze_ternary);
Datum
quantize_analyze_ternary(PG_FUNCTION_ARGS)
{
	Vector	   *original;
	VectorTernary *quantized;
	Vector	   *dequantized;

	double		mse;
	double		mae;
	double		max_error;
	double		sum_abs_original;
	double		relative_error;

	int			i;
	int			original_bytes;
	int			quantized_bytes;
	double		compression_ratio;
	StringInfoData json_buf;
	Jsonb	   *result_jsonb;

	int			byte_idx;
	int			bit_idx;
	uint8		value;
	float4		max_abs;

	original = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(original);

	mse = 0.0;
	mae = 0.0;
	max_error = 0.0;
	sum_abs_original = 0.0;
	relative_error = 0.0;
	max_abs = 0.0f;

	/* Find max absolute value for threshold */
	for (i = 0; i < original->dim; i++)
	{
		float4		abs_val = fabsf(original->data[i]);

		if (abs_val > max_abs)
			max_abs = abs_val;
	}

	/* Quantize the vector */
	quantized = quantize_vector_ternary(original);

	/* Dequantize */
	dequantized = new_vector(original->dim);
	for (i = 0; i < original->dim; i++)
	{
		byte_idx = (i * 2) / 8;
		bit_idx = (i * 2) % 8;
		value = (quantized->data[byte_idx] >> bit_idx) & 0x03;

		if (value == 2)
			dequantized->data[i] = 1.0f;	/* +1 */
		else if (value == 1)
			dequantized->data[i] = -1.0f;	/* -1 */
		else
			dequantized->data[i] = 0.0f;	/* 0 */
	}

	/* Compute error metrics */
	for (i = 0; i < original->dim; i++)
	{
		double		error = (double) original->data[i] - (double) dequantized->data[i];
		double		abs_error = fabs(error);
		double		abs_original = fabs((double) original->data[i]);

		mse += error * error;
		mae += abs_error;
		sum_abs_original += abs_original;

		if (abs_error > max_error)
			max_error = abs_error;
	}

	mse /= (double) original->dim;
	mae /= (double) original->dim;
	relative_error = (sum_abs_original > 0.0) ? (mae / sum_abs_original) : 0.0;

	/* Compute compression ratio */
	original_bytes = VECTOR_SIZE(original->dim);
	quantized_bytes = offsetof(VectorTernary, data) + ((original->dim * 2 + 7) / 8);
	compression_ratio = (double) original_bytes / (double) quantized_bytes;

	/* Build JSONB result */
	initStringInfo(&json_buf);
	appendStringInfo(&json_buf,
					 "{\"mse\":%.10f,\"mae\":%.10f,\"max_error\":%.10f,"
					 "\"compression_ratio\":%.2f,\"relative_error\":%.10f,"
					 "\"original_bytes\":%d,\"quantized_bytes\":%d}",
					 mse,
					 mae,
					 max_error,
					 compression_ratio,
					 relative_error,
					 original_bytes,
					 quantized_bytes);

	result_jsonb = DatumGetJsonbP(DirectFunctionCall1(
													  jsonb_in, CStringGetTextDatum(json_buf.data)));

	NDB_FREE(json_buf.data);
	NDB_FREE(quantized);
	NDB_FREE(dequantized);

	PG_RETURN_POINTER(result_jsonb);
}

/*
 * Quantization accuracy analysis: Compute error metrics for INT4 quantization
 */
PG_FUNCTION_INFO_V1(quantize_analyze_int4);
Datum
quantize_analyze_int4(PG_FUNCTION_ARGS)
{
	Vector	   *original;
	VectorI4   *quantized;
	Vector	   *dequantized;

	float4		max_abs;
	float4		scale;
	double		mse;
	double		mae;
	double		max_error;

	double		sum_abs_original;
	double		relative_error;
	int			i;
	int			original_bytes;

	int			quantized_bytes;
	double		compression_ratio;
	StringInfoData json_buf;
	Jsonb	   *result_jsonb;
	int			byte_idx;
	int			bit_idx;
	uint8		uvalue;
	int8		value;

	original = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(original);
	max_abs = 0.0f;
	mse = 0.0;
	mae = 0.0;
	max_error = 0.0;
	sum_abs_original = 0.0;
	relative_error = 0.0;

	/* Find maximum absolute value */
	for (i = 0; i < original->dim; i++)
	{
		float4		abs_val = fabsf(original->data[i]);

		if (abs_val > max_abs)
			max_abs = abs_val;
	}

	/* Quantize the vector */
	quantized = quantize_vector_int4(original);

	if (max_abs == 0.0f)
	{
		/* All zeros - perfect quantization */
		initStringInfo(&json_buf);
		appendStringInfo(&json_buf,
						 "{\"mse\":0.0,\"mae\":0.0,\"max_error\":0.0,"
						 "\"compression_ratio\":%.2f,\"relative_error\":0.0}",
						 8.0);
		result_jsonb = DatumGetJsonbP(DirectFunctionCall1(
														  jsonb_in, CStringGetTextDatum(json_buf.data)));
		NDB_FREE(json_buf.data);
		NDB_FREE(quantized);
		PG_RETURN_POINTER(result_jsonb);
	}

	scale = 7.0f / max_abs;

	/* Dequantize properly */
	dequantized = new_vector(original->dim);
	for (i = 0; i < original->dim; i++)
	{
		byte_idx = i / 2;
		bit_idx = (i % 2) * 4;
		uvalue = (quantized->data[byte_idx] >> bit_idx) & 0x0F;

		/* Convert unsigned 4-bit to signed */
		value = (uvalue < 8) ? ((int8) uvalue - 8) : ((int8) uvalue - 8);
		dequantized->data[i] = ((float4) value) / scale;
	}

	/* Compute error metrics */
	for (i = 0; i < original->dim; i++)
	{
		double		error = (double) original->data[i] - (double) dequantized->data[i];
		double		abs_error = fabs(error);
		double		abs_original = fabs((double) original->data[i]);

		mse += error * error;
		mae += abs_error;
		sum_abs_original += abs_original;

		if (abs_error > max_error)
			max_error = abs_error;
	}

	mse /= (double) original->dim;
	mae /= (double) original->dim;
	relative_error = (sum_abs_original > 0.0) ? (mae / sum_abs_original) : 0.0;

	/* Compute compression ratio */
	original_bytes = VECTOR_SIZE(original->dim);
	quantized_bytes = offsetof(VectorI4, data) + ((original->dim + 1) / 2);
	compression_ratio = (double) original_bytes / (double) quantized_bytes;

	/* Build JSONB result */
	initStringInfo(&json_buf);
	appendStringInfo(&json_buf,
					 "{\"mse\":%.10f,\"mae\":%.10f,\"max_error\":%.10f,"
					 "\"compression_ratio\":%.2f,\"relative_error\":%.10f,"
					 "\"original_bytes\":%d,\"quantized_bytes\":%d}",
					 mse,
					 mae,
					 max_error,
					 compression_ratio,
					 relative_error,
					 original_bytes,
					 quantized_bytes);

	result_jsonb = DatumGetJsonbP(DirectFunctionCall1(
													  jsonb_in, CStringGetTextDatum(json_buf.data)));

	NDB_FREE(json_buf.data);
	NDB_FREE(quantized);
	NDB_FREE(dequantized);

	PG_RETURN_POINTER(result_jsonb);
}

/*
 * UINT8 quantization: 8 bits per dimension, unsigned [0, 255]
 * Compression: 4x (8 bits vs 32 bits)
 */
VectorU8 *
quantize_vector_uint8(Vector *v)
{
	VectorU8   *result;
	int			size;
	float4		min_val = 0.0f;
	float4		max_val = 0.0f;
	float4		scale;
	int			i;
	bool		first = true;

	/* Find min and max values for scaling */
	for (i = 0; i < v->dim; i++)
	{
		if (first)
		{
			min_val = max_val = v->data[i];
			first = false;
		}
		else
		{
			if (v->data[i] < min_val)
				min_val = v->data[i];
			if (v->data[i] > max_val)
				max_val = v->data[i];
		}
	}

	size = offsetof(VectorU8, data) + sizeof(uint8) * v->dim;
	result = (VectorU8 *) palloc0(size);
	SET_VARSIZE(result, size);
	result->dim = v->dim;

	if (max_val == min_val)
		return result;

	scale = 255.0f / (max_val - min_val);

	for (i = 0; i < v->dim; i++)
	{
		float4		normalized = (v->data[i] - min_val) * scale;

		if (normalized > 255.0f)
			normalized = 255.0f;
		if (normalized < 0.0f)
			normalized = 0.0f;
		result->data[i] = (uint8) rintf(normalized);
	}

	return result;
}

PG_FUNCTION_INFO_V1(vector_to_uint8);
Datum
vector_to_uint8(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	VectorU8   *result;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	result = quantize_vector_uint8(v);
	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(uint8_to_vector);
Datum
uint8_to_vector(PG_FUNCTION_ARGS)
{
	VectorU8   *vu8 = (VectorU8 *) PG_GETARG_POINTER(0);
	Vector	   *result;
	int			i;

	/* Note: This is approximate - exact restoration requires min/max */
	result = new_vector(vu8->dim);
	for (i = 0; i < vu8->dim; i++)
		result->data[i] = ((float4) vu8->data[i]) / 255.0f;

	PG_RETURN_VECTOR_P(result);
}

/*
 * Ternary quantization: 2 bits per dimension (-1, 0, +1)
 * Compression: 16x (2 bits vs 32 bits)
 */
VectorTernary *
quantize_vector_ternary(Vector *v)
{
	VectorTernary *result;
	int			nbytes;
	int			size;
	int			i;
	int			byte_idx;
	int			bit_idx;
	float4		threshold = 0.0f;	/* Values > threshold = +1, < -threshold =
									 * -1, else 0 */

	/* Compute threshold as 1/3 of max absolute value */
	{
		float4		max_abs = 0.0f;

		for (i = 0; i < v->dim; i++)
		{
			float4		abs_val = fabsf(v->data[i]);

			if (abs_val > max_abs)
				max_abs = abs_val;
		}
		threshold = max_abs / 3.0f;
	}

	nbytes = (v->dim * 2 + 7) / 8;	/* 2 bits per dimension */
	size = offsetof(VectorTernary, data) + nbytes;
	result = (VectorTernary *) palloc0(size);
	SET_VARSIZE(result, size);
	result->dim = v->dim;

	for (i = 0; i < v->dim; i++)
	{
		uint8		value;

		if (v->data[i] > threshold)
			value = 2;			/* +1 */
		else if (v->data[i] < -threshold)
			value = 1;			/* -1 */
		else
			value = 0;			/* 0 */

		/* Pack 2 bits per dimension: 4 values per byte */
		byte_idx = (i * 2) / 8;
		bit_idx = (i * 2) % 8;
		result->data[byte_idx] |= (value << bit_idx);
	}

	return result;
}

PG_FUNCTION_INFO_V1(vector_to_ternary);
Datum
vector_to_ternary(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	VectorTernary *result;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	result = quantize_vector_ternary(v);
	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(ternary_to_vector);
Datum
ternary_to_vector(PG_FUNCTION_ARGS)
{
	VectorTernary *vt = (VectorTernary *) PG_GETARG_POINTER(0);
	Vector	   *result;
	int			i;
	int			byte_idx;
	int			bit_idx;
	uint8		value;

	result = new_vector(vt->dim);

	for (i = 0; i < vt->dim; i++)
	{
		byte_idx = (i * 2) / 8;
		bit_idx = (i * 2) % 8;
		value = (vt->data[byte_idx] >> bit_idx) & 0x03;

		if (value == 2)
			result->data[i] = 1.0f; /* +1 */
		else if (value == 1)
			result->data[i] = -1.0f;	/* -1 */
		else
			result->data[i] = 0.0f; /* 0 */
	}

	PG_RETURN_VECTOR_P(result);
}

/*
 * INT4 quantization: 4 bits per dimension, signed [-8, 7]
 * Compression: 8x (4 bits vs 32 bits)
 * GPU-accelerated when available
 */
VectorI4 *
quantize_vector_int4(Vector *v)
{
	VectorI4   *result;
	int			nbytes;
	int			size;
	int			i;
	int			byte_idx;
	int			bit_idx;
	float4		max_abs = 0.0f;
	float4		scale;

	/* Find maximum absolute value */
	for (i = 0; i < v->dim; i++)
	{
		float4		abs_val = fabsf(v->data[i]);

		if (abs_val > max_abs)
			max_abs = abs_val;
	}

	if (max_abs == 0.0f)
	{
		nbytes = (v->dim + 1) / 2;	/* 2 values per byte */
		size = offsetof(VectorI4, data) + nbytes;
		result = (VectorI4 *) palloc0(size);
		SET_VARSIZE(result, size);
		result->dim = v->dim;
		return result;
	}

	scale = 7.0f / max_abs;
	nbytes = (v->dim + 1) / 2;	/* 2 values per byte */
	size = offsetof(VectorI4, data) + nbytes;
	result = (VectorI4 *) palloc0(size);
	SET_VARSIZE(result, size);
	result->dim = v->dim;

	/* Try GPU first if available */
	if (neurondb_gpu_is_available())
	{
		neurondb_gpu_quantize_int4(v->data, result->data, v->dim);
		return result;
	}

	/* CPU fallback */
	{
		uint8		uvalue;

		for (i = 0; i < v->dim; i++)
		{
			int8		value;
			float4		scaled = v->data[i] * scale;

			if (scaled > 7.0f)
				value = 7;
			else if (scaled < -8.0f)
				value = -8;
			else
				value = (int8) rintf(scaled);

			/*
			 * Convert signed to unsigned 4-bit (0-7 = -8 to -1, 8-15 = 0 to
			 * 7)
			 */
			if (value < 0)
				uvalue = (uint8) (8 + value);	/* -8 to -1 -> 0 to 7 */
			else
				uvalue = (uint8) (8 + value);	/* 0 to 7 -> 8 to 15 */
			if (uvalue > 15)
				uvalue = 15;

			/* Pack 2 values per byte */
			byte_idx = i / 2;
			bit_idx = (i % 2) * 4;
			result->data[byte_idx] |= (uvalue << bit_idx);
		}
	}

	return result;
}

PG_FUNCTION_INFO_V1(vector_to_int4);
Datum
vector_to_int4(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	VectorI4   *result;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	result = quantize_vector_int4(v);
	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(int4_to_vector);
Datum
int4_to_vector(PG_FUNCTION_ARGS)
{
	VectorI4   *vi4 = (VectorI4 *) PG_GETARG_POINTER(0);
	Vector	   *result;
	int			i;
	int			byte_idx;
	int			bit_idx;
	uint8		uvalue;
	int8		value;

	result = new_vector(vi4->dim);

	for (i = 0; i < vi4->dim; i++)
	{
		byte_idx = i / 2;
		bit_idx = (i % 2) * 4;
		uvalue = (vi4->data[byte_idx] >> bit_idx) & 0x0F;

		/* Convert unsigned 4-bit to signed (0-7 = -8 to -1, 8-15 = 0 to 7) */
		value = (uvalue < 8) ? ((int8) uvalue - 8) : ((int8) uvalue - 8);

		/* Approximate dequantization */
		result->data[i] = ((float4) value) / 7.0f;
	}

	PG_RETURN_VECTOR_P(result);
}

/*-------------------------------------------------------------------------
 * halfvec type I/O functions (native PostgreSQL type for FP16 vectors)
 *-------------------------------------------------------------------------
 */

/*
 * halfvec_in: Parse text input like "[1.0, 2.0, 3.0]" and convert to VectorF16
 */
PG_FUNCTION_INFO_V1(halfvec_in);
Datum
halfvec_in(PG_FUNCTION_ARGS)
{
	char	   *str = PG_GETARG_CSTRING(0);
	VectorF16  *result;
	float4	   *temp_data;
	int			dim;
	int			capacity;
	char	   *ptr;
	char	   *endptr;
	int			size;
	int			i;

	dim = 0;
	capacity = 16;
	ptr = str;

	/* Skip whitespace */
	while (isspace((unsigned char) *ptr))
		ptr++;

	/* Expect '[' */
	if (*ptr == '[')
		ptr++;

	temp_data = (float4 *) palloc(sizeof(float4) * capacity);

	/* Parse comma-separated floats */
	while (*ptr && *ptr != ']')
	{
		while (isspace((unsigned char) *ptr) || *ptr == ',')
			ptr++;

		if (*ptr == ']' || *ptr == '\0')
			break;

		if (dim >= capacity)
		{
			capacity *= 2;
			temp_data = (float4 *) repalloc(
											temp_data, sizeof(float4) * capacity);
		}

		temp_data[dim] = strtof(ptr, &endptr);
		if (ptr == endptr)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("invalid input for halfvec")));

		ptr = endptr;
		dim++;
	}

	if (dim == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("halfvec must have at least 1 dimension")));

	if (dim > 4000)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("halfvec dimension %d exceeds maximum of 4000",
						dim)));

	/* Allocate VectorF16 */
	size = offsetof(VectorF16, data) + sizeof(uint16) * dim;
	result = (VectorF16 *) palloc0(size);
	SET_VARSIZE(result, size);
	result->dim = dim;

	/* Convert float32 to fp16 */
	for (i = 0; i < dim; i++)
		result->data[i] = float4_to_fp16(temp_data[i]);

	NDB_FREE(temp_data);
	PG_RETURN_POINTER(result);
}

/*
 * halfvec_out: Convert VectorF16 to text representation
 */
PG_FUNCTION_INFO_V1(halfvec_out);
Datum
halfvec_out(PG_FUNCTION_ARGS)
{
	VectorF16  *vf16 = (VectorF16 *) PG_GETARG_POINTER(0);
	StringInfoData buf;
	int			i;

	if (vf16 == NULL)
		PG_RETURN_CSTRING(pstrdup("NULL"));

	initStringInfo(&buf);
	appendStringInfoChar(&buf, '[');

	for (i = 0; i < vf16->dim; i++)
	{
		if (i > 0)
			appendStringInfoChar(&buf, ',');
		appendStringInfo(&buf, "%g", fp16_to_float(vf16->data[i]));
	}

	appendStringInfoChar(&buf, ']');
	PG_RETURN_CSTRING(buf.data);
}

/*
 * halfvec_recv: Binary receive function
 */
PG_FUNCTION_INFO_V1(halfvec_recv);
Datum
halfvec_recv(PG_FUNCTION_ARGS)
{
	StringInfo	buf = (StringInfo) PG_GETARG_POINTER(0);
	VectorF16  *result;
	int16		dim;
	int			size;
	int			i;

	dim = pq_getmsgint(buf, sizeof(int16));

	if (dim <= 0 || dim > 4000)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_BINARY_REPRESENTATION),
				 errmsg("invalid halfvec dimension: %d", dim)));

	size = offsetof(VectorF16, data) + sizeof(uint16) * dim;
	result = (VectorF16 *) palloc0(size);
	SET_VARSIZE(result, size);
	result->dim = dim;

	for (i = 0; i < dim; i++)
		result->data[i] = pq_getmsgint(buf, sizeof(uint16));

	PG_RETURN_POINTER(result);
}

/*
 * halfvec_send: Binary send function
 */
PG_FUNCTION_INFO_V1(halfvec_send);
Datum
halfvec_send(PG_FUNCTION_ARGS)
{
	VectorF16  *vf16 = (VectorF16 *) PG_GETARG_POINTER(0);
	StringInfoData buf;
	int			i;

	pq_begintypsend(&buf);
	pq_sendint(&buf, vf16->dim, sizeof(int16));

	for (i = 0; i < vf16->dim; i++)
		pq_sendint(&buf, vf16->data[i], sizeof(uint16));

	PG_RETURN_BYTEA_P(pq_endtypsend(&buf));
}

/*-------------------------------------------------------------------------
 * Comparison operators for halfvec type
 *-------------------------------------------------------------------------
 */

/*
 * halfvec_eq: Equality comparison for halfvec
 */
PG_FUNCTION_INFO_V1(halfvec_eq);
Datum
halfvec_eq(PG_FUNCTION_ARGS)
{
	VectorF16  *a = (VectorF16 *) PG_GETARG_POINTER(0);
	VectorF16  *b = (VectorF16 *) PG_GETARG_POINTER(1);
	int			i;

	/* Handle NULL vectors */
	if (a == NULL && b == NULL)
		PG_RETURN_BOOL(true);
	if (a == NULL || b == NULL)
		PG_RETURN_BOOL(false);

	if (a->dim != b->dim)
		PG_RETURN_BOOL(false);

	/* Compare fp16 values */
	for (i = 0; i < a->dim; i++)
	{
		if (a->data[i] != b->data[i])
			PG_RETURN_BOOL(false);
	}

	PG_RETURN_BOOL(true);
}

/*
 * halfvec_ne: Inequality comparison for halfvec
 */
PG_FUNCTION_INFO_V1(halfvec_ne);
Datum
halfvec_ne(PG_FUNCTION_ARGS)
{
	return DirectFunctionCall2(
							   halfvec_eq, PG_GETARG_DATUM(0), PG_GETARG_DATUM(1))
		? BoolGetDatum(false)
		: BoolGetDatum(true);
}

/*
 * halfvec_hash: Hash function for halfvec
 */
PG_FUNCTION_INFO_V1(halfvec_hash);
Datum
halfvec_hash(PG_FUNCTION_ARGS)
{
	VectorF16  *v = (VectorF16 *) PG_GETARG_POINTER(0);
	uint32		hash = 5381;
	int			i;

	if (v == NULL)
		PG_RETURN_UINT32(0);

	hash = ((hash << 5) + hash) + (uint32) v->dim;

	for (i = 0; i < v->dim && i < 16; i++)
	{
		uint32		val = (uint32) v->data[i];

		hash = ((hash << 5) + hash) + val;
	}

	if (v->dim > 16)
	{
		int			stride = v->dim / 16;

		for (i = 16; i < v->dim; i += stride)
		{
			uint32		val = (uint32) v->data[i];

			hash = ((hash << 5) + hash) + val;
		}
	}

	PG_RETURN_UINT32(hash);
}

/*-------------------------------------------------------------------------
 * Distance functions for halfvec type
 *-------------------------------------------------------------------------
 */

/*
 * halfvec_l2_distance: L2 distance between two halfvec vectors
 */
PG_FUNCTION_INFO_V1(halfvec_l2_distance);
Datum
halfvec_l2_distance(PG_FUNCTION_ARGS)
{
	VectorF16  *a = (VectorF16 *) PG_GETARG_POINTER(0);
	VectorF16  *b = (VectorF16 *) PG_GETARG_POINTER(1);
	double		sum = 0.0;
	int			i;
	float4		result;

	if (a == NULL || b == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot compute distance with NULL halfvec")));

	if (a->dim != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("halfvec dimensions must match: %d vs %d",
						a->dim,
						b->dim)));

	for (i = 0; i < a->dim; i++)
	{
		float		va = fp16_to_float(a->data[i]);
		float		vb = fp16_to_float(b->data[i]);
		double		diff = (double) va - (double) vb;

		sum += diff * diff;
	}

	result = (float4) sqrt(sum);
	PG_RETURN_FLOAT4(result);
}

/*
 * halfvec_cosine_distance: Cosine distance between two halfvec vectors
 */
PG_FUNCTION_INFO_V1(halfvec_cosine_distance);
Datum
halfvec_cosine_distance(PG_FUNCTION_ARGS)
{
	VectorF16  *a = (VectorF16 *) PG_GETARG_POINTER(0);
	VectorF16  *b = (VectorF16 *) PG_GETARG_POINTER(1);
	double		dot = 0.0,
				norm_a = 0.0,
				norm_b = 0.0;
	int			i;
	float4		result;

	if (a == NULL || b == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot compute distance with NULL halfvec")));

	if (a->dim != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("halfvec dimensions must match")));

	for (i = 0; i < a->dim; i++)
	{
		float		va = fp16_to_float(a->data[i]);
		float		vb = fp16_to_float(b->data[i]);

		dot += (double) va * (double) vb;
		norm_a += (double) va * (double) va;
		norm_b += (double) vb * (double) vb;
	}

	if (norm_a == 0.0 || norm_b == 0.0)
		PG_RETURN_FLOAT4(1.0);

	result = (float4) (1.0 - (dot / (sqrt(norm_a) * sqrt(norm_b))));
	PG_RETURN_FLOAT4(result);
}

/*
 * halfvec_inner_product: Inner product (negative for distance ordering)
 */
PG_FUNCTION_INFO_V1(halfvec_inner_product);
Datum
halfvec_inner_product(PG_FUNCTION_ARGS)
{
	VectorF16  *a = (VectorF16 *) PG_GETARG_POINTER(0);
	VectorF16  *b = (VectorF16 *) PG_GETARG_POINTER(1);
	double		sum = 0.0;
	int			i;

	if (a == NULL || b == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot compute distance with NULL halfvec")));

	if (a->dim != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("halfvec dimensions must match")));

	for (i = 0; i < a->dim; i++)
	{
		float		va = fp16_to_float(a->data[i]);
		float		vb = fp16_to_float(b->data[i]);

		sum += (double) va * (double) vb;
	}

	PG_RETURN_FLOAT4((float4) (-sum));
}

/*-------------------------------------------------------------------------
 * bit type support for binary vectors (pgvector compatibility)
 *-------------------------------------------------------------------------
 */

/*
 * vector_to_bit: Convert vector to PostgreSQL bit type (pgvector compatibility)
 * Returns bit type instead of bytea
 */
PG_FUNCTION_INFO_V1(vector_to_bit);
Datum
vector_to_bit(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	VectorBinary *vb;
	VarBit	   *result;
	int			nbits;
	int			nbytes;
	int			i;
	int			byte_idx;
	int			bit_idx;
	bits8	   *bit_data;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	if (v == NULL)
		PG_RETURN_NULL();

	vb = quantize_vector_binary(v);
	nbits = vb->dim;
	nbytes = VARBITTOTALLEN(nbits);

	result = (VarBit *) palloc0(nbytes);
	SET_VARSIZE(result, nbytes);
	VARBITLEN(result) = nbits;

	bit_data = VARBITS(result);

	/* Copy bits from VectorBinary to VarBit */
	for (i = 0; i < nbits; i++)
	{
		byte_idx = i / 8;
		bit_idx = i % 8;

		if (vb->data[byte_idx] & (1 << bit_idx))
		{
			int			bit_byte_idx = i / BITS_PER_BYTE;
			int			bit_bit_idx = i % BITS_PER_BYTE;

			bit_data[bit_byte_idx] |= (1 << (BITS_PER_BYTE - 1 - bit_bit_idx));
		}
	}

	NDB_FREE(vb);
	PG_RETURN_VARBIT_P(result);
}

/*
 * bit_to_vector: Convert PostgreSQL bit type to vector
 */
PG_FUNCTION_INFO_V1(bit_to_vector);
Datum
bit_to_vector(PG_FUNCTION_ARGS)
{
	VarBit	   *bit_vec = PG_GETARG_VARBIT_P(0);
	Vector	   *result;
	int			nbits;
	int			i;
	bits8	   *bit_data;

	if (bit_vec == NULL)
		PG_RETURN_NULL();

	nbits = VARBITLEN(bit_vec);
	result = new_vector(nbits);
	bit_data = VARBITS(bit_vec);

	/* Convert bits to vector (+1.0 or -1.0) */
	for (i = 0; i < nbits; i++)
	{
		int			bit_byte_idx = i / BITS_PER_BYTE;
		int			bit_bit_idx = i % BITS_PER_BYTE;
		int			bit_val = (bit_data[bit_byte_idx] >> (BITS_PER_BYTE - 1 - bit_bit_idx)) & 1;

		result->data[i] = bit_val ? 1.0f : -1.0f;
	}

	PG_RETURN_VECTOR_P(result);
}

/*-------------------------------------------------------------------------
 * Distance functions for bit type (Hamming distance)
 *-------------------------------------------------------------------------
 */

/*
 * bit_hamming_distance: Hamming distance between two bit vectors
 */
PG_FUNCTION_INFO_V1(bit_hamming_distance);
Datum
bit_hamming_distance(PG_FUNCTION_ARGS)
{
	VarBit	   *a = PG_GETARG_VARBIT_P(0);
	VarBit	   *b = PG_GETARG_VARBIT_P(1);
	int			nbits;
	int			count = 0;
	int			i;
	bits8	   *a_bits,
			   *b_bits;

	if (a == NULL || b == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot compute distance with NULL bit vectors")));

	nbits = VARBITLEN(a);
	if (VARBITLEN(b) != nbits)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("bit vector lengths must match: %d vs %d",
						nbits,
						VARBITLEN(b))));

	a_bits = VARBITS(a);
	b_bits = VARBITS(b);

	/* Count differing bits */
	for (i = 0; i < nbits; i++)
	{
		int			byte_idx = i / BITS_PER_BYTE;
		int			bit_idx = i % BITS_PER_BYTE;
		int			a_bit = (a_bits[byte_idx] >> (BITS_PER_BYTE - 1 - bit_idx)) & 1;
		int			b_bit = (b_bits[byte_idx] >> (BITS_PER_BYTE - 1 - bit_idx)) & 1;

		if (a_bit != b_bit)
			count++;
	}

	PG_RETURN_INT32(count);
}
