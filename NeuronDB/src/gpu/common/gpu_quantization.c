/*-------------------------------------------------------------------------
 *
 * gpu_quantization.c
 *    Host-side quantization helpers.
 *
 * This module provides quantization helper functions used by entry points
 * for vector compression and quantization operations.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/common/gpu_quantization.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"

#include "neurondb_gpu.h"
#include "neurondb_gpu_backend.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static uint16_t
float_to_half_bits(float value)
{
	union
	{
		float		f;
		uint32_t	u;
	}			v = {.f = value};
	uint32_t	sign = (v.u >> 16) & 0x8000;
	uint32_t	mant = v.u & 0x7fffff;
	int32_t		exp = ((int32_t) (v.u >> 23) & 0xff) - 127 + 15;

	if (exp <= 0)
	{
		uint32_t	shifted;

		if (exp < -10)
			return (uint16_t) sign;
		mant |= 0x800000;
		shifted = mant >> (1 - exp);
		return (uint16_t) (sign | ((shifted + 0x1000) >> 13));
	}
	else if (exp >= 31)
	{
		return (uint16_t) (sign | 0x7c00);
	}

	return (uint16_t) (sign | (exp << 10) | ((mant + 0x1000) >> 13));
}

static inline int8_t
quantize_int8_value(float value, float scale)
{
	float		scaled = roundf(value * scale);

	if (scaled > 127.0f)
		scaled = 127.0f;
	else if (scaled < -128.0f)
		scaled = -128.0f;
	return (int8_t) scaled;
}

void
neurondb_gpu_quantize_fp16(const float *input, void *output, int count)
{
	const		ndb_gpu_backend *backend;

	if (input == NULL || output == NULL || count <= 0)
		return;

	if (neurondb_gpu_is_available())
	{
		backend = ndb_gpu_get_active_backend();
		if (backend && backend->launch_quant_fp16)
		{
			if (backend->launch_quant_fp16(
										   input, output, count, NULL)
				== 0)
				return;
			elog(DEBUG1,
				 "neurondb: GPU quantize_fp16 failed, using CPU "
				 "fallback");
		}
	}

	{
		uint16_t   *dst = (uint16_t *) output;
		int			i;

		for (i = 0; i < count; i++)
			dst[i] = float_to_half_bits(input[i]);
	}
}

void
neurondb_gpu_quantize_int4(const float *input, unsigned char *output, int count)
{
	const		ndb_gpu_backend *backend;
	float		max_abs = 0.0f;
	float		scale;
	int			i;

	if (input == NULL || output == NULL || count <= 0)
		return;

	/* Find max absolute value */
	for (i = 0; i < count; i++)
	{
		float		abs_val = fabsf(input[i]);

		if (abs_val > max_abs)
			max_abs = abs_val;
	}

	if (max_abs < 1e-10f)
	{
		memset(output, 0, (count + 1) / 2);
		return;
	}

	scale = 7.0f / max_abs;

	if (neurondb_gpu_is_available())
	{
		backend = ndb_gpu_get_active_backend();
		if (backend && backend->launch_quant_int8)
		{
			/* Use int8 as fallback for int4 */
			if (backend->launch_quant_int8(
										   input, (int8_t *) output, count, scale, NULL) == 0)
				return;
			elog(DEBUG1,
				 "neurondb: GPU quantize_int4 failed, using CPU fallback");
		}
	}

	/* CPU fallback: pack 2 values per byte */
	for (i = 0; i < count; i += 2)
	{
		int8_t		val1,
					val2;
		unsigned char uval1,
					uval2;
		float		scaled1 = input[i] * scale;
		float		scaled2 = (i + 1 < count) ? input[i + 1] * scale : 0.0f;

		if (scaled1 > 7.0f)
			val1 = 7;
		else if (scaled1 < -8.0f)
			val1 = -8;
		else
			val1 = (int8_t) rintf(scaled1);

		if (scaled2 > 7.0f)
			val2 = 7;
		else if (scaled2 < -8.0f)
			val2 = -8;
		else
			val2 = (int8_t) rintf(scaled2);

		uval1 = (unsigned char) (8 + val1);
		uval2 = (unsigned char) (8 + val2);
		if (uval1 > 15)
			uval1 = 15;
		if (uval2 > 15)
			uval2 = 15;

		output[i / 2] = uval1 | (uval2 << 4);
	}
}

void
neurondb_gpu_quantize_fp8_e4m3(const float *input, unsigned char *output, int count)
{
	const		ndb_gpu_backend *backend;

	if (input == NULL || output == NULL || count <= 0)
		return;

	if (neurondb_gpu_is_available())
	{
		backend = ndb_gpu_get_active_backend();
		if (backend && backend->launch_quant_fp8_e4m3)
		{
			if (backend->launch_quant_fp8_e4m3(
											   input, output, count, NULL) == 0)
				return;
			elog(DEBUG1,
				 "neurondb: GPU quantize_fp8_e4m3 failed, using CPU fallback");
		}
	}

	/* CPU fallback */
	{
		int			i;

		for (i = 0; i < count; i++)
		{
			uint32_t	bits;
			uint8_t		sign,
						mant;
			int			exp;

			memcpy(&bits, &input[i], sizeof(float));
			sign = (bits >> 31) & 0x1;
			exp = ((bits >> 23) & 0xFF) - 127;
			mant = (bits >> 20) & 0x7;

			if (exp > 7)
				output[i] = (sign << 7) | 0x7F;
			else if (exp < -6)
				output[i] = 0;
			else
			{
				exp = exp + 7;
				output[i] = (sign << 7) | ((exp & 0xF) << 3) | (mant & 0x7);
			}
		}
	}
}

void
neurondb_gpu_quantize_fp8_e5m2(const float *input, unsigned char *output, int count)
{
	const		ndb_gpu_backend *backend;

	if (input == NULL || output == NULL || count <= 0)
		return;

	if (neurondb_gpu_is_available())
	{
		backend = ndb_gpu_get_active_backend();
		if (backend && backend->launch_quant_fp8_e5m2)
		{
			if (backend->launch_quant_fp8_e5m2(
											   input, output, count, NULL) == 0)
				return;
			elog(DEBUG1,
				 "neurondb: GPU quantize_fp8_e5m2 failed, using CPU fallback");
		}
	}

	/* CPU fallback */
	{
		int			i;

		for (i = 0; i < count; i++)
		{
			uint32_t	bits;
			uint8_t		sign,
						mant;
			int			exp;

			memcpy(&bits, &input[i], sizeof(float));
			sign = (bits >> 31) & 0x1;
			exp = ((bits >> 23) & 0xFF) - 127;
			mant = (bits >> 21) & 0x3;

			if (exp > 15)
				output[i] = (sign << 7) | 0x7F;
			else if (exp < -14)
				output[i] = 0;
			else
			{
				exp = exp + 15;
				output[i] = (sign << 7) | ((exp & 0x1F) << 2) | (mant & 0x3);
			}
		}
	}
}

void
neurondb_gpu_quantize_int8(const float *input, int8 * output, int count)
{
	const		ndb_gpu_backend *backend;
	float		max_abs = 0.0f;
	float		scale;
	int			i;

	if (input == NULL || output == NULL || count <= 0)
		return;

	for (i = 0; i < count; i++)
	{
		float		abs_val = fabsf(input[i]);

		if (abs_val > max_abs)
			max_abs = abs_val;
	}
	scale = (max_abs > 0.0f) ? (127.0f / max_abs) : 1.0f;

	if (neurondb_gpu_is_available())
	{
		backend = ndb_gpu_get_active_backend();
		if (backend && backend->launch_quant_int8)
		{
			if (backend->launch_quant_int8(
										   input, (int8_t *) output, count, scale, NULL)
				== 0)
				return;
			elog(DEBUG1,
				 "neurondb: GPU int8 quantization failed; using "
				 "CPU fallback");
		}
	}

	for (i = 0; i < count; i++)
		output[i] = quantize_int8_value(input[i], scale);
}

void
neurondb_gpu_quantize_binary(const float *input, uint8 * output, int count)
{
	const		ndb_gpu_backend *backend;

	if (input == NULL || output == NULL || count <= 0)
		return;

	if (neurondb_gpu_is_available())
	{
		backend = ndb_gpu_get_active_backend();
		if (backend && backend->launch_quant_binary)
		{
			if (backend->launch_quant_binary(
											 input, (uint8_t *) output, count, NULL)
				== 0)
				return;
			elog(DEBUG1,
				 "neurondb: GPU binary quantization failed; "
				 "using CPU fallback");
		}
	}

	{
		int			out_bytes = (count + 7) / 8;
		int			idx;

		memset(output, 0, out_bytes);

		for (idx = 0; idx < count; idx++)
		{
			if (input[idx] > 0.0f)
				output[idx / 8] |= (1u << (idx % 8));
		}
	}
}
