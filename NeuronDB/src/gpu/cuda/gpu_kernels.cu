/*-------------------------------------------------------------------------
 *
 * gpu_kernels.cu
 *    CUDA/HIP kernels for NeurondB GPU acceleration
 *
 *  This file implements GPU-accelerated fundamental computations such as
 *  vector distances (L2, cosine, inner product), quantization operations
 *  (float32 → float16, int8, and binary), and KMeans clustering steps,
 *  adhering strictly to PostgreSQL code style and robustness conventions.
 *  Importantly, this file is written according to PostgreSQL C style,
 *  maximizing clarity, predictability, and maintainability.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu_kernels.cu
 *
 *-------------------------------------------------------------------------
 */

#ifdef NDB_GPU_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#endif

#ifdef NDB_GPU_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#endif

#include <float.h>
#include <math.h>
#include "neurondb_cuda_launchers.h"

/*
 * Device kernel to compute L2 (Euclidean) distance between query and each vector row.
 */
extern "C" __global__ void
l2_distance_kernel(const float *__restrict__ query,
	const float *__restrict__ matrix,
	float *__restrict__ out,
	int rows,
	int dim)
{
	int row;
	const float *vec;
	double acc;
	int d;

	row = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (row < rows)
	{
		vec = matrix + (row * dim);
		acc = 0.0;
		for (d = 0; d < dim; d++)
		{
			float diff;

			diff = query[d] - vec[d];
			acc += (double)diff * (double)diff;
		}
		out[row] = sqrtf((float)acc);
	}
}

/*
 * Device kernel to compute cosine distance:
 * d(query, vec) = 1 - dot(query, vec) / (||query|| * ||vec||)
 */
extern "C" __global__ void
cosine_distance_kernel(const float *__restrict__ query,
	const float *__restrict__ matrix,
	float *__restrict__ out,
	int rows,
	int dim)
{
	int row;
	const float *vec;
	double dot, n_q, n_v;
	int d;
	float cosine_sim;

	row = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (row < rows)
	{
		vec = matrix + (row * dim);
		dot = 0.0;
		n_q = 0.0;
		n_v = 0.0;

		/* Loop unrolling in strict C style (4-way) */
		d = 0;
		for (; d <= dim - 4; d += 4)
		{
			float q0 = query[d], v0 = vec[d];
			float q1 = query[d + 1], v1 = vec[d + 1];
			float q2 = query[d + 2], v2 = vec[d + 2];
			float q3 = query[d + 3], v3 = vec[d + 3];

			dot += (double)q0 * (double)v0 + (double)q1 * (double)v1
				+ (double)q2 * (double)v2
				+ (double)q3 * (double)v3;

			n_q += (double)q0 * (double)q0 + (double)q1 * (double)q1
				+ (double)q2 * (double)q2
				+ (double)q3 * (double)q3;

			n_v += (double)v0 * (double)v0 + (double)v1 * (double)v1
				+ (double)v2 * (double)v2
				+ (double)v3 * (double)v3;
		}
		for (; d < dim; d++)
		{
			float q = query[d];
			float v = vec[d];

			dot += (double)q * (double)v;
			n_q += (double)q * (double)q;
			n_v += (double)v * (double)v;
		}

		/* Convert to norms (magnitude) */
		n_q = sqrt(n_q);
		n_v = sqrt(n_v);

		cosine_sim = 0.0f;
		if (n_q > 1e-10 && n_v > 1e-10)
			cosine_sim = (float)(dot / (n_q * n_v));

		/* Clamp to [-1,1] to avoid floating point error propagation */
		if (cosine_sim < -1.0f)
			cosine_sim = -1.0f;
		else if (cosine_sim > 1.0f)
			cosine_sim = 1.0f;

		out[row] = 1.0f - cosine_sim;
	}
}

/*
 * Device kernel to compute negative inner product for ANN distance.
 */
extern "C" __global__ void
inner_product_kernel(const float *__restrict__ query,
	const float *__restrict__ matrix,
	float *__restrict__ out,
	int rows,
	int dim)
{
	int row;
	const float *vec;
	double acc;
	int d;

	row = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (row < rows)
	{
		vec = matrix + (row * dim);
		acc = 0.0;
		d = 0;
		for (; d <= dim - 4; d += 4)
		{
			acc += (double)query[d] * (double)vec[d]
				+ (double)query[d + 1] * (double)vec[d + 1]
				+ (double)query[d + 2] * (double)vec[d + 2]
				+ (double)query[d + 3] * (double)vec[d + 3];
		}
		for (; d < dim; d++)
			acc += (double)query[d] * (double)vec[d];

		out[row] = (float)(-acc);
	}
}

/*
 * Device kernel for quantizing float32 to float16.
 */
extern "C" __global__ void
quantize_fp16_kernel(const float *__restrict__ input,
	__half *__restrict__ output,
	int count)
{
	int idx;

	idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < count)
	{
#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_HIP)
		output[idx] = __float2half(input[idx]);
#else
		/* Should not occur; fallback for compilation safety. */
		union {
			float f;
			unsigned u32;
		} u;
		u.f = input[idx];
		output[idx] = (__half)(u.u32 >> 16);
#endif
	}
}

/*
 * Device kernel for quantizing float32 to int8, with scaling.
 * Explicitly clamps to int8_t range.
 */
extern "C" __global__ void
quantize_int8_kernel(const float *__restrict__ input,
	signed char *__restrict__ output,
	int count,
	float scale)
{
	int idx;
	float val;

	idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < count)
	{
		val = input[idx] * scale;

		if (val > 127.0f)
			val = 127.0f;
		else if (val < -128.0f)
			val = -128.0f;

		output[idx] = (signed char)__float2int_rn(val);
	}
}

/*
 * Device kernel for binary quantization:
 * Packs groups of 8 input floats to one output byte.
 */
extern "C" __global__ void
quantize_binary_kernel(const float *__restrict__ input,
	unsigned char *__restrict__ output,
	int count)
{
	int idx;
	int byte_count;
	unsigned char byte;
	int start, in_idx, i;

	idx = blockIdx.x * blockDim.x + threadIdx.x;
	byte_count = (count + 7) / 8;

	if (idx < byte_count)
	{
		byte = 0;
		start = idx * 8;
#pragma unroll
		for (i = 0; i < 8; i++)
		{
			in_idx = start + i;
			if (in_idx < count && input[in_idx] > 0.0f)
				byte |= (1 << i);
		}
		output[idx] = byte;
	}
}

/*
 * Device kernel for INT4 quantization: 4 bits per dimension, packed 2 per byte
 */
extern "C" __global__ void
quantize_int4_kernel(const float *__restrict__ input,
	unsigned char *__restrict__ output,
	int count,
	float scale)
{
	int idx;
	unsigned char byte_val;
	int8_t val1, val2;
	unsigned char uval1, uval2;

	idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx * 2 < count)
	{
		/* Process two values per thread */
		float scaled1 = input[idx * 2] * scale;
		float scaled2 = (idx * 2 + 1 < count) ? input[idx * 2 + 1] * scale : 0.0f;

		/* Clamp to [-8, 7] range */
		if (scaled1 > 7.0f)
			val1 = 7;
		else if (scaled1 < -8.0f)
			val1 = -8;
		else
			val1 = (int8_t)__float2int_rn(scaled1);

		if (scaled2 > 7.0f)
			val2 = 7;
		else if (scaled2 < -8.0f)
			val2 = -8;
		else
			val2 = (int8_t)__float2int_rn(scaled2);

		/* Convert signed to unsigned 4-bit */
		uval1 = (unsigned char)(8 + val1);
		uval2 = (unsigned char)(8 + val2);
		if (uval1 > 15)
			uval1 = 15;
		if (uval2 > 15)
			uval2 = 15;

		/* Pack 2 values into 1 byte */
		byte_val = uval1 | (uval2 << 4);
		output[idx] = byte_val;
	}
}

/*
 * Device kernel for FP8 E4M3 quantization
 */
extern "C" __global__ void
quantize_fp8_e4m3_kernel(const float *__restrict__ input,
	unsigned char *__restrict__ output,
	int count)
{
	int idx;
	uint32_t bits;
	uint8_t sign, exp, mant;
	uint8_t result;

	idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < count)
	{
		float val = input[idx];

		if (val == 0.0f)
		{
			output[idx] = 0;
			return;
		}

		memcpy(&bits, &val, sizeof(float));
		sign = (bits >> 31) & 0x1;
		/* Use signed int for exponent calculation to allow negative checks */
		int exp_signed = ((bits >> 23) & 0xFF) - 127; /* FP32 exponent bias */
		mant = (bits >> 20) & 0x7; /* Top 3 mantissa bits */

		/* E4M3: 4 exponent bits (bias 7), 3 mantissa bits */
		if (exp_signed > 7)
		{
			/* Overflow: return max value */
			result = (sign << 7) | 0x7F;
		}
		else if (exp_signed < -6)
		{
			/* Underflow: return zero */
			result = 0;
		}
		else
		{
			exp = (uint8_t)(exp_signed + 7); /* E4M3 bias */
			result = (sign << 7) | ((exp & 0xF) << 3) | (mant & 0x7);
		}

		output[idx] = result;
	}
}

/*
 * Device kernel for FP8 E5M2 quantization
 */
extern "C" __global__ void
quantize_fp8_e5m2_kernel(const float *__restrict__ input,
	unsigned char *__restrict__ output,
	int count)
{
	int idx;
	uint32_t bits;
	uint8_t sign, exp, mant;
	uint8_t result;

	idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < count)
	{
		float val = input[idx];

		if (val == 0.0f)
		{
			output[idx] = 0;
			return;
		}

		memcpy(&bits, &val, sizeof(float));
		sign = (bits >> 31) & 0x1;
		/* Use signed int for exponent calculation to allow negative checks */
		int exp_signed = ((bits >> 23) & 0xFF) - 127; /* FP32 exponent bias */
		mant = (bits >> 21) & 0x3; /* Top 2 mantissa bits */

		/* E5M2: 5 exponent bits (bias 15), 2 mantissa bits */
		if (exp_signed > 15)
		{
			/* Overflow: return max value */
			result = (sign << 7) | 0x7F;
		}
		else if (exp_signed < -14)
		{
			/* Underflow: return zero */
			result = 0;
		}
		else
		{
			exp = (uint8_t)(exp_signed + 15); /* E5M2 bias */
			result = (sign << 7) | ((exp & 0x1F) << 2) | (mant & 0x3);
		}

		output[idx] = result;
	}
}

#if defined(NDB_GPU_CUDA)

#define NDB_KERNEL_LAUNCH_PARAMS \
	int block = 256; \
	int grid = (count + block - 1) / block

/*
 * Host launch wrapper for quantize_fp16_kernel (float32 → fp16).
 */
extern "C" cudaError_t
launch_quantize_fp32_to_fp16(const float *input,
	void *output,
	int count,
	cudaStream_t stream)
{
	if (count <= 0 || input == NULL || output == NULL)
		return cudaSuccess;

	NDB_KERNEL_LAUNCH_PARAMS;
	quantize_fp16_kernel<<<grid, block, 0, stream>>>(
		input, reinterpret_cast<__half *>(output), count);

	return cudaGetLastError();
}

/*
 * Host launch wrapper for quantize_int8_kernel (float32 → int8).
 */
extern "C" cudaError_t
launch_quantize_fp32_to_int8(const float *input,
	signed char *output,
	int count,
	float scale,
	cudaStream_t stream)
{
	if (count <= 0 || input == NULL || output == NULL)
		return cudaSuccess;

	NDB_KERNEL_LAUNCH_PARAMS;
	quantize_int8_kernel<<<grid, block, 0, stream>>>(
		input, output, count, scale);

	return cudaGetLastError();
}

/*
 * Host launch wrapper for quantize_binary_kernel (float32 → binary).
 */
extern "C" cudaError_t
launch_quantize_fp32_to_binary(const float *input,
	unsigned char *output,
	int count,
	cudaStream_t stream)
{
	int block;
	int grid;

	if (count <= 0 || input == NULL || output == NULL)
		return cudaSuccess;

	block = 256;
	grid = ((count + 7) / 8 + block - 1) / block;
	quantize_binary_kernel<<<grid, block, 0, stream>>>(
		input, output, count);

	return cudaGetLastError();
}

/*
 * Host launch wrapper for quantize_int4_kernel (float32 → int4).
 */
extern "C" cudaError_t
launch_quantize_fp32_to_int4(const float *input,
	unsigned char *output,
	int count,
	float scale,
	cudaStream_t stream)
{
	int block;
	int grid;

	if (count <= 0 || input == NULL || output == NULL)
		return cudaSuccess;

	block = 256;
	grid = ((count + 1) / 2 + block - 1) / block;
	quantize_int4_kernel<<<grid, block, 0, stream>>>(
		input, output, count, scale);

	return cudaGetLastError();
}

/*
 * Host launch wrapper for quantize_fp8_e4m3_kernel (float32 → FP8 E4M3).
 */
extern "C" cudaError_t
launch_quantize_fp32_to_fp8_e4m3(const float *input,
	unsigned char *output,
	int count,
	cudaStream_t stream)
{
	if (count <= 0 || input == NULL || output == NULL)
		return cudaSuccess;

	NDB_KERNEL_LAUNCH_PARAMS;
	quantize_fp8_e4m3_kernel<<<grid, block, 0, stream>>>(
		input, output, count);

	return cudaGetLastError();
}

/*
 * Host launch wrapper for quantize_fp8_e5m2_kernel (float32 → FP8 E5M2).
 */
extern "C" cudaError_t
launch_quantize_fp32_to_fp8_e5m2(const float *input,
	unsigned char *output,
	int count,
	cudaStream_t stream)
{
	if (count <= 0 || input == NULL || output == NULL)
		return cudaSuccess;

	NDB_KERNEL_LAUNCH_PARAMS;
	quantize_fp8_e5m2_kernel<<<grid, block, 0, stream>>>(
		input, output, count);

	return cudaGetLastError();
}

#undef NDB_KERNEL_LAUNCH_PARAMS

#endif /* NDB_GPU_CUDA */
