/*
 * gpu_cuda_kernels.cu
 *	  CUDA kernel implementations for GPU-accelerated vector operations
 *
 * IDENTIFICATION
 *	  src/gpu/gpu_cuda_kernels.cu
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * ------------------------------------------------------------------------
 * This file provides detailed, robust CUDA kernel implementations for
 * vector operations used in NeurondB, adhering strictly to the PostgreSQL
 * C coding standard. All device functions, error paths, types, naming,
 * and structure are crafted for maximum clarity and reliability.
 * ------------------------------------------------------------------------
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

#include "neurondb_cuda_launchers.h"

/*-------------------------------------------------------------------------
 *
 * Device kernel: L2 (Euclidean) distance between two float vectors.
 * Computes ||a-b||_2 for two vectors of dimension dim.
 * Only thread 0 writes result; others exit immediately.
 *
 * Args:
 *   a      - input vector a, device pointer, length dim
 *   b      - input vector b, device pointer, length dim
 *   result - output pointer, device memory, *result will contain distance
 *   dim    - vector dimensionality
 *
 * Safety: No host error reporting; intended for single-thread block use.
 *         Host should check errors and results carefully.
 *-------------------------------------------------------------------------
 */
__global__ void
cuda_l2_distance_kernel(const float *__restrict__ a,
	const float *__restrict__ b,
	float *__restrict__ result,
	int dim)
{
	int tid;

	tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid == 0)
	{
		double sum;
		int i;

		sum = 0.0;
		for (i = 0; i < dim; i++)
		{
			double diff;

			diff = (double)a[i] - (double)b[i];
			sum += diff * diff;
		}
		*result = (float)sqrt(sum);
	}
}

/*-------------------------------------------------------------------------
 *
 * Device kernel: Batch L2 distance.
 * Computes Euclidean distance between every query and every target vector,
 * filling [nq x nt] output matrix. Each block.y processes a query vector,
 * each thread processes one entry in its target row.
 *
 * Args:
 *   queries    - [nq x dim] matrix (row major), device pointer
 *   targets    - [nt x dim] matrix (row major), device pointer
 *   distances  - [nq x nt] output, device pointer, row major
 *   nq         - number of queries
 *   nt         - number of targets
 *   dim        - vector dimensionality
 *
 * Layout: distances[q * nt + t] = ||queries[q] - targets[t]||
 *-------------------------------------------------------------------------
 */
__global__ void
cuda_batch_l2_kernel(const float *__restrict__ queries,
	const float *__restrict__ targets,
	float *__restrict__ distances,
	int nq,
	int nt,
	int dim)
{
	int query_idx;
	int target_idx;

	query_idx = blockIdx.y;
	target_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (query_idx < nq && target_idx < nt)
	{
		const float *q;
		const float *t;
		double sum;
		int i;

		q = queries + (query_idx * dim);
		t = targets + (target_idx * dim);
		sum = 0.0;
		for (i = 0; i < dim; i++)
		{
			double diff;

			diff = (double)q[i] - (double)t[i];
			sum += diff * diff;
		}
		distances[query_idx * nt + target_idx] = (float)sqrt(sum);
	}
}

/*-------------------------------------------------------------------------
 *
 * Device kernel: Cosine distance between two float vectors.
 * Computes 1 - dot(a, b) / (||a|| * ||b||).
 * Only thread 0 writes result; others exit immediately.
 *
 * Note: Both norms must be nonzero. If either is zero, returns 1.0.
 *
 * Args:
 *   a      - input vector a, device pointer, length dim
 *   b      - input vector b, device pointer, length dim
 *   result - device output, *result will contain distance
 *   dim    - vector dimensionality
 *-------------------------------------------------------------------------
 */
__global__ void
cuda_cosine_distance_kernel(const float *__restrict__ a,
	const float *__restrict__ b,
	float *__restrict__ result,
	int dim)
{
	int tid;

	tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid == 0)
	{
		double dot;
		double norm_a;
		double norm_b;
		int i;

		dot = 0.0;
		norm_a = 0.0;
		norm_b = 0.0;
		for (i = 0; i < dim; i++)
		{
			double ai = (double)a[i];
			double bi = (double)b[i];

			dot += ai * bi;
			norm_a += ai * ai;
			norm_b += bi * bi;
		}
		norm_a = sqrt(norm_a);
		norm_b = sqrt(norm_b);

		if (norm_a > 0.0 && norm_b > 0.0)
		{
			double cosine_sim;

			cosine_sim = dot / (norm_a * norm_b);

			if (cosine_sim < -1.0)
				cosine_sim = -1.0;
			else if (cosine_sim > 1.0)
				cosine_sim = 1.0;

			*result = 1.0f - (float)cosine_sim;
		} else
		{
			/* At least one zero vector: define distance as 1 */
			*result = 1.0f;
		}
	}
}

/*-------------------------------------------------------------------------
 *
 * Device kernel: Inner product of two vectors.
 * Computes sum_i (a[i] * b[i]); only thread 0 writes result.
 *
 * Args:
 *   a      - input vector a, device pointer, length dim
 *   b      - input vector b, device pointer, length dim
 *   result - device output, *result will contain output
 *   dim    - vector dimensionality
 *-------------------------------------------------------------------------
 */
__global__ void
cuda_inner_product_kernel(const float *__restrict__ a,
	const float *__restrict__ b,
	float *__restrict__ result,
	int dim)
{
	int tid;

	tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid == 0)
	{
		double accum;
		int i;

		accum = 0.0;
		for (i = 0; i < dim; i++)
			accum += (double)a[i] * (double)b[i];
		*result = (float)accum;
	}
}

/*-------------------------------------------------------------------------
 *
 * Host utility: Launch/execute L2 distance kernel on device for two vectors.
 * Handles resource allocation, data transfer, and error checking.
 *
 * Returns: L2 distance (float) between inputs.
 * Follows PostgreSQL C style: all variables declared at start, complete cleanup,
 * strong error handling (returning HUGE_VALF for fatal GPU failures).
 *-------------------------------------------------------------------------
 */
extern "C" float
cuda_compute_l2_distance(const float *a, const float *b, int dim)
{
	float *d_a = NULL;
	float *d_b = NULL;
	float *d_result = NULL;
	float h_result = 0.0f;
	cudaError_t cerr;

	/* Allocate device memory for inputs and result */
	cerr = cudaMalloc((void **)&d_a, dim * sizeof(float));
	if (cerr != cudaSuccess)
		goto fail;
	cerr = cudaMalloc((void **)&d_b, dim * sizeof(float));
	if (cerr != cudaSuccess)
		goto fail;
	cerr = cudaMalloc((void **)&d_result, sizeof(float));
	if (cerr != cudaSuccess)
		goto fail;

	/* Copy input vectors to device */
	cerr = cudaMemcpy(d_a, a, dim * sizeof(float), cudaMemcpyHostToDevice);
	if (cerr != cudaSuccess)
		goto fail;
	cerr = cudaMemcpy(d_b, b, dim * sizeof(float), cudaMemcpyHostToDevice);
	if (cerr != cudaSuccess)
		goto fail;

	/* Launch kernel: single block, single thread */
	cuda_l2_distance_kernel<<<1, 1>>>(d_a, d_b, d_result, dim);
	cerr = cudaGetLastError();
	if (cerr != cudaSuccess)
		goto fail;

	/* Copy result back to host memory */
	cerr = cudaMemcpy(
		&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
	if (cerr != cudaSuccess)
		goto fail;

	/* Full cleanup on success */
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_result);

	return h_result;

fail:
	if (d_a)
		cudaFree(d_a);
	if (d_b)
		cudaFree(d_b);
	if (d_result)
		cudaFree(d_result);

	return HUGE_VALF;
}

/*-------------------------------------------------------------------------
 *
 * Host utility: Batch L2 distance computation for (nq x nt) pairs.
 * Allocates buffers and launches cuda_batch_l2_kernel.
 *
 * Args:
 *   queries    - [nq x dim] float, host
 *   targets    - [nt x dim] float, host
 *   distances  - [nq x nt] float, host; output
 *   nq         - number of queries
 *   nt         - number of targets
 *   dim        - vector dimension
 *
 * Robust resource cleanup and strong error handling. If error, distances
 * array is left in indeterminate state. All GPUs freed.
 *-------------------------------------------------------------------------
 */
extern "C" void
cuda_compute_batch_l2(const float *queries,
	const float *targets,
	float *distances,
	int nq,
	int nt,
	int dim)
{
	float *d_queries = NULL;
	float *d_targets = NULL;
	float *d_distances = NULL;
	cudaError_t cerr;
	dim3 block;
	dim3 grid;

	/* Allocate device memory for input/output matrices */
	cerr = cudaMalloc((void **)&d_queries, nq * dim * sizeof(float));
	if (cerr != cudaSuccess)
		goto fail;
	cerr = cudaMalloc((void **)&d_targets, nt * dim * sizeof(float));
	if (cerr != cudaSuccess)
		goto fail;
	cerr = cudaMalloc((void **)&d_distances, nq * nt * sizeof(float));
	if (cerr != cudaSuccess)
		goto fail;

	/* Copy host memory to device memory */
	cerr = cudaMemcpy(d_queries,
		queries,
		nq * dim * sizeof(float),
		cudaMemcpyHostToDevice);
	if (cerr != cudaSuccess)
		goto fail;
	cerr = cudaMemcpy(d_targets,
		targets,
		nt * dim * sizeof(float),
		cudaMemcpyHostToDevice);
	if (cerr != cudaSuccess)
		goto fail;

	block.x = 256;
	block.y = 1;
	block.z = 1;
	grid.x = (nt + block.x - 1) / block.x;
	grid.y = nq;
	grid.z = 1;

	cuda_batch_l2_kernel<<<grid, block>>>(
		d_queries, d_targets, d_distances, nq, nt, dim);
	cerr = cudaGetLastError();
	if (cerr != cudaSuccess)
		goto fail;

	cerr = cudaMemcpy(distances,
		d_distances,
		nq * nt * sizeof(float),
		cudaMemcpyDeviceToHost);
	if (cerr != cudaSuccess)
		goto fail;

	/* Free GPU memory */
	cudaFree(d_queries);
	cudaFree(d_targets);
	cudaFree(d_distances);
	return;

fail:
	if (d_queries)
		cudaFree(d_queries);
	if (d_targets)
		cudaFree(d_targets);
	if (d_distances)
		cudaFree(d_distances);
	return;
}

/*-------------------------------------------------------------------------
 *
 * Host utility: Cosine distance of two vectors (float).
 * Robust implementation, error paths included.
 *
 * Returns: 1-dot/(norms), or 1.0 if norms == 0 or any error (as per kernel).
 * Returns HUGE_VALF if device failure encountered.
 *-------------------------------------------------------------------------
 */
extern "C" float
cuda_compute_cosine_distance(const float *a, const float *b, int dim)
{
	float *d_a = NULL;
	float *d_b = NULL;
	float *d_result = NULL;
	float h_result = 1.0f;
	cudaError_t cerr;

	cerr = cudaMalloc((void **)&d_a, dim * sizeof(float));
	if (cerr != cudaSuccess)
		goto fail;
	cerr = cudaMalloc((void **)&d_b, dim * sizeof(float));
	if (cerr != cudaSuccess)
		goto fail;
	cerr = cudaMalloc((void **)&d_result, sizeof(float));
	if (cerr != cudaSuccess)
		goto fail;

	cerr = cudaMemcpy(d_a, a, dim * sizeof(float), cudaMemcpyHostToDevice);
	if (cerr != cudaSuccess)
		goto fail;
	cerr = cudaMemcpy(d_b, b, dim * sizeof(float), cudaMemcpyHostToDevice);
	if (cerr != cudaSuccess)
		goto fail;

	cuda_cosine_distance_kernel<<<1, 1>>>(d_a, d_b, d_result, dim);
	cerr = cudaGetLastError();
	if (cerr != cudaSuccess)
		goto fail;

	cerr = cudaMemcpy(
		&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
	if (cerr != cudaSuccess)
		goto fail;

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_result);
	return h_result;

fail:
	if (d_a)
		cudaFree(d_a);
	if (d_b)
		cudaFree(d_b);
	if (d_result)
		cudaFree(d_result);
	return HUGE_VALF;
}

/*-------------------------------------------------------------------------
 *
 * Host utility: Inner product of two vectors.
 * Robust resource management and error handling.
 * Returns HUGE_VALF on fatal device errors.
 *-------------------------------------------------------------------------
 */
extern "C" float
cuda_compute_inner_product(const float *a, const float *b, int dim)
{
	float *d_a = NULL;
	float *d_b = NULL;
	float *d_result = NULL;
	float h_result = 0.0f;
	cudaError_t cerr;

	cerr = cudaMalloc((void **)&d_a, dim * sizeof(float));
	if (cerr != cudaSuccess)
		goto fail;
	cerr = cudaMalloc((void **)&d_b, dim * sizeof(float));
	if (cerr != cudaSuccess)
		goto fail;
	cerr = cudaMalloc((void **)&d_result, sizeof(float));
	if (cerr != cudaSuccess)
		goto fail;

	cerr = cudaMemcpy(d_a, a, dim * sizeof(float), cudaMemcpyHostToDevice);
	if (cerr != cudaSuccess)
		goto fail;
	cerr = cudaMemcpy(d_b, b, dim * sizeof(float), cudaMemcpyHostToDevice);
	if (cerr != cudaSuccess)
		goto fail;

	cuda_inner_product_kernel<<<1, 1>>>(d_a, d_b, d_result, dim);
	cerr = cudaGetLastError();
	if (cerr != cudaSuccess)
		goto fail;

	cerr = cudaMemcpy(
		&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
	if (cerr != cudaSuccess)
		goto fail;

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_result);
	return h_result;

fail:
	if (d_a)
		cudaFree(d_a);
	if (d_b)
		cudaFree(d_b);
	if (d_result)
		cudaFree(d_result);
	return HUGE_VALF;
}

/*-------------------------------------------------------------------------
 *
 * Utility: Checks whether CUDA is present and usable on the system.
 * Robust style: zero for not present/usable/error, one for available.
 * No elog: callers must error out as needed.
 *-------------------------------------------------------------------------
 */
extern "C" int
cuda_check_available(void)
{
	int deviceCount = 0;
	cudaError_t error;

	error = cudaGetDeviceCount(&deviceCount);

	if (error != cudaSuccess || deviceCount == 0)
		return 0;
	return 1;
}

/*-------------------------------------------------------------------------
 *
 * Utility: Retrieve system CUDA device name.
 * Copies the device name (or fallback string) into buffer 'name'.
 *
 * Args:
 *   name   - char buffer (output)
 *   maxlen - buffer length (including NUL)
 *
 * On error, writes "CUDA Device (Unknown)"
 *-------------------------------------------------------------------------
 */
extern "C" void
cuda_get_device_name(char *name, int maxlen)
{
	cudaDeviceProp prop;
	int device = 0;
	cudaError_t error;

	error = cudaGetDeviceProperties(&prop, device);

	if (error == cudaSuccess)
	{
		snprintf(name, maxlen, "%s", prop.name);
	} else
	{
		snprintf(name, maxlen, "CUDA Device (Unknown)");
	}
}
