/*-------------------------------------------------------------------------
 *
 * gpu_pq_kernels.cu
 *    GPU CUDA/HIP kernels for Product Quantization (PQ)
 *
 * Implements GPU-accelerated PQ encoding and distance computation.
 * Optimized for batch processing of thousands of vectors.
 *
 * Performance: ~20-100x speedup vs CPU for large batches
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/gpu_pq_kernels.cu
 *
 *-------------------------------------------------------------------------
 */

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#define CUDA_KERNEL __global__
#define cudaSuccess hipSuccess
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaGetLastError hipGetLastError
#else
#include <cuda_runtime.h>
#define CUDA_KERNEL __global__
#endif

#include <stdint.h>
#include <float.h>
#include "neurondb_cuda_launchers.h"

/*
 * GPU kernel: PQ encoding
 *
 * Encode vectors using pre-trained codebooks.
 * Each thread processes one vector.
 *
 * Args:
 *   vectors:    Input vectors [nvec x dim]
 *   codebooks:  PQ codebooks [num_subspaces x codebook_size x subspace_dim]
 *   codes:      Output PQ codes [nvec x num_subspaces]
 *   nvec:       Number of vectors
 *   dim:        Vector dimensionality
 *   m:          Number of subspaces
 *   ks:         Codebook size
 */
CUDA_KERNEL void
pq_encode_kernel(const float *vectors,
	const float *codebooks,
	uint8_t *codes,
	int nvec,
	int dim,
	int m,
	int ks)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= nvec)
		return;

	const float *vec = vectors + idx * dim;
	uint8_t *code = codes + idx * m;

	int subspace_dim = dim / m;

	/* Encode each subspace */
	for (int sub = 0; sub < m; sub++)
	{
		const float *vec_sub = vec + sub * subspace_dim;
		const float *codebook = codebooks + sub * ks * subspace_dim;

		float min_dist = FLT_MAX;
		uint8_t best_idx = 0;

		/* Find nearest codeword */
		for (int k = 0; k < ks; k++)
		{
			const float *codeword = codebook + k * subspace_dim;
			float dist = 0.0f;

			for (int d = 0; d < subspace_dim; d++)
			{
				float diff = vec_sub[d] - codeword[d];
				dist += diff * diff;
			}

			if (dist < min_dist)
			{
				min_dist = dist;
				best_idx = (uint8_t)k;
			}
		}

		code[sub] = best_idx;
	}
}

/*
 * GPU kernel: PQ asymmetric distance computation
 *
 * Compute distances from query vector to PQ-encoded database vectors.
 * Uses asymmetric distance (query not quantized).
 *
 * Args:
 *   query:      Query vector [dim]
 *   codes:      PQ codes [nvec x m]
 *   codebooks:  PQ codebooks [m x ks x subspace_dim]
 *   distances:  Output distances [nvec]
 *   nvec:       Number of database vectors
 *   dim:        Vector dimensionality
 *   m:          Number of subspaces
 *   ks:         Codebook size
 */
CUDA_KERNEL void
pq_asymmetric_distance_kernel(const float *query,
	const uint8_t *codes,
	const float *codebooks,
	float *distances,
	int nvec,
	int dim,
	int m,
	int ks)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= nvec)
		return;

	const uint8_t *code = codes + idx * m;
	int subspace_dim = dim / m;
	float total_dist = 0.0f;

	/* Accumulate distance across subspaces */
	for (int sub = 0; sub < m; sub++)
	{
		const float *query_sub = query + sub * subspace_dim;
		const float *codebook = codebooks + sub * ks * subspace_dim;
		uint8_t code_idx = code[sub];

		const float *codeword = codebook + code_idx * subspace_dim;
		float dist_sub = 0.0f;

		for (int d = 0; d < subspace_dim; d++)
		{
			float diff = query_sub[d] - codeword[d];
			dist_sub += diff * diff;
		}

		total_dist += dist_sub;
	}

	distances[idx] = total_dist;
}

/*
 * Host wrapper: GPU PQ encoding
 */
extern "C" int
gpu_pq_encode_batch(const float *h_vectors,
	const float *h_codebooks,
	uint8_t *h_codes,
	int nvec,
	int dim,
	int m,
	int ks)
{
	float *d_vectors = NULL;
	float *d_codebooks = NULL;
	uint8_t *d_codes = NULL;

	int subspace_dim = dim / m;

	/* Allocate device memory */
	cudaMalloc(&d_vectors, nvec * dim * sizeof(float));
	cudaMalloc(&d_codebooks, m * ks * subspace_dim * sizeof(float));
	cudaMalloc(&d_codes, nvec * m * sizeof(uint8_t));

	/* Copy data to device */
	cudaMemcpy(d_vectors,
		h_vectors,
		nvec * dim * sizeof(float),
		cudaMemcpyHostToDevice);
	cudaMemcpy(d_codebooks,
		h_codebooks,
		m * ks * subspace_dim * sizeof(float),
		cudaMemcpyHostToDevice);

	/* Launch kernel */
	int threads = 256;
	int blocks = (nvec + threads - 1) / threads;
	pq_encode_kernel<<<blocks, threads>>>(
		d_vectors, d_codebooks, d_codes, nvec, dim, m, ks);

	/* Copy results back */
	cudaMemcpy(h_codes,
		d_codes,
		nvec * m * sizeof(uint8_t),
		cudaMemcpyDeviceToHost);

	/* Cleanup */
	cudaFree(d_vectors);
	cudaFree(d_codebooks);
	cudaFree(d_codes);

	return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/*
 * Host wrapper: GPU PQ asymmetric distance
 */
extern "C" int
gpu_pq_asymmetric_distance_batch(const float *h_query,
	const uint8_t *h_codes,
	const float *h_codebooks,
	float *h_distances,
	int nvec,
	int dim,
	int m,
	int ks)
{
	float *d_query = NULL;
	uint8_t *d_codes = NULL;
	float *d_codebooks = NULL;
	float *d_distances = NULL;

	int subspace_dim = dim / m;

	/* Allocate device memory */
	cudaMalloc(&d_query, dim * sizeof(float));
	cudaMalloc(&d_codes, nvec * m * sizeof(uint8_t));
	cudaMalloc(&d_codebooks, m * ks * subspace_dim * sizeof(float));
	cudaMalloc(&d_distances, nvec * sizeof(float));

	/* Copy data to device */
	cudaMemcpy(
		d_query, h_query, dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_codes,
		h_codes,
		nvec * m * sizeof(uint8_t),
		cudaMemcpyHostToDevice);
	cudaMemcpy(d_codebooks,
		h_codebooks,
		m * ks * subspace_dim * sizeof(float),
		cudaMemcpyHostToDevice);

	/* Launch kernel */
	int threads = 256;
	int blocks = (nvec + threads - 1) / threads;
	pq_asymmetric_distance_kernel<<<blocks, threads>>>(
		d_query, d_codes, d_codebooks, d_distances, nvec, dim, m, ks);

	/* Copy results back */
	cudaMemcpy(h_distances,
		d_distances,
		nvec * sizeof(float),
		cudaMemcpyDeviceToHost);

	/* Cleanup */
	cudaFree(d_query);
	cudaFree(d_codes);
	cudaFree(d_codebooks);
	cudaFree(d_distances);

	return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
