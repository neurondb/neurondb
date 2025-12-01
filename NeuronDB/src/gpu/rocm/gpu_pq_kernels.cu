/*-------------------------------------------------------------------------
 *
 * gpu_pq_kernels.cu
 *    GPU HIP kernels for Product Quantization (PQ)
 *
 * Implements GPU-accelerated PQ encoding and distance computation.
 * Optimized for batch processing of thousands of vectors on AMD GPUs.
 *
 * Performance: ~20-100x speedup vs CPU for large batches
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/rocm/gpu_pq_kernels.cu
 *
 *-------------------------------------------------------------------------
 */

#ifdef NDB_GPU_HIP

#include <hip/hip_runtime.h>
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
__global__ void
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
 * Host wrapper: GPU PQ encoding
 */
extern "C" int
gpu_pq_encode_batch_hip(const float *h_vectors,
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
	hipMalloc(&d_vectors, nvec * dim * sizeof(float));
	hipMalloc(&d_codebooks, m * ks * subspace_dim * sizeof(float));
	hipMalloc(&d_codes, nvec * m * sizeof(uint8_t));

	/* Copy data to device */
	hipMemcpy(d_vectors,
		h_vectors,
		nvec * dim * sizeof(float),
		hipMemcpyHostToDevice);
	hipMemcpy(d_codebooks,
		h_codebooks,
		m * ks * subspace_dim * sizeof(float),
		hipMemcpyHostToDevice);

	/* Launch kernel */
	int threads = 256;
	int blocks = (nvec + threads - 1) / threads;
	hipLaunchKernelGGL(pq_encode_kernel,
		dim3(blocks),
		dim3(threads),
		0,
		0,
		d_vectors, d_codebooks, d_codes, nvec, dim, m, ks);

	/* Copy results back */
	hipMemcpy(h_codes,
		d_codes,
		nvec * m * sizeof(uint8_t),
		hipMemcpyDeviceToHost);

	/* Cleanup */
	hipFree(d_vectors);
	hipFree(d_codebooks);
	hipFree(d_codes);

	return hipGetLastError() == hipSuccess ? 0 : -1;
}

#endif /* NDB_GPU_HIP */

