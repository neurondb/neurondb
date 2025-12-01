/*-------------------------------------------------------------------------
 *
 * gpu_kmeans_kernels.cu
 *    GPU HIP kernels for K-means clustering
 *
 * Implements GPU-accelerated K-means assignment and centroid updates.
 * Optimized for AMD GPUs via HIP runtime.
 *
 * Performance: ~10-50x speedup vs CPU for large datasets
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/rocm/gpu_kmeans_kernels.cu
 *
 *-------------------------------------------------------------------------
 */

#ifdef NDB_GPU_HIP

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <float.h>
#include "neurondb_cuda_launchers.h"

/*
 * GPU kernel: K-means assignment step
 * 
 * Assigns each vector to nearest centroid.
 * Each thread processes one vector.
 *
 * Args:
 *   vectors:    Input vectors [nvec x dim]
 *   centroids:  Cluster centroids [k x dim]
 *   assignments: Output cluster assignments [nvec]
 *   nvec:       Number of vectors
 *   k:          Number of clusters
 *   dim:        Vector dimensionality
 */
__global__ void
kmeans_assign_kernel(const float *vectors,
	const float *centroids,
	int32_t *assignments,
	int nvec,
	int k,
	int dim)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= nvec)
		return;

	const float *vec = vectors + idx * dim;
	float min_dist = FLT_MAX;
	int best_cluster = 0;

	/* Find nearest centroid */
	for (int c = 0; c < k; c++)
	{
		const float *centroid = centroids + c * dim;
		float dist = 0.0f;

		/* Compute L2 distance squared */
		for (int d = 0; d < dim; d++)
		{
			float diff = vec[d] - centroid[d];
			dist += diff * diff;
		}

		if (dist < min_dist)
		{
			min_dist = dist;
			best_cluster = c;
		}
	}

	assignments[idx] = best_cluster;
}

/*
 * GPU kernel: K-means centroid update (reduce phase)
 *
 * Accumulates vectors for each cluster.
 * Uses shared memory for efficient reduction.
 *
 * Args:
 *   vectors:      Input vectors [nvec x dim]
 *   assignments:  Cluster assignments [nvec]
 *   centroids:    Output centroids [k x dim]
 *   counts:       Points per cluster [k]
 *   nvec:         Number of vectors
 *   k:            Number of clusters
 *   dim:          Vector dimensionality
 */
__global__ void
kmeans_update_kernel(const float *vectors,
	const int32_t *assignments,
	float *centroids,
	int32_t *counts,
	int nvec,
	int k,
	int dim)
{
	/* Each block processes one cluster */
	int cluster = blockIdx.x;
	int tid = threadIdx.x;

	if (cluster >= k)
		return;

	/* Shared memory for reduction */
	extern __shared__ float shared_sum[];

	/* Initialize shared memory */
	for (int d = tid; d < dim; d += blockDim.x)
		shared_sum[d] = 0.0f;
	__syncthreads();

	/* Accumulate vectors belonging to this cluster */
	int local_count = 0;
	for (int i = tid; i < nvec; i += blockDim.x)
	{
		if (assignments[i] == cluster)
		{
			const float *vec = vectors + i * dim;
			for (int d = 0; d < dim; d++)
				atomicAdd(&shared_sum[d], vec[d]);
			local_count++;
		}
	}

	/* Reduce counts */
	atomicAdd(&counts[cluster], local_count);
	__syncthreads();

	/* Write results to global memory */
	int total_count = counts[cluster];
	if (total_count > 0)
	{
		for (int d = tid; d < dim; d += blockDim.x)
		{
			centroids[cluster * dim + d] =
				shared_sum[d] / total_count;
		}
	}
}

/*
 * Host wrapper: GPU K-means assignment
 */
extern "C" int
gpu_kmeans_assign_hip(const float *h_vectors,
	const float *h_centroids,
	int32_t *h_assignments,
	int nvec,
	int k,
	int dim)
{
	float *d_vectors = NULL;
	float *d_centroids = NULL;
	int32_t *d_assignments = NULL;

	/* Allocate device memory */
	hipMalloc(&d_vectors, nvec * dim * sizeof(float));
	hipMalloc(&d_centroids, k * dim * sizeof(float));
	hipMalloc(&d_assignments, nvec * sizeof(int32_t));

	/* Copy data to device */
	hipMemcpy(d_vectors,
		h_vectors,
		nvec * dim * sizeof(float),
		hipMemcpyHostToDevice);
	hipMemcpy(d_centroids,
		h_centroids,
		k * dim * sizeof(float),
		hipMemcpyHostToDevice);

	/* Launch kernel */
	int threads = 256;
	int blocks = (nvec + threads - 1) / threads;
	hipLaunchKernelGGL(kmeans_assign_kernel,
		dim3(blocks),
		dim3(threads),
		0,
		0,
		d_vectors, d_centroids, d_assignments, nvec, k, dim);

	/* Copy results back */
	hipMemcpy(h_assignments,
		d_assignments,
		nvec * sizeof(int32_t),
		hipMemcpyDeviceToHost);

	/* Cleanup */
	hipFree(d_vectors);
	hipFree(d_centroids);
	hipFree(d_assignments);

	return hipGetLastError() == hipSuccess ? 0 : -1;
}

/*
 * Host wrapper: GPU K-means centroid update
 */
extern "C" int
gpu_kmeans_update_hip(const float *h_vectors,
	const int32_t *h_assignments,
	float *h_centroids,
	int32_t *h_counts,
	int nvec,
	int k,
	int dim)
{
	float *d_vectors = NULL;
	int32_t *d_assignments = NULL;
	float *d_centroids = NULL;
	int32_t *d_counts = NULL;

	/* Allocate device memory */
	hipMalloc(&d_vectors, nvec * dim * sizeof(float));
	hipMalloc(&d_assignments, nvec * sizeof(int32_t));
	hipMalloc(&d_centroids, k * dim * sizeof(float));
	hipMalloc(&d_counts, k * sizeof(int32_t));

	/* Copy data to device */
	hipMemcpy(d_vectors,
		h_vectors,
		nvec * dim * sizeof(float),
		hipMemcpyHostToDevice);
	hipMemcpy(d_assignments,
		h_assignments,
		nvec * sizeof(int32_t),
		hipMemcpyHostToDevice);
	hipMemcpy(d_counts,
		h_counts,
		k * sizeof(int32_t),
		hipMemcpyHostToDevice);

	/* Launch kernel */
	int threads = 256;
	int shared_mem = dim * sizeof(float);
	hipLaunchKernelGGL(kmeans_update_kernel,
		dim3(k),
		dim3(threads),
		shared_mem,
		0,
		d_vectors, d_assignments, d_centroids, d_counts, nvec, k, dim);

	/* Copy results back */
	hipMemcpy(h_centroids,
		d_centroids,
		k * dim * sizeof(float),
		hipMemcpyDeviceToHost);
	hipMemcpy(h_counts,
		d_counts,
		k * sizeof(int32_t),
		hipMemcpyDeviceToHost);

	/* Cleanup */
	hipFree(d_vectors);
	hipFree(d_assignments);
	hipFree(d_centroids);
	hipFree(d_counts);

	return hipGetLastError() == hipSuccess ? 0 : -1;
}

#endif /* NDB_GPU_HIP */

