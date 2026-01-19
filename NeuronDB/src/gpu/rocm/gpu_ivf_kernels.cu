/*-------------------------------------------------------------------------
 *
 * gpu_ivf_kernels.cu
 *    HIP kernels for IVF (Inverted File) index search
 *
 * Implements GPU-accelerated IVF search with multi-probe support on AMD GPUs.
 * Supports both single and batch queries.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/rocm/gpu_ivf_kernels.cu
 *
 *-------------------------------------------------------------------------
 */

#ifdef NDB_GPU_HIP

#include <hip/hip_runtime.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

/*-------------------------------------------------------------------------
 * Device function: Compute L2 distance
 *-------------------------------------------------------------------------
 */
__device__ static float
compute_l2_distance(const float *vec1, const float *vec2, int dim)
{
	float sum = 0.0f;
	for (int i = 0; i < dim; i++)
	{
		float diff = vec1[i] - vec2[i];
		sum += diff * diff;
	}
	return sqrtf(sum);
}

/*-------------------------------------------------------------------------
 * Kernel: Single IVF search
 *-------------------------------------------------------------------------
 */
__global__ static void
ivf_search_kernel(const float *query,
				  const float *centroids,
				  const float *vectors,
				  const int32_t *list_offsets,
				  const int32_t *list_sizes,
				  int nlists,
				  int nprobe,
				  int dim,
				  int k,
				  uint32_t *result_indices,
				  float *result_distances)
{
	extern __shared__ char shared_mem[];
	uint32_t *topk_indices = (uint32_t *) shared_mem;
	float *topk_distances = (float *) (shared_mem + k * sizeof(uint32_t));

	int tid = threadIdx.x;

	if (tid < k)
	{
		topk_indices[tid] = 0xFFFFFFFF;
		topk_distances[tid] = FLT_MAX;
	}
	__syncthreads();

	float *centroid_dists = (float *) (shared_mem + k * (sizeof(uint32_t) + sizeof(float)));
	if (tid < nlists)
		centroid_dists[tid] = compute_l2_distance(query, centroids + tid * dim, dim);
	__syncthreads();

	int lists_to_probe = (nprobe < nlists) ? nprobe : nlists;
	int total_candidates = 0;
	for (int i = 0; i < lists_to_probe; i++)
		total_candidates += list_sizes[i];

	for (int candidate_idx = tid; candidate_idx < total_candidates; candidate_idx += blockDim.x)
	{
		int list_idx = 0;
		int offset_in_list = candidate_idx;
		
		for (int i = 0; i < lists_to_probe; i++)
		{
			if (offset_in_list < list_sizes[i])
			{
				list_idx = i;
				break;
			}
			offset_in_list -= list_sizes[i];
		}

		if (list_idx >= lists_to_probe)
			continue;

		int list_offset = list_offsets[list_idx];
		int global_idx = list_offset + offset_in_list;
		
		if (vectors != NULL)
		{
			const float *vec = vectors + global_idx * dim;
			float dist = compute_l2_distance(query, vec, dim);

			if (dist < topk_distances[k - 1])
			{
				int insert_pos = k - 1;
				while (insert_pos > 0 && dist < topk_distances[insert_pos - 1])
					insert_pos--;

				for (int j = k - 1; j > insert_pos; j--)
				{
					topk_indices[j] = topk_indices[j - 1];
					topk_distances[j] = topk_distances[j - 1];
				}
				topk_indices[insert_pos] = global_idx;
				topk_distances[insert_pos] = dist;
			}
		}
	}
	__syncthreads();

	if (tid < k)
	{
		result_indices[tid] = topk_indices[tid];
		result_distances[tid] = topk_distances[tid];
	}
}

/*-------------------------------------------------------------------------
 * Host function: Launch IVF search kernel
 *-------------------------------------------------------------------------
 */
extern "C" int
gpu_ivf_search_hip(const float *h_query,
					const float *h_centroids,
					const float *h_vectors,
					const int32_t *h_list_offsets,
					const int32_t *h_list_sizes,
					int nlists,
					int nprobe,
					int dim,
					int k,
					uint32_t *h_result_indices,
					float *h_result_distances)
{
	float *d_query = NULL;
	float *d_centroids = NULL;
	float *d_vectors = NULL;
	int32_t *d_list_offsets = NULL;
	int32_t *d_list_sizes = NULL;
	uint32_t *d_result_indices = NULL;
	float *d_result_distances = NULL;

	hipError_t err;
	size_t query_bytes = sizeof(float) * dim;
	size_t centroids_bytes = sizeof(float) * nlists * dim;
	size_t shared_mem_size = k * (sizeof(uint32_t) + sizeof(float)) + nlists * sizeof(float);

	err = hipMalloc(&d_query, query_bytes);
	if (err != hipSuccess)
		return -1;

	err = hipMalloc(&d_centroids, centroids_bytes);
	if (err != hipSuccess)
	{
		hipFree(d_query);
		return -1;
	}

	if (h_vectors != NULL)
	{
		int total_vectors = 0;
		if (h_list_offsets != NULL && h_list_sizes != NULL)
		{
			for (int i = 0; i < nlists; i++)
			{
				if (h_list_sizes[i] > 0 && h_list_offsets[i] + h_list_sizes[i] > total_vectors)
					total_vectors = h_list_offsets[i] + h_list_sizes[i];
			}
		}
		if (total_vectors == 0)
			total_vectors = 10000;

		size_t vectors_bytes = sizeof(float) * total_vectors * dim;
		err = hipMalloc(&d_vectors, vectors_bytes);
		if (err == hipSuccess)
			err = hipMemcpy(d_vectors, h_vectors, vectors_bytes, hipMemcpyHostToDevice);
		if (err != hipSuccess)
		{
			if (d_vectors) hipFree(d_vectors);
			hipFree(d_centroids);
			hipFree(d_query);
			return -1;
		}
	}

	if (h_list_offsets != NULL)
	{
		size_t list_offsets_bytes = sizeof(int32_t) * nlists;
		err = hipMalloc(&d_list_offsets, list_offsets_bytes);
		if (err == hipSuccess)
			err = hipMemcpy(d_list_offsets, h_list_offsets, list_offsets_bytes, hipMemcpyHostToDevice);
		if (err != hipSuccess)
		{
			if (d_list_offsets) hipFree(d_list_offsets);
			if (d_vectors) hipFree(d_vectors);
			hipFree(d_centroids);
			hipFree(d_query);
			return -1;
		}
	}

	if (h_list_sizes != NULL)
	{
		size_t list_sizes_bytes = sizeof(int32_t) * nlists;
		err = hipMalloc(&d_list_sizes, list_sizes_bytes);
		if (err == hipSuccess)
			err = hipMemcpy(d_list_sizes, h_list_sizes, list_sizes_bytes, hipMemcpyHostToDevice);
		if (err != hipSuccess)
		{
			if (d_list_sizes) hipFree(d_list_sizes);
			if (d_list_offsets) hipFree(d_list_offsets);
			if (d_vectors) hipFree(d_vectors);
			hipFree(d_centroids);
			hipFree(d_query);
			return -1;
		}
	}

	err = hipMalloc(&d_result_indices, k * sizeof(uint32_t));
	if (err != hipSuccess)
	{
		hipFree(d_centroids);
		hipFree(d_query);
		return -1;
	}

	err = hipMalloc(&d_result_distances, k * sizeof(float));
	if (err != hipSuccess)
	{
		hipFree(d_result_indices);
		hipFree(d_centroids);
		hipFree(d_query);
		return -1;
	}

	err = hipMemcpy(d_query, h_query, query_bytes, hipMemcpyHostToDevice);
	if (err == hipSuccess)
		err = hipMemcpy(d_centroids, h_centroids, centroids_bytes, hipMemcpyHostToDevice);

	if (err != hipSuccess)
	{
		hipFree(d_result_distances);
		hipFree(d_result_indices);
		hipFree(d_centroids);
		hipFree(d_query);
		return -1;
	}

	dim3 blocks(1);
	dim3 threads(256);
	hipLaunchKernelGGL(ivf_search_kernel,
		dim3(blocks),
		dim3(threads),
		shared_mem_size,
		0,
		d_query,
		d_centroids,
		d_vectors,
		d_list_offsets,
		d_list_sizes,
		nlists,
		nprobe,
		dim,
		k,
		d_result_indices,
		d_result_distances);

	err = hipGetLastError();
	if (err != hipSuccess)
	{
		hipFree(d_result_distances);
		hipFree(d_result_indices);
		hipFree(d_centroids);
		hipFree(d_query);
		return -1;
	}

	err = hipMemcpy(h_result_indices, d_result_indices, k * sizeof(uint32_t), hipMemcpyDeviceToHost);
	if (err == hipSuccess)
		err = hipMemcpy(h_result_distances, d_result_distances, k * sizeof(float), hipMemcpyDeviceToHost);

	hipFree(d_result_distances);
	hipFree(d_result_indices);
	if (d_list_sizes) hipFree(d_list_sizes);
	if (d_list_offsets) hipFree(d_list_offsets);
	if (d_vectors) hipFree(d_vectors);
	hipFree(d_centroids);
	hipFree(d_query);

	return (err == hipSuccess) ? 0 : -1;
}

/*-------------------------------------------------------------------------
 * Batch IVF search kernel
 *-------------------------------------------------------------------------
 */
__global__ static void
ivf_search_batch_kernel(const float *queries,
						const float *centroids,
						const float *vectors,
						const int32_t *list_offsets,
						const int32_t *list_sizes,
						int num_queries,
						int nlists,
						int nprobe,
						int dim,
						int k,
						uint32_t *result_indices,
						float *result_distances)
{
	int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (query_idx >= num_queries)
		return;

	const float *query = queries + query_idx * dim;
	uint32_t *query_results = result_indices + query_idx * k;
	float *query_distances = result_distances + query_idx * k;

	extern __shared__ char shared_mem[];
	uint32_t *topk_indices = (uint32_t *) (shared_mem + threadIdx.x * (k * (sizeof(uint32_t) + sizeof(float))));
	float *topk_distances = (float *) (topk_indices + k);

	for (int i = 0; i < k; i++)
	{
		topk_indices[i] = 0xFFFFFFFF;
		topk_distances[i] = FLT_MAX;
	}

	int lists_to_probe = (nprobe < nlists) ? nprobe : nlists;
	int total_candidates = 0;
	for (int i = 0; i < lists_to_probe; i++)
		total_candidates += list_sizes[i];

	for (int candidate_idx = 0; candidate_idx < total_candidates; candidate_idx++)
	{
		int list_idx = 0;
		int offset_in_list = candidate_idx;
		
		for (int i = 0; i < lists_to_probe; i++)
		{
			if (offset_in_list < list_sizes[i])
			{
				list_idx = i;
				break;
			}
			offset_in_list -= list_sizes[i];
		}

		if (list_idx >= lists_to_probe)
			continue;

		int list_offset = list_offsets[list_idx];
		int global_idx = list_offset + offset_in_list;
		
		if (vectors != NULL)
		{
			const float *vec = vectors + global_idx * dim;
			float dist = compute_l2_distance(query, vec, dim);

			if (dist < topk_distances[k - 1])
			{
				int insert_pos = k - 1;
				while (insert_pos > 0 && dist < topk_distances[insert_pos - 1])
					insert_pos--;

				for (int j = k - 1; j > insert_pos; j--)
				{
					topk_indices[j] = topk_indices[j - 1];
					topk_distances[j] = topk_distances[j - 1];
				}
				topk_indices[insert_pos] = global_idx;
				topk_distances[insert_pos] = dist;
			}
		}
	}

	for (int i = 0; i < k; i++)
	{
		query_results[i] = topk_indices[i];
		query_distances[i] = topk_distances[i];
	}
}

/*-------------------------------------------------------------------------
 * Host function: Launch batch IVF search kernel
 *-------------------------------------------------------------------------
 */
extern "C" int
gpu_ivf_search_batch_hip(const float *h_queries,
						   const float *h_centroids,
						   const float *h_vectors,
						   const int32_t *h_list_offsets,
						   const int32_t *h_list_sizes,
						   int num_queries,
						   int nlists,
						   int nprobe,
						   int dim,
						   int k,
						   uint32_t *h_result_indices,
						   float *h_result_distances)
{
	float *d_queries = NULL;
	float *d_centroids = NULL;
	float *d_vectors = NULL;
	int32_t *d_list_offsets = NULL;
	int32_t *d_list_sizes = NULL;
	uint32_t *d_result_indices = NULL;
	float *d_result_distances = NULL;

	hipError_t err;
	size_t queries_bytes = sizeof(float) * num_queries * dim;
	size_t centroids_bytes = sizeof(float) * nlists * dim;
	
	int total_vectors = 0;
	if (h_list_offsets != NULL && h_list_sizes != NULL)
	{
		for (int i = 0; i < nlists; i++)
		{
			if (h_list_sizes[i] > 0 && h_list_offsets[i] + h_list_sizes[i] > total_vectors)
				total_vectors = h_list_offsets[i] + h_list_sizes[i];
		}
	}
	if (total_vectors == 0)
		total_vectors = 10000;

	size_t vectors_bytes = sizeof(float) * total_vectors * dim;
	size_t shared_mem_size = k * (sizeof(uint32_t) + sizeof(float)) * 256;

	err = hipMalloc(&d_queries, queries_bytes);
	if (err != hipSuccess)
		return -1;

	err = hipMemcpy(d_queries, h_queries, queries_bytes, hipMemcpyHostToDevice);
	if (err != hipSuccess)
	{
		hipFree(d_queries);
		return -1;
	}

	err = hipMalloc(&d_centroids, centroids_bytes);
	if (err != hipSuccess)
	{
		hipFree(d_queries);
		return -1;
	}

	err = hipMemcpy(d_centroids, h_centroids, centroids_bytes, hipMemcpyHostToDevice);
	if (err != hipSuccess)
	{
		hipFree(d_centroids);
		hipFree(d_queries);
		return -1;
	}

	if (h_vectors != NULL)
	{
		err = hipMalloc(&d_vectors, vectors_bytes);
		if (err == hipSuccess)
			err = hipMemcpy(d_vectors, h_vectors, vectors_bytes, hipMemcpyHostToDevice);
		if (err != hipSuccess)
		{
			if (d_vectors) hipFree(d_vectors);
			hipFree(d_centroids);
			hipFree(d_queries);
			return -1;
		}
	}

	if (h_list_offsets != NULL)
	{
		size_t list_offsets_bytes = sizeof(int32_t) * nlists;
		err = hipMalloc(&d_list_offsets, list_offsets_bytes);
		if (err == hipSuccess)
			err = hipMemcpy(d_list_offsets, h_list_offsets, list_offsets_bytes, hipMemcpyHostToDevice);
		if (err != hipSuccess)
		{
			if (d_list_offsets) hipFree(d_list_offsets);
			if (d_vectors) hipFree(d_vectors);
			hipFree(d_centroids);
			hipFree(d_queries);
			return -1;
		}
	}

	if (h_list_sizes != NULL)
	{
		size_t list_sizes_bytes = sizeof(int32_t) * nlists;
		err = hipMalloc(&d_list_sizes, list_sizes_bytes);
		if (err == hipSuccess)
			err = hipMemcpy(d_list_sizes, h_list_sizes, list_sizes_bytes, hipMemcpyHostToDevice);
		if (err != hipSuccess)
		{
			if (d_list_sizes) hipFree(d_list_sizes);
			if (d_list_offsets) hipFree(d_list_offsets);
			if (d_vectors) hipFree(d_vectors);
			hipFree(d_centroids);
			hipFree(d_queries);
			return -1;
		}
	}

	err = hipMalloc(&d_result_indices, num_queries * k * sizeof(uint32_t));
	if (err != hipSuccess)
	{
		if (d_list_sizes) hipFree(d_list_sizes);
		if (d_list_offsets) hipFree(d_list_offsets);
		if (d_vectors) hipFree(d_vectors);
		hipFree(d_centroids);
		hipFree(d_queries);
		return -1;
	}

	err = hipMalloc(&d_result_distances, num_queries * k * sizeof(float));
	if (err != hipSuccess)
	{
		hipFree(d_result_indices);
		if (d_list_sizes) hipFree(d_list_sizes);
		if (d_list_offsets) hipFree(d_list_offsets);
		if (d_vectors) hipFree(d_vectors);
		hipFree(d_centroids);
		hipFree(d_queries);
		return -1;
	}

	hipMemset(d_result_indices, 0xFF, num_queries * k * sizeof(uint32_t));
	hipMemset(d_result_distances, 0, num_queries * k * sizeof(float));

	int threads_per_block = 256;
	int blocks = (num_queries + threads_per_block - 1) / threads_per_block;
	hipLaunchKernelGGL(ivf_search_batch_kernel,
		dim3(blocks),
		dim3(threads_per_block),
		shared_mem_size,
		0,
		d_queries,
		d_centroids,
		d_vectors,
		d_list_offsets,
		d_list_sizes,
		num_queries,
		nlists,
		nprobe,
		dim,
		k,
		d_result_indices,
		d_result_distances);

	err = hipGetLastError();
	if (err != hipSuccess)
	{
		hipFree(d_result_distances);
		hipFree(d_result_indices);
		if (d_list_sizes) hipFree(d_list_sizes);
		if (d_list_offsets) hipFree(d_list_offsets);
		if (d_vectors) hipFree(d_vectors);
		hipFree(d_centroids);
		hipFree(d_queries);
		return -1;
	}

	err = hipMemcpy(h_result_indices, d_result_indices, num_queries * k * sizeof(uint32_t), hipMemcpyDeviceToHost);
	if (err == hipSuccess)
		err = hipMemcpy(h_result_distances, d_result_distances, num_queries * k * sizeof(float), hipMemcpyDeviceToHost);

	hipFree(d_result_distances);
	hipFree(d_result_indices);
	if (d_list_sizes) hipFree(d_list_sizes);
	if (d_list_offsets) hipFree(d_list_offsets);
	if (d_vectors) hipFree(d_vectors);
	hipFree(d_centroids);
	hipFree(d_queries);

	return (err == hipSuccess) ? 0 : -1;
}

#endif /* NDB_GPU_HIP */
