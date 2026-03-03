/*-------------------------------------------------------------------------
 *
 * gpu_hnsw_kernels.cu
 *    HIP kernels for HNSW (Hierarchical Navigable Small World) index search
 *
 * Implements GPU-accelerated HNSW graph traversal for approximate nearest
 * neighbor search on AMD GPUs. Supports both single and batch queries.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/rocm/gpu_hnsw_kernels.cu
 *
 *-------------------------------------------------------------------------
 */

#ifdef NDB_GPU_HIP

#include <hip/hip_runtime.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

#define HNSW_MAX_LEVEL 16

/*-------------------------------------------------------------------------
 * Device function: Compute L2 distance between two vectors
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
 * Device structure for candidate queue element
 *-------------------------------------------------------------------------
 */
typedef struct
{
	uint32_t block;
	float distance;
} CandidateElement;

/*-------------------------------------------------------------------------
 * Device function: Insert into min-heap (candidate queue)
 *-------------------------------------------------------------------------
 */
__device__ static void
insert_candidate(CandidateElement *candidates, int *count, int capacity,
				 uint32_t block, float distance)
{
	if (*count >= capacity)
	{
		/* Find worst candidate to replace */
		int worst_idx = 0;
		float worst_dist = candidates[0].distance;

		for (int i = 1; i < *count; i++)
		{
			if (candidates[i].distance > worst_dist)
			{
				worst_dist = candidates[i].distance;
				worst_idx = i;
			}
		}

		if (distance < worst_dist)
		{
			candidates[worst_idx].block = block;
			candidates[worst_idx].distance = distance;
		}
		return;
	}

	/* Insert at end */
	int idx = (*count)++;
	candidates[idx].block = block;
	candidates[idx].distance = distance;

	/* Bubble up */
	while (idx > 0)
	{
		int parent = (idx - 1) / 2;
		if (candidates[parent].distance <= candidates[idx].distance)
			break;

		/* Swap */
		CandidateElement temp = candidates[parent];
		candidates[parent] = candidates[idx];
		candidates[idx] = temp;
		idx = parent;
	}
}

/*-------------------------------------------------------------------------
 * Device function: Extract min from heap
 *-------------------------------------------------------------------------
 */
__device__ static bool
extract_min(CandidateElement *candidates, int *count, uint32_t *block, float *distance)
{
	if (*count == 0)
		return false;

	*block = candidates[0].block;
	*distance = candidates[0].distance;

	/* Move last to root */
	(*count)--;
	if (*count > 0)
	{
		candidates[0] = candidates[*count];

		/* Bubble down */
		int idx = 0;
		while (true)
		{
			int left = 2 * idx + 1;
			int right = 2 * idx + 2;
			int smallest = idx;

			if (left < *count && candidates[left].distance < candidates[smallest].distance)
				smallest = left;
			if (right < *count && candidates[right].distance < candidates[smallest].distance)
				smallest = right;

			if (smallest == idx)
				break;

			CandidateElement temp = candidates[idx];
			candidates[idx] = candidates[smallest];
			candidates[smallest] = temp;
			idx = smallest;
		}
	}

	return true;
}

/*-------------------------------------------------------------------------
 * Kernel: Single HNSW search
 *-------------------------------------------------------------------------
 */
__global__ static void
hnsw_search_kernel(const float *query,
				   const float *nodes,
				   const uint32_t *neighbors,
				   const int32_t *neighbor_counts,
				   const int32_t *node_levels,
				   uint32_t entry_point,
				   int entry_level,
				   int dim,
				   int m,
				   int ef_search,
				   int k,
				   uint32_t *result_blocks,
				   float *result_distances)
{
	/* Shared memory for candidate queue */
	extern __shared__ char shared_mem[];
	CandidateElement *candidates = (CandidateElement *) shared_mem;
	int *candidate_count = (int *) (shared_mem + ef_search * sizeof(CandidateElement));

	/* Initialize */
	int tid = threadIdx.x;
	if (tid == 0)
		*candidate_count = 0;
	__syncthreads();

	/* Start from entry point */
	uint32_t current = entry_point;
	int current_level = entry_level;

	/* Navigate down to layer 0 */
	while (current_level > 0)
	{
		float min_dist = FLT_MAX;
		uint32_t closest = current;

		const uint32_t *level_neighbors = neighbors + current * (current_level + 1) * m * 2 + current_level * m * 2;
		int32_t ncount = neighbor_counts[current * HNSW_MAX_LEVEL + current_level];

		for (int i = 0; i < ncount && i < m * 2; i++)
		{
			uint32_t neighbor = level_neighbors[i];
			if (neighbor == 0xFFFFFFFF)
				continue;

			const float *neighbor_vec = nodes + neighbor * dim;
			float dist = compute_l2_distance(query, neighbor_vec, dim);

			if (dist < min_dist)
			{
				min_dist = dist;
				closest = neighbor;
			}
		}

		current = closest;
		current_level--;
	}

	/* Search at layer 0 */
	*candidate_count = 0;
	const float *entry_vec = nodes + current * dim;
	float entry_dist = compute_l2_distance(query, entry_vec, dim);
	insert_candidate(candidates, candidate_count, ef_search, current, entry_dist);

	/* Search loop */
	int max_iterations = ef_search * 10;
	int iterations = 0;

	while (*candidate_count > 0 && iterations < max_iterations)
	{
		uint32_t candidate_block;
		float candidate_dist;

		if (!extract_min(candidates, candidate_count, &candidate_block, &candidate_dist))
			break;

		/* Explore neighbors */
		const uint32_t *level_neighbors = neighbors + candidate_block * m * 2;
		int32_t ncount = neighbor_counts[candidate_block * HNSW_MAX_LEVEL + 0];

		for (int i = 0; i < ncount && i < m * 2; i++)
		{
			uint32_t neighbor = level_neighbors[i];
			if (neighbor == 0xFFFFFFFF)
				continue;

			/* Check if already in candidates */
			bool is_visited = false;
			for (int j = 0; j < *candidate_count; j++)
			{
				if (candidates[j].block == neighbor)
				{
					is_visited = true;
					break;
				}
			}

			if (!is_visited)
			{
				const float *neighbor_vec = nodes + neighbor * dim;
				float dist = compute_l2_distance(query, neighbor_vec, dim);

				if (*candidate_count < ef_search || dist < candidates[0].distance)
					insert_candidate(candidates, candidate_count, ef_search, neighbor, dist);
			}
		}

		iterations++;
	}

	/* Extract top-k results */
	__syncthreads();
	if (tid == 0)
	{
		for (int i = 0; i < k && i < *candidate_count; i++)
		{
			uint32_t best_block;
			float best_dist;
			if (extract_min(candidates, candidate_count, &best_block, &best_dist))
			{
				result_blocks[i] = best_block;
				result_distances[i] = best_dist;
			}
			else
			{
				result_blocks[i] = 0xFFFFFFFF;
				result_distances[i] = FLT_MAX;
			}
		}
	}
}

/*-------------------------------------------------------------------------
 * Host function: Launch HNSW search kernel
 *-------------------------------------------------------------------------
 */
extern "C" int
gpu_hnsw_search_hip(const float *h_query,
					 const float *h_nodes,
					 const uint32_t *h_neighbors,
					 const int32_t *h_neighbor_counts,
					 const int32_t *h_node_levels,
					 uint32_t entry_point,
					 int entry_level,
					 int dim,
					 int m,
					 int ef_search,
					 int k,
					 uint32_t *h_result_blocks,
					 float *h_result_distances)
{
	float *d_query = NULL;
	float *d_nodes = NULL;
	uint32_t *d_neighbors = NULL;
	int32_t *d_neighbor_counts = NULL;
	int32_t *d_node_levels = NULL;
	uint32_t *d_result_blocks = NULL;
	float *d_result_distances = NULL;

	hipError_t err;
	size_t query_bytes = sizeof(float) * dim;
	int estimated_nodes = 10000;
	if (h_neighbor_counts != NULL)
	{
		estimated_nodes = 0;
		for (int i = 0; i < 100000 && h_neighbor_counts[i] >= 0; i++)
		{
			if (h_neighbor_counts[i] > 0)
				estimated_nodes = i + 1;
		}
	}
	
	size_t nodes_bytes = sizeof(float) * estimated_nodes * dim;
	size_t neighbors_bytes = sizeof(uint32_t) * estimated_nodes * HNSW_MAX_LEVEL * m * 2;
	size_t neighbor_counts_bytes = sizeof(int32_t) * estimated_nodes * HNSW_MAX_LEVEL;
	size_t node_levels_bytes = sizeof(int32_t) * estimated_nodes;
	size_t shared_mem_size = ef_search * sizeof(CandidateElement) + sizeof(int);

	/* Allocate device memory */
	err = hipMalloc(&d_query, query_bytes);
	if (err != hipSuccess)
		return -1;

	if (h_nodes != NULL)
	{
		err = hipMalloc(&d_nodes, nodes_bytes);
		if (err == hipSuccess)
			err = hipMemcpy(d_nodes, h_nodes, nodes_bytes, hipMemcpyHostToDevice);
		if (err != hipSuccess)
		{
			if (d_nodes) hipFree(d_nodes);
			hipFree(d_query);
			return -1;
		}
	}

	if (h_neighbors != NULL)
	{
		err = hipMalloc(&d_neighbors, neighbors_bytes);
		if (err == hipSuccess)
			err = hipMemcpy(d_neighbors, h_neighbors, neighbors_bytes, hipMemcpyHostToDevice);
		if (err != hipSuccess)
		{
			if (d_neighbors) hipFree(d_neighbors);
			if (d_nodes) hipFree(d_nodes);
			hipFree(d_query);
			return -1;
		}
	}

	if (h_neighbor_counts != NULL)
	{
		err = hipMalloc(&d_neighbor_counts, neighbor_counts_bytes);
		if (err == hipSuccess)
			err = hipMemcpy(d_neighbor_counts, h_neighbor_counts, neighbor_counts_bytes, hipMemcpyHostToDevice);
		if (err != hipSuccess)
		{
			if (d_neighbor_counts) hipFree(d_neighbor_counts);
			if (d_neighbors) hipFree(d_neighbors);
			if (d_nodes) hipFree(d_nodes);
			hipFree(d_query);
			return -1;
		}
	}

	if (h_node_levels != NULL)
	{
		err = hipMalloc(&d_node_levels, node_levels_bytes);
		if (err == hipSuccess)
			err = hipMemcpy(d_node_levels, h_node_levels, node_levels_bytes, hipMemcpyHostToDevice);
		if (err != hipSuccess)
		{
			if (d_node_levels) hipFree(d_node_levels);
			if (d_neighbor_counts) hipFree(d_neighbor_counts);
			if (d_neighbors) hipFree(d_neighbors);
			if (d_nodes) hipFree(d_nodes);
			hipFree(d_query);
			return -1;
		}
	}

	err = hipMalloc(&d_result_blocks, k * sizeof(uint32_t));
	if (err != hipSuccess)
	{
		if (d_node_levels) hipFree(d_node_levels);
		if (d_neighbor_counts) hipFree(d_neighbor_counts);
		if (d_neighbors) hipFree(d_neighbors);
		if (d_nodes) hipFree(d_nodes);
		hipFree(d_query);
		return -1;
	}

	err = hipMalloc(&d_result_distances, k * sizeof(float));
	if (err != hipSuccess)
	{
		hipFree(d_result_blocks);
		if (d_node_levels) hipFree(d_node_levels);
		if (d_neighbor_counts) hipFree(d_neighbor_counts);
		if (d_neighbors) hipFree(d_neighbors);
		if (d_nodes) hipFree(d_nodes);
		hipFree(d_query);
		return -1;
	}

	err = hipMemcpy(d_query, h_query, query_bytes, hipMemcpyHostToDevice);
	if (err != hipSuccess)
	{
		hipFree(d_result_distances);
		hipFree(d_result_blocks);
		if (d_node_levels) hipFree(d_node_levels);
		if (d_neighbor_counts) hipFree(d_neighbor_counts);
		if (d_neighbors) hipFree(d_neighbors);
		if (d_nodes) hipFree(d_nodes);
		hipFree(d_query);
		return -1;
	}

	hipMemset(d_result_blocks, 0xFF, k * sizeof(uint32_t));
	hipMemset(d_result_distances, 0, k * sizeof(float));

	/* Launch kernel */
	dim3 blocks(1);
	dim3 threads(1);
	hipLaunchKernelGGL(hnsw_search_kernel,
		dim3(blocks),
		dim3(threads),
		shared_mem_size,
		0,
		d_query,
		d_nodes,
		d_neighbors,
		d_neighbor_counts,
		d_node_levels,
		entry_point,
		entry_level,
		dim,
		m,
		ef_search,
		k,
		d_result_blocks,
		d_result_distances);

	err = hipGetLastError();
	if (err != hipSuccess)
	{
		hipFree(d_result_distances);
		hipFree(d_result_blocks);
		if (d_node_levels) hipFree(d_node_levels);
		if (d_neighbor_counts) hipFree(d_neighbor_counts);
		if (d_neighbors) hipFree(d_neighbors);
		if (d_nodes) hipFree(d_nodes);
		hipFree(d_query);
		return -1;
	}

	err = hipMemcpy(h_result_blocks, d_result_blocks, k * sizeof(uint32_t), hipMemcpyDeviceToHost);
	if (err == hipSuccess)
		err = hipMemcpy(h_result_distances, d_result_distances, k * sizeof(float), hipMemcpyDeviceToHost);

	hipFree(d_result_distances);
	hipFree(d_result_blocks);
	if (d_node_levels) hipFree(d_node_levels);
	if (d_neighbor_counts) hipFree(d_neighbor_counts);
	if (d_neighbors) hipFree(d_neighbors);
	if (d_nodes) hipFree(d_nodes);
	hipFree(d_query);

	return (err == hipSuccess) ? 0 : -1;
}

/*-------------------------------------------------------------------------
 * Batch HNSW search kernel
 *-------------------------------------------------------------------------
 */
__global__ static void
hnsw_search_batch_kernel(const float *queries,
						 const float *nodes,
						 const uint32_t *neighbors,
						 const int32_t *neighbor_counts,
						 const int32_t *node_levels,
						 uint32_t entry_point,
						 int entry_level,
						 int num_queries,
						 int dim,
						 int m,
						 int ef_search,
						 int k,
						 uint32_t *result_blocks,
						 float *result_distances)
{
	int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (query_idx >= num_queries)
		return;

	const float *query = queries + query_idx * dim;
	uint32_t *query_results = result_blocks + query_idx * k;
	float *query_distances = result_distances + query_idx * k;

	extern __shared__ char shared_mem[];
	CandidateElement *candidates = (CandidateElement *) (shared_mem + threadIdx.x * (ef_search * sizeof(CandidateElement) + sizeof(int)));
	int *candidate_count = (int *) (candidates + ef_search);

	*candidate_count = 0;

	uint32_t current = entry_point;
	int current_level = entry_level;

	while (current_level > 0)
	{
		float min_dist = FLT_MAX;
		uint32_t closest = current;

		if (neighbors != NULL && neighbor_counts != NULL)
		{
			const uint32_t *level_neighbors = neighbors + current * (current_level + 1) * m * 2 + current_level * m * 2;
			int32_t ncount = neighbor_counts[current * HNSW_MAX_LEVEL + current_level];

			for (int i = 0; i < ncount && i < m * 2; i++)
			{
				uint32_t neighbor = level_neighbors[i];
				if (neighbor == 0xFFFFFFFF)
					continue;

				if (nodes != NULL)
				{
					const float *neighbor_vec = nodes + neighbor * dim;
					float dist = compute_l2_distance(query, neighbor_vec, dim);

					if (dist < min_dist)
					{
						min_dist = dist;
						closest = neighbor;
					}
				}
			}
		}

		current = closest;
		current_level--;
	}

	if (nodes != NULL)
	{
		const float *entry_vec = nodes + current * dim;
		float entry_dist = compute_l2_distance(query, entry_vec, dim);
		insert_candidate(candidates, candidate_count, ef_search, current, entry_dist);
	}

	int max_iterations = ef_search * 10;
	int iterations = 0;

	while (*candidate_count > 0 && iterations < max_iterations)
	{
		uint32_t candidate_block;
		float candidate_dist;

		if (!extract_min(candidates, candidate_count, &candidate_block, &candidate_dist))
			break;

		if (neighbors != NULL && neighbor_counts != NULL && nodes != NULL)
		{
			const uint32_t *level_neighbors = neighbors + candidate_block * m * 2;
			int32_t ncount = neighbor_counts[candidate_block * HNSW_MAX_LEVEL + 0];

			for (int i = 0; i < ncount && i < m * 2; i++)
			{
				uint32_t neighbor = level_neighbors[i];
				if (neighbor == 0xFFFFFFFF)
					continue;

				bool is_visited = false;
				for (int j = 0; j < *candidate_count; j++)
				{
					if (candidates[j].block == neighbor)
					{
						is_visited = true;
						break;
					}
				}

				if (!is_visited)
				{
					const float *neighbor_vec = nodes + neighbor * dim;
					float dist = compute_l2_distance(query, neighbor_vec, dim);

					if (*candidate_count < ef_search || dist < candidates[0].distance)
						insert_candidate(candidates, candidate_count, ef_search, neighbor, dist);
				}
			}
		}

		iterations++;
	}

	for (int i = 0; i < k && i < *candidate_count; i++)
	{
		uint32_t best_block;
		float best_dist;
		if (extract_min(candidates, candidate_count, &best_block, &best_dist))
		{
			query_results[i] = best_block;
			query_distances[i] = best_dist;
		}
		else
		{
			query_results[i] = 0xFFFFFFFF;
			query_distances[i] = FLT_MAX;
		}
	}
}

/*-------------------------------------------------------------------------
 * Host function: Launch batch HNSW search kernel
 *-------------------------------------------------------------------------
 */
extern "C" int
gpu_hnsw_search_batch_hip(const float *h_queries,
							const float *h_nodes,
							const uint32_t *h_neighbors,
							const int32_t *h_neighbor_counts,
							const int32_t *h_node_levels,
							uint32_t entry_point,
							int entry_level,
							int num_queries,
							int dim,
							int m,
							int ef_search,
							int k,
							uint32_t *h_result_blocks,
							float *h_result_distances)
{
	float *d_queries = NULL;
	float *d_nodes = NULL;
	uint32_t *d_neighbors = NULL;
	int32_t *d_neighbor_counts = NULL;
	int32_t *d_node_levels = NULL;
	uint32_t *d_result_blocks = NULL;
	float *d_result_distances = NULL;

	hipError_t err;
	size_t queries_bytes = sizeof(float) * num_queries * dim;
	int estimated_nodes = 10000;
	size_t nodes_bytes = sizeof(float) * estimated_nodes * dim;
	size_t neighbors_bytes = sizeof(uint32_t) * estimated_nodes * HNSW_MAX_LEVEL * m * 2;
	size_t neighbor_counts_bytes = sizeof(int32_t) * estimated_nodes * HNSW_MAX_LEVEL;
	size_t node_levels_bytes = sizeof(int32_t) * estimated_nodes;
	size_t shared_mem_size = (ef_search * sizeof(CandidateElement) + sizeof(int)) * 256;

	err = hipMalloc(&d_queries, queries_bytes);
	if (err != hipSuccess)
		return -1;

	err = hipMemcpy(d_queries, h_queries, queries_bytes, hipMemcpyHostToDevice);
	if (err != hipSuccess)
	{
		hipFree(d_queries);
		return -1;
	}

	if (h_nodes != NULL)
	{
		err = hipMalloc(&d_nodes, nodes_bytes);
		if (err == hipSuccess)
			err = hipMemcpy(d_nodes, h_nodes, nodes_bytes, hipMemcpyHostToDevice);
		if (err != hipSuccess)
		{
			if (d_nodes) hipFree(d_nodes);
			hipFree(d_queries);
			return -1;
		}
	}

	if (h_neighbors != NULL)
	{
		err = hipMalloc(&d_neighbors, neighbors_bytes);
		if (err == hipSuccess)
			err = hipMemcpy(d_neighbors, h_neighbors, neighbors_bytes, hipMemcpyHostToDevice);
		if (err != hipSuccess)
		{
			if (d_neighbors) hipFree(d_neighbors);
			if (d_nodes) hipFree(d_nodes);
			hipFree(d_queries);
			return -1;
		}
	}

	if (h_neighbor_counts != NULL)
	{
		err = hipMalloc(&d_neighbor_counts, neighbor_counts_bytes);
		if (err == hipSuccess)
			err = hipMemcpy(d_neighbor_counts, h_neighbor_counts, neighbor_counts_bytes, hipMemcpyHostToDevice);
		if (err != hipSuccess)
		{
			if (d_neighbor_counts) hipFree(d_neighbor_counts);
			if (d_neighbors) hipFree(d_neighbors);
			if (d_nodes) hipFree(d_nodes);
			hipFree(d_queries);
			return -1;
		}
	}

	if (h_node_levels != NULL)
	{
		err = hipMalloc(&d_node_levels, node_levels_bytes);
		if (err == hipSuccess)
			err = hipMemcpy(d_node_levels, h_node_levels, node_levels_bytes, hipMemcpyHostToDevice);
		if (err != hipSuccess)
		{
			if (d_node_levels) hipFree(d_node_levels);
			if (d_neighbor_counts) hipFree(d_neighbor_counts);
			if (d_neighbors) hipFree(d_neighbors);
			if (d_nodes) hipFree(d_nodes);
			hipFree(d_queries);
			return -1;
		}
	}

	err = hipMalloc(&d_result_blocks, num_queries * k * sizeof(uint32_t));
	if (err != hipSuccess)
	{
		if (d_node_levels) hipFree(d_node_levels);
		if (d_neighbor_counts) hipFree(d_neighbor_counts);
		if (d_neighbors) hipFree(d_neighbors);
		if (d_nodes) hipFree(d_nodes);
		hipFree(d_queries);
		return -1;
	}

	err = hipMalloc(&d_result_distances, num_queries * k * sizeof(float));
	if (err != hipSuccess)
	{
		hipFree(d_result_blocks);
		if (d_node_levels) hipFree(d_node_levels);
		if (d_neighbor_counts) hipFree(d_neighbor_counts);
		if (d_neighbors) hipFree(d_neighbors);
		if (d_nodes) hipFree(d_nodes);
		hipFree(d_queries);
		return -1;
	}

	hipMemset(d_result_blocks, 0xFF, num_queries * k * sizeof(uint32_t));
	hipMemset(d_result_distances, 0, num_queries * k * sizeof(float));

	int threads_per_block = 256;
	int blocks = (num_queries + threads_per_block - 1) / threads_per_block;
	hipLaunchKernelGGL(hnsw_search_batch_kernel,
		dim3(blocks),
		dim3(threads_per_block),
		shared_mem_size,
		0,
		d_queries,
		d_nodes,
		d_neighbors,
		d_neighbor_counts,
		d_node_levels,
		entry_point,
		entry_level,
		num_queries,
		dim,
		m,
		ef_search,
		k,
		d_result_blocks,
		d_result_distances);

	err = hipGetLastError();
	if (err != hipSuccess)
	{
		hipFree(d_result_distances);
		hipFree(d_result_blocks);
		if (d_node_levels) hipFree(d_node_levels);
		if (d_neighbor_counts) hipFree(d_neighbor_counts);
		if (d_neighbors) hipFree(d_neighbors);
		if (d_nodes) hipFree(d_nodes);
		hipFree(d_queries);
		return -1;
	}

	err = hipMemcpy(h_result_blocks, d_result_blocks, num_queries * k * sizeof(uint32_t), hipMemcpyDeviceToHost);
	if (err == hipSuccess)
		err = hipMemcpy(h_result_distances, d_result_distances, num_queries * k * sizeof(float), hipMemcpyDeviceToHost);

	hipFree(d_result_distances);
	hipFree(d_result_blocks);
	if (d_node_levels) hipFree(d_node_levels);
	if (d_neighbor_counts) hipFree(d_neighbor_counts);
	if (d_neighbors) hipFree(d_neighbors);
	if (d_nodes) hipFree(d_nodes);
	hipFree(d_queries);

	return (err == hipSuccess) ? 0 : -1;
}

#endif /* NDB_GPU_HIP */
