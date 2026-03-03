/*-------------------------------------------------------------------------
 *
 * gpu_pq_retrieval.c
 *    GPU-accelerated Product Quantization retrieval
 *
 * Implements fast asymmetric distance computation for PQ-encoded vectors.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/common/gpu_pq_retrieval.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb_gpu_backend.h"
#include "neurondb_gpu.h"
#include "neurondb_constants.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include <math.h>

/*
 * GPU-accelerated PQ asymmetric search
 *
 * Computes distances from query vector to PQ-encoded database vectors
 * and returns top-k candidates.
 */
int
neurondb_gpu_pq_asymmetric_search(const float *query,
								   const uint8_t *pq_codes,
								   const float *codebooks,
								   int num_vectors,
								   int dim,
								   int m,
								   int ks,
								   int k,
								   uint32_t *result_indices,
								   float *result_distances)
{
	const ndb_gpu_backend *backend;

	if (NDB_COMPUTE_MODE_IS_CPU())
		return -1;

	if (!neurondb_gpu_is_available())
		return -1;

	backend = ndb_gpu_get_active_backend();
	if (!backend || !backend->launch_pq_asymmetric_distance_batch)
		return -1;

	/* Allocate temporary arrays for distances and indices */
	float	   *all_distances = NULL;
	uint32_t   *all_indices = NULL;
	int			i, j;
	ndb_stream_t stream = NULL;
	int			rc;

	/* Allocate memory for all distances */
	nalloc(all_distances, float, num_vectors);
	nalloc(all_indices, uint32_t, num_vectors);
	NDB_CHECK_ALLOC(all_distances, "all_distances");
	NDB_CHECK_ALLOC(all_indices, "all_indices");

	/* Initialize indices */
	for (i = 0; i < num_vectors; i++)
		all_indices[i] = (uint32_t) i;

	/* Create stream for async execution if supported */
	if (backend->stream_create)
		backend->stream_create(&stream);

	/* Launch GPU kernel to compute all distances */
	rc = backend->launch_pq_asymmetric_distance_batch(query,
													  pq_codes,
													  codebooks,
													  all_distances,
													  num_vectors,
													  dim,
													  m,
													  ks,
													  stream);

	/* Synchronize stream if created */
	if (stream && backend->stream_synchronize)
		backend->stream_synchronize(stream);
	if (stream && backend->stream_destroy)
		backend->stream_destroy(stream);

	if (rc != 0)
	{
		pfree(all_distances);
		pfree(all_indices);
		return -1;
	}

	/* Select top-k using partial sort (heap-based for efficiency) */
	/* Use max-heap to find top-k smallest distances */
	{
		/* Build max-heap of size k */
		for (i = 0; i < k && i < num_vectors; i++)
		{
			/* Insert into heap */
			int			child = i;

			while (child > 0)
			{
				int			parent = (child - 1) / 2;

				if (all_distances[all_indices[parent]] >= all_distances[all_indices[child]])
					break;

				/* Swap */
				uint32_t	temp_idx = all_indices[parent];
				all_indices[parent] = all_indices[child];
				all_indices[child] = temp_idx;
				child = parent;
			}
		}

		/* Process remaining elements */
		for (i = k; i < num_vectors; i++)
		{
			if (all_distances[i] < all_distances[all_indices[0]])
			{
				/* Replace root */
				all_indices[0] = (uint32_t) i;

				/* Heapify down */
				int			parent = 0;

				while (true)
				{
					int			left = 2 * parent + 1;
					int			right = 2 * parent + 2;
					int			largest = parent;

					if (left < k && all_distances[all_indices[left]] > all_distances[all_indices[largest]])
						largest = left;
					if (right < k && all_distances[all_indices[right]] > all_distances[all_indices[largest]])
						largest = right;

					if (largest == parent)
						break;

					/* Swap */
					uint32_t	temp_idx = all_indices[parent];
					all_indices[parent] = all_indices[largest];
					all_indices[largest] = temp_idx;
					parent = largest;
				}
			}
		}

		/* Extract top-k in sorted order (smallest first) */
		for (i = k - 1; i >= 0; i--)
		{
			/* Swap root with last */
			uint32_t	temp_idx = all_indices[0];
			all_indices[0] = all_indices[i];
			all_indices[i] = temp_idx;

			/* Heapify down on smaller heap */
			int			parent = 0;

			while (true)
			{
				int			left = 2 * parent + 1;
				int			right = 2 * parent + 2;
				int			largest = parent;

				if (left < i && all_distances[all_indices[left]] > all_distances[all_indices[largest]])
					largest = left;
				if (right < i && all_distances[all_indices[right]] > all_distances[all_indices[largest]])
					largest = right;

				if (largest == parent)
					break;

				/* Swap */
				uint32_t	temp_idx2 = all_indices[parent];
				all_indices[parent] = all_indices[largest];
				all_indices[largest] = temp_idx2;
				parent = largest;
			}
		}
	}

	/* Copy results */
	for (i = 0; i < k && i < num_vectors; i++)
	{
		result_indices[i] = all_indices[i];
		result_distances[i] = all_distances[all_indices[i]];
	}

	pfree(all_distances);
	pfree(all_indices);

	return 0;
}

/*
 * Two-stage PQ search: coarse quantized + fine rerank
 */
int
neurondb_gpu_pq_two_stage_search(const float *query,
								 const uint8_t *pq_codes,
								 const float *codebooks,
								 const float *full_vectors,
								 int num_vectors,
								 int dim,
								 int m,
								 int ks,
								 int coarse_k,
								 int fine_k,
								 uint32_t *result_indices,
								 float *result_distances)
{
	uint32_t   *coarse_indices = NULL;
	float	   *coarse_distances = NULL;
	int			i;
	int			rc;

	/* Stage 1: Coarse search with PQ */
	nalloc(coarse_indices, uint32_t, coarse_k);
	nalloc(coarse_distances, float, coarse_k);
	NDB_CHECK_ALLOC(coarse_indices, "coarse_indices");
	NDB_CHECK_ALLOC(coarse_distances, "coarse_distances");

	rc = neurondb_gpu_pq_asymmetric_search(query,
										   pq_codes,
										   codebooks,
										   num_vectors,
										   dim,
										   m,
										   ks,
										   coarse_k,
										   coarse_indices,
										   coarse_distances);
	if (rc != 0)
	{
		pfree(coarse_indices);
		pfree(coarse_distances);
		return -1;
	}

	/* Stage 2: Fine rerank with full-precision vectors */
	/* Compute exact distances for top coarse_k candidates */
	for (i = 0; i < coarse_k && i < fine_k; i++)
	{
		uint32_t	idx = coarse_indices[i];
		const float *vec = full_vectors + idx * dim;
		float		dist = 0.0f;
		int			j;

		/* Compute L2 distance */
		for (j = 0; j < dim; j++)
		{
			float diff = query[j] - vec[j];
			dist += diff * diff;
		}
		dist = sqrtf(dist);

		result_indices[i] = idx;
		result_distances[i] = dist;
	}

	/* Sort by distance */
	for (i = 0; i < fine_k - 1; i++)
	{
		for (int j = i + 1; j < fine_k; j++)
		{
			if (result_distances[j] < result_distances[i])
			{
				uint32_t temp_idx = result_indices[i];
				float temp_dist = result_distances[i];
				result_indices[i] = result_indices[j];
				result_distances[i] = result_distances[j];
				result_indices[j] = temp_idx;
				result_distances[j] = temp_dist;
			}
		}
	}

	pfree(coarse_indices);
	pfree(coarse_distances);

	return 0;
}

