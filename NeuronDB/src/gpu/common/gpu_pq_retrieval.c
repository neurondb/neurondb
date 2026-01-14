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
	if (!backend)
		return -1;

	/* Use existing PQ asymmetric distance kernel if available */
	/* NOTE: launch_pq_asymmetric_distance_batch is not yet implemented in the backend structure.
	 * When implemented, it should be added to ndb_gpu_backend in
	 * include/neurondb_gpu_backend.h
	 */
	
	/* TODO: Implement GPU PQ asymmetric distance batch support */
	/* For now, fallback to CPU implementation */
	return -1;
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

