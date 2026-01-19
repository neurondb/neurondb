/*-------------------------------------------------------------------------
 *
 * gpu_hnsw.c
 *    Unified GPU interface for HNSW index search
 *
 * Provides a common interface for GPU-accelerated HNSW search across
 * different GPU backends (CUDA, ROCm, Metal).
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/common/gpu_hnsw.c
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

/*
 * GPU-accelerated HNSW search
 *
 * Returns 0 on success, -1 on failure
 */
int
neurondb_gpu_hnsw_search(const float *query,
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
	const ndb_gpu_backend *backend;

	if (NDB_COMPUTE_MODE_IS_CPU())
		return -1;

	if (!neurondb_gpu_is_available())
		return -1;

	backend = ndb_gpu_get_active_backend();
	if (!backend || !backend->hnsw_search)
		return -1;

	/* Create stream for async execution */
	ndb_stream_t stream = NULL;
	if (backend->stream_create)
	{
		if (backend->stream_create(&stream) != 0)
			return -1;
	}

	int rc = backend->hnsw_search(query,
								  nodes,
								  neighbors,
								  neighbor_counts,
								  node_levels,
								  entry_point,
								  entry_level,
								  dim,
								  m,
								  ef_search,
								  k,
								  result_blocks,
								  result_distances,
								  stream);

	/* Synchronize stream */
	if (stream && backend->stream_synchronize)
		backend->stream_synchronize(stream);

	if (stream && backend->stream_destroy)
		backend->stream_destroy(stream);

	return rc;
}

/*
 * GPU-accelerated filtered HNSW search
 *
 * Performs HNSW search with integrated filter evaluation on GPU.
 * Filter blocks represent valid candidates that pass the filter predicate.
 * This reduces data transfer by filtering during search traversal.
 */
int
neurondb_gpu_hnsw_search_filtered(const float *query,
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
								   const uint32_t *filter_blocks,
								   int filter_block_count,
								   uint32_t *result_blocks,
								   float *result_distances,
								   int *result_count)
{
	const ndb_gpu_backend *backend;

	if (NDB_COMPUTE_MODE_IS_CPU())
		return -1;

	if (!neurondb_gpu_is_available())
		return -1;

	backend = ndb_gpu_get_active_backend();
	if (!backend)
		return -1;

	/* Use filtered search if backend supports it */
	if (backend->hnsw_search_filtered != NULL)
	{
		ndb_stream_t stream = NULL;

		if (backend->stream_create)
		{
			if (backend->stream_create(&stream) != 0)
				return -1;
		}

		int rc = backend->hnsw_search_filtered(query,
											   nodes,
											   neighbors,
											   neighbor_counts,
											   node_levels,
											   entry_point,
											   entry_level,
											   dim,
											   m,
											   ef_search,
											   k,
											   filter_blocks,
											   filter_block_count,
											   result_blocks,
											   result_distances,
											   result_count,
											   stream);

		if (stream && backend->stream_synchronize)
			backend->stream_synchronize(stream);
		if (stream && backend->stream_destroy)
			backend->stream_destroy(stream);

		return rc;
	}

	/* Fallback: Use regular search and filter on CPU */
	/* This is less efficient but works for all backends */
	{
		uint32_t   *candidate_blocks = NULL;
		float	   *candidate_distances = NULL;
		int			candidate_count = 0;
		int			filtered_count = 0;
		int			i, j;
		int			rc;

		/* Allocate candidate arrays */
		nalloc(candidate_blocks, uint32_t, ef_search);
		nalloc(candidate_distances, float, ef_search);

		/* Perform regular HNSW search */
		rc = neurondb_gpu_hnsw_search(query,
									  nodes,
									  neighbors,
									  neighbor_counts,
									  node_levels,
									  entry_point,
									  entry_level,
									  dim,
									  m,
									  ef_search,
									  ef_search, /* Get more candidates */
									  candidate_blocks,
									  candidate_distances);

		if (rc != 0)
		{
			pfree(candidate_blocks);
			pfree(candidate_distances);
			return -1;
		}

		/* Count valid candidates */
		for (i = 0; i < ef_search; i++)
		{
			if (candidate_blocks[i] == 0xFFFFFFFF)
				break;
			candidate_count++;
		}

		/* Filter candidates using filter_blocks set */
		/* Build hash set for fast lookup (simplified - use linear search for small sets) */
		for (i = 0; i < candidate_count && filtered_count < k; i++)
		{
			uint32_t	block = candidate_blocks[i];
			bool		passes_filter = false;

			/* Check if block is in filter set */
			if (filter_blocks != NULL && filter_block_count > 0)
			{
				for (j = 0; j < filter_block_count; j++)
				{
					if (filter_blocks[j] == block)
					{
						passes_filter = true;
						break;
					}
				}
			}
			else
			{
				/* No filter - all pass */
				passes_filter = true;
			}

			if (passes_filter)
			{
				result_blocks[filtered_count] = block;
				result_distances[filtered_count] = candidate_distances[i];
				filtered_count++;
			}
		}

		/* Fill remaining slots with invalid markers */
		for (i = filtered_count; i < k; i++)
		{
			result_blocks[i] = 0xFFFFFFFF;
			result_distances[i] = FLT_MAX;
		}

		if (result_count != NULL)
			*result_count = filtered_count;

		pfree(candidate_blocks);
		pfree(candidate_distances);

		return 0;
	}
}

/*
 * GPU-accelerated batch HNSW search
 *
 * Processes multiple queries in parallel
 */
int
neurondb_gpu_hnsw_search_batch(const float *queries,
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
	const ndb_gpu_backend *backend;

	if (NDB_COMPUTE_MODE_IS_CPU())
		return -1;

	if (!neurondb_gpu_is_available())
		return -1;

	backend = ndb_gpu_get_active_backend();
	if (!backend || !backend->hnsw_search_batch)
		return -1;

	/* Create stream for async execution */
	ndb_stream_t stream = NULL;
	if (backend->stream_create)
	{
		if (backend->stream_create(&stream) != 0)
			return -1;
	}

	int rc = backend->hnsw_search_batch(queries,
										nodes,
										neighbors,
										neighbor_counts,
										node_levels,
										entry_point,
										entry_level,
										num_queries,
										dim,
										m,
										ef_search,
										k,
										result_blocks,
										result_distances,
										stream);

	/* Synchronize stream */
	if (stream && backend->stream_synchronize)
		backend->stream_synchronize(stream);

	if (stream && backend->stream_destroy)
		backend->stream_destroy(stream);

	return rc;
}







