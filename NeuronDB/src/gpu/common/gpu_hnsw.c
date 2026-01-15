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





