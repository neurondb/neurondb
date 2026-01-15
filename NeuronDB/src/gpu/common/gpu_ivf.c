/*-------------------------------------------------------------------------
 *
 * gpu_ivf.c
 *    Unified GPU interface for IVF index search
 *
 * Provides a common interface for GPU-accelerated IVF search across
 * different GPU backends (CUDA, ROCm, Metal).
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/common/gpu_ivf.c
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
 * GPU-accelerated IVF search
 *
 * Returns 0 on success, -1 on failure
 */
int
neurondb_gpu_ivf_search(const float *query,
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
	const ndb_gpu_backend *backend;

	if (NDB_COMPUTE_MODE_IS_CPU())
		return -1;

	if (!neurondb_gpu_is_available())
		return -1;

	backend = ndb_gpu_get_active_backend();
	if (!backend || !backend->ivf_search)
		return -1;

	/* Create stream for async execution */
	ndb_stream_t stream = NULL;
	if (backend->stream_create)
	{
		if (backend->stream_create(&stream) != 0)
			return -1;
	}

	int rc = backend->ivf_search(query,
								 centroids,
								 vectors,
								 list_offsets,
								 list_sizes,
								 nlists,
								 nprobe,
								 dim,
								 k,
								 result_indices,
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
 * GPU-accelerated batch IVF search
 *
 * Processes multiple queries in parallel
 */
int
neurondb_gpu_ivf_search_batch(const float *queries,
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
	const ndb_gpu_backend *backend;

	if (NDB_COMPUTE_MODE_IS_CPU())
		return -1;

	if (!neurondb_gpu_is_available())
		return -1;

	backend = ndb_gpu_get_active_backend();
	if (!backend || !backend->ivf_search_batch)
		return -1;

	/* Create stream for async execution */
	ndb_stream_t stream = NULL;
	if (backend->stream_create)
	{
		if (backend->stream_create(&stream) != 0)
			return -1;
	}

	int rc = backend->ivf_search_batch(queries,
									   centroids,
									   vectors,
									   list_offsets,
									   list_sizes,
									   num_queries,
									   nlists,
									   nprobe,
									   dim,
									   k,
									   result_indices,
									   result_distances,
									   stream);

	/* Synchronize stream */
	if (stream && backend->stream_synchronize)
		backend->stream_synchronize(stream);

	if (stream && backend->stream_destroy)
		backend->stream_destroy(stream);

	return rc;
}





