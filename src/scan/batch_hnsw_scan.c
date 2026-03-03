/*-------------------------------------------------------------------------
 *
 * batch_hnsw_scan.c
 *    Batch HNSW scan for multiple queries
 *
 * Processes multiple query vectors in a single GPU launch for
 * improved throughput.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/scan/batch_hnsw_scan.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_types.h"
#include "fmgr.h"
#include "access/relscan.h"
#include "utils/rel.h"
#include "storage/bufmgr.h"
#include "utils/builtins.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_gpu.h"
#include "gpu_hnsw.h"

/*
 * Batch HNSW scan state
 */
typedef struct BatchHnswScanState
{
	/* Query parameters */
	const float *queries;
	int			num_queries;
	int			dim;
	int			k;

	/* HNSW parameters */
	const float *nodes;
	const uint32_t *neighbors;
	const int32_t *neighbor_counts;
	const int32_t *node_levels;
	uint32_t	entry_point;
	int			entry_level;
	int			m;
	int			ef_search;

	/* Results */
	uint32_t   *result_blocks; /* [num_queries * k] */
	float	   *result_distances; /* [num_queries * k] */

	/* Current query position */
	int			current_query;
	int			current_result;
}			BatchHnswScanState;

/*
 * Initialize batch HNSW scan
 */
static BatchHnswScanState *
batch_hnsw_scan_init(const float *queries,
					 int num_queries,
					 int dim,
					 int k,
					 const float *nodes,
					 const uint32_t *neighbors,
					 const int32_t *neighbor_counts,
					 const int32_t *node_levels,
					 uint32_t entry_point,
					 int entry_level,
					 int m,
					 int ef_search)
{
	BatchHnswScanState *state = NULL;

	nalloc(state, BatchHnswScanState, 1);
	NDB_CHECK_ALLOC(state, "state");

	state->queries = queries;
	state->num_queries = num_queries;
	state->dim = dim;
	state->k = k;
	state->nodes = nodes;
	state->neighbors = neighbors;
	state->neighbor_counts = neighbor_counts;
	state->node_levels = node_levels;
	state->entry_point = entry_point;
	state->entry_level = entry_level;
	state->m = m;
	state->ef_search = ef_search;

	nalloc(state->result_blocks, uint32_t, num_queries * k);
	nalloc(state->result_distances, float, num_queries * k);

	state->current_query = 0;
	state->current_result = 0;

	return state;
}

/*
 * Execute batch HNSW search
 */
static int
batch_hnsw_scan_execute(BatchHnswScanState * state)
{
	/* Use GPU batch search if available */
	if (neurondb_gpu_is_available())
	{
		return neurondb_gpu_hnsw_search_batch(state->queries,
											  state->nodes,
											  state->neighbors,
											  state->neighbor_counts,
											  state->node_levels,
											  state->entry_point,
											  state->entry_level,
											  state->num_queries,
											  state->dim,
											  state->m,
											  state->ef_search,
											  state->k,
											  state->result_blocks,
											  state->result_distances);
	}

	/* CPU fallback: process queries sequentially */
	/* Note: This is inefficient - should use CPU batch operations when available */
	for (int i = 0; i < state->num_queries; i++)
	{
		/* For CPU fallback, we would call the standard HNSW search function */
		/* For now, mark as not fully implemented */
		/* In production, integrate with hnswSearchLayer0 from hnsw_scan.c */
		elog(WARNING, "batch_hnsw_scan: CPU fallback not fully implemented");
		return -1;
	}

	return 0;
}

/*
 * Get next result from batch scan
 */
static bool
batch_hnsw_scan_get_next(BatchHnswScanState * state,
						 uint32_t *block_out,
						 float *distance_out)
{
	if (state->current_query >= state->num_queries)
		return false;

	if (state->current_result >= state->k)
	{
		state->current_query++;
		state->current_result = 0;
		return batch_hnsw_scan_get_next(state, block_out, distance_out);
	}

	int			idx = state->current_query * state->k + state->current_result;

	if (state->result_blocks[idx] == 0xFFFFFFFF) /* InvalidBlockNumber */
		return false;

	*block_out = state->result_blocks[idx];
	*distance_out = state->result_distances[idx];
	state->current_result++;

	return true;
}

/*
 * Cleanup batch scan state
 */
static void
batch_hnsw_scan_cleanup(BatchHnswScanState * state)
{
	if (state == NULL)
		return;

	if (state->result_blocks != NULL)
		pfree(state->result_blocks);
	if (state->result_distances != NULL)
		pfree(state->result_distances);

	pfree(state);
}

