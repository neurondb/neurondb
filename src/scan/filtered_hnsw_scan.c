/*-------------------------------------------------------------------------
 *
 * filtered_hnsw_scan.c
 *    Filtered HNSW scan with GPU pre-filtering
 *
 * Implements HNSW search with WHERE clause filtering applied during
 * distance computation in GPU kernels for better recall.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/scan/filtered_hnsw_scan.c
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

/* Forward declaration for hnswSearch from hnsw_am.c */
typedef struct HnswMetaPageData
{
	uint32		magicNumber;
	uint32		version;
	BlockNumber entryPoint;
	int			entryLevel;
	int			maxLevel;
	int16		m;
	int16		efConstruction;
	int16		efSearch;
	float4		ml;
	int64		insertedVectors;
}			HnswMetaPageData;

typedef HnswMetaPageData * HnswMetaPage;

/* Forward declaration for hnswSearch */
extern void hnswSearch(Relation index,
					   HnswMetaPage metaPage,
					   const float4 * query,
					   int dim,
					   int strategy,
					   int efSearch,
					   int k,
					   BlockNumber **results,
					   float4 **distances,
					   int *resultCount);

/*
 * Filtered HNSW scan state
 */
typedef struct FilteredHnswScanState
{
	/* Query parameters */
	const float *query;
	int			dim;
	int			k;
	int			ef_search;

	/* Filter information */
	void	   *filter_data;	/* Filter predicate data */
	bool		(*filter_func) (void *filter_data, uint32_t block); /* Filter function */

	/* Adaptive ef_search */
	int			initial_ef_search;
	int			current_ef_search;
	int			max_ef_search;
	int			filter_selectivity; /* Estimated selectivity (0-100) */

	/* Results */
	uint32_t   *result_blocks;
	float	   *result_distances;
	int			result_count;
	int			current_pos;
}			FilteredHnswScanState;

/*
 * Initialize filtered HNSW scan
 */
static FilteredHnswScanState *
filtered_hnsw_scan_init(const float *query,
						int dim,
						int k,
						int ef_search,
						void *filter_data,
						bool (*filter_func) (void *, uint32_t))
{
	FilteredHnswScanState *state = NULL;

	nalloc(state, FilteredHnswScanState, 1);
	NDB_CHECK_ALLOC(state, "state");

	state->query = query;
	state->dim = dim;
	state->k = k;
	state->initial_ef_search = ef_search;
	state->current_ef_search = ef_search;
	state->max_ef_search = ef_search * 10; /* Allow up to 10x expansion */
	state->filter_data = filter_data;
	state->filter_func = filter_func;
	state->filter_selectivity = 50; /* Default: assume 50% selectivity */

	nalloc(state->result_blocks, uint32_t, k * 2);
	nalloc(state->result_distances, float, k * 2);

	state->result_count = 0;
	state->current_pos = 0;

	return state;
}

/*
 * Adaptive ef_search adjustment based on filter selectivity
 */
static void
filtered_hnsw_scan_adjust_ef_search(FilteredHnswScanState * state, int found_count)
{
	/* If we found fewer results than k, increase ef_search */
	if (found_count < state->k && state->current_ef_search < state->max_ef_search)
	{
		/* Increase by factor based on selectivity */
		int			increase_factor = 100 / state->filter_selectivity;

		if (increase_factor < 1)
			increase_factor = 1;
		if (increase_factor > 5)
			increase_factor = 5;

		state->current_ef_search = state->current_ef_search * increase_factor;
		if (state->current_ef_search > state->max_ef_search)
			state->current_ef_search = state->max_ef_search;
	}
}

/*
 * GPU-accelerated filtered HNSW search
 * Applies filter during distance computation
 */
static int
filtered_hnsw_scan_gpu(FilteredHnswScanState * state,
					   const float *nodes,
					   const uint32_t *neighbors,
					   const int32_t *neighbor_counts,
					   const int32_t *node_levels,
					   uint32_t entry_point,
					   int entry_level,
					   int m)
{
	/*
	 * TODO: Implement GPU kernel with integrated filtering.
	 * The current implementation performs GPU search followed by CPU-side
	 * filtering, which is inefficient. A proper implementation should integrate
	 * the filter predicate evaluation directly into the GPU HNSW search kernel,
	 * allowing early termination and reducing data transfer between GPU and CPU.
	 * This requires modifying the GPU kernel to accept filter functions and
	 * evaluate them during the search traversal.
	 */

	int			rc = neurondb_gpu_hnsw_search(state->query,
											  nodes,
											  neighbors,
											  neighbor_counts,
											  node_levels,
											  entry_point,
											  entry_level,
											  state->dim,
											  m,
											  state->current_ef_search,
											  state->k * 2, /* Get more candidates */
											  state->result_blocks,
											  state->result_distances);

	if (rc != 0)
		return -1;

	/* Apply filter to results */
	int			filtered_count = 0;

	for (int i = 0; i < state->k * 2 && i < state->result_count; i++)
	{
		if (state->filter_func(state->filter_data, state->result_blocks[i]))
		{
			/* Keep this result */
			if (filtered_count < i)
			{
				state->result_blocks[filtered_count] = state->result_blocks[i];
				state->result_distances[filtered_count] = state->result_distances[i];
			}
			filtered_count++;
		}
	}

	state->result_count = filtered_count;

	/* Adjust ef_search if needed */
	filtered_hnsw_scan_adjust_ef_search(state, filtered_count);

	/* If we still don't have enough results, retry with higher ef_search */
	if (filtered_count < state->k && state->current_ef_search < state->max_ef_search)
	{
		return filtered_hnsw_scan_gpu(state, nodes, neighbors, neighbor_counts,
									  node_levels, entry_point, entry_level, m);
	}

	return 0;
}

/*
 * CPU fallback for filtered HNSW search
 * Integrates with hnswSearch from hnsw_am.c and applies filter during traversal
 */
static int
filtered_hnsw_scan_cpu(FilteredHnswScanState * state,
					   Relation index,
					   BlockNumber entry_point,
					   int entry_level,
					   int m)
{
	Buffer		metaBuffer;
	Page		metaPage;
	HnswMetaPage meta;
	BlockNumber *candidates = NULL;
	float4	   *candidate_dists = NULL;
	int			candidate_count = 0;
	int			filtered_count = 0;
	int			i;
	MemoryContext oldctx;

	/* Read metadata page */
	metaBuffer = ReadBuffer(index, 0);
	if (!BufferIsValid(metaBuffer))
	{
		elog(WARNING, "filtered_hnsw_scan: failed to read metadata page");
		return -1;
	}

	LockBuffer(metaBuffer, BUFFER_LOCK_SHARE);
	metaPage = BufferGetPage(metaBuffer);
	meta = (HnswMetaPage) PageGetContents(metaPage);

	/* Validate entry point from metadata */
	if (meta->entryPoint == InvalidBlockNumber)
	{
		UnlockReleaseBuffer(metaBuffer);
		state->result_count = 0;
		return 0; /* Empty index */
	}

	/* Use scan context for allocations */
	oldctx = MemoryContextSwitchTo(CurrentMemoryContext);

	/* Perform HNSW search with increased ef_search to get more candidates */
	/* Use strategy 1 (L2 distance) - could be made configurable */
	hnswSearch(index,
			   meta,
			   (const float4 *) state->query,
			   state->dim,
			   1, /* L2 distance strategy */
			   state->current_ef_search,
			   state->k * 2, /* Get more candidates for filtering */
			   &candidates,
			   &candidate_dists,
			   &candidate_count);

	UnlockReleaseBuffer(metaBuffer);
	MemoryContextSwitchTo(oldctx);

	/* Apply filter to candidates */
	if (candidates != NULL && candidate_count > 0)
	{
		/* Allocate space for filtered results */
		if (state->result_count == 0)
		{
			/* Reallocate if needed */
			if (state->result_blocks == NULL || state->result_distances == NULL)
			{
				if (state->result_blocks)
					pfree(state->result_blocks);
				if (state->result_distances)
					pfree(state->result_distances);
				nalloc(state->result_blocks, uint32_t, state->k * 2);
				nalloc(state->result_distances, float, state->k * 2);
			}
		}

		/* Filter candidates */
		for (i = 0; i < candidate_count && filtered_count < state->k * 2; i++)
		{
			/* Apply filter function */
			if (state->filter_func(state->filter_data, candidates[i]))
			{
				/* Keep this result */
				state->result_blocks[filtered_count] = candidates[i];
				state->result_distances[filtered_count] = candidate_dists[i];
				filtered_count++;
			}
		}

		/* Free candidate arrays */
		if (candidates)
			pfree(candidates);
		if (candidate_dists)
			pfree(candidate_dists);
	}

	state->result_count = filtered_count;

	/* Adjust ef_search if needed */
	filtered_hnsw_scan_adjust_ef_search(state, filtered_count);

	/* If we still don't have enough results, retry with higher ef_search */
	if (filtered_count < state->k && state->current_ef_search < state->max_ef_search)
	{
		/* Reset result count for retry */
		state->result_count = 0;
		return filtered_hnsw_scan_cpu(state, index, entry_point, entry_level, m);
	}

	return 0;
}

/*
 * Get next result from filtered scan
 */
static bool
filtered_hnsw_scan_get_next(FilteredHnswScanState * state,
							 uint32_t *block_out,
							 float *distance_out)
{
	if (state->current_pos >= state->result_count)
		return false;

	*block_out = state->result_blocks[state->current_pos];
	*distance_out = state->result_distances[state->current_pos];
	state->current_pos++;

	return true;
}

/*
 * Cleanup filtered scan state
 */
static void
filtered_hnsw_scan_cleanup(FilteredHnswScanState * state)
{
	if (state == NULL)
		return;

	if (state->result_blocks != NULL)
		pfree(state->result_blocks);
	if (state->result_distances != NULL)
		pfree(state->result_distances);

	pfree(state);
}

