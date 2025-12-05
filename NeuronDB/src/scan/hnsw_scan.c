/*-------------------------------------------------------------------------
 *
 * hnsw_scan.c
 *		HNSW scan node implementation with ef_search tuning
 *
 * Implements scan-time logic for HNSW including:
 * - Layer-by-layer greedy search
 * - Candidate priority queue management
 * - Result set pruning with ef_search
 * - Distance computation caching
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/scan/hnsw_scan.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_scan.h"
#include "fmgr.h"
#include "access/relscan.h"
#include "utils/rel.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "storage/bufpage.h"
#include "utils/builtins.h"
#include <math.h>
#include <float.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/* HNSW node structure (from hnsw_am.c) */
#define HNSW_MAX_LEVEL 16
#define HNSW_DEFAULT_M 16

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

typedef struct HnswNodeData
{
	ItemPointerData heapPtr;
	int			level;
	int16		dim;
	int16		neighborCount[HNSW_MAX_LEVEL];

	/*
	 * Followed by: float4 vector[dim]; BlockNumber neighbors[level+1][M*2];
	 */
}			HnswNodeData;

typedef HnswNodeData * HnswNode;

#define HnswGetVector(node) \
	((float4 *)((char *)(node) + MAXALIGN(sizeof(HnswNodeData))))

/*
 * HnswGetNeighborsSafe - Get neighbors array pointer for a specific level
 * 
 * CRITICAL: This function uses meta->m, not HNSW_DEFAULT_M. The node layout
 * on disk is determined by the m value stored in the meta page when the
 * node was created. All nodes in an index must use the same m value.
 */
static inline BlockNumber *
HnswGetNeighborsSafe(HnswNode node, int level, int m)
{
	return (BlockNumber *)((char *)node + MAXALIGN(sizeof(HnswNodeData)) +
						   node->dim * sizeof(float4) +
						   level * m * 2 * sizeof(BlockNumber));
}

/* Legacy macro - DO NOT USE, use HnswGetNeighborsSafe instead */
#define HnswGetNeighbors(node, lev) \
	HnswGetNeighborsSafe(node, lev, HNSW_DEFAULT_M)

/* Forward declarations */
static BlockNumber hnswSearchLayerGreedy(Relation index,
										 BlockNumber entryPoint,
										 const float4 * query,
										 int dim,
										 int layer,
										 int m);
static void hnswSearchLayer0(Relation index,
							 BlockNumber entryPoint,
							 const float4 * query,
							 int dim,
							 int efSearch,
							 int k,
							 int m,
							 BlockNumber * *results,
							 float4 * *distances,
							 int *resultCount);

/* Compute L2 distance between two vectors */
static float4
compute_l2_distance(const float4 * vec1, const float4 * vec2, int dim)
{
	float4		sum = 0.0f;
	int			i;

	for (i = 0; i < dim; i++)
	{
		float4		diff = vec1[i] - vec2[i];

		sum += diff * diff;
	}
	return sqrtf(sum);
}

/*
 * Priority queue element for search
 */
typedef struct HnswSearchElement
{
	BlockNumber block;
	float4		distance;
}			HnswSearchElement;

/*
 * Search state for HNSW traversal
 */
typedef struct HnswSearchState
{
	/* Search parameters */
	const		float4 *query;
	int			dim;
	int			efSearch;
	int			k;

	/* Candidate sets */
	HnswSearchElement *candidates;	/* Min-heap of candidates */
	int			candidateCount;
	int			candidateCapacity;

	HnswSearchElement *visited; /* Visited nodes */
	int			visitedCount;
	int			visitedCapacity;

	/* Result set */
	HnswSearchElement *results; /* Top-k results */
	int			resultCount;
}			HnswSearchState;

/*
 * Initialize search state
 */
static HnswSearchState *
hnswInitSearchState(const float4 * query, int dim, int efSearch, int k)
{
	HnswSearchState *state = NULL;
	HnswSearchElement *candidates = NULL;
	HnswSearchElement *visited = NULL;
	HnswSearchElement *results = NULL;

	NDB_ALLOC(state, HnswSearchState, 1);
	NDB_CHECK_ALLOC(state, "state");
	state->query = query;
	state->dim = dim;
	state->efSearch = efSearch;
	state->k = k;

	state->candidateCapacity = efSearch * 2;
	NDB_ALLOC(candidates, HnswSearchElement, state->candidateCapacity);
	state->candidates = candidates;
	NDB_CHECK_ALLOC(state->candidates, "state->candidates");
	state->candidateCount = 0;

	state->visitedCapacity = efSearch * 4;
	NDB_ALLOC(visited, HnswSearchElement, state->visitedCapacity);
	state->visited = visited;
	NDB_CHECK_ALLOC(state->visited, "state->visited");
	state->visitedCount = 0;

	NDB_ALLOC(results, HnswSearchElement, k);
	state->results = results;
	NDB_CHECK_ALLOC(state->results, "state->results");
	state->resultCount = 0;

	return state;
}

/*
 * Free search state
 */
static void
hnswFreeSearchState(HnswSearchState * state)
{
	NDB_FREE(state->candidates);
	NDB_FREE(state->visited);
	NDB_FREE(state->results);
	NDB_FREE(state);
}

/*
 * Check if node has been visited
 */
static bool
hnswIsVisited(HnswSearchState * state, BlockNumber block)
{
	int			i;

	for (i = 0; i < state->visitedCount; i++)
	{
		if (state->visited[i].block == block)
			return true;
	}
	return false;
}

/*
 * Mark node as visited
 */
static void
hnswMarkVisited(HnswSearchState * state, BlockNumber block, float4 distance)
{
	if (state->visitedCount >= state->visitedCapacity)
	{
		state->visitedCapacity *= 2;
		state->visited = (HnswSearchElement *) repalloc(state->visited,
														state->visitedCapacity * sizeof(HnswSearchElement));
	}

	state->visited[state->visitedCount].block = block;
	state->visited[state->visitedCount].distance = distance;
	state->visitedCount++;
}

/*
 * Insert candidate into priority queue (min-heap by distance)
 */
static void
hnswInsertCandidate(HnswSearchState * state, BlockNumber block, float4 distance)
{
	int			i;
	int			parent;

	if (state->candidateCount >= state->candidateCapacity)
		return;					/* Queue full */

	/* Insert at end and bubble up */
	i = state->candidateCount;
	state->candidates[i].block = block;
	state->candidates[i].distance = distance;
	state->candidateCount++;

	while (i > 0)
	{
		parent = (i - 1) / 2;
		if (state->candidates[i].distance
			>= state->candidates[parent].distance)
			break;

		/* Swap with parent */
		{
			HnswSearchElement temp = state->candidates[i];

			state->candidates[i] = state->candidates[parent];
			state->candidates[parent] = temp;
		}
		i = parent;
	}
}

/*
 * Extract minimum candidate from priority queue
 */
static bool
hnswExtractMinCandidate(HnswSearchState * state,
						BlockNumber * block,
						float4 * distance)
{
	int			i;
	int			left,
				right,
				smallest;

	if (state->candidateCount == 0)
		return false;

	/* Return root */
	*block = state->candidates[0].block;
	*distance = state->candidates[0].distance;

	/* Move last element to root and bubble down */
	state->candidateCount--;
	if (state->candidateCount > 0)
	{
		state->candidates[0] = state->candidates[state->candidateCount];

		i = 0;
		while (1)
		{
			smallest = i;
			left = 2 * i + 1;
			right = 2 * i + 2;

			if (left < state->candidateCount
				&& state->candidates[left].distance
				< state->candidates[smallest].distance)
				smallest = left;

			if (right < state->candidateCount
				&& state->candidates[right].distance
				< state->candidates[smallest].distance)
				smallest = right;

			if (smallest == i)
				break;

			/* Swap with smallest child */
			{
				HnswSearchElement temp = state->candidates[i];

				state->candidates[i] =
					state->candidates[smallest];
				state->candidates[smallest] = temp;
			}
			i = smallest;
		}
	}

	return true;
}

/*
 * Add result to result set (maintains top-k by distance)
 */
static void
hnswAddResult(HnswSearchState * state, BlockNumber block, float4 distance)
{
	int			i;
	int			worstIdx = 0;
	float4		worstDist;

	/* If result set not full, just add */
	if (state->resultCount < state->k)
	{
		state->results[state->resultCount].block = block;
		state->results[state->resultCount].distance = distance;
		state->resultCount++;
		return;
	}

	/* Find worst result (maximum distance) */
	worstDist = state->results[0].distance;
	for (i = 1; i < state->resultCount; i++)
	{
		if (state->results[i].distance > worstDist)
		{
			worstDist = state->results[i].distance;
			worstIdx = i;
		}
	}

	/* Replace if new result is better */
	if (distance < worstDist)
	{
		state->results[worstIdx].block = block;
		state->results[worstIdx].distance = distance;
	}
}

/*
 * Main HNSW search algorithm
 *
 * Implements the search_layer algorithm from the HNSW paper.
 * This is an internal helper function that performs multi-layer search.
 *
 * Algorithm:
 * 1. Start at entry point at highest layer
 * 2. Greedy search through upper layers to find entry point for layer 0
 * 3. Search at layer 0 with ef_search to get k results
 */
void
hnsw_search_layer(Relation index,
				  BlockNumber entryPoint,
				  int entryLevel,
				  const float4 * query,
				  int dim,
				  int strategy,
				  int efSearch,
				  int k,
				  BlockNumber * *results,
				  float4 * *distances,
				  int *resultCount)
{
	BlockNumber currentEntry = entryPoint;
	int			currentLevel = entryLevel;
	Buffer		metaBuffer;
	Page		metaPage;
	HnswMetaPage meta;

	if (entryPoint == InvalidBlockNumber || entryLevel < 0)
	{
		*results = NULL;
		*distances = NULL;
		*resultCount = 0;
		return;
	}

	/* Read metadata to get entry point if not provided */
	if (entryPoint == InvalidBlockNumber)
	{
		metaBuffer = ReadBuffer(index, 0);
		if (!BufferIsValid(metaBuffer))
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: ReadBuffer failed for buffer")));
		}
		LockBuffer(metaBuffer, BUFFER_LOCK_SHARE);
		metaPage = BufferGetPage(metaBuffer);
		meta = (HnswMetaPage) PageGetContents(metaPage);

		if (meta->entryPoint == InvalidBlockNumber)
		{
			UnlockReleaseBuffer(metaBuffer);
			*results = NULL;
			*distances = NULL;
			*resultCount = 0;
			return;
		}

		currentEntry = meta->entryPoint;
		currentLevel = meta->entryLevel;
		UnlockReleaseBuffer(metaBuffer);
	}

	/* Get m from meta page */
	{
		Buffer		metaBuf;
		Page		metaPg;
		HnswMetaPage metaData;
		int			m;

		metaBuf = ReadBuffer(index, 0);
		LockBuffer(metaBuf, BUFFER_LOCK_SHARE);
		metaPg = BufferGetPage(metaBuf);
		metaData = (HnswMetaPage) PageGetContents(metaPg);
		m = metaData->m;
		UnlockReleaseBuffer(metaBuf);

		/* Step 1: Greedy search through upper layers */
		while (currentLevel > 0)
		{
			currentEntry = hnswSearchLayerGreedy(index,
												 currentEntry,
												 query,
												 dim,
												 currentLevel,
												 m);
			currentLevel--;
		}

		/* Step 2: Search at layer 0 with ef_search */
		hnswSearchLayer0(index,
						 currentEntry,
						 query,
						 dim,
						 efSearch,
						 k,
						 m,
						 results,
						 distances,
						 resultCount);
	}

	elog(DEBUG1,
		 "neurondb: HNSW search_layer completed: entry=%u, level=%d, results=%d",
		 currentEntry,
		 currentLevel,
		 *resultCount);
}

/*
 * Greedy search at a single layer
 *
 * Used for navigating upper layers to find entry point for next layer.
 */
static BlockNumber
hnswSearchLayerGreedy(Relation index,
					  BlockNumber entryPoint,
					  const float4 * query,
					  int dim,
					  int layer,
					  int m)
{
	BlockNumber best = entryPoint;
	bool		changed = true;

	/* Greedy hill climbing */
	while (changed)
	{
		Buffer		buf;
		Page		page;
		HnswNode	node;
		BlockNumber *neighbors;
		int16		neighborCount;
		float4	   *nodeVector;
		float4		bestDist;
		int			i;

		changed = false;

		/* Read current best node */
		buf = ReadBuffer(index, best);
		if (!BufferIsValid(buf))
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: ReadBuffer failed for buffer")));
		}
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);

		if (PageIsEmpty(page))
		{
			UnlockReleaseBuffer(buf);
			break;
		}

		node = (HnswNode) PageGetItem(page,
									  PageGetItemId(page, FirstOffsetNumber));
		if (node == NULL)
		{
			UnlockReleaseBuffer(buf);
			break;
		}

		/* Validate node level */
		if (node->level < 0 || node->level >= HNSW_MAX_LEVEL)
		{
			elog(WARNING, "hnsw: invalid node level %d in greedy search, breaking", node->level);
			UnlockReleaseBuffer(buf);
			break;
		}

		nodeVector = HnswGetVector(node);
		if (nodeVector == NULL)
		{
			UnlockReleaseBuffer(buf);
			break;
		}

		neighbors = HnswGetNeighborsSafe(node, layer, m);
		neighborCount = node->neighborCount[layer];

		/* Validate and clamp neighborCount */
		if (neighborCount < 0)
			neighborCount = 0;
		if (neighborCount > m * 2)
		{
			elog(WARNING, "hnsw: neighborCount %d exceeds maximum, clamping to %d", neighborCount, m * 2);
			neighborCount = m * 2;
		}

		bestDist = compute_l2_distance(query, nodeVector, dim);

		/* Check each neighbor before releasing buffer */
		for (i = 0; i < neighborCount; i++)
		{
			if (neighbors[i] == InvalidBlockNumber)
				continue;

			/* Validate block number */
			if (neighbors[i] >= RelationGetNumberOfBlocks(index))
			{
				elog(WARNING, "hnsw: invalid neighbor block %u in greedy search", neighbors[i]);
				continue;
			}

			/* Read neighbor node */
			{
				Buffer		neighborBuf;
				Page		neighborPage;
				HnswNode	neighborNode;
				float4	   *neighborVector;
				float4		neighborDist;

				neighborBuf = ReadBuffer(index, neighbors[i]);
				if (!BufferIsValid(neighborBuf))
				{
					elog(WARNING, "hnsw: ReadBuffer failed for neighbor block %u", neighbors[i]);
					continue;
				}
				LockBuffer(neighborBuf, BUFFER_LOCK_SHARE);
				neighborPage = BufferGetPage(neighborBuf);

				if (PageIsEmpty(neighborPage))
				{
					UnlockReleaseBuffer(neighborBuf);
					continue;
				}

				neighborNode = (HnswNode) PageGetItem(neighborPage,
													  PageGetItemId(neighborPage, FirstOffsetNumber));
				if (neighborNode == NULL)
				{
					UnlockReleaseBuffer(neighborBuf);
					continue;
				}

				neighborVector = HnswGetVector(neighborNode);
				if (neighborVector == NULL)
				{
					UnlockReleaseBuffer(neighborBuf);
					continue;
				}

				neighborDist = compute_l2_distance(query, neighborVector, dim);

				UnlockReleaseBuffer(neighborBuf);

				/* If neighbor is closer, move to it */
				if (neighborDist < bestDist)
				{
					best = neighbors[i];
					bestDist = neighborDist;
					changed = true;
				}
			}
		}

		UnlockReleaseBuffer(buf);
	}

	elog(DEBUG1,
		 "neurondb: Greedy search at layer %d found block %u",
		 layer,
		 best);
	return best;
}

/*
 * Search at layer 0 with ef_search
 *
 * This is the main search that returns k results using the ef parameter
 * for exploration.
 */
static void
hnswSearchLayer0(Relation index,
				 BlockNumber entryPoint,
				 const float4 * query,
				 int dim,
				 int efSearch,
				 int k,
				 int m,
				 BlockNumber * *results,
				 float4 * *distances,
				 int *resultCount)
{
	HnswSearchState *state;
	BlockNumber block;
	float4		distance;
	int			i;

	state = hnswInitSearchState(query, dim, efSearch, k);

	/* Start with entry point */
	hnswInsertCandidate(
						state, entryPoint, 0.0);	/* Distance would be
													 * computed */
	hnswMarkVisited(state, entryPoint, 0.0);

	/* Process candidates */
	while (hnswExtractMinCandidate(state, &block, &distance))
	{
		Buffer		buf;
		Page		page;
		HnswNode	node;
		BlockNumber *neighbors;
		int16		neighborCount;
		float4	   *nodeVector;
		float4		neighborDist;
		float4		furthestDist;
		int			j;

		/* Skip if distance already worse than kth result */
		if (state->resultCount >= k
			&& distance > state->results[k - 1].distance)
			continue;

		/* Read current node */
		buf = ReadBuffer(index, block);
		if (!BufferIsValid(buf))
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: ReadBuffer failed for buffer")));
		}
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);

		if (PageIsEmpty(page))
		{
			UnlockReleaseBuffer(buf);
			continue;
		}

		node = (HnswNode) PageGetItem(page,
									  PageGetItemId(page, FirstOffsetNumber));
		if (node == NULL)
		{
			UnlockReleaseBuffer(buf);
			continue;
		}

		/* Validate node level */
		if (node->level < 0 || node->level >= HNSW_MAX_LEVEL)
		{
			elog(WARNING, "hnsw: invalid node level %d in layer 0 search, skipping", node->level);
			UnlockReleaseBuffer(buf);
			continue;
		}

		nodeVector = HnswGetVector(node);
		if (nodeVector == NULL)
		{
			UnlockReleaseBuffer(buf);
			continue;
		}

		neighbors = HnswGetNeighborsSafe(node, 0, m);
		neighborCount = node->neighborCount[0];

		/* Validate and clamp neighborCount */
		if (neighborCount < 0)
			neighborCount = 0;
		if (neighborCount > m * 2)
		{
			elog(WARNING, "hnsw: neighborCount %d exceeds maximum, clamping to %d", neighborCount, m * 2);
			neighborCount = m * 2;
		}

		/* Compute distance to query for current node */
		distance = compute_l2_distance(query, nodeVector, dim);

		/* Get furthest distance in current results */
		furthestDist = (state->resultCount >= k)
			? state->results[k - 1].distance
			: FLT_MAX;

		UnlockReleaseBuffer(buf);

		/* Process neighbors at layer 0 */
		for (j = 0; j < neighborCount; j++)
		{
			if (neighbors[j] == InvalidBlockNumber)
				continue;

			/* Validate block number */
			if (neighbors[j] >= RelationGetNumberOfBlocks(index))
			{
				elog(WARNING, "hnsw: invalid neighbor block %u in layer 0 search", neighbors[j]);
				continue;
			}

			/* Skip if already visited */
			if (hnswIsVisited(state, neighbors[j]))
				continue;

			/* Read neighbor node */
			buf = ReadBuffer(index, neighbors[j]);
			if (!BufferIsValid(buf))
			{
				elog(WARNING, "hnsw: ReadBuffer failed for neighbor block %u", neighbors[j]);
				continue;
			}
			LockBuffer(buf, BUFFER_LOCK_SHARE);
			page = BufferGetPage(buf);

			if (PageIsEmpty(page))
			{
				UnlockReleaseBuffer(buf);
				continue;
			}

			node = (HnswNode) PageGetItem(page,
										  PageGetItemId(page, FirstOffsetNumber));
			if (node == NULL)
			{
				UnlockReleaseBuffer(buf);
				continue;
			}

			nodeVector = HnswGetVector(node);
			if (nodeVector == NULL)
			{
				UnlockReleaseBuffer(buf);
				continue;
			}

			neighborDist = compute_l2_distance(query, nodeVector, dim);

			UnlockReleaseBuffer(buf);

			/* Add to candidates if distance is better than furthest result */
			if (neighborDist < furthestDist
				|| state->resultCount < k)
			{
				hnswInsertCandidate(state, neighbors[j], neighborDist);
				hnswMarkVisited(state, neighbors[j], neighborDist);
			}
		}

		/* Add current to results */
		hnswAddResult(state, block, distance);
	}

	/* Copy results */
	*resultCount = state->resultCount;
	if (*resultCount > 0)
	{
		NDB_DECLARE(BlockNumber *, results_ptr);
		NDB_DECLARE(float4 *, distances_ptr);
		NDB_ALLOC(results_ptr, BlockNumber, *resultCount);
		NDB_ALLOC(distances_ptr, float4, *resultCount);
		*results = results_ptr;
		*distances = distances_ptr;
		NDB_CHECK_ALLOC(*results, "*results");
		NDB_CHECK_ALLOC(*distances, "*distances");

		for (i = 0; i < *resultCount; i++)
		{
			(*results)[i] = state->results[i].block;
			(*distances)[i] = state->results[i].distance;
		}
	}
	else
	{
		*results = NULL;
		*distances = NULL;
	}

	hnswFreeSearchState(state);

	elog(DEBUG1,
		 "neurondb: HNSW layer-0 search returned %d results",
		 *resultCount);
}
