/*-------------------------------------------------------------------------
 *
 * hnsw_am.c
 *	  HNSW (Hierarchical Navigable Small World) Index Access Method
 *
 * Implementation of HNSW index as a PostgreSQL Index Access Method:
 * - Probabilistic multi-layer graph
 * - Bidirectional link maintenance
 * - ef_construction and ef_search parameters
 * - Insert, delete, search, update, bulkdelete, vacuum, costestimate, etc.
 *
 * Based on the paper:
 * "Efficient and robust approximate nearest neighbor search using
 *  Hierarchical Navigable Small World graphs" by Malkov & Yashunin (2018)
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/index/hnsw_am.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_types.h"
#include "fmgr.h"

/* Forward declaration for fp16_to_float from quantization.c */
extern float fp16_to_float(uint16 h);
#include "access/amapi.h"
#include "access/generic_xlog.h"
#include "access/htup_details.h"
#include "access/reloptions.h"
#include "access/relscan.h"
#include "access/tableam.h"
#include "catalog/index.h"
#include "catalog/pg_am.h"
#include "catalog/pg_type.h"
#include "catalog/pg_namespace.h"
#include "commands/vacuum.h"
#include "miscadmin.h"
#include "nodes/execnodes.h"
#include "storage/bufmgr.h"
#include "storage/indexfsm.h"
#include "storage/lmgr.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "optimizer/cost.h"
#include "utils/typcache.h"
#include "utils/syscache.h"
#include "utils/lsyscache.h"
#include "parser/parse_type.h"
#include "nodes/parsenodes.h"
#include "nodes/makefuncs.h"
#include "funcapi.h"
#include "utils/varbit.h"
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/*
 * HNSW AM type definitions and constants
 *
 * IMPORTANT: HNSW index uses ONE NODE PER PAGE. This is a fundamental
 * design constraint. Each page contains exactly one HnswNode structure.
 * This assumption is used throughout the code for:
 * - Page layout (PageIsEmpty checks before insert)
 * - Node access (always uses FirstOffsetNumber)
 * - Neighbor removal (assumes single item per page)
 * - Bulk delete (assumes first item is the node)
 *
 * If this constraint is violated (e.g., by other code adding items
 * to HNSW index pages), the index will become corrupted.
 */
#define HNSW_DEFAULT_M			16
#define HNSW_DEFAULT_EF_CONSTRUCTION	200
#define HNSW_DEFAULT_EF_SEARCH		64
#define HNSW_DEFAULT_ML			0.36f
#define HNSW_MAX_LEVEL			16
#define HNSW_MAGIC_NUMBER		0x48534E57
#define HNSW_VERSION			1

/* Maximum and minimum values for m parameter */
#define HNSW_MIN_M				2
#define HNSW_MAX_M				128
#define HNSW_MIN_EF_CONSTRUCTION	4
#define HNSW_MAX_EF_CONSTRUCTION	10000
#define HNSW_MIN_EF_SEARCH		4
#define HNSW_MAX_EF_SEARCH		10000

/* Reloption kind - registered in _PG_init() */
extern int	relopt_kind_hnsw;

typedef struct HnswOptions
{
	int32		vl_len_;
	int			m;
	int			ef_construction;
	int			ef_search;
}			HnswOptions;

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

/* Maximum visited array size to prevent excessive memory allocation */
#define HNSW_MAX_VISITED_CAPACITY (1024 * 1024)  /* 1M entries max */

/*
 * HnswGetVector - Get vector data pointer from node
 */
#define HnswGetVector(node) \
	((float4 *)((char *)(node) + MAXALIGN(sizeof(HnswNodeData))))

/*
 * HnswGetNeighborsWithM - Get neighbors array pointer for a specific level.
 *
 * Node layout on disk is determined by the m value stored in the meta page
 * when the node was created. All nodes in an index must use the same m value.
 */
#define HnswGetNeighborsWithM(node, lev, m) \
	((BlockNumber *)((char *)(node) + MAXALIGN(sizeof(HnswNodeData)) \
		+ (node)->dim * sizeof(float4) \
		+ (lev) * (m) * 2 * sizeof(BlockNumber)))

/*
 * HnswGetNeighborsSafe - Get neighbors array using m from meta page.
 */
static inline BlockNumber *
HnswGetNeighborsSafe(HnswNode node, int level, int m)
{
	return (BlockNumber *)((char *)node + MAXALIGN(sizeof(HnswNodeData)) +
						   node->dim * sizeof(float4) +
						   level * m * 2 * sizeof(BlockNumber));
}

/*
 * HnswNodeSizeWithM - Calculate node size with specific m value.
 *
 * Node size depends on the m parameter. All nodes in an index must use
 * the same m value that matches meta->m.
 */
static inline Size
HnswNodeSizeWithM(int dim, int level, int m)
{
	return MAXALIGN(sizeof(HnswNodeData) +
					dim * sizeof(float4) +
					(level + 1) * m * 2 * sizeof(BlockNumber));
}

/* Legacy macros - use HnswNodeSizeWithM and HnswGetNeighborsSafe instead */
#define HnswNodeSize(dim, level) \
	HnswNodeSizeWithM(dim, level, HNSW_DEFAULT_M)

#define HnswGetNeighbors(node, lev) \
	HnswGetNeighborsWithM(node, lev, HNSW_DEFAULT_M)

/*
 * Build state for index build
 */
typedef struct HnswBuildState
{
	Relation	heap;
	Relation	index;
	IndexInfo  *indexInfo;
	double		indtuples;
	MemoryContext tmpCtx;
}			HnswBuildState;

/*
 * Opaque for scan state
 */
typedef struct HnswScanOpaqueData
{
	int			efSearch;
	int			strategy;
	Vector	   *queryVector;
	int			k;
	bool		firstCall;
	int			resultCount;
	BlockNumber *results;
	float4	   *distances;
	int			currentResult;
}			HnswScanOpaqueData;

typedef HnswScanOpaqueData * HnswScanOpaque;

/*
 * Forward declarations
 */
static IndexBuildResult * hnswbuild(Relation heap, Relation index, IndexInfo * indexInfo);
static void hnswbuildempty(Relation index);
static bool hnswinsert(Relation index, Datum * values, bool *isnull, ItemPointer ht_ctid,
					   Relation heapRel, IndexUniqueCheck checkUnique,
					   bool indexUnchanged, struct IndexInfo *indexInfo);
static IndexBulkDeleteResult * hnswbulkdelete(IndexVacuumInfo * info,
											  IndexBulkDeleteResult * stats,
											  IndexBulkDeleteCallback callback,
											  void *callback_state);
static IndexBulkDeleteResult * hnswvacuumcleanup(IndexVacuumInfo * info,
												 IndexBulkDeleteResult * stats);
static bool hnswdelete(Relation index, ItemPointer tid, Datum * values, bool *isnull,
					   Relation heapRel, struct IndexInfo *indexInfo) __attribute__((unused));
static bool hnswupdate(Relation index, ItemPointer tid, Datum * values, bool *isnull,
					   ItemPointer otid, Relation heapRel, struct IndexInfo *indexInfo) __attribute__((unused));
static void hnswcostestimate(struct PlannerInfo *root, struct IndexPath *path, double loop_count,
							 Cost * indexStartupCost, Cost * indexTotalCost,
							 Selectivity * indexSelectivity, double *indexCorrelation,
							 double *indexPages);
static bytea * hnswoptions(Datum reloptions, bool validate);
static void hnswRemoveNodeFromNeighbor(Relation index,
									   BlockNumber neighborBlkno,
									   BlockNumber nodeBlkno,
									   int level);
static bool hnswproperty(Oid index_oid, int attno, IndexAMProperty prop,
						 const char *propname, bool *res, bool *isnull);
static IndexScanDesc hnswbeginscan(Relation index, int nkeys, int norderbys);
static void hnswrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys);
static bool hnswgettuple(IndexScanDesc scan, ScanDirection dir);
static void hnswendscan(IndexScanDesc scan);
static void hnswLoadOptions(Relation index, HnswOptions *opts_out);

static void hnswInitMetaPage(Buffer metaBuffer, int16 m, int16 efConstruction, int16 efSearch, float4 ml);
static int	hnswGetRandomLevel(float4 ml);
static float4 hnswComputeDistance(const float4 * vec1, const float4 * vec2, int dim, int strategy) __attribute__((unused));
static void hnswSearch(Relation index, HnswMetaPage metaPage, const float4 * query,
					   int dim, int strategy, int efSearch, int k,
					   BlockNumber * *results, float4 * *distances, int *resultCount);
static void hnswInsertNode(Relation index, HnswMetaPage metaPage,
						   const float4 * vector, int dim, ItemPointer heapPtr);
static float4 * hnswExtractVectorData(Datum value, Oid typeOid, int *out_dim, MemoryContext ctx);
static Oid hnswGetKeyType(Relation index, int attno);
static void hnswBuildCallback(Relation index, ItemPointer tid, Datum * values,
							  bool *isnull, bool tupleIsAlive, void *state);

/* Safety validation helpers */
static int16 hnswValidateNeighborCount(int16 neighborCount, int m, int level);
static bool hnswValidateLevelSafe(int level);  /* Returns false instead of ERROR */
static bool hnswValidateBlockNumber(BlockNumber blkno, Relation index);
static Size hnswComputeNodeSizeSafe(int dim, int level, int m, bool *overflow);
static void hnswCacheTypeOids(void);

/* Cached type OIDs - initialized once */
static Oid cached_vectorOid = InvalidOid;
static Oid cached_halfvecOid = InvalidOid;
static Oid cached_sparsevecOid = InvalidOid;
static Oid cached_bitOid = InvalidOid;
static bool typeOidsCached = false;

/*
 * SQL-callable handler function
 */
PG_FUNCTION_INFO_V1(hnsw_handler);

Datum
hnsw_handler(PG_FUNCTION_ARGS)
{
	IndexAmRoutine *amroutine;

	amroutine = makeNode(IndexAmRoutine);
	amroutine->amstrategies = 0;
	amroutine->amsupport = 1;
	amroutine->amoptsprocnum = 0;
	amroutine->amcanorder = false;
	amroutine->amcanorderbyop = true;
	amroutine->amcanbackward = false;
	amroutine->amcanunique = false;
	amroutine->amcanmulticol = false;
	amroutine->amoptionalkey = true;
	amroutine->amsearcharray = false;
	amroutine->amsearchnulls = false;
	amroutine->amstorage = false;
	amroutine->amclusterable = false;
	amroutine->ampredlocks = false;
	amroutine->amcanparallel = true;
	amroutine->amcaninclude = false;
	amroutine->amusemaintenanceworkmem = false;
	amroutine->amsummarizing = false;
	amroutine->amparallelvacuumoptions = 0;
	amroutine->amkeytype = InvalidOid;

	amroutine->ambuild = hnswbuild;
	amroutine->ambuildempty = hnswbuildempty;
	amroutine->aminsert = hnswinsert;
	amroutine->ambulkdelete = hnswbulkdelete;
	amroutine->amvacuumcleanup = hnswvacuumcleanup;
	amroutine->amcanreturn = NULL;
	amroutine->amcostestimate = hnswcostestimate;
	amroutine->amoptions = hnswoptions;
	amroutine->amproperty = hnswproperty;
	amroutine->ambuildphasename = NULL;
	amroutine->amvalidate = NULL;
	amroutine->amadjustmembers = NULL;
	amroutine->ambeginscan = hnswbeginscan;
	amroutine->amrescan = hnswrescan;
	amroutine->amgettuple = hnswgettuple;
	amroutine->amgetbitmap = NULL;
	amroutine->amendscan = hnswendscan;
	amroutine->ammarkpos = NULL;
	amroutine->amrestrpos = NULL;
	amroutine->amestimateparallelscan = NULL;
	amroutine->aminitparallelscan = NULL;
	amroutine->amparallelrescan = NULL;

	PG_RETURN_POINTER(amroutine);
}

/*
 * Index Build
 */
static IndexBuildResult *
hnswbuild(Relation heap, Relation index, IndexInfo * indexInfo)
{
	HnswBuildState buildstate = {0};
	Buffer		metaBuffer;
	Page		metaPage = NULL;  /* Suppress unused variable warning */
	HnswOptions *options;
	IndexBuildResult *result = NULL;
	int			m,
				ef_construction,
				ef_search;

	elog(INFO, "neurondb: Building HNSW index on %s", RelationGetRelationName(index));

	buildstate.heap = heap;
	buildstate.index = index;
	buildstate.indexInfo = indexInfo;
	buildstate.indtuples = 0;
	buildstate.tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
											  "HNSW build temporary context",
											  ALLOCSET_DEFAULT_SIZES);

	/* Initialize metadata page on block 0 */
	metaBuffer = ReadBuffer(index, 0);
	if (!BufferIsValid(metaBuffer))
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ReadBuffer failed")));
	}
	LockBuffer(metaBuffer, BUFFER_LOCK_EXCLUSIVE);
	metaPage = BufferGetPage(metaBuffer);
	if (PageIsNew(metaPage))
		PageInit(metaPage, BufferGetPageSize(metaBuffer), sizeof(HnswMetaPageData));

	options = (HnswOptions *) indexInfo->ii_AmCache;
	if (options == NULL)
	{
		HnswOptions opts;

		HnswOptions *cached_opts = NULL;
		hnswLoadOptions(index, &opts);
		NDB_ALLOC(cached_opts, HnswOptions, 1);
		*cached_opts = opts;
		options = cached_opts;
		indexInfo->ii_AmCache = (void *) options;
	}
	m = options->m;
	ef_construction = options->ef_construction;
	ef_search = options->ef_search;

	hnswInitMetaPage(metaBuffer, m, ef_construction, ef_search, HNSW_DEFAULT_ML);

	MarkBufferDirty(metaBuffer);
	UnlockReleaseBuffer(metaBuffer);

	/* Use parallel scan if available */
	buildstate.indtuples = table_index_build_scan(heap, index, indexInfo,
												  true, true, hnswBuildCallback,
												  (void *) &buildstate, NULL);

	{
		NDB_ALLOC(result, IndexBuildResult, 1);
		result->heap_tuples = buildstate.indtuples;
		result->index_tuples = buildstate.indtuples;

		MemoryContextDelete(buildstate.tmpCtx);
		elog(INFO, "neurondb: HNSW index build complete, indexed %.0f tuples",
			 buildstate.indtuples);

		return result;
	}
}

/*
 * hnswBuildCallback
 *    Callback function invoked during index build for each heap tuple.
 *
 * This function is called by PostgreSQL's index build infrastructure for
 * each tuple in the heap relation being indexed. It extracts the vector
 * value from the tuple, determines its target layer level using the
 * probabilistic level assignment algorithm, and inserts it into the HNSW
 * graph structure at the appropriate layers. The insertion process
 * maintains bidirectional links between nodes, ensuring that each node
 * has connections to its nearest neighbors at each level it participates
 * in. This callback operates within a transaction context and uses the
 * temporary memory context provided in the build state for intermediate
 * allocations, ensuring that memory is properly managed during bulk index
 * construction operations.
 */
static void
hnswBuildCallback(Relation index, ItemPointer tid, Datum * values,
				  bool *isnull, bool tupleIsAlive, void *state)
{
	HnswBuildState *buildstate = (HnswBuildState *) state;

	hnswinsert(index, values, isnull, tid, buildstate->heap,
			   UNIQUE_CHECK_NO, true, buildstate->indexInfo);

	buildstate->indtuples++;
}

static void
hnswbuildempty(Relation index)
{
	Buffer		metaBuffer;
	Page		metaPage;
	HnswOptions opts;

	/* Load options from relation to match CREATE INDEX reloptions */
	hnswLoadOptions(index, &opts);

	/* Initialize metadata page on block 0 */
	metaBuffer = ReadBuffer(index, 0);
	if (!BufferIsValid(metaBuffer))
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ReadBuffer failed")));
	}
	LockBuffer(metaBuffer, BUFFER_LOCK_EXCLUSIVE);
	metaPage = BufferGetPage(metaBuffer);
	if (PageIsNew(metaPage))
		PageInit(metaPage, BufferGetPageSize(metaBuffer), sizeof(HnswMetaPageData));

	hnswInitMetaPage(metaBuffer,
					 opts.m,
					 opts.ef_construction,
					 opts.ef_search,
					 HNSW_DEFAULT_ML);

	MarkBufferDirty(metaBuffer);
	UnlockReleaseBuffer(metaBuffer);
}

static bool
hnswinsert(Relation index,
		   Datum * values,
		   bool *isnull,
		   ItemPointer ht_ctid,
		   Relation heapRel,
		   IndexUniqueCheck checkUnique,
		   bool indexUnchanged,
		   struct IndexInfo *indexInfo)
{
	float4	   *vectorData;
	int			dim;
	Buffer		metaBuffer;
	Page		metaPage;
	HnswMetaPage meta;
	Oid			keyType;
	MemoryContext oldctx;

	if (isnull[0])
		return false;

	keyType = hnswGetKeyType(index, 1);

	oldctx = MemoryContextSwitchTo(CurrentMemoryContext);
	vectorData = hnswExtractVectorData(values[0], keyType, &dim, CurrentMemoryContext);
	MemoryContextSwitchTo(oldctx);

	if (vectorData == NULL)
		return false;

	metaBuffer = InvalidBuffer;
	PG_TRY();
	{
		metaBuffer = ReadBuffer(index, 0);
		LockBuffer(metaBuffer, BUFFER_LOCK_EXCLUSIVE);
		metaPage = BufferGetPage(metaBuffer);
		meta = (HnswMetaPage) PageGetContents(metaPage);

		hnswInsertNode(index, meta, vectorData, dim, ht_ctid);

		MarkBufferDirty(metaBuffer);
		UnlockReleaseBuffer(metaBuffer);
		metaBuffer = InvalidBuffer;
	}
	PG_CATCH();
	{
		if (BufferIsValid(metaBuffer))
		{
			LockBuffer(metaBuffer, BUFFER_LOCK_UNLOCK);
			ReleaseBuffer(metaBuffer);
			metaBuffer = InvalidBuffer;
		}
		NDB_FREE(vectorData);
		vectorData = NULL;
		PG_RE_THROW();
	}
	PG_END_TRY();

	NDB_FREE(vectorData);
	vectorData = NULL;

	return true;
}

/*
 * Bulk delete implementation: iteratively calls callback and removes nodes
 * from HNSW graph structure.
 */
static IndexBulkDeleteResult *
hnswbulkdelete(IndexVacuumInfo * info,
			   IndexBulkDeleteResult * stats,
			   IndexBulkDeleteCallback callback,
			   void *callback_state)
{
	Relation	index = info->index;
	BlockNumber blkno;
	Buffer		metaBuffer;
	Page		metaPage;
	HnswMetaPage meta;
	Buffer		nodeBuf;
	Page		nodePage;
	OffsetNumber maxoff;
	OffsetNumber offnum;
	HnswNode	node;
	BlockNumber *neighbors;
	int16		neighborCount;
	int			level;
	int			i;
	bool		foundNewEntry;
	ItemId		itemId;

	NDB_DECLARE(IndexBulkDeleteResult *, new_stats);

	if (stats == NULL)
	{
		NDB_ALLOC(new_stats, IndexBulkDeleteResult, 1);
		memset(new_stats, 0, sizeof(IndexBulkDeleteResult));
		stats = new_stats;
	}

	/* Read metadata page */
	metaBuffer = ReadBuffer(index, 0);
	LockBuffer(metaBuffer, BUFFER_LOCK_EXCLUSIVE);
	metaPage = BufferGetPage(metaBuffer);
	meta = (HnswMetaPage) PageGetContents(metaPage);

	/* Scan all pages in the index */
	for (blkno = 1; blkno < RelationGetNumberOfBlocks(index); blkno++)
	{
		nodeBuf = ReadBuffer(index, blkno);
		LockBuffer(nodeBuf, BUFFER_LOCK_EXCLUSIVE);
		nodePage = BufferGetPage(nodeBuf);

		if (PageIsNew(nodePage) || PageIsEmpty(nodePage))
		{
			UnlockReleaseBuffer(nodeBuf);
			continue;
		}

		maxoff = PageGetMaxOffsetNumber(nodePage);
		for (offnum = FirstOffsetNumber; offnum <= maxoff;
			 offnum = OffsetNumberNext(offnum))
		{
			itemId = PageGetItemId(nodePage, offnum);

			if (!ItemIdIsValid(itemId) || ItemIdIsDead(itemId))
				continue;

			node = (HnswNode) PageGetItem(nodePage, itemId);
			if (node == NULL)
				continue;

			/* Validate node level */
			if (!hnswValidateLevelSafe(node->level))
			{
				elog(WARNING, "hnsw: invalid node level %d in bulk delete at block %u, skipping",
					 node->level, blkno);
				continue;
			}

			/* Check callback to see if this tuple should be deleted */
			if (callback(&node->heapPtr, callback_state))
			{
				/* Remove node from graph structure */
				for (level = 0; level <= node->level; level++)
				{
					neighbors = HnswGetNeighborsSafe(node, level, meta->m);
					neighborCount = node->neighborCount[level];

					/* Validate and clamp neighborCount */
					neighborCount = hnswValidateNeighborCount(neighborCount, meta->m, level);

					for (i = 0; i < neighborCount; i++)
					{
						if (neighbors[i] != InvalidBlockNumber &&
							hnswValidateBlockNumber(neighbors[i], index))
						{
							hnswRemoveNodeFromNeighbor(index,
													   neighbors[i],
													   blkno,
													   level);
						}
					}
				}

				if (meta->entryPoint == blkno)
				{
					foundNewEntry = false;

					for (level = node->level;
						 level >= 0 && !foundNewEntry;
						 level--)
					{
						neighbors = HnswGetNeighborsSafe(node, level, meta->m);
						neighborCount = node->neighborCount[level];

						/* Validate and clamp neighborCount */
						neighborCount = hnswValidateNeighborCount(neighborCount, meta->m, level);

						for (i = 0; i < neighborCount && !foundNewEntry; i++)
						{
							if (neighbors[i] != InvalidBlockNumber &&
								hnswValidateBlockNumber(neighbors[i], index))
							{
								/* Use first valid neighbor as new entry point */
								{
									Buffer		tmpBuf;
									Page		tmpPage;
									HnswNode	tmpNode;

									tmpBuf = ReadBuffer(index, neighbors[i]);
									LockBuffer(tmpBuf, BUFFER_LOCK_SHARE);
									tmpPage = BufferGetPage(tmpBuf);
									if (!PageIsEmpty(tmpPage))
									{
										tmpNode = (HnswNode) PageGetItem(tmpPage,
																		 PageGetItemId(tmpPage,
																					   FirstOffsetNumber));
										if (tmpNode != NULL && hnswValidateLevelSafe(tmpNode->level))
										{
											meta->entryPoint = neighbors[i];
											meta->entryLevel = tmpNode->level;
											foundNewEntry = true;
										}
									}
									UnlockReleaseBuffer(tmpBuf);
								}
							}
						}
					}

					/* If no neighbor found, mark entry as invalid */
					if (!foundNewEntry)
					{
						meta->entryPoint = InvalidBlockNumber;
						meta->entryLevel = -1;
					}
				}

				/* Mark node as deleted */
				ItemIdSetDead(itemId);
				MarkBufferDirty(nodeBuf);

				stats->tuples_removed++;
				meta->insertedVectors--;
				if (meta->insertedVectors < 0)
					meta->insertedVectors = 0;
			}
		}

		UnlockReleaseBuffer(nodeBuf);
	}

	if (stats->tuples_removed > 0)
		MarkBufferDirty(metaBuffer);

	UnlockReleaseBuffer(metaBuffer);

	return stats;
}

/*
 * Vacuum cleanup: just create result if stats not provided
 */
static IndexBulkDeleteResult *
hnswvacuumcleanup(IndexVacuumInfo * info, IndexBulkDeleteResult * stats)
{
	NDB_DECLARE(IndexBulkDeleteResult *, new_stats);

	if (stats == NULL)
	{
		NDB_ALLOC(new_stats, IndexBulkDeleteResult, 1);
		memset(new_stats, 0, sizeof(IndexBulkDeleteResult));
		stats = new_stats;
	}
	return stats;
}

static void
hnswcostestimate(struct PlannerInfo *root,
				 struct IndexPath *path,
				 double loop_count,
				 Cost * indexStartupCost,
				 Cost * indexTotalCost,
				 Selectivity * indexSelectivity,
				 double *indexCorrelation,
				 double *indexPages)
{
	Relation	index;
	BlockNumber numPages;
	double		numTuples;
	double		efSearch = 64.0;	/* Default, can be improved by reading from meta */
	double		cpu_cost = 0.0025;	/* Default CPU operator cost */

	/* Get relation from index OID */
	index = index_open(path->indexinfo->indexoid, AccessShareLock);

	/* Get index size */
	numPages = RelationGetNumberOfBlocks(index);
	numTuples = index->rd_rel->reltuples;
	if (numTuples < 1.0)
		numTuples = 1.0;

	/* Estimate pages based on actual index size */
	*indexPages = (double) numPages;

	/* Startup cost: reading meta page + initial search setup */
	*indexStartupCost = 1.0;

	/* Total cost: based on ef_search and index size
	 * HNSW search typically examines ef_search candidates
	 * Cost per tuple is roughly log(numTuples) * ef_search operations
	 */
	*indexTotalCost = *indexStartupCost + (log(numTuples) * efSearch * cpu_cost);

	/* Release lock */
	index_close(index, AccessShareLock);

	/* Selectivity: approximate based on k / total tuples */
	if (path->indexselectivity > 0.0)
		*indexSelectivity = path->indexselectivity;
	else
		*indexSelectivity = Min(1.0, 10.0 / numTuples);	/* Default k=10 */

	*indexCorrelation = 0.0;	/* HNSW is not correlated with physical order */
}

static bytea *
hnswoptions(Datum reloptions, bool validate)
{
	static const relopt_parse_elt tab[] = {
		{"m", RELOPT_TYPE_INT, offsetof(HnswOptions, m)},
		{"ef_construction", RELOPT_TYPE_INT, offsetof(HnswOptions, ef_construction)},
		{"ef_search", RELOPT_TYPE_INT, offsetof(HnswOptions, ef_search)}
	};
	HnswOptions *opts;
	bytea	   *result;

	/* Lazy initialization: ensure relopt_kind_hnsw is registered */
	if (relopt_kind_hnsw == 0)
	{
		relopt_kind_hnsw = add_reloption_kind();
	}

	result = (bytea *) build_reloptions(reloptions, validate, relopt_kind_hnsw,
									   sizeof(HnswOptions),
									   tab, lengthof(tab));

	/* Validate parameter ranges if validate is true */
	if (validate && result != NULL)
	{
		opts = (HnswOptions *) VARDATA(result);

		/* Validate m */
		if (opts->m < HNSW_MIN_M || opts->m > HNSW_MAX_M)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("hnsw: parameter m must be between %d and %d, got %d",
							HNSW_MIN_M, HNSW_MAX_M, opts->m)));
		}

		/* Validate ef_construction */
		if (opts->ef_construction < HNSW_MIN_EF_CONSTRUCTION ||
			opts->ef_construction > HNSW_MAX_EF_CONSTRUCTION)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("hnsw: parameter ef_construction must be between %d and %d, got %d",
							HNSW_MIN_EF_CONSTRUCTION, HNSW_MAX_EF_CONSTRUCTION,
							opts->ef_construction)));
		}

		/* Validate ef_search */
		if (opts->ef_search < HNSW_MIN_EF_SEARCH ||
			opts->ef_search > HNSW_MAX_EF_SEARCH)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("hnsw: parameter ef_search must be between %d and %d, got %d",
							HNSW_MIN_EF_SEARCH, HNSW_MAX_EF_SEARCH, opts->ef_search)));
		}

		/* Ensure ef_construction >= m */
		if (opts->ef_construction < opts->m)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("hnsw: parameter ef_construction (%d) must be >= m (%d)",
							opts->ef_construction, opts->m)));
		}

		/* Ensure ef_search >= m */
		if (opts->ef_search < opts->m)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("hnsw: parameter ef_search (%d) must be >= m (%d)",
							opts->ef_search, opts->m)));
		}
	}

	return result;
}

static bool
hnswproperty(Oid index_oid,
			 int attno,
			 IndexAMProperty prop,
			 const char *propname,
			 bool *res,
			 bool *isnull)
{
	return false;
}

static IndexScanDesc
hnswbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;

	NDB_DECLARE(HnswScanOpaque, so);

	scan = RelationGetIndexScan(index, nkeys, norderbys);
	NDB_ALLOC(so, HnswScanOpaqueData, 1);
	so->efSearch = HNSW_DEFAULT_EF_SEARCH;
	so->strategy = 1;
	so->firstCall = true;
	so->k = 0;
	so->queryVector = NULL;
	so->results = NULL;
	so->distances = NULL;
	so->resultCount = 0;
	so->currentResult = 0;

	scan->opaque = so;

	return scan;
}

static void
hnswrescan(IndexScanDesc scan,
		   ScanKey keys,
		   int nkeys,
		   ScanKey orderbys,
		   int norderbys)
{
	extern int	neurondb_hnsw_ef_search;
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;

	so->firstCall = true;
	so->currentResult = 0;
	so->resultCount = 0;

	if (norderbys > 0)
		so->strategy = orderbys[0].sk_strategy;
	else
		so->strategy = 1;

	if (neurondb_hnsw_ef_search > 0)
		so->efSearch = neurondb_hnsw_ef_search;
	else
		{
			Buffer		metaBuffer = ReadBuffer(scan->indexRelation, 0);
			Page		metaPage;
			HnswMetaPage meta;

			LockBuffer(metaBuffer, BUFFER_LOCK_SHARE);
			metaPage = BufferGetPage(metaBuffer); 
			meta = (HnswMetaPage) PageGetContents(metaPage);
			so->efSearch = meta->efSearch;
			UnlockReleaseBuffer(metaBuffer);
		}

		if (so->efSearch > 100000)
		{
			elog(WARNING, "hnsw: ef_search %d exceeds maximum, clamping to 100000", so->efSearch);
			so->efSearch = 100000;
		}

	if (norderbys > 0 && orderbys[0].sk_argument != 0)
	{
		float4	   *vectorData;
		int			dim;
		Oid			queryType;
		MemoryContext oldctx;

		queryType = TupleDescAttr(scan->indexRelation->rd_att, 0)->atttypid;
		oldctx = MemoryContextSwitchTo(scan->indexRelation->rd_indexcxt);
		vectorData = hnswExtractVectorData(orderbys[0].sk_argument, queryType, &dim,
										   scan->indexRelation->rd_indexcxt);
		MemoryContextSwitchTo(oldctx);

		if (vectorData != NULL)
		{
			char	   *queryVector_raw = NULL;
			if (so->queryVector)
			{
				NDB_FREE(so->queryVector);
				so->queryVector = NULL;
			}
			NDB_ALLOC(queryVector_raw, char, VECTOR_SIZE(dim));
			so->queryVector = (Vector *) queryVector_raw;
			SET_VARSIZE(so->queryVector, VECTOR_SIZE(dim));
			so->queryVector->dim = dim;
			memcpy(so->queryVector->data, vectorData, dim * sizeof(float4));
			NDB_FREE(vectorData);
			vectorData = NULL;
		}
		/* Get k from GUC or default to 10 */
		so->k = (neurondb_hnsw_k > 0) ? neurondb_hnsw_k : 10;
	}
}

static bool
hnswgettuple(IndexScanDesc scan, ScanDirection dir)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Buffer		metaBuffer;
	Page		metaPage;
	HnswMetaPage meta;

	if (so->firstCall)
	{
		metaBuffer = ReadBuffer(scan->indexRelation, 0);
		LockBuffer(metaBuffer, BUFFER_LOCK_SHARE);
		metaPage = BufferGetPage(metaBuffer);
		meta = (HnswMetaPage) PageGetContents(metaPage);

		if (!so->queryVector)
		{
			UnlockReleaseBuffer(metaBuffer);
			return false;
		}

		hnswSearch(scan->indexRelation, meta,
				   so->queryVector->data, so->queryVector->dim,
				   so->strategy, so->efSearch, so->k,
				   &so->results, &so->distances, &so->resultCount);

		UnlockReleaseBuffer(metaBuffer);
		so->firstCall = false;
		so->currentResult = 0;
	}

	if (so->currentResult < so->resultCount)
	{
		/* Set scan->xs_heaptid for identified tuple */
		BlockNumber resultBlkno = so->results[so->currentResult];
		Buffer		buf;
		Page		page;
		HnswNode	node;

		if (!hnswValidateBlockNumber(resultBlkno, scan->indexRelation))
		{
			elog(WARNING, "hnsw: invalid result block %u in gettuple, skipping", resultBlkno);
			so->currentResult++;
			return false;
		}

		/* Read the node to get its heap pointer */
		buf = ReadBuffer(scan->indexRelation, resultBlkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);

		/* If page is empty, skip this result and try next */
		if (PageIsEmpty(page))
		{
			UnlockReleaseBuffer(buf);
			so->currentResult++;
			return false;
		}

		node = (HnswNode) PageGetItem(page, PageGetItemId(page, FirstOffsetNumber));
		if (node != NULL)
		{
			scan->xs_heaptid = node->heapPtr;
		}
		else
		{
			elog(WARNING, "hnsw: null node at block %u in gettuple", resultBlkno);
			UnlockReleaseBuffer(buf);
			so->currentResult++;
			return false;
		}

		UnlockReleaseBuffer(buf);
		so->currentResult++;
		return true;
	}

	return false;
}

static void
hnswendscan(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;

	if (so == NULL)
		return;

	if (so->results)
	{
		NDB_FREE(so->results);
		so->results = NULL;
	}
	if (so->distances)
	{
		NDB_FREE(so->distances);
		so->distances = NULL;
	}
	if (so->queryVector)
	{
		NDB_FREE(so->queryVector);
		so->queryVector = NULL;
	}

	NDB_FREE(so);
	so = NULL;
}


/*
 * hnswInitMetaPage - Initialize HNSW metadata page.
 */
static void
hnswInitMetaPage(Buffer metaBuffer, int16 m, int16 efConstruction, int16 efSearch, float4 ml)
{
	Page		page;
	HnswMetaPage meta;

	page = BufferGetPage(metaBuffer);
	PageInit(page, BufferGetPageSize(metaBuffer), sizeof(HnswMetaPageData));

	meta = (HnswMetaPage) PageGetContents(page);
	meta->magicNumber = HNSW_MAGIC_NUMBER;
	meta->version = HNSW_VERSION;
	meta->entryPoint = InvalidBlockNumber;
	meta->entryLevel = -1;
	meta->maxLevel = -1;
	meta->m = m;
	meta->efConstruction = efConstruction;
	meta->efSearch = efSearch;
	meta->ml = ml;
	meta->insertedVectors = 0;
}

/*
 * Load HNSW index options from relation, with defaults if not set
 */
static void
hnswLoadOptions(Relation index, HnswOptions *opts_out)
{
	static const relopt_parse_elt tab[] = {
		{"m", RELOPT_TYPE_INT, offsetof(HnswOptions, m)},
		{"ef_construction", RELOPT_TYPE_INT, offsetof(HnswOptions, ef_construction)},
		{"ef_search", RELOPT_TYPE_INT, offsetof(HnswOptions, ef_search)}
	};
	HnswOptions *opts;
	Datum		relopts = PointerGetDatum(index->rd_options);

	/* Lazy initialization: ensure relopt_kind_hnsw is registered */
	if (relopt_kind_hnsw == 0)
	{
		relopt_kind_hnsw = add_reloption_kind();
	}

	opts = (HnswOptions *) build_reloptions(relopts, false,
										   relopt_kind_hnsw,
										   sizeof(HnswOptions),
										   tab, lengthof(tab));

	/* Copy to output with defaults */
	opts_out->m = opts ? opts->m : HNSW_DEFAULT_M;
	opts_out->ef_construction = opts ? opts->ef_construction : HNSW_DEFAULT_EF_CONSTRUCTION;
	opts_out->ef_search = opts ? opts->ef_search : HNSW_DEFAULT_EF_SEARCH;
}

static int
hnswGetRandomLevel(float4 ml)
{
	double		r;
	int			level;

	r = (double) random() / (double) RAND_MAX;
	while (r == 0.0)
		r = (double) random() / (double) RAND_MAX;

	level = (int) (-log(r) * ml);

	if (level > HNSW_MAX_LEVEL - 1)
		level = HNSW_MAX_LEVEL - 1;
	if (level < 0)
		level = 0;

	return level;
}

/*
 * Validate and clamp neighborCount to prevent array bounds violations.
 * Returns a safe neighborCount value clamped to [0, m*2].
 */
static int16
hnswValidateNeighborCount(int16 neighborCount, int m, int level)
{
	int16		maxNeighbors = m * 2;

	if (neighborCount < 0)
	{
		elog(WARNING, "hnsw: invalid negative neighborCount %d at level %d, clamping to 0",
			 neighborCount, level);
		return 0;
	}
	if (neighborCount > maxNeighbors)
	{
		elog(WARNING, "hnsw: neighborCount %d exceeds maximum %d at level %d, clamping",
			 neighborCount, maxNeighbors, level);
		return maxNeighbors;
	}
	return neighborCount;
}

/*
 * hnswValidateLevelSafe - Validate level value against HNSW_MAX_LEVEL.
 *
 * Returns true if valid, false otherwise. Does not raise ERROR to avoid
 * issues when called with held locks. Callers should check return value
 * and handle errors appropriately, releasing locks before raising errors.
 */
static bool
hnswValidateLevelSafe(int level)
{
	if (level < 0 || level >= HNSW_MAX_LEVEL)
	{
		return false;
	}
	return true;
}

/*
 * Validate level and raise ERROR if invalid.
 * Use this only when not holding locks.
 */
static void __attribute__((unused))
hnswValidateLevel(int level)
{
	if (!hnswValidateLevelSafe(level))
	{
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("hnsw: invalid node level %d (valid range: 0-%d)",
						level, HNSW_MAX_LEVEL - 1)));
	}
}

/*
 * Validate block number is within valid range for the index.
 * Returns true if valid, false otherwise.
 */
static bool
hnswValidateBlockNumber(BlockNumber blkno, Relation index)
{
	BlockNumber maxBlocks = RelationGetNumberOfBlocks(index);

	if (blkno == InvalidBlockNumber)
		return false;

	if (blkno >= maxBlocks)
	{
		elog(WARNING, "hnsw: block number %u exceeds index size %u",
			 blkno, maxBlocks);
		return false;
	}
	return true;
}

/*
 * hnswComputeNodeSizeSafe - Compute node size with overflow checking.
 *
 * Returns the computed size and sets *overflow to true if overflow detected.
 * Uses the m parameter from meta page, not HNSW_DEFAULT_M.
 */
static Size
hnswComputeNodeSizeSafe(int dim, int level, int m, bool *overflow)
{
	size_t		vectorSize;
	size_t		neighborSize;
	size_t		totalSize;
	Size		result;

	*overflow = false;

	/* Validate m parameter */
	if (m < HNSW_MIN_M || m > HNSW_MAX_M)
	{
		*overflow = true;
		return 0;
	}

	/* Check vector size overflow */
	vectorSize = (size_t) dim * sizeof(float4);
	if (vectorSize / sizeof(float4) != (size_t) dim)
	{
		*overflow = true;
		return 0;
	}

	/* Check neighbor size overflow - uses m parameter */
	neighborSize = (size_t)(level + 1) * m * 2 * sizeof(BlockNumber);
	if (neighborSize / sizeof(BlockNumber) != (size_t)(level + 1) * m * 2)
	{
		*overflow = true;
		return 0;
	}

	/* Check total size overflow */
	totalSize = sizeof(HnswNodeData) + vectorSize + neighborSize;
	if (totalSize < sizeof(HnswNodeData) || totalSize < vectorSize || totalSize < neighborSize)
	{
		*overflow = true;
		return 0;
	}

	result = MAXALIGN(totalSize);
	if (result < totalSize)  /* MAXALIGN overflow */
	{
		*overflow = true;
		return 0;
	}

	return result;
}

/*
 * Distance computation for L2, Cosine, or negative-InnerProduct distances
 */
static float4
hnswComputeDistance(const float4 * vec1, const float4 * vec2, int dim, int strategy)
{
	int			i;
	double		sum = 0.0,
				dot_product = 0.0,
				norm1 = 0.0,
				norm2 = 0.0;

	switch (strategy)
	{
		case 1:					/* L2 */
			for (i = 0; i < dim; i++)
			{
				double		d = vec1[i] - vec2[i];

				sum += d * d;
			}
			return (float4) sqrt(sum);

		case 2:					/* Cosine */
			for (i = 0; i < dim; i++)
			{
				dot_product += vec1[i] * vec2[i];
				norm1 += vec1[i] * vec1[i];
				norm2 += vec2[i] * vec2[i];
			}
			norm1 = sqrt(norm1);
			norm2 = sqrt(norm2);
			if (norm1 == 0.0 || norm2 == 0.0)
				return 2.0f;
			return (float4) (1.0f - (dot_product / (norm1 * norm2)));

		case 3:					/* Negative inner product */
			for (i = 0; i < dim; i++)
				dot_product += vec1[i] * vec2[i];
			return (float4) (-dot_product);

		default:
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("hnsw: unsupported distance strategy %d", strategy)));
			return 0.0f;
	}
}

/*
 * Cache type OIDs once to avoid expensive lookups on every call
 */
static void
hnswCacheTypeOids(void)
{
	if (typeOidsCached)
		return;

	{
		List	   *names;

		names = list_make2(makeString("public"), makeString("vector"));
		cached_vectorOid = LookupTypeNameOid(NULL, makeTypeNameFromNameList(names), true);
		if (!OidIsValid(cached_vectorOid))
		{
			ereport(ERROR,
					(errcode(ERRCODE_UNDEFINED_OBJECT),
					 errmsg("hnsw requires public.vector type from pgvector extension"),
					 errhint("Install the pgvector extension: CREATE EXTENSION vector")));
		}
		list_free(names);
		names = list_make2(makeString("public"), makeString("halfvec"));
		cached_halfvecOid = LookupTypeNameOid(NULL, makeTypeNameFromNameList(names), true);
		if (!OidIsValid(cached_halfvecOid))
		{
			ereport(ERROR,
					(errcode(ERRCODE_UNDEFINED_OBJECT),
					 errmsg("hnsw requires public.halfvec type from pgvector extension"),
					 errhint("Install the pgvector extension: CREATE EXTENSION vector")));
		}
		list_free(names);
		names = list_make2(makeString("public"), makeString("sparsevec"));
		cached_sparsevecOid = LookupTypeNameOid(NULL, makeTypeNameFromNameList(names), true);
		if (!OidIsValid(cached_sparsevecOid))
		{
			ereport(ERROR,
					(errcode(ERRCODE_UNDEFINED_OBJECT),
					 errmsg("hnsw requires public.sparsevec type from pgvector extension"),
					 errhint("Install the pgvector extension: CREATE EXTENSION vector")));
		}
		list_free(names);
		cached_bitOid = BITOID;
	}

	typeOidsCached = true;
}

/*
 * hnswExtractVectorData - Extract vector from datum for type OID.
 *
 * Supports vector, halfvec, sparsevec, and bit types. For sparsevec, the
 * result buffer is zero-initialized before populating non-zero entries to
 * ensure correct distance computations.
 */
static float4 *
hnswExtractVectorData(Datum value, Oid typeOid, int *out_dim, MemoryContext ctx)
{
	MemoryContext oldctx;
	Oid			vectorOid,
				halfvecOid,
				sparsevecOid,
				bitOid;
	int			i;

	NDB_DECLARE(float4 *, result);

	/* Cache OIDs on first call */
	hnswCacheTypeOids();

	vectorOid = cached_vectorOid;
	halfvecOid = cached_halfvecOid;
	sparsevecOid = cached_sparsevecOid;
	bitOid = cached_bitOid;

	oldctx = MemoryContextSwitchTo(ctx);

	if (typeOid == vectorOid)
	{
		Vector	   *v = DatumGetVector(value);

		NDB_CHECK_VECTOR_VALID(v);
		*out_dim = v->dim;
		NDB_CHECK_ALLOC_SIZE((size_t) v->dim * sizeof(float4), "vector data");
		NDB_ALLOC(result, float4, v->dim);
		NDB_CHECK_ALLOC(result, "vector data");
		for (i = 0; i < v->dim; i++)
			result[i] = v->data[i];
	}
	else if (typeOid == halfvecOid)
	{
		VectorF16  *hv = (VectorF16 *) PG_DETOAST_DATUM(value);

		NDB_CHECK_NULL(hv, "halfvec");
		if (hv->dim <= 0 || hv->dim > 32767)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("hnsw: invalid halfvec dimension %d", hv->dim)));
		*out_dim = hv->dim;
		NDB_CHECK_ALLOC_SIZE((size_t) hv->dim * sizeof(float4), "halfvec data");
		NDB_ALLOC(result, float4, hv->dim);
		NDB_CHECK_ALLOC(result, "halfvec data");
		for (i = 0; i < hv->dim; i++)
			result[i] = fp16_to_float(hv->data[i]);
	}
	else if (typeOid == sparsevecOid)
	{
		VectorMap  *sv = (VectorMap *) PG_DETOAST_DATUM(value);

		NDB_CHECK_NULL(sv, "sparsevec");
		if (sv->total_dim <= 0 || sv->total_dim > 32767)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("hnsw: invalid sparsevec total_dim %d", sv->total_dim)));
		{
			int32	   *indices = VECMAP_INDICES(sv);
			float4	   *values = VECMAP_VALUES(sv);

			*out_dim = sv->total_dim;
			NDB_CHECK_ALLOC_SIZE((size_t) sv->total_dim * sizeof(float4), "sparsevec data");
			NDB_ALLOC(result, float4, sv->total_dim);
			NDB_CHECK_ALLOC(result, "sparsevec data");

			/* Zero-initialize buffer: sparsevec only stores non-zero entries */
			memset(result, 0, sv->total_dim * sizeof(float4));

			/* Populate non-zero entries */
			for (i = 0; i < sv->nnz; i++)
			{
				if (indices[i] >= 0 && indices[i] < sv->total_dim)
					result[indices[i]] = values[i];
			}
		}
	}
	else if (typeOid == bitOid)
	{
		VarBit	   *bit_vec = (VarBit *) PG_DETOAST_DATUM(value);

		NDB_CHECK_NULL(bit_vec, "bit vector");
		{
			int			nbits;
			bits8	   *bit_data;

			nbits = VARBITLEN(bit_vec);
			if (nbits <= 0 || nbits > 32767)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("hnsw: invalid bit vector length %d", nbits)));
			bit_data = VARBITS(bit_vec);
			*out_dim = nbits;
			NDB_CHECK_ALLOC_SIZE((size_t) nbits * sizeof(float4), "bit vector data");
			NDB_ALLOC(result, float4, nbits);
			NDB_CHECK_ALLOC(result, "bit vector data");
			for (i = 0; i < nbits; i++)
			{
				int			byte_idx = i / BITS_PER_BYTE;
				int			bit_idx = i % BITS_PER_BYTE;
				int			bit_val = (bit_data[byte_idx] >> (BITS_PER_BYTE - 1 - bit_idx)) & 1;

				result[i] = bit_val ? 1.0f : -1.0f;
			}
		}
	}
	else
	{
		MemoryContextSwitchTo(oldctx);
		ereport(ERROR,
				(errcode(ERRCODE_DATATYPE_MISMATCH),
				 errmsg("hnsw: unsupported type OID %u", typeOid)));
	}
	MemoryContextSwitchTo(oldctx);
	return result;
}

static Oid
hnswGetKeyType(Relation index, int attno)
{
	TupleDesc	indexDesc = RelationGetDescr(index);
	Form_pg_attribute attr;

	if (attno < 1 || attno > indexDesc->natts)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("hnsw: invalid attribute number %d", attno)));

	attr = TupleDescAttr(indexDesc, attno - 1);
	return attr->atttypid;
}

/*
 * hnswSearch
 *	Find k nearest neighbors via greedy layer traversal and ef-search.
 *
 * Entry point: start from metaPage->entryPoint; descend with greedy search.
 * Then use ef-search (candidate heap, visited set) at level 0.
 *
 * Results: (*results, *distances, *resultCount) filled on success.
 */
static void
hnswSearch(Relation index,
		   HnswMetaPage metaPage,
		   const float4 * query,
		   int dim,
		   int strategy,
		   int efSearch,
		   int k,
		   BlockNumber * *results,
		   float4 * *distances,
		   int *resultCount)
{
	BlockNumber current;
	int			currentLevel;
	volatile	Buffer nodeBuf = InvalidBuffer;
	Page		nodePage;
	HnswNode	node;
	float4	   *nodeVector;
	float4		currentDist;
	int			level;
	int			i,
				j;

	NDB_DECLARE(BlockNumber *, candidates);
	NDB_DECLARE(float4 *, candidateDists);
	int			candidateCount = 0;

	NDB_DECLARE(BlockNumber *, visited);
	int			visitedCount = 0;
	int			visitedCapacity = 0;

	NDB_DECLARE(bool *, visitedSet);
	BlockNumber *neighbors;
	int16		neighborCount;

	NDB_DECLARE(BlockNumber *, topK);
	NDB_DECLARE(float4 *, topKDists);
	int			topKCount = 0;

	NDB_DECLARE(int *, indices);
	int			minIdx,
				temp;
	int			l,
				worstIdx;
	float4		worstDist,
				minDist;

	/* Defensive: no vectors yet */
	if (metaPage->entryPoint == InvalidBlockNumber)
	{
		*results = NULL;
		*distances = NULL;
		*resultCount = 0;
		return;
	}

	PG_TRY();
	{
		BlockNumber numBlocks;	/* Computed once and reused throughout function */

		current = metaPage->entryPoint;
		currentLevel = metaPage->entryLevel;

		/* Validate entry level */
		if (currentLevel < 0 || currentLevel >= HNSW_MAX_LEVEL)
		{
			elog(WARNING, "hnsw: invalid entryLevel %d, resetting to 0", currentLevel);
			currentLevel = 0;
		}

		visitedCapacity = (efSearch > 1 ? efSearch * 2 : 32);
		NDB_ALLOC(visited, BlockNumber, visitedCapacity);

		/* Allocate visitedSet with overflow checking */
		numBlocks = RelationGetNumberOfBlocks(index);
		{
			size_t visitedSetSize = (size_t) numBlocks * sizeof(bool);
			if (visitedSetSize / sizeof(bool) != (size_t) numBlocks)
			{
				ereport(ERROR,
						(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
						 errmsg("hnsw: visitedSet size calculation overflow (%u blocks)",
								numBlocks)));
			}
			NDB_ALLOC(visitedSet, bool, numBlocks);
			memset(visitedSet, 0, visitedSetSize);
		}
		visitedCount = 0;

		NDB_ALLOC(candidates, BlockNumber, efSearch);
		NDB_ALLOC(candidateDists, float4, efSearch);
		candidateCount = 0;

		for (level = currentLevel; level > 0; level--)
		{
			bool		foundBetter;

			do
			{
				foundBetter = false;
				if (!hnswValidateBlockNumber(current, index))
				{
					elog(WARNING, "hnsw: invalid current block %u in greedy search", current);
					break;
				}

				nodeBuf = ReadBuffer(index, current);
				LockBuffer(nodeBuf, BUFFER_LOCK_SHARE);
				nodePage = BufferGetPage(nodeBuf);

				if (PageIsNew(nodePage) || PageIsEmpty(nodePage))
				{
					UnlockReleaseBuffer(nodeBuf);
					break;
				}

				node = (HnswNode) PageGetItem(nodePage,
											  PageGetItemId(nodePage, FirstOffsetNumber));
				if (node == NULL)
				{
					UnlockReleaseBuffer(nodeBuf);
					break;
				}

				/* Validate node level */
				if (!hnswValidateLevelSafe(node->level))
				{
					UnlockReleaseBuffer(nodeBuf);
					break;
				}

				nodeVector = HnswGetVector(node);
				if (nodeVector == NULL)
				{
					UnlockReleaseBuffer(nodeBuf);
					break;
				}

				currentDist = hnswComputeDistance(query, nodeVector, dim, strategy);

				if (node->level >= level)
				{
					neighbors = HnswGetNeighborsSafe(node, level, metaPage->m);
					neighborCount = node->neighborCount[level];

					/* Validate and clamp neighborCount */
					neighborCount = hnswValidateNeighborCount(neighborCount, metaPage->m, level);

					for (i = 0; i < neighborCount; i++)
					{
						Buffer		neighborBuf;
						Page		neighborPage;
						HnswNode	neighbor;
						float4	   *neighborVector;
						float4		neighborDist;

						if (neighbors[i] == InvalidBlockNumber)
							continue;

						if (!hnswValidateBlockNumber(neighbors[i], index))
						{
							elog(WARNING, "hnsw: invalid neighbor block %u at level %d",
								 neighbors[i], level);
							continue;
						}

						neighborBuf = ReadBuffer(index, neighbors[i]);
						LockBuffer(neighborBuf, BUFFER_LOCK_SHARE);
						neighborPage = BufferGetPage(neighborBuf);

						if (PageIsNew(neighborPage) || PageIsEmpty(neighborPage))
						{
							UnlockReleaseBuffer(neighborBuf);
							continue;
						}

						neighbor = (HnswNode) PageGetItem(neighborPage,
														  PageGetItemId(neighborPage, FirstOffsetNumber));
						if (neighbor == NULL)
						{
							UnlockReleaseBuffer(neighborBuf);
							continue;
						}

						neighborVector = HnswGetVector(neighbor);
						if (neighborVector == NULL)
						{
							UnlockReleaseBuffer(neighborBuf);
							continue;
						}

						neighborDist = hnswComputeDistance(query, neighborVector, dim, strategy);

						if (neighborDist < currentDist)
						{
							current = neighbors[i];
							currentDist = neighborDist;
							foundBetter = true;
						}

						UnlockReleaseBuffer(neighborBuf);
					}
				}
				UnlockReleaseBuffer(nodeBuf);
			} while (foundBetter);
		}

		if (!hnswValidateBlockNumber(current, index))
		{
			elog(WARNING, "hnsw: invalid current block %u for level 0 search", current);
			*results = NULL;
			*distances = NULL;
			*resultCount = 0;
			NDB_FREE(visited);
			NDB_FREE(visitedSet);
			NDB_FREE(candidates);
			NDB_FREE(candidateDists);
			return;
		}

		candidates[0] = current;
		nodeBuf = ReadBuffer(index, current);
		LockBuffer(nodeBuf, BUFFER_LOCK_SHARE);
		nodePage = BufferGetPage(nodeBuf);

		if (PageIsNew(nodePage) || PageIsEmpty(nodePage))
		{
			UnlockReleaseBuffer(nodeBuf);
			*results = NULL;
			*distances = NULL;
			*resultCount = 0;
			NDB_FREE(visited);
			NDB_FREE(visitedSet);
			NDB_FREE(candidates);
			NDB_FREE(candidateDists);
			return;
		}

		node = (HnswNode) PageGetItem(nodePage,
									  PageGetItemId(nodePage, FirstOffsetNumber));
		if (node == NULL)
		{
			UnlockReleaseBuffer(nodeBuf);
			*results = NULL;
			*distances = NULL;
			*resultCount = 0;
			NDB_FREE(visited);
			NDB_FREE(visitedSet);
			NDB_FREE(candidates);
			NDB_FREE(candidateDists);
			return;
		}

		if (!hnswValidateLevelSafe(node->level))
		{
			UnlockReleaseBuffer(nodeBuf);
			*results = NULL;
			*distances = NULL;
			*resultCount = 0;
			NDB_FREE(visited);
			NDB_FREE(visitedSet);
			NDB_FREE(candidates);
			NDB_FREE(candidateDists);
			return;
		}

		nodeVector = HnswGetVector(node);
		if (nodeVector == NULL)
		{
			UnlockReleaseBuffer(nodeBuf);
			*results = NULL;
			*distances = NULL;
			*resultCount = 0;
			NDB_FREE(visited);
			NDB_FREE(visitedSet);
			NDB_FREE(candidates);
			NDB_FREE(candidateDists);
			return;
		}

		candidateDists[0] = hnswComputeDistance(query, nodeVector, dim, strategy);
		candidateCount = 1;
		visited[visitedCount++] = current;
			/* Use numBlocks from outer scope - computed once at function start */
			if (current < numBlocks)
				visitedSet[current] = true;
		UnlockReleaseBuffer(nodeBuf);

		for (i = 0; i < candidateCount && candidateCount < efSearch; i++)
		{
			BlockNumber candidate = candidates[i];

			if (!hnswValidateBlockNumber(candidate, index))
			{
				elog(WARNING, "hnsw: invalid candidate block %u, skipping", candidate);
				continue;
			}

			nodeBuf = ReadBuffer(index, candidate);
			LockBuffer(nodeBuf, BUFFER_LOCK_SHARE);
			nodePage = BufferGetPage(nodeBuf);

			if (PageIsNew(nodePage) || PageIsEmpty(nodePage))
			{
				UnlockReleaseBuffer(nodeBuf);
				continue;
			}

			node = (HnswNode) PageGetItem(nodePage,
										  PageGetItemId(nodePage, FirstOffsetNumber));
			if (node == NULL)
			{
				UnlockReleaseBuffer(nodeBuf);
				continue;
			}

			if (!hnswValidateLevelSafe(node->level))
			{
				UnlockReleaseBuffer(nodeBuf);
				continue;
			}

			neighbors = HnswGetNeighborsSafe(node, 0, metaPage->m);
			neighborCount = node->neighborCount[0];

			/* Validate and clamp neighborCount */
			neighborCount = hnswValidateNeighborCount(neighborCount, metaPage->m, 0);

			for (j = 0; j < neighborCount; j++)
			{
				Buffer		neighborBuf;
				Page		neighborPage;
				HnswNode	neighbor;
				float4	   *neighborVector;
				float4		neighborDist;

				if (neighbors[j] == InvalidBlockNumber)
					continue;

				if (!hnswValidateBlockNumber(neighbors[j], index))
				{
					elog(WARNING, "hnsw: invalid neighbor block %u, skipping", neighbors[j]);
					continue;
				}

				/* Check visitedSet with bounds validation */
				if (neighbors[j] < numBlocks && visitedSet[neighbors[j]])
					continue;

				neighborBuf = ReadBuffer(index, neighbors[j]);
				LockBuffer(neighborBuf, BUFFER_LOCK_SHARE);
				neighborPage = BufferGetPage(neighborBuf);

				if (PageIsNew(neighborPage) || PageIsEmpty(neighborPage))
				{
					UnlockReleaseBuffer(neighborBuf);
					continue;
				}

				neighbor = (HnswNode) PageGetItem(neighborPage,
												  PageGetItemId(neighborPage, FirstOffsetNumber));
				if (neighbor == NULL)
				{
					UnlockReleaseBuffer(neighborBuf);
					continue;
				}

				neighborVector = HnswGetVector(neighbor);
				if (neighborVector == NULL)
				{
					UnlockReleaseBuffer(neighborBuf);
					continue;
				}

				neighborDist = hnswComputeDistance(query, neighborVector, dim, strategy);
				UnlockReleaseBuffer(neighborBuf);

				if (neighbors[j] < numBlocks)
					visitedSet[neighbors[j]] = true;

				visited[visitedCount++] = neighbors[j];
				if (visitedCount >= visitedCapacity)
				{
					/* Limit reallocation to prevent excessive memory usage */
					if (visitedCapacity >= HNSW_MAX_VISITED_CAPACITY)
					{
						elog(WARNING, "hnsw: visited array reached maximum capacity %d, stopping expansion",
							 HNSW_MAX_VISITED_CAPACITY);
						/* Continue without expanding - may miss some neighbors but prevents OOM */
					}
					else
					{
						int newCapacity = Min(visitedCapacity * 2, HNSW_MAX_VISITED_CAPACITY);
						if (newCapacity > visitedCapacity)
						{
							visitedCapacity = newCapacity;
							visited = (BlockNumber *) repalloc(visited,
															   visitedCapacity * sizeof(BlockNumber));
						}
					}
				}

				/* Ensure candidateCount doesn't exceed efSearch */
				if (candidateCount < efSearch)
				{
					candidates[candidateCount] = neighbors[j];
					candidateDists[candidateCount] = neighborDist;
					candidateCount++;
				}
				else
				{
					worstIdx = 0;
					worstDist = candidateDists[0];
					for (l = 1; l < candidateCount && l < efSearch; l++)
					{
						if (candidateDists[l] > worstDist)
						{
							worstDist = candidateDists[l];
							worstIdx = l;
						}
					}

					if (neighborDist < worstDist)
					{
						candidates[worstIdx] = neighbors[j];
						candidateDists[worstIdx] = neighborDist;
					}
				}
			}
			UnlockReleaseBuffer(nodeBuf);
		}

		NDB_ALLOC(indices, int, candidateCount);
		for (i = 0; i < candidateCount; i++)
		{
			CHECK_FOR_INTERRUPTS();
			indices[i] = i;
		}

		for (i = 0; i < k && i < candidateCount; i++)
		{
			CHECK_FOR_INTERRUPTS();
			minIdx = i;
			minDist = candidateDists[indices[i]];

			for (j = i + 1; j < candidateCount; j++)
			{
				if (candidateDists[indices[j]] < minDist)
				{
					minDist = candidateDists[indices[j]];
					minIdx = j;
				}
			}
			if (minIdx != i)
			{
				temp = indices[i];
				indices[i] = indices[minIdx];
				indices[minIdx] = temp;
			}
		}

		topKCount = Min(k, candidateCount);
		NDB_ALLOC(topK, BlockNumber, topKCount);
		NDB_ALLOC(topKDists, float4, topKCount);
		for (i = 0; i < topKCount; i++)
		{
			topK[i] = candidates[indices[i]];
			topKDists[i] = candidateDists[indices[i]];
		}

		NDB_FREE(indices);
		indices = NULL;

		*results = topK;
		*distances = topKDists;
		*resultCount = topKCount;

		NDB_FREE(candidates);
		candidates = NULL;
		NDB_FREE(candidateDists);
		candidateDists = NULL;
		NDB_FREE(visited);
		visited = NULL;
		NDB_FREE(visitedSet);
		visitedSet = NULL;
	}
	PG_CATCH();
	{
		if (BufferIsValid(nodeBuf))
		{
			LockBuffer(nodeBuf, BUFFER_LOCK_UNLOCK);
			ReleaseBuffer(nodeBuf);
			nodeBuf = InvalidBuffer;
		}
		if (candidates)
		{
			NDB_FREE(candidates);
			candidates = NULL;
		}
		if (candidateDists)
		{
			NDB_FREE(candidateDists);
			candidateDists = NULL;
		}
		if (visited)
		{
			NDB_FREE(visited);
			visited = NULL;
		}
		if (visitedSet)
		{
			NDB_FREE(visitedSet);
			visitedSet = NULL;
		}
		if (topK)
		{
			NDB_FREE(topK);
			topK = NULL;
		}
		if (topKDists)
		{
			NDB_FREE(topKDists);
			topKDists = NULL;
		}
		if (indices)
		{
			NDB_FREE(indices);
			indices = NULL;
		}
		*results = NULL;
		*distances = NULL;
		*resultCount = 0;
		PG_RE_THROW();
	}
	PG_END_TRY();
}

/*
 * hnswInsertNode - Insert a vector into the HNSW graph structure.
 *
 * Assigns the new node to a random level using exponential distribution,
 * searches for nearest neighbors at each level starting from entry point,
 * and establishes bidirectional links maintaining at most M connections
 * per level. If the new node is at a level higher than the current maximum,
 * it becomes the new entry point.
 */
static void
hnswInsertNode(Relation index,
			   HnswMetaPage metaPage,
			   const float4 * vector,
			   int dim,
			   ItemPointer heapPtr)
{
	int			level;
	Buffer		buf = InvalidBuffer;
	Page		page;
	BlockNumber blkno;
	Size		nodeSize;
	int			i;
	HnswNode	node = NULL;
	char	   *node_raw = NULL;

	level = hnswGetRandomLevel(metaPage->ml);

	/* Enforce limit on level */
	if (level >= HNSW_MAX_LEVEL)
		level = HNSW_MAX_LEVEL - 1;

	/* Validate level */
	if (!hnswValidateLevelSafe(level))
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("hnsw: failed to generate valid level")));
	}

	{
		bool		overflow = false;
		int			m = metaPage->m;  /* Use m from meta page */
		BlockNumber *neighbors;
		int			l;

		nodeSize = hnswComputeNodeSizeSafe(dim, level, m, &overflow);
		if (overflow || nodeSize == 0)
		{
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("hnsw: node size calculation overflow (dim=%d, level=%d, m=%d)",
							dim, level, m)));
		}
		NDB_ALLOC(node_raw, char, nodeSize);
		node = (HnswNode) node_raw;
		ItemPointerCopy(heapPtr, &node->heapPtr);
		node->level = level;
		node->dim = dim;
		for (i = 0; i < HNSW_MAX_LEVEL; i++)
			node->neighborCount[i] = 0;
		memcpy(HnswGetVector(node), vector, dim * sizeof(float4));

		/* Initialize neighbor arrays to InvalidBlockNumber for safety */
		for (l = 0; l <= level; l++)
		{
			node->neighborCount[l] = 0;
			neighbors = HnswGetNeighborsSafe(node, l, m);
			memset(neighbors, 0xFF, m * 2 * sizeof(BlockNumber));	/* InvalidBlockNumber */
		}
		for (; l < HNSW_MAX_LEVEL; l++)
			node->neighborCount[l] = 0;
	}

	/* Step 3: Find insertion point by greedy search from entry point */
	{
		BlockNumber bestEntry = metaPage->entryPoint;
		float4		bestDist = FLT_MAX;
		Buffer		entryBuf;
		Page		entryPage;
		HnswNode	entryNode;
		float4	   *entryVector;
		BlockNumber *entryNeighbors;
		int16		entryNeighborCount;
		bool		improved = true;
		int			iterations = 0;
		const int	maxIterations = 10;

		if (bestEntry != InvalidBlockNumber && level > 0)
		{
			while (improved && iterations < maxIterations)
			{
				improved = false;
				iterations++;

				if (!hnswValidateBlockNumber(bestEntry, index))
				{
					elog(WARNING, "hnsw: invalid bestEntry block %u in insert", bestEntry);
					break;
				}

				entryBuf = ReadBuffer(index, bestEntry);
				LockBuffer(entryBuf, BUFFER_LOCK_SHARE);
				entryPage = BufferGetPage(entryBuf);

				if (PageIsNew(entryPage) || PageIsEmpty(entryPage))
				{
					UnlockReleaseBuffer(entryBuf);
					break;
				}

				entryNode = (HnswNode) PageGetItem(entryPage,
												   PageGetItemId(entryPage, FirstOffsetNumber));
				if (entryNode == NULL)
				{
					UnlockReleaseBuffer(entryBuf);
					break;
				}

				/* Validate entry node level */
				if (!hnswValidateLevelSafe(entryNode->level))
				{
					UnlockReleaseBuffer(entryBuf);
					break;
				}

				if (entryNode->level >= level)
				{
					entryVector = HnswGetVector(entryNode);
					if (entryVector == NULL)
					{
						UnlockReleaseBuffer(entryBuf);
						break;
					}

					bestDist = hnswComputeDistance(vector, entryVector, dim, 1);
					entryNeighbors = HnswGetNeighborsSafe(entryNode, level, metaPage->m);
					entryNeighborCount = entryNode->neighborCount[level];

					/* Validate and clamp neighborCount */
					entryNeighborCount = hnswValidateNeighborCount(entryNeighborCount, metaPage->m, level);

					for (i = 0; i < entryNeighborCount; i++)
					{
						CHECK_FOR_INTERRUPTS();

						if (entryNeighbors[i] == InvalidBlockNumber)
							continue;

						if (entryNeighbors[i] >= RelationGetNumberOfBlocks(index))
						{
							elog(WARNING, "hnsw: invalid neighbor block %u in insert",
								 entryNeighbors[i]);
							continue;
						}

						{
							Buffer		neighborBuf;
							Page		neighborPage;
							HnswNode	neighbor;
							float4	   *neighborVector;
							float4		neighborDist;

							neighborBuf = ReadBuffer(index, entryNeighbors[i]);
							LockBuffer(neighborBuf, BUFFER_LOCK_SHARE);
							neighborPage = BufferGetPage(neighborBuf);

							if (PageIsNew(neighborPage) || PageIsEmpty(neighborPage))
							{
								UnlockReleaseBuffer(neighborBuf);
								continue;
							}

							neighbor = (HnswNode) PageGetItem(neighborPage,
															  PageGetItemId(neighborPage, FirstOffsetNumber));
							if (neighbor == NULL)
							{
								UnlockReleaseBuffer(neighborBuf);
								continue;
							}

							neighborVector = HnswGetVector(neighbor);
							if (neighborVector == NULL)
							{
								UnlockReleaseBuffer(neighborBuf);
								continue;
							}

							neighborDist = hnswComputeDistance(vector, neighborVector, dim, 1);

							if (neighborDist < bestDist)
							{
								bestDist = neighborDist;
								bestEntry = entryNeighbors[i];
								improved = true;
							}

							UnlockReleaseBuffer(neighborBuf);
						}
					}
				}

				UnlockReleaseBuffer(entryBuf);
			}
		}
	}

	/* Step 4: Insert the node into the index (1 node per page) */
	blkno = RelationGetNumberOfBlocks(index);

	PG_TRY();
	{
		buf = ReadBuffer(index, P_NEW);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
		page = BufferGetPage(buf);

		if (PageIsNew(page))
			PageInit(page, BufferGetPageSize(buf), 0);

		if (!PageIsEmpty(page))
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("hnsw: expected new page to be empty")));

		if (PageGetFreeSpace(page) < nodeSize)
			ereport(ERROR,
					(errcode(ERRCODE_INSUFFICIENT_RESOURCES),
					 errmsg("hnsw: not enough space for new node (needed %zu, available %zu)",
							nodeSize, PageGetFreeSpace(page))));

		if (PageAddItem(page, (Item) node, nodeSize, InvalidOffsetNumber, false, false) == InvalidOffsetNumber)
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("hnsw: failed to add node to page")));

		MarkBufferDirty(buf);
		UnlockReleaseBuffer(buf);
		buf = InvalidBuffer;
	}
	PG_CATCH();
	{
		if (BufferIsValid(buf))
		{
			LockBuffer(buf, BUFFER_LOCK_UNLOCK);
			ReleaseBuffer(buf);
			buf = InvalidBuffer;
		}
		NDB_FREE(node);
		node = NULL;
		PG_RE_THROW();
	}
	PG_END_TRY();

	/* Step 5: Link neighbors at each level bidirectionally */
	{
		int			entryLevel = metaPage->entryLevel;
		int			m = metaPage->m;
		int			efConstruction = metaPage->efConstruction;

		/* Skip neighbor linking if this is the first node in the index */
		if (metaPage->entryPoint != InvalidBlockNumber && entryLevel >= 0)
		{
			NDB_DECLARE(BlockNumber **, selectedNeighborsPerLevel);
			NDB_DECLARE(int *, selectedCountPerLevel);
			int			currentLevel;
			int			maxLevel = Min(level, entryLevel);
			Buffer		newNodeBuf = InvalidBuffer;
			Page		newNodePage;
			HnswNode	newNode;
			int			idx,
						j;

			if (blkno == InvalidBlockNumber ||
				blkno >= RelationGetNumberOfBlocks(index))
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("hnsw: invalid block number %u after insert", blkno)));
			}

			NDB_ALLOC(selectedNeighborsPerLevel, BlockNumber *, maxLevel + 1);
			NDB_ALLOC(selectedCountPerLevel, int, maxLevel + 1);

			/* Step 5a: Search for neighbors at each level */
			for (currentLevel = maxLevel; currentLevel >= 0; currentLevel--)
			{
				NDB_DECLARE(BlockNumber *, candidates);
				NDB_DECLARE(float4 *, candidateDistances);
				NDB_DECLARE(BlockNumber *, selectedNeighbors);
				NDB_DECLARE(float4 *, selectedDistances);
				int			candidateCount = 0;
				int			selectedCount;

				/* Find neighbor candidates for this level */
				hnswSearch(index,
						   metaPage,
						   vector,
						   dim,
						   1,	/* L2 distance */
						   efConstruction,
						   efConstruction,
						   &candidates,
						   &candidateDistances,
						   &candidateCount);

				selectedCount = Min(m, candidateCount);
				if (selectedCount > 0)
				{
					NDB_ALLOC(selectedNeighbors, BlockNumber, selectedCount);
					NDB_ALLOC(selectedDistances, float4, selectedCount);
				}
				else
				{
					selectedNeighbors = NULL;
					selectedDistances = NULL;
				}

				/* Sort by distance: select top m */
				for (idx = 0; idx < selectedCount && candidates != NULL && candidateDistances != NULL; idx++)
				{
					int			bestIdx = idx;
					float4		bestDist = candidateDistances[idx];

					for (j = idx + 1; j < candidateCount; j++)
					{
						if (candidateDistances[j] < bestDist)
						{
							bestDist = candidateDistances[j];
							bestIdx = j;
						}
					}
					if (bestIdx != idx)
					{
						BlockNumber tempBlk = candidates[idx];
						float4		tempDist = candidateDistances[idx];

						candidates[idx] = candidates[bestIdx];
						candidateDistances[idx] = candidateDistances[bestIdx];
						candidates[bestIdx] = tempBlk;
						candidateDistances[bestIdx] = tempDist;
					}
					selectedNeighbors[idx] = candidates[idx];
					selectedDistances[idx] = candidateDistances[idx];
				}

				/* Now lock newNodeBuf and write neighbors */
				newNodeBuf = ReadBuffer(index, blkno);
				LockBuffer(newNodeBuf, BUFFER_LOCK_EXCLUSIVE);
				newNodePage = BufferGetPage(newNodeBuf);
				if (PageIsNew(newNodePage) || PageIsEmpty(newNodePage))
				{
					UnlockReleaseBuffer(newNodeBuf);
					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg("hnsw: newly inserted page is empty at block %u", blkno)));
				}
				newNode = (HnswNode) PageGetItem(newNodePage,
												 PageGetItemId(newNodePage, FirstOffsetNumber));
				if (newNode == NULL)
				{
					UnlockReleaseBuffer(newNodeBuf);
					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg("hnsw: null node at newly inserted block %u", blkno)));
				}

				/*
				 * Link new node to neighbors, and each neighbor back
				 * (bidirectional)
				 */
				{
					BlockNumber *newNodeNeighbors = HnswGetNeighborsSafe(newNode, currentLevel, m);
					for (idx = 0; idx < selectedCount; idx++)
					{
						Buffer		neighborBuf;
						Page		neighborPage;
						HnswNode	neighborNode;
						BlockNumber *neighborNeighbors;
						int16		neighborNeighborCount;
						int			insertPos;
						bool		needsPruning = false;

						if (idx < m)
						{
							newNodeNeighbors[idx] = selectedNeighbors[idx];
							newNode->neighborCount[currentLevel] = idx + 1;
						}

						neighborBuf = ReadBuffer(index, selectedNeighbors[idx]);
				LockBuffer(neighborBuf, BUFFER_LOCK_EXCLUSIVE);
				neighborPage = BufferGetPage(neighborBuf);
				neighborNode = (HnswNode)
					PageGetItem(neighborPage, PageGetItemId(neighborPage, FirstOffsetNumber));
				if (neighborNode == NULL)
				{
					UnlockReleaseBuffer(neighborBuf);
					continue;
				}

				/* Validate neighbor node level */
				if (!hnswValidateLevelSafe(neighborNode->level))
				{
					UnlockReleaseBuffer(neighborBuf);
					continue;
				}

				neighborNeighbors = HnswGetNeighborsSafe(neighborNode, currentLevel, m);
				neighborNeighborCount = neighborNode->neighborCount[currentLevel];

				/* Validate and clamp neighborCount */
				neighborNeighborCount = hnswValidateNeighborCount(neighborNeighborCount, m, currentLevel);

				insertPos = neighborNeighborCount;
				for (j = 0; j < neighborNeighborCount; j++)
				{
					if (neighborNeighbors[j] == InvalidBlockNumber)
					{
						insertPos = j;
						break;
					}
				}

				if (insertPos < m * 2)
				{
					neighborNeighbors[insertPos] = blkno;
					if (insertPos >= neighborNeighborCount)
						neighborNode->neighborCount[currentLevel] = insertPos + 1;
					MarkBufferDirty(neighborBuf);
				}

				/* Prune to at most m*2 nearest neighbors */
				if (neighborNode->neighborCount[currentLevel] > m * 2)
					needsPruning = true;

				if (needsPruning)
				{
					float4	   *neighborVector = HnswGetVector(neighborNode);
					int16		pruneCount = neighborNode->neighborCount[currentLevel];
					float4	   *neighborDists = NULL;
					int		   *neighborIndices = NULL;

					/* Ensure pruneCount is within valid bounds */
					if (pruneCount > m * 2)
						pruneCount = m * 2;
					if (pruneCount < 0)
						pruneCount = 0;

					NDB_ALLOC(neighborDists, float4, pruneCount);
					NDB_ALLOC(neighborIndices, int, pruneCount);

					for (j = 0; j < pruneCount; j++)
					{
						if (neighborNeighbors[j] == InvalidBlockNumber)
							break;
						neighborIndices[j] = j;
						if (neighborNeighbors[j] == blkno)
							neighborDists[j] = selectedDistances[idx];
						else
						{
							Buffer		otherBuf;
							Page		otherPage;
							HnswNode	otherNode;
							float4	   *otherVector;

							if (!hnswValidateBlockNumber(neighborNeighbors[j], index))
							{
								neighborDists[j] = FLT_MAX;  /* Mark as invalid */
								continue;
							}

							otherBuf = ReadBuffer(index, neighborNeighbors[j]);
							LockBuffer(otherBuf, BUFFER_LOCK_SHARE);
							otherPage = BufferGetPage(otherBuf);

							if (PageIsNew(otherPage) || PageIsEmpty(otherPage))
							{
								UnlockReleaseBuffer(otherBuf);
								neighborDists[j] = FLT_MAX;  /* Mark as invalid */
								continue;
							}

							otherNode = (HnswNode)
								PageGetItem(otherPage, PageGetItemId(otherPage, FirstOffsetNumber));
							if (otherNode == NULL)
							{
								UnlockReleaseBuffer(otherBuf);
								neighborDists[j] = FLT_MAX;  /* Mark as invalid */
								continue;
							}

							otherVector = HnswGetVector(otherNode);
							if (otherVector == NULL)
							{
								UnlockReleaseBuffer(otherBuf);
								neighborDists[j] = FLT_MAX;  /* Mark as invalid */
								continue;
							}

							neighborDists[j] = hnswComputeDistance(neighborVector, otherVector, dim, 1);
							UnlockReleaseBuffer(otherBuf);
						}
					}
					for (j = 0; j < pruneCount - 1; j++)
					{
						int			k;

						for (k = j + 1; k < pruneCount; k++)
						{
							if (neighborDists[k] < neighborDists[j])
							{
								float4		tmpDist = neighborDists[j];
								int			tmpIdx = neighborIndices[j];

								neighborDists[j] = neighborDists[k];
								neighborIndices[j] = neighborIndices[k];
								neighborDists[k] = tmpDist;
								neighborIndices[k] = tmpIdx;
							}
						}
					}
					neighborNode->neighborCount[currentLevel] = m * 2;
					for (j = 0; j < m * 2; j++)
						neighborNeighbors[j] = neighborNeighbors[neighborIndices[j]];
					/* Neighbors array for this level has exactly m*2 slots,
					 * so no need to clear beyond that. */

					NDB_FREE(neighborDists);
					neighborDists = NULL;
					NDB_FREE(neighborIndices);
					neighborIndices = NULL;
					MarkBufferDirty(neighborBuf);
				}
				UnlockReleaseBuffer(neighborBuf);
					}
				}

			if (selectedNeighbors)
			{
				NDB_FREE(selectedNeighbors);
				selectedNeighbors = NULL;
			}
			if (selectedDistances)
			{
				NDB_FREE(selectedDistances);
				selectedDistances = NULL;
			}

			if (candidates)
			{
				NDB_FREE(candidates);
				candidates = NULL;
			}
			if (candidateDistances)
			{
				NDB_FREE(candidateDistances);
				candidateDistances = NULL;
			}
			}
			if (BufferIsValid(newNodeBuf))
			{
				MarkBufferDirty(newNodeBuf);
				UnlockReleaseBuffer(newNodeBuf);
				newNodeBuf = InvalidBuffer;
			}
		}
	}

	/* Step 6: Update entry point and meta info if necessary */
	if (metaPage->entryPoint == InvalidBlockNumber || level > metaPage->entryLevel)
	{
		if (hnswValidateBlockNumber(blkno, index))
		{
			metaPage->entryPoint = blkno;
			metaPage->entryLevel = level;
		}
		else
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("hnsw: invalid block number %u for entry point", blkno)));
		}
	}

	metaPage->insertedVectors++;
	if (level > metaPage->maxLevel)
		metaPage->maxLevel = level;

	NDB_FREE(node);
	node = NULL;
}

/*
 * hnswFindNodeByTid - Find HNSW node by ItemPointer.
 *
 * Scans all pages in the index to locate the node matching the given
 * ItemPointer. Returns true if found, setting *outBlkno and *outOffset.
 */
static bool
hnswFindNodeByTid(Relation index,
				  ItemPointer tid,
				  BlockNumber * outBlkno,
				  OffsetNumber * outOffset)
{
	BlockNumber blkno;
	Buffer		buf;
	Page		page;
	OffsetNumber maxoff;
	OffsetNumber offnum;
	HnswNode	node;

	*outBlkno = InvalidBlockNumber;
	*outOffset = InvalidOffsetNumber;

	/* Scan all pages in the index */
	for (blkno = 1; blkno < RelationGetNumberOfBlocks(index); blkno++)
	{
		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);

		if (PageIsNew(page) || PageIsEmpty(page))
		{
			UnlockReleaseBuffer(buf);
			continue;
		}

		maxoff = PageGetMaxOffsetNumber(page);

		/* Enforce one-node-per-page invariant */
		if (maxoff != FirstOffsetNumber)
		{
			elog(WARNING, "hnsw: page %u has %d items, expected 1 (one-node-per-page invariant violated)",
				 blkno, maxoff);
			UnlockReleaseBuffer(buf);
			continue;
		}

		for (offnum = FirstOffsetNumber; offnum <= maxoff; offnum = OffsetNumberNext(offnum))
		{
			ItemId		itemId = PageGetItemId(page, offnum);

			if (!ItemIdIsValid(itemId))
				continue;

			node = (HnswNode) PageGetItem(page, itemId);

			/* Check if this node matches the ItemPointer */
			if (ItemPointerEquals(&node->heapPtr, tid))
			{
				*outBlkno = blkno;
				*outOffset = offnum;
				UnlockReleaseBuffer(buf);
				return true;
			}
		}

		UnlockReleaseBuffer(buf);
	}

	return false;
}

/*
 * Helper: Remove node from neighbor's neighbor list
 */
static void
hnswRemoveNodeFromNeighbor(Relation index,
						   BlockNumber neighborBlkno,
						   BlockNumber nodeBlkno,
						   int level)
{
	Buffer		buf;
	Page		page;
	HnswNode	neighbor;
	BlockNumber *neighbors;
	int16		neighborCount;
	int			i,
				j;
	bool		found = false;

	if (!hnswValidateBlockNumber(neighborBlkno, index))
	{
		elog(WARNING, "hnsw: invalid neighbor block %u in RemoveNodeFromNeighbor", neighborBlkno);
		return;
	}

	if (!hnswValidateLevelSafe(level))
	{
		elog(WARNING, "hnsw: invalid level %d in RemoveNodeFromNeighbor", level);
		return;
	}

	buf = ReadBuffer(index, neighborBlkno);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	page = BufferGetPage(buf);

	/* Get first item on page (assuming one node per page for simplicity) */
	if (PageIsEmpty(page))
	{
		UnlockReleaseBuffer(buf);
		return;
	}

	neighbor = (HnswNode) PageGetItem(page, PageGetItemId(page, FirstOffsetNumber));
	if (neighbor == NULL)
	{
		UnlockReleaseBuffer(buf);
		return;
	}

	/* Validate neighbor level */
	if (!hnswValidateLevelSafe(neighbor->level))
	{
		elog(WARNING, "hnsw: invalid neighbor level %d in RemoveNodeFromNeighbor", neighbor->level);
		UnlockReleaseBuffer(buf);
		return;
	}

	/* Get m from meta page - we need to read it */
	{
		Buffer		metaBuf;
		Page		metaPage;
		HnswMetaPage meta;
		int			m;

		metaBuf = ReadBuffer(index, 0);
		LockBuffer(metaBuf, BUFFER_LOCK_SHARE);
		metaPage = BufferGetPage(metaBuf);
		meta = (HnswMetaPage) PageGetContents(metaPage);
		m = meta->m;
		UnlockReleaseBuffer(metaBuf);

		neighbors = HnswGetNeighborsSafe(neighbor, level, m);
		neighborCount = neighbor->neighborCount[level];

		/* Validate and clamp neighborCount */
		neighborCount = hnswValidateNeighborCount(neighborCount, m, level);
	}

	for (i = 0; i < neighborCount; i++)
	{
		if (neighbors[i] == nodeBlkno)
		{
			found = true;
			/* Shift remaining neighbors */
			for (j = i; j < neighborCount - 1; j++)
				neighbors[j] = neighbors[j + 1];
			neighbors[neighborCount - 1] = InvalidBlockNumber;
			neighbor->neighborCount[level]--;
			break;
		}
	}

	if (found)
	{
		MarkBufferDirty(buf);
	}

	UnlockReleaseBuffer(buf);
}

static bool
hnswdelete(Relation index,
		   ItemPointer tid,
		   Datum * values,
		   bool *isnull,
		   Relation heapRel,
		   struct IndexInfo *indexInfo)
{
	BlockNumber nodeBlkno;
	OffsetNumber nodeOffset;
	Buffer		nodeBuf;
	Page		nodePage;
	HnswNode	node;
	Buffer		metaBuffer;
	Page		metaPage;
	HnswMetaPage meta;
	int			level;
	int			i;
	BlockNumber *neighbors;
	int16		neighborCount;

	if (!hnswFindNodeByTid(index, tid, &nodeBlkno, &nodeOffset))
	{
		/* Node not found - already deleted or never existed */
		return true;
	}

	/* Read metadata */
	metaBuffer = ReadBuffer(index, 0);
	LockBuffer(metaBuffer, BUFFER_LOCK_EXCLUSIVE);
	metaPage = BufferGetPage(metaBuffer);
	meta = (HnswMetaPage) PageGetContents(metaPage);

	/* Read the node to be deleted */
	nodeBuf = ReadBuffer(index, nodeBlkno);
	LockBuffer(nodeBuf, BUFFER_LOCK_EXCLUSIVE);
	nodePage = BufferGetPage(nodeBuf);
	node = (HnswNode) PageGetItem(nodePage, PageGetItemId(nodePage, nodeOffset));

	/* Validate node level */
	if (!hnswValidateLevelSafe(node->level))
	{
		UnlockReleaseBuffer(nodeBuf);
		UnlockReleaseBuffer(metaBuffer);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("hnsw: invalid node level %d in delete", node->level)));
	}

	/*
	 * For each level where this node exists, remove it from neighbor
	 * connections
	 */
	for (level = 0; level <= node->level; level++)
	{
		neighbors = HnswGetNeighborsSafe(node, level, meta->m);
		neighborCount = node->neighborCount[level];

		/* Validate and clamp neighborCount */
		neighborCount = hnswValidateNeighborCount(neighborCount, meta->m, level);

		for (i = 0; i < neighborCount; i++)
		{
			if (neighbors[i] != InvalidBlockNumber &&
				hnswValidateBlockNumber(neighbors[i], index))
			{
				hnswRemoveNodeFromNeighbor(index, neighbors[i], nodeBlkno, level);
			}
		}
	}

	if (meta->entryPoint == nodeBlkno)
	{
		bool		foundNewEntry = false;
		int			bestLevel = -1;
		BlockNumber bestEntry = InvalidBlockNumber;

		for (level = node->level; level >= 0; level--)
		{
			neighbors = HnswGetNeighborsSafe(node, level, meta->m);
			neighborCount = node->neighborCount[level];

			/* Validate and clamp neighborCount */
			neighborCount = hnswValidateNeighborCount(neighborCount, meta->m, level);

			for (i = 0; i < neighborCount; i++)
			{
				if (neighbors[i] != InvalidBlockNumber &&
					hnswValidateBlockNumber(neighbors[i], index))
				{
					/* Check the actual level of this neighbor */
					Buffer		neighborBuf;
					Page		neighborPage;
					HnswNode	neighborNode;
					ItemId		neighborItemId;

					neighborBuf = ReadBuffer(index, neighbors[i]);
					LockBuffer(neighborBuf, BUFFER_LOCK_SHARE);
					neighborPage = BufferGetPage(neighborBuf);

					if (!PageIsEmpty(neighborPage))
					{
						neighborItemId = PageGetItemId(neighborPage, FirstOffsetNumber);
						if (ItemIdIsValid(neighborItemId))
						{
							neighborNode = (HnswNode) PageGetItem(neighborPage, neighborItemId);
							if (neighborNode != NULL &&
								hnswValidateLevelSafe(neighborNode->level) &&
								neighborNode->level > bestLevel)
							{
								bestLevel = neighborNode->level;
								bestEntry = neighbors[i];
								foundNewEntry = true;
							}
						}
					}

					UnlockReleaseBuffer(neighborBuf);
				}
			}
		}

		/* Set new entry point if found */
		if (foundNewEntry)
		{
			meta->entryPoint = bestEntry;
			meta->entryLevel = bestLevel;
		}
		else
		{
			/* If no neighbor found, mark entry as invalid */
			meta->entryPoint = InvalidBlockNumber;
			meta->entryLevel = -1;
		}
	}

	/* Mark node page for deletion (actual deletion handled by vacuum) */
	/* For now, we mark the item as deleted */
	{
		ItemId		itemId = PageGetItemId(nodePage, nodeOffset);

		if (ItemIdIsValid(itemId))
		{
			ItemIdSetDead(itemId);
			MarkBufferDirty(nodeBuf);
		}
	}

	meta->insertedVectors--;
	if (meta->insertedVectors < 0)
		meta->insertedVectors = 0;
	MarkBufferDirty(metaBuffer);

	UnlockReleaseBuffer(nodeBuf);
	UnlockReleaseBuffer(metaBuffer);

	return true;
}

/*
 * Update: delete old value, insert new value
 * This is the standard HNSW update pattern: remove old node from graph,
 * then insert new node with updated vector.
 */
static bool
hnswupdate(Relation index,
		   ItemPointer tid,
		   Datum * values,
		   bool *isnull,
		   ItemPointer otid,
		   Relation heapRel,
		   struct IndexInfo *indexInfo)
{
	bool		deleteResult;
	bool		insertResult;

	/*
	 * Generic HNSW update = delete old, insert new. First delete the old
	 * value, then insert the new one.
	 */
	deleteResult = hnswdelete(index, otid, values, isnull, heapRel, indexInfo);
	if (!deleteResult)
	{
		/*
		 * If delete failed (e.g., old node not found), still try to insert
		 * new value
		 */
		elog(DEBUG1,
			 "neurondb: HNSW update: delete of old value failed (may not exist), "
			 "proceeding with insert");
	}

	/* Insert the new value */
	insertResult = hnswinsert(index, values, isnull, tid, heapRel,
							  UNIQUE_CHECK_NO, false, indexInfo);

	/*
	 * Update succeeds if insert succeeds (delete failure is acceptable if
	 * node didn't exist)
	 */
	return insertResult;
}

