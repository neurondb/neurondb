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
 */
#define HNSW_DEFAULT_M			16
#define HNSW_DEFAULT_EF_CONSTRUCTION	200
#define HNSW_DEFAULT_EF_SEARCH		64
#define HNSW_DEFAULT_ML			0.36f
#define HNSW_MAX_LEVEL			16
#define HNSW_MAGIC_NUMBER		0x48534E57
#define HNSW_VERSION			1

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

#define HnswNodeSize(dim, level) \
	(MAXALIGN(sizeof(HnswNodeData) + (dim) * sizeof(float4) \
		+ ((level) + 1) * HNSW_DEFAULT_M * 2 * sizeof(BlockNumber)))

#define HnswGetVector(node) \
	((float4 *)((char *)(node) + MAXALIGN(sizeof(HnswNodeData))))

#define HnswGetNeighbors(node, lev) \
	((BlockNumber *)((char *)(node) + MAXALIGN(sizeof(HnswNodeData)) \
		+ (node)->dim * sizeof(float4) \
		+ (lev) * HNSW_DEFAULT_M * 2 * sizeof(BlockNumber)))

/*
 * Build state for index build
 */
typedef struct HnswBuildState
{
	Relation	heap;
	Relation	index;
	IndexInfo  *indexInfo;
	HnswMetaPage metaPage;
	double		indtuples;
	Buffer		metaBuffer;
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
static bool hnswinsert(Relation index, Datum *values, bool *isnull, ItemPointer ht_ctid,
					   Relation heapRel, IndexUniqueCheck checkUnique,
					   bool indexUnchanged, struct IndexInfo *indexInfo);
static IndexBulkDeleteResult * hnswbulkdelete(IndexVacuumInfo * info,
											  IndexBulkDeleteResult * stats,
											  IndexBulkDeleteCallback callback,
											  void *callback_state);
static IndexBulkDeleteResult * hnswvacuumcleanup(IndexVacuumInfo * info,
												 IndexBulkDeleteResult * stats);
static bool hnswdelete(Relation index, ItemPointer tid, Datum *values, bool *isnull,
					   Relation heapRel, struct IndexInfo *indexInfo) __attribute__((unused));
static bool hnswupdate(Relation index, ItemPointer tid, Datum *values, bool *isnull,
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
static void hnswBuildCallback(Relation index, ItemPointer tid, Datum *values,
							  bool *isnull, bool tupleIsAlive, void *state);

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
	IndexBuildResult *result;
	HnswBuildState buildstate;
	Buffer		metaBuffer;
	Page		metaPage;
	HnswOptions *options;
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

	metaBuffer = ReadBuffer(index, P_NEW);
	LockBuffer(metaBuffer, BUFFER_LOCK_EXCLUSIVE);
	metaPage = BufferGetPage(metaBuffer);

	options = (HnswOptions *) indexInfo->ii_AmCache;
	if (options == NULL)
	{
		static const relopt_parse_elt tab[] = {
			{"m", RELOPT_TYPE_INT, offsetof(HnswOptions, m)},
			{"ef_construction", RELOPT_TYPE_INT, offsetof(HnswOptions, ef_construction)},
			{"ef_search", RELOPT_TYPE_INT, offsetof(HnswOptions, ef_search)}
		};
		Datum		relopts = PointerGetDatum(index->rd_options);

		/* Lazy initialization: ensure relopt_kind_hnsw is registered */
		if (relopt_kind_hnsw == 0)
		{
			relopt_kind_hnsw = add_reloption_kind();
			elog(DEBUG1, "neurondb: lazily registered relopt_kind_hnsw = %d", relopt_kind_hnsw);
		}

		options = (HnswOptions *) build_reloptions(relopts, false,
												   relopt_kind_hnsw,
												   sizeof(HnswOptions), tab, lengthof(tab));
		indexInfo->ii_AmCache = (void *) options;
	}
	m = options ? options->m : HNSW_DEFAULT_M;
	ef_construction = options ? options->ef_construction : HNSW_DEFAULT_EF_CONSTRUCTION;
	ef_search = options ? options->ef_search : HNSW_DEFAULT_EF_SEARCH;

	hnswInitMetaPage(metaBuffer, m, ef_construction, ef_search, HNSW_DEFAULT_ML);

	buildstate.metaBuffer = metaBuffer;
	buildstate.metaPage = (HnswMetaPage) PageGetContents(metaPage);

	MarkBufferDirty(metaBuffer);
	UnlockReleaseBuffer(metaBuffer);

	/* Use parallel scan if available */
	buildstate.indtuples = table_index_build_scan(heap, index, indexInfo,
												  true, true, hnswBuildCallback,
												  (void *) &buildstate, NULL);

	result = (IndexBuildResult *) palloc(sizeof(IndexBuildResult));
	result->heap_tuples = buildstate.indtuples;
	result->index_tuples = buildstate.indtuples;

	MemoryContextDelete(buildstate.tmpCtx);
	elog(INFO, "neurondb: HNSW index build complete, indexed %.0f tuples",
		 buildstate.indtuples);

	return result;
}

static void
hnswBuildCallback(Relation index, ItemPointer tid, Datum *values,
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

	metaBuffer = ReadBuffer(index, P_NEW);
	LockBuffer(metaBuffer, BUFFER_LOCK_EXCLUSIVE);

	hnswInitMetaPage(metaBuffer,
					 HNSW_DEFAULT_M,
					 HNSW_DEFAULT_EF_CONSTRUCTION,
					 HNSW_DEFAULT_EF_SEARCH,
					 HNSW_DEFAULT_ML);

	MarkBufferDirty(metaBuffer);
	UnlockReleaseBuffer(metaBuffer);
}

static bool
hnswinsert(Relation index,
		   Datum *values,
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
		LockBuffer(metaBuffer, BUFFER_LOCK_SHARE);
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

	if (stats == NULL)
		stats = (IndexBulkDeleteResult *) palloc0(sizeof(IndexBulkDeleteResult));

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

			/* Check callback to see if this tuple should be deleted */
			if (callback(&node->heapPtr, callback_state))
			{
				/* Remove node from graph structure */
				/* For each level where this node exists */
				for (level = 0; level <= node->level; level++)
				{
					neighbors = HnswGetNeighbors(node, level);
					neighborCount = node->neighborCount[level];

					/* Remove this node from each neighbor's neighbor list */
					for (i = 0; i < neighborCount; i++)
					{
						if (neighbors[i] != InvalidBlockNumber)
						{
							hnswRemoveNodeFromNeighbor(index,
													   neighbors[i],
													   blkno,
													   level);
						}
					}
				}

				/* Update entry point if this node was the entry point */
				if (meta->entryPoint == blkno)
				{
					foundNewEntry = false;

					/* Find a new entry point from neighbors at highest level */
					for (level = node->level;
						 level >= 0 && !foundNewEntry;
						 level--)
					{
						neighbors = HnswGetNeighbors(node, level);
						neighborCount = node->neighborCount[level];
						for (i = 0; i < neighborCount && !foundNewEntry; i++)
						{
							if (neighbors[i] != InvalidBlockNumber)
							{
								/* Use first valid neighbor as new entry point */
								meta->entryPoint = neighbors[i];
								/* Find actual level of neighbor */
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
										meta->entryLevel = tmpNode->level;
									}
									UnlockReleaseBuffer(tmpBuf);
								}
								foundNewEntry = true;
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

				/* Update statistics */
				stats->tuples_removed++;
				meta->insertedVectors--;
				if (meta->insertedVectors < 0)
					meta->insertedVectors = 0;
			}
		}

		UnlockReleaseBuffer(nodeBuf);
	}

	/* Update metadata if changed */
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
	if (stats == NULL)
		stats = (IndexBulkDeleteResult *) palloc0(sizeof(IndexBulkDeleteResult));
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
	*indexStartupCost = 50.0;
	*indexTotalCost = 100.0;
	*indexSelectivity = 0.01;
	*indexCorrelation = 0.0;
	*indexPages = 10;
}

static bytea *
hnswoptions(Datum reloptions, bool validate)
{
	static const relopt_parse_elt tab[] = {
		{"m", RELOPT_TYPE_INT, offsetof(HnswOptions, m)},
		{"ef_construction", RELOPT_TYPE_INT, offsetof(HnswOptions, ef_construction)},
		{"ef_search", RELOPT_TYPE_INT, offsetof(HnswOptions, ef_search)}
	};

	/* Lazy initialization: ensure relopt_kind_hnsw is registered */
	if (relopt_kind_hnsw == 0)
	{
		relopt_kind_hnsw = add_reloption_kind();
		elog(DEBUG1, "neurondb: lazily registered relopt_kind_hnsw = %d", relopt_kind_hnsw);
	}

	return (bytea *) build_reloptions(reloptions, validate, relopt_kind_hnsw,
									  sizeof(HnswOptions),
									  tab, lengthof(tab));
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
	HnswScanOpaque so;

	scan = RelationGetIndexScan(index, nkeys, norderbys);
	so = (HnswScanOpaque) palloc0(sizeof(HnswScanOpaqueData));
	so->efSearch = HNSW_DEFAULT_EF_SEARCH;
	so->strategy = 1;
	so->firstCall = true;

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
			if (so->queryVector)
			{
				NDB_FREE(so->queryVector);
				so->queryVector = NULL;
			}
			so->queryVector = (Vector *) palloc(VECTOR_SIZE(dim));
			SET_VARSIZE(so->queryVector, VECTOR_SIZE(dim));
			so->queryVector->dim = dim;
			memcpy(so->queryVector->data, vectorData, dim * sizeof(float4));
			NDB_FREE(vectorData);
			vectorData = NULL;
		}
		so->k = 10;
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

		/* Read the node to get its heap pointer */
		buf = ReadBuffer(scan->indexRelation, resultBlkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);

		if (!PageIsEmpty(page))
		{
			node = (HnswNode) PageGetItem(page, PageGetItemId(page, FirstOffsetNumber));
			scan->xs_heaptid = node->heapPtr;
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

/* ------- HNSW Core Operations: Node/MetaPage/Distance/Search/Insert/Update/Delete ------- */

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
 * Extract vector from datum for type OID (vector/halfvec/sparsevec/bit)
 */
static float4 *
hnswExtractVectorData(Datum value, Oid typeOid, int *out_dim, MemoryContext ctx)
{
	MemoryContext oldctx;
	Oid			vectorOid,
				halfvecOid,
				sparsevecOid,
				bitOid;
	float4	   *result = NULL;
	int			i;

	oldctx = MemoryContextSwitchTo(ctx);

	{
		List	   *names;

		names = list_make2(makeString("public"), makeString("vector"));
		vectorOid = LookupTypeNameOid(NULL, makeTypeNameFromNameList(names), false);
		list_free(names);
		names = list_make2(makeString("public"), makeString("halfvec"));
		halfvecOid = LookupTypeNameOid(NULL, makeTypeNameFromNameList(names), false);
		list_free(names);
		names = list_make2(makeString("public"), makeString("sparsevec"));
		sparsevecOid = LookupTypeNameOid(NULL, makeTypeNameFromNameList(names), false);
		list_free(names);
		bitOid = BITOID;
	}

	if (typeOid == vectorOid)
	{
		Vector	   *v = DatumGetVector(value);

		NDB_CHECK_VECTOR_VALID(v);
		*out_dim = v->dim;
		NDB_CHECK_ALLOC_SIZE((size_t) v->dim * sizeof(float4), "vector data");
		result = (float4 *) palloc(v->dim * sizeof(float4));
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
		result = (float4 *) palloc(hv->dim * sizeof(float4));
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
			result = (float4 *) palloc0(sv->total_dim * sizeof(float4));
			NDB_CHECK_ALLOC(result, "sparsevec data");
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
			result = (float4 *) palloc(nbits * sizeof(float4));
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
	BlockNumber *candidates = NULL;
	float4	   *candidateDists = NULL;
	int			candidateCount = 0;
	BlockNumber *visited = NULL;
	int			visitedCount = 0;
	int			visitedCapacity = 0;
	bool	   *visitedSet = NULL;
	BlockNumber *neighbors;
	int16		neighborCount;
	BlockNumber *topK = NULL;
	float4	   *topKDists = NULL;
	int			topKCount = 0;
	int		   *indices = NULL;
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
		current = metaPage->entryPoint;
		currentLevel = metaPage->entryLevel;

		visitedCapacity = (efSearch > 1 ? efSearch * 2 : 32);
		visited = (BlockNumber *) palloc(visitedCapacity * sizeof(BlockNumber));
		visitedSet = (bool *) palloc0(RelationGetNumberOfBlocks(index) * sizeof(bool));
		visitedCount = 0;

		candidates = (BlockNumber *) palloc(efSearch * sizeof(BlockNumber));
		candidateDists = (float4 *) palloc(efSearch * sizeof(float4));
		candidateCount = 0;

		/* Step 1: Greedy search - entry at top down to level 1 */
		for (level = currentLevel; level > 0; level--)
		{
			bool		foundBetter;

			do
			{
				foundBetter = false;
				nodeBuf = ReadBuffer(index, current);
				LockBuffer(nodeBuf, BUFFER_LOCK_SHARE);
				nodePage = BufferGetPage(nodeBuf);
				node = (HnswNode) PageGetItem(nodePage,
											  PageGetItemId(nodePage, FirstOffsetNumber));
				nodeVector = HnswGetVector(node);
				currentDist = hnswComputeDistance(query, nodeVector, dim, strategy);

				if (node->level >= level)
				{
					neighbors = HnswGetNeighbors(node, level);
					neighborCount = node->neighborCount[level];

					for (i = 0; i < neighborCount; i++)
					{
						Buffer		neighborBuf;
						Page		neighborPage;
						HnswNode	neighbor;
						float4	   *neighborVector;
						float4		neighborDist;

						if (neighbors[i] == InvalidBlockNumber)
							continue;

						neighborBuf = ReadBuffer(index, neighbors[i]);
						LockBuffer(neighborBuf, BUFFER_LOCK_SHARE);
						neighborPage = BufferGetPage(neighborBuf);
						neighbor = (HnswNode) PageGetItem(neighborPage,
														  PageGetItemId(neighborPage, FirstOffsetNumber));
						neighborVector = HnswGetVector(neighbor);
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

		/* Step 2: ef-search at level 0 */
		candidates[0] = current;
		nodeBuf = ReadBuffer(index, current);
		LockBuffer(nodeBuf, BUFFER_LOCK_SHARE);
		nodePage = BufferGetPage(nodeBuf);
		node = (HnswNode) PageGetItem(nodePage,
									  PageGetItemId(nodePage, FirstOffsetNumber));
		nodeVector = HnswGetVector(node);
		candidateDists[0] = hnswComputeDistance(query, nodeVector, dim, strategy);
		candidateCount = 1;
		visited[visitedCount++] = current;
		if (current < RelationGetNumberOfBlocks(index))
			visitedSet[current] = true;
		UnlockReleaseBuffer(nodeBuf);

		for (i = 0; i < candidateCount && candidateCount < efSearch; i++)
		{
			BlockNumber candidate = candidates[i];

			nodeBuf = ReadBuffer(index, candidate);
			LockBuffer(nodeBuf, BUFFER_LOCK_SHARE);
			nodePage = BufferGetPage(nodeBuf);
			node = (HnswNode) PageGetItem(nodePage,
										  PageGetItemId(nodePage, FirstOffsetNumber));
			neighbors = HnswGetNeighbors(node, 0);
			neighborCount = node->neighborCount[0];

			for (j = 0; j < neighborCount; j++)
			{
				Buffer		neighborBuf;
				Page		neighborPage;
				HnswNode	neighbor;
				float4	   *neighborVector;
				float4		neighborDist;

				if (neighbors[j] == InvalidBlockNumber)
					continue;

				if (neighbors[j] < RelationGetNumberOfBlocks(index) &&
					visitedSet[neighbors[j]])
					continue;

				neighborBuf = ReadBuffer(index, neighbors[j]);
				LockBuffer(neighborBuf, BUFFER_LOCK_SHARE);
				neighborPage = BufferGetPage(neighborBuf);
				neighbor = (HnswNode) PageGetItem(neighborPage,
												  PageGetItemId(neighborPage, FirstOffsetNumber));
				neighborVector = HnswGetVector(neighbor);
				neighborDist = hnswComputeDistance(query, neighborVector, dim, strategy);
				UnlockReleaseBuffer(neighborBuf);

				if (neighbors[j] < RelationGetNumberOfBlocks(index))
					visitedSet[neighbors[j]] = true;
				visited[visitedCount++] = neighbors[j];
				if (visitedCount >= visitedCapacity)
				{
					visitedCapacity = Max(32, visitedCapacity * 2);
					visited = (BlockNumber *) repalloc(visited,
													   visitedCapacity * sizeof(BlockNumber));
				}

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
					for (l = 1; l < candidateCount; l++)
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

		/* Step 3: Partial selection sort for top-k */
		indices = (int *) palloc(candidateCount * sizeof(int));
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
		topK = (BlockNumber *) palloc(topKCount * sizeof(BlockNumber));
		topKDists = (float4 *) palloc(topKCount * sizeof(float4));
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
 * hnswInsertNode
 *		Insert a node into the HNSW graph, updating neighbor links.
 *
 * Follows PostgreSQL C coding standards and structure.
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
	HnswNode	node;
	BlockNumber blkno;
	Size		nodeSize;
	int			i;

	/* Step 1: Assign random level using exponential law */
	level = hnswGetRandomLevel(metaPage->ml);

	/* Enforce limit on level */
	if (level >= HNSW_MAX_LEVEL)
		level = HNSW_MAX_LEVEL - 1;

	/* Step 2: Allocate and initialize the node */
	nodeSize = HnswNodeSize(dim, level);
	node = (HnswNode) palloc0(nodeSize);
	ItemPointerCopy(heapPtr, &node->heapPtr);
	node->level = level;
	node->dim = dim;
	for (i = 0; i < HNSW_MAX_LEVEL; i++)
		node->neighborCount[i] = 0;
	memcpy(HnswGetVector(node), vector, dim * sizeof(float4));
	/* Neighbors will be assigned below */

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

				entryBuf = ReadBuffer(index, bestEntry);
				LockBuffer(entryBuf, BUFFER_LOCK_SHARE);
				entryPage = BufferGetPage(entryBuf);
				entryNode = (HnswNode) PageGetItem(entryPage,
												   PageGetItemId(entryPage, FirstOffsetNumber));

				if (entryNode->level >= level)
				{
					entryVector = HnswGetVector(entryNode);
					bestDist = hnswComputeDistance(vector, entryVector, dim, 1);
					entryNeighbors = HnswGetNeighbors(entryNode, level);
					entryNeighborCount = entryNode->neighborCount[level];

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
		BlockNumber *candidates = NULL;
		float4	   *candidateDistances = NULL;
		int			candidateCount = 0;
		int			currentLevel;
		BlockNumber *newNodeNeighbors;
		Buffer		newNodeBuf;
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

		for (currentLevel = Min(level, entryLevel);
			 currentLevel >= 0;
			 currentLevel--)
		{
			BlockNumber *selectedNeighbors;
			int			selectedCount;
			float4	   *selectedDistances;

			/* Find neighbor candidates for this level */
			hnswSearch(index,
					   metaPage,
					   vector,
					   dim,
					   1,		/* L2 distance */
					   efConstruction,
					   efConstruction,
					   &candidates,
					   &candidateDistances,
					   &candidateCount);

			selectedCount = Min(m, candidateCount);
			selectedNeighbors = (BlockNumber *) palloc(selectedCount * sizeof(BlockNumber));
			selectedDistances = (float4 *) palloc(selectedCount * sizeof(float4));

			/* Sort by distance: select top m */
			for (idx = 0; idx < selectedCount; idx++)
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

			/*
			 * Link new node to neighbors, and each neighbor back
			 * (bidirectional)
			 */
			newNodeNeighbors = HnswGetNeighbors(newNode, currentLevel);
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
				neighborNeighbors = HnswGetNeighbors(neighborNode, currentLevel);
				neighborNeighborCount = neighborNode->neighborCount[currentLevel];

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
					float4	   *neighborDists = (float4 *)
						palloc(neighborNode->neighborCount[currentLevel] * sizeof(float4));
					int		   *neighborIndices = (int *)
						palloc(neighborNode->neighborCount[currentLevel] * sizeof(int));

					for (j = 0; j < neighborNode->neighborCount[currentLevel]; j++)
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

							otherBuf = ReadBuffer(index, neighborNeighbors[j]);
							LockBuffer(otherBuf, BUFFER_LOCK_SHARE);
							otherPage = BufferGetPage(otherBuf);
							otherNode = (HnswNode)
								PageGetItem(otherPage, PageGetItemId(otherPage, FirstOffsetNumber));
							otherVector = HnswGetVector(otherNode);

							neighborDists[j] = hnswComputeDistance(neighborVector, otherVector, dim, 1);
							UnlockReleaseBuffer(otherBuf);
						}
					}
					for (j = 0; j < neighborNode->neighborCount[currentLevel] - 1; j++)
					{
						int			k;

						for (k = j + 1; k < neighborNode->neighborCount[currentLevel]; k++)
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
					for (j = m * 2; j < HNSW_DEFAULT_M * 2; j++)
						neighborNeighbors[j] = InvalidBlockNumber;

					NDB_FREE(neighborDists);
					neighborDists = NULL;
					NDB_FREE(neighborIndices);
					neighborIndices = NULL;
					MarkBufferDirty(neighborBuf);
				}
				UnlockReleaseBuffer(neighborBuf);
			}

			NDB_FREE(selectedNeighbors);
			selectedNeighbors = NULL;
			NDB_FREE(selectedDistances);
			selectedDistances = NULL;

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
		MarkBufferDirty(newNodeBuf);
		UnlockReleaseBuffer(newNodeBuf);
	}

	/* Step 6: Update entry point and meta info if necessary */
	if (metaPage->entryPoint == InvalidBlockNumber || level > metaPage->entryLevel)
	{
		metaPage->entryPoint = blkno;
		metaPage->entryLevel = level;
	}

	/* Step 7: Update meta statistics */
	metaPage->insertedVectors++;
	if (level > metaPage->maxLevel)
		metaPage->maxLevel = level;

	NDB_FREE(node);
	node = NULL;
}

/*
 * Delete vector from HNSW index.
 * Implementation: Removes node from graph, updates neighbor connections,
 * and handles entry point reassignment if needed.
 */
/*
 * Helper: Find HNSW node by ItemPointer
 * Returns the block number and offset if found, InvalidBlockNumber otherwise
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
	neighbors = HnswGetNeighbors(neighbor, level);
	neighborCount = neighbor->neighborCount[level];

	/* Find and remove nodeBlkno from neighbors */
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
		   Datum *values,
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

	/* Find the node by ItemPointer */
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

	/*
	 * For each level where this node exists, remove it from neighbor
	 * connections
	 */
	for (level = 0; level <= node->level; level++)
	{
		neighbors = HnswGetNeighbors(node, level);
		neighborCount = node->neighborCount[level];

		/* Remove this node from each neighbor's neighbor list */
		for (i = 0; i < neighborCount; i++)
		{
			if (neighbors[i] != InvalidBlockNumber)
			{
				hnswRemoveNodeFromNeighbor(index, neighbors[i], nodeBlkno, level);
			}
		}
	}

	/* Update entry point if this node was the entry point */
	if (meta->entryPoint == nodeBlkno)
	{
		bool		foundNewEntry = false;
		int			bestLevel = -1;
		BlockNumber bestEntry = InvalidBlockNumber;

		/* Find the highest level neighbor to use as new entry point */
		for (level = node->level; level >= 0; level--)
		{
			neighbors = HnswGetNeighbors(node, level);
			neighborCount = node->neighborCount[level];
			for (i = 0; i < neighborCount; i++)
			{
				if (neighbors[i] != InvalidBlockNumber)
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
							if (neighborNode->level > bestLevel)
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

	/* Update metadata */
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
		   Datum *values,
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
