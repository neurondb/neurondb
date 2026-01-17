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
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/index/hnsw_am.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_types.h"
#include "neurondb_replication.h"
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
#define HNSW_VERSION			2	/* Version bumped for HOT update support */
#define HNSW_HEAPTIDS			10	/* Maximum heap TIDs per node for HOT update support */

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
	int32		vl_len_;		/* varlena header (do not touch directly!) */
	int			m;				/* number of connections */
	int			efConstruction; /* size of dynamic candidate list */
	int			ef_search;		/* size of dynamic candidate list for search */
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
	ItemPointerData heaptids[HNSW_HEAPTIDS];	/* Multiple heap TIDs for HOT update support */
	uint8		heaptidsLength;					/* Number of valid heap TIDs */
	uint8		unused1;						/* Padding for alignment */
	uint16		unused2;						/* Padding for alignment */
	int			level;
	int16		dim;
	int16		neighborCount[HNSW_MAX_LEVEL];

	/*
	 * Followed by: float4 vector[dim]; BlockNumber neighbors[level+1][M*2];
	 */
}			HnswNodeData;

typedef HnswNodeData * HnswNode;

/*
 * Maximum visited array size to prevent excessive memory allocation.
 * When this limit is reached during search, extra neighbors are dropped
 * rather than expanding further. This is acceptable for correctness but
 * may reduce recall for very large ef_search values.
 */
#define HNSW_MAX_VISITED_CAPACITY (1024 * 1024)  /* 1M entries max */

/*
 * HnswGetVector - Get vector data pointer from node
 * Vector area is MAXALIGNED to ensure proper neighbor array alignment
 */
#define HnswGetVector(node) \
	((float4 *)((char *)(node) + MAXALIGN(sizeof(HnswNodeData))))

/*
 * HnswGetVectorSize - Get aligned size of vector area
 */
#define HnswGetVectorSize(dim) \
	MAXALIGN((dim) * sizeof(float4))

/*
 * HnswGetNeighborsWithM - Get neighbors array pointer for a specific level.
 *
 * Node layout on disk is determined by the m value stored in the meta page
 * when the node was created. All nodes in an index must use the same m value.
 * Vector area is MAXALIGNED before neighbors start.
 */
#define HnswGetNeighborsWithM(node, lev, m) \
	((BlockNumber *)((char *)(node) + MAXALIGN(sizeof(HnswNodeData)) \
		+ HnswGetVectorSize((node)->dim) \
		+ (lev) * (m) * 2 * sizeof(BlockNumber)))

/*
 * HnswGetNeighborsSafe - Get neighbors array using m from meta page.
 */
static inline BlockNumber *
HnswGetNeighborsSafe(HnswNode node, int level, int m)
{
	/* Validate inputs before pointer arithmetic */
	if (node == NULL)
	{
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("hnsw: HnswGetNeighborsSafe called with NULL node")));
		return NULL;
	}
	
	if (level < 0 || level >= HNSW_MAX_LEVEL)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("hnsw: HnswGetNeighborsSafe called with invalid level %d (max: %d)",
						level, HNSW_MAX_LEVEL - 1)));
		return NULL;
	}
	
	if (m <= 0 || m > HNSW_MAX_M)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("hnsw: HnswGetNeighborsSafe called with invalid m %d (max: %d)",
						m, HNSW_MAX_M)));
		return NULL;
	}
	
	/* Validate dimension to prevent overflow in size calculation */
	if (node->dim <= 0 || node->dim > 32767)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("hnsw: HnswGetNeighborsSafe called with invalid dim %d",
						node->dim)));
		return NULL;
	}
	
	/* Check for integer overflow in offset calculation */
	if ((size_t) level > SIZE_MAX / (size_t) m / 2 / sizeof(BlockNumber))
	{
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("hnsw: HnswGetNeighborsSafe offset calculation would overflow")));
		return NULL;
	}
	
	return (BlockNumber *)((char *)node + MAXALIGN(sizeof(HnswNodeData)) +
						   HnswGetVectorSize(node->dim) +
						   level * m * 2 * sizeof(BlockNumber));
}

/*
 * HnswNodeSizeWithM - Calculate node size with specific m value.
 *
 * Node size depends on the m parameter. All nodes in an index must use
 * the same m value that matches meta->m.
 */
static inline Size __attribute__((used))
HnswNodeSizeWithM(int dim, int level, int m)
{
	Size headerSize = MAXALIGN(sizeof(HnswNodeData));
	Size vectorSize = HnswGetVectorSize(dim);
	Size neighborSize = (level + 1) * m * 2 * sizeof(BlockNumber);
	
	/* Total size without additional outer MAXALIGN - components are already aligned */
	return headerSize + vectorSize + neighborSize;
}

/* Legacy macros - use HnswNodeSizeWithM and HnswGetNeighborsSafe instead */
#define HnswNodeSize(dim, level) \
	HnswNodeSizeWithM(dim, level, HNSW_DEFAULT_M)

#define HnswGetNeighbors(node, lev) \
	HnswGetNeighborsWithM(node, lev, HNSW_DEFAULT_M)

/*
 * Build state for index build
 */
/*
 * In-memory graph element for fast building
 * Similar to pgvector's HnswElement but optimized for NeuronDB
 */
typedef struct HnswInMemoryElement
{
	struct HnswInMemoryElement *next;	/* Linked list of all elements */
	ItemPointerData heaptids[HNSW_HEAPTIDS];
	uint8		heaptidsLength;
	int			level;
	int16		dim;
	float4	   *vector;					/* Vector data (allocated separately) */
	BlockNumber *neighbors[HNSW_MAX_LEVEL];	/* Neighbors at each level */
	int16		neighborCount[HNSW_MAX_LEVEL];
	float4	   *neighborDistances[HNSW_MAX_LEVEL];	/* Distances to neighbors */
	BlockNumber blkno;					/* Set during flush */
	OffsetNumber offno;					/* Set during flush */
	BlockNumber neighborPage;			/* Set during flush */
	OffsetNumber neighborOffno;			/* Set during flush */
}			HnswInMemoryElement;

/*
 * In-memory graph structure
 */
typedef struct HnswInMemoryGraph
{
	HnswInMemoryElement *head;			/* Linked list head */
	HnswInMemoryElement *entryPoint;	/* Entry point for searches */
	int			entryLevel;				/* Level of entry point */
	Size		memoryUsed;				/* Memory used so far */
	Size		memoryTotal;			/* Total memory available */
	bool		flushed;				/* True if graph has been flushed to disk */
	slock_t		lock;					/* Spinlock for graph modifications */
}			HnswInMemoryGraph;

typedef struct HnswBuildState
{
	Relation	heap;
	Relation	index;
	IndexInfo  *indexInfo;
	double		indtuples;
	MemoryContext tmpCtx;
	MemoryContext graphCtx;				/* Context for in-memory graph */
	HnswInMemoryGraph *graph;			/* In-memory graph (NULL if using on-disk) */
	int			m;
	int			efConstruction;
	int			dim;
}			HnswBuildState;

/*
 * Opaque for scan state
 */
typedef struct HnswScanOpaqueData
{
	int			efSearch;
	int			strategy;
	float4	   *query;			/* Query vector as plain float4 array (no varlena) */
	int			queryDim;		/* Query vector dimensionality */
	int			k;
	bool		firstCall;
	int			resultCount;
	BlockNumber *results;
	float4	   *distances;
	int			currentResult;
	MemoryContext scanCtx;		/* Dedicated context for scan allocations */
	double		maxMemory;		/* Max memory for iterative scans (bytes) */
	/* Iterative scan support */
	bool		iterativeScanEnabled;
	int			iterativeScanMode;	/* 0=off, 1=strict_order, 2=relaxed_order */
	int			maxScanTuples;
	int			scannedTuples;		/* Tuples scanned so far */
	bool		needsMoreResults;	/* True if we need to scan more */
	Relation	heapRel;			/* Heap relation for filtering */
	ExprState  *qualExpr;			/* Qual expression for filtering */
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
static inline float4 hnswComputeDistanceSquaredL2(const float4 *vec1, const float4 *vec2, int dim);
static void hnswSearch(Relation index, HnswMetaPage metaPage, const float4 * query,
					   int dim, int strategy, int efSearch, int k,
					   BlockNumber * *results, float4 * *distances, int *resultCount);
static void hnswInsertNode(Relation index, HnswMetaPage metaPage,
						   const float4 * vector, int dim, ItemPointer heapPtr);
static float4 * hnswExtractVectorData(Datum value, Oid typeOid, int *out_dim, MemoryContext ctx);
static Oid hnswGetKeyType(Relation index, int attno);
static void hnswBuildCallback(Relation index, ItemPointer tid, Datum * values,
							  bool *isnull, bool tupleIsAlive, void *state);

/* HOT update support functions */
static void hnswAddHeapTid(HnswNode node, ItemPointer heaptid);
static bool hnswNodeHasHeapTid(HnswNode node, ItemPointer heaptid);
static bool hnswIsNodeCompatible(HnswNode node, uint32 version);

/* Safety validation helpers */
static int16 hnswValidateNeighborCount(int16 neighborCount, int m, int level);
static bool hnswValidateLevelSafe(int level);  /* Returns false instead of ERROR */
static bool hnswValidateBlockNumber(BlockNumber blkno, Relation index);
static Size hnswComputeNodeSizeSafe(int dim, int level, int m, bool *overflow);
static void hnswCacheTypeOids(void);
static inline Buffer hnswReadBufferChecked(Relation index, BlockNumber blkno, const char *ctx);

/* Neighbor selection helper - for comparing distances during sorting */
typedef struct {
	BlockNumber blk;
	float4 dist;
} NeighborCandidate;

static int hnswCompareNeighborDist(const void *a, const void *b);

/* Convenience wrapper for consistent logging context */
#define HNSW_READBUFFER(index, blkno) hnswReadBufferChecked((index), (blkno), __func__)

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
	IndexAmRoutine *amroutine = NULL;

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
	amroutine->amcanparallel = false;	/* TODO: Parallel build not yet implemented */
	amroutine->amcaninclude = false;
	amroutine->amusemaintenanceworkmem = true;	/* Enable in-memory graph building */
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
/*
 * Initialize in-memory graph
 */
static void
InitInMemoryGraph(HnswInMemoryGraph *graph, Size memoryTotal)
{
	graph->head = NULL;
	graph->entryPoint = NULL;
	graph->entryLevel = -1;
	graph->memoryUsed = 0;
	graph->memoryTotal = memoryTotal;
	graph->flushed = false;
	SpinLockInit(&graph->lock);
}

/*
 * Flush in-memory graph to disk
 */
static void
FlushInMemoryGraph(Relation index, HnswInMemoryGraph *graph, HnswBuildState *buildstate)
{
	HnswInMemoryElement *element;
	Buffer		metaBuffer;
	HnswMetaPage metaPage;
	Page		page;
	BlockNumber entryBlkno = InvalidBlockNumber;
	int			entryLevel = -1;

	if (graph == NULL || graph->flushed)
		return;

	/* Initialize metadata page */
	metaBuffer = HNSW_READBUFFER(index, 0);
	LockBuffer(metaBuffer, BUFFER_LOCK_EXCLUSIVE);
	page = BufferGetPage(metaBuffer);
	metaPage = (HnswMetaPage) PageGetContents(page);

	/* Traverse graph and write elements to disk */
	element = graph->head;
	while (element != NULL)
	{
		HnswInMemoryElement *next = element->next;
		BlockNumber blkno;
		Buffer		buf;
		Page		page;
		HnswNode	node;
		Size		nodeSize;
		int			i, l;
		bool		overflow = false;
		char	   *nodeBuf;

		/* Calculate node size */
		nodeSize = hnswComputeNodeSizeSafe(element->dim, element->level, buildstate->m, &overflow);
		if (overflow || nodeSize == 0)
		{
			UnlockReleaseBuffer(metaBuffer);
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("hnsw: node size calculation overflow during flush")));
		}

		/* Allocate new page for this node */
		buf = HNSW_READBUFFER(index, P_NEW);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
		page = BufferGetPage(buf);
		PageInit(page, BufferGetPageSize(buf), 0);
		blkno = BufferGetBlockNumber(buf);

		/* Allocate node buffer */
		nodeBuf = (char *) palloc(nodeSize);
		node = (HnswNode) nodeBuf;
		MemSet(node, 0, nodeSize);

		/* Copy element data to node */
		node->level = element->level;
		node->dim = element->dim;
		node->heaptidsLength = element->heaptidsLength;
		for (i = 0; i < HNSW_HEAPTIDS; i++)
			node->heaptids[i] = element->heaptids[i];
		for (l = 0; l < HNSW_MAX_LEVEL; l++)
			node->neighborCount[l] = element->neighborCount[l];

		/* Copy vector */
		memcpy(HnswGetVector(node), element->vector, element->dim * sizeof(float4));

		/* Copy neighbors - will be updated in second pass */
		for (l = 0; l <= element->level; l++)
		{
			BlockNumber *nodeNeighbors = HnswGetNeighborsSafe(node, l, buildstate->m);
			/* Initialize to InvalidBlockNumber */
			int			lm = (l == 0 ? buildstate->m * 2 : buildstate->m);
			for (i = 0; i < lm; i++)
				nodeNeighbors[i] = InvalidBlockNumber;
		}

		/* Add node to page */
		if (PageAddItem(page, (Item) node, nodeSize, InvalidOffsetNumber, false, false) == InvalidOffsetNumber)
		{
			pfree(nodeBuf);
			UnlockReleaseBuffer(buf);
			UnlockReleaseBuffer(metaBuffer);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("hnsw: failed to add node to page")));
		}

		pfree(nodeBuf);

		/* Store block numbers for later neighbor linking */
		element->blkno = blkno;
		element->offno = FirstOffsetNumber;

		/* Update entry point if this is the highest level */
		if (element->level > entryLevel)
		{
			entryLevel = element->level;
			entryBlkno = blkno;
		}

		MarkBufferDirty(buf);
		UnlockReleaseBuffer(buf);

		element = next;
	}

	/* Update metadata */
	if (BlockNumberIsValid(entryBlkno))
	{
		metaPage->entryPoint = entryBlkno;
		metaPage->entryLevel = entryLevel;
		metaPage->maxLevel = entryLevel;
		metaPage->insertedVectors = buildstate->indtuples;
	}

	MarkBufferDirty(metaBuffer);
	UnlockReleaseBuffer(metaBuffer);

	/* Second pass: Update neighbors by converting element pointers to block numbers */
	/* Neighbors were found during in-memory insert phase and stored as element pointers */
	element = graph->head;
	while (element != NULL)
	{
		HnswInMemoryElement *next = element->next;
		Buffer		buf;
		Page		page;
		HnswNode	node;
		int			l;

	buf = HNSW_READBUFFER(index, element->blkno);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	page = BufferGetPage(buf);
	
	/* Check if page has valid items before accessing */
	if (PageIsNew(page) || PageIsEmpty(page) ||
		PageGetMaxOffsetNumber(page) < FirstOffsetNumber)
	{
		UnlockReleaseBuffer(buf);
		return;  /* Skip this element */
	}
	
	{
		ItemId elementItemId = PageGetItemId(page, FirstOffsetNumber);
		if (!ItemIdIsValid(elementItemId) || !ItemIdHasStorage(elementItemId))
		{
			UnlockReleaseBuffer(buf);
			return;  /* Skip this element */
		}
		
		node = (HnswNode) PageGetItem(page, elementItemId);
	}
	
	if (node == NULL)
	{
		UnlockReleaseBuffer(buf);
		return;  /* Skip this element */
	}

	/* Update neighbors for each level */
		for (l = 0; l <= element->level; l++)
		{
			BlockNumber *nodeNeighbors = HnswGetNeighborsSafe(node, l, buildstate->m);
			int			lm = (l == 0 ? buildstate->m * 2 : buildstate->m);
			int			count = element->neighborCount[l];

			if (count > 0 && element->neighbors[l] != NULL)
			{
				/* Convert element pointers (stored as BlockNumber) to actual block numbers */
				for (int i = 0; i < count && i < lm; i++)
				{
					/* Element pointer was stored as (BlockNumber)(uintptr_t)element */
					HnswInMemoryElement *neighborElem = 
						(HnswInMemoryElement *) (uintptr_t) element->neighbors[l][i];
					
					/* Validate neighbor element exists and has a block number */
					if (neighborElem != NULL && BlockNumberIsValid(neighborElem->blkno))
					{
						nodeNeighbors[i] = neighborElem->blkno;
					}
					else
					{
						/* Invalid neighbor - shouldn't happen, but handle gracefully */
						nodeNeighbors[i] = InvalidBlockNumber;
						count = i;	/* Adjust count */
						break;
					}
				}
				node->neighborCount[l] = count;
			}
			else
			{
				/* No neighbors found during insert - this can happen for first element */
				node->neighborCount[l] = 0;
			}
		}

		MarkBufferDirty(buf);
		UnlockReleaseBuffer(buf);

		element = next;
	}

	graph->flushed = true;
}

/*
 * Find neighbors in in-memory graph using efficient greedy search
 * 
 * This implements the HNSW insertion algorithm: for each level l where
 * the element exists, find efConstruction candidates via greedy search
 * starting from entry point, then select top m neighbors.
 */
static void
FindNeighborsInMemory(HnswInMemoryElement *element, HnswInMemoryElement *entryPoint,
					  HnswInMemoryGraph *graph, int m, int efConstruction, int dim, int strategy)
{
	int			level = element->level;
	int			l;
	MemoryContext oldCtx;

	if (entryPoint == NULL || graph->head == element)
	{
		/* First element or no entry point - no neighbors yet */
		return;
	}

	/* Use graph context for allocations */
	oldCtx = MemoryContextSwitchTo(graph->head->vector ? 
								   CurrentMemoryContext : CurrentMemoryContext);

	/* Process each level from top to bottom */
	for (l = level; l >= 0; l--)
	{
		int			lm = (l == 0 ? m * 2 : m);
		int			maxCandidates = Min(efConstruction, 200);
		HnswInMemoryElement *current;
		HnswInMemoryElement **candidates;
		float4	   *candidateDists;
		int			candidateCount = 0;
		bool	   *visited;
		int			i, j;

		/* Allocate candidate arrays */
		candidates = (HnswInMemoryElement **) palloc(maxCandidates * sizeof(HnswInMemoryElement *));
		candidateDists = (float4 *) palloc(maxCandidates * sizeof(float4));
		visited = (bool *) palloc0(maxCandidates * sizeof(bool));

		/* Find starting point at level l or higher */
		current = entryPoint;
		while (current != NULL && current != element && current->level < l)
		{
			/* Find closest element at level l or higher by scanning graph */
			HnswInMemoryElement *best = NULL;
			float4		bestDist = FLT_MAX;
			HnswInMemoryElement *scan = graph->head;

			while (scan != NULL && scan != element)
			{
				if (scan->level >= l)
				{
					/* Use squared distance for L2 comparisons */
					float4		dist = (strategy == 1) ?
									  hnswComputeDistanceSquaredL2(element->vector, scan->vector, dim) :
									  hnswComputeDistance(element->vector, scan->vector, dim, strategy);
					if (dist < bestDist)
					{
						bestDist = dist;
						best = scan;
					}
				}
				scan = scan->next;
			}
			if (best != NULL)
				current = best;
			else
				break;
		}

		if (current == NULL || current == element)
		{
			pfree(candidates);
			pfree(candidateDists);
			pfree(visited);
			continue;
		}

		/* Greedy search for efConstruction candidates */
		for (i = 0; i < maxCandidates && current != NULL && current != element; i++)
		{
			/* Use squared distance for L2 comparisons to avoid sqrt overhead */
			float4		currentDist = (strategy == 1) ? 
									  hnswComputeDistanceSquaredL2(element->vector, current->vector, dim) :
									  hnswComputeDistance(element->vector, current->vector, dim, strategy);
			bool		isVisited = false;
			HnswInMemoryElement *closer;
			float4		closerDist;

			/* Check if already visited */
			for (j = 0; j < candidateCount; j++)
			{
				if (candidates[j] == current)
				{
					isVisited = true;
					break;
				}
			}

			if (!isVisited)
			{
				/* Add to candidates or replace worst */
				if (candidateCount < maxCandidates)
				{
					candidates[candidateCount] = current;
					candidateDists[candidateCount] = currentDist;
					candidateCount++;
				}
				else
				{
					/* Find worst candidate */
					int			worstIdx = 0;
					for (j = 1; j < candidateCount; j++)
					{
						if (candidateDists[j] > candidateDists[worstIdx])
							worstIdx = j;
					}
					if (currentDist < candidateDists[worstIdx])
					{
						candidates[worstIdx] = current;
						candidateDists[worstIdx] = currentDist;
					}
				}
			}

			/* Find closer unvisited neighbor at this level */
			closer = NULL;
			closerDist = currentDist;

			/* Check neighbors of current element at this level */
			if (current->neighborCount[l] > 0 && current->neighbors[l] != NULL)
			{
				/* Neighbors are stored as BlockNumber pointers (temporary) during in-memory phase */
				/* For now, scan all elements at level l to find closest */
				HnswInMemoryElement *scan = graph->head;
				while (scan != NULL && scan != element)
				{
					if (scan->level >= l && scan != current)
					{
						bool		alreadyCandidate = false;
						for (j = 0; j < candidateCount; j++)
						{
							if (candidates[j] == scan)
							{
								alreadyCandidate = true;
								break;
							}
						}
						if (!alreadyCandidate)
						{
							/* Use squared distance for L2 comparisons */
							float4		scanDist = (strategy == 1) ?
												  hnswComputeDistanceSquaredL2(element->vector, scan->vector, dim) :
												  hnswComputeDistance(element->vector, scan->vector, dim, strategy);
							if (scanDist < closerDist)
							{
								closerDist = scanDist;
								closer = scan;
							}
						}
					}
					scan = scan->next;
				}
			}
			else
			{
				/* No neighbors yet, scan all elements at this level */
				HnswInMemoryElement *scan = graph->head;
				while (scan != NULL && scan != element)
				{
					if (scan->level >= l && scan != current)
					{
						bool		alreadyCandidate = false;
						for (j = 0; j < candidateCount; j++)
						{
							if (candidates[j] == scan)
							{
								alreadyCandidate = true;
								break;
							}
						}
						if (!alreadyCandidate)
						{
							/* Use squared distance for L2 comparisons */
							float4		scanDist = (strategy == 1) ?
												  hnswComputeDistanceSquaredL2(element->vector, scan->vector, dim) :
												  hnswComputeDistance(element->vector, scan->vector, dim, strategy);
							if (scanDist < closerDist)
							{
								closerDist = scanDist;
								closer = scan;
							}
						}
					}
					scan = scan->next;
				}
			}

			if (closer != NULL && closerDist < currentDist)
				current = closer;
			else
				break;		/* No closer neighbor found, done */
		}

		/* Select top lm neighbors from candidates using sorting */
		if (candidateCount > 0)
		{
			/* Create temporary array for sorting */
			typedef struct {
				HnswInMemoryElement *elem;
				float4 dist;
			} InMemoryNeighborCandidate;
			InMemoryNeighborCandidate *sortedCandidates = 
				(InMemoryNeighborCandidate *) palloc(candidateCount * sizeof(InMemoryNeighborCandidate));
			int			selectedCount;

			for (i = 0; i < candidateCount; i++)
			{
				sortedCandidates[i].elem = candidates[i];
				sortedCandidates[i].dist = candidateDists[i];
			}

			/* Sort by distance using selection sort (optimized for small arrays) */
			/* For inner product (strategy 3), larger distance is better (descending) */
			/* For L2/cosine, smaller distance is better (ascending) */
			for (i = 0; i < candidateCount - 1; i++)
			{
				int			bestIdx = i;
				for (j = i + 1; j < candidateCount; j++)
				{
					if (strategy == 3)
					{
						/* Inner product: larger is better */
						if (sortedCandidates[j].dist > sortedCandidates[bestIdx].dist)
							bestIdx = j;
					}
					else
					{
						/* L2/cosine: smaller is better */
						if (sortedCandidates[j].dist < sortedCandidates[bestIdx].dist)
							bestIdx = j;
					}
				}
				if (bestIdx != i)
				{
					InMemoryNeighborCandidate temp = sortedCandidates[i];
					sortedCandidates[i] = sortedCandidates[bestIdx];
					sortedCandidates[bestIdx] = temp;
				}
			}

			/* Select top lm */
			selectedCount = Min(lm, candidateCount);

			element->neighbors[l] = (BlockNumber *) palloc0(lm * sizeof(BlockNumber));
			element->neighborDistances[l] = (float4 *) palloc0(lm * sizeof(float4));
			element->neighborCount[l] = selectedCount;

			for (i = 0; i < selectedCount; i++)
			{
				/* Store element pointer temporarily as BlockNumber (will be converted during flush) */
				element->neighbors[l][i] = (BlockNumber) (uintptr_t) sortedCandidates[i].elem;
				element->neighborDistances[l][i] = sortedCandidates[i].dist;
			}

			pfree(sortedCandidates);
		}

		pfree(candidates);
		pfree(candidateDists);
		pfree(visited);
	}

	MemoryContextSwitchTo(oldCtx);
}

/*
 * Insert tuple into in-memory graph
 */
static bool
InsertTupleInMemory(Relation index, Datum *values, bool *isnull, ItemPointer tid,
					HnswBuildState *buildstate)
{
	HnswInMemoryGraph *graph = buildstate->graph;
	HnswInMemoryElement *element;
	float4	   *vector;
	int			dim;
	int			level;
	Size		elementSize;
	MemoryContext oldCtx;
	Oid			keyType;

	if (graph == NULL || graph->flushed)
		return false;

	/* Extract vector */
	if (isnull[0])
		return false;

	keyType = hnswGetKeyType(index, 1);
	vector = hnswExtractVectorData(values[0], keyType, &dim, buildstate->tmpCtx);
	if (vector == NULL)
		return false;

	if (buildstate->dim == 0)
		buildstate->dim = dim;

	/* Calculate level */
	level = hnswGetRandomLevel(HNSW_DEFAULT_ML);
	if (level >= HNSW_MAX_LEVEL)
		level = HNSW_MAX_LEVEL - 1;

	/* Estimate element size */
	elementSize = sizeof(HnswInMemoryElement) + 
				  dim * sizeof(float4) + 
				  (level + 1) * buildstate->m * 2 * sizeof(BlockNumber) +
				  (level + 1) * buildstate->m * 2 * sizeof(float4);

	/* Check memory limit */
	SpinLockAcquire(&graph->lock);
	if (graph->memoryUsed + elementSize > graph->memoryTotal)
	{
		SpinLockRelease(&graph->lock);
		return false;	/* Out of memory, fall back to on-disk */
	}
	graph->memoryUsed += elementSize;
	SpinLockRelease(&graph->lock);

	/* Allocate element in graph context */
	oldCtx = MemoryContextSwitchTo(buildstate->graphCtx);
	element = (HnswInMemoryElement *) palloc0(sizeof(HnswInMemoryElement));
	element->vector = (float4 *) palloc(dim * sizeof(float4));
	memcpy(element->vector, vector, dim * sizeof(float4));
	element->dim = dim;
	element->level = level;
	element->heaptidsLength = 1;
	element->heaptids[0] = *tid;

	/* Initialize neighbor arrays */
	for (int l = 0; l <= level; l++)
	{
		element->neighbors[l] = NULL;
		element->neighborDistances[l] = NULL;
		element->neighborCount[l] = 0;
	}

	/* Add to graph FIRST so it's available for neighbor finding */
	{
		HnswInMemoryElement *oldEntryPoint;
		
		SpinLockAcquire(&graph->lock);
		element->next = graph->head;
		oldEntryPoint = graph->entryPoint;

		/* Update entry point if needed */
		if (graph->entryPoint == NULL || element->level > graph->entryLevel)
		{
			graph->entryPoint = element;
			graph->entryLevel = element->level;
		}
		graph->head = element;
		SpinLockRelease(&graph->lock);

		/* Find neighbors using in-memory search - use old entry point as starting point */
		/* Use strategy 1 (L2) - should match index operator class */
		{
			int			strategy = 1;	/* Default to L2, could be extracted from index if needed */
			
			FindNeighborsInMemory(element, oldEntryPoint ? oldEntryPoint : element, graph, 
								 buildstate->m, buildstate->efConstruction, dim, strategy);
		}
	}

	MemoryContextSwitchTo(oldCtx);
	return true;
}

static IndexBuildResult *
hnswbuild(Relation heap, Relation index, IndexInfo * indexInfo)
{
	HnswBuildState buildstate = {0};
	Buffer		metaBuffer;
	HnswOptions *options = NULL;
	IndexBuildResult *result = NULL;
	int			m,
				ef_construction,
				ef_search;
	Size		memoryTotal;
	HnswInMemoryGraph *graph = NULL;


	buildstate.heap = heap;
	buildstate.index = index;
	buildstate.indexInfo = indexInfo;
	buildstate.indtuples = 0;
	buildstate.tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
											  "HNSW build temporary context",
											  ALLOCSET_DEFAULT_SIZES);

	/* Initialize metadata page on block 0 */
	metaBuffer = HNSW_READBUFFER(index, P_NEW);
	LockBuffer(metaBuffer, BUFFER_LOCK_EXCLUSIVE);

	options = (HnswOptions *) indexInfo->ii_AmCache;
	if (options == NULL)
	{
		HnswOptions opts;

		/* Load options using proper function that handles defaults */
		hnswLoadOptions(index, &opts);
		
		elog(DEBUG1, "[HNSW_BUILD_OPTIONS] Loaded: m=%d efConstruction=%d ef_search=%d",
			 opts.m, opts.efConstruction, opts.ef_search);
		
		/* Allocate in CurrentMemoryContext for proper lifecycle during index build
		 * Note: ii_AmCache is NOT a varlena type - it's a plain pointer cache.
		 * Do NOT use SET_VARSIZE here. The varlena header is only for bytea storage
		 * in rd_options, not for the cache pointer. */
		options = (HnswOptions *) palloc0(sizeof(HnswOptions));
		*options = opts;
		indexInfo->ii_AmCache = (void *) options;
	}
	m = options->m;
	ef_construction = options->efConstruction;
	ef_search = options->ef_search;

	buildstate.m = m;
	buildstate.efConstruction = ef_construction;
	buildstate.dim = 0;

	hnswInitMetaPage(metaBuffer, m, ef_construction, ef_search, HNSW_DEFAULT_ML);

	MarkBufferDirty(metaBuffer);
	UnlockReleaseBuffer(metaBuffer);

	/* Try to use in-memory graph building */
	memoryTotal = (Size) maintenance_work_mem * 1024L;
	if (memoryTotal > 0)
	{
		/* Create graph context and initialize in-memory graph */
		buildstate.graphCtx = AllocSetContextCreate(CurrentMemoryContext,
													"HNSW build graph context",
													ALLOCSET_DEFAULT_SIZES);
		graph = (HnswInMemoryGraph *) MemoryContextAlloc(buildstate.graphCtx, sizeof(HnswInMemoryGraph));
		InitInMemoryGraph(graph, memoryTotal);
		buildstate.graph = graph;
	}

	/* Use parallel scan if available */
	buildstate.indtuples = table_index_build_scan(heap, index, indexInfo,
												  true, true, hnswBuildCallback,
												  (void *) &buildstate, NULL);

	/* Flush in-memory graph to disk if we built one */
	if (graph != NULL && !graph->flushed)
	{
		FlushInMemoryGraph(index, graph, &buildstate);
	}

	{
		nalloc(result, IndexBuildResult, 1);
		result->heap_tuples = buildstate.indtuples;
		result->index_tuples = buildstate.indtuples;

		if (buildstate.graphCtx != NULL)
			MemoryContextDelete(buildstate.graphCtx);
		MemoryContextDelete(buildstate.tmpCtx);

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
	bool		inserted = false;

	/* Try in-memory insert first if graph is available */
	if (buildstate->graph != NULL && !buildstate->graph->flushed)
	{
		inserted = InsertTupleInMemory(index, values, isnull, tid, buildstate);
	}

	/* Fall back to on-disk insert if in-memory failed */
	if (!inserted)
	{
		/* If we ran out of memory, flush and switch to on-disk */
		if (buildstate->graph != NULL && !buildstate->graph->flushed)
		{
			FlushInMemoryGraph(index, buildstate->graph, buildstate);
		}

		hnswinsert(index, values, isnull, tid, buildstate->heap,
				   UNIQUE_CHECK_NO, true, buildstate->indexInfo);
	}

	buildstate->indtuples++;
}

static void
hnswbuildempty(Relation index)
{
	Buffer		metaBuffer;
	HnswOptions opts;

	/* Load options from relation to match CREATE INDEX reloptions */
	hnswLoadOptions(index, &opts);

	/* Initialize metadata page on block 0 */
	metaBuffer = HNSW_READBUFFER(index, P_NEW);
	LockBuffer(metaBuffer, BUFFER_LOCK_EXCLUSIVE);
	(void) BufferGetPage(metaBuffer);  /* Ensure page is valid */

	hnswInitMetaPage(metaBuffer,
					 opts.m,
					 opts.efConstruction,
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
	float4	   *vectorData = NULL;
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

	/* Validate heap TID before inserting - prevent storing invalid TIDs that could cause VACUUM issues */
	if (!ItemPointerIsValid(ht_ctid))
	{
		elog(WARNING, "hnsw: invalid heap TID provided, skipping insert");
		pfree(vectorData);
		return false;
	}

	metaBuffer = HNSW_READBUFFER(index, 0);
	LockBuffer(metaBuffer, BUFFER_LOCK_EXCLUSIVE);
	metaPage = BufferGetPage(metaBuffer);
	meta = (HnswMetaPage) PageGetContents(metaPage);

	hnswInsertNode(index, meta, vectorData, dim, ht_ctid);

	MarkBufferDirty(metaBuffer);
	UnlockReleaseBuffer(metaBuffer);

	/* Call replication hook if replication is enabled */
	/* Note: Block number tracking would be added in full implementation */
	if (neurondb_replication_enabled())
	{
		/* In full implementation, track the inserted block number */
		neurondb_hnsw_replication_hook(index, InvalidBlockNumber, true);
	}

	pfree(vectorData);

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
	BlockNumber *neighbors = NULL;
	int16		neighborCount;
	int			level;
	int			i;
	bool		foundNewEntry;
	ItemId		itemId;

	IndexBulkDeleteResult *new_stats = NULL;

	if (stats == NULL)
	{
		nalloc(new_stats, IndexBulkDeleteResult, 1);
		stats = new_stats;
	}

	/* Read metadata page */
	metaBuffer = HNSW_READBUFFER(index, 0);
	LockBuffer(metaBuffer, BUFFER_LOCK_EXCLUSIVE);
	metaPage = BufferGetPage(metaBuffer);
	meta = (HnswMetaPage) PageGetContents(metaPage);

	/* Scan all pages in the index */
	for (blkno = 1; blkno < RelationGetNumberOfBlocks(index); blkno++)
	{
		nodeBuf = HNSW_READBUFFER(index, blkno);
		LockBuffer(nodeBuf, BUFFER_LOCK_EXCLUSIVE);
		nodePage = BufferGetPage(nodeBuf);

		if (PageIsNew(nodePage) || PageIsEmpty(nodePage))
		{
			UnlockReleaseBuffer(nodeBuf);
			continue;
		}

		maxoff = PageGetMaxOffsetNumber(nodePage);
		
		/* Enforce one-node-per-page invariant: HNSW index uses ONE NODE PER PAGE */
		if (maxoff != FirstOffsetNumber)
		{
			elog(WARNING, "hnsw: page %u has %d items, expected 1 (one-node-per-page invariant violated), skipping",
				 blkno, maxoff);
			UnlockReleaseBuffer(nodeBuf);
			continue;
		}
		
		offnum = FirstOffsetNumber;
		itemId = PageGetItemId(nodePage, offnum);
		
		if (!ItemIdIsValid(itemId) || ItemIdIsDead(itemId))
		{
			UnlockReleaseBuffer(nodeBuf);
			continue;
		}

		if (!ItemIdHasStorage(itemId))
		{
			UnlockReleaseBuffer(nodeBuf);
			continue;
		}

		node = (HnswNode) PageGetItem(nodePage, itemId);
		if (node == NULL)
		{
			UnlockReleaseBuffer(nodeBuf);
			continue;
		}

		/* Validate node level */
		if (!hnswValidateLevelSafe(node->level))
		{
			elog(WARNING, "hnsw: invalid node level %d in bulk delete at block %u, skipping",
				 node->level, blkno);
			UnlockReleaseBuffer(nodeBuf);
			continue;
		}

		/* Validate node dim before using */
		if (node->dim <= 0 || node->dim > 32767)
		{
			elog(WARNING, "hnsw: node at block %u has invalid dim %d in bulk delete, skipping", blkno, node->dim);
			UnlockReleaseBuffer(nodeBuf);
			continue;
		}

		/* Check all heap TIDs for HOT update support */
		{
			int			tidIdx;
			int			aliveCount = 0;
			int			deadCount __attribute__((unused)) = 0;
			ItemPointerData aliveTids[HNSW_HEAPTIDS];
			bool		hasDeadTids = false;
			int			heaptidsLength;
			GenericXLogState *state;
			Page		updatePage;
			HnswNode	updateNode;
			ItemId		updateItemId;
		
		/* Handle backwards compatibility: if heaptidsLength is 0, treat as 1 */
		/* Validate heaptidsLength to prevent out-of-bounds access */
		if (node->heaptidsLength > HNSW_HEAPTIDS)
		{
			elog(WARNING, "hnsw: corrupted heaptidsLength %d (max: %d), clamping to %d",
				 node->heaptidsLength, HNSW_HEAPTIDS, HNSW_HEAPTIDS);
			heaptidsLength = HNSW_HEAPTIDS;
		}
		else if (node->heaptidsLength == 0 && ItemPointerIsValid(&node->heaptids[0]))
			heaptidsLength = 1;
		else
			heaptidsLength = node->heaptidsLength;
		
		/* Check each TID and collect alive ones */
		for (tidIdx = 0; tidIdx < heaptidsLength && tidIdx < HNSW_HEAPTIDS; tidIdx++)
		{
			if (callback(&node->heaptids[tidIdx], callback_state))
			{
				hasDeadTids = true;
				deadCount++;
				stats->tuples_removed++;
			}
			else
			{
				if (aliveCount < HNSW_HEAPTIDS)
				{
					aliveTids[aliveCount] = node->heaptids[tidIdx];
					aliveCount++;
				}
				stats->num_index_tuples++;
			}
		}
		
		/* If all TIDs are dead, delete the entire node */
		if (aliveCount == 0)
		{
			/* Remove node from graph structure */
				/* Copy neighbor block numbers before unlocking nodeBuf to prevent deadlock */
				BlockNumber **neighborBlocksPerLevel = NULL;
				int *neighborCountPerLevel = NULL;
				int			maxLevel = node->level;

				if (maxLevel >= 0)
				{
					nalloc(neighborBlocksPerLevel, BlockNumber *, maxLevel + 1);
					nalloc(neighborCountPerLevel, int, maxLevel + 1);

			for (level = 0; level <= maxLevel; level++)
			{
				BlockNumber *tempNeighbors;
				int16		tempCount;
				
				/* Validate node dim before using */
				if (node->dim <= 0 || node->dim > 32767)
				{
					elog(WARNING, "hnsw: node at block %u has invalid dim %d in bulkdelete, skipping", blkno, node->dim);
					break;  /* Break out of level loop if dim is invalid */
				}
				
				/* Validate level bounds BEFORE array access to prevent out-of-bounds read */
				if (level < 0 || level >= HNSW_MAX_LEVEL)
				{
					elog(WARNING, "hnsw: invalid level %d in bulkdelete, skipping (max: %d)",
						 level, HNSW_MAX_LEVEL - 1);
					continue;
				}
				
				tempNeighbors = HnswGetNeighborsSafe(node, level, meta->m);
				tempCount = node->neighborCount[level];

						/* Validate and clamp neighborCount */
						tempCount = hnswValidateNeighborCount(tempCount, meta->m, level);

						if (tempCount > 0)
						{
							int			validCount = 0;

							nalloc(neighborBlocksPerLevel[level], BlockNumber, tempCount);
							for (i = 0; i < tempCount; i++)
							{
								if (tempNeighbors[i] != InvalidBlockNumber &&
									hnswValidateBlockNumber(tempNeighbors[i], index))
								{
									neighborBlocksPerLevel[level][validCount++] = tempNeighbors[i];
								}
							}
							neighborCountPerLevel[level] = validCount;
						}
						else
						{
							neighborBlocksPerLevel[level] = NULL;
							neighborCountPerLevel[level] = 0;
						}
					}
				}

				/* Unlock nodeBuf before processing neighbors to avoid deadlock */
				UnlockReleaseBuffer(nodeBuf);
				nodeBuf = InvalidBuffer;

				/* Now process each level's neighbors */
				for (level = 0; level <= maxLevel; level++)
				{
					if (neighborCountPerLevel && neighborCountPerLevel[level] > 0)
					{
						for (i = 0; i < neighborCountPerLevel[level]; i++)
						{
							hnswRemoveNodeFromNeighbor(index,
													   neighborBlocksPerLevel[level][i],
													   blkno,
													   level);
						}
					}
				}

				/* Free neighbor block arrays */
				if (neighborBlocksPerLevel)
				{
					for (level = 0; level <= maxLevel; level++)
					{
						if (neighborBlocksPerLevel[level])
							pfree(neighborBlocksPerLevel[level]);
					}
					pfree(neighborBlocksPerLevel);
				}
				if (neighborCountPerLevel)
					pfree(neighborCountPerLevel);

				/* Relock nodeBuf if needed for entry point update */
				nodeBuf = HNSW_READBUFFER(index, blkno);
				LockBuffer(nodeBuf, BUFFER_LOCK_EXCLUSIVE);
				nodePage = BufferGetPage(nodeBuf);

			/* Re-validate page state after unlock/relock */
			if (PageIsNew(nodePage) || PageIsEmpty(nodePage))
			{
				UnlockReleaseBuffer(nodeBuf);
				nodeBuf = InvalidBuffer;
				/* If this was the entry point, mark it invalid */
				if (meta->entryPoint == blkno)
				{
					meta->entryPoint = InvalidBlockNumber;
					meta->entryLevel = -1;
				}
				continue;
			}

			itemId = PageGetItemId(nodePage, FirstOffsetNumber);

			if (!ItemIdIsValid(itemId) || !ItemIdHasStorage(itemId))
			{
				UnlockReleaseBuffer(nodeBuf);
				nodeBuf = InvalidBuffer;
				/* If this was the entry point, mark it invalid */
				if (meta->entryPoint == blkno)
				{
					meta->entryPoint = InvalidBlockNumber;
					meta->entryLevel = -1;
				}
				continue;
			}

			node = (HnswNode) PageGetItem(nodePage, itemId);
			if (node == NULL)
			{
				UnlockReleaseBuffer(nodeBuf);
				nodeBuf = InvalidBuffer;
				/* If this was the entry point, mark it invalid */
				if (meta->entryPoint == blkno)
				{
					meta->entryPoint = InvalidBlockNumber;
					meta->entryLevel = -1;
				}
				continue;
			}

			/* Validate node dim before using */
			if (node->dim <= 0 || node->dim > 32767)
			{
				elog(WARNING, "hnsw: node at block %u has invalid dim %d when finding entry point, skipping", blkno, node->dim);
				UnlockReleaseBuffer(nodeBuf);
				nodeBuf = InvalidBuffer;
				if (meta->entryPoint == blkno)
				{
					meta->entryPoint = InvalidBlockNumber;
					meta->entryLevel = -1;
				}
				continue;
			}

			/* Update entry point if this node was the entry point */
			if (meta->entryPoint == blkno)
			{
				foundNewEntry = false;

				for (level = node->level;
					 level >= 0 && !foundNewEntry;
					 level--)
				{
					/* Validate level bounds BEFORE array access to prevent out-of-bounds read */
					if (level < 0 || level >= HNSW_MAX_LEVEL)
					{
						elog(WARNING, "hnsw: invalid level %d when finding new entry point, skipping (max: %d)",
							 level, HNSW_MAX_LEVEL - 1);
						continue;
					}
					
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
							Buffer		tmpBuf;
							Page		tmpPage;
							HnswNode	tmpNode;
							ItemId		tmpItemId;

							tmpBuf = HNSW_READBUFFER(index, neighbors[i]);
							LockBuffer(tmpBuf, BUFFER_LOCK_SHARE);
							tmpPage = BufferGetPage(tmpBuf);

							/* Skip if page is empty or node is NULL */
							if (PageIsNew(tmpPage) || PageIsEmpty(tmpPage))
							{
								UnlockReleaseBuffer(tmpBuf);
								continue;
							}

							tmpItemId = PageGetItemId(tmpPage, FirstOffsetNumber);
							if (!ItemIdIsValid(tmpItemId))
							{
								UnlockReleaseBuffer(tmpBuf);
								continue;
							}

							tmpNode = (HnswNode) PageGetItem(tmpPage, tmpItemId);
							if (tmpNode != NULL && hnswValidateLevelSafe(tmpNode->level))
							{
								meta->entryPoint = neighbors[i];
								meta->entryLevel = tmpNode->level;
								foundNewEntry = true;
							}
							UnlockReleaseBuffer(tmpBuf);
						}
					}
				}

				/* If no neighbor found, mark entry as invalid */
				if (!foundNewEntry)
				{
					elog(WARNING, "hnsw: no valid neighbor found to replace entry point "
								  "at block %u, marking entry point invalid", blkno);
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

			UnlockReleaseBuffer(nodeBuf);
		}
		else if (hasDeadTids && aliveCount > 0)
		{
			/* Some TIDs are dead but not all - update node in place using GenericXLog */
			/* Use GenericXLog for efficient in-place update */
			state = GenericXLogStart(index);
			updatePage = GenericXLogRegisterBuffer(state, nodeBuf, 0);
			updateItemId = PageGetItemId(updatePage, FirstOffsetNumber);
			updateNode = (HnswNode) PageGetItem(updatePage, updateItemId);
			
			/* Update heap TIDs array: move alive TIDs to front */
			updateNode->heaptidsLength = aliveCount;
			for (tidIdx = 0; tidIdx < aliveCount; tidIdx++)
			{
				updateNode->heaptids[tidIdx] = aliveTids[tidIdx];
			}
			
			/* Mark remaining slots as invalid */
			for (tidIdx = aliveCount; tidIdx < HNSW_HEAPTIDS; tidIdx++)
			{
				ItemPointerSetInvalid(&updateNode->heaptids[tidIdx]);
			}
			
			GenericXLogFinish(state);
			MarkBufferDirty(nodeBuf);
			UnlockReleaseBuffer(nodeBuf);
		}
		}  /* End of HOT update support block */
	}  /* End of for loop */

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
	IndexBulkDeleteResult *new_stats = NULL;

	if (stats == NULL)
	{
		nalloc(new_stats, IndexBulkDeleteResult, 1);
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
		{"ef_construction", RELOPT_TYPE_INT, offsetof(HnswOptions, efConstruction)},
		{"ef_search", RELOPT_TYPE_INT, offsetof(HnswOptions, ef_search)}
	};

	/* relopt_kind_hnsw must be initialized in worker_init.c before use */
	if (relopt_kind_hnsw == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("relopt_kind_hnsw not initialized")));
	
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
	if (isnull != NULL)
		*isnull = true;
	return false;
}

static IndexScanDesc
hnswbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	HnswScanOpaque so = NULL;
	MemoryContext oldctx;

	scan = RelationGetIndexScan(index, nkeys, norderbys);
	
	/* Allocate scan opaque in index relation's context */
	oldctx = MemoryContextSwitchTo(scan->indexRelation->rd_indexcxt);
	nalloc(so, HnswScanOpaqueData, 1);
	
	/* Create dedicated memory context for scan operations */
	/* Use smaller initial allocation to allow more scans before hitting work_mem */
	/* Use a lower max allocation size than default to allow scanning more
	 * tuples for iterative search before exceeding work_mem
	 */
	so->scanCtx = AllocSetContextCreate(CurrentMemoryContext,
										"HNSW scan temporary context",
										0, 8 * 1024, 256 * 1024);
	
	/* Calculate max memory for iterative scans */
	{
		extern double hnsw_scan_mem_multiplier;
		double		maxMemory;
		
		/* Add 256 extra bytes to fill last block when close */
		maxMemory = (double) work_mem * hnsw_scan_mem_multiplier * 1024.0 + 256;
		so->maxMemory = Min(maxMemory, (double) SIZE_MAX);
	}
	
	so->efSearch = HNSW_DEFAULT_EF_SEARCH;
	so->strategy = 1;
	so->firstCall = true;
	so->k = 0;
	so->query = NULL;
	so->queryDim = 0;
	so->results = NULL;
	so->distances = NULL;
	so->resultCount = 0;
	so->currentResult = 0;
	/* Initialize iterative scan fields */
	so->iterativeScanEnabled = false;
	so->iterativeScanMode = 0;
	so->maxScanTuples = 20000;
	so->scannedTuples = 0;
	so->needsMoreResults = false;
	so->heapRel = NULL;
	so->qualExpr = NULL;

	MemoryContextSwitchTo(oldctx);
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
	extern int	neurondb_hnsw_k;
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	MemoryContext oldctx;
	MemoryContext indexCxt = scan->indexRelation->rd_indexcxt;

	/* Reset scan context to free previous search allocations */
	if (so->scanCtx != NULL)
		MemoryContextReset(so->scanCtx);
	
	so->results = NULL;
	so->distances = NULL;
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

	/* Initialize iterative scan settings from GUC */
	{
		extern int	hnsw_iterative_scan;
		extern int	hnsw_max_scan_tuples;

		if (hnsw_iterative_scan > 0)
		{
			so->iterativeScanMode = hnsw_iterative_scan;
			so->iterativeScanEnabled = true;
			so->maxScanTuples = hnsw_max_scan_tuples;
		}
		else
		{
			so->iterativeScanMode = 0;
			so->iterativeScanEnabled = false;
			so->maxScanTuples = 20000;	/* Default */
		}
		so->scannedTuples = 0;
		so->needsMoreResults = false;
	}

		if (so->efSearch > 100000)
		{
			elog(WARNING, "hnsw: ef_search %d exceeds maximum, clamping to 100000", so->efSearch);
			so->efSearch = 100000;
		}

	if (norderbys > 0 && orderbys[0].sk_argument != 0)
	{
		float4 *vectorData = NULL;
		int			dim;
		Oid			queryType;

		queryType = TupleDescAttr(scan->indexRelation->rd_att, 0)->atttypid;
		oldctx = MemoryContextSwitchTo(indexCxt);
		vectorData = hnswExtractVectorData(orderbys[0].sk_argument, queryType, &dim,
										   indexCxt);

		if (vectorData != NULL)
		{
			/* Free old query buffer in rd_indexcxt where it was allocated */
			if (so->query)
			{
				pfree(so->query);
				so->query = NULL;
			}
			/* Store query as plain float4 array - no varlena construction */
			so->query = vectorData;  /* Already allocated in indexCxt */
			so->queryDim = dim;
		}
		MemoryContextSwitchTo(oldctx);
		/* Get k from GUC or default to 10 */
		so->k = (neurondb_hnsw_k > 0) ? neurondb_hnsw_k : 10;
	}
}

static bool
hnswgettuple(IndexScanDesc scan, ScanDirection dir)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Page		metaPage;
	HnswMetaPage meta;

	if (so->firstCall)
	{
		MemoryContext oldctx;
		Buffer		metaBuffer;

		metaBuffer = ReadBuffer(scan->indexRelation, 0);
		LockBuffer(metaBuffer, BUFFER_LOCK_SHARE);
		metaPage = BufferGetPage(metaBuffer);
		meta = (HnswMetaPage) PageGetContents(metaPage);

		if (!so->query)
		{
			UnlockReleaseBuffer(metaBuffer);
			return false;
		}

		/* Use dedicated scan context for search allocations */
		oldctx = MemoryContextSwitchTo(so->scanCtx);
		hnswSearch(scan->indexRelation, meta,
				   so->query, so->queryDim,
				   so->strategy, so->efSearch, so->k,
				   &so->results, &so->distances, &so->resultCount);
		MemoryContextSwitchTo(oldctx);

		UnlockReleaseBuffer(metaBuffer);
		so->firstCall = false;
		so->currentResult = 0;
	}

	/* Iterative scan: if we've exhausted results and iterative scan is enabled, scan more */
	if (so->currentResult >= so->resultCount && so->iterativeScanEnabled)
	{
		Buffer		metaBuffer;
		Page		metaPage;
		HnswMetaPage meta;
		MemoryContext oldctx;
		int			oldEfSearch = so->efSearch;
		int			newEfSearch;

		/* Check if we've reached max tuples or memory limit */
		if (so->scannedTuples >= so->maxScanTuples ||
			MemoryContextMemAllocated(so->scanCtx, false) > so->maxMemory)
		{
			return false;
		}

		/* Increase ef_search for next scan (double it, but cap at maxScanTuples) */
		newEfSearch = Min(so->efSearch * 2, so->maxScanTuples);
		if (newEfSearch > 100000)
			newEfSearch = 100000;	/* Hard cap */
		
		if (newEfSearch <= oldEfSearch)
		{
			/* Can't increase further, give up */
			return false;
		}

		so->efSearch = newEfSearch;

		/* Re-scan with increased ef_search */
		metaBuffer = ReadBuffer(scan->indexRelation, 0);
		LockBuffer(metaBuffer, BUFFER_LOCK_SHARE);
		metaPage = BufferGetPage(metaBuffer);
		meta = (HnswMetaPage) PageGetContents(metaPage);

		if (!so->query)
		{
			UnlockReleaseBuffer(metaBuffer);
			return false;
		}

		/* Free old results */
		if (so->results)
		{
			pfree(so->results);
			so->results = NULL;
		}
		if (so->distances)
		{
			pfree(so->distances);
			so->distances = NULL;
		}

		/* Use dedicated scan context for search allocations */
		oldctx = MemoryContextSwitchTo(so->scanCtx);
		hnswSearch(scan->indexRelation, meta,
				   so->query, so->queryDim,
				   so->strategy, so->efSearch, so->k,
				   &so->results, &so->distances, &so->resultCount);
		MemoryContextSwitchTo(oldctx);

		UnlockReleaseBuffer(metaBuffer);
		so->currentResult = 0;
		so->scannedTuples += so->resultCount;
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

	/* Check if page has valid items before accessing */
	if (PageIsNew(page) || PageIsEmpty(page) ||
		PageGetMaxOffsetNumber(page) < FirstOffsetNumber)
	{
		UnlockReleaseBuffer(buf);
		so->currentResult++;
		return false;
	}
	
	{
		ItemId scanItemId = PageGetItemId(page, FirstOffsetNumber);
		if (!ItemIdIsValid(scanItemId) || !ItemIdHasStorage(scanItemId))
		{
			UnlockReleaseBuffer(buf);
			so->currentResult++;
			return false;
		}
		
		node = (HnswNode) PageGetItem(page, scanItemId);
	}
	
	if (node != NULL)
	{
		/* Validate node structure before use */
		if (node->dim <= 0 || node->dim > 32767)
		{
			BlockNumber blkno = buf != InvalidBuffer ? BufferGetBlockNumber(buf) : 0;
			elog(WARNING, "hnsw: node at block %u has invalid dim %d in scan, skipping", blkno, node->dim);
			UnlockReleaseBuffer(buf);
			so->currentResult++;
			return false;
		}

		if (!hnswValidateLevelSafe(node->level))
		{
			elog(WARNING, "hnsw: node at block %u has invalid level %d in scan, skipping", buf != InvalidBuffer ? BufferGetBlockNumber(buf) : 0, node->level);
			UnlockReleaseBuffer(buf);
			so->currentResult++;
			return false;
		}
		
		/* Return first valid heap TID (HOT update support) */
		if (node->heaptidsLength > 0 && ItemPointerIsValid(&node->heaptids[0]))
			scan->xs_heaptid = node->heaptids[0];
		else if (ItemPointerIsValid(&node->heaptids[0]))	/* Backwards compat */
			scan->xs_heaptid = node->heaptids[0];
		else
			ItemPointerSetInvalid(&scan->xs_heaptid);
		scan->xs_recheck = false;
		scan->xs_recheckorderby = false;
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
	MemoryContext oldctx;
	MemoryContext indexCxt;

	if (so == NULL)
		return;

	/* All scan opaque data is allocated in rd_indexcxt, so free it there */
	indexCxt = scan->indexRelation->rd_indexcxt;
	oldctx = MemoryContextSwitchTo(indexCxt);

	/* Delete dedicated scan context (frees all allocations within it) */
	/* NOTE: After this, so->results and so->distances are invalid - do NOT access them */
	if (so->scanCtx != NULL)
	{
		MemoryContextDelete(so->scanCtx);
		so->scanCtx = NULL;
		/* Clear pointers to prevent accidental use-after-free */
		so->results = NULL;
		so->distances = NULL;
		so->resultCount = 0;
	}

	/* results and distances are allocated in scanCtx, so they're already freed */
	/* query is allocated in index context, so it needs explicit freeing */
	if (so->query)
	{
		nfree(so->query);
		so->query = NULL;
	}

	nfree(so);

	MemoryContextSwitchTo(oldctx);
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
	HnswOptions *opts = NULL;

	/* index->rd_options is already a processed bytea from build_reloptions (with validate=true)
	 * We can access it directly - it's a varlena with vl_len_ header */
	if (index->rd_options != NULL)
	{
		opts = (HnswOptions *) index->rd_options;
		opts_out->m = opts->m;
		opts_out->efConstruction = opts->efConstruction;
		opts_out->ef_search = opts->ef_search;
	}
	else
	{
		opts_out->m = HNSW_DEFAULT_M;
		opts_out->efConstruction = HNSW_DEFAULT_EF_CONSTRUCTION;
		opts_out->ef_search = HNSW_DEFAULT_EF_SEARCH;
	}
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
	/* Fast path: only reject obviously invalid values */
	if (blkno == InvalidBlockNumber)
	{
		return false;
	}
	
	/* Quick sanity check: block numbers > 1000000 are almost certainly garbage */
	if (blkno > 1000000)
	{
		return false;
	}
	
	/* During index build, skip expensive validation for performance.
	 * The cached value is good enough to catch most issues.
	 */
	return true;
}

/*
 * Comparison function for qsort - compare neighbor candidates by distance
 */
static int
hnswCompareNeighborDist(const void *a, const void *b)
{
	const NeighborCandidate *na = (const NeighborCandidate *) a;
	const NeighborCandidate *nb = (const NeighborCandidate *) b;
	
	if (na->dist < nb->dist)
		return -1;
	else if (na->dist > nb->dist)
		return 1;
	else
		return 0;
}

/*
 * hnswReadBufferChecked - wrapper around ReadBuffer that logs suspicious
 * block numbers before PostgreSQL errors out with "could not read blocks ..."
 */
static inline Buffer
hnswReadBufferChecked(Relation index, BlockNumber blkno, const char *ctx)
{
	/* Minimal validation - let ReadBuffer handle the real validation */
	(void) ctx; /* unused in optimized build */
	
	return ReadBuffer(index, blkno);
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
	size_t		headerSize;
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

	/*
	 * IMPORTANT: This size calculation must match the node layout macros:
	 * - Vector starts at MAXALIGN(sizeof(HnswNodeData))
	 * - Neighbors start after HnswGetVectorSize(dim) (MAXALIGNED vector area)
	 *
	 * If we under-allocate here, memset/memcpy of neighbors can scribble past
	 * node_raw and corrupt PortalContext (chunk header damage).
	 */
	headerSize = (size_t) MAXALIGN(sizeof(HnswNodeData));

	/* Check aligned vector size overflow */
	vectorSize = (size_t) HnswGetVectorSize(dim);
	if (vectorSize < (size_t) dim * sizeof(float4))
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
	totalSize = headerSize + vectorSize + neighborSize;
	if (totalSize < headerSize || totalSize < vectorSize || totalSize < neighborSize)
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

	/* Sanity: must match the canonical size computation */
	Assert(result == HnswNodeSizeWithM(dim, level, m));

	return result;
}

/*
 * Optimized L2 squared distance (no sqrt) for comparisons
 * Uses SIMD if available via external function, falls back to scalar
 * This avoids sqrt() overhead when we only need to compare distances
 */
static inline float4
hnswComputeDistanceSquaredL2(const float4 *vec1, const float4 *vec2, int dim)
{
	/* Use existing SIMD function - it's linked from neurondb_simd_impl.c */
	/* This function has SIMD optimizations (AVX2/NEON) with scalar fallback */
	extern double neurondb_l2_distance_squared(const float *a, const float *b, int n);
	
	return (float4) neurondb_l2_distance_squared((const float *) vec1, (const float *) vec2, dim);
}

/*
 * Distance computation for L2, Cosine, or negative-InnerProduct distances
 * 
 * For L2 distance, this returns sqrt(sum), but for comparisons use
 * hnswComputeDistanceSquaredL2() instead to avoid sqrt overhead.
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
			/* For comparisons, caller should use hnswComputeDistanceSquaredL2 */
			/* Here we need actual distance (with sqrt) */
			sum = (double) hnswComputeDistanceSquaredL2(vec1, vec2, dim);
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
		List *names = NULL;

		names = list_make2(makeString("public"), makeString("vector"));
		cached_vectorOid = LookupTypeNameOid(NULL, makeTypeNameFromNameList(names), true);
		if (!OidIsValid(cached_vectorOid))
		{
			ereport(ERROR,
					(errcode(ERRCODE_UNDEFINED_OBJECT),
					 errmsg("hnsw requires public.vector type"),
					 errhint("Ensure the vector type is available in the public schema")));
		}
		list_free(names);
		names = list_make2(makeString("public"), makeString("halfvec"));
		cached_halfvecOid = LookupTypeNameOid(NULL, makeTypeNameFromNameList(names), true);
		if (!OidIsValid(cached_halfvecOid))
		{
			ereport(ERROR,
					(errcode(ERRCODE_UNDEFINED_OBJECT),
					 errmsg("hnsw requires public.halfvec type"),
					 errhint("Ensure the halfvec type is available in the public schema")));
		}
		list_free(names);
		names = list_make2(makeString("public"), makeString("sparsevec"));
		cached_sparsevecOid = LookupTypeNameOid(NULL, makeTypeNameFromNameList(names), true);
		if (!OidIsValid(cached_sparsevecOid))
		{
			ereport(ERROR,
					(errcode(ERRCODE_UNDEFINED_OBJECT),
					 errmsg("hnsw requires public.sparsevec type"),
					 errhint("Ensure the sparsevec type is available in the public schema")));
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

	float4 *result = NULL;

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
		nalloc(result, float4, v->dim);
		NDB_CHECK_ALLOC(result, "vector data");
		for (i = 0; i < v->dim; i++)
			result[i] = v->data[i];
	}
	else if (typeOid == halfvecOid)
	{
		VectorF16  *hv = (VectorF16 *) PG_DETOAST_DATUM(value);
		bool		needsFree = false;

		if (hv != (VectorF16 *) DatumGetPointer(value))
			needsFree = true;

		NDB_CHECK_NULL(hv, "halfvec");
		if (hv->dim <= 0 || hv->dim > 32767)
		{
			if (needsFree)
				pfree(hv);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("hnsw: invalid halfvec dimension %d", hv->dim)));
		}
		*out_dim = hv->dim;
		NDB_CHECK_ALLOC_SIZE((size_t) hv->dim * sizeof(float4), "halfvec data");
		nalloc(result, float4, hv->dim);
		NDB_CHECK_ALLOC(result, "halfvec data");
		for (i = 0; i < hv->dim; i++)
			result[i] = fp16_to_float(hv->data[i]);

		if (needsFree)
			pfree(hv);
	}
	else if (typeOid == sparsevecOid)
	{
		VectorMap  *sv = (VectorMap *) PG_DETOAST_DATUM(value);
		bool		needsFree = false;

		if (sv != (VectorMap *) DatumGetPointer(value))
			needsFree = true;

		NDB_CHECK_NULL(sv, "sparsevec");
		if (sv->total_dim <= 0 || sv->total_dim > 32767)
		{
			if (needsFree)
				pfree(sv);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("hnsw: invalid sparsevec total_dim %d", sv->total_dim)));
		}
		{
			int32	   *indices = VECMAP_INDICES(sv);
			float4	   *values = VECMAP_VALUES(sv);

			/* Validate pointers are not null (macros now check this, but double-check) */
			if (indices == NULL || values == NULL)
			{
				if (needsFree)
					pfree(sv);
				ereport(ERROR,
						(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
						 errmsg("hnsw: sparsevec indices or values pointer is NULL")));
			}

			*out_dim = sv->total_dim;
			NDB_CHECK_ALLOC_SIZE((size_t) sv->total_dim * sizeof(float4), "sparsevec data");
			nalloc(result, float4, sv->total_dim);
			NDB_CHECK_ALLOC(result, "sparsevec data");

			/* Zero-initialize buffer: sparsevec only stores non-zero entries */
			/* Check for overflow in size calculation */
			if (sv->total_dim > 0 && (size_t) sv->total_dim > SIZE_MAX / sizeof(float4))
			{
				if (needsFree)
					pfree(sv);
				ereport(ERROR,
						(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
						 errmsg("hnsw: sparsevec total_dim %d too large for memset", sv->total_dim)));
			}
			memset(result, 0, sv->total_dim * sizeof(float4));

			/* Populate non-zero entries with comprehensive bounds checking */
			for (i = 0; i < sv->nnz; i++)
			{
				/* Validate index is within bounds - error on out-of-bounds to prevent silent corruption */
				/* This ensures indices[i] is valid for both sv->total_dim and result array (which has same size) */
				if (indices[i] < 0 || indices[i] >= sv->total_dim)
				{
					if (needsFree)
						pfree(sv);
					ereport(ERROR,
							(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
							 errmsg("hnsw: sparsevec index %d out of bounds (dim=%d, nnz=%d)",
									indices[i], sv->total_dim, sv->nnz)));
				}
				result[indices[i]] = values[i];
			}
		}

		if (needsFree)
			pfree(sv);
	}
	else if (typeOid == bitOid)
	{
		VarBit	   *bit_vec = (VarBit *) PG_DETOAST_DATUM(value);
		bool		needsFree = false;

		if (bit_vec != (VarBit *) DatumGetPointer(value))
			needsFree = true;

		NDB_CHECK_NULL(bit_vec, "bit vector");
		{
			int			nbits;
			bits8 *bit_data = NULL;

			nbits = VARBITLEN(bit_vec);
			if (nbits <= 0 || nbits > 32767)
			{
				if (needsFree)
					pfree(bit_vec);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("hnsw: invalid bit vector length %d", nbits)));
			}
			bit_data = VARBITS(bit_vec);
			if (bit_data == NULL)
			{
				if (needsFree)
					pfree(bit_vec);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("hnsw: bit vector data is NULL")));
			}
			*out_dim = nbits;
			NDB_CHECK_ALLOC_SIZE((size_t) nbits * sizeof(float4), "bit vector data");
			nalloc(result, float4, nbits);
			NDB_CHECK_ALLOC(result, "bit vector data");
			for (i = 0; i < nbits; i++)
			{
				int			byte_idx = i / BITS_PER_BYTE;
				int			bit_idx = i % BITS_PER_BYTE;
				int			bit_val = (bit_data[byte_idx] >> (BITS_PER_BYTE - 1 - bit_idx)) & 1;

				result[i] = bit_val ? 1.0f : -1.0f;
			}
		}

		if (needsFree)
			pfree(bit_vec);
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
/*
 * hnswSearch - Find k nearest neighbors using HNSW graph traversal
 *
 * Memory Management:
 * - All allocations use CurrentMemoryContext (expected to be scan context)
 * - Large arrays (visitedSet, candidates, results) are allocated once
 * - Memory is bounded by efSearch parameter
 * - Caller is responsible for context lifecycle
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
	float4	   *nodeVector = NULL;
	float4		currentDist;
	int			level;
	int			i,
				j;

	BlockNumber *candidates = NULL;
	float4 *candidateDists = NULL;
	int			candidateCount = 0;
	int			candidateCapacity = 0;

	BlockNumber *visited = NULL;
	int			visitedCount = 0;
	int			visitedCapacity = 0;

	bool *visitedSet = NULL;
	BlockNumber *neighbors = NULL;
	int16		neighborCount;

	BlockNumber *topK = NULL;
	float4 *topKDists = NULL;
	int			topKCount = 0;

	int *indices = NULL;
	int			minIdx,
				temp;
	int			l,
				worstIdx;
	float4		worstDist,
				minDist;
	BlockNumber numBlocks;	/* Computed once and reused throughout function */

	elog(DEBUG1, "[HNSW_SEARCH_START] dim=%d strategy=%d efSearch=%d k=%d entryPoint=%u entryLevel=%d insertedVectors=%lld",
		 dim, strategy, efSearch, k, metaPage->entryPoint, metaPage->entryLevel, (long long) metaPage->insertedVectors);

	/* Defensive: no vectors yet */
	if (metaPage->entryPoint == InvalidBlockNumber)
	{
		elog(DEBUG1, "[HNSW_SEARCH_EMPTY] entryPoint is InvalidBlockNumber, returning empty results");
		*results = NULL;
		*distances = NULL;
		*resultCount = 0;
		return;
	}

	/* Enforce k <= efSearch: effective k is min(k, efSearch) */
	if (k > efSearch)
		k = efSearch;

	current = metaPage->entryPoint;
	currentLevel = metaPage->entryLevel;

	/* Validate entry level */
	if (currentLevel < 0 || currentLevel >= HNSW_MAX_LEVEL)
	{
		elog(WARNING, "hnsw: invalid entryLevel %d, resetting to 0", currentLevel);
		currentLevel = 0;
	}

	/* Initialize data structures with capacity bounds */
	numBlocks = RelationGetNumberOfBlocks(index);
	
	/* Allocate visited array - bounded by efSearch for memory efficiency */
	visitedCapacity = (efSearch > 1 ? efSearch * 2 : 32);
	if (visitedCapacity > 100000)
		visitedCapacity = 100000;  /* Cap to prevent excessive allocation */
	nalloc(visited, BlockNumber, visitedCapacity);

	/* Allocate visitedSet with overflow checking - must be zero-initialized */
	/* Use bit array only if index size is reasonable to avoid huge allocations */
	if (numBlocks <= 1000000)  /* Cap at ~1M blocks (~8GB index) */
	{
		size_t visitedSetSize = (size_t) numBlocks * sizeof(bool);
		if (visitedSetSize / sizeof(bool) != (size_t) numBlocks)
		{
			/* Overflow - fall back to array-based visited tracking */
			visitedSet = NULL;
		}
		else
		{
			visitedSet = (bool *) palloc0(visitedSetSize);
		}
	}
	else
	{
		/* Index too large for bit array - use array-based tracking only */
		visitedSet = NULL;
	}
	visitedCount = 0;

	/* Allocate candidate arrays with exact capacity */
	candidateCapacity = efSearch;
	if (candidateCapacity > 100000)
		candidateCapacity = 100000;  /* Cap to prevent excessive allocation */
	nalloc(candidates, BlockNumber, candidateCapacity);
	nalloc(candidateDists, float4, candidateCapacity);
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

				nodeBuf = HNSW_READBUFFER(index, current);
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

			/* Validate node dim before using */
			if (node->dim <= 0 || node->dim > 32767)
			{
				elog(WARNING, "hnsw: node at block %u has invalid dim %d in search, skipping", current, node->dim);
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
				BlockNumber *neighborBlocks = NULL;
				int			validNeighborCount = 0;
				int maxNeighbors;
				
				/* Validate level bounds BEFORE array access to prevent out-of-bounds read */
				if (level < 0 || level >= HNSW_MAX_LEVEL)
				{
					elog(WARNING, "hnsw: invalid level %d in search, skipping (max: %d)",
						 level, HNSW_MAX_LEVEL - 1);
					UnlockReleaseBuffer(nodeBuf);
					break;
				}
				
				neighbors = HnswGetNeighborsSafe(node, level, metaPage->m);
				neighborCount = node->neighborCount[level];

				elog(DEBUG1, "[HNSW_SEARCH_LEVEL_READ] current=%u level=%d node->level=%d neighborCount=%d m=%d",
					 current, level, node->level, neighborCount, metaPage->m);

				/* Validate and clamp neighborCount BEFORE accessing neighbors array */
				neighborCount = hnswValidateNeighborCount(neighborCount, metaPage->m, level);
				
				/* Additional safety: clamp to maximum possible neighbors */
				maxNeighbors = metaPage->m * 2;
					if (neighborCount > maxNeighbors)
					{
						elog(WARNING, "hnsw: neighborCount %d exceeds maximum %d at level %d, clamping",
							 neighborCount, maxNeighbors, level);
						neighborCount = maxNeighbors;
					}
					
					/* Ensure neighborCount is non-negative */
					if (neighborCount < 0)
						neighborCount = 0;

					/* Copy neighbor block numbers to avoid deadlock - unlock nodeBuf first */
					if (neighborCount > 0)
					{
						nalloc(neighborBlocks, BlockNumber, neighborCount);
						for (i = 0; i < neighborCount; i++)
						{
							BlockNumber neighborBlk = neighbors[i];
							
							/* Strict validation: must be valid block number and within index bounds */
							if (neighborBlk == InvalidBlockNumber)
								continue;
							
							/* Additional sanity check: block number should be reasonable */
							if (neighborBlk > 1000000)  /* Sanity check for corrupted data */
							{
								elog(WARNING, "hnsw: suspiciously large neighbor block %u at level %d index %d, skipping",
									 neighborBlk, level, i);
								continue;
							}
							
							/* Validate block number is within index bounds */
							if (!hnswValidateBlockNumber(neighborBlk, index))
							{
								elog(WARNING, "hnsw: invalid neighbor block %u at level %d index %d, skipping",
									 neighborBlk, level, i);
								continue;
							}
							
							/* Valid neighbor - add to array */
							neighborBlocks[validNeighborCount++] = neighborBlk;
						}
					}

					/* Unlock nodeBuf before processing neighbors to avoid deadlock */
					UnlockReleaseBuffer(nodeBuf);
					nodeBuf = InvalidBuffer;

					/* Now process each neighbor one at a time */
					for (i = 0; i < validNeighborCount; i++)
					{
						Buffer		neighborBuf;
						Page		neighborPage;
						HnswNode	neighbor;
						float4 *neighborVector = NULL;
						float4		neighborDist;
						BlockNumber neighborBlk = neighborBlocks[i];
						ItemId		itemId;

						/* Double-check validation before reading buffer */
						if (!hnswValidateBlockNumber(neighborBlk, index))
						{
							elog(WARNING, "hnsw: neighbor block %u failed validation before read at level %d, skipping",
								 neighborBlk, level);
							continue;
						}

						neighborBuf = HNSW_READBUFFER(index, neighborBlk);
						LockBuffer(neighborBuf, BUFFER_LOCK_SHARE);
						neighborPage = BufferGetPage(neighborBuf);

						if (PageIsNew(neighborPage) || PageIsEmpty(neighborPage))
						{
							UnlockReleaseBuffer(neighborBuf);
							continue;
						}

						/* Check if page has valid items before accessing */
						if (PageGetMaxOffsetNumber(neighborPage) < FirstOffsetNumber)
						{
							UnlockReleaseBuffer(neighborBuf);
							continue;
						}

						itemId = PageGetItemId(neighborPage, FirstOffsetNumber);
					if (!ItemIdIsValid(itemId) || !ItemIdHasStorage(itemId))
					{
						UnlockReleaseBuffer(neighborBuf);
						continue;
					}

					neighbor = (HnswNode) PageGetItem(neighborPage, itemId);
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
							current = neighborBlk;
							currentDist = neighborDist;
							foundBetter = true;
						}

						UnlockReleaseBuffer(neighborBuf);
					}

					if (neighborBlocks)
						pfree(neighborBlocks);
				}
				else
				{
					UnlockReleaseBuffer(nodeBuf);
					nodeBuf = InvalidBuffer;
				}
			} while (foundBetter);
		}

	if (!hnswValidateBlockNumber(current, index))
	{
		elog(WARNING, "hnsw: invalid current block %u for level 0 search", current);
		*results = NULL;
		*distances = NULL;
		*resultCount = 0;
		nfree(visited);
		if (visitedSet != NULL)
			pfree(visitedSet);
		nfree(candidates);
		nfree(candidateDists);
		return;
	}

		/* Bounds check before accessing candidate array */
		if (candidateCapacity > 0)
		{
			candidates[0] = current;
		}
		else
		{
			elog(ERROR, "hnsw: candidateCapacity is 0, cannot store entry point");
		}
		nodeBuf = HNSW_READBUFFER(index, current);
		LockBuffer(nodeBuf, BUFFER_LOCK_SHARE);
		nodePage = BufferGetPage(nodeBuf);

	if (PageIsNew(nodePage) || PageIsEmpty(nodePage))
	{
		UnlockReleaseBuffer(nodeBuf);
		*results = NULL;
		*distances = NULL;
		*resultCount = 0;
		nfree(visited);
		if (visitedSet != NULL)
			pfree(visitedSet);
		nfree(candidates);
		nfree(candidateDists);
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
		nfree(visited);
		if (visitedSet != NULL)
			pfree(visitedSet);
		nfree(candidates);
		nfree(candidateDists);
		return;
	}

	if (!hnswValidateLevelSafe(node->level))
	{
		UnlockReleaseBuffer(nodeBuf);
		*results = NULL;
		*distances = NULL;
		*resultCount = 0;
		nfree(visited);
		if (visitedSet != NULL)
			pfree(visitedSet);
		nfree(candidates);
		nfree(candidateDists);
		return;
	}

	nodeVector = HnswGetVector(node);
	if (nodeVector == NULL)
	{
		UnlockReleaseBuffer(nodeBuf);
		*results = NULL;
		*distances = NULL;
		*resultCount = 0;
		nfree(visited);
		if (visitedSet != NULL)
			pfree(visitedSet);
		nfree(candidates);
		nfree(candidateDists);
		return;
	}

		/* Bounds check before accessing candidate arrays */
		if (candidateCapacity > 0)
		{
			candidateDists[0] = hnswComputeDistance(query, nodeVector, dim, strategy);
			candidates[0] = current;
			candidateCount = 1;
		}
		else
		{
			elog(ERROR, "hnsw: candidateCapacity is 0, cannot store candidates");
		}
		/* Bounds check before accessing visited array */
		if (visitedCount < visitedCapacity)
		{
			visited[visitedCount++] = current;
		}
		else
		{
			elog(WARNING, "hnsw: visited array at capacity %d, skipping initial entry", visitedCapacity);
		}
		if (visitedSet != NULL && current < numBlocks)
			visitedSet[current] = true;
		UnlockReleaseBuffer(nodeBuf);

	/* Process candidates - use a fixed limit to avoid infinite loop */
	{
		int processedCount = 0;
		int maxIterations = Min(efSearch * 2, candidateCapacity);  /* Limit iterations */
		int maxNeighbors;
		int jj;
		
		while (processedCount < candidateCount && processedCount < maxIterations)
		{
			BlockNumber candidate;
			BlockNumber *neighborBlocks = NULL;
			int			validNeighborCount = 0;
			BlockNumber *candNeighbors = NULL;
			int16		candNeighborCount;
			OffsetNumber maxOff;

			/* Bounds check before accessing candidates array */
			if (processedCount >= candidateCount || processedCount >= candidateCapacity)
			{
				elog(WARNING, "hnsw: processedCount %d out of bounds (candidateCount=%d, candidateCapacity=%d), breaking",
					 processedCount, candidateCount, candidateCapacity);
				break;
			}
			candidate = candidates[processedCount];
			processedCount++; /* Increment at start to avoid infinite loop */
			
			CHECK_FOR_INTERRUPTS();

			if (!hnswValidateBlockNumber(candidate, index))
			{
				elog(WARNING, "hnsw: invalid candidate block %u, skipping", candidate);
				continue;
			}

			nodeBuf = HNSW_READBUFFER(index, candidate);
			LockBuffer(nodeBuf, BUFFER_LOCK_SHARE);
			nodePage = BufferGetPage(nodeBuf);

			if (PageIsNew(nodePage) || PageIsEmpty(nodePage))
			{
				UnlockReleaseBuffer(nodeBuf);
				continue;
			}

			/* Validate page structure before reading node */
			maxOff = PageGetMaxOffsetNumber(nodePage);
				/* Sanity check: maxOffset should be reasonable (max ~200-300 items per page) */
				if (maxOff > 1000)
				{
					UnlockReleaseBuffer(nodeBuf);
					continue;
				}
				
				/* HNSW constraint: one node per page, so maxOffset must be FirstOffsetNumber */
				if (maxOff != FirstOffsetNumber && maxOff != InvalidOffsetNumber)
				{
					UnlockReleaseBuffer(nodeBuf);
					continue;
				}
				
				{
					ItemId itemId = PageGetItemId(nodePage, FirstOffsetNumber);
					if (!ItemIdIsValid(itemId) || !ItemIdHasStorage(itemId))
					{
						UnlockReleaseBuffer(nodeBuf);
						continue;
					}
					node = (HnswNode) PageGetItem(nodePage, itemId);
				}
				if (node == NULL)
				{
					UnlockReleaseBuffer(nodeBuf);
					continue;
				}

			/* Validate node structure immediately after reading - check for all-zeros corruption */
			if (node->dim == 0 && node->level == 0 && node->neighborCount[0] == 0)
			{
				/* Check if entire node structure is zeroed (corruption indicator) */
				bool allZeros = true;
				for (int z = 0; z < HNSW_HEAPTIDS && allZeros; z++)
				{
					if (ItemPointerIsValid(&node->heaptids[z]))
						allZeros = false;
				}
				if (allZeros)
				{
					UnlockReleaseBuffer(nodeBuf);
					continue;
				}
			}
			
			if (!hnswValidateLevelSafe(node->level))
			{
				UnlockReleaseBuffer(nodeBuf);
				continue;
			}

			/* Validate node dim before using */
			if (node->dim <= 0 || node->dim > 32767)
			{
				elog(WARNING, "hnsw: candidate node at block %u has invalid dim %d, skipping", candidate, node->dim);
				UnlockReleaseBuffer(nodeBuf);
				continue;
			}

		/* Validate level bounds BEFORE array access (level 0 is always valid) */
		candNeighbors = HnswGetNeighborsSafe(node, 0, metaPage->m);
			candNeighborCount = node->neighborCount[0];

			/* Validate and clamp neighborCount BEFORE accessing neighbors array */
			candNeighborCount = hnswValidateNeighborCount(candNeighborCount, metaPage->m, 0);
			
			/* Additional safety: clamp to maximum possible neighbors */
			maxNeighbors = metaPage->m * 2;
			if (candNeighborCount > maxNeighbors)
			{
				elog(WARNING, "hnsw: neighborCount %d exceeds maximum %d at level 0, clamping",
					 candNeighborCount, maxNeighbors);
				candNeighborCount = maxNeighbors;
			}
			
			/* Ensure neighborCount is non-negative */
			if (candNeighborCount < 0)
				candNeighborCount = 0;

			/* Copy neighbor block numbers to avoid deadlock - unlock nodeBuf first */
			/* Validate neighbors array bounds before accessing */
			if (candNeighborCount > 0)
			{
				nalloc(neighborBlocks, BlockNumber, candNeighborCount);
			for (jj = 0; jj < candNeighborCount; jj++)
			{
				BlockNumber neighborBlk = candNeighbors[jj];
				
				/* Strict validation: must be valid block number and within index bounds */
				if (neighborBlk == InvalidBlockNumber)
					continue;
				
				/* Additional sanity check: block number should be reasonable */
				if (neighborBlk > 1000000)  /* Sanity check for corrupted data */
				{
					elog(WARNING, "hnsw: suspiciously large neighbor block %u at index %d, skipping",
						 neighborBlk, jj);
					continue;
				}
				
				/* Validate block number is within index bounds */
				if (!hnswValidateBlockNumber(neighborBlk, index))
				{
					elog(WARNING, "hnsw: invalid neighbor block %u at index %d, skipping",
						 neighborBlk, jj);
					continue;
				}
						
						/* Check if already visited (if visitedSet is available) */
						if (visitedSet != NULL && neighborBlk < numBlocks && visitedSet[neighborBlk])
							continue;
						
						/* Valid neighbor - add to array */
						neighborBlocks[validNeighborCount++] = neighborBlk;
					}
				}

				/* Unlock nodeBuf before processing neighbors to avoid deadlock */
				UnlockReleaseBuffer(nodeBuf);
				nodeBuf = InvalidBuffer;

				/* Now process each neighbor one at a time */
				for (j = 0; j < validNeighborCount; j++)
				{
					Buffer		neighborBuf;
					Page		neighborPage;
					HnswNode	neighbor;
					float4 *neighborVector = NULL;
					float4		neighborDist;
					BlockNumber neighborBlk = neighborBlocks[j];
					ItemId		itemId;

					/* Double-check validation before reading buffer */
					if (!hnswValidateBlockNumber(neighborBlk, index))
					{
						elog(WARNING, "hnsw: neighbor block %u failed validation before read, skipping",
							 neighborBlk);
						continue;
					}

					neighborBuf = HNSW_READBUFFER(index, neighborBlk);
					LockBuffer(neighborBuf, BUFFER_LOCK_SHARE);
					neighborPage = BufferGetPage(neighborBuf);

					if (PageIsNew(neighborPage) || PageIsEmpty(neighborPage))
					{
						UnlockReleaseBuffer(neighborBuf);
						continue;
					}

					/* Check if page has valid items before accessing */
					if (PageGetMaxOffsetNumber(neighborPage) < FirstOffsetNumber)
					{
						UnlockReleaseBuffer(neighborBuf);
						continue;
					}

					itemId = PageGetItemId(neighborPage, FirstOffsetNumber);
					if (!ItemIdIsValid(itemId) || !ItemIdHasStorage(itemId))
					{
						UnlockReleaseBuffer(neighborBuf);
						continue;
					}

					neighbor = (HnswNode) PageGetItem(neighborPage, itemId);
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

					/* Use neighborBlocks[j] instead of neighbors[j] after unlocking nodeBuf */
					{
						BlockNumber nb = neighborBlk;
						if (visitedSet != NULL && nb < numBlocks)
							visitedSet[nb] = true;

				/* Check capacity before writing to prevent overflow */
				if (visitedCount >= visitedCapacity)
				{
					if (visitedCapacity >= HNSW_MAX_VISITED_CAPACITY)
					{
						elog(WARNING, "hnsw: visited array reached maximum capacity %d, "
									  "dropping extra neighbors",
							 HNSW_MAX_VISITED_CAPACITY);
						/* Skip recording this neighbor */
						continue;
					}
					else
					{
						/* Check for integer overflow in capacity calculation */
						int newCapacity;
						if (visitedCapacity > INT_MAX / 2)
						{
							/* Would overflow - cap at maximum */
							newCapacity = HNSW_MAX_VISITED_CAPACITY;
						}
						else
						{
							newCapacity = Min(visitedCapacity * 2, HNSW_MAX_VISITED_CAPACITY);
						}
						
						if (newCapacity > visitedCapacity)
						{
							/* Check for overflow in size calculation before repalloc */
							size_t newSize = (size_t) newCapacity * sizeof(BlockNumber);
							if (newSize / sizeof(BlockNumber) != (size_t) newCapacity)
							{
								elog(WARNING, "hnsw: visited array size calculation overflow, capping at current capacity");
								newCapacity = visitedCapacity;
							}
							else
							{
								visitedCapacity = newCapacity;
								visited = (BlockNumber *) repalloc(visited, newSize);
								if (visited == NULL)
								{
									elog(ERROR, "hnsw: repalloc failed for visited array");
								}
							}
						}
					}
				}

				/* Bounds check before writing to visited array */
				if (visitedCount < visitedCapacity)
				{
					visited[visitedCount++] = nb;
				}
				else
				{
					elog(WARNING, "hnsw: visited array at capacity %d, skipping neighbor %u", visitedCapacity, nb);
				}

						/* Ensure candidateCount doesn't exceed candidateCapacity */
						if (candidateCount < candidateCapacity)
						{
							candidates[candidateCount] = neighborBlk;
							candidateDists[candidateCount] = neighborDist;
							candidateCount++;
						}
						else
						{
							/* Find worst candidate to replace */
							if (candidateCount > 0 && candidateCount <= candidateCapacity)
							{
								worstIdx = 0;
								worstDist = candidateDists[0];
								for (l = 1; l < candidateCount; l++)
								{
									if (l < candidateCapacity && candidateDists[l] > worstDist)
									{
										worstDist = candidateDists[l];
										worstIdx = l;
									}
								}

								/* Bounds check before writing */
								if (worstIdx >= 0 && worstIdx < candidateCapacity && neighborDist < worstDist)
								{
									candidates[worstIdx] = neighborBlk;
									candidateDists[worstIdx] = neighborDist;
								}
							}
						}
					}
				}

				if (neighborBlocks)
					pfree(neighborBlocks);
			}
		}

		nalloc(indices, int, candidateCount);
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
		if (topKCount > 0)
		{
			nalloc(topK, BlockNumber, topKCount);
			nalloc(topKDists, float4, topKCount);
			for (i = 0; i < topKCount; i++)
			{
				topK[i] = candidates[indices[i]];
				topKDists[i] = candidateDists[indices[i]];
			}
		}
		else
		{
			topK = NULL;
			topKDists = NULL;
		}

		pfree(indices);
		indices = NULL;

	*results = topK;
	*distances = topKDists;
	*resultCount = topKCount;

	nfree(candidates);
	candidates = NULL;
	nfree(candidateDists);
	candidateDists = NULL;
	nfree(visited);
	visited = NULL;
	if (visitedSet != NULL)
	{
		pfree(visitedSet);
		visitedSet = NULL;
	}
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

	elog(DEBUG1, "[HNSW_INSERT_START] dim=%d insertedVectors=%lld entryPoint=%u entryLevel=%d m=%d efConstruction=%d",
		 dim, (long long) metaPage->insertedVectors, metaPage->entryPoint, metaPage->entryLevel,
		 metaPage->m, metaPage->efConstruction);

	level = hnswGetRandomLevel(metaPage->ml);
	
	elog(DEBUG1, "[HNSW_INSERT_LEVEL] generated level=%d", level);

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
		BlockNumber *neighbors = NULL;
		int			l;

		nodeSize = hnswComputeNodeSizeSafe(dim, level, m, &overflow);
		if (overflow || nodeSize == 0)
		{
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("hnsw: node size calculation overflow (dim=%d, level=%d, m=%d)",
							dim, level, m)));
		}
		nalloc(node_raw, char, nodeSize);
		node = (HnswNode) node_raw;
		
		/* Initialize heap TIDs array */
		node->heaptidsLength = 0;
		for (i = 0; i < HNSW_HEAPTIDS; i++)
			ItemPointerSetInvalid(&node->heaptids[i]);
		hnswAddHeapTid(node, heapPtr);
		
		node->level = level;
		node->dim = dim;
		for (i = 0; i < HNSW_MAX_LEVEL; i++)
			node->neighborCount[i] = 0;
		/* Check for overflow in memcpy size calculation */
		if (dim > 0 && (size_t) dim > SIZE_MAX / sizeof(float4))
		{
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("hnsw: vector dimension %d too large for memcpy", dim)));
		}
		memcpy(HnswGetVector(node), vector, dim * sizeof(float4));

		/* Initialize neighbor arrays to InvalidBlockNumber for safety */
		for (l = 0; l <= level; l++)
		{
			node->neighborCount[l] = 0;
			neighbors = HnswGetNeighborsSafe(node, l, m);
			
			/* Verify pointer is within allocated node buffer */
			if ((char*)neighbors < node_raw || 
				(char*)neighbors + m * 2 * sizeof(BlockNumber) > node_raw + nodeSize)
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("hnsw: neighbor pointer out of bounds for level %d (node_raw=%p, neighbors=%p, nodeSize=%zu)",
								l, node_raw, neighbors, nodeSize)));
			}
			
			/* Explicit loop is safer and more readable than memset */
			for (i = 0; i < m * 2; i++)
			{
				neighbors[i] = InvalidBlockNumber;  /* 0xFFFFFFFF */
			}
			
			/* Verify first and last neighbors were set correctly */
			if (neighbors[0] != InvalidBlockNumber || neighbors[m * 2 - 1] != InvalidBlockNumber)
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("hnsw: failed to set InvalidBlockNumber (first=%u, last=%u, expected=%u)",
								neighbors[0], neighbors[m * 2 - 1], InvalidBlockNumber)));
			}
			
			elog(DEBUG1, "[HNSW_INIT] level=%d m=%d: initialized %d neighbors to InvalidBlockNumber (0x%08X), neighborCount[%d]=%d",
				 l, m, m * 2, InvalidBlockNumber, l, node->neighborCount[l]);
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
		float4 *entryVector = NULL;
		BlockNumber *entryNeighbors = NULL;
		int16		entryNeighborCount;
		bool		improved = true;
		int			iterations = 0;
		/* Make maxIterations dynamic based on entryLevel to handle tall graphs */
		int			maxIterations = Min(metaPage->entryLevel + 1, 32);

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

			entryBuf = HNSW_READBUFFER(index, bestEntry);
			LockBuffer(entryBuf, BUFFER_LOCK_SHARE);
			entryPage = BufferGetPage(entryBuf);

			if (PageIsNew(entryPage) || PageIsEmpty(entryPage))
			{
				UnlockReleaseBuffer(entryBuf);
				break;
			}

			/* Check if page has valid items before accessing */
			if (PageGetMaxOffsetNumber(entryPage) < FirstOffsetNumber)
			{
				UnlockReleaseBuffer(entryBuf);
				break;
			}

			{
				ItemId entryItemId = PageGetItemId(entryPage, FirstOffsetNumber);
				if (!ItemIdIsValid(entryItemId) || !ItemIdHasStorage(entryItemId))
				{
					UnlockReleaseBuffer(entryBuf);
					break;
				}

				entryNode = (HnswNode) PageGetItem(entryPage, entryItemId);
			}
			
			if (entryNode == NULL)
			{
				UnlockReleaseBuffer(entryBuf);
				break;
			}

			/* Validate entry node structure - must be done BEFORE any use */
			if (entryNode->dim <= 0 || entryNode->dim > 32767)
			{
				elog(WARNING, "hnsw: entry node has invalid dim %d at block %u, skipping", entryNode->dim, bestEntry);
				UnlockReleaseBuffer(entryBuf);
				break;
			}

			/* Validate entry node level */
			if (!hnswValidateLevelSafe(entryNode->level))
			{
				elog(WARNING, "hnsw: entry node has invalid level %d at block %u, skipping", entryNode->level, bestEntry);
				UnlockReleaseBuffer(entryBuf);
				break;
			}

			if (entryNode->level >= level)
			{
				/* Validate level bounds BEFORE array access to prevent out-of-bounds read */
				if (level < 0 || level >= HNSW_MAX_LEVEL)
				{
					elog(WARNING, "hnsw: invalid level %d in insert entry search, skipping (max: %d)",
						 level, HNSW_MAX_LEVEL - 1);
					UnlockReleaseBuffer(entryBuf);
					break;
				}
				
				/* Double-check dim is still valid before using */
				if (entryNode->dim <= 0 || entryNode->dim > 32767)
				{
					elog(WARNING, "hnsw: entry node dim %d became invalid, skipping", entryNode->dim);
					UnlockReleaseBuffer(entryBuf);
					break;
				}
				
				entryVector = HnswGetVector(entryNode);
				if (entryVector == NULL)
				{
					UnlockReleaseBuffer(entryBuf);
					break;
				}

				bestDist = hnswComputeDistance(vector, entryVector, dim, 1);
				
				/* Final validation before calling HnswGetNeighborsSafe to catch any issues */
				if (entryNode->dim <= 0 || entryNode->dim > 32767)
				{
					ereport(ERROR,
							(errcode(ERRCODE_DATA_CORRUPTED),
							 errmsg("hnsw: entry node dim %d is invalid (expected %d) - index corruption at block %u",
									entryNode->dim, dim, bestEntry)));
				}
				
				entryNeighbors = HnswGetNeighborsSafe(entryNode, level, metaPage->m);
					entryNeighborCount = entryNode->neighborCount[level];

					elog(DEBUG1, "[HNSW_INSERT_ENTRY_READ] bestEntry=%u level=%d entryNode->level=%d entryNeighborCount=%d m=%d",
						 bestEntry, level, entryNode->level, entryNeighborCount, metaPage->m);
					
					if (entryNeighborCount > 0)
					{
						for (int dbg_i = 0; dbg_i < Min(entryNeighborCount, 5); dbg_i++)
						{
							elog(DEBUG1, "[HNSW_INSERT_ENTRY_NEIGHBOR] i=%d entryNeighbors[%d]=%u (0x%08X) InvalidBlockNumber=0x%08X",
								 dbg_i, dbg_i, entryNeighbors[dbg_i], entryNeighbors[dbg_i], InvalidBlockNumber);
						}
					}

					/* Validate and clamp neighborCount */
					entryNeighborCount = hnswValidateNeighborCount(entryNeighborCount, metaPage->m, level);

					for (i = 0; i < entryNeighborCount; i++)
					{
						CHECK_FOR_INTERRUPTS();

						if (entryNeighbors[i] == InvalidBlockNumber)
						{
							elog(DEBUG1, "[HNSW_INSERT_ENTRY_SKIP] i=%d entryNeighbors[%d]=InvalidBlockNumber, skipping",
								 i, i);
							continue;
						}

						elog(DEBUG1, "[HNSW_INSERT_ENTRY_CHECK] i=%d entryNeighbors[%d]=%u insertedVectors=%lld RelationBlocks=%u",
							 i, i, entryNeighbors[i], (long long) metaPage->insertedVectors, RelationGetNumberOfBlocks(index));

						/* Use proper validation function that checks actual relation size */
						if (!hnswValidateBlockNumber(entryNeighbors[i], index))
						{
							elog(WARNING, "[HNSW_INSERT_ENTRY_SKIP] invalid neighbor block %u failed validation, skipping",
								 entryNeighbors[i]);
							continue;
						}

						{
							Buffer		neighborBuf;
							Page		neighborPage;
							HnswNode	neighbor;
							float4	   *neighborVector = NULL;
							float4		neighborDist;

							neighborBuf = HNSW_READBUFFER(index, entryNeighbors[i]);
							LockBuffer(neighborBuf, BUFFER_LOCK_SHARE);
						neighborPage = BufferGetPage(neighborBuf);

						if (PageIsNew(neighborPage) || PageIsEmpty(neighborPage))
						{
							UnlockReleaseBuffer(neighborBuf);
							continue;
						}

						/* Check if page has valid items before accessing */
						if (PageGetMaxOffsetNumber(neighborPage) < FirstOffsetNumber)
						{
							UnlockReleaseBuffer(neighborBuf);
							continue;
						}

						{
							ItemId neighborItemId2 = PageGetItemId(neighborPage, FirstOffsetNumber);
							if (!ItemIdIsValid(neighborItemId2) || !ItemIdHasStorage(neighborItemId2))
							{
								UnlockReleaseBuffer(neighborBuf);
								continue;
							}

							neighbor = (HnswNode) PageGetItem(neighborPage, neighborItemId2);
						}
						
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
	buf = HNSW_READBUFFER(index, P_NEW);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	
	/* Get the actual block number after allocation */
	blkno = BufferGetBlockNumber(buf);
	
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

	elog(DEBUG1, "[HNSW_INSERT_PRE_PAGEADD] blkno=%u nodeSize=%zu level=%d dim=%d m=%d heap=(%u,%u) heaptidsLength=%d",
		 blkno, nodeSize, node->level, node->dim, metaPage->m,
		 ItemPointerGetBlockNumber(&node->heaptids[0]), ItemPointerGetOffsetNumber(&node->heaptids[0]), node->heaptidsLength);
	
	/* Log first few neighbors before PageAddItem */
	/* Validate node dim before debug logging */
	if (node->dim > 0 && node->dim <= 32767)
	{
		for (i = 0; i <= Min(level, 2); i++)
		{
			BlockNumber *check_neighbors = HnswGetNeighborsSafe(node, i, metaPage->m);
			elog(DEBUG1, "[HNSW_INSERT_PRE_PAGEADD] level=%d neighborCount=%d first_neighbor=0x%08X (should be 0x%08X)",
				 i, node->neighborCount[i], check_neighbors[0], InvalidBlockNumber);
		}
	}
	else
	{
		elog(WARNING, "[HNSW_INSERT_PRE_PAGEADD] node has invalid dim %d, skipping debug log", node->dim);
	}
	
	/* Validate node before adding to page */
	if (node->dim != dim || node->dim <= 0 || node->dim > 32767)
	{
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("hnsw: node has invalid dim %d before PageAddItem (expected %d)",
						node->dim, dim)));
	}
	if (node->level != level || !hnswValidateLevelSafe(node->level))
	{
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("hnsw: node has invalid level %d before PageAddItem (expected %d)",
						node->level, level)));
	}
	
	/* Ensure page is truly empty - double check before adding */
	if (PageGetMaxOffsetNumber(page) != InvalidOffsetNumber)
	{
		UnlockReleaseBuffer(buf);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("hnsw: page %u already has items (maxOffset=%u), cannot add node (HNSW uses one node per page)",
						blkno, PageGetMaxOffsetNumber(page))));
	}
	
	if (PageAddItem(page, (Item) node, nodeSize, InvalidOffsetNumber, false, false) == InvalidOffsetNumber)
	{
		UnlockReleaseBuffer(buf);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("hnsw: failed to add node to page %u (nodeSize=%zu, freeSpace=%zu)",
						blkno, nodeSize, PageGetFreeSpace(page))));
	}

	/* Immediately verify the node was written correctly and page still has only one item */
	{
		OffsetNumber maxOffset;
		ItemId		itemId;
		Size		itemSize;

		maxOffset = PageGetMaxOffsetNumber(page);
		if (maxOffset != FirstOffsetNumber)
		{
			UnlockReleaseBuffer(buf);
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("hnsw: page %u has %u items after PageAddItem (expected 1) - HNSW violation",
							blkno, maxOffset)));
		}
		
		itemId = PageGetItemId(page, FirstOffsetNumber);
		if (!ItemIdIsValid(itemId) || !ItemIdHasStorage(itemId))
		{
			UnlockReleaseBuffer(buf);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("hnsw: item ID invalid after PageAddItem at block %u", blkno)));
		}
		
		/* Verify item size matches expected node size */
		itemSize = ItemIdGetLength(itemId);
		if (itemSize != nodeSize)
		{
			UnlockReleaseBuffer(buf);
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("hnsw: item size mismatch after PageAddItem at block %u: got %zu, expected %zu",
							blkno, itemSize, nodeSize)));
		}
		
		{
			HnswNode verifyNode = (HnswNode) PageGetItem(page, itemId);
			if (verifyNode == NULL)
			{
				UnlockReleaseBuffer(buf);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("hnsw: null node after PageGetItem at block %u", blkno)));
			}
			
			/* Validate node structure immediately after reading */
			if (verifyNode->dim != dim || verifyNode->dim <= 0 || verifyNode->dim > 32767)
			{
				UnlockReleaseBuffer(buf);
				ereport(ERROR,
						(errcode(ERRCODE_DATA_CORRUPTED),
						 errmsg("hnsw: node corruption detected immediately after PageAddItem at block %u: dim=%d (expected %d)",
								blkno, verifyNode->dim, dim)));
			}
			if (verifyNode->level != level || !hnswValidateLevelSafe(verifyNode->level))
			{
				UnlockReleaseBuffer(buf);
				ereport(ERROR,
						(errcode(ERRCODE_DATA_CORRUPTED),
						 errmsg("hnsw: node level corruption detected immediately after PageAddItem at block %u: level=%d (expected %d)",
								blkno, verifyNode->level, level)));
			}
			
			elog(DEBUG1, "[HNSW_INSERT_VERIFY_SUCCESS] block=%u dim=%d level=%d verified correctly", blkno, verifyNode->dim, verifyNode->level);
		}
	}

	MarkBufferDirty(buf);
	/* Release the buffer immediately. We'll re-read it later when linking neighbors. */
	UnlockReleaseBuffer(buf);
	buf = InvalidBuffer;
	
	elog(DEBUG1, "[HNSW_INSERT_POST_PAGEADD] blkno=%u successfully added to page, buffer released", blkno);

	/* Step 5: Link neighbors at each level bidirectionally */
	{
		int			entryLevel = metaPage->entryLevel;
		int			m = metaPage->m;
		int			efConstruction = metaPage->efConstruction;

		/* Skip neighbor linking if this is the first node in the index */
		if (metaPage->entryPoint != InvalidBlockNumber && entryLevel >= 0)
		{
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

			/* Step 5a: Search for neighbors at each level */
			for (currentLevel = maxLevel; currentLevel >= 0; currentLevel--)
			{
				BlockNumber *candidates = NULL;
				float4 *candidateDistances = NULL;
				BlockNumber *selectedNeighbors = NULL;
				float4 *selectedDistances = NULL;
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
					nalloc(selectedNeighbors, BlockNumber, selectedCount);
					nalloc(selectedDistances, float4, selectedCount);
				}
				else
				{
					selectedNeighbors = NULL;
					selectedDistances = NULL;
				}

				/* Efficiently select top m neighbors using partial sort */
				if (selectedCount > 0 && candidates != NULL && candidateDistances != NULL)
				{
					/* For small arrays, use simple approach. For larger, use qsort */
					if (candidateCount <= 100)
					{
						/* Simple selection sort for small arrays - optimized */
						for (idx = 0; idx < selectedCount && idx < candidateCount; idx++)
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
					}
					else
					{
						/* For larger arrays, use qsort for better performance */
						NeighborCandidate *sortedCandidates;
						int			i;

						/* Create sorted candidate array */
						sortedCandidates = (NeighborCandidate *) palloc(candidateCount * sizeof(NeighborCandidate));
						for (i = 0; i < candidateCount; i++)
						{
							sortedCandidates[i].blk = candidates[i];
							sortedCandidates[i].dist = candidateDistances[i];
						}

						/* Sort by distance */
						qsort(sortedCandidates, candidateCount, sizeof(NeighborCandidate),
							  hnswCompareNeighborDist);

						/* Copy top m to selected arrays */
						for (idx = 0; idx < selectedCount; idx++)
						{
							selectedNeighbors[idx] = sortedCandidates[idx].blk;
							selectedDistances[idx] = sortedCandidates[idx].dist;
						}

						pfree(sortedCandidates);
					}
				}

				/* Read the buffer we just inserted to link neighbors */
				newNodeBuf = HNSW_READBUFFER(index, blkno);
				LockBuffer(newNodeBuf, BUFFER_LOCK_EXCLUSIVE);
				newNodePage = BufferGetPage(newNodeBuf);
				
				/* Verify we got the correct buffer - check block number matches */
				if (BufferGetBlockNumber(newNodeBuf) != blkno)
				{
					UnlockReleaseBuffer(newNodeBuf);
					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg("hnsw: block number mismatch: expected %u, got %u",
									blkno, BufferGetBlockNumber(newNodeBuf))));
				}
				
				if (PageIsNew(newNodePage) || PageIsEmpty(newNodePage))
				{
					UnlockReleaseBuffer(newNodeBuf);
					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg("hnsw: newly inserted page is empty at block %u (this may indicate a buffer sync issue)", blkno)));
				}
				
				{
					ItemId		itemId = PageGetItemId(newNodePage, FirstOffsetNumber);
					
					if (!ItemIdIsValid(itemId))
					{
						UnlockReleaseBuffer(newNodeBuf);
						ereport(ERROR,
								(errcode(ERRCODE_INTERNAL_ERROR),
								 errmsg("hnsw: invalid item ID at newly inserted block %u", blkno)));
					}
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

				/* Validate newNode structure before using */
				if (newNode->dim <= 0 || newNode->dim > 32767)
				{
					UnlockReleaseBuffer(newNodeBuf);
					ereport(ERROR,
							(errcode(ERRCODE_DATA_CORRUPTED),
							 errmsg("hnsw: newly inserted node at block %u has invalid dim %d (expected %d)",
									blkno, newNode->dim, dim)));
				}

				/*
				 * Link new node to neighbors, and each neighbor back
				 * (bidirectional)
				 * 
				 * Copy new node neighbor data while holding lock to prevent deadlock,
				 * then unlock newNodeBuf before processing neighbors.
				 * Relock newNodeBuf only when updating it.
				 */
				{
					BlockNumber *newNodeNeighbors = NULL;
					int16		newNodeNeighborCount = 0;

				/* Copy new node neighbor data while holding lock */
				{
					BlockNumber *tempNeighbors;
					int16		tempCount;
					
					/* Validate level bounds BEFORE array access to prevent out-of-bounds read */
					if (currentLevel < 0 || currentLevel >= HNSW_MAX_LEVEL)
					{
						elog(WARNING, "hnsw: invalid currentLevel %d in insert, skipping (max: %d)",
							 currentLevel, HNSW_MAX_LEVEL - 1);
						/* Unlock buffer and free memory before continuing */
						if (BufferIsValid(newNodeBuf))
						{
							UnlockReleaseBuffer(newNodeBuf);
							newNodeBuf = InvalidBuffer;
						}
						if (selectedNeighbors)
						{
							pfree(selectedNeighbors);
							selectedNeighbors = NULL;
						}
						if (selectedDistances)
						{
							pfree(selectedDistances);
							selectedDistances = NULL;
						}
						continue;
					}
					
					tempNeighbors = HnswGetNeighborsSafe(newNode, currentLevel, m);
					tempCount = newNode->neighborCount[currentLevel];

						/* Validate and clamp */
						tempCount = hnswValidateNeighborCount(tempCount, m, currentLevel);

						if (tempCount > 0 || selectedCount > 0)
						{
							nalloc(newNodeNeighbors, BlockNumber, m * 2);
							if (tempCount > 0)
							{
							/* Check for overflow in memcpy size calculation */
							if ((size_t) tempCount > SIZE_MAX / sizeof(BlockNumber))
							{
								ereport(ERROR,
										(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
										 errmsg("hnsw: tempCount %d too large for memcpy", tempCount)));
							}
								memcpy(newNodeNeighbors, tempNeighbors, tempCount * sizeof(BlockNumber));
							}
							newNodeNeighborCount = tempCount;
						}
					}

					/* Unlock newNodeBuf before processing neighbors to avoid deadlock */
					MarkBufferDirty(newNodeBuf);
					UnlockReleaseBuffer(newNodeBuf);
					newNodeBuf = InvalidBuffer;

					/* Process each neighbor one at a time */
					for (idx = 0; idx < selectedCount; idx++)
					{
						Buffer		neighborBuf = InvalidBuffer;
						Page		neighborPage;
						HnswNode	neighborNode;
						BlockNumber *neighborNeighbors = NULL;
						int16		neighborNeighborCount;
						int			insertPos;
						bool		needsPruning = false;
						bool		hasNeighborLock = false;
						OffsetNumber maxOff;

						/* Update new node's neighbor list (in memory) */
						if (idx < m && newNodeNeighbors)
						{
							newNodeNeighbors[idx] = selectedNeighbors[idx];
							if (idx + 1 > newNodeNeighborCount)
								newNodeNeighborCount = idx + 1;
						}

						/* Lock and update neighbor node (one at a time to avoid deadlock) */
						if (metaPage->insertedVectors >= 0 &&
							metaPage->insertedVectors < (int64) MaxBlockNumber &&
							selectedNeighbors[idx] > (BlockNumber) (metaPage->insertedVectors + 1))
						{
							elog(WARNING, "hnsw: selected neighbor block %u exceeds expected max %lld, skipping",
								 selectedNeighbors[idx], (long long) (metaPage->insertedVectors + 1));
							continue;
						}
						if (!hnswValidateBlockNumber(selectedNeighbors[idx], index))
						{
							elog(WARNING, "hnsw: invalid selected neighbor block %u at level %d, skipping",
								 selectedNeighbors[idx], currentLevel);
							continue;
						}
						neighborBuf = HNSW_READBUFFER(index, selectedNeighbors[idx]);
						LockBuffer(neighborBuf, BUFFER_LOCK_EXCLUSIVE);
						hasNeighborLock = true;
						neighborPage = BufferGetPage(neighborBuf);
						
						/* Check if page has valid items before accessing */
						if (PageIsNew(neighborPage) || PageIsEmpty(neighborPage))
						{
							UnlockReleaseBuffer(neighborBuf);
							hasNeighborLock = false;
							neighborBuf = InvalidBuffer;
							continue;
						}
						
						/* Validate page structure - check for corrupted page headers */
						maxOff = PageGetMaxOffsetNumber(neighborPage);
						if (maxOff > 1000)
						{
							UnlockReleaseBuffer(neighborBuf);
							hasNeighborLock = false;
							neighborBuf = InvalidBuffer;
							continue;
						}
						
						if (maxOff != FirstOffsetNumber && maxOff != InvalidOffsetNumber)
						{
							UnlockReleaseBuffer(neighborBuf);
							hasNeighborLock = false;
							neighborBuf = InvalidBuffer;
							continue;
						}
						
						if (maxOff < FirstOffsetNumber)
						{
							UnlockReleaseBuffer(neighborBuf);
							hasNeighborLock = false;
							neighborBuf = InvalidBuffer;
							continue;
						}
					
					{
						ItemId neighborItemId = PageGetItemId(neighborPage, FirstOffsetNumber);
						if (!ItemIdIsValid(neighborItemId) || !ItemIdHasStorage(neighborItemId))
						{
							UnlockReleaseBuffer(neighborBuf);
							hasNeighborLock = false;
							neighborBuf = InvalidBuffer;
							continue;
						}
						
					neighborNode = (HnswNode) PageGetItem(neighborPage, neighborItemId);
				}
				if (neighborNode == NULL)
				{
					UnlockReleaseBuffer(neighborBuf);
					hasNeighborLock = false;
					neighborBuf = InvalidBuffer;
					continue;
				}

					/* Check for zeroed node corruption */
					if (neighborNode->dim == 0 && neighborNode->level == 0 && neighborNode->neighborCount[0] == 0)
					{
						bool allZeros = true;
						for (int z = 0; z < HNSW_HEAPTIDS && allZeros; z++)
						{
							if (ItemPointerIsValid(&neighborNode->heaptids[z]))
								allZeros = false;
						}
						if (allZeros)
						{
							UnlockReleaseBuffer(neighborBuf);
							hasNeighborLock = false;
							neighborBuf = InvalidBuffer;
							continue;
						}
					}

					/* Validate neighbor node structure before use */
					if (neighborNode->dim <= 0 || neighborNode->dim > 32767)
					{
						elog(WARNING, "hnsw: neighbor node at block %u has invalid dim %d in insert, skipping", selectedNeighbors[idx], neighborNode->dim);
						UnlockReleaseBuffer(neighborBuf);
						hasNeighborLock = false;
						neighborBuf = InvalidBuffer;
						continue;
					}

					/* Validate neighbor node level */
					if (!hnswValidateLevelSafe(neighborNode->level))
					{
						UnlockReleaseBuffer(neighborBuf);
						hasNeighborLock = false;
						neighborBuf = InvalidBuffer;
						continue;
					}

					/* Validate level bounds BEFORE array access to prevent out-of-bounds read */
					if (currentLevel < 0 || currentLevel >= HNSW_MAX_LEVEL)
					{
						elog(WARNING, "hnsw: invalid currentLevel %d when accessing neighbor, skipping (max: %d)",
							 currentLevel, HNSW_MAX_LEVEL - 1);
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
					float4	   *neighborVectorCopy = NULL;
					int16		pruneCount = neighborNode->neighborCount[currentLevel];
					float4 *neighborDists = NULL;
					int *neighborIndices = NULL;
					BlockNumber *pruneNeighborBlocks = NULL;
					int			validPruneCount = 0;

					/* Ensure pruneCount is within valid bounds */
					if (pruneCount > m * 2)
						pruneCount = m * 2;
					if (pruneCount < 0)
						pruneCount = 0;

					/* Copy neighbor block numbers before unlocking neighborBuf to prevent deadlock */
					if (pruneCount > 0)
					{
						nalloc(pruneNeighborBlocks, BlockNumber, pruneCount);
						for (j = 0; j < pruneCount; j++)
						{
							if (neighborNeighbors[j] != InvalidBlockNumber &&
								hnswValidateBlockNumber(neighborNeighbors[j], index))
							{
								pruneNeighborBlocks[validPruneCount++] = neighborNeighbors[j];
							}
						}
					}

					/*
					 * neighborVector points into the buffer page. Since we unlock/release
					 * neighborBuf before computing distances (to avoid deadlocks), we must
					 * copy the vector data now.
					 */
					if (neighborVector != NULL && dim > 0)
					{
					/* Check for overflow in size calculation */
					if ((size_t) dim > SIZE_MAX / sizeof(float4))
					{
						ereport(ERROR,
								(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
								 errmsg("hnsw: dimension %d too large for vector copy", dim)));
					}
						nalloc(neighborVectorCopy, float4, dim);
						memcpy(neighborVectorCopy, neighborVector, dim * sizeof(float4));
					}

					/* Unlock neighborBuf before processing other neighbors for pruning */
					UnlockReleaseBuffer(neighborBuf);
					hasNeighborLock = false;
					neighborBuf = InvalidBuffer;

					/* Allocate arrays with bounds checking */
					if (validPruneCount > 0)
					{
						nalloc(neighborIndices, int, validPruneCount);
						nalloc(neighborDists, float4, validPruneCount);
					}
					else
					{
						neighborIndices = NULL;
						neighborDists = NULL;
					}

					/* Process each neighbor for distance calculation */
					
					for (j = 0; j < validPruneCount; j++)
					{
						/* Bounds check before array access */
						if (j >= 0 && j < validPruneCount)
						{
							neighborIndices[j] = j;
							if (pruneNeighborBlocks[j] == blkno)
							{
								/* Bounds check before accessing selectedDistances */
								if (idx >= 0 && idx < selectedCount)
									neighborDists[j] = selectedDistances[idx];
								else
									neighborDists[j] = FLT_MAX;
							}
							else
							{
								Buffer		otherBuf;
								Page		otherPage;
								HnswNode	otherNode;
								float4 *otherVector = NULL;
								ItemId		itemId;

								/* Bounds check before accessing pruneNeighborBlocks */
								if (j >= 0 && j < validPruneCount)
									otherBuf = HNSW_READBUFFER(index, pruneNeighborBlocks[j]);
								else
									continue;
								LockBuffer(otherBuf, BUFFER_LOCK_SHARE);
								otherPage = BufferGetPage(otherBuf);

								if (PageIsNew(otherPage) || PageIsEmpty(otherPage))
								{
									UnlockReleaseBuffer(otherBuf);
									neighborDists[j] = FLT_MAX;  /* Mark as invalid */
									continue;
								}

								/* Check if page has valid items before accessing */
								if (PageGetMaxOffsetNumber(otherPage) < FirstOffsetNumber)
								{
									UnlockReleaseBuffer(otherBuf);
									neighborDists[j] = FLT_MAX;  /* Mark as invalid */
									continue;
								}

								itemId = PageGetItemId(otherPage, FirstOffsetNumber);
						if (!ItemIdIsValid(itemId) || !ItemIdHasStorage(itemId))
						{
							UnlockReleaseBuffer(otherBuf);
							neighborDists[j] = FLT_MAX;  /* Mark as invalid */
							continue;
						}

						otherNode = (HnswNode) PageGetItem(otherPage, itemId);
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

						neighborDists[j] = hnswComputeDistance(neighborVectorCopy ? neighborVectorCopy : neighborVector,
															  otherVector, dim, 1);
						UnlockReleaseBuffer(otherBuf);
					}
				}
			}

			/* Sort by distance with bounds checking */
					if (validPruneCount > 1 && neighborDists != NULL && neighborIndices != NULL)
					{
						for (j = 0; j < validPruneCount - 1; j++)
						{
							int			k;

							/* Bounds check before accessing arrays */
							if (j < 0 || j >= validPruneCount)
								break;
							
							for (k = j + 1; k < validPruneCount; k++)
							{
								/* Bounds check before accessing arrays */
								if (k < 0 || k >= validPruneCount)
									break;
								
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
					}

					/* Relock neighborBuf to write pruned list */
					{
						ItemId		itemId;

						if (!hnswValidateBlockNumber(selectedNeighbors[idx], index))
						{
							elog(WARNING, "hnsw: invalid selected neighbor block %u during prune relock, skipping",
								 selectedNeighbors[idx]);
							pfree(neighborDists);
							pfree(neighborIndices);
							if (pruneNeighborBlocks)
								pfree(pruneNeighborBlocks);
							if (neighborVectorCopy)
								pfree(neighborVectorCopy);
							continue;
						}
						neighborBuf = HNSW_READBUFFER(index, selectedNeighbors[idx]);
						LockBuffer(neighborBuf, BUFFER_LOCK_EXCLUSIVE);
						hasNeighborLock = true;
						neighborPage = BufferGetPage(neighborBuf);
						
						/* Check if page has valid items before accessing */
						if (PageIsNew(neighborPage) || PageIsEmpty(neighborPage) ||
							PageGetMaxOffsetNumber(neighborPage) < FirstOffsetNumber)
						{
							UnlockReleaseBuffer(neighborBuf);
							hasNeighborLock = false;
							neighborBuf = InvalidBuffer;
							pfree(neighborDists);
							pfree(neighborIndices);
							if (pruneNeighborBlocks)
								pfree(pruneNeighborBlocks);
							if (neighborVectorCopy)
								pfree(neighborVectorCopy);
							continue;
						}
						
						itemId = PageGetItemId(neighborPage, FirstOffsetNumber);
						if (!ItemIdIsValid(itemId) || !ItemIdHasStorage(itemId))
						{
							UnlockReleaseBuffer(neighborBuf);
							hasNeighborLock = false;
							neighborBuf = InvalidBuffer;
							pfree(neighborDists);
							pfree(neighborIndices);
							if (pruneNeighborBlocks)
								pfree(pruneNeighborBlocks);
							if (neighborVectorCopy)
								pfree(neighborVectorCopy);
							continue;
						}
						
						neighborNode = (HnswNode) PageGetItem(neighborPage, itemId);
						if (neighborNode == NULL)
						{
							UnlockReleaseBuffer(neighborBuf);
							hasNeighborLock = false;
							neighborBuf = InvalidBuffer;
							pfree(neighborDists);
							pfree(neighborIndices);
							if (pruneNeighborBlocks)
								pfree(pruneNeighborBlocks);
							if (neighborVectorCopy)
								pfree(neighborVectorCopy);
							continue;
						}

						/* Validate neighbor node structure before use */
						if (neighborNode->dim <= 0 || neighborNode->dim > 32767)
						{
							elog(WARNING, "hnsw: neighbor node at block %u has invalid dim %d during prune, skipping", selectedNeighbors[idx], neighborNode->dim);
							UnlockReleaseBuffer(neighborBuf);
							hasNeighborLock = false;
							neighborBuf = InvalidBuffer;
							pfree(neighborDists);
							pfree(neighborIndices);
							if (pruneNeighborBlocks)
								pfree(pruneNeighborBlocks);
							if (neighborVectorCopy)
								pfree(neighborVectorCopy);
							continue;
						}

						if (hnswValidateLevelSafe(neighborNode->level))
						{
							/* Assert that selectedNeighbors[idx] is still valid */
							if (selectedNeighbors[idx] >= RelationGetNumberOfBlocks(index))
							{
								elog(WARNING, "hnsw: selectedNeighbors[%d] = %u exceeds index size %u",
									 idx, selectedNeighbors[idx], RelationGetNumberOfBlocks(index));
								UnlockReleaseBuffer(neighborBuf);
								hasNeighborLock = false;
								neighborBuf = InvalidBuffer;
								pfree(neighborDists);
								pfree(neighborIndices);
								if (pruneNeighborBlocks)
									pfree(pruneNeighborBlocks);
								if (neighborVectorCopy)
									pfree(neighborVectorCopy);
								continue;
							}
						}
						
						{
							int finalCount;
						
							neighborNeighbors = HnswGetNeighborsSafe(neighborNode, currentLevel, m);
							if (neighborNeighbors == NULL)
							{
								UnlockReleaseBuffer(neighborBuf);
								hasNeighborLock = false;
								neighborBuf = InvalidBuffer;
								pfree(neighborDists);
								pfree(neighborIndices);
								if (pruneNeighborBlocks)
									pfree(pruneNeighborBlocks);
								if (neighborVectorCopy)
									pfree(neighborVectorCopy);
								ereport(ERROR,
										(errcode(ERRCODE_INTERNAL_ERROR),
										 errmsg("hnsw: HnswGetNeighborsSafe returned NULL")));
							}
							
							finalCount = Min(m * 2, validPruneCount);
							neighborNode->neighborCount[currentLevel] = finalCount;
							
							/* Bounds check before accessing arrays */
							for (j = 0; j < finalCount && j < m * 2; j++)
							{
								/* Validate neighborIndices[j] is within bounds */
								if (neighborIndices[j] >= 0 && neighborIndices[j] < validPruneCount)
								{
									neighborNeighbors[j] = pruneNeighborBlocks[neighborIndices[j]];
								}
								else
								{
									elog(WARNING, "hnsw: neighborIndices[%d]=%d out of bounds (validPruneCount=%d), using InvalidBlockNumber",
										 j, neighborIndices[j], validPruneCount);
									neighborNeighbors[j] = InvalidBlockNumber;
								}
							}
							/* Clear remaining slots */
							for (j = finalCount; j < m * 2; j++)
								neighborNeighbors[j] = InvalidBlockNumber;
							MarkBufferDirty(neighborBuf);
							UnlockReleaseBuffer(neighborBuf);
							hasNeighborLock = false;
							neighborBuf = InvalidBuffer;
						}
					}

					pfree(neighborDists);
					neighborDists = NULL;
					pfree(neighborIndices);
					neighborIndices = NULL;
					if (pruneNeighborBlocks)
						pfree(pruneNeighborBlocks);
					if (neighborVectorCopy)
						pfree(neighborVectorCopy);
				}
				else
		{
			/* No pruning needed - unlock neighborBuf since we locked it earlier */
			if (hasNeighborLock)
			{
				UnlockReleaseBuffer(neighborBuf);
				hasNeighborLock = false;
				neighborBuf = InvalidBuffer;
			}
		}

					/* Assert invariant: neighborBuf should not be locked at end of loop */
					if (hasNeighborLock)
					{
						elog(WARNING, "hnsw: neighborBuf still locked at end of neighbor loop, unlocking");
						UnlockReleaseBuffer(neighborBuf);
						hasNeighborLock = false;
						neighborBuf = InvalidBuffer;
					}
				}

		/* Write newNodeNeighbors back to the new node page before freeing */
		if (newNodeNeighbors && newNodeNeighborCount > 0)
					{
						/* Relock newNodeBuf to write neighbors back */
						newNodeBuf = HNSW_READBUFFER(index, blkno);
						if (!BufferIsValid(newNodeBuf))
						{
							pfree(newNodeNeighbors);
							ereport(ERROR,
									(errcode(ERRCODE_INTERNAL_ERROR),
									 errmsg("hnsw: ReadBuffer failed for new node block %u", blkno)));
						}
						LockBuffer(newNodeBuf, BUFFER_LOCK_EXCLUSIVE);
						newNodePage = BufferGetPage(newNodeBuf);
						
					/* Reload node from page */
					newNode = (HnswNode) PageGetItem(newNodePage,
													 PageGetItemId(newNodePage, FirstOffsetNumber));
					if (newNode == NULL)
					{
						UnlockReleaseBuffer(newNodeBuf);
						pfree(newNodeNeighbors);
						ereport(ERROR,
								(errcode(ERRCODE_INTERNAL_ERROR),
								 errmsg("hnsw: failed to reload new node at block %u", blkno)));
					}
					
					/* Validate newNode structure after reload */
					if (newNode->dim <= 0 || newNode->dim > 32767)
					{
						UnlockReleaseBuffer(newNodeBuf);
						pfree(newNodeNeighbors);
						ereport(ERROR,
								(errcode(ERRCODE_DATA_CORRUPTED),
								 errmsg("hnsw: reloaded node at block %u has invalid dim %d (expected %d)",
										blkno, newNode->dim, dim)));
					}
					
					/* Write neighbor count and neighbors array */
					newNode->neighborCount[currentLevel] = newNodeNeighborCount;
					{
						BlockNumber *neighbors = HnswGetNeighborsSafe(newNode, currentLevel, m);
						/* Check for overflow in memcpy size calculation */
						if (newNodeNeighborCount > 0 && (size_t) newNodeNeighborCount > SIZE_MAX / sizeof(BlockNumber))
						{
							ereport(ERROR,
									(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
									 errmsg("hnsw: newNodeNeighborCount %d too large for memcpy", newNodeNeighborCount)));
						}
							memcpy(neighbors, newNodeNeighbors, newNodeNeighborCount * sizeof(BlockNumber));
						}
						
						MarkBufferDirty(newNodeBuf);
						UnlockReleaseBuffer(newNodeBuf);
						newNodeBuf = InvalidBuffer;
					}

					/* Free newNodeNeighbors after writing back */
					if (newNodeNeighbors)
					{
						pfree(newNodeNeighbors);
						newNodeNeighbors = NULL;
					}
				}

			if (selectedNeighbors)
			{
				pfree(selectedNeighbors);
				selectedNeighbors = NULL;
			}
			if (selectedDistances)
			{
				pfree(selectedDistances);
				selectedDistances = NULL;
			}

			if (candidates)
			{
				pfree(candidates);
				candidates = NULL;
			}
			if (candidateDistances)
			{
				pfree(candidateDistances);
				candidateDistances = NULL;
			}

			/* newNodeBuf should already be unlocked at this point */
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

	pfree(node);
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
		buf = HNSW_READBUFFER(index, blkno);
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

			if (!ItemIdIsValid(itemId) || !ItemIdHasStorage(itemId))
				continue;

			node = (HnswNode) PageGetItem(page, itemId);

			/* Check if this node matches the ItemPointer (check all TIDs for HOT support) */
			{
				bool		found = false;
				int			tidIdx;
				
				/* Validate heaptidsLength to prevent out-of-bounds access */
				int validTidLength = (node->heaptidsLength > HNSW_HEAPTIDS) ? HNSW_HEAPTIDS : node->heaptidsLength;
				
				for (tidIdx = 0; tidIdx < validTidLength && tidIdx < HNSW_HEAPTIDS; tidIdx++)
				{
					if (ItemPointerEquals(&node->heaptids[tidIdx], tid))
					{
						found = true;
						break;
					}
				}
				
				/* Backwards compatibility: check heaptids[0] if length is 0 */
				if (!found && node->heaptidsLength == 0 && ItemPointerIsValid(&node->heaptids[0]))
				{
					if (ItemPointerEquals(&node->heaptids[0], tid))
						found = true;
				}
				
				if (found)
				{
					*outBlkno = blkno;
					*outOffset = offnum;
					UnlockReleaseBuffer(buf);
					return true;
				}
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
	BlockNumber *neighbors = NULL;
	int16		neighborCount;
	int			i,
				j;
	bool		found = false;
	int			m;

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

	buf = HNSW_READBUFFER(index, neighborBlkno);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	page = BufferGetPage(buf);

	/* Get first item on page (assuming one node per page for simplicity) */
	{
		ItemId		itemId;

		if (PageIsNew(page) || PageIsEmpty(page))
		{
			UnlockReleaseBuffer(buf);
			return;
		}

		/* Check if page has valid items before accessing */
		if (PageGetMaxOffsetNumber(page) < FirstOffsetNumber)
		{
			UnlockReleaseBuffer(buf);
			return;
		}

		itemId = PageGetItemId(page, FirstOffsetNumber);
		if (!ItemIdIsValid(itemId) || !ItemIdHasStorage(itemId))
		{
			UnlockReleaseBuffer(buf);
			return;
		}

		neighbor = (HnswNode) PageGetItem(page, itemId);
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

			metaBuf = HNSW_READBUFFER(index, 0);
			LockBuffer(metaBuf, BUFFER_LOCK_SHARE);
			metaPage = BufferGetPage(metaBuf);
			meta = (HnswMetaPage) PageGetContents(metaPage);
			m = meta->m;
			UnlockReleaseBuffer(metaBuf);

			/* Validate level bounds BEFORE array access to prevent out-of-bounds read */
			if (level < 0 || level >= HNSW_MAX_LEVEL)
			{
				elog(WARNING, "hnsw: invalid level %d in removeNodeFromNeighbor, skipping (max: %d)",
					 level, HNSW_MAX_LEVEL - 1);
				UnlockReleaseBuffer(buf);
				return;
			}
			
			neighbors = HnswGetNeighborsSafe(neighbor, level, m);
			neighborCount = neighbor->neighborCount[level];

			/* Validate and clamp neighborCount */
			neighborCount = hnswValidateNeighborCount(neighborCount, m, level);
		}

		for (i = 0; i < neighborCount; i++)
		{
			if (neighbors[i] == nodeBlkno)
			{
				int maxNeighbors;
				
				found = true;
				/* Shift remaining neighbors with bounds checking */
				maxNeighbors = m * 2;
				for (j = i; j < neighborCount - 1 && j + 1 < maxNeighbors; j++)
				{
					/* Bounds check before accessing neighbors[j + 1] */
					if (j + 1 < neighborCount && j + 1 < maxNeighbors)
						neighbors[j] = neighbors[j + 1];
					else
						break;
				}
				/* Bounds check before writing to last position */
				if (neighborCount > 0 && neighborCount - 1 < maxNeighbors)
					neighbors[neighborCount - 1] = InvalidBlockNumber;
				/* Validate level bounds BEFORE array write to prevent out-of-bounds write */
				if (level >= 0 && level < HNSW_MAX_LEVEL)
				{
					neighbor->neighborCount[level]--;
				}
				else
				{
					elog(WARNING, "hnsw: invalid level %d when decrementing neighborCount, skipping write (max: %d)",
						 level, HNSW_MAX_LEVEL - 1);
				}
				break;
			}
		}

		if (found)
		{
			MarkBufferDirty(buf);
		}

		UnlockReleaseBuffer(buf);
	}
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
	BlockNumber *neighbors = NULL;
	int16		neighborCount;

	if (!hnswFindNodeByTid(index, tid, &nodeBlkno, &nodeOffset))
	{
		/* Node not found - already deleted or never existed */
		return true;
	}

	/* Read metadata */
	metaBuffer = HNSW_READBUFFER(index, 0);
	LockBuffer(metaBuffer, BUFFER_LOCK_EXCLUSIVE);
	metaPage = BufferGetPage(metaBuffer);
	meta = (HnswMetaPage) PageGetContents(metaPage);

	/* Read the node to be deleted */
	nodeBuf = HNSW_READBUFFER(index, nodeBlkno);
	LockBuffer(nodeBuf, BUFFER_LOCK_EXCLUSIVE);
	nodePage = BufferGetPage(nodeBuf);
	
	/* Check if page has valid items before accessing */
	if (PageIsNew(nodePage) || PageIsEmpty(nodePage))
	{
		elog(WARNING, "hnsw: page %u is new or empty, cannot delete node at offset %u", nodeBlkno, nodeOffset);
		UnlockReleaseBuffer(nodeBuf);
		UnlockReleaseBuffer(metaBuffer);
		return false;
	}
	
	if (PageGetMaxOffsetNumber(nodePage) < nodeOffset)
	{
		elog(WARNING, "hnsw: offset %u exceeds max offset on page %u", nodeOffset, nodeBlkno);
		UnlockReleaseBuffer(nodeBuf);
		UnlockReleaseBuffer(metaBuffer);
		return false;
	}
	
	{
		ItemId deleteItemId = PageGetItemId(nodePage, nodeOffset);
		if (!ItemIdIsValid(deleteItemId) || !ItemIdHasStorage(deleteItemId))
		{
			elog(WARNING, "hnsw: invalid ItemId at offset %u on page %u", nodeOffset, nodeBlkno);
			UnlockReleaseBuffer(nodeBuf);
			UnlockReleaseBuffer(metaBuffer);
			return false;
		}
		
		node = (HnswNode) PageGetItem(nodePage, deleteItemId);
	}
	
	if (node == NULL)
	{
		elog(WARNING, "hnsw: failed to read node at offset %u on page %u", nodeOffset, nodeBlkno);
		UnlockReleaseBuffer(nodeBuf);
		UnlockReleaseBuffer(metaBuffer);
		return false;
	}
	
	/* Validate node structure before use */
	if (node->dim <= 0 || node->dim > 32767)
	{
		elog(WARNING, "hnsw: node at offset %u on page %u has invalid dim %d", nodeOffset, nodeBlkno, node->dim);
		UnlockReleaseBuffer(nodeBuf);
		UnlockReleaseBuffer(metaBuffer);
		return false;
	}

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
		/* Validate level bounds BEFORE array access to prevent out-of-bounds read */
		if (level < 0 || level >= HNSW_MAX_LEVEL)
		{
			elog(WARNING, "hnsw: invalid level %d in delete node cleanup, skipping (max: %d)",
				 level, HNSW_MAX_LEVEL - 1);
			continue;
		}
		
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
			/* Validate level bounds BEFORE array access to prevent out-of-bounds read */
			if (level < 0 || level >= HNSW_MAX_LEVEL)
			{
				elog(WARNING, "hnsw: invalid level %d when finding new entry point, skipping (max: %d)",
					 level, HNSW_MAX_LEVEL - 1);
				continue;
			}
			
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

					neighborBuf = HNSW_READBUFFER(index, neighbors[i]);
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

	/* Call replication hook if replication is enabled */
	if (neurondb_replication_enabled())
	{
		neurondb_hnsw_replication_hook(index, nodeBlkno, false);
	}

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

/*
 * HOT Update Support Functions
 */

/*
 * hnswAddHeapTid - Add a heap TID to a node's heap TID array
 *
 * If the TID already exists, does nothing. If the array is full,
 * returns false. Otherwise adds the TID and returns true.
 */
static void
hnswAddHeapTid(HnswNode node, ItemPointer heaptid)
{
	int			i;

	/* Validate heap TID before storing - prevent storing invalid TIDs that could cause VACUUM issues */
	if (!ItemPointerIsValid(heaptid))
	{
		elog(WARNING, "hnsw: attempt to add invalid heap TID, skipping");
		return;
	}

	/* Validate heaptidsLength to prevent out-of-bounds access */
	if (node->heaptidsLength > HNSW_HEAPTIDS)
	{
		elog(WARNING, "hnsw: corrupted heaptidsLength %d (max: %d), clamping to %d",
			 node->heaptidsLength, HNSW_HEAPTIDS, HNSW_HEAPTIDS);
		node->heaptidsLength = HNSW_HEAPTIDS;
	}
	
	/* Check if already exists */
	for (i = 0; i < node->heaptidsLength && i < HNSW_HEAPTIDS; i++)
	{
		if (ItemPointerEquals(&node->heaptids[i], heaptid))
			return;					/* Already exists, no-op */
	}

	/* Check if array is full */
	if (node->heaptidsLength >= HNSW_HEAPTIDS)
	{
		ereport(WARNING,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("hnsw: heap TID array full (%d), cannot add more TIDs",
						HNSW_HEAPTIDS)));
		return;
	}

	/* Add to array */
	ItemPointerCopy(heaptid, &node->heaptids[node->heaptidsLength]);
	node->heaptidsLength++;
}

/*
 * hnswNodeHasHeapTid - Check if a node has a specific heap TID
 */
static bool __attribute__((unused))
hnswNodeHasHeapTid(HnswNode node, ItemPointer heaptid)
{
	int			i;
	int			validLength;

	/* Validate heaptidsLength to prevent out-of-bounds access */
	if (node->heaptidsLength > HNSW_HEAPTIDS)
	{
		elog(WARNING, "hnsw: corrupted heaptidsLength %d (max: %d), clamping to %d",
			 node->heaptidsLength, HNSW_HEAPTIDS, HNSW_HEAPTIDS);
		validLength = HNSW_HEAPTIDS;
	}
	else
	{
		validLength = node->heaptidsLength;
	}

	for (i = 0; i < validLength && i < HNSW_HEAPTIDS; i++)
	{
		if (ItemPointerEquals(&node->heaptids[i], heaptid))
			return true;
	}

	return false;
}

/*
 * hnswIsNodeCompatible - Check if a node structure is compatible with version
 *
 * For version 1 nodes, heaptidsLength would be uninitialized (0), and
 * heaptids[0] would contain what was previously heapPtr.
 * For version 2+ nodes, heaptidsLength should be >= 1.
 */
static bool __attribute__((unused))
hnswIsNodeCompatible(HnswNode node, uint32 version)
{
	if (version >= 2)
	{
		/* Version 2+ should have at least one heap TID */
		return node->heaptidsLength > 0 && node->heaptidsLength <= HNSW_HEAPTIDS;
	}
	else
	{
		/* Version 1 nodes: heaptidsLength would be 0, but heaptids[0] should be valid */
		/* For backwards compatibility, we treat version 1 nodes as having 1 TID */
		return true;
	}
}

