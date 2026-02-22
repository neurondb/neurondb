/*-------------------------------------------------------------------------
 *
 * ivf_am.c
 *		IVF (Inverted File) Index Access Method with KMeans clustering
 *
 * This implements a complete IVF index as a PostgreSQL IndexAM with:
 * - KMeans clustering for centroid computation
 * - Inverted list construction and maintenance
 * - Multi-probe search with nprobe parameter
 * - Dynamic centroid assignment
 *
 * Based on the paper:
 * "Product Quantization for Nearest Neighbor Search" by Jégou et al. (2011)
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/index/ivf_am.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_types.h"
#include "fmgr.h"
#include "access/amapi.h"
#include "access/generic_xlog.h"
#include "access/reloptions.h"
#include "access/relscan.h"
#include "access/tableam.h"
#include "catalog/pg_type.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "utils/varbit.h"
#include "utils/lsyscache.h"
#include "parser/parse_type.h"
#include "nodes/parsenodes.h"
#include "nodes/makefuncs.h"
#include <math.h>
#include <float.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "access/parallel.h"

/* Forward declarations for type conversion */
extern float fp16_to_float(uint16 fp16);

/* IVF parameters */
#define IVF_DEFAULT_NLISTS 100	/* Number of clusters/centroids */
#define IVF_DEFAULT_NPROBE 10	/* Number of lists to probe */
#define IVF_MAX_ITERATIONS 50	/* KMeans max iterations */
#define IVF_CONVERGENCE_THRESHOLD 0.001 /* KMeans convergence */

/*
 * IVF index options
 */
typedef struct IvfOptions
{
	int			nlists;			/* Number of clusters */
	int			nprobe;			/* Number of lists to probe */
}			IvfOptions;

/* Reloption kind - registered in _PG_init() */
extern int	relopt_kind_ivf;

/*
 * IVF metadata page (block 0)
 */
typedef struct IvfMetaPageData
{
	uint32		magicNumber;
	uint32		version;
	int			nlists;			/* Number of inverted lists */
	int			nprobe;			/* Default nprobe */
	int			dim;			/* Vector dimension */
	BlockNumber centroidsBlock; /* Block containing centroids */
	int64		insertedVectors;
}			IvfMetaPageData;

typedef IvfMetaPageData * IvfMetaPage;

#define IVF_MAGIC_NUMBER 0x49564646 /* "IVFF" in hex */
#define IVF_VERSION 1

/*
 * Centroid data (stored in dedicated page(s))
 */
typedef struct IvfCentroidData
{
	int			listId;			/* Inverted list ID */
	int			dim;			/* Vector dimension */
	int64		memberCount;	/* Vectors in this list */
	BlockNumber firstBlock;		/* First block of inverted list */
	/* Followed by float4 centroid[dim] */
}			IvfCentroidData;

typedef IvfCentroidData * IvfCentroid;

#define IvfGetCentroidVector(centroid) \
	((float4 *)((char *)(centroid) + MAXALIGN(sizeof(IvfCentroidData))))

/*
 * Type conversion helpers for multi-type support (same as HNSW)
 */

/*
 * Extract vector data from any supported type (vector, halfvec, sparsevec, bit)
 * Returns allocated float4 array and dimension via out_dim
 * Caller must pfree the result
 */
static float4 *
ivfExtractVectorData(Datum value, Oid typeOid, int *out_dim, MemoryContext ctx)
{
	float4 *result = NULL;
	int			i;
	MemoryContext oldctx;
	Oid			vectorOid,
				halfvecOid,
				sparsevecOid,
				bitOid;

	if (out_dim == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("ivf: out_dim cannot be NULL")));

	/* Type OIDs are cached at index build time in build state */
	bitOid = BITOID;			/* PostgreSQL built-in type */

	/* Get type OIDs - use LookupTypeNameOid for proper namespace lookup */
	{
		List *names = NULL;

		names = list_make2(makeString("public"), makeString("vector"));
		vectorOid = LookupTypeNameOid(NULL, makeTypeNameFromNameList(names), false);
		list_free(names);

		names = list_make2(makeString("public"), makeString("halfvec"));
		halfvecOid = LookupTypeNameOid(NULL, makeTypeNameFromNameList(names), false);
		list_free(names);

		names = list_make2(makeString("public"), makeString("sparsevec"));
		sparsevecOid = LookupTypeNameOid(NULL, makeTypeNameFromNameList(names), false);
		list_free(names);
	}

	oldctx = MemoryContextSwitchTo(ctx);

	/* Check type and extract accordingly */
	if (typeOid == vectorOid)
	{
		Vector	   *v = DatumGetVector(value);

		*out_dim = v->dim;
		nalloc(result, float4, v->dim);
		for (i = 0; i < v->dim; i++)
			result[i] = v->data[i];
	}
	else if (typeOid == halfvecOid)
	{
		VectorF16  *hv = (VectorF16 *) PG_DETOAST_DATUM(value);

		*out_dim = hv->dim;
		nalloc(result, float4, hv->dim);
		for (i = 0; i < hv->dim; i++)
			result[i] = fp16_to_float(hv->data[i]);
	}
	else if (typeOid == sparsevecOid)
	{
		VectorMap  *sv = (VectorMap *) PG_DETOAST_DATUM(value);
		int32	   *indices = VECMAP_INDICES(sv);
		float4	   *values = VECMAP_VALUES(sv);

		*out_dim = sv->total_dim;
		nalloc(result, float4, sv->total_dim);
		/* Zero-initialize the result array */
		memset(result, 0, sv->total_dim * sizeof(float4));
		/* Populate non-zero entries with comprehensive bounds checking */
		for (i = 0; i < sv->nnz; i++)
		{
			/* Validate index is within bounds - error on out-of-bounds to prevent silent corruption */
			if (indices[i] < 0 || indices[i] >= sv->total_dim)
			{
				ereport(ERROR,
						(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
						 errmsg("ivf: sparsevec index %d out of bounds (dim=%d, nnz=%d)",
								indices[i], sv->total_dim, sv->nnz)));
			}
			result[indices[i]] = values[i];
		}
	}
	else if (typeOid == bitOid)
	{
		VarBit	   *bit_vec = (VarBit *) PG_DETOAST_DATUM(value);
		int			nbits = VARBITLEN(bit_vec);
		bits8	   *bit_data = VARBITS(bit_vec);

		*out_dim = nbits;
		nalloc(result, float4, nbits);
		for (i = 0; i < nbits; i++)
		{
			int			byte_idx = i / BITS_PER_BYTE;
			int			bit_idx = i % BITS_PER_BYTE;
			int			bit_val = (bit_data[byte_idx] >> (BITS_PER_BYTE - 1 - bit_idx)) & 1;

			result[i] = bit_val ? 1.0f : -1.0f;
		}
	}
	else
	{
		MemoryContextSwitchTo(oldctx);
		ereport(ERROR,
				(errcode(ERRCODE_DATATYPE_MISMATCH),
				 errmsg("ivf: unsupported type OID %u", typeOid)));
		return NULL;			/* not reached */
	}

	MemoryContextSwitchTo(oldctx);
	return result;
}

/*
 * Get type OID from index key attribute
 */
static Oid
ivfGetKeyType(Relation index, int attno)
{
	TupleDesc	indexDesc = RelationGetDescr(index);
	Form_pg_attribute attr;

	if (attno < 1 || attno > indexDesc->natts)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("ivf: invalid attribute number %d", attno)));

	attr = TupleDescAttr(indexDesc, attno - 1);
	return attr->atttypid;
}

/*
 * Inverted list page header (stored in special space)
 */
typedef struct IvfListPageHeader
{
	BlockNumber nextBlock;		/* Next block in chain, InvalidBlockNumber if
								 * last */
	int32		entryCount;		/* Number of entries on this page */
}			IvfListPageHeader;

/*
 * Inverted list entry
 */
typedef struct IvfListEntryData
{
	ItemPointerData heapPtr;
	int16		dim;
	/* Followed by float4 vector[dim] */
}			IvfListEntryData;

typedef IvfListEntryData * IvfListEntry;

#define IvfGetListPageHeader(page) \
	((IvfListPageHeader *)PageGetSpecialPointer(page))

/*
 * IVF scan opaque state
 */
typedef struct IvfScanOpaqueData
{
	Vector	   *queryVector;	/* Query vector */
	int			strategy;		/* Distance strategy (1=L2, 2=Cosine, etc.) */
	int			nprobe;			/* Number of clusters to probe */
	int			k;				/* Number of results to return */
	bool		firstCall;		/* First call to gettuple */
	int			resultCount;	/* Number of results found */
	ItemPointerData *results;	/* Result heap TIDs */
	float4	   *distances;		/* Result distances */
	int			currentResult;	/* Current result index */
	int		   *selectedClusters;
	int			currentCluster; /* Current cluster being scanned */
	BlockNumber currentListBlock;	/* Current list block */
	int			currentListOffset;	/* Current offset in list */
	/* Iterative scan support */
	bool		iterativeScanEnabled;
	int			iterativeScanMode;	/* 0=off, 1=strict_order, 2=relaxed_order */
	int			maxProbes;			/* Maximum probes for iterative scan */
	int			initialNprobe;		/* Initial nprobe value */
}			IvfScanOpaqueData;

typedef IvfScanOpaqueData * IvfScanOpaque;

/*
 * KMeans clustering state
 */
typedef struct KMeansState
{
	int			k;				/* Number of clusters */
	int			dim;			/* Vector dimension */
	int			maxIter;		/* Max iterations */
	float4		threshold;		/* Convergence threshold */

	/* Centroids */
	float4	  **centroids;		/* k x dim */
	int		   *assignments;	/* Vector assignments */
	int		   *counts;			/* Points per cluster */

	/* Data */
	float4	  **data;			/* n x dim training data */
	int			n;				/* Number of data points */

	MemoryContext ctx;
}			KMeansState;

/* Forward declarations */
static IndexBuildResult *
ivfbuild(Relation heap, Relation index, struct IndexInfo *indexInfo);
static void ivfbuildempty(Relation index);
static bool ivfinsert(Relation index,
					  Datum * values,
					  bool *isnull,
					  ItemPointer ht_ctid,
					  Relation heapRel,
					  IndexUniqueCheck checkUnique,
					  bool indexUnchanged,
					  struct IndexInfo *indexInfo);
static IndexBulkDeleteResult * ivfbulkdelete(IndexVacuumInfo * info,
											 IndexBulkDeleteResult * stats,
											 IndexBulkDeleteCallback callback,
											 void *callback_state);
static IndexBulkDeleteResult * ivfvacuumcleanup(IndexVacuumInfo * info,
												IndexBulkDeleteResult * stats);
static bool ivfdelete(Relation index,
					  ItemPointer tid,
					  Datum * values,
					  bool *isnull,
					  Relation heapRel,
					  struct IndexInfo *indexInfo) __attribute__((unused));
static bool ivfupdate(Relation index,
					  ItemPointer tid,
					  Datum * values,
					  bool *isnull,
					  ItemPointer otid,
					  Relation heapRel,
					  struct IndexInfo *indexInfo) __attribute__((unused));
static void ivfcostestimate(struct PlannerInfo *root,
							struct IndexPath *path,
							double loop_count,
							Cost * indexStartupCost,
							Cost * indexTotalCost,
							Selectivity * indexSelectivity,
							double *indexCorrelation,
							double *indexPages);
static bytea * ivfoptions(Datum reloptions, bool validate);
static bool ivfproperty(Oid index_oid,
						int attno,
						IndexAMProperty prop,
						const char *propname,
						bool *res,
						bool *isnull);
static IndexScanDesc ivfbeginscan(Relation index, int nkeys, int norderbys);
static void ivfrescan(IndexScanDesc scan,
					  ScanKey keys,
					  int nkeys,
					  ScanKey orderbys,
					  int norderbys);
static bool ivfgettuple(IndexScanDesc scan, ScanDirection dir);
static void ivfendscan(IndexScanDesc scan);

/* Build callback */
static void ivfBuildCallback(Relation index,
							 ItemPointer tid,
							 Datum * values,
							 bool *isnull,
							 bool tupleIsAlive,
							 void *state);

/* KMeans helper functions */
static KMeansState * kmeans_init(int k, int dim, float4 * *data, int n);
static void kmeans_run(KMeansState * state);
static void kmeans_assign(KMeansState * state);
static void kmeans_update_centroids(KMeansState * state);
static float4 kmeans_compute_cost(KMeansState * state);
static void kmeans_free(KMeansState * state);
static float4 vector_distance_l2(const float4 * v1, const float4 * v2, int dim);
static int	find_nearest_centroid(KMeansState * state, const float4 * vector);

/*
 * SQL-callable handler
 */
PG_FUNCTION_INFO_V1(ivf_handler);

Datum
ivf_handler(PG_FUNCTION_ARGS)
{
	IndexAmRoutine *amroutine = makeNode(IndexAmRoutine);

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
	/*
	 * Parallel build not implemented: amestimateparallelscan, aminitparallelscan,
	 * and amparallelrescan callbacks would be required. Set to false until
	 * parallel build is implemented.
	 */
	amroutine->amcanparallel = false;
	amroutine->amcaninclude = false;
	amroutine->amusemaintenanceworkmem = false;
	amroutine->amsummarizing = false;
	amroutine->amparallelvacuumoptions = 0;
	amroutine->amkeytype = InvalidOid;

	amroutine->ambuild = ivfbuild;
	amroutine->ambuildempty = ivfbuildempty;
	amroutine->aminsert = ivfinsert;
	amroutine->ambulkdelete = ivfbulkdelete;
	amroutine->amvacuumcleanup = ivfvacuumcleanup;
	amroutine->amcanreturn = NULL;
	amroutine->amcostestimate = ivfcostestimate;
	amroutine->amoptions = ivfoptions;
	amroutine->amproperty = ivfproperty;
	amroutine->ambuildphasename = NULL;
	amroutine->amvalidate = NULL;
	amroutine->amadjustmembers = NULL;
	amroutine->ambeginscan = ivfbeginscan;
	amroutine->amrescan = ivfrescan;
	amroutine->amgettuple = ivfgettuple;
	amroutine->amgetbitmap = NULL;
	amroutine->amendscan = ivfendscan;
	amroutine->ammarkpos = NULL;
	amroutine->amrestrpos = NULL;
	/* Parallel scan callbacks - not needed for parallel build (handled automatically) */
	amroutine->amestimateparallelscan = NULL;
	amroutine->aminitparallelscan = NULL;
	amroutine->amparallelrescan = NULL;

	PG_RETURN_POINTER(amroutine);
}

/*
 * Build IVF index
 */
/*
 * Build callback: Collects vectors for KMeans sampling
 */
typedef struct IvfBuildState
{
	Relation	heap;
	Relation	index;
	IndexInfo  *indexInfo;
	MemoryContext tmpCtx;
	double		indtuples;
	float4	  **sampleVectors;	/* Sampled vectors for KMeans */
	int			sampleCount;
	int			maxSamples;
	int			dim;
	Oid			keyType;
}			IvfBuildState;

static void
ivfBuildCallback(Relation index,
				 ItemPointer tid,
				 Datum * values,
				 bool *isnull,
				 bool tupleIsAlive,
				 void *state)
{
	IvfBuildState *buildstate = (IvfBuildState *) state;
	float4	   *vectorData = NULL;
	int			dim;

	if (isnull[0])
		return;

	/* Extract vector data */
	vectorData = ivfExtractVectorData(values[0],
									  buildstate->keyType,
									  &dim,
									  buildstate->tmpCtx);

	if (vectorData == NULL)
		return;

	/* Store dimension on first vector */
	if (buildstate->dim == 0)
		buildstate->dim = dim;

	/* Sample vectors for KMeans (up to maxSamples) */
	if (buildstate->sampleCount < buildstate->maxSamples)
	{
		buildstate->sampleVectors[buildstate->sampleCount] =
			(float4 *) MemoryContextAlloc(buildstate->tmpCtx,
										  dim * sizeof(float4));
		memcpy(buildstate->sampleVectors[buildstate->sampleCount],
			   vectorData,
			   dim * sizeof(float4));
		buildstate->sampleCount++;
	}

	buildstate->indtuples++;
	pfree(vectorData);
}

static IndexBuildResult *
ivfbuild(Relation heap, Relation index, struct IndexInfo *indexInfo)
{
	IndexBuildResult *result = NULL;
	IvfBuildState buildstate;
	IvfOptions *options = NULL;
	Buffer		metaBuffer;
	Page		metaPage;
	IvfMetaPage meta;
	KMeansState *kmeans = NULL;
	int			nlists;
	int			i;
	BlockNumber centroidsBlock;
	Buffer		centroidsBuf;
	Page		centroidsPage;
	Size		centroidSize;
	OffsetNumber offnum;


	/* Initialize build state */
	memset(&buildstate, 0, sizeof(buildstate));
	buildstate.heap = heap;
	buildstate.index = index;
	buildstate.indexInfo = indexInfo;
	buildstate.tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
											  "IVF build temporary context",
											  ALLOCSET_DEFAULT_SIZES);
	buildstate.keyType = ivfGetKeyType(index, 1);

	/* Get index options */
	options = (IvfOptions *) indexInfo->ii_AmCache;
	if (options == NULL)
	{
		IvfOptions opts;
		IvfOptions *opts_ptr = NULL;

		/* Safely get options from index->rd_options */
		/* Validate rd_options before accessing to prevent NULL pointer dereference */
		if (index->rd_options != NULL)
		{
			/* index->rd_options is a bytea created by build_reloptions */
			/* Validate that it's large enough and properly formatted */
			Size bytea_size = VARSIZE(index->rd_options);
			Size expected_size = VARHDRSZ + MAXALIGN(sizeof(IvfOptions));
			
			/* Only access if size is reasonable */
			if (bytea_size >= expected_size)
			{
				bytea *opts_bytea = index->rd_options;
				opts_ptr = (IvfOptions *) VARDATA(opts_bytea);
				
				/* Validate pointer is not NULL and structure is valid */
				if (opts_ptr != NULL)
				{
					/* Copy structure safely */
					opts.nlists = opts_ptr->nlists;
					opts.nprobe = opts_ptr->nprobe;
					
					/* Validate values are reasonable */
					if (opts.nlists <= 0)
						opts.nlists = IVF_DEFAULT_NLISTS;
					if (opts.nprobe <= 0)
						opts.nprobe = IVF_DEFAULT_NPROBE;
				}
				else
				{
					/* Use defaults if pointer is NULL */
					opts.nlists = IVF_DEFAULT_NLISTS;
					opts.nprobe = IVF_DEFAULT_NPROBE;
				}
			}
			else
			{
				/* Size mismatch - use defaults */
				opts.nlists = IVF_DEFAULT_NLISTS;
				opts.nprobe = IVF_DEFAULT_NPROBE;
			}
		}
		else
		{
			/* Use defaults during index build if rd_options is NULL */
			opts.nlists = IVF_DEFAULT_NLISTS;
			opts.nprobe = IVF_DEFAULT_NPROBE;
		}

		nalloc(options, IvfOptions, 1);
		*options = opts;
		indexInfo->ii_AmCache = (void *) options;
	}
	nlists = options ? options->nlists : IVF_DEFAULT_NLISTS;

	/* Initialize metadata page on block 0 */
	/* Use P_NEW with RBM_NORMAL to extend file and create block 0 */
	metaBuffer = ReadBufferExtended(index, MAIN_FORKNUM, P_NEW, RBM_NORMAL, NULL);
	if (!BufferIsValid(metaBuffer))
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ReadBuffer for block 0 failed")));
	}
	LockBuffer(metaBuffer, BUFFER_LOCK_EXCLUSIVE);
	metaPage = BufferGetPage(metaBuffer);
	/* Metadata goes in page contents, not special space */
	PageInit(metaPage, BufferGetPageSize(metaBuffer), 0);
	meta = (IvfMetaPage) PageGetContents(metaPage);
	meta->magicNumber = IVF_MAGIC_NUMBER;
	meta->version = IVF_VERSION;
	meta->nlists = nlists;
	meta->nprobe = options ? options->nprobe : IVF_DEFAULT_NPROBE;
	meta->dim = 0;				/* Will be set after sampling */
	meta->centroidsBlock = InvalidBlockNumber;
	meta->insertedVectors = 0;

	MarkBufferDirty(metaBuffer);
	UnlockReleaseBuffer(metaBuffer);

	/* Step 1: Sample vectors from heap for KMeans */
	buildstate.maxSamples = Min(10000, nlists * 100);	/* Sample up to 10k or
														 * nlists*100 */
	buildstate.sampleVectors = (float4 * *) MemoryContextAlloc(buildstate.tmpCtx,
															   buildstate.maxSamples * sizeof(float4 *));
	buildstate.sampleCount = 0;

	/* Scan heap and collect sample vectors */
	buildstate.indtuples = table_index_build_scan(heap,
												  index,
												  indexInfo,
												  true, /* allow_sync */
												  true, /* progress */
												  ivfBuildCallback,
											  (void *) &buildstate,
											  NULL);

	/* Allow nlists to be up to sampleCount, but warn if it's too high */
	if (buildstate.sampleCount < nlists)
	{
		/* If user explicitly set nlists > sampleCount, use sampleCount instead */
		if (nlists > buildstate.sampleCount && buildstate.sampleCount > 0)
		{
			ereport(WARNING,
					(errmsg("ivf: reducing nlists from %d to %d (not enough sample vectors)",
							nlists, buildstate.sampleCount)));
			nlists = buildstate.sampleCount;
		}
		else if (buildstate.sampleCount == 0)
		{
			/* Empty table - create empty index */
			ereport(WARNING,
					(errmsg("ivf: building empty index (no sample vectors available)")));
			nlists = 1; /* Use minimum nlists for empty index */
		}
		else
		{
			ereport(ERROR,
					(errcode(ERRCODE_INSUFFICIENT_RESOURCES),
					 errmsg("ivf: not enough sample vectors (%d < %d)",
							buildstate.sampleCount,
							nlists)));
		}
	}

	/* Set dimension in metadata - re-read block 0 */
	metaBuffer = ReadBuffer(index, 0);
	if (!BufferIsValid(metaBuffer))
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ReadBuffer failed")));
	}
	LockBuffer(metaBuffer, BUFFER_LOCK_EXCLUSIVE);
	meta = (IvfMetaPage) PageGetContents(BufferGetPage(metaBuffer));
	meta->dim = buildstate.dim;
	MarkBufferDirty(metaBuffer);
	UnlockReleaseBuffer(metaBuffer);

	/* Step 2: Run KMeans clustering */
	kmeans = kmeans_init(nlists,
						 buildstate.dim,
						 buildstate.sampleVectors,
						 buildstate.sampleCount);
	if (kmeans == NULL)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("ivf: kmeans_init failed")));
	}
	kmeans_run(kmeans);

	/* Validate kmeans state */
	if (kmeans->centroids == NULL)
	{
		kmeans_free(kmeans);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("ivf: kmeans centroids are NULL")));
	}

	/* Step 3: Store centroids in dedicated page(s) */
	centroidsBuf = ReadBufferExtended(index, MAIN_FORKNUM, P_NEW, RBM_NORMAL, NULL);
	if (!BufferIsValid(centroidsBuf))
	{
		kmeans_free(kmeans);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ReadBuffer failed")));
	}
	LockBuffer(centroidsBuf, BUFFER_LOCK_EXCLUSIVE);
	centroidsPage = BufferGetPage(centroidsBuf);
	centroidsBlock = BufferGetBlockNumber(centroidsBuf);
	PageInit(centroidsPage,
			 BufferGetPageSize(centroidsBuf),
			 0);

	/* Centroid size calculation must match IvfGetCentroidVector macro:
	 * The macro uses: (char*)(centroid) + MAXALIGN(sizeof(IvfCentroidData))
	 * So total size is: MAXALIGN(header) + vector_data
	 */
	centroidSize = MAXALIGN(sizeof(IvfCentroidData)) + 
				   buildstate.dim * sizeof(float4);

	for (i = 0; i < nlists; i++)
	{
		IvfCentroidData *centroid = NULL;
		char *centroid_raw = NULL;

		if (kmeans->centroids[i] == NULL)
		{
			UnlockReleaseBuffer(centroidsBuf);
			kmeans_free(kmeans);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("ivf: centroid %d is NULL", i)));
		}

		centroid_raw = (char *) palloc0(centroidSize);
		centroid = (IvfCentroidData *) centroid_raw;

		centroid->listId = i;
		centroid->dim = buildstate.dim;
		centroid->memberCount = 0;
		centroid->firstBlock = InvalidBlockNumber;

		/* Copy centroid vector (vector data starts after aligned header) */
		memcpy(IvfGetCentroidVector(centroid),
			   kmeans->centroids[i],
			   buildstate.dim * sizeof(float4));

		/* Add to page */
		offnum = PageAddItem(centroidsPage,
							 (Item) centroid,
							 centroidSize,
							 InvalidOffsetNumber,
							 false,
							 false);
		
		if (offnum == InvalidOffsetNumber)
		{
			/* Current page is full, allocate a new page */
			MarkBufferDirty(centroidsBuf);
			UnlockReleaseBuffer(centroidsBuf);

			/* Allocate new page for remaining centroids */
			centroidsBuf = ReadBufferExtended(index, MAIN_FORKNUM, P_NEW, RBM_NORMAL, NULL);
			if (!BufferIsValid(centroidsBuf))
			{
				pfree(centroid_raw);
				kmeans_free(kmeans);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: ReadBuffer failed for centroid overflow page")));
			}
			LockBuffer(centroidsBuf, BUFFER_LOCK_EXCLUSIVE);
			centroidsPage = BufferGetPage(centroidsBuf);
			PageInit(centroidsPage,
					 BufferGetPageSize(centroidsBuf),
					 0);

			/* Try again on the new page */
			offnum = PageAddItem(centroidsPage,
								 (Item) centroid,
								 centroidSize,
								 InvalidOffsetNumber,
								 false,
								 false);
			if (offnum == InvalidOffsetNumber)
			{
				/* Still failed - centroid too large for even an empty page */
				pfree(centroid_raw);
				UnlockReleaseBuffer(centroidsBuf);
				kmeans_free(kmeans);
				ereport(ERROR,
						(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
						 errmsg("ivf: centroid size (%zu bytes) exceeds page size", centroidSize),
						 errhint("Reduce vector dimensionality or use a different index type")));
			}
		}

		/*
		 * PageAddItem copies the data, so we can free the temporary
		 * allocation
		 */
		pfree(centroid_raw);
	}

	MarkBufferDirty(centroidsBuf);
	UnlockReleaseBuffer(centroidsBuf);

	/* Update metadata with centroids block - re-read block 0 */
	metaBuffer = ReadBuffer(index, 0);
	if (!BufferIsValid(metaBuffer))
	{
		kmeans_free(kmeans);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ReadBuffer failed")));
	}
	LockBuffer(metaBuffer, BUFFER_LOCK_EXCLUSIVE);
	meta = (IvfMetaPage) PageGetContents(BufferGetPage(metaBuffer));
	meta->centroidsBlock = centroidsBlock;
	MarkBufferDirty(metaBuffer);
	UnlockReleaseBuffer(metaBuffer);

	/* Step 4 & 5: Assign all vectors to lists and build inverted lists */
	/* This would require a second pass through the heap */
	/* For now, we mark the index as built with centroids ready */
	/* Actual list building happens during inserts */

	/* Clean up kmeans (allocated in CurrentMemoryContext, not tmpCtx) */
	kmeans_free(kmeans);

	/* Clean up */
	MemoryContextDelete(buildstate.tmpCtx);

	/* Create result */
	nalloc(result, IndexBuildResult, 1);
	result->heap_tuples = buildstate.indtuples;
	result->index_tuples = buildstate.indtuples;	/* Simplified */

	return result;
}

/*
 * Build empty index
 */
static void
ivfbuildempty(Relation index)
{
	Buffer		buf;
	Page		page;
	IvfMetaPage meta;
	IvfOptions *options = NULL;
	int			nlists = IVF_DEFAULT_NLISTS;
	int			nprobe = IVF_DEFAULT_NPROBE;

	/* Get index options */
	if (index->rd_options != NULL)
	{
		/* rd_options is a varlena (bytea) containing IvfOptions in VARDATA */
		options = (IvfOptions *) VARDATA(index->rd_options);
		if (options != NULL)
		{
			nlists = (options->nlists > 0) ? options->nlists : IVF_DEFAULT_NLISTS;
			nprobe = (options->nprobe > 0) ? options->nprobe : IVF_DEFAULT_NPROBE;
		}
	}

	/* Initialize metadata page on block 0 */
	buf = ReadBuffer(index, 0);
	if (!BufferIsValid(buf))
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ReadBuffer failed")));
	}
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

	page = BufferGetPage(buf);
	if (PageIsNew(page))
		/* Metadata goes in page contents, not special space */
		PageInit(page, BufferGetPageSize(buf), 0);

	meta = (IvfMetaPage) PageGetContents(page);
	memset(meta, 0, sizeof(IvfMetaPageData));

	meta->magicNumber = IVF_MAGIC_NUMBER;
	meta->version = IVF_VERSION;
	meta->nlists = nlists;
	meta->nprobe = nprobe;
	meta->dim = 0;
	meta->centroidsBlock = InvalidBlockNumber;
	meta->insertedVectors = 0;

	MarkBufferDirty(buf);
	UnlockReleaseBuffer(buf);
}

/*
 * Insert into IVF index
 */
static bool
ivfinsert(Relation index,
		  Datum * values,
		  bool *isnull,
		  ItemPointer ht_ctid,
		  Relation heapRel,
		  IndexUniqueCheck checkUnique,
		  bool indexUnchanged,
		  struct IndexInfo *indexInfo)
{
	Vector	   *input_vec = NULL;
	BlockNumber meta_blkno = 0;
	Buffer		meta_buf;
	IvfMetaPageData *meta = NULL;
	BlockNumber centroidsBlock = InvalidBlockNumber;  /* Copied from meta page */
	int			i,
				min_idx = 0,
				nlist;
	float4		min_dist = FLT_MAX;
	float4		dist;

	if (isnull[0])
		return false;			/* don't insert NULLs */

	/* Extract vector data (handles vector, halfvec, sparsevec, bit) */
	{
		float4 *vectorData = NULL;
		int			dim;
		Oid			keyType;
		MemoryContext oldctx;
		char *input_vec_raw = NULL;

		/* Get key type from index */
		keyType = ivfGetKeyType(index, 1);

		/* Extract vector data */
		oldctx = MemoryContextSwitchTo(CurrentMemoryContext);
		vectorData = ivfExtractVectorData(values[0], keyType, &dim, CurrentMemoryContext);
		MemoryContextSwitchTo(oldctx);

		if (vectorData == NULL)
			return false;

		/* Allocate Vector structure for compatibility */
		nalloc(input_vec_raw, char, VECTOR_SIZE(dim));
		input_vec = (Vector *) input_vec_raw;
		SET_VARSIZE(input_vec, VECTOR_SIZE(dim));
		input_vec->dim = dim;
		memcpy(input_vec->data, vectorData, dim * sizeof(float4));
		pfree(vectorData);
	}

	/*
	 * Step 1: Read IVF metadata and centroids
	 */
	meta_buf = ReadBuffer(index, meta_blkno);
	if (!BufferIsValid(meta_buf))
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ReadBuffer failed")));
	}
	LockBuffer(meta_buf, BUFFER_LOCK_SHARE);
	meta = (IvfMetaPageData *) PageGetContents(BufferGetPage(meta_buf));
	
	/* Copy all needed fields from meta page into locals before unlocking */
	centroidsBlock = meta->centroidsBlock;
	nlist = meta->nlists;

	if (nlist <= 0)
	{
		UnlockReleaseBuffer(meta_buf);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("IVF index has no centroids (nlists=%d)", nlist)));
		return false;
	}

	UnlockReleaseBuffer(meta_buf);
	meta_buf = InvalidBuffer;  /* Prevent accidental reuse */
	meta = NULL;  /* Prevent accidental reuse */

	/*
	 * Step 2: Find nearest centroid by L2 distance
	 */
	if (centroidsBlock != InvalidBlockNumber)
	{
		Buffer		centroidsBuf;
		Page		centroidsPage;
		OffsetNumber maxoff;
		OffsetNumber offnum;
		IvfCentroid centroid;
		float4 *centroidVector = NULL;
		float4		accum;
		int			k;

		centroidsBuf = ReadBuffer(index, centroidsBlock);
		if (!BufferIsValid(centroidsBuf))
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: ReadBuffer failed")));
		}
		LockBuffer(centroidsBuf, BUFFER_LOCK_SHARE);
		centroidsPage = BufferGetPage(centroidsBuf);

		if (PageIsNew(centroidsPage) || PageIsEmpty(centroidsPage))
		{
			UnlockReleaseBuffer(centroidsBuf);
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("IVF index centroids page is empty")));
			return false;
		}

		maxoff = PageGetMaxOffsetNumber(centroidsPage);
		for (i = 0; i < nlist && i < maxoff; i++)
		{
			offnum = FirstOffsetNumber + i;
			if (offnum > maxoff)
				break;

			centroid = (IvfCentroid) PageGetItem(centroidsPage,
												 PageGetItemId(centroidsPage, offnum));

			if (centroid->dim != input_vec->dim)
				continue;

			centroidVector = IvfGetCentroidVector(centroid);

			/* Compute L2 distance */
			accum = 0.0f;
			for (k = 0; k < input_vec->dim; k++)
			{
				float4		diff = input_vec->data[k] - centroidVector[k];

				accum += diff * diff;
			}
			dist = sqrtf(accum);

			if (dist < min_dist)
			{
				min_dist = dist;
				min_idx = i;
			}
		}

		UnlockReleaseBuffer(centroidsBuf);
	}
	else
	{
		/* No centroids block - cannot assign vector */
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("IVF index has no centroids block")));
		return false;
	}

	/*
	 * Step 3: Append to selected inverted list
	 */
	{
		Buffer		centroidsBuf;
		Page		centroidsPage;
		IvfCentroid centroid;
		OffsetNumber offnum;
		BlockNumber listBlock;
		Buffer		listBuf;
		Page		listPage;
		IvfListPageHeader *listHeader = NULL;
		Size		entrySize;
		OffsetNumber newOffnum;
		bool		needNewBlock = false;

		/* Get centroid for selected list */
		centroidsBuf = ReadBuffer(index, centroidsBlock);
		if (!BufferIsValid(centroidsBuf))
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: ReadBuffer failed")));
		}
		LockBuffer(centroidsBuf, BUFFER_LOCK_EXCLUSIVE);
		centroidsPage = BufferGetPage(centroidsBuf);
		offnum = FirstOffsetNumber + min_idx;
		centroid = (IvfCentroid) PageGetItem(centroidsPage,
											 PageGetItemId(centroidsPage, offnum));

		/* Calculate entry size */
		entrySize = MAXALIGN(sizeof(IvfListEntryData)) +
			MAXALIGN(input_vec->dim * sizeof(float4));

		/* Find last block in chain or create first block */
		listBlock = centroid->firstBlock;
		if (listBlock == InvalidBlockNumber)
		{
			/* Create first block for this list */
			listBlock = P_NEW;
			needNewBlock = true;
		}
		else
		{
			/* Traverse to last block in chain */
			Buffer		tempBuf;
			Page		tempPage;
			IvfListPageHeader *tempHeader = NULL;

			tempBuf = ReadBuffer(index, listBlock);
			if (!BufferIsValid(tempBuf))
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: ReadBuffer failed")));
			}
			LockBuffer(tempBuf, BUFFER_LOCK_SHARE);
			tempPage = BufferGetPage(tempBuf);
			tempHeader = IvfGetListPageHeader(tempPage);

			while (tempHeader->nextBlock != InvalidBlockNumber)
			{
				BlockNumber nextBlock = tempHeader->nextBlock;

				UnlockReleaseBuffer(tempBuf);
				tempBuf = ReadBuffer(index, nextBlock);
				if (!BufferIsValid(tempBuf))
				{
					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg("neurondb: ReadBuffer failed")));
				}
				LockBuffer(tempBuf, BUFFER_LOCK_SHARE);
				tempPage = BufferGetPage(tempBuf);
				tempHeader = IvfGetListPageHeader(tempPage);
				listBlock = nextBlock;
			}

			UnlockReleaseBuffer(tempBuf);
		}

		/* Get or create list block */
		if (needNewBlock)
		{
			listBuf = ReadBuffer(index, listBlock);
			if (!BufferIsValid(listBuf))
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: ReadBuffer failed")));
			}
			LockBuffer(listBuf, BUFFER_LOCK_EXCLUSIVE);
			listPage = BufferGetPage(listBuf);
			PageInit(listPage, BufferGetPageSize(listBuf), sizeof(IvfListPageHeader));
			listHeader = IvfGetListPageHeader(listPage);
			listHeader->nextBlock = InvalidBlockNumber;
			listHeader->entryCount = 0;

			/* Update centroid to point to first block */
			centroid->firstBlock = BufferGetBlockNumber(listBuf);
		}
		else
		{
			listBuf = ReadBuffer(index, listBlock);
			if (!BufferIsValid(listBuf))
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: ReadBuffer failed")));
			}
			LockBuffer(listBuf, BUFFER_LOCK_EXCLUSIVE);
			listPage = BufferGetPage(listBuf);
			listHeader = IvfGetListPageHeader(listPage);

			/* Check if page has space, otherwise allocate new block */
			if (PageGetFreeSpace(listPage) < entrySize)
			{
				/* Allocate new block and chain it */
				BlockNumber newBlock = P_NEW;
				Buffer		newBuf;
				Page		newPage;
				IvfListPageHeader *newHeader = NULL;

				newBuf = ReadBuffer(index, newBlock);
				if (!BufferIsValid(newBuf))
				{
					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg("neurondb: ReadBuffer failed")));
				}
				LockBuffer(newBuf, BUFFER_LOCK_EXCLUSIVE);
				newPage = BufferGetPage(newBuf);
				PageInit(newPage, BufferGetPageSize(newBuf), sizeof(IvfListPageHeader));
				newHeader = IvfGetListPageHeader(newPage);
				newHeader->nextBlock = InvalidBlockNumber;
				newHeader->entryCount = 0;

				/* Chain new block */
				listHeader->nextBlock = BufferGetBlockNumber(newBuf);
				MarkBufferDirty(listBuf);
				UnlockReleaseBuffer(listBuf);

				/* Use new block */
				listBuf = newBuf;
				listPage = newPage;
				listHeader = newHeader;
				listBlock = BufferGetBlockNumber(listBuf);
			}
		}

		/* Construct entry in temporary buffer */
		{
			char *entryData = NULL;
			IvfListEntry tempEntry;
			float4 *entryVector = NULL;
			nalloc(entryData, char, entrySize);
			tempEntry = (IvfListEntry) entryData;

			ItemPointerCopy(ht_ctid, &tempEntry->heapPtr);
			tempEntry->dim = input_vec->dim;
			entryVector = (float4 *) (entryData + MAXALIGN(sizeof(IvfListEntryData)));
			memcpy(entryVector, input_vec->data, input_vec->dim * sizeof(float4));

			/* Append entry to page */
			newOffnum = PageAddItem(listPage,
									entryData,
									entrySize,
									InvalidOffsetNumber,
									0,
									false);

			pfree(entryData);
		}

		if (newOffnum == InvalidOffsetNumber)
		{
			UnlockReleaseBuffer(listBuf);
			UnlockReleaseBuffer(centroidsBuf);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("ivfinsert: failed to add entry to list page")));
			return false;
		}

		/* Update page header */
		listHeader->entryCount++;
		MarkBufferDirty(listBuf);
		UnlockReleaseBuffer(listBuf);

		/* Update centroid member count */
		centroid->memberCount++;
		MarkBufferDirty(centroidsBuf);
		UnlockReleaseBuffer(centroidsBuf);

		/* Update metadata */
		meta_buf = ReadBuffer(index, 0);
		if (!BufferIsValid(meta_buf))
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: ReadBuffer failed")));
		}
		LockBuffer(meta_buf, BUFFER_LOCK_EXCLUSIVE);
		meta = (IvfMetaPageData *) PageGetContents(BufferGetPage(meta_buf));
		meta->insertedVectors++;
		MarkBufferDirty(meta_buf);
		UnlockReleaseBuffer(meta_buf);
	}


	pfree(input_vec);
	return true;
}

/*
 * Bulk delete: scan all inverted lists and remove entries based on callback
 */
static IndexBulkDeleteResult *
ivfbulkdelete(IndexVacuumInfo * info,
			  IndexBulkDeleteResult * stats,
			  IndexBulkDeleteCallback callback,
			  void *callback_state)
{
	Relation	index = info->index;
	Buffer		metaBuf;
	Page		metaPage;
	IvfMetaPage meta;
	Buffer		centroidsBuf;
	Page		centroidsPage;
	OffsetNumber maxoff;
	OffsetNumber offnum;
	IvfCentroid centroid;
		BlockNumber listBlock;
		Buffer		listBuf;
		Page		listPage;
		IvfListPageHeader *listHeader = NULL;
		OffsetNumber listMaxoff;
	OffsetNumber listOffnum;
	IvfListEntry entry;
	ItemId		itemId;
	int			i;
	int			tuplesRemoved = 0;
	int			tuplesRemovedThisPage = 0;

	IndexBulkDeleteResult *new_stats = NULL;

	if (stats == NULL)
	{
		nalloc(new_stats, IndexBulkDeleteResult, 1);
		stats = new_stats;
	}

	/* Read metadata */
	metaBuf = ReadBuffer(index, 0);
	if (!BufferIsValid(metaBuf))
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ReadBuffer failed")));
	}
	LockBuffer(metaBuf, BUFFER_LOCK_SHARE);
	metaPage = BufferGetPage(metaBuf);
	meta = (IvfMetaPage) PageGetContents(metaPage);

	if (meta->centroidsBlock == InvalidBlockNumber)
	{
		UnlockReleaseBuffer(metaBuf);
		return stats;
	}

	/* Read centroids */
	centroidsBuf = ReadBuffer(index, meta->centroidsBlock);
	if (!BufferIsValid(centroidsBuf))
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ReadBuffer failed")));
	}
	LockBuffer(centroidsBuf, BUFFER_LOCK_SHARE);
	centroidsPage = BufferGetPage(centroidsBuf);
	maxoff = PageGetMaxOffsetNumber(centroidsPage);

	/* Scan all centroids and their lists */
	for (i = 0; i < meta->nlists && i < maxoff; i++)
	{
		int			removedFromThisList = 0;

		offnum = FirstOffsetNumber + i;
		if (offnum > maxoff)
			break;

		centroid = (IvfCentroid) PageGetItem(centroidsPage,
											 PageGetItemId(centroidsPage, offnum));

		if (centroid->firstBlock == InvalidBlockNumber)
			continue;

		/* Traverse all blocks in this list's chain */
		listBlock = centroid->firstBlock;
		while (listBlock != InvalidBlockNumber)
		{
			listBuf = ReadBuffer(index, listBlock);
			if (!BufferIsValid(listBuf))
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: ReadBuffer failed")));
			}
			LockBuffer(listBuf, BUFFER_LOCK_EXCLUSIVE);
			listPage = BufferGetPage(listBuf);
			listHeader = IvfGetListPageHeader(listPage);
			listMaxoff = PageGetMaxOffsetNumber(listPage);
			tuplesRemovedThisPage = 0;

			/* Scan all entries on this page */
			for (listOffnum = FirstOffsetNumber; listOffnum <= listMaxoff;
				 listOffnum = OffsetNumberNext(listOffnum))
			{
				itemId = PageGetItemId(listPage, listOffnum);

				if (!ItemIdIsValid(itemId) || ItemIdIsDead(itemId))
					continue;

				if (!ItemIdHasStorage(itemId))
					continue;

				entry = (IvfListEntry) PageGetItem(listPage, itemId);

				/* Check callback to see if this tuple should be deleted */
				if (callback(&entry->heapPtr, callback_state))
				{
					/* Mark as deleted */
					ItemIdSetDead(itemId);
					tuplesRemovedThisPage++;
					tuplesRemoved++;
				}
			}

			/* Update page header if entries were removed */
			if (tuplesRemovedThisPage > 0)
			{
				listHeader->entryCount -= tuplesRemovedThisPage;
				if (listHeader->entryCount < 0)
					listHeader->entryCount = 0;
				removedFromThisList += tuplesRemovedThisPage;
				MarkBufferDirty(listBuf);
			}

			/* Move to next block in chain */
			listBlock = listHeader->nextBlock;
			UnlockReleaseBuffer(listBuf);
		}

		/* Update centroid member count if entries were removed from this list */
		if (removedFromThisList > 0)
		{
			/* Release SHARE lock and reacquire EXCLUSIVE */
			UnlockReleaseBuffer(centroidsBuf);

			centroidsBuf = ReadBuffer(index, meta->centroidsBlock);
			if (!BufferIsValid(centroidsBuf))
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: ReadBuffer failed")));
			}
			LockBuffer(centroidsBuf, BUFFER_LOCK_EXCLUSIVE);
			centroidsPage = BufferGetPage(centroidsBuf);
			centroid = (IvfCentroid) PageGetItem(centroidsPage,
												 PageGetItemId(centroidsPage, offnum));
			centroid->memberCount -= removedFromThisList;
			if (centroid->memberCount < 0)
				centroid->memberCount = 0;
			MarkBufferDirty(centroidsBuf);
			UnlockReleaseBuffer(centroidsBuf);

			/* Reacquire SHARE lock for next iteration */
			centroidsBuf = ReadBuffer(index, meta->centroidsBlock);
			if (!BufferIsValid(centroidsBuf))
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: ReadBuffer failed")));
			}
			LockBuffer(centroidsBuf, BUFFER_LOCK_SHARE);
			centroidsPage = BufferGetPage(centroidsBuf);
		}
	}

	UnlockReleaseBuffer(centroidsBuf);

	/* Update metadata */
	LockBuffer(metaBuf, BUFFER_LOCK_EXCLUSIVE);
	meta = (IvfMetaPage) PageGetContents(BufferGetPage(metaBuf));
	meta->insertedVectors -= tuplesRemoved;
	if (meta->insertedVectors < 0)
		meta->insertedVectors = 0;
	MarkBufferDirty(metaBuf);
	UnlockReleaseBuffer(metaBuf);

	/* Update stats */
	stats->tuples_removed = tuplesRemoved;
	stats->num_index_tuples = meta->insertedVectors;

	return stats;
}

static IndexBulkDeleteResult *
ivfvacuumcleanup(IndexVacuumInfo * info, IndexBulkDeleteResult * stats)
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
ivfcostestimate(struct PlannerInfo *root,
				struct IndexPath *path,
				double loop_count,
				Cost * indexStartupCost,
				Cost * indexTotalCost,
				Selectivity * indexSelectivity,
				double *indexCorrelation,
				double *indexPages)
{
	*indexStartupCost = 25.0;
	*indexTotalCost = 50.0;
	*indexSelectivity = 0.01;
	*indexCorrelation = 0.0;
	*indexPages = 5;
}

static bytea *
ivfoptions(Datum reloptions, bool validate)
{
	static const relopt_parse_elt tab[] = {
		{"lists", RELOPT_TYPE_INT, offsetof(IvfOptions, nlists)},
		{"probes", RELOPT_TYPE_INT, offsetof(IvfOptions, nprobe)}
	};
	IvfOptions *opts = NULL;
	bytea *result = NULL;

	/* Handle NULL reloptions safely */
	if (reloptions == (Datum) 0 || reloptions == PointerGetDatum(NULL))
		reloptions = (Datum) 0;

	result = (bytea *) build_reloptions(reloptions, validate, relopt_kind_ivf,
										sizeof(IvfOptions), tab, lengthof(tab));
	
	/* WORKAROUND: build_reloptions appears to match by position, not by name.
	 * Manually parse reloptions array and correct the values. */
	if (result != NULL && reloptions != (Datum) 0 && DatumGetPointer(reloptions))
	{
		ArrayType *arr = DatumGetArrayTypeP(reloptions);
		if (arr != NULL && ARR_NDIM(arr) == 1 && ARR_ELEMTYPE(arr) == TEXTOID)
		{
			Datum *elems = NULL;
			bool *nulls = NULL;
			int nelems = 0;
			int parsed_nlists = 0;
			int parsed_nprobe = 0;
			bool found_nlists = false;
			bool found_nprobe = false;
			
			deconstruct_array(arr, TEXTOID, -1, false, 'i', &elems, &nulls, &nelems);
			
			/* Parse each element to extract parameter name and value */
			{
				int			i;
				
				for (i = 0; i < nelems; i++)
				{
					if (!nulls[i] && elems[i] != (Datum) 0)
					{
						char *elem_str = text_to_cstring(DatumGetTextP(elems[i]));
						char *eq_pos = strchr(elem_str, '=');
						
						if (eq_pos != NULL)
						{
							char *param_name;
							char *param_value_str;
							int param_value;
							
							*eq_pos = '\0';
							param_name = elem_str;
							param_value_str = eq_pos + 1;
							param_value = atoi(param_value_str);
							
							if (strcmp(param_name, "lists") == 0)
							{
								parsed_nlists = param_value;
								found_nlists = true;
							}
							else if (strcmp(param_name, "probes") == 0)
							{
								parsed_nprobe = param_value;
								found_nprobe = true;
							}
						}
						pfree(elem_str);
					}
				}
			}
			
			/* Free deconstruct_array outputs */
			if (elems)
				pfree(elems);
			if (nulls)
				pfree(nulls);
			
			/* Apply the correctly parsed values */
			opts = (IvfOptions *) VARDATA(result);
			if (found_nlists)
			{
				opts->nlists = parsed_nlists;
			}
			if (found_nprobe)
			{
				opts->nprobe = parsed_nprobe;
			}
		}
	}
	
	return result;
}

static bool
ivfproperty(Oid index_oid,
			int attno,
			IndexAMProperty prop,
			const char *propname,
			bool *res,
			bool *isnull)
{
	return false;
}

static IndexScanDesc
ivfbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	IvfScanOpaque so = NULL;

	scan = RelationGetIndexScan(index, nkeys, norderbys);
	nalloc(so, IvfScanOpaqueData, 1);
	so->strategy = 1;			/* Default to L2 */
	so->nprobe = IVF_DEFAULT_NPROBE;
	so->k = 10;					/* Default k */
	so->firstCall = true;
	so->resultCount = 0;
	so->currentResult = 0;
	so->currentCluster = 0;
	so->currentListBlock = InvalidBlockNumber;
	so->currentListOffset = 0;
	so->queryVector = NULL;
	so->results = NULL;
	so->distances = NULL;
	so->selectedClusters = NULL;
	/* Initialize iterative scan fields */
	so->iterativeScanEnabled = false;
	so->iterativeScanMode = 0;
	so->maxProbes = 100;
	so->initialNprobe = IVF_DEFAULT_NPROBE;

	scan->opaque = so;

	return scan;
}

static void
ivfrescan(IndexScanDesc scan,
		  ScanKey keys,
		  int nkeys,
		  ScanKey orderbys,
		  int norderbys)
{
	IvfScanOpaque so = (IvfScanOpaque) scan->opaque;
	Buffer		metaBuffer;
	Page		metaPage;
	IvfMetaPage meta;
	IvfOptions *options = NULL;

	if (so == NULL)
		return;

	/* Reset scan state */
	so->firstCall = true;
	so->currentResult = 0;
	so->resultCount = 0;
	so->currentCluster = 0;
	so->currentListBlock = InvalidBlockNumber;
	so->currentListOffset = 0;

	/* Free previous results */
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
	if (so->selectedClusters)
	{
		pfree(so->selectedClusters);
		so->selectedClusters = NULL;
	}

	/* Get strategy from orderbys */
	if (norderbys > 0)
		so->strategy = orderbys[0].sk_strategy;
	else
		so->strategy = 1;

	/* Get nprobe from index options or metadata */
	if (scan->indexRelation->rd_options != NULL)
	{
		options = (IvfOptions *) VARDATA(scan->indexRelation->rd_options);
		if (options != NULL && options->nprobe > 0)
			so->nprobe = options->nprobe;
		else
			so->nprobe = IVF_DEFAULT_NPROBE;
	}
	else
	{
		metaBuffer = ReadBuffer(scan->indexRelation, 0);
		if (!BufferIsValid(metaBuffer))
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: ReadBuffer failed")));
		}
		LockBuffer(metaBuffer, BUFFER_LOCK_SHARE);
		metaPage = BufferGetPage(metaBuffer);
		meta = (IvfMetaPage) PageGetContents(metaPage);
		so->nprobe = meta->nprobe;
		UnlockReleaseBuffer(metaBuffer);
	}

	if (so->nprobe <= 0)
		so->nprobe = IVF_DEFAULT_NPROBE;

	/* Initialize iterative scan settings from GUC */
	{
		extern int	ivf_iterative_scan;
		extern int	ivf_max_probes;
		extern int	neurondb_ivf_probes;

		/* Use neurondb.ivf_probes if set, otherwise default */
		int			probes = so->nprobe;
		if (neurondb_ivf_probes > 0)
			probes = neurondb_ivf_probes;

		if (ivf_iterative_scan > 0)
		{
			so->iterativeScanMode = ivf_iterative_scan;
			so->iterativeScanEnabled = true;
			/* maxProbes should be at least as large as initial probes */
			so->maxProbes = Max(ivf_max_probes, probes);
		}
		else
		{
			so->iterativeScanMode = 0;
			so->iterativeScanEnabled = false;
			so->maxProbes = probes;	/* Use current probes as max when iterative scan is off */
		}
		so->initialNprobe = probes;
		so->nprobe = probes;
	}

	/* Extract query vector from orderbys */
	if (norderbys > 0 && orderbys[0].sk_argument != 0)
	{
		float4 *vectorData = NULL;
		int			dim;
		Oid			queryType;
		MemoryContext oldctx;

		queryType = TupleDescAttr(scan->indexRelation->rd_att, 0)->atttypid;
		oldctx = MemoryContextSwitchTo(scan->indexRelation->rd_indexcxt);
		vectorData = ivfExtractVectorData(orderbys[0].sk_argument,
										  queryType,
										  &dim,
										  scan->indexRelation->rd_indexcxt);
		MemoryContextSwitchTo(oldctx);

		if (vectorData != NULL)
		{
			char *queryVector_raw = NULL;
			if (so->queryVector)
				pfree(so->queryVector);
			nalloc(queryVector_raw, char, VECTOR_SIZE(dim));
			so->queryVector = (Vector *) queryVector_raw;
			SET_VARSIZE(so->queryVector, VECTOR_SIZE(dim));
			so->queryVector->dim = dim;
			memcpy(so->queryVector->data, vectorData, dim * sizeof(float4));
			pfree(vectorData);
		}

		/*
		 * Extract k from orderbys if available (stored in sk_flags or
		 * similar)
		 */
		/* For now, use default or extract from scan context */
		so->k = 10;				/* Default, could be extracted from plan */
	}
}

/*
 * Helper: Compute distance between two vectors
 */
static float4
ivfComputeDistance(const float4 * vec1, const float4 * vec2, int dim, int strategy)
{
	int			i;
	float4		sum = 0.0f;
	float4		dot_product = 0.0f;
	float4		norm1 = 0.0f;
	float4		norm2 = 0.0f;

	switch (strategy)
	{
		case 1:					/* L2 */
			for (i = 0; i < dim; i++)
			{
				float4		diff = vec1[i] - vec2[i];

				sum += diff * diff;
			}
			return sqrtf(sum);

		case 2:					/* Cosine */
			for (i = 0; i < dim; i++)
			{
				dot_product += vec1[i] * vec2[i];
				norm1 += vec1[i] * vec1[i];
				norm2 += vec2[i] * vec2[i];
			}
			norm1 = sqrtf(norm1);
			norm2 = sqrtf(norm2);
			if (norm1 == 0.0f || norm2 == 0.0f)
				return 1.0f;
			return 1.0f - (dot_product / (norm1 * norm2));

		case 3:					/* Negative inner product */
			for (i = 0; i < dim; i++)
				dot_product += vec1[i] * vec2[i];
			return (float4) (-dot_product);

		default:				/* Default to L2 */
			for (i = 0; i < dim; i++)
			{
				float4		diff = vec1[i] - vec2[i];

				sum += diff * diff;
			}
			return sqrtf(sum);
	}
}

/*
 * Helper: Find nprobe closest clusters to query vector
 */
static void
ivfSelectClusters(Relation index,
				  IvfMetaPage meta,
				  const float4 * queryVector,
				  int dim,
				  int nprobe,
				  int *selectedClusters)
{
	Buffer		centroidsBuf;
	Page		centroidsPage;
	OffsetNumber maxoff;
	OffsetNumber offnum;
	IvfCentroid centroid;
	float4	   *centroidVector = NULL;
	float4	   *clusterDistances = NULL;
	int			i,
				j;
	int			nlists;

	if (meta->centroidsBlock == InvalidBlockNumber)
	{
		/* No centroids - cannot select clusters */
		for (i = 0; i < nprobe; i++)
			selectedClusters[i] = -1;
		return;
	}

	nlists = meta->nlists;
	if (nprobe > nlists)
		nprobe = nlists;

	centroidsBuf = ReadBuffer(index, meta->centroidsBlock);
	if (!BufferIsValid(centroidsBuf))
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ReadBuffer failed")));
	}
	LockBuffer(centroidsBuf, BUFFER_LOCK_SHARE);
	centroidsPage = BufferGetPage(centroidsBuf);

	if (PageIsNew(centroidsPage) || PageIsEmpty(centroidsPage))
	{
		UnlockReleaseBuffer(centroidsBuf);
		for (i = 0; i < nprobe; i++)
			selectedClusters[i] = -1;
		return;
	}

	maxoff = PageGetMaxOffsetNumber(centroidsPage);

	/* Cap nlists to actual number of centroids available */
	if (nlists > maxoff)
		nlists = maxoff;
	if (nprobe > nlists)
		nprobe = nlists;

	/* Allocate distance array and initialize all entries to FLT_MAX */
	nalloc(clusterDistances, float4, nlists);
	for (i = 0; i < nlists; i++)
		clusterDistances[i] = FLT_MAX;

	/* Compute distances to all centroids */
	for (i = 0; i < nlists && i < maxoff; i++)
	{
		offnum = FirstOffsetNumber + i;
		if (offnum > maxoff)
			break;

		centroid = (IvfCentroid) PageGetItem(centroidsPage,
											 PageGetItemId(centroidsPage, offnum));

		if (centroid->dim != dim)
		{
			clusterDistances[i] = FLT_MAX;
			continue;
		}

		centroidVector = IvfGetCentroidVector(centroid);
		clusterDistances[i] = ivfComputeDistance(queryVector,
												 centroidVector,
												 dim,
												 1);	/* Use L2 for cluster
														 * selection */
	}

	UnlockReleaseBuffer(centroidsBuf);

	/* Select nprobe closest clusters (simple selection sort) */
	for (i = 0; i < nprobe; i++)
	{
		int			bestIdx = -1;
		float4		bestDist = FLT_MAX;

		for (j = 0; j < nlists; j++)
		{
			/* Check if already selected */
			bool		alreadySelected = false;
			int			k;

			for (k = 0; k < i; k++)
			{
				if (selectedClusters[k] == j)
				{
					alreadySelected = true;
					break;
				}
			}

			if (!alreadySelected && clusterDistances[j] < bestDist)
			{
				bestDist = clusterDistances[j];
				bestIdx = j;
			}
		}

		selectedClusters[i] = bestIdx;
	}

	pfree(clusterDistances);
}

/*
 * Helper: Collect candidates from selected clusters
 */
static void
ivfCollectCandidates(Relation index,
					 IvfMetaPage meta,
					 const float4 * queryVector,
					 int dim,
					 int strategy,
					 int *selectedClusters,
					 int nprobe,
					 int k,
					 ItemPointerData * *results,
					 float4 * *distances,
					 int *resultCount)
{
	Buffer		centroidsBuf;
	Page		centroidsPage;
	OffsetNumber maxoff;
	OffsetNumber offnum;
	IvfCentroid centroid;
	ItemPointerData *candidates = NULL;
	float4	   *candidateDistances = NULL;
	int			candidateCount = 0;
	int			maxCandidates = k * 10; /* Collect more than k for better
										 * results */
	int			i,
				j;

	/* Allocate candidate arrays */
	nalloc(candidates, ItemPointerData, maxCandidates);
	nalloc(candidateDistances, float4, maxCandidates);

	centroidsBuf = ReadBuffer(index, meta->centroidsBlock);
	if (!BufferIsValid(centroidsBuf))
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ReadBuffer failed")));
	}
	LockBuffer(centroidsBuf, BUFFER_LOCK_SHARE);
	centroidsPage = BufferGetPage(centroidsBuf);
	maxoff = PageGetMaxOffsetNumber(centroidsPage);

	/* Scan each selected cluster */
	for (i = 0; i < nprobe && candidateCount < maxCandidates; i++)
	{
		int			clusterId = selectedClusters[i];

		if (clusterId < 0 || clusterId >= maxoff)
			continue;

		offnum = FirstOffsetNumber + clusterId;
		if (offnum > maxoff)
			continue;

		centroid = (IvfCentroid) PageGetItem(centroidsPage,
											 PageGetItemId(centroidsPage, offnum));

		if (centroid->firstBlock == InvalidBlockNumber)
			continue;

		/* Scan inverted list for this cluster */
		{
			BlockNumber listBlock = centroid->firstBlock;
			Buffer		listBuf;
			Page		listPage;
			IvfListPageHeader *listHeader = NULL;
			OffsetNumber listMaxoff;
			OffsetNumber listOffnum;
			IvfListEntry entry;
			float4 *entryVector = NULL;

			/* Traverse all blocks in chain */
			while (listBlock != InvalidBlockNumber && candidateCount < maxCandidates)
			{
				listBuf = ReadBuffer(index, listBlock);
				if (!BufferIsValid(listBuf))
				{
					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg("neurondb: ReadBuffer failed")));
				}
				LockBuffer(listBuf, BUFFER_LOCK_SHARE);
				listPage = BufferGetPage(listBuf);
				listHeader = IvfGetListPageHeader(listPage);

				if (!PageIsNew(listPage) && !PageIsEmpty(listPage))
				{
					listMaxoff = PageGetMaxOffsetNumber(listPage);

					for (listOffnum = FirstOffsetNumber;
						 listOffnum <= listMaxoff && candidateCount < maxCandidates;
						 listOffnum = OffsetNumberNext(listOffnum))
					{
						ItemId		itemId = PageGetItemId(listPage, listOffnum);

						if (!ItemIdIsValid(itemId) || ItemIdIsDead(itemId))
							continue;

						entry = (IvfListEntry) PageGetItem(listPage, itemId);

						if (entry->dim != dim)
							continue;

						entryVector = (float4 *) ((char *) entry + MAXALIGN(sizeof(IvfListEntryData)));

						/* Compute distance */
						candidateDistances[candidateCount] =
							ivfComputeDistance(queryVector,
											   entryVector,
											   dim,
											   strategy);
						candidates[candidateCount] = entry->heapPtr;
						candidateCount++;
					}
				}

				/* Move to next block in chain */
				listBlock = listHeader->nextBlock;
				UnlockReleaseBuffer(listBuf);
			}
		}
	}

	UnlockReleaseBuffer(centroidsBuf);

	/* Sort candidates by distance and keep top-k */
	if (candidateCount > 0)
	{
		int *indices = NULL;
		int			actualK = Min(k, candidateCount);
		int			temp;
		ItemPointerData *results_ptr = NULL;
		float4 *distances_ptr = NULL;

		/* Create index array for sorting */
		nalloc(indices, int, candidateCount);
		for (i = 0; i < candidateCount; i++)
			indices[i] = i;

		/* Simple selection sort (could use qsort for better performance) */
		for (i = 0; i < actualK; i++)
		{
			int			bestIdx = i;
			float4		bestDist = candidateDistances[indices[i]];

			for (j = i + 1; j < candidateCount; j++)
			{
				if (candidateDistances[indices[j]] < bestDist)
				{
					bestDist = candidateDistances[indices[j]];
					bestIdx = j;
				}
			}

			if (bestIdx != i)
			{
				temp = indices[i];
				indices[i] = indices[bestIdx];
				indices[bestIdx] = temp;
			}
		}

		/* Allocate result arrays */
		nalloc(results_ptr, ItemPointerData, actualK);
		nalloc(distances_ptr, float4, actualK);
		*results = results_ptr;
		*distances = distances_ptr;

		/* Copy top-k results */
		for (i = 0; i < actualK; i++)
		{
			(*results)[i] = candidates[indices[i]];
			(*distances)[i] = candidateDistances[indices[i]];
		}

		*resultCount = actualK;

		pfree(indices);
	}
	else
	{
		*results = NULL;
		*distances = NULL;
		*resultCount = 0;
	}

	pfree(candidates);
	pfree(candidateDistances);
}

static bool
ivfgettuple(IndexScanDesc scan, ScanDirection dir)
{
	IvfScanOpaque so = (IvfScanOpaque) scan->opaque;
	Buffer		metaBuffer;
	Page		metaPage;
	IvfMetaPage meta;

	if (so == NULL)
		return false;

	/* Check if query vector is available */
	if (!so->queryVector)
	{
		return false;
	}

	/* On first call, perform search */
	if (so->firstCall)
	{
		int *selectedClusters = NULL;
		metaBuffer = ReadBuffer(scan->indexRelation, 0);
		if (!BufferIsValid(metaBuffer))
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: ReadBuffer failed")));
		}
		LockBuffer(metaBuffer, BUFFER_LOCK_SHARE);
		metaPage = BufferGetPage(metaBuffer);
		meta = (IvfMetaPage) PageGetContents(metaPage);

		if (meta->magicNumber != IVF_MAGIC_NUMBER)
		{
			UnlockReleaseBuffer(metaBuffer);
			return false;
		}

		/* Check if index is empty */
		if (meta->insertedVectors == 0)
		{
			UnlockReleaseBuffer(metaBuffer);
			so->firstCall = false;
			so->resultCount = 0;
			return false;
		}

		/* Validate query vector dimension matches index */
		if (meta->dim > 0 && so->queryVector->dim != meta->dim)
		{
			UnlockReleaseBuffer(metaBuffer);
			so->firstCall = false;
			so->resultCount = 0;
			return false;
		}

		/* Allocate selected clusters array */
		nalloc(selectedClusters, int, so->nprobe);
		so->selectedClusters = selectedClusters;

		/* Select nprobe closest clusters */
		ivfSelectClusters(scan->indexRelation,
						  meta,
						  so->queryVector->data,
						  so->queryVector->dim,
						  so->nprobe,
						  so->selectedClusters);

		/* Collect candidates from selected clusters */
		ivfCollectCandidates(scan->indexRelation,
							 meta,
							 so->queryVector->data,
							 so->queryVector->dim,
							 so->strategy,
							 so->selectedClusters,
							 so->nprobe,
							 so->k,
							 &so->results,
							 &so->distances,
							 &so->resultCount);

		/* Check if any results were found */
		if (so->resultCount == 0)
		{
		}

		UnlockReleaseBuffer(metaBuffer);
		so->firstCall = false;
		so->currentResult = 0;
	}

	/* Iterative scan: if we've exhausted results and iterative scan is enabled, probe more */
	if (so->currentResult >= so->resultCount && so->iterativeScanEnabled && so->nprobe < so->maxProbes)
	{
		int			oldNprobe = so->nprobe;
		int			newNprobe;

		/* Increase nprobe for next scan (double it, but cap at maxProbes) */
		newNprobe = Min(so->nprobe * 2, so->maxProbes);
		
		if (newNprobe <= oldNprobe)
		{
			/* Can't increase further, give up */
			return false;
		}

		so->nprobe = newNprobe;

		/* Re-scan with increased nprobe */
		metaBuffer = ReadBuffer(scan->indexRelation, 0);
		if (!BufferIsValid(metaBuffer))
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: ReadBuffer failed")));
		}
		LockBuffer(metaBuffer, BUFFER_LOCK_SHARE);
		metaPage = BufferGetPage(metaBuffer);
		meta = (IvfMetaPage) PageGetContents(metaPage);

		if (meta->magicNumber != IVF_MAGIC_NUMBER)
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
		if (so->selectedClusters)
		{
			pfree(so->selectedClusters);
			so->selectedClusters = NULL;
		}

		/* Allocate selected clusters array */
		nalloc(so->selectedClusters, int, so->nprobe);

		/* Select nprobe closest clusters */
		ivfSelectClusters(scan->indexRelation,
						  meta,
						  so->queryVector->data,
						  so->queryVector->dim,
						  so->nprobe,
						  so->selectedClusters);

		/* Collect candidates from selected clusters */
		ivfCollectCandidates(scan->indexRelation,
							 meta,
							 so->queryVector->data,
							 so->queryVector->dim,
							 so->strategy,
							 so->selectedClusters,
							 so->nprobe,
							 so->k,
							 &so->results,
							 &so->distances,
							 &so->resultCount);

		UnlockReleaseBuffer(metaBuffer);
		so->currentResult = 0;
	}

	/* Return next result */
	if (so->currentResult < so->resultCount)
	{
		scan->xs_heaptid = so->results[so->currentResult];
		scan->xs_recheck = false;
		
		/* Tell PostgreSQL to recompute distances via recheckorderby,
		 * similar to how HNSW handles ORDER BY queries */
		scan->xs_recheckorderby = false;

		so->currentResult++;
		return true;
	}

	return false;
}

static void
ivfendscan(IndexScanDesc scan)
{
	IvfScanOpaque so = (IvfScanOpaque) scan->opaque;

	if (so == NULL)
		return;

	if (so->results)
		pfree(so->results);
	if (so->distances)
		pfree(so->distances);
	if (so->selectedClusters)
		pfree(so->selectedClusters);
	if (so->queryVector)
		pfree(so->queryVector);

	pfree(so);
	scan->opaque = NULL;
}

/*
 * kmeans_init - Initialize KMeans state
 *
 * Allocates and initializes a KMeansState structure for performing
 * K-means clustering. Sets up the state with k clusters, dimension,
 * and initializes centroids.
 *
 * Parameters:
 *   k - Number of clusters
 *   dim - Vector dimension
 *   data - Array of data points (unused in current implementation)
 *   n - Number of data points (unused in current implementation)
 *
 * Returns:
 *   Pointer to initialized KMeansState structure
 *
 * Notes:
 *   This function is currently marked as unused. Memory is allocated
 *   in CurrentMemoryContext.
 */
__attribute__((unused)) static KMeansState *
kmeans_init(int k, int dim, float4 * *data, int n)
{
	KMeansState *state = NULL;
	int			i,
				j;
	float4	  **centroids = NULL;
	int *assignments = NULL;
	int *counts = NULL;

	state = (KMeansState *) palloc0(sizeof(KMeansState));
	state->k = k;
	state->dim = dim;
	state->maxIter = IVF_MAX_ITERATIONS;
	state->threshold = IVF_CONVERGENCE_THRESHOLD;
	state->n = n;
	state->data = data;
	state->ctx = CurrentMemoryContext;

	/* Allocate centroids */
	nalloc(centroids, float4 *, k);
	state->centroids = centroids;
	for (i = 0; i < k; i++)
	{
		float4 *centroid = NULL;
		nalloc(centroid, float4, dim);
		state->centroids[i] = centroid;

		/* Initialize with random data points (KMeans++) */
		if (i < n)
		{
			for (j = 0; j < dim; j++)
				state->centroids[i][j] = data[i][j];
		}
	}

	nalloc(assignments, int, n);
	nalloc(counts, int, k);
	state->assignments = assignments;
	state->counts = counts;

	return state;
}

/*
 * Run KMeans clustering (Lloyd's algorithm)
 */
__attribute__((unused)) static void
kmeans_run(KMeansState * state)
{
	int			iter;
	float4		prevCost = FLT_MAX;
	float4		cost;


	for (iter = 0; iter < state->maxIter; iter++)
	{
		/* Assignment step */
		kmeans_assign(state);

		/* Update centroids */
		kmeans_update_centroids(state);

		/* Check convergence */
		cost = kmeans_compute_cost(state);

		if (fabs(prevCost - cost) < state->threshold)
		{
			break;
		}

		prevCost = cost;
	}
}

/*
 * Assign each vector to nearest centroid
 */
static void
kmeans_assign(KMeansState * state)
{
	int			i;

	memset(state->counts, 0, state->k * sizeof(int));

	for (i = 0; i < state->n; i++)
	{
		state->assignments[i] =
			find_nearest_centroid(state, state->data[i]);
		state->counts[state->assignments[i]]++;
	}
}

/*
 * Update centroids to mean of assigned vectors
 */
static void
kmeans_update_centroids(KMeansState * state)
{
	int			i,
				j,
				c;

	/* Zero centroids */
	for (c = 0; c < state->k; c++)
	{
		for (j = 0; j < state->dim; j++)
			state->centroids[c][j] = 0.0;
	}

	/* Sum assigned vectors */
	for (i = 0; i < state->n; i++)
	{
		c = state->assignments[i];
		for (j = 0; j < state->dim; j++)
			state->centroids[c][j] += state->data[i][j];
	}

	/* Divide by count to get mean */
	for (c = 0; c < state->k; c++)
	{
		if (state->counts[c] > 0)
		{
			for (j = 0; j < state->dim; j++)
				state->centroids[c][j] /= state->counts[c];
		}
	}
}

/*
 * Compute total cost (sum of squared distances)
 */
static float4
kmeans_compute_cost(KMeansState * state)
{
	float4		cost = 0.0;
	int			i,
				c;

	for (i = 0; i < state->n; i++)
	{
		c = state->assignments[i];
		cost += vector_distance_l2(
								   state->data[i], state->centroids[c], state->dim);
	}

	return cost;
}

/*
 * Free KMeans state
 */
static void
kmeans_free(KMeansState * state)
{
	int			i;

	for (i = 0; i < state->k; i++)
		pfree(state->centroids[i]);

	pfree(state->centroids);
	pfree(state->assignments);
	pfree(state->counts);
	pfree(state);
}

/*
 * Compute L2 distance (squared)
 */
static float4
vector_distance_l2(const float4 * v1, const float4 * v2, int dim)
{
	float4		sum = 0.0;
	int			i;

	for (i = 0; i < dim; i++)
	{
		float4		diff = v1[i] - v2[i];

		sum += diff * diff;
	}

	return sum;
}

/*
 * Find nearest centroid to vector
 */
static int
find_nearest_centroid(KMeansState * state, const float4 * vector)
{
	int			best = 0;
	float4		bestDist = FLT_MAX;
	int			c;

	for (c = 0; c < state->k; c++)
	{
		float4		dist = vector_distance_l2(
											  vector, state->centroids[c], state->dim);

		if (dist < bestDist)
		{
			bestDist = dist;
			best = c;
		}
	}

	return best;
}

/*
 * Delete a vector from the IVF index
 * IVF deletion requires removing the vector from its assigned inverted list.
 * For now, we mark the tuple as deleted but don't rebuild the lists.
 */
static bool
ivfdelete(Relation index,
		  ItemPointer tid,
		  Datum * values,
		  bool *isnull,
		  Relation heapRel,
		  struct IndexInfo *indexInfo)
{
	Buffer		metaBuf;
	Page		metaPage;
	IvfMetaPage meta;
	Buffer		centroidsBuf;
	Page		centroidsPage;
	IvfCentroid centroid;
	float4	   *centroidVector = NULL;
	BlockNumber listBlock;
	Buffer		listBuf;
	Page		listPage;
	OffsetNumber maxoff;
	OffsetNumber offnum;
	IvfListEntry entry;
	bool		found = false;
	int			i;
	int			minIdx = -1;
	float4		minDist = FLT_MAX;
	Vector	   *inputVec = NULL;
	int			metaBlkno = 0;
	BlockNumber centroidsBlock;
	int			nlists;

	/* Step 1: Read metadata under SHARE lock, copy needed fields */
	metaBuf = ReadBuffer(index, metaBlkno);
	if (!BufferIsValid(metaBuf))
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ReadBuffer failed")));
	}
	LockBuffer(metaBuf, BUFFER_LOCK_SHARE);
	metaPage = BufferGetPage(metaBuf);
	meta = (IvfMetaPage) PageGetContents(metaPage);

	if (meta->centroidsBlock == InvalidBlockNumber)
	{
		UnlockReleaseBuffer(metaBuf);
		return false;
	}

	/* Copy fields we need before unlocking */
	centroidsBlock = meta->centroidsBlock;
	nlists = meta->nlists;

	UnlockReleaseBuffer(metaBuf);

	/* Step 2: Get vector from heap to find which centroid it belongs to */
	if (values != NULL && !isnull[0])
	{
		float4 *vectorData = NULL;
		int			dim;
		Oid			keyType;
		MemoryContext oldctx;

		keyType = ivfGetKeyType(index, 1);
		oldctx = MemoryContextSwitchTo(CurrentMemoryContext);
		vectorData = ivfExtractVectorData(values[0], keyType, &dim, CurrentMemoryContext);
		MemoryContextSwitchTo(oldctx);

		if (vectorData != NULL)
		{
			char *inputVec_raw = NULL;
			nalloc(inputVec_raw, char, VECTOR_SIZE(dim));
			inputVec = (Vector *) inputVec_raw;
			SET_VARSIZE(inputVec, VECTOR_SIZE(dim));
			inputVec->dim = dim;
			memcpy(inputVec->data, vectorData, dim * sizeof(float4));
			pfree(vectorData);
		}
	}

	/* If we don't have the vector, we need to scan all lists */
	if (inputVec == NULL)
	{
		/* Scan all centroids and their lists */
		centroidsBuf = ReadBuffer(index, centroidsBlock);
		if (!BufferIsValid(centroidsBuf))
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: ReadBuffer failed")));
		}
		LockBuffer(centroidsBuf, BUFFER_LOCK_SHARE);
		centroidsPage = BufferGetPage(centroidsBuf);
		maxoff = PageGetMaxOffsetNumber(centroidsPage);

		for (i = 0; i < nlists && i < maxoff; i++)
		{
			offnum = FirstOffsetNumber + i;
			if (offnum > maxoff)
				break;

			centroid = (IvfCentroid) PageGetItem(centroidsPage,
												 PageGetItemId(centroidsPage, offnum));
			listBlock = centroid->firstBlock;

			/* Scan this list for the ItemPointer */
			{
				IvfListPageHeader *listHeader = NULL;

				while (listBlock != InvalidBlockNumber && !found)
				{
					listBuf = ReadBuffer(index, listBlock);
					if (!BufferIsValid(listBuf))
					{
						ereport(ERROR,
								(errcode(ERRCODE_INTERNAL_ERROR),
								 errmsg("neurondb: ReadBuffer failed")));
					}
					LockBuffer(listBuf, BUFFER_LOCK_EXCLUSIVE);
					listPage = BufferGetPage(listBuf);
					listHeader = IvfGetListPageHeader(listPage);
					maxoff = PageGetMaxOffsetNumber(listPage);

					for (offnum = FirstOffsetNumber; offnum <= maxoff; offnum++)
					{
						ItemId		itemId = PageGetItemId(listPage, offnum);

						if (!ItemIdIsValid(itemId) || ItemIdIsDead(itemId))
							continue;

						entry = (IvfListEntry) PageGetItem(listPage, itemId);
						if (ItemPointerEquals(&entry->heapPtr, tid))
						{
							/* Found it - mark as deleted */
							ItemIdSetDead(itemId);
							MarkBufferDirty(listBuf);
							found = true;
							minIdx = i;
							break;
						}
					}

					/* Move to next block in chain */
					listBlock = listHeader->nextBlock;
					UnlockReleaseBuffer(listBuf);
				}
			}

			if (found)
			{
				/* Update centroid member count - need EXCLUSIVE lock */
				UnlockReleaseBuffer(centroidsBuf);
				centroidsBuf = ReadBuffer(index, centroidsBlock);
				if (!BufferIsValid(centroidsBuf))
				{
					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg("neurondb: ReadBuffer failed")));
				}
				LockBuffer(centroidsBuf, BUFFER_LOCK_EXCLUSIVE);
				centroidsPage = BufferGetPage(centroidsBuf);
				centroid = (IvfCentroid) PageGetItem(centroidsPage,
													 PageGetItemId(centroidsPage, FirstOffsetNumber + minIdx));
				centroid->memberCount--;
				if (centroid->memberCount < 0)
					centroid->memberCount = 0;
				MarkBufferDirty(centroidsBuf);
				UnlockReleaseBuffer(centroidsBuf);
				break;
			}
		}

		if (!found)
			UnlockReleaseBuffer(centroidsBuf);
	}
	else
	{
		/* Step 3: Find nearest centroid */
		centroidsBuf = ReadBuffer(index, centroidsBlock);
		if (!BufferIsValid(centroidsBuf))
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: ReadBuffer failed")));
		}
		LockBuffer(centroidsBuf, BUFFER_LOCK_SHARE);
		centroidsPage = BufferGetPage(centroidsBuf);
		maxoff = PageGetMaxOffsetNumber(centroidsPage);

		for (i = 0; i < nlists && i < maxoff; i++)
		{
			float4		dist;
			float4		accum = 0.0f;
			int			k;

			offnum = FirstOffsetNumber + i;
			if (offnum > maxoff)
				break;

			centroid = (IvfCentroid) PageGetItem(centroidsPage,
												 PageGetItemId(centroidsPage, offnum));

			if (centroid->dim != inputVec->dim)
				continue;

			centroidVector = IvfGetCentroidVector(centroid);

			/* Compute L2 distance */
			for (k = 0; k < inputVec->dim; k++)
			{
				float4		diff = inputVec->data[k] - centroidVector[k];

				accum += diff * diff;
			}
			dist = sqrtf(accum);

			if (dist < minDist)
			{
				minDist = dist;
				minIdx = i;
			}
		}

		UnlockReleaseBuffer(centroidsBuf);

		/* Step 4: Remove from selected list */
		if (minIdx >= 0)
		{
			centroidsBuf = ReadBuffer(index, centroidsBlock);
			if (!BufferIsValid(centroidsBuf))
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: ReadBuffer failed")));
			}
			LockBuffer(centroidsBuf, BUFFER_LOCK_EXCLUSIVE);
			centroidsPage = BufferGetPage(centroidsBuf);
			centroid = (IvfCentroid) PageGetItem(centroidsPage,
												 PageGetItemId(centroidsPage, FirstOffsetNumber + minIdx));
			listBlock = centroid->firstBlock;

			/* Scan list for the ItemPointer */
			{
				IvfListPageHeader *listHeader = NULL;

				while (listBlock != InvalidBlockNumber && !found)
				{
					listBuf = ReadBuffer(index, listBlock);
					if (!BufferIsValid(listBuf))
					{
						ereport(ERROR,
								(errcode(ERRCODE_INTERNAL_ERROR),
								 errmsg("neurondb: ReadBuffer failed")));
					}
					LockBuffer(listBuf, BUFFER_LOCK_EXCLUSIVE);
					listPage = BufferGetPage(listBuf);
					listHeader = IvfGetListPageHeader(listPage);
					maxoff = PageGetMaxOffsetNumber(listPage);

					for (offnum = FirstOffsetNumber; offnum <= maxoff; offnum++)
					{
						ItemId		itemId = PageGetItemId(listPage, offnum);

						if (!ItemIdIsValid(itemId) || ItemIdIsDead(itemId))
							continue;

						entry = (IvfListEntry) PageGetItem(listPage, itemId);
						if (ItemPointerEquals(&entry->heapPtr, tid))
						{
							/* Found it - mark as deleted */
							ItemIdSetDead(itemId);
							MarkBufferDirty(listBuf);
							centroid->memberCount--;
							if (centroid->memberCount < 0)
								centroid->memberCount = 0;
							found = true;
							break;
						}
					}

					/* Move to next block in chain */
					listBlock = listHeader->nextBlock;
					UnlockReleaseBuffer(listBuf);
				}
			}

			MarkBufferDirty(centroidsBuf);
			UnlockReleaseBuffer(centroidsBuf);
		}

		pfree(inputVec);
	}

	/* Step 5: Update metadata - reacquire EXCLUSIVE lock */
	metaBuf = ReadBuffer(index, metaBlkno);
	if (!BufferIsValid(metaBuf))
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ReadBuffer failed")));
	}
	LockBuffer(metaBuf, BUFFER_LOCK_EXCLUSIVE);
	meta = (IvfMetaPage) PageGetContents(BufferGetPage(metaBuf));
	meta->insertedVectors--;
	if (meta->insertedVectors < 0)
		meta->insertedVectors = 0;
	MarkBufferDirty(metaBuf);
	UnlockReleaseBuffer(metaBuf);

	return found;
}

/*
 * Update a vector in the IVF index
 * This requires deleting the old vector and inserting the new one
 */
static bool
ivfupdate(Relation index,
		  ItemPointer tid,
		  Datum * values,
		  bool *isnull,
		  ItemPointer otid,
		  Relation heapRel,
		  struct IndexInfo *indexInfo)
{
	/*
	 * For proper vector updates (including upserts), we need to: 1. Find and
	 * remove the old vector from its assigned list 2. Assign the new vector
	 * to the appropriate list and insert it
	 */
	if (!ivfdelete(index, otid, values, isnull, heapRel, indexInfo))
	{
		/* If delete failed, still try to insert new value */
	}

	return ivfinsert(index, values, isnull, tid, heapRel,
					 UNIQUE_CHECK_NO, false, indexInfo);
}

/*
 * Parallel index build support
 * 
 * Note: Parallel index BUILD is handled automatically by PostgreSQL when
 * amcanparallel = true. The amestimateparallelscan, aminitparallelscan, and
 * amparallelrescan callbacks are for parallel index SCANNING (reading from
 * the index), not building. We set them to NULL since we're enabling parallel
 * build, not parallel scan.
 */
