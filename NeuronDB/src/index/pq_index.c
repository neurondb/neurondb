/*-------------------------------------------------------------------------
 *
 * pq_index.c
 *    Product Quantization (PQ) Index Access Method
 *
 * Implements a two-stage retrieval system:
 * - Stage 1: Coarse search using PQ-encoded vectors (fast, approximate)
 * - Stage 2: Fine rerank with full-precision vectors (accurate, slower)
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/index/pq_index.c
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
#include "catalog/index.h"
#include "catalog/pg_type.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_ml.h"
#include "access/heapam.h"
#include "access/tableam.h"
#include "utils/relcache.h"
#include "utils/reloptions.h"
#include "utils/lsyscache.h"
#include "catalog/namespace.h"
#include "parser/parse_type.h"
#include "nodes/makefuncs.h"
#include "optimizer/cost.h"
#include "access/table.h"
#include <math.h>
#include <float.h>
#include <stdlib.h>

extern double neurondb_l2_distance_squared(const float *a, const float *b, int n);

/* PQ index parameters */
#define PQ_DEFAULT_M 8			/* Number of subspaces */
#define PQ_DEFAULT_KS 256		/* Codebook size */
#define PQ_DEFAULT_RERANK_K 100 /* Number of candidates for reranking */

/*
 * PQ index options
 */
typedef struct PqOptions
{
	int			m;				/* Number of subspaces */
	int			ks;				/* Codebook size */
	int			rerank_k;		/* Number of candidates for reranking */
}			PqOptions;

/* Reloption kind - registered in _PG_init() */
extern int	relopt_kind_pq;

/*
 * PQ metadata page (block 0)
 */
typedef struct PqMetaPageData
{
	uint32		magicNumber;
	uint32		version;
	int			m;				/* Number of subspaces */
	int			ks;				/* Codebook size */
	int			dim;			/* Vector dimension */
	int			subspace_dim;	/* Dimension per subspace */
	BlockNumber codebooksBlock; /* Block containing codebooks */
	int64		insertedVectors;
}			PqMetaPageData;

typedef PqMetaPageData * PqMetaPage;

#define PQ_MAGIC_NUMBER 0x50514944 /* "PQID" in hex */
#define PQ_VERSION 1

/*
 * PQ code entry (stored in index pages)
 * Note: codes array size is variable based on m parameter
 * Use PqCodeEntrySize(m) to get actual size
 */
typedef struct PqCodeEntry
{
	ItemPointerData heapPtr;
	uint8_t		codes[FLEXIBLE_ARRAY_MEMBER]; /* PQ codes for each subspace (size = m) */
}			PqCodeEntry;

#define PqCodeEntrySize(m) (offsetof(PqCodeEntry, codes) + (m) * sizeof(uint8_t))

/*
 * PQ build state
 */
typedef struct PqBuildState
{
	Relation	heap;
	Relation	index;
	IndexInfo  *indexInfo;
	double		indtuples;
	MemoryContext tmpCtx;
	int			m;
	int			ks;
	int			dim;
	int			subspace_dim;
	float	   **vectors;			/* Collected vectors for training */
	ItemPointer *tids;				/* Corresponding TIDs */
	int			vector_count;
	int			vector_capacity;	/* Current capacity of vectors/tids arrays */
	float	  ***codebooks;			/* Trained codebooks [m][ks][subspace_dim] */
}			PqBuildState;

/*
 * PQ scan state
 */
typedef struct PqScanOpaqueData
{
	float4	   *query;			/* Query vector */
	int			queryDim;		/* Query dimension */
	int			k;				/* Number of results to return */
	int			rerank_k;		/* Number of candidates for reranking */
	bool		firstCall;		/* True on first call to amgettuple */
	int			resultCount;	/* Number of results found */
	ItemPointer *results;		/* Result heap TIDs */
	float4	   *distances;		/* Result distances */
	int			currentResult;	/* Current result index */
	MemoryContext scanCtx;		/* Scan memory context */
	float	  ***codebooks;		/* Loaded codebooks */
	int			m;				/* Number of subspaces */
	int			ks;				/* Codebook size */
	int			subspace_dim;	/* Dimension per subspace */
}			PqScanOpaqueData;

typedef PqScanOpaqueData * PqScanOpaque;

/*
 * Forward declarations
 */
static IndexBuildResult * pqbuild(Relation heap, Relation index, IndexInfo * indexInfo);
static void pqBuildCallback(Relation index, ItemPointer tid, Datum * values,
							bool *isnull, bool tupleIsAlive, void *state);
static void train_pq_codebooks(PqBuildState * buildstate);
static uint8_t **encode_vectors_pq(PqBuildState * buildstate);
static BlockNumber store_codebooks(Relation index, PqBuildState * buildstate);
static void store_encoded_vectors(Relation index, PqBuildState * buildstate, uint8_t **encoded_codes);
static bytea * pqoptions(Datum reloptions, bool validate);
static void pqLoadOptions(Relation index, PqOptions *opts_out);
static IndexScanDesc pqbeginscan(Relation index, int nkeys, int norderbys);
static bool pqgettuple(IndexScanDesc scan, ScanDirection dir);
static void pqrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys);
static void pqendscan(IndexScanDesc scan);
static void pq_coarse_search(Relation index, PqMetaPage meta, PqScanOpaque so,
							 ItemPointer **coarse_tids, float **coarse_dists, int *coarse_count);
static void pq_fine_rerank(Relation index, PqMetaPage meta, PqScanOpaque so,
						   ItemPointer *coarse_tids, float *coarse_dists, int coarse_count);
static IndexBulkDeleteResult * pqbulkdelete(IndexVacuumInfo * info,
											IndexBulkDeleteResult * stats,
											IndexBulkDeleteCallback callback,
											void *callback_state);
static IndexBulkDeleteResult * pqvacuumcleanup(IndexVacuumInfo * info, IndexBulkDeleteResult * stats);
static void pqcostestimate(struct PlannerInfo *root, struct IndexPath *path, double loop_count,
						   Cost * indexStartupCost, Cost * indexTotalCost,
						   Selectivity * indexSelectivity, double *indexCorrelation,
						   double *indexPages);

/*
 * Train k-means codebook for a single subspace
 * This is a simplified version of train_subspace_kmeans from ml_product_quantization.c
 */
static void
train_subspace_kmeans_pq(float **subspace_data,
						int nvec,
						int dsub,
						int k,
						float **centroids,
						int max_iters)
{
	bool		changed = true;
	int			c, d, i, idx, iter;
	int		   *assignments = NULL;
	int		   *counts = NULL;

	nalloc(assignments, int, nvec);
	NDB_CHECK_ALLOC(assignments, "assignments");

	/* Random initialization */
	for (c = 0; c < k; c++)
	{
		idx = rand() % nvec;
		memcpy(centroids[c], subspace_data[idx], sizeof(float) * dsub);
	}

	/* Lloyd's algorithm */
	for (iter = 0; iter < max_iters && changed; iter++)
	{
		changed = false;

		/* Assignment step */
		for (i = 0; i < nvec; i++)
		{
			double		min_dist = DBL_MAX;
			int			best = -1;

			for (c = 0; c < k; c++)
			{
				double		dist = 0.0;

				for (d = 0; d < dsub; d++)
				{
					double		diff = (double) subspace_data[i][d] - (double) centroids[c][d];
					dist += diff * diff;
				}
				if (dist < min_dist)
				{
					min_dist = dist;
					best = c;
				}
			}
			if (assignments[i] != best)
			{
				assignments[i] = best;
				changed = true;
			}
		}

		if (!changed)
			break;

		/* Update step */
		nalloc(counts, int, k);
		NDB_CHECK_ALLOC(counts, "counts");
		for (c = 0; c < k; c++)
			memset(centroids[c], 0, sizeof(float) * dsub);

		for (i = 0; i < nvec; i++)
		{
			c = assignments[i];
			for (d = 0; d < dsub; d++)
				centroids[c][d] += subspace_data[i][d];
			counts[c]++;
		}

		for (c = 0; c < k; c++)
		{
			if (counts[c] > 0)
			{
				for (d = 0; d < dsub; d++)
					centroids[c][d] /= counts[c];
			}
		}
		nfree(counts);
	}

	nfree(assignments);
}

/*
 * Load PQ index options from relation (stored bytea from pqoptions).
 */
static void
pqLoadOptions(Relation index, PqOptions *opts_out)
{
	Datum		reloptions;
	bytea	   *opt;

	reloptions = RelationGetReloptions(index);
	if (reloptions == (Datum) 0 || DatumGetPointer(reloptions) == NULL)
	{
		opts_out->m = PQ_DEFAULT_M;
		opts_out->ks = PQ_DEFAULT_KS;
		opts_out->rerank_k = PQ_DEFAULT_RERANK_K;
		return;
	}

	opt = (bytea *) DatumGetPointer(reloptions);
	if (VARSIZE_ANY_EXHDR(opt) < (Size) sizeof(PqOptions))
	{
		opts_out->m = PQ_DEFAULT_M;
		opts_out->ks = PQ_DEFAULT_KS;
		opts_out->rerank_k = PQ_DEFAULT_RERANK_K;
		return;
	}

	{
		PqOptions *p = (PqOptions *) VARDATA(opt);

		opts_out->m = p->m;
		opts_out->ks = p->ks;
		opts_out->rerank_k = p->rerank_k;
		if (opts_out->m < 1 || opts_out->m > 32)
			opts_out->m = PQ_DEFAULT_M;
		if (opts_out->ks < 16 || opts_out->ks > 65536)
			opts_out->ks = PQ_DEFAULT_KS;
		if (opts_out->rerank_k < 1 || opts_out->rerank_k > 10000)
			opts_out->rerank_k = PQ_DEFAULT_RERANK_K;
	}
}

/* Placeholder for future implementation */
static bool
pqbuildempty(Relation index)
{
	Buffer		metaBuffer;
	Page		metaPage;
	PqMetaPage meta;

	metaBuffer = ReadBuffer(index, 0);
	LockBuffer(metaBuffer, BUFFER_LOCK_EXCLUSIVE);
	metaPage = BufferGetPage(metaBuffer);
	meta = (PqMetaPage) PageGetContents(metaPage);

	meta->magicNumber = PQ_MAGIC_NUMBER;
	meta->version = PQ_VERSION;
	meta->m = PQ_DEFAULT_M;
	meta->ks = PQ_DEFAULT_KS;
	meta->dim = 0;				/* Will be set during build */
	meta->subspace_dim = 0;
	meta->codebooksBlock = InvalidBlockNumber;
	meta->insertedVectors = 0;

	MarkBufferDirty(metaBuffer);
	UnlockReleaseBuffer(metaBuffer);

	return true;
}

/*
 * Index AM handler
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pqhandler);
Datum
pqhandler(PG_FUNCTION_ARGS)
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
	amroutine->amcanparallel = false;
	amroutine->amcanbuildparallel = false;
	amroutine->amcaninclude = false;
	amroutine->amusemaintenanceworkmem = false;
	amroutine->amparallelvacuumoptions = 0;
	amroutine->amkeytype = InvalidOid;

	/* Interface functions */
	amroutine->ambuildempty = pqbuildempty;
	amroutine->ambuild = pqbuild;
	amroutine->aminsert = pqinsert;
	amroutine->ambulkdelete = pqbulkdelete;
	amroutine->amvacuumcleanup = pqvacuumcleanup;
	amroutine->amcanreturn = NULL;
	amroutine->amcostestimate = pqcostestimate;
	amroutine->amoptions = pqoptions;
	amroutine->amproperty = NULL;
	amroutine->ambuildphasename = NULL;
	amroutine->amvalidate = NULL;
	amroutine->ambeginscan = pqbeginscan;
	amroutine->amrescan = pqrescan;
	amroutine->amgettuple = pqgettuple;
	amroutine->amendscan = pqendscan;
	amroutine->amgetbitmap = NULL;
	amroutine->amendscan = NULL;
	amroutine->ammarkpos = NULL;
	amroutine->amrestrpos = NULL;

	PG_RETURN_POINTER(amroutine);
}

/*
 * PQ index build callback - collects vectors and TIDs
 */
static void
pqBuildCallback(Relation index, ItemPointer tid, Datum * values,
				bool *isnull, bool tupleIsAlive, void *state)
{
	PqBuildState *buildstate = (PqBuildState *) state;
	Vector	   *vec = NULL;
	float4	   *vec_data = NULL;
	Oid			keyType;
	int			dim;

	if (!tupleIsAlive || isnull[0])
		return;

	/* Extract vector */
	{
		TupleDesc	indexDesc = RelationGetDescr(index);
		Form_pg_attribute attr;

		if (indexDesc->natts < 1)
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("pq: index must have at least one column")));

		attr = TupleDescAttr(indexDesc, 0);
		keyType = attr->atttypid;
	}

	/* Get vector type OID for comparison */
	{
		Oid			vectorOid = InvalidOid;
		List	   *names = list_make2(makeString("public"), makeString("vector"));

		vectorOid = LookupTypeNameOid(NULL, makeTypeNameFromNameList(names), false);
		list_free(names);

		if (keyType != vectorOid)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("pq: only vector type supported, got type OID %u", keyType)));
	}

	vec = DatumGetVectorP(values[0]);
	NDB_CHECK_VECTOR_VALID(vec);
	dim = vec->dim;
	vec_data = vec->data;

	/* Initialize dimension on first vector */
	if (buildstate->dim == 0)
	{
		buildstate->dim = dim;
		buildstate->subspace_dim = dim / buildstate->m;
		if (dim % buildstate->m != 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("pq: vector dimension %d must be divisible by m=%d",
						dim, buildstate->m)));
	}
	else if (buildstate->dim != dim)
	{
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("pq: vector dimension mismatch: expected %d, got %d",
					buildstate->dim, dim)));
	}

	/* Allocate space for vector and TID - use dynamic growth */
	if (buildstate->vectors == NULL)
	{
		/* Initial allocation */
		buildstate->vector_capacity = 1000;
		nalloc(buildstate->vectors, float *, buildstate->vector_capacity);
		nalloc(buildstate->tids, ItemPointer, buildstate->vector_capacity);
		buildstate->vector_count = 0;
	}
	else if (buildstate->vector_count >= buildstate->vector_capacity)
	{
		/* Grow capacity - double it */
		buildstate->vector_capacity *= 2;
		buildstate->vectors = (float **) repalloc(buildstate->vectors,
												   sizeof(float *) * buildstate->vector_capacity);
		buildstate->tids = (ItemPointer *) repalloc(buildstate->tids,
													 sizeof(ItemPointer) * buildstate->vector_capacity);
	}

	/* Copy vector data */
	{
		float	   *vec_copy = NULL;

		nalloc(vec_copy, float, dim);
		memcpy(vec_copy, vec_data, sizeof(float) * dim);
		buildstate->vectors[buildstate->vector_count] = vec_copy;
		buildstate->tids[buildstate->vector_count] = *tid;
		buildstate->vector_count++;
	}

	buildstate->indtuples++;
}

/*
 * Train PQ codebooks for all subspaces
 */
static void
train_pq_codebooks(PqBuildState * buildstate)
{
	int			sub, i, c;
	float	  **subspace_data = NULL;

	/* Allocate codebooks [m][ks][subspace_dim] */
	nalloc(buildstate->codebooks, float **, buildstate->m);
	NDB_CHECK_ALLOC(buildstate->codebooks, "codebooks");

	for (sub = 0; sub < buildstate->m; sub++)
	{
		float	  **sub_centroids = NULL;

		nalloc(sub_centroids, float *, buildstate->ks);
		NDB_CHECK_ALLOC(sub_centroids, "sub_centroids");
		buildstate->codebooks[sub] = sub_centroids;

		for (c = 0; c < buildstate->ks; c++)
		{
			float	   *centroid = NULL;

			nalloc(centroid, float, buildstate->subspace_dim);
			NDB_CHECK_ALLOC(centroid, "centroid");
			buildstate->codebooks[sub][c] = centroid;
		}
	}

	/* Train codebook for each subspace */
	for (sub = 0; sub < buildstate->m; sub++)
	{
		int			start_dim = sub * buildstate->subspace_dim;

		/* Extract subspace vectors */
		nalloc(subspace_data, float *, buildstate->vector_count);
		NDB_CHECK_ALLOC(subspace_data, "subspace_data");

		for (i = 0; i < buildstate->vector_count; i++)
		{
			float	   *sub_vec = NULL;

			nalloc(sub_vec, float, buildstate->subspace_dim);
			NDB_CHECK_ALLOC(sub_vec, "sub_vec");
			memcpy(sub_vec, &buildstate->vectors[i][start_dim],
				   sizeof(float) * buildstate->subspace_dim);
			subspace_data[i] = sub_vec;
		}

		/* Train k-means for this subspace */
		train_subspace_kmeans_pq(subspace_data,
								 buildstate->vector_count,
								 buildstate->subspace_dim,
								 buildstate->ks,
								 buildstate->codebooks[sub],
								 100);	/* max iterations */

		/* Free subspace data */
		for (i = 0; i < buildstate->vector_count; i++)
			nfree(subspace_data[i]);
		nfree(subspace_data);
	}
}

/*
 * Encode all vectors using trained codebooks
 * Returns allocated array of encoded codes
 */
static uint8_t **
encode_vectors_pq(PqBuildState * buildstate)
{
	int			i, sub, c, d;
	int			start_dim;
	uint8_t   **encoded_codes = NULL;

	/* Allocate encoded codes array */
	nalloc(encoded_codes, uint8_t *, buildstate->vector_count);
	NDB_CHECK_ALLOC(encoded_codes, "encoded_codes");

	for (i = 0; i < buildstate->vector_count; i++)
	{
		uint8_t    *codes = NULL;

		nalloc(codes, uint8_t, buildstate->m);
		NDB_CHECK_ALLOC(codes, "codes");
		encoded_codes[i] = codes;

		/* Encode each subspace */
		for (sub = 0; sub < buildstate->m; sub++)
		{
			start_dim = sub * buildstate->subspace_dim;
			double		min_dist = DBL_MAX;
			int			best = 0;

			/* Find closest centroid */
			for (c = 0; c < buildstate->ks; c++)
			{
				double		dist = 0.0;

				for (d = 0; d < buildstate->subspace_dim; d++)
				{
					double		diff = (double) buildstate->vectors[i][start_dim + d]
						- (double) buildstate->codebooks[sub][c][d];
					dist += diff * diff;
				}
				if (dist < min_dist)
				{
					min_dist = dist;
					best = c;
				}
			}
			codes[sub] = (uint8_t) best;
		}
	}

	return encoded_codes;
}

/*
 * Store codebooks in index pages
 * Returns block number where codebooks are stored
 */
static BlockNumber
store_codebooks(Relation index, PqBuildState * buildstate)
{
	Buffer		codebookBuffer;
	Page		codebookPage;
	char	   *page_data = NULL;
	int			offset = 0;
	int			sub, c;
	Size		codebook_size;
	BlockNumber codebook_block;

	/* Calculate codebook size */
	codebook_size = sizeof(int) * 3 + /* m, ks, subspace_dim */
		buildstate->m * buildstate->ks * buildstate->subspace_dim * sizeof(float);

	/* Check if codebook fits in one page */
	if (codebook_size > BLCKSZ - SizeOfPageHeaderData)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("pq: codebook size %zu exceeds page size", codebook_size)));

	/* Allocate codebook page */
	codebookBuffer = ReadBuffer(index, P_NEW);
	LockBuffer(codebookBuffer, BUFFER_LOCK_EXCLUSIVE);
	codebookPage = BufferGetPage(codebookBuffer);
	PageInit(codebookPage, BLCKSZ, 0);

	/* Write codebook data to page */
	page_data = PageGetContents(codebookPage);
	memcpy(page_data + offset, &buildstate->m, sizeof(int));
	offset += sizeof(int);
	memcpy(page_data + offset, &buildstate->ks, sizeof(int));
	offset += sizeof(int);
	memcpy(page_data + offset, &buildstate->subspace_dim, sizeof(int));
	offset += sizeof(int);

	/* Write centroids */
	for (sub = 0; sub < buildstate->m; sub++)
	{
		for (c = 0; c < buildstate->ks; c++)
		{
			memcpy(page_data + offset, buildstate->codebooks[sub][c],
				   sizeof(float) * buildstate->subspace_dim);
			offset += sizeof(float) * buildstate->subspace_dim;
		}
	}

	codebook_block = BufferGetBlockNumber(codebookBuffer);
	MarkBufferDirty(codebookBuffer);
	UnlockReleaseBuffer(codebookBuffer);

	return codebook_block;
}

/*
 * Store encoded vectors in index pages
 */
static void
store_encoded_vectors(Relation index, PqBuildState * buildstate, uint8_t **encoded_codes)
{
	Buffer		vectorBuffer = InvalidBuffer;
	Page		vectorPage;
	OffsetNumber offnum;
	int			i;
	int			entries_per_page;
	Size		entry_size;

	/* Calculate entry size - ItemPointer + m codes */
	entry_size = MAXALIGN(PqCodeEntrySize(buildstate->m));
	entries_per_page = (BLCKSZ - SizeOfPageHeaderData) / entry_size;

	if (entries_per_page <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("pq: entry size %zu too large for page", entry_size)));

	/* Store encoded vectors */
	for (i = 0; i < buildstate->vector_count; i++)
	{
		int			page_offset = i % entries_per_page;

		if (page_offset == 0)
		{
			/* Need new page */
			if (BufferIsValid(vectorBuffer))
				UnlockReleaseBuffer(vectorBuffer);

			vectorBuffer = ReadBuffer(index, P_NEW);
			LockBuffer(vectorBuffer, BUFFER_LOCK_EXCLUSIVE);
			vectorPage = BufferGetPage(vectorBuffer);
			PageInit(vectorPage, BLCKSZ, 0);
		}

		/* Create code entry */
		{
			PqCodeEntry *entry = NULL;
			char	   *entry_buf = NULL;

			/* Allocate entry data */
			nalloc(entry_buf, char, entry_size);
			entry = (PqCodeEntry *) entry_buf;

			/* Set heap pointer and codes */
			entry->heapPtr = buildstate->tids[i];
			memcpy(entry->codes, encoded_codes[i], buildstate->m * sizeof(uint8_t));

			/* Add to page */
			offnum = PageAddItem(vectorPage,
								 (Item) entry,
								 entry_size,
								 InvalidOffsetNumber,
								 false,
								 false);
			if (offnum == InvalidOffsetNumber)
			{
				nfree(entry_buf);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("pq: failed to add item to page")));
			}

			/* Entry is now on page, can free temporary buffer */
			nfree(entry_buf);
		}

		MarkBufferDirty(vectorBuffer);
	}

	if (BufferIsValid(vectorBuffer))
		UnlockReleaseBuffer(vectorBuffer);
}

/*
 * PQ index build
 */
static IndexBuildResult *
pqbuild(Relation heap, Relation index, IndexInfo * indexInfo)
{
	PqBuildState buildstate = {0};
	Buffer		metaBuffer;
	PqOptions	options;
	IndexBuildResult *result = NULL;
	BlockNumber codebook_block = InvalidBlockNumber;
	uint8_t   **encoded_codes = NULL;
	int			i;

	/* Initialize build state */
	buildstate.heap = heap;
	buildstate.index = index;
	buildstate.indexInfo = indexInfo;
	buildstate.indtuples = 0;
	buildstate.vector_count = 0;
	buildstate.dim = 0;
	buildstate.tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
											  "PQ build temporary context",
											  ALLOCSET_DEFAULT_SIZES);

	/* Load options */
	pqLoadOptions(index, &options);
	buildstate.m = options.m;
	buildstate.ks = options.ks;

	/* Initialize metadata page */
	metaBuffer = ReadBuffer(index, 0);
	LockBuffer(metaBuffer, BUFFER_LOCK_EXCLUSIVE);
	{
		Page		metaPage = BufferGetPage(metaBuffer);
		PqMetaPage meta = (PqMetaPage) PageGetContents(metaPage);

		meta->magicNumber = PQ_MAGIC_NUMBER;
		meta->version = PQ_VERSION;
		meta->m = buildstate.m;
		meta->ks = buildstate.ks;
		meta->dim = 0;			/* Will be set after collecting vectors */
		meta->subspace_dim = 0;
		meta->codebooksBlock = InvalidBlockNumber;
		meta->insertedVectors = 0;
	}
	MarkBufferDirty(metaBuffer);
	UnlockReleaseBuffer(metaBuffer);

	/* Collect all vectors using table scan */
	buildstate.indtuples = table_index_build_scan(heap, index, indexInfo,
												  true, true, pqBuildCallback,
												  (void *) &buildstate, NULL);

	if (buildstate.vector_count == 0)
	{
		/* No vectors to index */
		nalloc(result, IndexBuildResult, 1);
		result->heap_tuples = 0;
		result->index_tuples = 0;
		MemoryContextDelete(buildstate.tmpCtx);
		return result;
	}

	/* Validate dimension is divisible by m */
	if (buildstate.dim % buildstate.m != 0)
	{
		MemoryContextDelete(buildstate.tmpCtx);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("pq: vector dimension %d must be divisible by m=%d",
					buildstate.dim, buildstate.m)));
	}

	buildstate.subspace_dim = buildstate.dim / buildstate.m;

	/* Train codebooks */
	elog(DEBUG1, "pq: training codebooks for %d vectors, dim=%d, m=%d, ks=%d",
		 buildstate.vector_count, buildstate.dim, buildstate.m, buildstate.ks);
	train_pq_codebooks(&buildstate);

	/* Encode vectors */
	elog(DEBUG1, "pq: encoding %d vectors", buildstate.vector_count);
	encoded_codes = encode_vectors_pq(&buildstate);

	/* Store codebooks */
	elog(DEBUG1, "pq: storing codebooks");
	codebook_block = store_codebooks(index, &buildstate);

	/* Store encoded vectors */
	elog(DEBUG1, "pq: storing %d encoded vectors", buildstate.vector_count);
	store_encoded_vectors(index, &buildstate, encoded_codes);

	/* Update metadata page */
	metaBuffer = ReadBuffer(index, 0);
	LockBuffer(metaBuffer, BUFFER_LOCK_EXCLUSIVE);
	{
		Page		metaPage = BufferGetPage(metaBuffer);
		PqMetaPage meta = (PqMetaPage) PageGetContents(metaPage);

		meta->dim = buildstate.dim;
		meta->subspace_dim = buildstate.subspace_dim;
		meta->codebooksBlock = codebook_block;
		meta->insertedVectors = buildstate.vector_count;
	}
	MarkBufferDirty(metaBuffer);
	UnlockReleaseBuffer(metaBuffer);

	/* Cleanup */
	for (i = 0; i < buildstate.vector_count; i++)
	{
		if (buildstate.vectors[i] != NULL)
			nfree(buildstate.vectors[i]);
		if (encoded_codes[i] != NULL)
			nfree(encoded_codes[i]);
	}
	if (buildstate.vectors != NULL)
		nfree(buildstate.vectors);
	if (buildstate.tids != NULL)
		nfree(buildstate.tids);
	if (encoded_codes != NULL)
		nfree(encoded_codes);

	/* Free codebooks */
	if (buildstate.codebooks != NULL)
	{
		for (i = 0; i < buildstate.m; i++)
		{
			if (buildstate.codebooks[i] != NULL)
			{
				int			c;

				for (c = 0; c < buildstate.ks; c++)
				{
					if (buildstate.codebooks[i][c] != NULL)
						nfree(buildstate.codebooks[i][c]);
				}
				nfree(buildstate.codebooks[i]);
			}
		}
		nfree(buildstate.codebooks);
	}

	nalloc(result, IndexBuildResult, 1);
	result->heap_tuples = buildstate.indtuples;
	result->index_tuples = buildstate.vector_count;

	MemoryContextDelete(buildstate.tmpCtx);

	return result;
}

/*
 * PQ index options handler - parse reloptions and return PqOptions bytea.
 */
static bytea *
pqoptions(Datum reloptions, bool validate)
{
	static const relopt_parse_elt tab[] = {
		{"m", RELOPT_TYPE_INT, offsetof(PqOptions, m)},
		{"ks", RELOPT_TYPE_INT, offsetof(PqOptions, ks)},
		{"rerank_k", RELOPT_TYPE_INT, offsetof(PqOptions, rerank_k)}
	};
	bytea	   *result;

	if (relopt_kind_pq == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("relopt_kind_pq not initialized")));

	if (reloptions == (Datum) 0 || reloptions == PointerGetDatum(NULL))
		reloptions = (Datum) 0;

	result = (bytea *) build_reloptions(reloptions, validate, relopt_kind_pq,
										sizeof(PqOptions), tab, lengthof(tab));
	if (result == NULL)
	{
		result = (bytea *) palloc(sizeof(PqOptions) + VARHDRSZ);
		SET_VARSIZE(result, sizeof(PqOptions) + VARHDRSZ);
		{
			PqOptions *opts = (PqOptions *) VARDATA(result);

			opts->m = PQ_DEFAULT_M;
			opts->ks = PQ_DEFAULT_KS;
			opts->rerank_k = PQ_DEFAULT_RERANK_K;
		}
	}
	return result;
}

/*
 * Load codebooks from index
 */
static float ***
load_codebooks(Relation index, PqMetaPage meta, MemoryContext ctx)
{
	Buffer		codebookBuffer;
	Page		codebookPage;
	char	   *page_data = NULL;
	float	  ***codebooks = NULL;
	int			sub, c;
	int			offset = 0;
	int			m, ks, dsub;
	MemoryContext oldctx;

	if (!BlockNumberIsValid(meta->codebooksBlock))
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("pq: codebooks not found in index")));

	oldctx = MemoryContextSwitchTo(ctx);

	codebookBuffer = ReadBuffer(index, meta->codebooksBlock);
	LockBuffer(codebookBuffer, BUFFER_LOCK_SHARE);
	codebookPage = BufferGetPage(codebookBuffer);
	page_data = PageGetContents(codebookPage);

	/* Read header */
	memcpy(&m, page_data + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(&ks, page_data + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(&dsub, page_data + offset, sizeof(int));
	offset += sizeof(int);

	/* Validate */
	if (m != meta->m || ks != meta->ks || dsub != meta->subspace_dim)
	{
		UnlockReleaseBuffer(codebookBuffer);
		MemoryContextSwitchTo(oldctx);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("pq: codebook metadata mismatch")));
	}

	/* Allocate codebooks */
	nalloc(codebooks, float **, m);
	for (sub = 0; sub < m; sub++)
	{
		float	  **sub_centroids = NULL;

		nalloc(sub_centroids, float *, ks);
		codebooks[sub] = sub_centroids;

		for (c = 0; c < ks; c++)
		{
			float	   *centroid = NULL;

			nalloc(centroid, float, dsub);
			memcpy(centroid, page_data + offset, sizeof(float) * dsub);
			offset += sizeof(float) * dsub;
			codebooks[sub][c] = centroid;
		}
	}

	UnlockReleaseBuffer(codebookBuffer);
	MemoryContextSwitchTo(oldctx);

	return codebooks;
}

/*
 * Encode a single vector using codebooks
 */
static uint8_t *
encode_vector_pq_single(const float *vector, int dim, float ***codebooks, int m, int ks, int dsub, MemoryContext ctx)
{
	uint8_t    *codes = NULL;
	int			sub, c, d;
	int			start_dim;
	MemoryContext oldctx;

	oldctx = MemoryContextSwitchTo(ctx);
	nalloc(codes, uint8_t, m);

	for (sub = 0; sub < m; sub++)
	{
		start_dim = sub * dsub;
		double		min_dist = DBL_MAX;
		int			best = 0;

		/* Find closest centroid */
		for (c = 0; c < ks; c++)
		{
			double		dist = 0.0;

			for (d = 0; d < dsub; d++)
			{
				double		diff = (double) vector[start_dim + d] - (double) codebooks[sub][c][d];
				dist += diff * diff;
			}
			if (dist < min_dist)
			{
				min_dist = dist;
				best = c;
			}
		}
		codes[sub] = (uint8_t) best;
	}

	MemoryContextSwitchTo(oldctx);
	return codes;
}

/*
 * PQ index insert
 */
static bool
pqinsert(Relation index, Datum * values, bool *isnull, ItemPointer ht_ctid,
		 Relation heapRel, IndexUniqueCheck checkUnique,
		 bool indexUnchanged, struct IndexInfo *indexInfo)
{
	Buffer		metaBuffer;
	Page		metaPage;
	PqMetaPage meta;
	Vector	   *vec = NULL;
	float4	   *vector_data = NULL;
	int			dim;
	uint8_t    *codes = NULL;
	Buffer		vectorBuffer = InvalidBuffer;
	Page		vectorPage;
	OffsetNumber offnum;
	Size		entry_size;
	PqCodeEntry *entry = NULL;
	char	   *entry_buf = NULL;
	Oid			keyType;
	float	  ***codebooks = NULL;
	MemoryContext insertCtx;
	MemoryContext oldctx;

	if (isnull[0])
		return false;

	/* Validate heap TID */
	if (!ItemPointerIsValid(ht_ctid))
	{
		elog(WARNING, "pq: invalid heap TID provided, skipping insert");
		return false;
	}

	/* Get key type */
	{
		TupleDesc	indexDesc = RelationGetDescr(index);
		Form_pg_attribute attr;

		if (indexDesc->natts < 1)
			return false;

		attr = TupleDescAttr(indexDesc, 0);
		keyType = attr->atttypid;
	}

	/* Extract vector */
	{
		Oid			vectorOid = InvalidOid;
		List	   *names = list_make2(makeString("public"), makeString("vector"));

		vectorOid = LookupTypeNameOid(NULL, makeTypeNameFromNameList(names), false);
		list_free(names);

		if (keyType != vectorOid)
			return false;
	}

	vec = DatumGetVectorP(values[0]);
	NDB_CHECK_VECTOR_VALID(vec);
	dim = vec->dim;
	vector_data = vec->data;

	/* Create memory context for insert */
	insertCtx = AllocSetContextCreate(CurrentMemoryContext,
									  "PQ insert temporary context",
									  ALLOCSET_DEFAULT_SIZES);
	oldctx = MemoryContextSwitchTo(insertCtx);

	/* Read metadata */
	metaBuffer = ReadBuffer(index, 0);
	LockBuffer(metaBuffer, BUFFER_LOCK_SHARE);
	metaPage = BufferGetPage(metaBuffer);
	meta = (PqMetaPage) PageGetContents(metaPage);

	/* Validate metadata */
	if (meta->magicNumber != PQ_MAGIC_NUMBER)
	{
		UnlockReleaseBuffer(metaBuffer);
		MemoryContextSwitchTo(oldctx);
		MemoryContextDelete(insertCtx);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("pq: invalid index magic number")));
	}

	if (meta->dim != dim)
	{
		UnlockReleaseBuffer(metaBuffer);
		MemoryContextSwitchTo(oldctx);
		MemoryContextDelete(insertCtx);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("pq: vector dimension %d does not match index dimension %d",
					dim, meta->dim)));
	}

	/* Load codebooks */
	codebooks = load_codebooks(index, meta, insertCtx);
	UnlockReleaseBuffer(metaBuffer);

	/* Encode vector */
	codes = encode_vector_pq_single(vector_data, dim, codebooks, meta->m, meta->ks,
									meta->subspace_dim, insertCtx);

	/* Calculate entry size */
	entry_size = MAXALIGN(PqCodeEntrySize(meta->m));

	/* Find a page with space or allocate new page */
	/* For simplicity, always append to a new page - full implementation would reuse pages */
	vectorBuffer = ReadBuffer(index, P_NEW);
	LockBuffer(vectorBuffer, BUFFER_LOCK_EXCLUSIVE);
	vectorPage = BufferGetPage(vectorBuffer);
	PageInit(vectorPage, BLCKSZ, 0);

	/* Create entry */
	nalloc(entry_buf, char, entry_size);
	entry = (PqCodeEntry *) entry_buf;
	entry->heapPtr = *ht_ctid;
	memcpy(entry->codes, codes, meta->m * sizeof(uint8_t));

	/* Add to page */
	offnum = PageAddItem(vectorPage,
						 (Item) entry,
						 entry_size,
						 InvalidOffsetNumber,
						 false,
						 false);
	if (offnum == InvalidOffsetNumber)
	{
		nfree(entry_buf);
		UnlockReleaseBuffer(vectorBuffer);
		MemoryContextSwitchTo(oldctx);
		MemoryContextDelete(insertCtx);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("pq: failed to add item to page")));
	}

	MarkBufferDirty(vectorBuffer);
	UnlockReleaseBuffer(vectorBuffer);

	/* Update metadata - increment count */
	metaBuffer = ReadBuffer(index, 0);
	LockBuffer(metaBuffer, BUFFER_LOCK_EXCLUSIVE);
	metaPage = BufferGetPage(metaBuffer);
	meta = (PqMetaPage) PageGetContents(metaPage);
	meta->insertedVectors++;
	MarkBufferDirty(metaBuffer);
	UnlockReleaseBuffer(metaBuffer);

	/* Cleanup */
	nfree(entry_buf);
	MemoryContextSwitchTo(oldctx);
	MemoryContextDelete(insertCtx);

	return true;
}

/*
 * PQ index scan begin
 */
static IndexScanDesc
pqbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	PqScanOpaque so;
	MemoryContext oldctx;

	scan = RelationGetIndexScan(index, nkeys, norderbys);
	so = (PqScanOpaque) palloc0(sizeof(PqScanOpaqueData));
	scan->opaque = so;

	/* Create scan memory context */
	so->scanCtx = AllocSetContextCreate(CurrentMemoryContext,
										"PQ scan context",
										ALLOCSET_DEFAULT_SIZES);

	so->firstCall = true;
	so->currentResult = 0;
	so->resultCount = 0;
	so->results = NULL;
	so->distances = NULL;
	so->query = NULL;
	so->k = 10;			/* Default */
	so->rerank_k = PQ_DEFAULT_RERANK_K;

	return scan;
}

/*
 * PQ index rescan
 */
static void
pqrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	PqScanOpaque so = (PqScanOpaque) scan->opaque;
	MemoryContext oldctx;
	Oid			queryType;

	/* Reset scan state */
	if (so->scanCtx != NULL)
		MemoryContextReset(so->scanCtx);

	so->results = NULL;
	so->distances = NULL;
	so->firstCall = true;
	so->currentResult = 0;
	so->resultCount = 0;

	/* Extract query vector from orderbys */
	if (norderbys > 0 && orderbys[0].sk_argument != 0)
	{
		Vector	   *vec = NULL;
		float4	   *vectorData = NULL;
		int			dim;

		queryType = TupleDescAttr(scan->indexRelation->rd_att, 0)->atttypid;
		oldctx = MemoryContextSwitchTo(so->scanCtx);

		vec = DatumGetVectorP(orderbys[0].sk_argument);
		NDB_CHECK_VECTOR_VALID(vec);
		dim = vec->dim;

		/* Copy vector data to scan context */
		nalloc(vectorData, float4, dim);
		memcpy(vectorData, vec->data, sizeof(float4) * dim);
		so->query = vectorData;
		so->queryDim = dim;

		MemoryContextSwitchTo(oldctx);
		so->k = 10;		/* Default, could be from GUC */
	}
}

/*
 * PQ index end scan
 */
static void
pqendscan(IndexScanDesc scan)
{
	PqScanOpaque so = (PqScanOpaque) scan->opaque;

	if (so->scanCtx != NULL)
		MemoryContextDelete(so->scanCtx);

	pfree(so);
}

/*
 * Coarse PQ search - find candidates using PQ codes
 */
static void
pq_coarse_search(Relation index, PqMetaPage meta, PqScanOpaque so,
				 ItemPointer **coarse_tids, float **coarse_dists, int *coarse_count)
{
	Buffer		vectorBuffer;
	Page		vectorPage;
	BlockNumber blkno;
	ItemPointer *tids = NULL;
	float	   *dists = NULL;
	int			count = 0;
	int			capacity = so->rerank_k * 2;	/* Get more candidates than needed */
	int			i;
	uint8_t    *query_codes = NULL;
	float	  **codebook_dists = NULL;	/* Precomputed distances from query to codebooks */
	MemoryContext oldctx;

	oldctx = MemoryContextSwitchTo(so->scanCtx);

	/* Encode query vector */
	query_codes = encode_vector_pq_single(so->query, so->queryDim, so->codebooks,
										  so->m, so->ks, so->subspace_dim, so->scanCtx);

	/* Precompute distances from query to all codebook centroids */
	nalloc(codebook_dists, float *, so->m);
	for (i = 0; i < so->m; i++)
	{
		float	   *sub_dists = NULL;
		int			c, d;

		nalloc(sub_dists, float, so->ks);
		for (c = 0; c < so->ks; c++)
		{
			float		dist = 0.0f;
			int			start_dim = i * so->subspace_dim;

			for (d = 0; d < so->subspace_dim; d++)
			{
				float		diff = so->query[start_dim + d] - so->codebooks[i][c][d];
				dist += diff * diff;
			}
			sub_dists[c] = dist;
		}
		codebook_dists[i] = sub_dists;
	}

	/* Allocate result arrays */
	nalloc(tids, ItemPointer, capacity);
	nalloc(dists, float, capacity);

	/* Scan all index pages and compute PQ distances */
	/* For now, scan sequentially - full implementation would use more efficient approach */
	for (blkno = 1; blkno < RelationGetNumberOfBlocks(index); blkno++)
	{
		OffsetNumber maxoff;
		OffsetNumber offnum;

		vectorBuffer = ReadBuffer(index, blkno);
		LockBuffer(vectorBuffer, BUFFER_LOCK_SHARE);
		vectorPage = BufferGetPage(vectorBuffer);

		if (PageIsNew(vectorPage) || PageIsEmpty(vectorPage))
		{
			UnlockReleaseBuffer(vectorBuffer);
			continue;
		}

		maxoff = PageGetMaxOffsetNumber(vectorPage);
		for (offnum = FirstOffsetNumber; offnum <= maxoff; offnum = OffsetNumberNext(offnum))
		{
			ItemId		itemId = PageGetItemId(vectorPage, offnum);
			PqCodeEntry *entry = NULL;
			float		pq_dist = 0.0f;
			int			sub;

			if (!ItemIdIsValid(itemId) || !ItemIdHasStorage(itemId))
				continue;

			entry = (PqCodeEntry *) PageGetItem(vectorPage, itemId);

			/* Compute PQ distance using precomputed codebook distances */
			for (sub = 0; sub < so->m; sub++)
			{
				uint8_t		code = entry->codes[sub];
				pq_dist += codebook_dists[sub][code];
			}

			/* Add to candidates if within capacity or better than worst */
			if (count < capacity)
			{
				tids[count] = entry->heapPtr;
				dists[count] = pq_dist;
				count++;
			}
			else
			{
				/* Find worst candidate and replace if better */
				int			worst_idx = 0;
				float		worst_dist = dists[0];

				for (i = 1; i < count; i++)
				{
					if (dists[i] > worst_dist)
					{
						worst_dist = dists[i];
						worst_idx = i;
					}
				}

				if (pq_dist < worst_dist)
				{
					tids[worst_idx] = entry->heapPtr;
					dists[worst_idx] = pq_dist;
				}
			}
		}

		UnlockReleaseBuffer(vectorBuffer);
	}

	/* Sort by distance and take top rerank_k */
	if (count > so->rerank_k)
	{
		/* Simple selection sort for top-k */
		int			i, j;

		for (i = 0; i < so->rerank_k; i++)
		{
			int			best_idx = i;
			float		best_dist = dists[i];

			for (j = i + 1; j < count; j++)
			{
				if (dists[j] < best_dist)
				{
					best_dist = dists[j];
					best_idx = j;
				}
			}

			if (best_idx != i)
			{
				/* Swap */
				ItemPointer temp_tid = tids[i];
				float		temp_dist = dists[i];

				tids[i] = tids[best_idx];
				dists[i] = dists[best_idx];
				tids[best_idx] = temp_tid;
				dists[best_idx] = temp_dist;
			}
		}
		count = so->rerank_k;
	}

	*coarse_tids = tids;
	*coarse_dists = dists;
	*coarse_count = count;

	MemoryContextSwitchTo(oldctx);
}

/*
 * Fine rerank - compute exact distances for top candidates by fetching
 * full vectors from the heap and computing exact L2 distance.
 */
static void
pq_fine_rerank(Relation index, PqMetaPage meta, PqScanOpaque so,
			   ItemPointer *coarse_tids, float *coarse_dists, int coarse_count)
{
	ItemPointerData *tid_storage = NULL;
	ItemPointer *results = NULL;
	float	   *distances = NULL;
	Relation	heapRel = NULL;
	Snapshot	snapshot;
	TupleDesc	heapTupdesc;
	int			vec_attnum;
	int			i;
	int			n;
	MemoryContext oldctx;

	oldctx = MemoryContextSwitchTo(so->scanCtx);

	n = Min(coarse_count, so->k);
	if (n <= 0)
	{
		so->results = NULL;
		so->distances = NULL;
		so->resultCount = 0;
		MemoryContextSwitchTo(oldctx);
		return;
	}

	tid_storage = (ItemPointerData *) palloc(n * sizeof(ItemPointerData));
	nalloc(results, ItemPointer, so->k);
	nalloc(distances, float, so->k);
	for (i = 0; i < so->k; i++)
		results[i] = &tid_storage[i < n ? i : 0];

	/* Get heap relation for the index */
	heapRel = NULL;
	{
		Oid			heapOid = IndexGetRelation(index->rd_id, false);

		if (!OidIsValid(heapOid))
			goto use_coarse;
		heapRel = table_open(heapOid, AccessShareLock);
		if (!RelationIsValid(heapRel))
		{
			heapRel = NULL;
			goto use_coarse;
		}
	}

	snapshot = GetActiveSnapshot();
	heapTupdesc = RelationGetDescr(heapRel);
	vec_attnum = index->rd_index->indkey[0];
	if (vec_attnum <= 0)
	{
		table_close(heapRel, AccessShareLock);
		goto use_coarse;
	}

	for (i = 0; i < n; i++)
	{
		HeapTupleData tupleData;
		HeapTuple	tuple = &tupleData;
		Buffer		heapBuf;
		bool		found;
		bool		isnull;
		Datum		vec_datum;
		Vector	   *vec = NULL;
		double		dist_sq;

		ItemPointerCopy(coarse_tids[i], &tid_storage[i]);
		ItemPointerCopy(&tid_storage[i], &tupleData.t_self);
		found = heap_fetch(heapRel, snapshot, tuple, &heapBuf, false);
		if (!found || !HeapTupleIsValid(tuple))
		{
			distances[i] = (float) coarse_dists[i];
			continue;
		}

		vec_datum = heap_getattr(tuple, vec_attnum, heapTupdesc, &isnull);
		if (isnull)
		{
			distances[i] = (float) coarse_dists[i];
			ReleaseBuffer(heapBuf);
			continue;
		}

		vec = DatumGetVectorP(vec_datum);
		if (vec == NULL || vec->dim != so->queryDim)
		{
			distances[i] = (float) coarse_dists[i];
			ReleaseBuffer(heapBuf);
			continue;
		}

		dist_sq = neurondb_l2_distance_squared(so->query, vec->data, so->queryDim);
		distances[i] = (float) sqrt(dist_sq);
		ReleaseBuffer(heapBuf);
	}

	table_close(heapRel, AccessShareLock);

	/* Sort by exact distance */
	for (i = 0; i < n - 1; i++)
	{
		int			j;
		int			best_idx = i;

		for (j = i + 1; j < n; j++)
		{
			if (distances[j] < distances[best_idx])
				best_idx = j;
		}

		if (best_idx != i)
		{
			ItemPointerData temp_tid;
			float		temp_dist;

			ItemPointerCopy(results[i], &temp_tid);
			ItemPointerCopy(results[best_idx], results[i]);
			ItemPointerCopy(&temp_tid, results[best_idx]);
			temp_dist = distances[i];
			distances[i] = distances[best_idx];
			distances[best_idx] = temp_dist;
		}
	}

	so->results = results;
	so->distances = distances;
	so->resultCount = n;
	MemoryContextSwitchTo(oldctx);
	return;

use_coarse:
	/* Fallback: use coarse distances when heap is unavailable */
	for (i = 0; i < n; i++)
	{
		ItemPointerCopy(coarse_tids[i], &tid_storage[i]);
		distances[i] = (float) coarse_dists[i];
	}
	so->results = results;
	so->distances = distances;
	so->resultCount = n;
	if (heapRel != NULL && RelationIsValid(heapRel))
		table_close(heapRel, AccessShareLock);
	MemoryContextSwitchTo(oldctx);
}

/*
 * PQ index get tuple
 */
static bool
pqgettuple(IndexScanDesc scan, ScanDirection dir)
{
	PqScanOpaque so = (PqScanOpaque) scan->opaque;
	Buffer		metaBuffer;
	Page		metaPage;
	PqMetaPage meta;
	ItemPointer *coarse_tids = NULL;
	float	   *coarse_dists = NULL;
	int			coarse_count = 0;

	if (so->firstCall)
	{
		/* Read metadata and load codebooks */
		metaBuffer = ReadBuffer(scan->indexRelation, 0);
		LockBuffer(metaBuffer, BUFFER_LOCK_SHARE);
		metaPage = BufferGetPage(metaBuffer);
		meta = (PqMetaPage) PageGetContents(metaPage);

		if (meta->magicNumber != PQ_MAGIC_NUMBER)
		{
			UnlockReleaseBuffer(metaBuffer);
			return false;
		}

		so->m = meta->m;
		so->ks = meta->ks;
		so->subspace_dim = meta->subspace_dim;

		/* Load codebooks */
		so->codebooks = load_codebooks(scan->indexRelation, meta, so->scanCtx);

		if (!so->query)
		{
			UnlockReleaseBuffer(metaBuffer);
			return false;
		}

		/* Perform two-stage search */
		pq_coarse_search(scan->indexRelation, meta, so, &coarse_tids, &coarse_dists, &coarse_count);
		pq_fine_rerank(scan->indexRelation, meta, so, coarse_tids, coarse_dists, coarse_count);

		UnlockReleaseBuffer(metaBuffer);
		so->firstCall = false;
		so->currentResult = 0;
	}

	if (so->currentResult < so->resultCount)
	{
		/* Return next result */
		scan->xs_heaptid = so->results[so->currentResult];
		so->currentResult++;
		return true;
	}

	return false;
}

/*
 * PQ index bulk delete
 */
static IndexBulkDeleteResult *
pqbulkdelete(IndexVacuumInfo * info,
			 IndexBulkDeleteResult * stats,
			 IndexBulkDeleteCallback callback,
			 void *callback_state)
{
	Relation	index = info->index;
	BlockNumber blkno;
	Buffer		metaBuffer;
	Page		metaPage;
	PqMetaPage meta;
	Buffer		vectorBuffer;
	Page		vectorPage;
	OffsetNumber maxoff;
	OffsetNumber offnum;
	PqCodeEntry *entry = NULL;
	ItemId		itemId;
	IndexBulkDeleteResult *new_stats = NULL;
	int			m;
	Size		entry_size;

	if (stats == NULL)
	{
		nalloc(new_stats, IndexBulkDeleteResult, 1);
		stats = new_stats;
		stats->num_index_tuples = 0;
		stats->tuples_removed = 0;
	}

	/* Read metadata */
	metaBuffer = ReadBuffer(index, 0);
	LockBuffer(metaBuffer, BUFFER_LOCK_SHARE);
	metaPage = BufferGetPage(metaBuffer);
	meta = (PqMetaPage) PageGetContents(metaPage);
	m = meta->m;
	entry_size = PqCodeEntrySize(m);
	UnlockReleaseBuffer(metaBuffer);

	/* Scan all index pages (skip block 0 which is metadata) */
	for (blkno = 1; blkno < RelationGetNumberOfBlocks(index); blkno++)
	{
		vectorBuffer = ReadBuffer(index, blkno);
		LockBuffer(vectorBuffer, BUFFER_LOCK_EXCLUSIVE);
		vectorPage = BufferGetPage(vectorBuffer);

		if (PageIsNew(vectorPage) || PageIsEmpty(vectorPage))
		{
			UnlockReleaseBuffer(vectorBuffer);
			continue;
		}

		maxoff = PageGetMaxOffsetNumber(vectorPage);
		for (offnum = FirstOffsetNumber; offnum <= maxoff; offnum = OffsetNumberNext(offnum))
		{
			itemId = PageGetItemId(vectorPage, offnum);

			if (!ItemIdIsValid(itemId) || ItemIdIsDead(itemId))
				continue;

			if (!ItemIdHasStorage(itemId))
				continue;

			entry = (PqCodeEntry *) PageGetItem(vectorPage, itemId);

			/* Check if TID should be deleted */
			if (callback(&entry->heapPtr, callback_state))
			{
				/* Mark item as dead */
				ItemIdMarkDead(itemId);
				MarkBufferDirty(vectorBuffer);
				stats->tuples_removed++;
			}
			else
			{
				stats->num_index_tuples++;
			}
		}

		UnlockReleaseBuffer(vectorBuffer);
	}

	return stats;
}

/*
 * PQ index vacuum cleanup
 */
static IndexBulkDeleteResult *
pqvacuumcleanup(IndexVacuumInfo * info, IndexBulkDeleteResult * stats)
{
	Relation	index = info->index;
	Buffer		metaBuffer;
	Page		metaPage;
	PqMetaPage meta;
	BlockNumber blkno;
	int			pages_deleted = 0;

	if (stats == NULL)
		return NULL;

	/* Update metadata with new count */
	metaBuffer = ReadBuffer(index, 0);
	LockBuffer(metaBuffer, BUFFER_LOCK_EXCLUSIVE);
	metaPage = BufferGetPage(metaBuffer);
	meta = (PqMetaPage) PageGetContents(metaPage);

	/* Update inserted vectors count (subtract removed) */
	if (stats->tuples_removed > 0)
	{
		if (meta->insertedVectors >= (int64) stats->tuples_removed)
			meta->insertedVectors -= stats->tuples_removed;
		else
			meta->insertedVectors = 0;
	}

	MarkBufferDirty(metaBuffer);
	UnlockReleaseBuffer(metaBuffer);

	/* Try to reclaim empty pages at the end */
	/* Full implementation would compact pages and update free space map */
	for (blkno = RelationGetNumberOfBlocks(index) - 1; blkno > 0; blkno--)
	{
		Buffer		buf = ReadBuffer(index, blkno);
		Page		page = BufferGetPage(buf);

		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

		if (PageIsNew(page) || PageIsEmpty(page))
		{
			/* Page is empty, can be truncated */
			/* Full implementation would use RelationTruncate or similar */
			UnlockReleaseBuffer(buf);
			pages_deleted++;
		}
		else
		{
			UnlockReleaseBuffer(buf);
			break;		/* Found non-empty page, stop */
		}
	}

	stats->pages_deleted = pages_deleted;
	stats->pages_free = 0;	/* Would be computed in full implementation */

	return stats;
}

/*
 * PQ index cost estimation
 */
static void
pqcostestimate(struct PlannerInfo *root, struct IndexPath *path, double loop_count,
			   Cost * indexStartupCost, Cost * indexTotalCost,
			   Selectivity * indexSelectivity, double *indexCorrelation,
			   double *indexPages)
{
	Relation	index = path->indexinfo->index;
	Buffer		metaBuffer;
	Page		metaPage;
	PqMetaPage meta;
	double		indexSelectivityEstimate;
	double		numIndexPages;
	double		numIndexTuples;
	Cost		indexStartup;
	Cost		indexTotal;
	PqOptions	options;

	/* Load options to get rerank_k */
	pqLoadOptions(index, &options);

	/* Read metadata */
	metaBuffer = ReadBuffer(index, 0);
	LockBuffer(metaBuffer, BUFFER_LOCK_SHARE);
	metaPage = BufferGetPage(metaBuffer);
	meta = (PqMetaPage) PageGetContents(metaPage);
	numIndexTuples = (double) meta->insertedVectors;
	UnlockReleaseBuffer(metaBuffer);

	/* Estimate number of index pages */
	numIndexPages = RelationGetNumberOfBlocks(index);

	/* Estimate selectivity - PQ is approximate, assume high selectivity */
	indexSelectivityEstimate = 0.01;	/* 1% selectivity estimate */

	/* Cost model for two-stage PQ search:
	 * - Startup: Load codebooks (1 page read)
	 * - Per-tuple: Coarse PQ search (scan all pages) + Fine rerank (fetch rerank_k vectors)
	 */
	indexStartup = 1.0 * random_page_cost;	/* Codebook page read */

	/* Coarse search: scan all index pages */
	/* Fine rerank: fetch rerank_k vectors from heap */
	indexTotal = indexStartup +
		(numIndexPages * seq_page_cost) +	/* Coarse search page reads */
		(options.rerank_k * (random_page_cost + cpu_index_tuple_cost));	/* Fine rerank */

	*indexStartupCost = indexStartup;
	*indexTotalCost = indexTotal * loop_count;
	*indexSelectivity = indexSelectivityEstimate;
	*indexCorrelation = 0.0;	/* No correlation assumption */
	*indexPages = numIndexPages;
}



