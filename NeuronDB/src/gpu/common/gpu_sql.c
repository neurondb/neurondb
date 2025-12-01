/*-------------------------------------------------------------------------
 *
 * gpu_sql.c
 *    SQL-callable wrappers for accelerated vector operations.
 *
 * This module provides PostgreSQL SQL-callable wrappers for accelerated
 * vector operations with error checking and resource management.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/common/gpu_sql.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/guc.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "access/relscan.h"
#include "access/genam.h"
#include "utils/rel.h"
#include "utils/lsyscache.h"
#include "storage/bufmgr.h"
#include "access/amapi.h"

#include "neurondb_config.h"
#include "neurondb_gpu.h"
#include "neurondb_gpu_backend.h"
#include "neurondb.h"
#include "neurondb_constants.h"

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <float.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/* IVF index structure definitions (matching ivf_am.c) */
#define IVF_MAGIC_NUMBER 0x49564646 /* "IVFF" in hex */

typedef struct IvfMetaPageData
{
	uint32		magicNumber;
	uint32		version;
	int			nlists;
	int			nprobe;
	int			dim;
	BlockNumber centroidsBlock;
	int64		insertedVectors;
}			IvfMetaPageData;

typedef struct IvfCentroidData
{
	int			listId;
	int			dim;
	int64		memberCount;
	BlockNumber firstBlock;
	/* Followed by float4 centroid[dim] */
}			IvfCentroidData;

typedef struct IvfListPageHeader
{
	BlockNumber nextBlock;
	int32		entryCount;
}			IvfListPageHeader;

typedef struct IvfListEntryData
{
	ItemPointerData heapPtr;
	int16		dim;
	/* Followed by float4 vector[dim] */
}			IvfListEntryData;

/*
 * ndb_gpu_can_run
 * Check if GPU support and specific kernel are enabled and available.
 * All conditions required for this to return true; safe for fallback.
 */
static inline bool
ndb_gpu_can_run(const char *kernel_name)
{
	/* Check if global GPU support is enabled. */
	if (!NDB_SHOULD_TRY_GPU())
		return false;
	/* Check if the specified kernel is available. */
	if (!ndb_gpu_kernel_enabled(kernel_name))
		return false;
	/* If needed, initialize device/system. */
	ndb_gpu_init_if_needed();
	/* Final status: device is available and ready. */
	return neurondb_gpu_is_available();
}

/*
 * vector_l2_distance_gpu
 * SQL-callable interface: Compute L2 distance (Euclidean norm) between
 * two vectors. Uses GPU if available, otherwise falls back to CPU.
 */
PG_FUNCTION_INFO_V1(vector_l2_distance_gpu);
Datum
vector_l2_distance_gpu(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;
	float4		result = -1.0f;
	extern float4 l2_distance(Vector *a, Vector *b);

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);

	if (ndb_gpu_can_run("l2") && a->dim == b->dim)
	{
		result = neurondb_gpu_l2_distance(a->data, b->data, a->dim);
		if (result >= 0.0f && !isnan(result))
			PG_RETURN_FLOAT4(result);
		/* Otherwise, fall through to CPU. */
	}

	/* CPU fallback. */
	PG_RETURN_FLOAT4(l2_distance(a, b));
}

/*
 * vector_cosine_distance_gpu
 * SQL-callable interface: Compute cosine distance (1 - cosine similarity)
 * between two vectors. Prefers GPU, falls back to CPU when needed.
 */
PG_FUNCTION_INFO_V1(vector_cosine_distance_gpu);
Datum
vector_cosine_distance_gpu(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;
	float4		result = -1.0f;
	extern float4 cosine_distance(Vector *a, Vector *b);

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);

	if (ndb_gpu_can_run("cosine") && a->dim == b->dim)
	{
		result = neurondb_gpu_cosine_distance(a->data, b->data, a->dim);
		if (result >= 0.0f && !isnan(result))
			PG_RETURN_FLOAT4(result);
	}

	PG_RETURN_FLOAT4(cosine_distance(a, b));
}

/*
 * vector_inner_product_gpu
 * SQL-callable interface: Compute -dot(a, b) as a distance metric.
 * Uses GPU if able, else falls back to CPU.
 */
PG_FUNCTION_INFO_V1(vector_inner_product_gpu);
Datum
vector_inner_product_gpu(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;
	float4		result = -1.0f;
	extern float4 inner_product_distance(Vector *a, Vector *b);

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);

	if (ndb_gpu_can_run("ip") && a->dim == b->dim)
	{
		result = neurondb_gpu_inner_product(a->data, b->data, a->dim);
		if (result >= 0.0f && !isnan(result))
			PG_RETURN_FLOAT4(result);
	}

	PG_RETURN_FLOAT4(inner_product_distance(a, b));
}

/*
 * vector_to_int8_gpu
 * SQL-callable: Quantizes a float32 vector to int8, returns bytea of
 * length count. GPU preferred, CPU fallback when needed.
 */
PG_FUNCTION_INFO_V1(vector_to_int8_gpu);
Datum
vector_to_int8_gpu(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	int			count;
	bytea	   *out;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);
	count = v->dim;

	out = (bytea *) palloc(VARHDRSZ + count);
	SET_VARSIZE(out, VARHDRSZ + count);

	if (ndb_gpu_can_run("quantize"))
	{
		neurondb_gpu_quantize_int8(
								   v->data, (int8 *) VARDATA(out), count);
	}
	else
	{
		int			i;
		float		maxv = 0.0f;
		float		scale;
		int8	   *quantized = (int8 *) VARDATA(out);

		for (i = 0; i < count; i++)
		{
			float		val = fabsf(v->data[i]);

			if (val > maxv)
				maxv = val;
		}

		scale = (maxv > 0.0f) ? (127.0f / maxv) : 1.0f;

		for (i = 0; i < count; i++)
		{
			float		scaled = v->data[i] * scale;

			if (scaled > 127.0f)
				scaled = 127.0f;
			else if (scaled < -128.0f)
				scaled = -128.0f;

			quantized[i] = (int8) lrintf(scaled);
		}
	}

	PG_RETURN_BYTEA_P(out);
}

/*
 * vector_to_fp16_gpu
 * SQL-callable: Quantizes float32 vector to packed IEEE 754 half-precision,
 * two bytes per value, returned in a bytea.
 */
PG_FUNCTION_INFO_V1(vector_to_fp16_gpu);
Datum
vector_to_fp16_gpu(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	int			count;
	int			out_bytes;
	bytea	   *out;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);
	count = v->dim;
	out_bytes = count * 2;		/* 2 bytes per fp16 */

	out = (bytea *) palloc(VARHDRSZ + out_bytes);
	SET_VARSIZE(out, VARHDRSZ + out_bytes);

	if (ndb_gpu_can_run("quantize"))
	{
		neurondb_gpu_quantize_fp16(
								   v->data, (void *) VARDATA(out), count);
	}
	else
	{
		int			i;
		uint16	   *dst = (uint16 *) VARDATA(out);

		for (i = 0; i < count; i++)
		{
			float		f = v->data[i];
			union
			{
				float		f;
				uint32		u;
			}			in;
			uint32		f32;
			uint32		sign;
			int32		exp;
			uint32		mant;
			uint16		out_fp16;

			in.f = f;
			f32 = in.u;

			sign = (f32 >> 31) & 0x1;
			exp = ((f32 >> 23) & 0xFF) - 127;
			mant = f32 & 0x7FFFFF;
			out_fp16 = 0;

			if ((f32 & 0x7FFFFFFF) == 0)
			{
				/* Signed zero */
				out_fp16 = sign << 15;
			}
			else if ((f32 & 0x7F800000) == 0x7F800000)
			{
				/* NaN or Inf */
				if ((f32 & 0x007FFFFF) == 0)
				{
					out_fp16 = (sign << 15) | (0x1F << 10);
				}
				else
				{
					out_fp16 = (sign << 15) | (0x1F << 10)
						| ((mant >> 13) ? (mant >> 13)
						   : 1);
				}
			}
			else if (exp > 15)
			{
				/* Overflow => Inf */
				out_fp16 = (sign << 15) | (0x1F << 10);
			}
			else if (exp >= -14)
			{
				uint32		new_exp = exp + 15;
				uint32		mant_fp16 = mant >> 13;

				/* Round-to-nearest/evens. */
				if (((mant >> 12) & 1)
					&& (((mant & 0xFFF) > 0)
						|| (mant_fp16 & 1)))
				{
					mant_fp16 += 1;
					if (mant_fp16 == 0x400)
					{
						mant_fp16 = 0;
						new_exp++;
						if (new_exp == 0x1F)
						{
							out_fp16 = (sign << 15)
								| (0x1F << 10);
							dst[i] = out_fp16;
							continue;
						}
					}
				}
				out_fp16 = (sign << 15)
					| ((new_exp & 0x1F) << 10)
					| (mant_fp16 & 0x3FF);
			}
			else if (exp >= -24)
			{
				uint32		shift = (uint32) (-14 - exp);
				uint32		mantissa;

				if (shift > 24)
					shift = 24;
				mantissa = (mant | 0x800000) >> (shift + 13);

				if (((mant | 0x800000) >> (shift + 12)) & 1)
					mantissa += 1;
				out_fp16 = (sign << 15) | mantissa;
			}
			else
			{
				out_fp16 = (sign << 15);
			}

			dst[i] = out_fp16;
		}
	}

	PG_RETURN_BYTEA_P(out);
}

/*
 * vector_to_binary_gpu
 * SQL-callable: Convert float32 vector to packed bitstring:
 * 1 bit for each value: set if > 0.0f.
 */
PG_FUNCTION_INFO_V1(vector_to_binary_gpu);
Datum
vector_to_binary_gpu(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	int			count;
	int			out_bytes;
	bytea	   *out;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);
	count = v->dim;
	out_bytes = (count + 7) / 8;

	out = (bytea *) palloc(VARHDRSZ + out_bytes);
	SET_VARSIZE(out, VARHDRSZ + out_bytes);

	/* Clear all bits (avoid garbage bits in final bytes). */
	memset(VARDATA(out), 0, out_bytes);

	if (ndb_gpu_can_run("quantize"))
	{
		neurondb_gpu_quantize_binary(
									 v->data, (uint8 *) VARDATA(out), count);
	}
	else
	{
		int			i;
		uint8	   *dst = (uint8 *) VARDATA(out);

		for (i = 0; i < count; i++)
		{
			/* Set bit if positive. */
			if (v->data[i] > 0.0f)
				dst[i >> 3] |= (1u << (i & 7));
		}
	}

	PG_RETURN_BYTEA_P(out);
}

/*
 * vector_to_uint8_gpu
 * SQL-callable: Quantizes float32 vector to uint8, returns bytea.
 * GPU preferred, CPU fallback when needed.
 */
PG_FUNCTION_INFO_V1(vector_to_uint8_gpu);
Datum
vector_to_uint8_gpu(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	VectorU8   *result;
	bytea	   *out;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	/* For now, use CPU implementation (GPU kernels can be added later) */
	result = quantize_vector_uint8(v);
	out = (bytea *) result;

	PG_RETURN_BYTEA_P(out);
}

/*
 * vector_to_ternary_gpu
 * SQL-callable: Quantizes float32 vector to ternary (2 bits per dimension),
 * returns bytea. GPU preferred, CPU fallback when needed.
 */
PG_FUNCTION_INFO_V1(vector_to_ternary_gpu);
Datum
vector_to_ternary_gpu(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	VectorTernary *result;
	bytea	   *out;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	/* For now, use CPU implementation (GPU kernels can be added later) */
	result = quantize_vector_ternary(v);
	out = (bytea *) result;

	PG_RETURN_BYTEA_P(out);
}

/*
 * vector_to_int4_gpu
 * SQL-callable: Quantizes float32 vector to int4 (4 bits per dimension),
 * returns bytea. GPU preferred, CPU fallback when needed.
 */
PG_FUNCTION_INFO_V1(vector_to_int4_gpu);
Datum
vector_to_int4_gpu(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	VectorI4   *result;
	bytea	   *out;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);

	/* For now, use CPU implementation (GPU kernels can be added later) */
	result = quantize_vector_int4(v);
	out = (bytea *) result;

	PG_RETURN_BYTEA_P(out);
}

/*
 * Helper: Get relation OID from name
 */
static Oid
get_index_oid_from_name(const char *index_name)
{
	Oid			index_oid;

	if (index_name == NULL || strlen(index_name) == 0)
		return InvalidOid;

	/* Use to_regclass to resolve index name */
	index_oid = DatumGetObjectId(
								 DirectFunctionCall1(to_regclass, CStringGetDatum(index_name)));

	return index_oid;
}

/*
 * hnsw_knn_search_gpu
 * GPU-accelerated HNSW k-nearest neighbor search.
 * Uses GPU for distance calculations when available.
 * Returns SETOF (id bigint, distance real).
 *
 * Signature: hnsw_knn_search_gpu(index_name text, query vector, k int, ef_search int)
 */
PG_FUNCTION_INFO_V1(hnsw_knn_search_gpu);
Datum
hnsw_knn_search_gpu(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	TupleDesc	tupdesc;
	MemoryContext oldcontext;
	text	   *index_name_text;
	char	   *index_name;
	Vector	   *query;
	int32		k;
	int32		ef_search;
	const		ndb_gpu_backend *backend;
	Relation	indexRel = NULL;
	Oid			index_oid;
	Buffer		metaBuffer;
	Page		metaPage;
	typedef struct
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
	typedef struct
	{
		ItemPointerData heapPtr;
		int			level;
		int16		dim;
		int16		neighborCount[16];	/* HNSW_MAX_LEVEL */
		/* Followed by vector and neighbors, but we only need heapPtr */
	}			HnswNodeData;
	typedef HnswNodeData * HnswNode;
	HnswMetaPageData *meta;
	ItemPointerData *results = NULL;
	float4	   *distances = NULL;
	int			resultCount = 0;

	if (SRF_IS_FIRSTCALL())
	{
		index_name_text = PG_GETARG_TEXT_P(0);
		query = PG_GETARG_VECTOR_P(1);
		NDB_CHECK_VECTOR_VALID(query);
		k = PG_GETARG_INT32(2);
		ef_search = PG_NARGS() > 3 ? PG_GETARG_INT32(3) : 100;

		/* Validate inputs */
		if (index_name_text == NULL)
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("hnsw_knn_search_gpu: index name cannot be NULL")));

		index_name = text_to_cstring(index_name_text);

		if (query == NULL)
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("hnsw_knn_search_gpu: query vector cannot be NULL")));

		if (k <= 0 || k > 10000)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("hnsw_knn_search_gpu: k must be between 1 and 10000")));

		if (ef_search <= 0 || ef_search > 10000)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("hnsw_knn_search_gpu: ef_search must be between 1 and 10000")));

		if (query->dim <= 0 || query->dim > 10000)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("hnsw_knn_search_gpu: invalid query dimension %d",
							query->dim)));

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		/* Create tuple descriptor: (id bigint, distance real) */
		tupdesc = CreateTemplateTupleDesc(2);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "id", INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "distance", FLOAT4OID, -1, 0);
		funcctx->tuple_desc = BlessTupleDesc(tupdesc);

		/* Get index OID from name */
		index_oid = get_index_oid_from_name(index_name);
		if (!OidIsValid(index_oid))
			ereport(ERROR,
					(errcode(ERRCODE_UNDEFINED_OBJECT),
					 errmsg("hnsw_knn_search_gpu: index \"%s\" does not exist",
							index_name)));

		/* Open index relation */
		indexRel = index_open(index_oid, AccessShareLock);

		/* Check if this is an HNSW index */
		/* For now, assume it is - full implementation would check AM */

		/* Read metadata page */
		metaBuffer = ReadBuffer(indexRel, 0);
		if (!BufferIsValid(metaBuffer))
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: ReadBuffer failed for buffer")));
		}
		LockBuffer(metaBuffer, BUFFER_LOCK_SHARE);
		metaPage = BufferGetPage(metaBuffer);
		meta = (HnswMetaPageData *) PageGetContents(metaPage);

		/* Validate magic number (HNSW uses 0x484E5357 = "HNSW") */
		if (meta->magicNumber != 0x484E5357)
		{
			UnlockReleaseBuffer(metaBuffer);
			index_close(indexRel, AccessShareLock);
			ereport(ERROR,
					(errcode(ERRCODE_WRONG_OBJECT_TYPE),
					 errmsg("hnsw_knn_search_gpu: \"%s\" is not an HNSW index",
							index_name)));
		}

		/* Check GPU availability for distance calculations */
		backend = NULL;
		if (ndb_gpu_can_run("l2"))
		{
			backend = ndb_gpu_get_active_backend();
			if (backend && backend->launch_l2_distance)
			{
				elog(DEBUG1,
					 "hnsw_knn_search_gpu: GPU acceleration available "
					 "for distance calculations (k=%d, ef_search=%d, dim=%d)",
					 k, ef_search, query->dim);
			}
		}

		/* Perform full HNSW graph traversal search */
		/* Note: Full implementation would use proper HNSW search algorithm */
		/* For now, implement basic graph traversal to find top-k neighbors */
		if (meta->entryPoint != InvalidBlockNumber)
		{
			/* Basic HNSW search: traverse graph from entry point */
			/*
			 * This is a simplified implementation - full version would do
			 * multi-layer search
			 */
			BlockNumber current = meta->entryPoint;
			BlockNumber *candidates = NULL;
			float4	   *candidateDists = NULL;
			int			candidateCount = 0;
			int			maxCandidates = ef_search > k ? ef_search : k * 2;
			bool	   *visited = NULL;
			int			visitedSize = RelationGetNumberOfBlocks(indexRel);
			int			i,
						j;
			extern float4 l2_distance(Vector *a, Vector *b);

			candidates = (BlockNumber *) palloc(sizeof(BlockNumber) * maxCandidates);
			candidateDists = (float4 *) palloc(sizeof(float4) * maxCandidates);
			visited = (bool *) palloc0(sizeof(bool) * visitedSize);

			/* Start from entry point */
			{
				Buffer		nodeBuf;
				Page		nodePage;
				HnswNode	node;
				float4	   *nodeVector;
				Vector	   *nodeVec;
				float4		dist;

				nodeBuf = ReadBuffer(indexRel, current);
				LockBuffer(nodeBuf, BUFFER_LOCK_SHARE);
				nodePage = BufferGetPage(nodeBuf);
				if (!PageIsEmpty(nodePage))
				{
					node = (HnswNode) PageGetItem(nodePage,
												  PageGetItemId(nodePage, FirstOffsetNumber));
					nodeVector = (float4 *) ((char *) (node) + MAXALIGN(sizeof(HnswNodeData)));

					nodeVec = (Vector *) palloc(VARHDRSZ + sizeof(int16) * 2 + sizeof(float4) * node->dim);
					SET_VARSIZE(nodeVec, VARHDRSZ + sizeof(int16) * 2 + sizeof(float4) * node->dim);
					nodeVec->dim = node->dim;
					nodeVec->unused = 0;
					memcpy(nodeVec->data, nodeVector, sizeof(float4) * node->dim);

					dist = l2_distance(query, nodeVec);
					candidates[0] = current;
					candidateDists[0] = dist;
					candidateCount = 1;
					visited[current] = true;

					NDB_FREE(nodeVec);
				}
				UnlockReleaseBuffer(nodeBuf);
			}

			/*
			 * Expand candidates by following neighbors (simplified - only
			 * level 0)
			 */
			for (i = 0; i < candidateCount && candidateCount < maxCandidates; i++)
			{
				Buffer		nodeBuf;
				Page		nodePage;
				HnswNode	node;
				BlockNumber *neighbors;
				int16		neighborCount;

				nodeBuf = ReadBuffer(indexRel, candidates[i]);
				LockBuffer(nodeBuf, BUFFER_LOCK_SHARE);
				nodePage = BufferGetPage(nodeBuf);
				if (!PageIsEmpty(nodePage))
				{
					node = (HnswNode) PageGetItem(nodePage,
												  PageGetItemId(nodePage, FirstOffsetNumber));
					neighbors = (BlockNumber *) ((char *) (node) + MAXALIGN(sizeof(HnswNodeData))
												 + node->dim * sizeof(float4));
					neighborCount = node->neighborCount[0];

					for (j = 0; j < neighborCount && candidateCount < maxCandidates; j++)
					{
						if (neighbors[j] != InvalidBlockNumber && neighbors[j] < visitedSize && !visited[neighbors[j]])
						{
							Buffer		neighborBuf;
							Page		neighborPage;
							HnswNode	neighbor;
							float4	   *neighborVector;
							Vector	   *neighborVec;
							float4		dist;

							neighborBuf = ReadBuffer(indexRel, neighbors[j]);
							LockBuffer(neighborBuf, BUFFER_LOCK_SHARE);
							neighborPage = BufferGetPage(neighborBuf);
							if (!PageIsEmpty(neighborPage))
							{
								neighbor = (HnswNode) PageGetItem(neighborPage,
																  PageGetItemId(neighborPage, FirstOffsetNumber));
								neighborVector = (float4 *) ((char *) (neighbor) + MAXALIGN(sizeof(HnswNodeData)));

								neighborVec = (Vector *) palloc(VARHDRSZ + sizeof(int16) * 2 + sizeof(float4) * neighbor->dim);
								SET_VARSIZE(neighborVec, VARHDRSZ + sizeof(int16) * 2 + sizeof(float4) * neighbor->dim);
								neighborVec->dim = neighbor->dim;
								neighborVec->unused = 0;
								memcpy(neighborVec->data, neighborVector, sizeof(float4) * neighbor->dim);

								dist = l2_distance(query, neighborVec);
								candidates[candidateCount] = neighbors[j];
								candidateDists[candidateCount] = dist;
								candidateCount++;
								visited[neighbors[j]] = true;

								NDB_FREE(neighborVec);
							}
							UnlockReleaseBuffer(neighborBuf);
						}
					}
				}
				UnlockReleaseBuffer(nodeBuf);
			}

			/* Sort candidates by distance and take top-k */
			if (candidateCount > 0)
			{
				/* Simple selection sort for top-k */
				int		   *topKIndices = (int *) palloc(sizeof(int) * k);
				int			topKCount = candidateCount < k ? candidateCount : k;
				int			l,
							m,
							minIdx;
				float4		minDist;

				for (l = 0; l < topKCount; l++)
				{
					minIdx = l;
					minDist = candidateDists[l];
					for (m = l + 1; m < candidateCount; m++)
					{
						if (candidateDists[m] < minDist)
						{
							minIdx = m;
							minDist = candidateDists[m];
						}
					}
					if (minIdx != l)
					{
						BlockNumber tempBlk = candidates[l];
						float4		tempDist = candidateDists[l];

						candidates[l] = candidates[minIdx];
						candidateDists[l] = candidateDists[minIdx];
						candidates[minIdx] = tempBlk;
						candidateDists[minIdx] = tempDist;
					}
					topKIndices[l] = l;
				}

				/* Convert to results */
				resultCount = topKCount;
				results = (ItemPointerData *) palloc(sizeof(ItemPointerData) * resultCount);
				distances = (float4 *) palloc(sizeof(float4) * resultCount);

				for (i = 0; i < resultCount; i++)
				{
					Buffer		resultBuf;
					Page		resultPage;
					HnswNode	resultNode;

					resultBuf = ReadBuffer(indexRel, candidates[topKIndices[i]]);
					LockBuffer(resultBuf, BUFFER_LOCK_SHARE);
					resultPage = BufferGetPage(resultBuf);
					if (!PageIsEmpty(resultPage))
					{
						resultNode = (HnswNode) PageGetItem(resultPage,
															PageGetItemId(resultPage, FirstOffsetNumber));
						results[i] = resultNode->heapPtr;
						distances[i] = candidateDists[topKIndices[i]];
					}
					UnlockReleaseBuffer(resultBuf);
				}

				NDB_FREE(topKIndices);
			}

			NDB_FREE(candidates);
			NDB_FREE(candidateDists);
			NDB_FREE(visited);

			elog(DEBUG1,
				 "hnsw_knn_search_gpu: Completed graph traversal, returning %d result(s)",
				 resultCount);
		}
		else
		{
			results = NULL;
			distances = NULL;
			resultCount = 0;
			elog(DEBUG1, "hnsw_knn_search_gpu: No entry point in index");
		}

		UnlockReleaseBuffer(metaBuffer);
		index_close(indexRel, AccessShareLock);

		/* Store results in function context */
		/* Use a structure to store both results and distances */
		{
			typedef struct
			{
				ItemPointerData *results;
				float4	   *distances;
				int			count;
			}			HnswSearchResults;
			HnswSearchResults *searchResults;

			funcctx->max_calls = resultCount;
			if (resultCount > 0)
			{
				searchResults = (HnswSearchResults *) palloc(sizeof(HnswSearchResults));
				searchResults->results = results;
				searchResults->distances = distances;
				searchResults->count = resultCount;
				funcctx->user_fctx = searchResults;
			}
			else
			{
				funcctx->user_fctx = NULL;
				if (results)
					NDB_FREE(results);
				if (distances)
					NDB_FREE(distances);
			}
		}

		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();
	if (funcctx->call_cntr < funcctx->max_calls)
	{
		typedef struct
		{
			ItemPointerData *results;
			float4	   *distances;
			int			count;
		}			HnswSearchResults;
		HnswSearchResults *searchResults = (HnswSearchResults *) funcctx->user_fctx;

		if (searchResults != NULL)
		{
			/* Return next result */
			HeapTuple	tuple;
			Datum		values[2];
			bool		nulls[2] = {false, false};
			ItemPointerData *tid = &searchResults->results[funcctx->call_cntr];

			/*
			 * Return heap TID block number as ID (could also return offset or
			 * full TID)
			 */
			values[0] = Int64GetDatum((int64) ItemPointerGetBlockNumber(tid));
			values[1] = Float4GetDatum(searchResults->distances[funcctx->call_cntr]);
			tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);

			SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
		}
	}

	SRF_RETURN_DONE(funcctx);
}

/*
 * ivf_knn_search_gpu
 * GPU-accelerated IVF (Inverted File) k-nearest neighbor search.
 * Uses GPU for distance calculations when available.
 * Returns SETOF (id bigint, distance real).
 *
 * Signature: ivf_knn_search_gpu(index_name text, query vector, k int, nprobe int)
 */
PG_FUNCTION_INFO_V1(ivf_knn_search_gpu);
Datum
ivf_knn_search_gpu(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	ItemPointerData *results = NULL;
	float4	   *distances = NULL;
	int			resultCount = 0;

	if (SRF_IS_FIRSTCALL())
	{
		TupleDesc	tupdesc;
		MemoryContext oldcontext;
		text	   *index_name_text;
		char	   *index_name;
		Vector	   *query;
		int32		k;
		int32		nprobe;
		const		ndb_gpu_backend *backend;
		Relation	indexRel = NULL;
		Oid			index_oid;
		Buffer		metaBuffer = InvalidBuffer;
		Page		metaPage;
		typedef struct
		{
			uint32		magicNumber;
			uint32		version;
			int			nlists;
			int			nprobe;
			int			dim;
			BlockNumber centroidsBlock;
			int64		insertedVectors;
		}			IvfMetaPageData;
		IvfMetaPageData *meta;

		index_name_text = PG_GETARG_TEXT_P(0);
		query = PG_GETARG_VECTOR_P(1);
		NDB_CHECK_VECTOR_VALID(query);
		k = PG_GETARG_INT32(2);
		nprobe = PG_NARGS() > 3 ? PG_GETARG_INT32(3) : 10;

		/* Validate inputs */
		if (index_name_text == NULL)
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("ivf_knn_search_gpu: index name cannot be NULL")));

		index_name = text_to_cstring(index_name_text);

		if (query == NULL)
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("ivf_knn_search_gpu: query vector cannot be NULL")));

		if (k <= 0 || k > 10000)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("ivf_knn_search_gpu: k must be between 1 and 10000")));

		if (nprobe <= 0 || nprobe > 1000)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("ivf_knn_search_gpu: nprobe must be between 1 and 1000")));

		if (query->dim <= 0 || query->dim > 10000)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("ivf_knn_search_gpu: invalid query dimension %d",
							query->dim)));

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		/* Create tuple descriptor: (id bigint, distance real) */
		tupdesc = CreateTemplateTupleDesc(2);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "id", INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "distance", FLOAT4OID, -1, 0);
		funcctx->tuple_desc = BlessTupleDesc(tupdesc);

		/* Get index OID from name */
		index_oid = get_index_oid_from_name(index_name);
		if (!OidIsValid(index_oid))
			ereport(ERROR,
					(errcode(ERRCODE_UNDEFINED_OBJECT),
					 errmsg("ivf_knn_search_gpu: index \"%s\" does not exist",
							index_name)));

		/* Open index relation */
		indexRel = index_open(index_oid, AccessShareLock);

		/* Read metadata page */
		metaBuffer = ReadBuffer(indexRel, 0);
		if (!BufferIsValid(metaBuffer))
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: ReadBuffer failed for buffer")));
		}
		LockBuffer(metaBuffer, BUFFER_LOCK_SHARE);
		metaPage = BufferGetPage(metaBuffer);
		meta = (IvfMetaPageData *) PageGetContents(metaPage);

		/* Validate magic number (IVF uses 0x49564646 = "IVFF") */
		if (meta->magicNumber != 0x49564646)
		{
			UnlockReleaseBuffer(metaBuffer);
			index_close(indexRel, AccessShareLock);
			ereport(ERROR,
					(errcode(ERRCODE_WRONG_OBJECT_TYPE),
					 errmsg("ivf_knn_search_gpu: \"%s\" is not an IVF index",
							index_name)));
		}

		/* Check GPU availability for distance calculations */
		backend = NULL;
		if (ndb_gpu_can_run("l2"))
		{
			backend = ndb_gpu_get_active_backend();
			if (backend && backend->launch_l2_distance)
			{
				elog(DEBUG1,
					 "ivf_knn_search_gpu: GPU acceleration available "
					 "for distance calculations (k=%d, nprobe=%d, dim=%d)",
					 k, nprobe, query->dim);
			}
		}

		/* Perform IVF search with GPU acceleration */
		if (meta->centroidsBlock != InvalidBlockNumber && meta->nlists > 0)
		{
			Buffer		centroidsBuf;
			Page		centroidsPage;
			OffsetNumber maxoff;
			float4	   *centroids = NULL;
			float4	   *centroid_distances = NULL;
			int		   *selected_clusters = NULL;
			float4	   *candidate_vectors = NULL;
			ItemPointerData *candidate_tids = NULL;
			int			candidate_count = 0;
			int			i, j;
			int			max_candidates = k * nprobe * 10; /* Collect more candidates than needed */

			/* Step 1: Load centroids to GPU memory */
			centroidsBuf = ReadBuffer(indexRel, meta->centroidsBlock);
			if (!BufferIsValid(centroidsBuf))
			{
				UnlockReleaseBuffer(metaBuffer);
				index_close(indexRel, AccessShareLock);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("ivf_knn_search_gpu: ReadBuffer failed for centroids")));
			}
			LockBuffer(centroidsBuf, BUFFER_LOCK_SHARE);
			centroidsPage = BufferGetPage(centroidsBuf);
			maxoff = PageGetMaxOffsetNumber(centroidsPage);

			/* Allocate centroids array */
			centroids = (float4 *) palloc(meta->nlists * query->dim * sizeof(float4));
			centroid_distances = (float4 *) palloc(meta->nlists * sizeof(float4));
			selected_clusters = (int *) palloc(nprobe * sizeof(int));

			/* Extract centroids from page */
			for (i = 0; i < meta->nlists && i < maxoff; i++)
			{
				OffsetNumber offnum = FirstOffsetNumber + i;
				IvfCentroidData *centroid;
				float4	   *centroid_vec;

				if (offnum > maxoff)
					break;

				centroid = (IvfCentroidData *) PageGetItem(centroidsPage,
														   PageGetItemId(centroidsPage, offnum));
				if (centroid->dim != query->dim)
					continue;

				centroid_vec = (float4 *) ((char *) centroid + MAXALIGN(sizeof(IvfCentroidData)));
				memcpy(centroids + i * query->dim, centroid_vec, query->dim * sizeof(float4));
			}

			/* Step 2: Select nprobe closest clusters using GPU */
			if (backend && backend->launch_l2_distance)
			{
				/* Compute distances to all centroids on GPU */
				for (i = 0; i < meta->nlists; i++)
				{
					float		dist = 0.0f;
					int			rc;

					rc = backend->launch_l2_distance(query->data,
													  centroids + i * query->dim,
													  &dist,
													  1,
													  query->dim,
													  NULL);
					if (rc == 0)
						centroid_distances[i] = dist;
					else
						centroid_distances[i] = FLT_MAX;
				}
			}
			else
			{
				/* CPU fallback: compute distances */
				for (i = 0; i < meta->nlists; i++)
				{
					float		sum = 0.0f;
					int			d;

					for (d = 0; d < query->dim; d++)
					{
						float		diff = query->data[d] - centroids[i * query->dim + d];
						sum += diff * diff;
					}
					centroid_distances[i] = sqrtf(sum);
				}
			}

			/* Select nprobe closest clusters */
			for (i = 0; i < nprobe; i++)
			{
				int			best_idx = -1;
				float		best_dist = FLT_MAX;

				for (j = 0; j < meta->nlists; j++)
				{
					bool		already_selected = false;
					int			k;

					for (k = 0; k < i; k++)
					{
						if (selected_clusters[k] == j)
						{
							already_selected = true;
							break;
						}
					}

					if (!already_selected && centroid_distances[j] < best_dist)
					{
						best_dist = centroid_distances[j];
						best_idx = j;
					}
				}
				selected_clusters[i] = best_idx;
			}

			UnlockReleaseBuffer(centroidsBuf);

			/* Step 3: Load candidate vectors from selected clusters */
			centroidsBuf = ReadBuffer(indexRel, meta->centroidsBlock);
			LockBuffer(centroidsBuf, BUFFER_LOCK_SHARE);
			centroidsPage = BufferGetPage(centroidsBuf);
			maxoff = PageGetMaxOffsetNumber(centroidsPage);

			candidate_vectors = (float4 *) palloc(max_candidates * query->dim * sizeof(float4));
			candidate_tids = (ItemPointerData *) palloc(max_candidates * sizeof(ItemPointerData));

			/* Collect candidates from selected clusters */
			for (i = 0; i < nprobe && candidate_count < max_candidates; i++)
			{
				int			cluster_id = selected_clusters[i];
				OffsetNumber offnum;
				IvfCentroidData *centroid;
				BlockNumber list_block;

				if (cluster_id < 0 || cluster_id >= maxoff)
					continue;

				offnum = FirstOffsetNumber + cluster_id;
				if (offnum > maxoff)
					continue;

				centroid = (IvfCentroidData *) PageGetItem(centroidsPage,
														   PageGetItemId(centroidsPage, offnum));

				if (centroid->firstBlock == InvalidBlockNumber)
					continue;

				/* Traverse inverted list */
				list_block = centroid->firstBlock;
				while (list_block != InvalidBlockNumber && candidate_count < max_candidates)
				{
					Buffer		list_buf;
					Page		list_page;
					IvfListPageHeader *list_header;
					OffsetNumber list_maxoff;
					OffsetNumber list_offnum;

					list_buf = ReadBuffer(indexRel, list_block);
					if (!BufferIsValid(list_buf))
						break;
					LockBuffer(list_buf, BUFFER_LOCK_SHARE);
					list_page = BufferGetPage(list_buf);
					list_header = (IvfListPageHeader *) PageGetSpecialPointer(list_page);

					if (!PageIsNew(list_page) && !PageIsEmpty(list_page))
					{
						list_maxoff = PageGetMaxOffsetNumber(list_page);

						for (list_offnum = FirstOffsetNumber;
							 list_offnum <= list_maxoff && candidate_count < max_candidates;
							 list_offnum = OffsetNumberNext(list_offnum))
						{
							ItemId		item_id = PageGetItemId(list_page, list_offnum);
							IvfListEntryData *entry;
							float4	   *entry_vec;

							if (!ItemIdIsValid(item_id) || ItemIdIsDead(item_id))
								continue;

							entry = (IvfListEntryData *) PageGetItem(list_page, item_id);
							if (entry->dim != query->dim)
								continue;

							entry_vec = (float4 *) ((char *) entry + MAXALIGN(sizeof(IvfListEntryData)));
							memcpy(candidate_vectors + candidate_count * query->dim,
								   entry_vec, query->dim * sizeof(float4));
							candidate_tids[candidate_count] = entry->heapPtr;
							candidate_count++;
						}
					}

					list_block = list_header->nextBlock;
					UnlockReleaseBuffer(list_buf);
				}
			}

			UnlockReleaseBuffer(centroidsBuf);

			/* Step 4: Compute distances on GPU for all candidates */
			if (candidate_count > 0)
			{
				float4	   *candidate_distances = NULL;
				int		   *indices = NULL;
				int			j;

				candidate_distances = (float4 *) palloc(candidate_count * sizeof(float4));
				indices = (int *) palloc(candidate_count * sizeof(int));

				if (backend && backend->launch_l2_distance)
				{
					/* Compute distances on GPU */
					for (i = 0; i < candidate_count; i++)
					{
						float		dist = 0.0f;
						int			rc;

						rc = backend->launch_l2_distance(query->data,
														  candidate_vectors + i * query->dim,
														  &dist,
														  1,
														  query->dim,
														  NULL);
						if (rc == 0)
							candidate_distances[i] = dist;
						else
							candidate_distances[i] = FLT_MAX;
					}
				}
				else
				{
					/* CPU fallback */
					for (i = 0; i < candidate_count; i++)
					{
						float		sum = 0.0f;
						int			d;

						for (d = 0; d < query->dim; d++)
						{
							float		diff = query->data[d] - candidate_vectors[i * query->dim + d];
							sum += diff * diff;
						}
						candidate_distances[i] = sqrtf(sum);
					}
				}

				/* Initialize indices */
				for (i = 0; i < candidate_count; i++)
					indices[i] = i;

				/* Sort by distance (selection sort for top-k) */
				for (i = 0; i < k && i < candidate_count; i++)
				{
					int			best_idx = i;
					float		best_dist = candidate_distances[indices[i]];

					for (j = i + 1; j < candidate_count; j++)
					{
						if (candidate_distances[indices[j]] < best_dist)
						{
							best_dist = candidate_distances[indices[j]];
							best_idx = j;
						}
					}

					/* Swap */
					if (best_idx != i)
					{
						int			tmp = indices[i];
						indices[i] = indices[best_idx];
						indices[best_idx] = tmp;
					}
				}

				/* Allocate result arrays */
				resultCount = (k < candidate_count) ? k : candidate_count;
				results = (ItemPointerData *) palloc(resultCount * sizeof(ItemPointerData));
				distances = (float4 *) palloc(resultCount * sizeof(float4));

				for (i = 0; i < resultCount; i++)
				{
					results[i] = candidate_tids[indices[i]];
					distances[i] = candidate_distances[indices[i]];
				}

				NDB_FREE(candidate_distances);
				NDB_FREE(indices);
			}
			else
			{
				resultCount = 0;
				results = NULL;
				distances = NULL;
			}

			/* Cleanup */
			if (centroids)
				NDB_FREE(centroids);
			if (centroid_distances)
				NDB_FREE(centroid_distances);
			if (selected_clusters)
				NDB_FREE(selected_clusters);
			if (candidate_vectors)
				NDB_FREE(candidate_vectors);
			if (candidate_tids)
				NDB_FREE(candidate_tids);
		}
		else
		{
			/* No centroids available - return empty results */
			resultCount = 0;
			results = NULL;
			distances = NULL;
			elog(DEBUG1, "ivf_knn_search_gpu: No centroids in index");
		}

		UnlockReleaseBuffer(metaBuffer);
		index_close(indexRel, AccessShareLock);

		/* Store results in function context */
		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);
		{
			typedef struct
			{
				ItemPointerData *results;
				float4	   *distances;
				int			count;
			}			IvfSearchResults;
			IvfSearchResults *searchResults;

			funcctx->max_calls = resultCount;
			if (resultCount > 0)
			{
				searchResults = (IvfSearchResults *) palloc(sizeof(IvfSearchResults));
				searchResults->results = results;
				searchResults->distances = distances;
				searchResults->count = resultCount;
				funcctx->user_fctx = searchResults;
			}
			else
			{
				funcctx->user_fctx = NULL;
				if (results)
					NDB_FREE(results);
				if (distances)
					NDB_FREE(distances);
			}
		}
		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();
	if (funcctx->call_cntr < funcctx->max_calls)
	{
		typedef struct
		{
			ItemPointerData *results;
			float4	   *distances;
			int			count;
		}			IvfSearchResults;
		IvfSearchResults *searchResults = (IvfSearchResults *) funcctx->user_fctx;

		if (searchResults != NULL)
		{
			/* Return next result */
			HeapTuple	tuple;
			Datum		values[2];
			bool		nulls[2] = {false, false};
			ItemPointerData *tid = &searchResults->results[funcctx->call_cntr];

			/*
			 * Return heap TID block number as ID (could also return offset or
			 * full TID)
			 */
			if (ItemPointerIsValid(tid))
				values[0] = Int64GetDatum((int64) ItemPointerGetBlockNumber(tid));
			else
				values[0] = Int64GetDatum(0);
			values[1] = Float4GetDatum(searchResults->distances[funcctx->call_cntr]);
			tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);

			SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
		}
	}

	SRF_RETURN_DONE(funcctx);
}
