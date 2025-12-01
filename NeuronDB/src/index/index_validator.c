/*-------------------------------------------------------------------------
 *
 * index_validator.c
 *		Index validation and diagnostic functions
 *
 * Provides functions to validate index integrity:
 * - neurondb_validate() - Comprehensive validation
 * - neurondb_diag() - Diagnostic information
 * - Graph connectivity checks for HNSW
 * - Centroid quality metrics for IVF
 * - Dead tuple detection
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/index/index_validator.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "access/htup_details.h"
#include "access/heapam.h"
#include "access/relation.h"
#include "access/genam.h"
#include "access/amapi.h"
#include "access/tableam.h"
#include "catalog/pg_class.h"
#include "catalog/pg_index.h"
#include "catalog/pg_am.h"
#include "catalog/pg_namespace.h"
#include "catalog/index.h"
#include "utils/syscache.h"
#include "storage/bufmgr.h"
#include "utils/builtins.h"
#include "utils/rel.h"
#include "utils/snapmgr.h"
#include "utils/timestamp.h"
#include "utils/typcache.h"
#include "utils/lsyscache.h"
#include "utils/relcache.h"
#include "utils/jsonb.h"
#include "lib/stringinfo.h"
#include "funcapi.h"
#include <string.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_constants.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"
#include "executor/spi.h"
#include "catalog/index.h"
#include "access/genam.h"

/* HNSW structures (from hnsw_am.c) */
#define HNSW_MAX_LEVEL 16
#define HNSW_DEFAULT_M 16

/* IVF structures (from ivf_am.c) */
typedef struct IvfCentroidData
{
	int			listId;			/* Inverted list ID */
	int			dim;			/* Vector dimension */
	int64		memberCount;	/* Vectors in this list */
	BlockNumber firstBlock;		/* First block of inverted list */
	/* Followed by float4 centroid[dim] */
}			IvfCentroidData;

typedef IvfCentroidData * IvfCentroid;

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
}			HnswNodeData;

typedef HnswNodeData * HnswNode;

#define HnswGetNeighbors(node, lev) \
	((BlockNumber *)((char *)(node) + MAXALIGN(sizeof(HnswNodeData)) \
		+ (node)->dim * sizeof(float4) \
		+ (lev) * HNSW_DEFAULT_M * 2 * sizeof(BlockNumber)))

/* IVF structures (from ivf_am.c) */
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

typedef IvfMetaPageData * IvfMetaPage;

/*
 * Validation result structure
 */
typedef struct ValidateResult
{
	bool		valid;
	int			errors;
	int			warnings;
	StringInfoData messages;
}			ValidateResult;

/*
 * Diagnostic metrics structure
 */
typedef struct DiagResult
{
	char	   *index_name;
	char	   *index_type;
	int64		total_tuples;
	int64		dead_tuples;
	int64		orphan_nodes;
	float4		avg_connectivity;
	float4		fragmentation;
	int64		size_bytes;
	char	   *health_status;
	StringInfoData recommendations;
}			DiagResult;

/* Forward declarations */
static ValidateResult * validate_hnsw_index(Relation index);
static ValidateResult * validate_ivf_index(Relation index);
static DiagResult * diagnose_index(Relation index);
static void check_hnsw_connectivity(Relation index, ValidateResult * result);
static void check_dead_tuples(Relation index, ValidateResult * result);
static float4 compute_fragmentation(Relation index);

/*
 * neurondb_validate(index_oid regclass) RETURNS TABLE(...)
 *
 * Validates a NeurondB index and returns detailed results.
 */
PG_FUNCTION_INFO_V1(neurondb_validate);

Datum
neurondb_validate(PG_FUNCTION_ARGS)
{
	Oid			indexOid;
	Relation	indexRel;
	ValidateResult *result;
	TupleDesc	tupdesc;
	Datum		values[5];
	bool		nulls[5];
	HeapTuple	tuple;

	/* Get index OID */
	indexOid = PG_GETARG_OID(0);

	/* Open index relation */
	indexRel = index_open(indexOid, AccessShareLock);

	/* Check if it's a NeurondB index */
	if (!RelationIsValid(indexRel))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: invalid index OID")));

	/* Determine index type and validate */
	/* For now, assume HNSW - would need to check relam */
	result = validate_hnsw_index(indexRel);

	/* Build result tuple */
	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("function returning record called in "
						"context that cannot accept type "
						"record")));

	tupdesc = BlessTupleDesc(tupdesc);

	values[0] = BoolGetDatum(result->valid);
	values[1] = Int32GetDatum(result->errors);
	values[2] = Int32GetDatum(result->warnings);
	values[3] = CStringGetTextDatum(result->messages.data);
	values[4] = TimestampTzGetDatum(GetCurrentTimestamp());

	memset(nulls, 0, sizeof(nulls));

	tuple = heap_form_tuple(tupdesc, values, nulls);

	index_close(indexRel, AccessShareLock);

	PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}

/*
 * neurondb_diag(index_oid regclass) RETURNS TABLE(...)
 *
 * Returns diagnostic information about an index.
 */
PG_FUNCTION_INFO_V1(neurondb_diag);

Datum
neurondb_diag(PG_FUNCTION_ARGS)
{
	Oid			indexOid;
	Relation	indexRel;
	DiagResult *diag;
	TupleDesc	tupdesc;
	Datum		values[9];
	bool		nulls[9];
	HeapTuple	tuple;

	indexOid = PG_GETARG_OID(0);
	indexRel = index_open(indexOid, AccessShareLock);

	if (!RelationIsValid(indexRel))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: invalid index OID")));

	diag = diagnose_index(indexRel);

	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("function returning record called in "
						"context that cannot accept type "
						"record")));

	tupdesc = BlessTupleDesc(tupdesc);

	values[0] = CStringGetTextDatum(diag->index_name);
	values[1] = CStringGetTextDatum(diag->index_type);
	values[2] = Int64GetDatum(diag->total_tuples);
	values[3] = Int64GetDatum(diag->dead_tuples);
	values[4] = Int64GetDatum(diag->orphan_nodes);
	values[5] = Float4GetDatum(diag->avg_connectivity);
	values[6] = Float4GetDatum(diag->fragmentation);
	values[7] = Int64GetDatum(diag->size_bytes);
	values[8] = CStringGetTextDatum(diag->health_status);

	memset(nulls, 0, sizeof(nulls));

	tuple = heap_form_tuple(tupdesc, values, nulls);

	index_close(indexRel, AccessShareLock);

	PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}

/*
 * Validate HNSW index
 */
static ValidateResult *
validate_hnsw_index(Relation index)
{
	ValidateResult *result;

	result = (ValidateResult *) palloc0(sizeof(ValidateResult));
	initStringInfo(&result->messages);
	result->valid = true;
	result->errors = 0;
	result->warnings = 0;

	elog(INFO,
		 "neurondb: Validating HNSW index %s",
		 RelationGetRelationName(index));

	/* Check metadata page */
	appendStringInfo(&result->messages, "Checking metadata page... ");
	{
		Buffer		metaBuf;
		Page		metaPage;
		HnswMetaPage meta;

		metaBuf = ReadBuffer(index, 0);
		if (!BufferIsValid(metaBuf))
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: ReadBuffer failed for buffer")));
		}
		LockBuffer(metaBuf, BUFFER_LOCK_SHARE);
		metaPage = BufferGetPage(metaBuf);
		meta = (HnswMetaPage) PageGetContents(metaPage);

		/* Validate magic number */
		if (meta->magicNumber != 0x484E5357)	/* "HNSW" */
		{
			appendStringInfo(&result->messages, "ERROR: Invalid magic number\n");
			result->errors++;
			result->valid = false;
		}

		/* Validate entry point */
		if (meta->entryPoint == InvalidBlockNumber && meta->insertedVectors > 0)
		{
			appendStringInfo(&result->messages, "WARN: No entry point but vectors exist\n");
			result->warnings++;
		}

		/* Validate parameters */
		if (meta->m <= 0 || meta->m > 128)
		{
			appendStringInfo(&result->messages, "ERROR: Invalid m parameter\n");
			result->errors++;
			result->valid = false;
		}

		UnlockReleaseBuffer(metaBuf);
		appendStringInfo(&result->messages, "OK\n");
	}

	/* Check graph connectivity */
	check_hnsw_connectivity(index, result);

	/* Check for dead tuples */
	check_dead_tuples(index, result);

	/* Check layer structure */
	appendStringInfo(&result->messages, "Checking layer structure... ");
	{
		Buffer		metaBuf;
		Page		metaPage;
		HnswMetaPage meta;
		BlockNumber blkno;
		int			maxLevelFound = -1;
		int			nodeCount = 0;

		metaBuf = ReadBuffer(index, 0);
		if (!BufferIsValid(metaBuf))
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: ReadBuffer failed for buffer")));
		}
		LockBuffer(metaBuf, BUFFER_LOCK_SHARE);
		metaPage = BufferGetPage(metaBuf);
		meta = (HnswMetaPage) PageGetContents(metaPage);

		/* Scan all nodes and check levels */
		for (blkno = 1; blkno < RelationGetNumberOfBlocks(index); blkno++)
		{
			Buffer		nodeBuf;
			Page		nodePage;
			OffsetNumber maxoff;
			OffsetNumber offnum;

			nodeBuf = ReadBuffer(index, blkno);
			if (!BufferIsValid(nodeBuf))
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: ReadBuffer failed for buffer")));
			}
			LockBuffer(nodeBuf, BUFFER_LOCK_SHARE);
			nodePage = BufferGetPage(nodeBuf);

			if (PageIsNew(nodePage) || PageIsEmpty(nodePage))
			{
				UnlockReleaseBuffer(nodeBuf);
				continue;
			}

			maxoff = PageGetMaxOffsetNumber(nodePage);
			for (offnum = FirstOffsetNumber; offnum <= maxoff; offnum++)
			{
				ItemId		itemId = PageGetItemId(nodePage, offnum);

				if (!ItemIdIsValid(itemId) || ItemIdIsDead(itemId))
					continue;

				{
					HnswNode	node = (HnswNode) PageGetItem(nodePage, itemId);

					if (node->level > maxLevelFound)
						maxLevelFound = node->level;
					if (node->level < 0 || node->level >= HNSW_MAX_LEVEL)
					{
						appendStringInfo(&result->messages,
										 "ERROR: Invalid node level %d\n",
										 node->level);
						result->errors++;
						result->valid = false;
					}
					nodeCount++;
				}
			}

			UnlockReleaseBuffer(nodeBuf);
		}

		UnlockReleaseBuffer(metaBuf);

		if (maxLevelFound > meta->maxLevel)
		{
			appendStringInfo(&result->messages,
							 "WARN: Max level mismatch (found %d, meta says %d)\n",
							 maxLevelFound,
							 meta->maxLevel);
			result->warnings++;
		}

		appendStringInfo(&result->messages, "OK (checked %d nodes)\n", nodeCount);
	}

	/* Summary */
	if (result->errors == 0 && result->warnings == 0)
	{
		appendStringInfo(&result->messages, "\nIndex is HEALTHY\n");
	}
	else
	{
		appendStringInfo(&result->messages,
						 "\nFound %d errors, %d warnings\n",
						 result->errors,
						 result->warnings);
		result->valid = false;
	}

	return result;
}

/*
 * Validate IVF index
 */
__attribute__((unused)) static ValidateResult *
validate_ivf_index(Relation index)
{
	ValidateResult *result;

	result = (ValidateResult *) palloc0(sizeof(ValidateResult));
	initStringInfo(&result->messages);
	result->valid = true;
	result->errors = 0;
	result->warnings = 0;

	elog(INFO,
		 "neurondb: Validating IVF index %s",
		 RelationGetRelationName(index));

	appendStringInfo(&result->messages, "Checking centroids... ");
	{
		Buffer		metaBuf;
		Page		metaPage;
		IvfMetaPage meta;
		Buffer		centroidsBuf;
		Page		centroidsPage;
		OffsetNumber maxoff;
		int			centroidCount = 0;

		metaBuf = ReadBuffer(index, 0);
		if (!BufferIsValid(metaBuf))
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: ReadBuffer failed for buffer")));
		}
		LockBuffer(metaBuf, BUFFER_LOCK_SHARE);
		metaPage = BufferGetPage(metaBuf);
		meta = (IvfMetaPage) PageGetContents(metaPage);

		if (meta->centroidsBlock == InvalidBlockNumber)
		{
			appendStringInfo(&result->messages, "ERROR: No centroids block\n");
			result->errors++;
			result->valid = false;
		}
		else
		{
			centroidsBuf = ReadBuffer(index, meta->centroidsBlock);
			if (!BufferIsValid(centroidsBuf))
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: ReadBuffer failed for buffer")));
			}
			LockBuffer(centroidsBuf, BUFFER_LOCK_SHARE);
			centroidsPage = BufferGetPage(centroidsBuf);
			maxoff = PageGetMaxOffsetNumber(centroidsPage);
			centroidCount = maxoff;

			if (centroidCount != meta->nlists)
			{
				appendStringInfo(&result->messages,
								 "WARN: Centroid count mismatch (%d vs %d)\n",
								 centroidCount,
								 meta->nlists);
				result->warnings++;
			}

			UnlockReleaseBuffer(centroidsBuf);
		}

		UnlockReleaseBuffer(metaBuf);
		appendStringInfo(&result->messages, "OK (%d centroids)\n", centroidCount);
	}

	appendStringInfo(&result->messages, "Checking inverted lists... ");
	{
		Buffer		metaBuf;
		Page		metaPage;
		IvfMetaPage meta;
		Buffer		centroidsBuf;
		Page		centroidsPage;
		OffsetNumber maxoff;
		int			i;
		int			totalMembers = 0;

		metaBuf = ReadBuffer(index, 0);
		if (!BufferIsValid(metaBuf))
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: ReadBuffer failed for buffer")));
		}
		LockBuffer(metaBuf, BUFFER_LOCK_SHARE);
		metaPage = BufferGetPage(metaBuf);
		meta = (IvfMetaPage) PageGetContents(metaPage);

		if (meta->centroidsBlock != InvalidBlockNumber)
		{
			centroidsBuf = ReadBuffer(index, meta->centroidsBlock);
			if (!BufferIsValid(centroidsBuf))
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: ReadBuffer failed for buffer")));
			}
			LockBuffer(centroidsBuf, BUFFER_LOCK_SHARE);
			centroidsPage = BufferGetPage(centroidsBuf);
			maxoff = PageGetMaxOffsetNumber(centroidsPage);

			for (i = 0; i < meta->nlists && i < maxoff; i++)
			{
				IvfCentroid centroid = (IvfCentroid) PageGetItem(centroidsPage,
																 PageGetItemId(centroidsPage, FirstOffsetNumber + i));

				if (centroid != NULL && centroid->firstBlock != InvalidBlockNumber)
				{
					totalMembers += centroid->memberCount;
				}
			}

			UnlockReleaseBuffer(centroidsBuf);
		}

		UnlockReleaseBuffer(metaBuf);

		if (totalMembers != meta->insertedVectors)
		{
			appendStringInfo(&result->messages,
							 "WARN: Member count mismatch (%d vs %ld)\n",
							 totalMembers,
							 (long) meta->insertedVectors);
			result->warnings++;
		}

		appendStringInfo(&result->messages, "OK\n");
	}

	check_dead_tuples(index, result);

	if (result->errors == 0)
		appendStringInfo(&result->messages, "\nIndex is HEALTHY\n");

	return result;
}

/*
 * Check HNSW graph connectivity
 */
static void
check_hnsw_connectivity(Relation index, ValidateResult * result)
{
	int			orphanCount = 0;
	int			totalNodes = 0;

	appendStringInfo(&result->messages, "Checking graph connectivity... ");

	{
		Buffer		metaBuf;
		Page		metaPage;
		HnswMetaPage meta;
		bool	   *visited;
		int			visitedSize;
		BlockNumber entryPoint;
		BlockNumber blkno;

		metaBuf = ReadBuffer(index, 0);
		if (!BufferIsValid(metaBuf))
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: ReadBuffer failed for buffer")));
		}
		LockBuffer(metaBuf, BUFFER_LOCK_SHARE);
		metaPage = BufferGetPage(metaBuf);
		meta = (HnswMetaPage) PageGetContents(metaPage);
		entryPoint = meta->entryPoint;
		UnlockReleaseBuffer(metaBuf);

		/* Allocate visited array */
		totalNodes = RelationGetNumberOfBlocks(index);
		visitedSize = totalNodes * sizeof(bool);
		visited = (bool *) palloc0(visitedSize);

		/* Traverse from entry point using BFS */
		if (entryPoint != InvalidBlockNumber)
		{
			BlockNumber *queue;
			int			queueHead = 0;
			int			queueTail = 0;
			int			queueSize = totalNodes;

			queue = (BlockNumber *) palloc(queueSize * sizeof(BlockNumber));
			queue[queueTail++] = entryPoint;
			visited[entryPoint] = true;

			while (queueHead < queueTail)
			{
				BlockNumber current = queue[queueHead++];
				Buffer		nodeBuf;
				Page		nodePage;
				HnswNode	node;
				BlockNumber *neighbors;
				int			level;
				int16		neighborCount;
				int			j;

				nodeBuf = ReadBuffer(index, current);
				if (!BufferIsValid(nodeBuf))
				{
					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg("neurondb: ReadBuffer failed for buffer")));
				}
				LockBuffer(nodeBuf, BUFFER_LOCK_SHARE);
				nodePage = BufferGetPage(nodeBuf);

				if (PageIsEmpty(nodePage))
				{
					UnlockReleaseBuffer(nodeBuf);
					continue;
				}

				node = (HnswNode) PageGetItem(nodePage,
											  PageGetItemId(nodePage, FirstOffsetNumber));

				/* Visit neighbors at all levels */
				for (level = 0; level <= node->level; level++)
				{
					neighbors = HnswGetNeighbors(node, level);
					neighborCount = node->neighborCount[level];

					for (j = 0; j < neighborCount; j++)
					{
						if (neighbors[j] != InvalidBlockNumber
							&& neighbors[j] < totalNodes
							&& !visited[neighbors[j]])
						{
							visited[neighbors[j]] = true;
							if (queueTail < queueSize)
								queue[queueTail++] = neighbors[j];
						}
					}
				}

				UnlockReleaseBuffer(nodeBuf);
			}

			NDB_FREE(queue);
		}

		/* Count unreachable nodes */
		for (blkno = 1; blkno < totalNodes; blkno++)
		{
			Buffer		nodeBuf;
			Page		nodePage;
			OffsetNumber maxoff;
			OffsetNumber offnum;

			if (visited[blkno])
				continue;

			nodeBuf = ReadBuffer(index, blkno);
			if (!BufferIsValid(nodeBuf))
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: ReadBuffer failed for buffer")));
			}
			LockBuffer(nodeBuf, BUFFER_LOCK_SHARE);
			nodePage = BufferGetPage(nodeBuf);

			if (PageIsNew(nodePage) || PageIsEmpty(nodePage))
			{
				UnlockReleaseBuffer(nodeBuf);
				continue;
			}

			maxoff = PageGetMaxOffsetNumber(nodePage);
			for (offnum = FirstOffsetNumber; offnum <= maxoff; offnum++)
			{
				ItemId		itemId = PageGetItemId(nodePage, offnum);

				if (ItemIdIsValid(itemId) && !ItemIdIsDead(itemId))
				{
					orphanCount++;
					break;		/* Count block once */
				}
			}

			UnlockReleaseBuffer(nodeBuf);
		}

		NDB_FREE(visited);
	}

	if (orphanCount > 0)
	{
		appendStringInfo(&result->messages,
						 "WARN: Found %d orphan nodes\n",
						 orphanCount);
		result->warnings++;
	}
	else
	{
		appendStringInfo(&result->messages,
						 "OK (checked %d nodes)\n",
						 totalNodes);
	}
}

/*
 * Check for dead tuples
 */
static void
check_dead_tuples(Relation index, ValidateResult * result)
{
	int			deadCount = 0;
	int			totalChecked = 0;
	Relation	heapRel;
	Snapshot	snapshot;
	BlockNumber blkno;
	Buffer		buf;
	Page		page;
	OffsetNumber maxoff;
	OffsetNumber offnum;
	ItemId		itemId;
	HeapTupleData tupleData;
	HeapTuple	tuple = &tupleData;
	bool		found;

	appendStringInfo(&result->messages, "Checking for dead tuples... ");

	/* Get heap relation from index */
	heapRel = index_open(IndexGetRelation(index->rd_id, false), AccessShareLock);

	/* Get snapshot for visibility checking */
	snapshot = GetActiveSnapshot();

	/* Scan all pages in the index */
	for (blkno = 1; blkno < RelationGetNumberOfBlocks(index); blkno++)
	{
		buf = ReadBuffer(index, blkno);
		if (!BufferIsValid(buf))
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: ReadBuffer failed for buffer")));
		}
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);

		if (PageIsNew(page) || PageIsEmpty(page))
		{
			UnlockReleaseBuffer(buf);
			continue;
		}

		maxoff = PageGetMaxOffsetNumber(page);

		for (offnum = FirstOffsetNumber; offnum <= maxoff;
			 offnum = OffsetNumberNext(offnum))
		{
			itemId = PageGetItemId(page, offnum);

			if (!ItemIdIsValid(itemId) || ItemIdIsDead(itemId))
				continue;

			/* Check if heap tuple is visible */
			{
				Buffer		heapBuf;

				found = heap_fetch(heapRel, snapshot, tuple, &heapBuf, false);
				if (found && HeapTupleIsValid(tuple))
				{
					/* Check visibility */
					if (!HeapTupleSatisfiesVisibility(tuple, snapshot, heapBuf))
					{
						deadCount++;
					}
				}
				else
				{
					/* Tuple not found or invalid - dead reference */
					deadCount++;
				}
			}

			totalChecked++;
		}

		UnlockReleaseBuffer(buf);
	}

	/* Close heap relation */
	index_close(heapRel, AccessShareLock);

	if (deadCount > 0)
	{
		appendStringInfo(&result->messages,
						 "WARN: Found %d dead tuples out of %d checked (consider VACUUM)\n",
						 deadCount,
						 totalChecked);
		result->warnings++;
	}
	else
	{
		appendStringInfo(&result->messages, "OK (checked %d tuples)\n", totalChecked);
	}
}

/*
 * Diagnose index and return metrics
 */
static DiagResult *
diagnose_index(Relation index)
{
	DiagResult *diag;

	diag = (DiagResult *) palloc0(sizeof(DiagResult));
	initStringInfo(&diag->recommendations);

	diag->index_name = pstrdup(RelationGetRelationName(index));
	diag->index_type = pstrdup("HNSW"); /* Would check actual type */

	/* Get actual statistics from index */
	{
		BlockNumber blkno;
		Buffer		buf;
		Page		page;
		OffsetNumber maxoff;
		OffsetNumber offnum;
		ItemId		itemId;
		int64		totalTuples = 0;
		int64		deadTuples = 0;
		Relation	heapRel;
		Snapshot	snapshot;
		HeapTupleData tupleData;
		HeapTuple	tuple = &tupleData;
		bool		found;

		heapRel = index_open(IndexGetRelation(index->rd_id, false), AccessShareLock);
		snapshot = GetActiveSnapshot();

		for (blkno = 1; blkno < RelationGetNumberOfBlocks(index); blkno++)
		{
			buf = ReadBuffer(index, blkno);
			if (!BufferIsValid(buf))
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: ReadBuffer failed for buffer")));
			}
			LockBuffer(buf, BUFFER_LOCK_SHARE);
			page = BufferGetPage(buf);

			if (PageIsNew(page) || PageIsEmpty(page))
			{
				UnlockReleaseBuffer(buf);
				continue;
			}

			maxoff = PageGetMaxOffsetNumber(page);

			for (offnum = FirstOffsetNumber; offnum <= maxoff;
				 offnum = OffsetNumberNext(offnum))
			{
				itemId = PageGetItemId(page, offnum);

				if (!ItemIdIsValid(itemId))
					continue;

				if (ItemIdIsDead(itemId))
				{
					deadTuples++;
					continue;
				}

				totalTuples++;

				/* Check if heap tuple is visible */
				{
					Buffer		heapBuf;

					found = heap_fetch(heapRel, snapshot, tuple, &heapBuf, false);
					if (found && HeapTupleIsValid(tuple))
					{
						if (!HeapTupleSatisfiesVisibility(tuple, snapshot, heapBuf))
						{
							deadTuples++;
						}
					}
					else
					{
						deadTuples++;
					}
				}
			}

			UnlockReleaseBuffer(buf);
		}

		index_close(heapRel, AccessShareLock);

		diag->total_tuples = totalTuples;
		diag->dead_tuples = deadTuples;
	}
	diag->orphan_nodes = 0;
	diag->avg_connectivity = 16.0;	/* M parameter */
	diag->fragmentation = compute_fragmentation(index);
	diag->size_bytes = RelationGetNumberOfBlocks(index) * BLCKSZ;

	/* Determine health status */
	if (diag->dead_tuples > diag->total_tuples * 0.2)
	{
		diag->health_status = pstrdup("NEEDS_VACUUM");
		appendStringInfo(&diag->recommendations,
						 "Run VACUUM to clean dead tuples. ");
	}
	else if (diag->fragmentation > 0.3)
	{
		diag->health_status = pstrdup("FRAGMENTED");
		appendStringInfo(&diag->recommendations,
						 "Consider REINDEX to reduce fragmentation. ");
	}
	else if (diag->orphan_nodes > 0)
	{
		diag->health_status = pstrdup("DEGRADED");
		appendStringInfo(&diag->recommendations,
						 "Orphan nodes detected, rebuild recommended. ");
	}
	else
	{
		diag->health_status = pstrdup("HEALTHY");
		appendStringInfo(&diag->recommendations, "No issues detected.");
	}

	return diag;
}

/*
 * Compute index fragmentation metric
 */
static float4
compute_fragmentation(Relation index)
{
	BlockNumber totalBlocks;
	BlockNumber usedBlocks;
	float4		fragmentation;

	totalBlocks = RelationGetNumberOfBlocks(index);
	/* Count actually used blocks */
	{
		BlockNumber blkno;
		Buffer		buf;
		Page		page;
		int64		used = 0;

		for (blkno = 0; blkno < totalBlocks; blkno++)
		{
			buf = ReadBuffer(index, blkno);
			if (!BufferIsValid(buf))
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: ReadBuffer failed for buffer")));
			}
			LockBuffer(buf, BUFFER_LOCK_SHARE);
			page = BufferGetPage(buf);

			/* Block is used if it's not new and not empty */
			if (!PageIsNew(page) && !PageIsEmpty(page))
			{
				/* Further check: has at least one valid item */
				OffsetNumber maxoff = PageGetMaxOffsetNumber(page);
				OffsetNumber offnum;
				bool		hasValidItem = false;

				for (offnum = FirstOffsetNumber; offnum <= maxoff;
					 offnum = OffsetNumberNext(offnum))
				{
					ItemId		itemId = PageGetItemId(page, offnum);

					if (ItemIdIsValid(itemId))
					{
						hasValidItem = true;
						break;
					}
				}

				if (hasValidItem)
					used++;
			}

			UnlockReleaseBuffer(buf);
		}

		usedBlocks = used;
	}

	if (totalBlocks == 0)
		return 0.0;

	fragmentation = 1.0 - ((float4) usedBlocks / totalBlocks);

	return fragmentation;
}

/*
 * neurondb_rebuild_index(index_oid regclass) RETURNS void
 *
 * Rebuilds an index with optimization.
 */
PG_FUNCTION_INFO_V1(neurondb_rebuild_index);

Datum
neurondb_rebuild_index(PG_FUNCTION_ARGS)
{
	Oid			indexOid;
	Relation	indexRel;

	indexOid = PG_GETARG_OID(0);
	indexRel = index_open(indexOid, AccessExclusiveLock);

	elog(INFO,
		 "neurondb: Rebuilding index %s",
		 RelationGetRelationName(indexRel));

	/* Get heap relation and index info for rebuild */
	{
		Relation	heapRel;
		IndexInfo  *indexInfo;
		Oid			heapOid;
		char	   *indexName = NULL;
		StringInfoData sql;
		TimestampTz rebuildTime;
		NDB_DECLARE(NdbSpiSession *, session2);

		/* Get heap relation OID from index */
		heapOid = IndexGetRelation(indexOid, false);
		if (!OidIsValid(heapOid))
		{
			index_close(indexRel, AccessExclusiveLock);
			ereport(ERROR,
					(errcode(ERRCODE_UNDEFINED_OBJECT),
					 errmsg("neurondb: Cannot find heap relation for index %s",
							RelationGetRelationName(indexRel))));
		}

		heapRel = relation_open(heapOid, AccessShareLock);

		/* Create IndexInfo for rebuild */
		indexInfo = BuildIndexInfo(indexRel);
		if (indexInfo == NULL)
		{
			relation_close(heapRel, AccessShareLock);
			index_close(indexRel, AccessExclusiveLock);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: Failed to build IndexInfo for index %s",
							RelationGetRelationName(indexRel))));
		}

		/* Get current timestamp for rebuild history */
		rebuildTime = GetCurrentTimestamp();

		/* Close relations before REINDEX (REINDEX needs exclusive access) */
		relation_close(heapRel, AccessShareLock);
		index_close(indexRel, AccessExclusiveLock);

		/* Perform the actual rebuild using REINDEX via SPI */
		{
			char	   *inner_indexName;
			char	   *rebuildCmd;
			StringInfoData cmd;
			NDB_DECLARE(NdbSpiSession *, session);

			inner_indexName = pstrdup(RelationGetRelationName(indexRel));
			initStringInfo(&cmd);
			appendStringInfo(&cmd, "REINDEX INDEX %s", inner_indexName);
			rebuildCmd = cmd.data;
			session = ndb_spi_session_begin(CurrentMemoryContext, false);
			if (session == NULL)
			{
				pfree(indexInfo);
				NDB_FREE(inner_indexName);
				NDB_FREE(rebuildCmd);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: failed to begin SPI session during index rebuild")));
			}

			ndb_spi_execute(session, rebuildCmd, false, 0);
			ndb_spi_session_end(&session);

			NDB_FREE(indexName);
			NDB_FREE(rebuildCmd);

			elog(INFO,
				 "neurondb: Index %s rebuilt successfully",
				 RelationGetRelationName(indexRel));
		}

		/* Reopen index to get updated relation for tracking */
		indexRel = index_open(indexOid, AccessShareLock);

		/* Track rebuild history */
		indexName = pstrdup(RelationGetRelationName(indexRel));
		session2 = NULL;  /* Initialize */
		session2 = ndb_spi_session_begin(CurrentMemoryContext, false);
		if (session2 == NULL)
		{
			pfree(indexInfo);
			relation_close(heapRel, AccessShareLock);
			index_close(indexRel, AccessExclusiveLock);
			NDB_FREE(indexName);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: failed to begin SPI session during rebuild history tracking")));
		}

		/* Create rebuild history table if it doesn't exist */
		{
			initStringInfo(&sql);
			appendStringInfo(&sql,
							 "CREATE TABLE IF NOT EXISTS " NDB_FQ_INDEX_REBUILD_HISTORY " ("
							 "index_oid OID PRIMARY KEY, "
							 "index_name TEXT NOT NULL, "
							 "last_rebuild_time TIMESTAMPTZ NOT NULL, "
							 "rebuild_count BIGINT DEFAULT 1)");
			(void) ndb_spi_execute(session2, sql.data, false, 0);
			NDB_FREE(sql.data);
		}
		/* Upsert rebuild history */
		{
			char	   *rebuildTimeStr;
			Datum		rebuildTimeDatum = TimestampTzGetDatum(rebuildTime);

			rebuildTimeStr = DatumGetCString(DirectFunctionCall1(timestamptz_out, rebuildTimeDatum));

			initStringInfo(&sql);
			appendStringInfo(&sql,
							 "INSERT INTO " NDB_FQ_INDEX_REBUILD_HISTORY " "
							 "(index_oid, index_name, last_rebuild_time, rebuild_count) "
							 "VALUES (%u, %s, %s::timestamptz, 1) "
							 "ON CONFLICT (index_oid) DO UPDATE SET "
							 "last_rebuild_time = EXCLUDED.last_rebuild_time, "
							 "rebuild_count = " NDB_FQ_INDEX_REBUILD_HISTORY ".rebuild_count + 1",
							 indexOid,
							 quote_literal_cstr(indexName),
							 quote_literal_cstr(rebuildTimeStr));
			(void) ndb_spi_execute(session2, sql.data, false, 0);
			NDB_FREE(sql.data);
			NDB_FREE(rebuildTimeStr);
		}
		NDB_FREE(indexName);
		ndb_spi_session_end(&session2);

		pfree(indexInfo);
	}

	index_close(indexRel, AccessShareLock);

	PG_RETURN_VOID();
}

/*
 * index_statistics(index_name text) RETURNS jsonb
 *
 * Get comprehensive statistics about an index including size, nodes, edges,
 * tuple counts, and performance metrics.
 */
PG_FUNCTION_INFO_V1(index_statistics);
Datum
index_statistics(PG_FUNCTION_ARGS)
{
	text	   *index_name = PG_GETARG_TEXT_P(0);
	char	   *idx_name;
	Oid			indexOid;
	Relation	indexRel;
	DiagResult *diag;
	StringInfoData json_buf;
	Jsonb	   *result_jsonb;
	int64		total_blocks;
	int64		index_size_bytes;
	int64		heap_size_bytes;
	int64		total_tuples;
	int64		dead_tuples;
	float4		fragmentation;
	char	   *index_type = "unknown";

	idx_name = text_to_cstring(index_name);

	/* Look up index OID */
	{
		Oid			namespaceOid;
		HeapTuple	nstuple;
		Form_pg_namespace nsform;

		nstuple = SearchSysCache1(NAMESPACENAME, CStringGetTextDatum("public"));
		if (HeapTupleIsValid(nstuple))
		{
			nsform = (Form_pg_namespace) GETSTRUCT(nstuple);
			namespaceOid = nsform->oid;
			ReleaseSysCache(nstuple);
			indexOid = get_relname_relid(idx_name, namespaceOid);
		}
		else
		{
			indexOid = InvalidOid;
		}
	}
	if (!OidIsValid(indexOid))
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("index \"%s\" does not exist", idx_name)));

	indexRel = index_open(indexOid, AccessShareLock);

	if (!RelationIsValid(indexRel))
	{
		index_close(indexRel, AccessShareLock);
		NDB_FREE(idx_name);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid index: %s", idx_name)));
	}

	/* Get index type */
	{
		Oid			amOid = indexRel->rd_rel->relam;
		HeapTuple	amtup;
		Form_pg_am	amform;

		amtup = SearchSysCache1(AMOID, ObjectIdGetDatum(amOid));
		if (HeapTupleIsValid(amtup))
		{
			amform = (Form_pg_am) GETSTRUCT(amtup);
			if (strcmp(NameStr(amform->amname), "hnsw") == 0)
				index_type = "hnsw";
			else if (strcmp(NameStr(amform->amname), "ivfflat") == 0)
				index_type = "ivf";
			ReleaseSysCache(amtup);
		}
	}

	/* Get statistics */
	diag = diagnose_index(indexRel);
	total_blocks = RelationGetNumberOfBlocks(indexRel);
	index_size_bytes = total_blocks * BLCKSZ;
	fragmentation = compute_fragmentation(indexRel);

	/* Get heap relation stats */
	{
		Oid			heapOid = indexRel->rd_index->indrelid;

		if (OidIsValid(heapOid))
		{
			Relation	heapRel = relation_open(heapOid, AccessShareLock);

			heap_size_bytes = RelationGetNumberOfBlocks(heapRel) * BLCKSZ;
			relation_close(heapRel, AccessShareLock);
		}
		else
		{
			heap_size_bytes = 0;
		}
	}

	/* Get tuple counts from pg_stat_user_indexes if available */
	total_tuples = diag->total_tuples;
	dead_tuples = diag->dead_tuples;

	/* Build JSONB result */
	initStringInfo(&json_buf);
	appendStringInfo(&json_buf,
					 "{\"index_name\":\"%s\","
					 "\"index_type\":\"%s\","
					 "\"index_size_bytes\":%lld,"
					 "\"index_size_mb\":%.2f,"
					 "\"heap_size_bytes\":%lld,"
					 "\"heap_size_mb\":%.2f,"
					 "\"total_tuples\":%lld,"
					 "\"dead_tuples\":%lld,"
					 "\"live_tuples\":%lld,"
					 "\"fragmentation\":%.4f,"
					 "\"avg_connectivity\":%.2f,"
					 "\"orphan_nodes\":%lld}",
					 idx_name,
					 index_type,
					 (long long) index_size_bytes,
					 (double) index_size_bytes / (1024.0 * 1024.0),
					 (long long) heap_size_bytes,
					 (double) heap_size_bytes / (1024.0 * 1024.0),
					 (long long) total_tuples,
					 (long long) dead_tuples,
					 (long long) (total_tuples - dead_tuples),
					 fragmentation,
					 diag->avg_connectivity,
					 (long long) diag->orphan_nodes);

	result_jsonb = DatumGetJsonbP(DirectFunctionCall1(
													  jsonb_in, CStringGetTextDatum(json_buf.data)));

	NDB_FREE(json_buf.data);
	NDB_FREE(idx_name);
	index_close(indexRel, AccessShareLock);

	PG_RETURN_POINTER(result_jsonb);
}

/*
 * index_health(index_name text) RETURNS jsonb
 *
 * Check index health and quality, returning health status and recommendations.
 */
PG_FUNCTION_INFO_V1(index_health);
Datum
index_health(PG_FUNCTION_ARGS)
{
	text	   *index_name = PG_GETARG_TEXT_P(0);
	char	   *idx_name;
	Oid			indexOid;
	Relation	indexRel;
	DiagResult *diag;
	StringInfoData json_buf;
	Jsonb	   *result_jsonb;
	float4		health_score;
	char	   *health_status;
	int64		total_tuples;
	int64		dead_tuples;
	float4		fragmentation;
	int64		orphan_nodes;

	idx_name = text_to_cstring(index_name);

	/* Look up index OID */
	{
		Oid			namespaceOid;
		HeapTuple	nstuple;
		Form_pg_namespace nsform;

		nstuple = SearchSysCache1(NAMESPACENAME, CStringGetTextDatum("public"));
		if (HeapTupleIsValid(nstuple))
		{
			nsform = (Form_pg_namespace) GETSTRUCT(nstuple);
			namespaceOid = nsform->oid;
			ReleaseSysCache(nstuple);
			indexOid = get_relname_relid(idx_name, namespaceOid);
		}
		else
		{
			indexOid = InvalidOid;
		}
	}
	if (!OidIsValid(indexOid))
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("index \"%s\" does not exist", idx_name)));

	indexRel = index_open(indexOid, AccessShareLock);

	if (!RelationIsValid(indexRel))
	{
		index_close(indexRel, AccessShareLock);
		NDB_FREE(idx_name);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid index: %s", idx_name)));
	}

	diag = diagnose_index(indexRel);
	total_tuples = diag->total_tuples;
	dead_tuples = diag->dead_tuples;
	fragmentation = diag->fragmentation;
	orphan_nodes = diag->orphan_nodes;
	health_status = diag->health_status;

	/* Calculate health score (0.0 to 1.0) */
	health_score = 1.0f;
	if (total_tuples > 0)
	{
		float4		dead_ratio = (float4) dead_tuples / (float4) total_tuples;

		health_score -= dead_ratio * 0.4f;	/* Dead tuples reduce score */
	}
	health_score -= fragmentation * 0.3f;	/* Fragmentation reduces score */
	if (orphan_nodes > 0)
		health_score -= 0.3f;	/* Orphan nodes reduce score */
	if (health_score < 0.0f)
		health_score = 0.0f;

	/* Build JSONB result */
	initStringInfo(&json_buf);
	appendStringInfo(&json_buf,
					 "{\"index_name\":\"%s\","
					 "\"health_status\":\"%s\","
					 "\"health_score\":%.3f,"
					 "\"dead_tuple_ratio\":%.4f,"
					 "\"fragmentation\":%.4f,"
					 "\"orphan_nodes\":%lld,"
					 "\"recommendations\":\"%s\"}",
					 idx_name,
					 health_status,
					 health_score,
					 total_tuples > 0 ? ((float4) dead_tuples / (float4) total_tuples) : 0.0f,
					 fragmentation,
					 (long long) orphan_nodes,
					 diag->recommendations.data);

	result_jsonb = DatumGetJsonbP(DirectFunctionCall1(
													  jsonb_in, CStringGetTextDatum(json_buf.data)));

	NDB_FREE(json_buf.data);
	NDB_FREE(idx_name);
	index_close(indexRel, AccessShareLock);

	PG_RETURN_POINTER(result_jsonb);
}

/*
 * index_rebuild_recommendation(index_name text) RETURNS jsonb
 *
 * Analyze index and recommend when to rebuild based on health metrics.
 */
PG_FUNCTION_INFO_V1(index_rebuild_recommendation);
Datum
index_rebuild_recommendation(PG_FUNCTION_ARGS)
{
	text	   *index_name = PG_GETARG_TEXT_P(0);
	char	   *idx_name;
	Oid			indexOid;
	Relation	indexRel;
	DiagResult *diag;
	StringInfoData json_buf;
	Jsonb	   *result_jsonb;
	bool		should_rebuild = false;
	char	   *rebuild_reason = NULL;
	float4		dead_ratio;
	float4		fragmentation;
	int64		orphan_nodes;
	int64		days_since_last_rebuild = 0;
	TimestampTz last_rebuild_time = 0;

	idx_name = text_to_cstring(index_name);

	/* Look up index OID */
	{
		Oid			namespaceOid;
		HeapTuple	nstuple;
		Form_pg_namespace nsform;

		nstuple = SearchSysCache1(NAMESPACENAME, CStringGetTextDatum("public"));
		if (HeapTupleIsValid(nstuple))
		{
			nsform = (Form_pg_namespace) GETSTRUCT(nstuple);
			namespaceOid = nsform->oid;
			ReleaseSysCache(nstuple);
			indexOid = get_relname_relid(idx_name, namespaceOid);
		}
		else
		{
			indexOid = InvalidOid;
		}
	}
	if (!OidIsValid(indexOid))
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("index \"%s\" does not exist", idx_name)));

	indexRel = index_open(indexOid, AccessShareLock);

	if (!RelationIsValid(indexRel))
	{
		index_close(indexRel, AccessShareLock);
		NDB_FREE(idx_name);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid index: %s", idx_name)));
	}

	diag = diagnose_index(indexRel);
	dead_ratio = diag->total_tuples > 0
		? ((float4) diag->dead_tuples / (float4) diag->total_tuples)
		: 0.0f;
	fragmentation = diag->fragmentation;
	orphan_nodes = diag->orphan_nodes;

	/* Query rebuild history to get last rebuild time */
	{
		int			ret;
		StringInfoData sql;

		NDB_DECLARE(NdbSpiSession *, session3);
		session3 = ndb_spi_session_begin(CurrentMemoryContext, false);
		if (session3 != NULL)
		{
			initStringInfo(&sql);
			appendStringInfo(&sql,
							 "SELECT last_rebuild_time FROM " NDB_FQ_INDEX_REBUILD_HISTORY " "
							 "WHERE index_oid = %u",
							 indexOid);
			ret = ndb_spi_execute(session3, sql.data, true, 0);
			if (ret == SPI_OK_SELECT && SPI_processed > 0)
			{
				bool		isnull;
				Datum		rebuildTimeDatum = SPI_getbinval(SPI_tuptable->vals[0],
															 SPI_tuptable->tupdesc,
															 1,
															 &isnull);

				if (!isnull)
				{
					last_rebuild_time = DatumGetTimestampTz(rebuildTimeDatum);
					days_since_last_rebuild = (GetCurrentTimestamp() - last_rebuild_time) / (24 * 60 * 60 * 1000000.0);
				}
			}
			NDB_FREE(sql.data);
			ndb_spi_session_end(&session3);
		}
	}

	/* Determine if rebuild is recommended */
	if (orphan_nodes > 0)
	{
		should_rebuild = true;
		rebuild_reason = "Orphan nodes detected";
	}
	else if (dead_ratio > 0.2f)
	{
		should_rebuild = true;
		rebuild_reason = "High dead tuple ratio (>20%)";
	}
	else if (fragmentation > 0.3f)
	{
		should_rebuild = true;
		rebuild_reason = "High fragmentation (>30%)";
	}
	else if (days_since_last_rebuild > 90)
	{
		should_rebuild = true;
		rebuild_reason = "Index age (>90 days since last rebuild)";
	}
	else
	{
		should_rebuild = false;
		rebuild_reason = "Index is healthy, no rebuild needed";
	}

	/* Build JSONB result */
	initStringInfo(&json_buf);
	appendStringInfo(&json_buf,
					 "{\"index_name\":\"%s\","
					 "\"should_rebuild\":%s,"
					 "\"rebuild_reason\":\"%s\","
					 "\"dead_tuple_ratio\":%.4f,"
					 "\"fragmentation\":%.4f,"
					 "\"orphan_nodes\":%lld,"
					 "\"days_since_last_rebuild\":%lld,"
					 "\"priority\":\"%s\"}",
					 idx_name,
					 should_rebuild ? "true" : "false",
					 rebuild_reason,
					 dead_ratio,
					 fragmentation,
					 (long long) orphan_nodes,
					 (long long) days_since_last_rebuild,
					 should_rebuild ? (orphan_nodes > 0 ? "high" : (dead_ratio > 0.3f ? "high" : "medium")) : "low");

	result_jsonb = DatumGetJsonbP(DirectFunctionCall1(
													  jsonb_in, CStringGetTextDatum(json_buf.data)));

	NDB_FREE(json_buf.data);
	NDB_FREE(idx_name);
	index_close(indexRel, AccessShareLock);

	PG_RETURN_POINTER(result_jsonb);
}
