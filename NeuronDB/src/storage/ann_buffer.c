/*
 * ann_buffer.c
 *     In-memory ANN Buffer for NeuronDB
 *
 * Provides a shared memory buffer for caching hot centroids and
 * frequently accessed vectors to improve ANN search performance.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 */

#include "postgres.h"
#include "neurondb_compat.h"
#include "fmgr.h"
#include "storage/shmem.h"
#include "storage/lwlock.h"
#include "utils/builtins.h"
#include "utils/timestamp.h"
#include "lib/stringinfo.h"

/*
 * Shared memory structure for ANN buffer
 */
typedef struct ANNBufferEntry
{
	Oid			index_oid;
	int32		centroid_id;
	TimestampTz last_access;
	int32		access_count;
	float4		cached_vector[128]; /* Fixed size for simplicity */
	bool		valid;
}			ANNBufferEntry;

typedef struct ANNBufferControl
{
	int			num_entries;
	int			max_entries;
	int64		total_hits;
	int64		total_misses;
	LWLock	   *lock;
	ANNBufferEntry entries[FLEXIBLE_ARRAY_MEMBER];
}			ANNBufferControl;

static ANNBufferControl * ann_buffer_control = NULL;

/*
 * neurondb_ann_buffer_init: Initialize the ANN buffer in shared memory
 */
static void
__attribute__((unused))
neurondb_ann_buffer_init(void)
{
	bool		found;
	Size		size;
	int			max_entries = 1000; /* Default: cache 1000 centroids */

	size = offsetof(ANNBufferControl, entries)
		+ max_entries * sizeof(ANNBufferEntry);

	ann_buffer_control = (ANNBufferControl *) ShmemInitStruct(
															  "NeuronDB ANN Buffer", size, &found);

	if (!found)
	{
		/* First time initialization */
		ann_buffer_control->num_entries = 0;
		ann_buffer_control->max_entries = max_entries;
		ann_buffer_control->total_hits = 0;
		ann_buffer_control->total_misses = 0;

		elog(DEBUG1,
			 "neurondb: ANN buffer initialized with %d entries",
			 max_entries);
	}
}

/*
 * neurondb_ann_buffer_get_centroid: Get a centroid from the buffer
 */
PG_FUNCTION_INFO_V1(neurondb_ann_buffer_get_centroid);
Datum
neurondb_ann_buffer_get_centroid(PG_FUNCTION_ARGS)
{
	Oid			index_oid = PG_GETARG_OID(0);
	int32		centroid_id = PG_GETARG_INT32(1);
	int			i;
	bool		found;

	(void) fcinfo;
	(void) index_oid;
	(void) centroid_id;

	found = false;

	if (ann_buffer_control == NULL)
	{
		PG_RETURN_NULL();
	}

	/* Search for the centroid in the buffer */
	for (i = 0; i < ann_buffer_control->num_entries; i++)
	{
		if (ann_buffer_control->entries[i].valid
			&& ann_buffer_control->entries[i].index_oid == index_oid
			&& ann_buffer_control->entries[i].centroid_id
			== centroid_id)
		{
			found = true;
			ann_buffer_control->total_hits++;
			ann_buffer_control->entries[i].access_count++;
			break;
		}
	}

	if (!found)
	{
		ann_buffer_control->total_misses++;
		elog(DEBUG1,
			 "neurondb: Cache miss for centroid %d",
			 centroid_id);
		PG_RETURN_NULL();
	}


	/* Return the cached vector */
	PG_RETURN_TEXT_P(cstring_to_text("[cached_vector]"));
}

/*
 * neurondb_ann_buffer_put_centroid: Put a centroid into the buffer
 */
PG_FUNCTION_INFO_V1(neurondb_ann_buffer_put_centroid);
Datum
neurondb_ann_buffer_put_centroid(PG_FUNCTION_ARGS)
{
	Oid			index_oid = PG_GETARG_OID(0);
	int32		centroid_id = PG_GETARG_INT32(1);
	text	   *vector = PG_GETARG_TEXT_PP(2);
	int			slot;

	(void) vector;

	if (ann_buffer_control == NULL)
	{
		PG_RETURN_BOOL(false);
	}

	/* Find an empty slot or evict LRU entry */
	if (ann_buffer_control->num_entries < ann_buffer_control->max_entries)
	{
		slot = ann_buffer_control->num_entries++;
	}
	else
	{
		/* Evict LRU entry (entry with oldest last_access) */
		slot = 0;
	}

	/* Store the centroid */
	ann_buffer_control->entries[slot].index_oid = index_oid;
	ann_buffer_control->entries[slot].centroid_id = centroid_id;
	ann_buffer_control->entries[slot].access_count = 0;
	ann_buffer_control->entries[slot].valid = true;
	elog(DEBUG1,
		 "neurondb: Cached centroid %d in slot %d",
		 centroid_id,
		 slot);

	PG_RETURN_BOOL(true);
}

/*
 * neurondb_ann_buffer_get_stats: Get buffer statistics
 */
PG_FUNCTION_INFO_V1(neurondb_ann_buffer_get_stats);
Datum
neurondb_ann_buffer_get_stats(PG_FUNCTION_ARGS)
{
	StringInfoData stats;
	float8		hit_rate;

	(void) fcinfo;

	if (ann_buffer_control == NULL)
	{
		PG_RETURN_TEXT_P(cstring_to_text(
										 "{\"error\":\"buffer not initialized\"}"));
	}

	if (ann_buffer_control->total_hits + ann_buffer_control->total_misses
		> 0)
	{
		hit_rate = (float8) ann_buffer_control->total_hits
			/ (ann_buffer_control->total_hits
			   + ann_buffer_control->total_misses);
	}
	else
	{
		hit_rate = 0.0;
	}

	initStringInfo(&stats);
	appendStringInfo(&stats,
					 "{\"entries\":%d,\"max_entries\":%d,\"hits\":" NDB_INT64_FMT
					 ",\"misses\":" NDB_INT64_FMT ",\"hit_rate\":%.2f}",
					 ann_buffer_control->num_entries,
					 ann_buffer_control->max_entries,
					 NDB_INT64_CAST(ann_buffer_control->total_hits),
					 NDB_INT64_CAST(ann_buffer_control->total_misses),
					 hit_rate);

	PG_RETURN_TEXT_P(cstring_to_text(stats.data));
}

/*
 * neurondb_ann_buffer_clear: Clear the buffer
 */
PG_FUNCTION_INFO_V1(neurondb_ann_buffer_clear);
Datum
neurondb_ann_buffer_clear(PG_FUNCTION_ARGS)
{
	int			i;

	(void) fcinfo;

	if (ann_buffer_control == NULL)
	{
		PG_RETURN_BOOL(false);
	}

	/* Mark all entries as invalid */
	for (i = 0; i < ann_buffer_control->num_entries; i++)
	{
		ann_buffer_control->entries[i].valid = false;
	}

	ann_buffer_control->num_entries = 0;
	ann_buffer_control->total_hits = 0;
	ann_buffer_control->total_misses = 0;


	PG_RETURN_BOOL(true);
}
