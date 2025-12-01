/*-------------------------------------------------------------------------
 *
 * index_cache.c
 *		HNSW entrypoint cache for fast query startup
 *
 * Caches HNSW graph entry points in shared memory to avoid
 * repeated metadata page reads. Uses LRU eviction.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/index/index_cache.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_scan.h"
#include "fmgr.h"
#include "access/htup_details.h"
#include "storage/lwlock.h"
#include "storage/shmem.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/timestamp.h"
#include "utils/typcache.h"
#include "funcapi.h"

#define MAX_CACHE_ENTRIES 128

/*
 * Cached entry point
 */
typedef struct CacheEntry
{
	Oid			indexOid;
	BlockNumber entryPoint;
	int			entryLevel;
	int			maxLevel;
	TimestampTz lastAccess;
	bool		valid;
}			CacheEntry;

/*
 * Entrypoint cache (in shared memory)
 */
typedef struct EntrypointCache
{
	LWLock	   *lock;
	int			maxEntries;
	int			currentEntries;
	CacheEntry	entries[MAX_CACHE_ENTRIES];
}			EntrypointCache;

static EntrypointCache * ep_cache = NULL;
static int	cache_size = MAX_CACHE_ENTRIES;

/*
 * Initialize GUC
 */
void
entrypoint_cache_init_guc(void)
{
	DefineCustomIntVariable("neurondb.entrypoint_cache_size",
							"Number of HNSW entrypoints to cache",
							NULL,
							&cache_size,
							MAX_CACHE_ENTRIES,
							16,
							1024,
							PGC_POSTMASTER,
							0,
							NULL,
							NULL,
							NULL);
}

/*
 * Shared memory sizing
 */
Size
entrypoint_cache_shmem_size(void)
{
	return MAXALIGN(sizeof(EntrypointCache));
}

/*
 * Initialize shared memory
 */
void
entrypoint_cache_shmem_init(void)
{
	bool		found;

	ep_cache =
		(EntrypointCache *) ShmemInitStruct("neurondb_entrypoint_cache",
											entrypoint_cache_shmem_size(),
											&found);

	if (!found)
	{
		ep_cache->lock =
			&(GetNamedLWLockTranche("neurondb_entrypoint_cache"))
			->lock;
		ep_cache->maxEntries = cache_size;
		ep_cache->currentEntries = 0;

		for (int i = 0; i < MAX_CACHE_ENTRIES; i++)
		{
			ep_cache->entries[i].valid = false;
		}
	}
}

/*
 * Lookup entrypoint in cache
 */
bool
entrypoint_cache_lookup(Oid indexOid,
						BlockNumber * entryPoint,
						int *entryLevel,
						int *maxLevel)
{
	bool		found = false;
	int			i;

	if (ep_cache == NULL)
		return false;

	LWLockAcquire(ep_cache->lock, LW_SHARED);

	for (i = 0; i < ep_cache->currentEntries; i++)
	{
		if (ep_cache->entries[i].valid
			&& ep_cache->entries[i].indexOid == indexOid)
		{
			*entryPoint = ep_cache->entries[i].entryPoint;
			*entryLevel = ep_cache->entries[i].entryLevel;
			*maxLevel = ep_cache->entries[i].maxLevel;
			ep_cache->entries[i].lastAccess = GetCurrentTimestamp();
			found = true;
			break;
		}
	}

	LWLockRelease(ep_cache->lock);

	if (found)
		elog(DEBUG1,
			 "neurondb: Entrypoint cache HIT for index %u",
			 indexOid);
	else
		elog(DEBUG1,
			 "neurondb: Entrypoint cache MISS for index %u",
			 indexOid);

	return found;
}

/*
 * Store entrypoint in cache
 */
void
entrypoint_cache_store(Oid indexOid,
					   BlockNumber entryPoint,
					   int entryLevel,
					   int maxLevel)
{
	int			victim = -1;
	TimestampTz oldest = GetCurrentTimestamp();
	int			i;

	if (ep_cache == NULL)
		return;

	LWLockAcquire(ep_cache->lock, LW_EXCLUSIVE);

	/* Check if already exists */
	for (i = 0; i < ep_cache->currentEntries; i++)
	{
		if (ep_cache->entries[i].valid
			&& ep_cache->entries[i].indexOid == indexOid)
		{
			/* Update existing */
			ep_cache->entries[i].entryPoint = entryPoint;
			ep_cache->entries[i].entryLevel = entryLevel;
			ep_cache->entries[i].maxLevel = maxLevel;
			ep_cache->entries[i].lastAccess = GetCurrentTimestamp();
			LWLockRelease(ep_cache->lock);
			return;
		}
	}

	/* Find empty slot or LRU victim */
	if (ep_cache->currentEntries < ep_cache->maxEntries)
	{
		victim = ep_cache->currentEntries;
		ep_cache->currentEntries++;
	}
	else
	{
		/* Find LRU entry */
		for (i = 0; i < ep_cache->currentEntries; i++)
		{
			if (ep_cache->entries[i].lastAccess < oldest)
			{
				oldest = ep_cache->entries[i].lastAccess;
				victim = i;
			}
		}
	}

	/* Store entry */
	if (victim >= 0)
	{
		ep_cache->entries[victim].indexOid = indexOid;
		ep_cache->entries[victim].entryPoint = entryPoint;
		ep_cache->entries[victim].entryLevel = entryLevel;
		ep_cache->entries[victim].maxLevel = maxLevel;
		ep_cache->entries[victim].lastAccess = GetCurrentTimestamp();
		ep_cache->entries[victim].valid = true;

		elog(DEBUG1,
			 "neurondb: Stored entrypoint in cache for index %u",
			 indexOid);
	}

	LWLockRelease(ep_cache->lock);
}

/*
 * Invalidate cached entrypoint
 */
void
entrypoint_cache_invalidate(Oid indexOid)
{
	int			i;

	if (ep_cache == NULL)
		return;

	LWLockAcquire(ep_cache->lock, LW_EXCLUSIVE);

	for (i = 0; i < ep_cache->currentEntries; i++)
	{
		if (ep_cache->entries[i].valid
			&& ep_cache->entries[i].indexOid == indexOid)
		{
			ep_cache->entries[i].valid = false;
			elog(DEBUG1,
				 "neurondb: Invalidated entrypoint cache for "
				 "index %u",
				 indexOid);
			break;
		}
	}

	LWLockRelease(ep_cache->lock);
}

/*
 * Clear entire cache
 */
PG_FUNCTION_INFO_V1(neurondb_clear_entrypoint_cache);

Datum
neurondb_clear_entrypoint_cache(PG_FUNCTION_ARGS)
{
	int			i;

	if (ep_cache == NULL)
		ereport(ERROR,
				(errmsg("neurondb: Entrypoint cache not initialized")));

	LWLockAcquire(ep_cache->lock, LW_EXCLUSIVE);

	for (i = 0; i < MAX_CACHE_ENTRIES; i++)
	{
		ep_cache->entries[i].valid = false;
	}
	ep_cache->currentEntries = 0;

	LWLockRelease(ep_cache->lock);


	PG_RETURN_VOID();
}

/*
 * Get cache statistics
 */
PG_FUNCTION_INFO_V1(neurondb_entrypoint_cache_stats);

Datum
neurondb_entrypoint_cache_stats(PG_FUNCTION_ARGS)
{
	TupleDesc	tupdesc;
	Datum		values[3];
	bool		nulls[3];
	HeapTuple	tuple;
	int			valid_count = 0;
	int			i;

	if (ep_cache == NULL)
		ereport(ERROR,
				(errmsg("neurondb: Entrypoint cache not initialized")));

	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("function returning record called in "
						"context that cannot accept type "
						"record")));

	tupdesc = BlessTupleDesc(tupdesc);

	LWLockAcquire(ep_cache->lock, LW_SHARED);

	for (i = 0; i < ep_cache->currentEntries; i++)
	{
		if (ep_cache->entries[i].valid)
			valid_count++;
	}

	values[0] = Int32GetDatum(ep_cache->maxEntries);
	values[1] = Int32GetDatum(ep_cache->currentEntries);
	values[2] = Int32GetDatum(valid_count);

	LWLockRelease(ep_cache->lock);

	memset(nulls, 0, sizeof(nulls));

	tuple = heap_form_tuple(tupdesc, values, nulls);

	PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}
