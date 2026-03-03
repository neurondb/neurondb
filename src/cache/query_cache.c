/*-------------------------------------------------------------------------
 *
 * query_cache.c
 *    Intelligent caching layer for frequent vector queries
 *
 * Implements a query result cache for vector similarity searches
 * to improve performance for repeated queries.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/cache/query_cache.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_compat.h"
#include "fmgr.h"
#include "funcapi.h"
#include "access/htup_details.h"
#include "access/tupdesc.h"
#include "utils/builtins.h"
#include "utils/memutils.h"
#include "common/hashfn.h"
#include "lib/stringinfo.h"
#include "utils/timestamp.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/* GUC variables */
int			neurondb_query_cache_size = 1000;	/* Maximum cache entries */
int			neurondb_query_cache_ttl = 3600;		/* Default TTL in seconds */

/*
 * Query cache entry structure
 * Stored in hash table with query_hash as key
 */
typedef struct QueryCacheEntry
{
	char		query_hash[64];		/* Hash key (null-terminated) */
	ItemPointerData *results;		/* Cached result TIDs */
	float4	   *distances;			/* Cached distances */
	int			result_count;		/* Number of results */
	TimestampTz cached_at;			/* When this was cached */
	TimestampTz expires_at;			/* When this expires */
	uint64		hits;				/* Cache hit count */
} QueryCacheEntry;

/*
 * Cache statistics
 */
typedef struct QueryCacheStats
{
	uint64		hits;
	uint64		misses;
	uint64		entries;
	uint64		evictions;
} QueryCacheStats;

/*
 * Global cache hash table and statistics
 */
static HTAB *query_cache_hash = NULL;
static QueryCacheStats cache_stats = {0, 0, 0, 0};
static slock_t cache_lock;			/* Spinlock for cache access */

/*
 * Generate cache key hash from query vector and parameters
 * Returns a 64-character hex string hash
 */
static void
neurondb_generate_cache_key(const float4 *query_vector, int dim, int k, int strategy,
							char *key_out, size_t key_size)
{
	uint32		hash_value;
	uint32		hash1, hash2;
	int			i;

	if (key_size < 64)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("cache key buffer too small")));

	/* Use PostgreSQL's hash_any for hashing */
	hash1 = hash_any((const unsigned char *) &dim, sizeof(dim));
	hash1 = hash_combine(hash1, hash_any((const unsigned char *) &k, sizeof(k)));
	hash1 = hash_combine(hash1, hash_any((const unsigned char *) &strategy, sizeof(strategy)));

	/* Hash query vector */
	hash2 = hash_any((const unsigned char *) query_vector, sizeof(float4) * dim);
	hash_value = hash_combine(hash1, hash2);

	/* Convert to hex string (use 32-bit hash, pad to 16 hex chars) */
	snprintf(key_out, key_size, "%08x%08x", hash_value, hash_value);
}

/*
 * Hash table key comparison function
 */
static int
query_cache_key_match(const void *key1, const void *key2, Size keysize)
{
	(void) keysize;  /* keysize is always 64 for our cache keys */
	return strcmp((const char *) key1, (const char *) key2);
}

/*
 * Initialize query cache
 * Note: This function is called from _PG_init() in neurondb.c
 */
void
neurondb_init_query_cache(void)
{
	HASHCTL		ctl;

	if (query_cache_hash != NULL)
		return;					/* Already initialized */

	SpinLockInit(&cache_lock);

	MemSet(&ctl, 0, sizeof(HASHCTL));
	ctl.keysize = 64;			/* Hash key size */
	ctl.entrysize = sizeof(QueryCacheEntry);
	ctl.hcxt = CacheMemoryContext;	/* Use cache memory context */
	ctl.hash = string_hash;
	ctl.match = query_cache_key_match;

	query_cache_hash = hash_create("neurondb query cache",
								   neurondb_query_cache_size,
								   &ctl,
								   HASH_ELEM | HASH_FUNCTION | HASH_COMPARE | HASH_CONTEXT);
}

/*
 * Evict expired entries from cache
 */
static void
neurondb_query_cache_evict_expired(void)
{
	HASH_SEQ_STATUS status;
	QueryCacheEntry *entry = NULL;
	TimestampTz now = GetCurrentTimestamp();
	int			evicted = 0;

	if (query_cache_hash == NULL)
		return;

	hash_seq_init(&status, query_cache_hash);

	while ((entry = (QueryCacheEntry *) hash_seq_search(&status)) != NULL)
	{
		if (entry->expires_at < now)
		{
			/* Entry expired - free memory and remove */
			if (entry->results)
				pfree(entry->results);
			if (entry->distances)
				pfree(entry->distances);

			hash_search(query_cache_hash, entry->query_hash, HASH_REMOVE, NULL);
			evicted++;
		}
	}

	if (evicted > 0)
	{
		SpinLockAcquire(&cache_lock);
		cache_stats.evictions += evicted;
		cache_stats.entries -= evicted;
		SpinLockRelease(&cache_lock);
	}
}

/*
 * Lookup cached query results
 * Returns true if found and not expired, false otherwise
 */
bool
neurondb_query_cache_lookup(const float4 *query_vector, int dim, int k, int strategy,
							ItemPointerData **results, float4 **distances, int *result_count)
{
	QueryCacheEntry *entry = NULL;
	char		cache_key[64];
	TimestampTz now;
	MemoryContext oldctx;

	if (query_cache_hash == NULL)
		neurondb_init_query_cache();

	if (query_vector == NULL || dim <= 0 || k <= 0)
		return false;

	/* Generate cache key */
	neurondb_generate_cache_key(query_vector, dim, k, strategy, cache_key, sizeof(cache_key));

	/* Evict expired entries periodically */
	now = GetCurrentTimestamp();
	if (cache_stats.entries > 0 && (cache_stats.entries % 100 == 0))
		neurondb_query_cache_evict_expired();

	/* Lookup entry */
	entry = (QueryCacheEntry *) hash_search(query_cache_hash, cache_key, HASH_FIND, NULL);

	if (entry == NULL)
	{
		SpinLockAcquire(&cache_lock);
		cache_stats.misses++;
		SpinLockRelease(&cache_lock);
		return false;
	}

	/* Check if expired */
	if (entry->expires_at < now)
	{
		/* Entry expired - remove it */
		if (entry->results)
			pfree(entry->results);
		if (entry->distances)
			pfree(entry->distances);
		hash_search(query_cache_hash, cache_key, HASH_REMOVE, NULL);

		SpinLockAcquire(&cache_lock);
		cache_stats.misses++;
		cache_stats.entries--;
		cache_stats.evictions++;
		SpinLockRelease(&cache_lock);
		return false;
	}

	/* Cache hit - copy results to caller's memory context */
	oldctx = MemoryContextSwitchTo(CurrentMemoryContext);

	if (entry->result_count > 0)
	{
		nalloc(*results, ItemPointerData, entry->result_count);
		nalloc(*distances, float4, entry->result_count);
		memcpy(*results, entry->results, sizeof(ItemPointerData) * entry->result_count);
		memcpy(*distances, entry->distances, sizeof(float4) * entry->result_count);
		*result_count = entry->result_count;
	}
	else
	{
		*results = NULL;
		*distances = NULL;
		*result_count = 0;
	}

	MemoryContextSwitchTo(oldctx);

	/* Update hit statistics */
	SpinLockAcquire(&cache_lock);
	cache_stats.hits++;
	entry->hits++;
	SpinLockRelease(&cache_lock);

	return true;
}

/*
 * Store query results in cache
 */
void
neurondb_query_cache_store(const float4 *query_vector, int dim, int k, int strategy,
						   ItemPointerData *results, float4 *distances, int result_count,
						   int ttl_seconds)
{
	QueryCacheEntry *entry = NULL;
	char		cache_key[64];
	bool		found;
	TimestampTz now;
	MemoryContext oldctx;

	if (query_cache_hash == NULL)
		neurondb_init_query_cache();

	if (query_vector == NULL || dim <= 0 || k <= 0)
		return;

	if (ttl_seconds <= 0)
		ttl_seconds = neurondb_query_cache_ttl;

	/* Generate cache key */
	neurondb_generate_cache_key(query_vector, dim, k, strategy, cache_key, sizeof(cache_key));

	/* Check cache size and evict if needed */
	if (cache_stats.entries >= (uint64) neurondb_query_cache_size)
	{
		neurondb_query_cache_evict_expired();
		/* If still full, evict oldest entry (simple LRU would be better) */
		if (cache_stats.entries >= (uint64) neurondb_query_cache_size)
		{
			HASH_SEQ_STATUS status;
			QueryCacheEntry *oldest = NULL;
			TimestampTz oldest_time = GetCurrentTimestamp() + 1;

			hash_seq_init(&status, query_cache_hash);
			while ((entry = (QueryCacheEntry *) hash_seq_search(&status)) != NULL)
			{
				if (entry->cached_at < oldest_time)
				{
					oldest_time = entry->cached_at;
					oldest = entry;
				}
			}

			if (oldest != NULL)
			{
				if (oldest->results)
					pfree(oldest->results);
				if (oldest->distances)
					pfree(oldest->distances);
				hash_search(query_cache_hash, oldest->query_hash, HASH_REMOVE, NULL);

				SpinLockAcquire(&cache_lock);
				cache_stats.entries--;
				cache_stats.evictions++;
				SpinLockRelease(&cache_lock);
			}
		}
	}

	now = GetCurrentTimestamp();

	/* Enter or update entry */
	entry = (QueryCacheEntry *) hash_search(query_cache_hash, cache_key, HASH_ENTER, &found);

	if (entry == NULL)
		return;					/* Out of memory */

	/* Free old data if updating */
	if (found)
	{
		if (entry->results)
			pfree(entry->results);
		if (entry->distances)
			pfree(entry->distances);
	}
	else
	{
		/* New entry */
		SpinLockAcquire(&cache_lock);
		cache_stats.entries++;
		SpinLockRelease(&cache_lock);
	}

	/* Copy key */
	strlcpy(entry->query_hash, cache_key, sizeof(entry->query_hash));

	/* Allocate and copy results */
	oldctx = MemoryContextSwitchTo(CacheMemoryContext);

	if (result_count > 0)
	{
		nalloc(entry->results, ItemPointerData, result_count);
		nalloc(entry->distances, float4, result_count);
		memcpy(entry->results, results, sizeof(ItemPointerData) * result_count);
		memcpy(entry->distances, distances, sizeof(float4) * result_count);
	}
	else
	{
		entry->results = NULL;
		entry->distances = NULL;
	}

	entry->result_count = result_count;
	entry->cached_at = now;
	entry->expires_at = TimestampTzPlusMilliseconds(now, ttl_seconds * 1000);
	if (!found)
		entry->hits = 0;

	MemoryContextSwitchTo(oldctx);
}

/*
 * Clear query cache
 */
void
neurondb_query_cache_clear(void)
{
	HASH_SEQ_STATUS status;
	QueryCacheEntry *entry = NULL;

	if (query_cache_hash == NULL)
		return;

	hash_seq_init(&status, query_cache_hash);

	while ((entry = (QueryCacheEntry *) hash_seq_search(&status)) != NULL)
	{
		if (entry->results)
			pfree(entry->results);
		if (entry->distances)
			pfree(entry->distances);
	}

	hash_destroy(query_cache_hash);
	query_cache_hash = NULL;

	SpinLockAcquire(&cache_lock);
	cache_stats.entries = 0;
	cache_stats.hits = 0;
	cache_stats.misses = 0;
	cache_stats.evictions = 0;
	SpinLockRelease(&cache_lock);
}

/*
 * Get cache statistics
 */
void
neurondb_query_cache_get_stats(uint64 *hits, uint64 *misses, uint64 *entries, uint64 *evictions)
{
	SpinLockAcquire(&cache_lock);
	if (hits)
		*hits = cache_stats.hits;
	if (misses)
		*misses = cache_stats.misses;
	if (entries)
		*entries = cache_stats.entries;
	if (evictions)
		*evictions = cache_stats.evictions;
	SpinLockRelease(&cache_lock);
}

