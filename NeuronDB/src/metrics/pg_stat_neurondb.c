/*-------------------------------------------------------------------------
 *
 * pg_stat_neurondb.c
 *      Statistics view for NeuronDB ANN operations – DETAILED IMPLEMENTATION
 *
 * Implements the pg_stat_neurondb view that exposes internal ANN/statistics:
 *  - Query type/counts by index (HNSW, IVF, Hybrid, Total)
 *  - Latency (average and max in ms, rolling)
 *  - Recall@K metrics (1, 10, 100, floating-point [0.0-1.0])
 *  - Cache statistics (hits, misses)
 *  - Index rebuild count and last reset timestamp
 *
 * All routines pay careful attention to Postgres context management,
 * set-returning function API, type safety, and zero cost for unused stats.
 * Thread/concurrency safety is not handled here (assumes single backend stats).
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/pg_stat_neurondb.c
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/timestamp.h"
#include "catalog/pg_type.h"
#include "access/htup_details.h"
#include "access/tupdesc.h"
#include "utils/rel.h"
#include "utils/syscache.h"
#include "utils/lsyscache.h"
#include "executor/spi.h"
#include "funcapi.h"
#include "utils/guc.h"
#include "utils/tuplestore.h"
#include <math.h>
#include <string.h>

/*
 * Structure holding all NeuronDB statistics for this backend.
 *
 * All fields are 64-bit so they do not wrap during long sessions.
 * Latency is maintained as a "rolling average" for simplicity. Recall is measured as "percent * 100".
 * last_reset is a TimestampTz (internally, also a 64-bit integer).
 */
typedef struct NeuronDBStats
{
	uint64		queries_total;	/* Total number of ANN/search queries */
	uint64		queries_hnsw;	/* HNSW index queries */
	uint64		queries_ivf;	/* IVF index queries */
	uint64		queries_hybrid; /* Hybrid index queries */
	uint64		avg_latency_ms; /* Rolling average query latency, ms */
	uint64		max_latency_ms; /* Maximum query latency, ms */
	uint64		recall_at_1;	/* Recall@1, as (fraction*100), e.g. 98.2% ->
								 * 9820 */
	uint64		recall_at_10;	/* Recall@10, as (fraction*100) */
	uint64		recall_at_100;	/* Recall@100, as (fraction*100) */
	uint64		cache_hits;		/* Number of plan/vector cache hits */
	uint64		cache_misses;	/* Number of plan/vector cache misses */
	uint64		index_rebuilds; /* Number of index rebuilds triggered in
								 * session */
	uint64		last_reset;		/* TimestampTz, last statistics reset */
}			NeuronDBStats;

/*
 * The global statistics variable.
 * This is one-per-backend (not shared memory); session-only statistics.
 * Zero-initialized.
 */
static NeuronDBStats g_stats =
{
	0
};

/*
 * SQL-callable function: pg_stat_neurondb
 * Returns a single-row set exposing the statistics above.
 *
 * Carefully handles Postgres SRF setup:
 *  - Allocates tuple descriptor if needed
 *  - Materializes a single row with all values
 *  - Columns: queries_total, queries_hnsw, queries_ivf, queries_hybrid,
 *             avg_latency_ms, max_latency_ms,
 *             recall_at_1, recall_at_10, recall_at_100 (float8 fraction 0-1),
 *             cache_hits, cache_misses, index_rebuilds, last_reset (timestamptz)
 */
PG_FUNCTION_INFO_V1(pg_stat_neurondb);
Datum
pg_stat_neurondb(PG_FUNCTION_ARGS)
{
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	TupleDesc	tupdesc = NULL;
	Tuplestorestate *tupstore = NULL;
	MemoryContext per_query_ctx = NULL;
	MemoryContext oldcontext = NULL;
	Datum		values[13];
	bool		nulls[13];

	/*
	 * Validate that this function is properly called in SRF context.
	 */
	if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("neurondb: set-valued function called "
						"in context that cannot accept a set")));

	/*
	 * Build or acquire the returned tuple descriptor (rowtype). This is
	 * either provided by the caller or constructed by us.
	 */
	if (rsinfo->expectedDesc == NULL)
	{
		if (get_call_result_type(fcinfo, NULL, &tupdesc)
			!= TYPEFUNC_COMPOSITE)
			elog(ERROR, "neurondb: return type must be a row type");
		rsinfo->expectedDesc = tupdesc;
	}
	tupdesc = rsinfo->expectedDesc;

	/*
	 * Switch to per-query memory context for allocation of the tuplestore.
	 */
	per_query_ctx = rsinfo->econtext->ecxt_per_query_memory;
	oldcontext = MemoryContextSwitchTo(per_query_ctx);

	/*
	 * Initialize the tuplestore (used by SRFs for materialized sets).
	 * Estimation: only one row, but 1k slots allowed for future expansion.
	 */
	tupstore = tuplestore_begin_heap(true, false, 1024);

	rsinfo->returnMode = SFRM_Materialize;
	rsinfo->setResult = tupstore;
	rsinfo->setDesc = tupdesc;

	MemoryContextSwitchTo(oldcontext);

	/*
	 * Prepare output values/nulls. All columns always filled, so 'nulls'
	 * array can stay all false.
	 */
	memset(values, 0, sizeof(values));
	memset(nulls, 0, sizeof(nulls));

	/*
	 * Fill statistic columns, matching order of view definition. Recall
	 * values are floating-point (division by 100.0 to convert to fraction);
	 * all other metrics are directly cast to Datum.
	 */
	values[0] = Int64GetDatum(g_stats.queries_total);
	values[1] = Int64GetDatum(g_stats.queries_hnsw);
	values[2] = Int64GetDatum(g_stats.queries_ivf);
	values[3] = Int64GetDatum(g_stats.queries_hybrid);
	values[4] = Int64GetDatum(g_stats.avg_latency_ms);
	values[5] = Int64GetDatum(g_stats.max_latency_ms);
	values[6] = Float8GetDatum(
							   (double) g_stats.recall_at_1 / 100.0);
	values[7] = Float8GetDatum((double) g_stats.recall_at_10
							   / 100.0);
	values[8] = Float8GetDatum((double) g_stats.recall_at_100
							   / 100.0);
	values[9] = Int64GetDatum(g_stats.cache_hits);
	values[10] = Int64GetDatum(g_stats.cache_misses);
	values[11] = Int64GetDatum(g_stats.index_rebuilds);
	values[12] = TimestampTzGetDatum(
									 g_stats.last_reset);

	/*
	 * Materialize the constructed row to the tuplestore.
	 */
	tuplestore_putvalues(tupstore, tupdesc, values, nulls);

	PG_RETURN_NULL();
}

/*
 * SQL-callable function: pg_neurondb_stat_reset()
 * Resets all collected statistics for this backend to zero and updates last_reset timestamp.
 */
PG_FUNCTION_INFO_V1(pg_neurondb_stat_reset);
Datum
pg_neurondb_stat_reset(PG_FUNCTION_ARGS)
{
	(void) fcinfo;

	memset(&g_stats, 0, sizeof(NeuronDBStats));
	g_stats.last_reset = GetCurrentTimestamp();

	PG_RETURN_VOID();
}

/*
 * Internal helper: neurondb_update_query_stats
 * Updates counters for query, latency, recall. Not thread-safe; for backend-local use.
 *  - index_type: 1 (HNSW), 2 (IVF), 3 (Hybrid)
 *  - latency_ms: observed query latency in milliseconds
 *  - recall_percent: recall as a percent (e.g., 95.20 for 95.2%), *100 for tracking as int
 * Precision: latency average is a simple rolling mean (should be improved for true EWMA).
 */
static void
__attribute__((unused))
neurondb_update_query_stats(int index_type, int latency_ms, int recall_percent)
{
	g_stats.queries_total++;

	switch (index_type)
	{
		case 1:
			g_stats.queries_hnsw++;
			break;
		case 2:
			g_stats.queries_ivf++;
			break;
		case 3:
			g_stats.queries_hybrid++;
			break;
		default:
			break;
	}

	/*
	 * Update latency via rolling mean (biased but simple, to keep atomicity);
	 * for each incoming, mean = (old + new) / 2. More advanced EWMA can be
	 * used if needed.
	 */
	g_stats.avg_latency_ms = (g_stats.avg_latency_ms == 0)
		? (uint64) latency_ms
		: (g_stats.avg_latency_ms + (uint64) latency_ms) / 2;

	if ((uint64) latency_ms > g_stats.max_latency_ms)
		g_stats.max_latency_ms = (uint64) latency_ms;

	/*
	 * Update recall stats if recall_percent positive (fraction*100 as int).
	 * Again, rolling mean, same comments as above.
	 */
	if (recall_percent > 0)
	{
		uint64		recall_u = (uint64) recall_percent;

		g_stats.recall_at_1 = (g_stats.recall_at_1 == 0)
			? recall_u
			: (g_stats.recall_at_1 + recall_u) / 2;
		g_stats.recall_at_10 = (g_stats.recall_at_10 == 0)
			? recall_u
			: (g_stats.recall_at_10 + recall_u) / 2;
		g_stats.recall_at_100 = (g_stats.recall_at_100 == 0)
			? recall_u
			: (g_stats.recall_at_100 + recall_u) / 2;

		/*
		 * If/when supports true per-K stats, extend here; for now assumes
		 * recall_percent covers K=1,10,100 identically.
		 */
	}
}

/*
 * Internal helper: neurondb_update_cache_stats
 * Updates cache_hits/cache_misses; backend-local only.
 */
static void
__attribute__((unused))
neurondb_update_cache_stats(bool hit)
{
	if (hit)
		g_stats.cache_hits++;
	else
		g_stats.cache_misses++;
}

/*
 * Internal helper: neurondb_update_rebuild_stats
 * Increment index_rebuilds count each time a rebuild event occurs.
 */
static void
__attribute__((unused))
neurondb_update_rebuild_stats(void)
{
	g_stats.index_rebuilds++;
}
