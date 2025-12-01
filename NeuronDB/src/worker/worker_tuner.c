/*-------------------------------------------------------------------------
 *
 * bgworker_tuner.c
 *		Background worker: neuranmon - Auto-tuner and monitoring
 *
 * This worker samples queries, updates ef_search and hybrid weights from
 * live SLOs, rotates caches, records recall@k metrics, and exports
 * Prometheus metrics.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/bgworker_tuner.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb_compat.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "postmaster/bgworker.h"
#include "storage/ipc.h"
#include "storage/latch.h"
#include "storage/lwlock.h"
#include "storage/proc.h"
#include "storage/shmem.h"
#include "executor/spi.h"
#include "utils/guc.h"
#include "utils/timestamp.h"
#include "utils/builtins.h"
#include "utils/snapmgr.h"
#include "lib/stringinfo.h"
#include "catalog/pg_type.h"
#include "access/xact.h"
#include <stdlib.h>
#include <string.h>

#include "neurondb_bgworkers.h"
#include "neurondb_validation.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_guc.h"

/* GUC variables are now centralized in neurondb_guc.c */
/* Use GetConfigOption() to retrieve GUC values */

/* Helper to get int GUC value */
static int
get_guc_int(const char *name, int default_val)
{
	const char *val = GetConfigOption(name, true, false);
	return val ? atoi(val) : default_val;
}

/* Helper to get double GUC value */
static double
get_guc_double(const char *name, double default_val)
{
	const char *val = GetConfigOption(name, true, false);
	return val ? atof(val) : default_val;
}

/* Helper to get bool GUC value */
static bool
get_guc_bool(const char *name, bool default_val)
{
	const char *val = GetConfigOption(name, true, false);
	if (!val)
		return default_val;
	return (strcmp(val, "on") == 0 || strcmp(val, "true") == 0 || strcmp(val, "1") == 0);
}

/* Shared memory structure */
typedef struct NeuranmonSharedState
{
	LWLock	   *lock;
	int64		queries_sampled;
	int64		adjustments_made;
	double		avg_latency_ms;
	double		avg_recall;
	int			current_ef_search;
	double		current_hybrid_weight;
	TimestampTz last_heartbeat;
	pid_t		worker_pid;
}			NeuranmonSharedState;

static NeuranmonSharedState * neuranmon_state = NULL;

/* Forward declarations */
PGDLLEXPORT void neuranmon_main(Datum main_arg);
static void neuranmon_sigterm(SIGNAL_ARGS);
static void neuranmon_sighup(SIGNAL_ARGS);
static void sample_and_tune(void);
static void rotate_caches(void);
static void record_metrics(void);
static void export_prometheus_metrics(void);

static volatile sig_atomic_t got_sigterm = false;
static volatile sig_atomic_t got_sighup = false;

/*
 * Signal handlers
 */
static void
neuranmon_sigterm(SIGNAL_ARGS)
{
	int			save_errno = errno;

	(void) postgres_signal_arg; /* Unused */

	got_sigterm = true;
	SetLatch(MyLatch);

	errno = save_errno;
}

static void
neuranmon_sighup(SIGNAL_ARGS)
{
	int			save_errno = errno;

	(void) postgres_signal_arg; /* Unused */

	got_sighup = true;
	SetLatch(MyLatch);

	errno = save_errno;
}

/* GUC initialization is now centralized in neurondb_guc.c */

/*
 * Estimate shared memory size
 */
Size
neuranmon_shmem_size(void)
{
	return MAXALIGN(sizeof(NeuranmonSharedState));
}

/*
 * Initialize shared memory
 */
void
neuranmon_shmem_init(void)
{
	bool		found;

	LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);

	neuranmon_state = ShmemInitStruct(
									  "NeuronDB Tuner Worker State", neuranmon_shmem_size(), &found);

	if (!found)
	{
		neuranmon_state->lock =
			&(GetNamedLWLockTranche("neurondb_tuner"))->lock;
		neuranmon_state->queries_sampled = 0;
		neuranmon_state->adjustments_made = 0;
		neuranmon_state->avg_latency_ms = 0.0;
		neuranmon_state->avg_recall = 0.0;
		neuranmon_state->current_ef_search = 64;	/* default */
		neuranmon_state->current_hybrid_weight = 0.7;	/* default */
		neuranmon_state->last_heartbeat = GetCurrentTimestamp();
		neuranmon_state->worker_pid = 0;
	}

	LWLockRelease(AddinShmemInitLock);
}

/*
 * Main entry point for tuner worker
 */
PGDLLEXPORT void
neuranmon_main(Datum main_arg)
{
	(void) main_arg;			/* Unused */

	/* Defensive: check shared memory and lock */
	if (neuranmon_state == NULL)
	{
		elog(LOG, "neurondb: neuranmon_state is NULL, shared memory not initialized. Exiting to prevent segfault.");
		proc_exit(1);
	}
	if (neuranmon_state->lock == NULL)
	{
		elog(LOG, "neurondb: neuranmon_state->lock is NULL, LWLock not initialized. Exiting to prevent segfault.");
		proc_exit(1);
	}

	/* Establish signal handlers */
	pqsignal(SIGTERM, neuranmon_sigterm);
	pqsignal(SIGHUP, neuranmon_sighup);
	/* Ignore SIGPIPE to prevent crashes when writing to broken pipes */
	pqsignal(SIGPIPE, SIG_IGN);

	BackgroundWorkerUnblockSignals();

	/* Connect to database */
	BackgroundWorkerInitializeConnection("postgres", NULL, 0);

	/* Initialize shared state */
	LWLockAcquire(neuranmon_state->lock, LW_EXCLUSIVE);
	neuranmon_state->worker_pid = MyProcPid;
	neuranmon_state->last_heartbeat = GetCurrentTimestamp();
	LWLockRelease(neuranmon_state->lock);

	elog(LOG, "neurondb: neuranmon worker started (PID %d)", MyProcPid);

	/* Main loop */
	while (!got_sigterm)
	{
		int			rc;

		if (got_sighup)
		{
			got_sighup = false;
			ProcessConfigFile(PGC_SIGHUP);
			elog(LOG, "neurondb: neuranmon reloaded configuration");
		}

		if (!get_guc_bool("neurondb.neuranmon_enabled", true))
		{
			elog(LOG, "neurondb: neuranmon disabled, exiting");
			proc_exit(0);
		}

		/* Update heartbeat */
		LWLockAcquire(neuranmon_state->lock, LW_EXCLUSIVE);
		neuranmon_state->last_heartbeat = GetCurrentTimestamp();
		LWLockRelease(neuranmon_state->lock);

		/* Perform tuning tasks with error handling */
		PG_TRY();
		{
			StartTransactionCommand();
			PushActiveSnapshot(GetTransactionSnapshot());

			sample_and_tune();
			rotate_caches();
			record_metrics();
			export_prometheus_metrics();

			PopActiveSnapshot();
			CommitTransactionCommand();
		}
		PG_CATCH();
		{
			/* Clean up on error */
			if (IsTransactionState())
				AbortCurrentTransaction();

			EmitErrorReport();
			FlushErrorState();
			elog(LOG,
				 "neurondb: tuner worker caught exception, all "
				 "cleaned up. Restarting loop.");
		}
		PG_END_TRY();

		/* Wait for next cycle */
		rc = WaitLatch(MyLatch,
					   WL_LATCH_SET | WL_TIMEOUT | WL_POSTMASTER_DEATH,
					   get_guc_int("neurondb.neuranmon_naptime", 60000),
					   0);
		ResetLatch(MyLatch);

		if (rc & WL_POSTMASTER_DEATH)
			proc_exit(1);
	}

	elog(LOG, "neurondb: neuranmon worker shutting down");
	proc_exit(0);
}

/*
 * Sample queries and adjust parameters based on SLOs
 */
static void
sample_and_tune(void)
{
	StringInfoData sql;
	int			ret;
	double		avg_latency = 0.0;
	double		avg_recall = 0.0;
	int			new_ef_search;
	double		new_hybrid_weight;
	bool		needs_adjustment = false;

	NDB_DECLARE(NdbSpiSession *, session);
	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		elog(ERROR, "neurondb: failed to begin SPI session in neuranmon");

	/* Check if table exists (before CREATE EXTENSION, this is expected) */
	ret = ndb_spi_execute(session, "SELECT 1 FROM pg_tables WHERE schemaname = "
							   "'neurondb' AND tablename = 'neurondb_query_metrics'",
							   true,
							   0);
	if (ret != SPI_OK_SELECT || SPI_processed == 0)
	{
		/*
		 * Table doesn't exist yet - extension not created. This is normal at
		 * startup.
		 */
		ndb_spi_session_end(&session);
		elog(DEBUG1,
			 "neurondb: tuner waiting for extension to be created");
		return;
	}

	/*
	 * Sample queries from pg_stat_statements if available, otherwise from
	 * query_history
	 */
	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT AVG(latency_ms) as avg_latency, "
					 "       AVG(recall_at_k) as avg_recall "
					 "FROM neurondb.query_metrics "
					 "WHERE query_timestamp > now() - interval '5 minutes' "
					 "  AND recall_at_k IS NOT NULL");

	ret = ndb_spi_execute(session, sql.data, true, 1);

	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		bool		isnull;
		Datum		latency_datum,
					recall_datum;

		latency_datum = SPI_getbinval(SPI_tuptable->vals[0],
									  SPI_tuptable->tupdesc,
									  1,
									  &isnull);
		if (!isnull)
			avg_latency = DatumGetFloat8(latency_datum);

		recall_datum = SPI_getbinval(SPI_tuptable->vals[0],
									 SPI_tuptable->tupdesc,
									 2,
									 &isnull);
		if (!isnull)
			avg_recall = DatumGetFloat8(recall_datum);
	}

	/* Defensive: check shared state before accessing */
	if (neuranmon_state == NULL || neuranmon_state->lock == NULL)
	{
		elog(LOG,
			 "neurondb: neuranmon_state invalid in sample_and_tune, "
			 "skipping adjustment");
		NDB_FREE(sql.data);
		ndb_spi_session_end(&session);
		return;
	}

	/* Adjust ef_search based on latency/recall trade-off */
	LWLockAcquire(neuranmon_state->lock, LW_SHARED);
	new_ef_search = neuranmon_state->current_ef_search;
	new_hybrid_weight = neuranmon_state->current_hybrid_weight;
	LWLockRelease(neuranmon_state->lock);

	if (avg_recall < get_guc_double("neurondb.neuranmon_target_recall", 0.95)
		&& avg_latency < get_guc_double("neurondb.neuranmon_target_latency", 100.0))
	{
		/* Recall too low, latency acceptable -> increase ef_search */
		new_ef_search = (int) (new_ef_search * 1.2);
		if (new_ef_search > 512)
			new_ef_search = 512;
		needs_adjustment = true;
	}
	else if (avg_recall >= get_guc_double("neurondb.neuranmon_target_recall", 0.95)
			 && avg_latency > get_guc_double("neurondb.neuranmon_target_latency", 100.0))
	{
		/* Recall good, latency too high -> decrease ef_search */
		new_ef_search = (int) (new_ef_search * 0.8);
		if (new_ef_search < 16)
			new_ef_search = 16;
		needs_adjustment = true;
	}

	/* Update shared state */
	if (needs_adjustment)
	{
		LWLockAcquire(neuranmon_state->lock, LW_EXCLUSIVE);
		neuranmon_state->current_ef_search = new_ef_search;
		neuranmon_state->current_hybrid_weight = new_hybrid_weight;
		neuranmon_state->adjustments_made++;
		neuranmon_state->avg_latency_ms = avg_latency;
		neuranmon_state->avg_recall = avg_recall;
		LWLockRelease(neuranmon_state->lock);

		elog(LOG,
			 "neurondb: tuned ef_search=%d (latency=%.2fms, "
			 "recall=%.3f)",
			 new_ef_search,
			 avg_latency,
			 avg_recall);

		/* Update GUC variable to apply the new ef_search value */
		PG_TRY();
		{
			initStringInfo(&sql);
			appendStringInfo(&sql,
							 "SET neurondb.hnsw_ef_search = %d",
							 new_ef_search);
			ret = ndb_spi_execute(session, sql.data, false, 0);
			if (ret != SPI_OK_UTILITY)
			{
				elog(WARNING,
					 "neurondb: failed to set neurondb.hnsw_ef_search to %d",
					 new_ef_search);
			}
			NDB_FREE(sql.data);
		}
		PG_CATCH();
		{
			NDB_FREE(sql.data);
			EmitErrorReport();
			FlushErrorState();
			elog(WARNING,
				 "neurondb: error setting neurondb.hnsw_ef_search, continuing");
		}
		PG_END_TRY();
	}
	/* Update statistics - defensive check before accessing */
	if (neuranmon_state != NULL && neuranmon_state->lock != NULL)
	{
		LWLockAcquire(neuranmon_state->lock, LW_EXCLUSIVE);
		neuranmon_state->queries_sampled += get_guc_int("neurondb.neuranmon_sample_size", 1000);
		LWLockRelease(neuranmon_state->lock);
	}

	/* Clean up StringInfo - must be freed before session end destroys context */
	NDB_FREE(sql.data);
	ndb_spi_session_end(&session);
}

/*
 * Rotate caches to maintain freshness
 */
static void
rotate_caches(void)
{
	StringInfoData sql;
	int			ret;
	NDB_DECLARE(NdbSpiSession *, session);

	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		return;

	PG_TRY();
	{
		/* Check if table exists first */
		ret = ndb_spi_execute(session,
								   "SELECT 1 FROM pg_tables WHERE schemaname = 'neurondb' "
								   "AND tablename = 'neurondb_embedding_cache'",
								   true,
								   0);
		if (ret != SPI_OK_SELECT || SPI_processed == 0)
		{
			/* Table doesn't exist, skip cache rotation */
			ndb_spi_session_end(&session);
			return;
		}

		/* Evict old cache entries */
		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "DELETE FROM neurondb.embedding_cache "
						 "WHERE last_accessed < now() - interval '1 hour' "
						 "  AND access_count < 10");

		ret = ndb_spi_execute(session, sql.data, false, 0);
		if (ret != SPI_OK_DELETE)
		{
			elog(WARNING,
				 "neurondb: cache rotation DELETE returned unexpected code %d",
				 ret);
		}
		else
		{
			elog(DEBUG1,
				 "neurondb: rotated cache (removed " NDB_UINT64_FMT
				 " entries)",
				 NDB_UINT64_CAST(SPI_processed));
		}

		NDB_FREE(sql.data);
	}
	PG_CATCH();
	{
		ndb_spi_session_end(&session);
		EmitErrorReport();
		FlushErrorState();
		elog(LOG, "neurondb: cache rotation failed, continuing");
	}
	PG_END_TRY();

	ndb_spi_session_end(&session);
}

/*
 * Record detailed metrics for analysis
 */
static void
record_metrics(void)
{
	StringInfoData sql;
	int			ret;

	NDB_DECLARE(NdbSpiSession *, session);
	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		return;

	PG_TRY();
	{
		/* Check if tables exist first */
		ret = ndb_spi_execute(session,
								   "SELECT 1 FROM pg_tables WHERE schemaname = 'neurondb' "
								   "AND tablename = 'neurondb_histograms'",
								   true,
								   0);
		if (ret != SPI_OK_SELECT || SPI_processed == 0)
		{
			/* Tables don't exist, skip metrics recording */
			ndb_spi_session_end(&session);
			return;
		}

		/* Record latency histogram */
		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "INSERT INTO neurondb.histograms "
						 "(metric_name, bucket_start, bucket_end, count) "
						 "SELECT 'latency_ms', "
						 "       floor(latency_ms / 10) * 10, "
						 "       floor(latency_ms / 10) * 10 + 10, "
						 "       COUNT(*) "
						 "FROM neurondb.query_metrics "
						 "WHERE query_timestamp > now() - interval '5 minutes' "
						 "GROUP BY floor(latency_ms / 10)");

		ret = ndb_spi_execute(session, sql.data, false, 0);
		if (ret != SPI_OK_INSERT)
		{
			elog(WARNING,
				 "neurondb: histogram INSERT returned unexpected code %d",
				 ret);
		}
		NDB_FREE(sql.data);
	}
	PG_CATCH();
	{
		ndb_spi_session_end(&session);
		EmitErrorReport();
		FlushErrorState();
		elog(LOG, "neurondb: metrics recording failed, continuing");
	}
	PG_END_TRY();

	ndb_spi_session_end(&session);
}

/*
 * Export Prometheus metrics
 */
static void
export_prometheus_metrics(void)
{
	StringInfoData sql;
	StringInfoData metrics;
	int			ret;
	NDB_DECLARE(NdbSpiSession *, session);

	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		return;

	PG_TRY();
	{
		int64		queries_sampled;
		int64		adjustments_made;
		double		avg_latency_ms;
		double		avg_recall;
		int			current_ef_search;

		/* Check if table exists first */
		ret = ndb_spi_execute(session,
								   "SELECT 1 FROM pg_tables WHERE schemaname = 'neurondb' "
								   "AND tablename = 'neurondb_prometheus_metrics'",
								   true,
								   0);
		NDB_CHECK_SPI_TUPTABLE();
		if (ret != SPI_OK_SELECT || SPI_processed == 0)
		{
			/* Table doesn't exist, skip metrics export */
			ndb_spi_session_end(&session);
			return;
		}

		/* Defensive: check shared state before accessing */
		if (neuranmon_state == NULL || neuranmon_state->lock == NULL)
		{
			ndb_spi_session_end(&session);
			return;
		}

		initStringInfo(&sql);
		initStringInfo(&metrics);

		/* Read metrics with lock protection */

		LWLockAcquire(neuranmon_state->lock, LW_SHARED);
		queries_sampled = neuranmon_state->queries_sampled;
		adjustments_made = neuranmon_state->adjustments_made;
		avg_latency_ms = neuranmon_state->avg_latency_ms;
		avg_recall = neuranmon_state->avg_recall;
		current_ef_search = neuranmon_state->current_ef_search;
		LWLockRelease(neuranmon_state->lock);

		/* Export key metrics to Prometheus metrics table */
		appendStringInfo(&sql,
						 "INSERT INTO neurondb.prometheus_metrics "
						 "(metric_name, metric_value) "
						 "VALUES "
						 "  ('neurondb_queries_sampled_total', " NDB_INT64_FMT
						 "), "
						 "  ('neurondb_adjustments_made_total', " NDB_INT64_FMT
						 "), "
						 "  ('neurondb_avg_latency_ms', %.2f), "
						 "  ('neurondb_avg_recall', %.3f), "
						 "  ('neurondb_current_ef_search', %d) "
						 "ON CONFLICT (metric_name) DO UPDATE SET "
						 "  metric_value = EXCLUDED.metric_value, "
						 "  last_updated = now()",
						 NDB_INT64_CAST(queries_sampled),
						 NDB_INT64_CAST(adjustments_made),
						 avg_latency_ms,
						 avg_recall,
						 current_ef_search);

		ret = ndb_spi_execute(session, sql.data, false, 0);
		if (ret != SPI_OK_INSERT)
		{
			elog(WARNING,
				 "neurondb: Prometheus metrics INSERT returned unexpected code %d",
				 ret);
		}
		NDB_FREE(sql.data);
		NDB_FREE(metrics.data);
	}
	PG_CATCH();
	{
		ndb_spi_session_end(&session);
		EmitErrorReport();
		FlushErrorState();
		elog(LOG,
			 "neurondb: Prometheus metrics export failed, "
			 "continuing");
	}
	PG_END_TRY();

	ndb_spi_session_end(&session);
}

/*
 * Manual execution function for operators
 */
PG_FUNCTION_INFO_V1(neuranmon_sample);
Datum
neuranmon_sample(PG_FUNCTION_ARGS)
{
	(void) fcinfo;				/* Unused */

	/* Function is called from user session, already in transaction */
	sample_and_tune();

	PG_RETURN_BOOL(true);
}
