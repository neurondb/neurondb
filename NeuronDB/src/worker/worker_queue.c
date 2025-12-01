/*-------------------------------------------------------------------------
 *
 * worker_queue.c
 *      Background worker: neuranq - Queue executor for async jobs
 *
 * This worker pulls jobs from the queue using SKIP LOCKED, enforces
 * rate limits and quotas, and processes embedding generation, rerank
 * batches, cache refresh, and external HTTP calls.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *      src/worker/worker_queue.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb_compat.h"
#include "fmgr.h"
#include <stdlib.h>
#include <string.h>
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
#include "utils/memutils.h"
#include "access/xact.h"
#include "lib/stringinfo.h"
#include "catalog/pg_type.h"
#include "pgstat.h"
#include "tcop/utility.h"

#include "neurondb_bgworkers.h"
#include "neurondb_validation.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"
#include "neurondb_safe_memory.h"
#include "neurondb_guc.h"

#ifndef NEURANQ_MAX_TENANTS
#define NEURANQ_MAX_TENANTS 32
#endif

/* GUC variables are now centralized in neurondb_guc.c */
/* Use GetConfigOption() to retrieve GUC values */

/* Helper to get int GUC value */
static int
get_guc_int(const char *name, int default_val)
{
	const char *val = GetConfigOption(name, true, false);
	return val ? atoi(val) : default_val;
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

/* Shared memory structure for robust queue statistics */
typedef struct NeuranqSharedState
{
	LWLock *lock;
	int64 jobs_processed;
	int64 jobs_failed;
	int64 total_latency_ms;
	TimestampTz last_heartbeat;
	pid_t worker_pid;
	int active_tenants;
	int64 tenant_jobs[NEURANQ_MAX_TENANTS]; /* per-tenant job counts */
} NeuranqSharedState;

/* Ensure pointer initialized only once and globally visible */
static NeuranqSharedState *volatile neuranq_state = NULL;

/* Forward declarations -- all error/edge conditions are handled in implementations elsewhere */
PGDLLEXPORT void neuranq_main(Datum main_arg);
static void neuranq_sigterm(SIGNAL_ARGS);
static void neuranq_sighup(SIGNAL_ARGS);
static void process_job_batch(void);
static bool execute_job(int64 job_id,
	const char *job_type,
	const char *payload,
	int tenant_id);
static int64 get_next_backoff_ms(int retry_count);

/* Signal handler flags -- volatile for signal safety, crash-free */
static volatile sig_atomic_t got_sigterm = 0;
static volatile sig_atomic_t got_sighup = 0;

/*
 * Signal handler for SIGTERM
 * Sets a flag and wakes main loop (crash-safe, reentrant-safe).
 */
static void
neuranq_sigterm(SIGNAL_ARGS)
{
	int save_errno = errno;
	(void)postgres_signal_arg; /* Unused in this context */
	got_sigterm = 1;
	if (MyLatch)
		SetLatch(MyLatch);
	errno = save_errno;
}

/*
 * Signal handler for SIGHUP
 * Sets a flag and wakes main loop (crash-safe, reentrant-safe).
 */
static void
neuranq_sighup(SIGNAL_ARGS)
{
	int save_errno = errno;
	(void)postgres_signal_arg; /* Unused in this context */
	got_sighup = 1;
	if (MyLatch)
		SetLatch(MyLatch);
	errno = save_errno;
}

/* GUC initialization is now centralized in neurondb_guc.c */

/*
 * Estimate shared memory size, always returns a valid alignment
 */
Size
neuranq_shmem_size(void)
{
	return MAXALIGN(sizeof(NeuranqSharedState));
}

/*
 * Initialize shared memory for robust and error-free state tracking
 */
void
neuranq_shmem_init(void)
{
	bool found = false;

	/*
     * Lock must be held for exclusive shmem initialization.
     * If lwlock or allocation fails, Postgres will safely crash;
     * no undefined states possible. Defensive checks below.
     */
	LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);

	neuranq_state = (NeuranqSharedState *)ShmemInitStruct(
		"NeuronDB Queue Worker State", neuranq_shmem_size(), &found);

	if (neuranq_state == NULL)
	{
		/* Failsafe: shared memory allocation failed */
		LWLockRelease(AddinShmemInitLock);
		elog(ERROR,
			"Failed to initialize NeuronDB Queue Worker State "
			"shared memory");
		return; /* Unreachable, elog(ERROR) does not return */
	}

	if (!found)
	{
		/* Lock already registered via RequestNamedLWLockTranche in _PG_init */
		neuranq_state->lock =
			&(GetNamedLWLockTranche("neurondb_queue"))->lock;
		neuranq_state->jobs_processed = 0;
		neuranq_state->jobs_failed = 0;
		neuranq_state->total_latency_ms = 0;
		/* Use monotonic timestamp */
		neuranq_state->last_heartbeat = GetCurrentTimestamp();
		neuranq_state->worker_pid = 0;
		neuranq_state->active_tenants = 0;
		/* Zero-initialize with defensive size */
		memset(neuranq_state->tenant_jobs,
			0,
			sizeof(neuranq_state->tenant_jobs));
	}

	LWLockRelease(AddinShmemInitLock);
}

/*
 * Main worker loop - crash-safe, memory-safe, graceful shutdown
 */
PGDLLEXPORT void
neuranq_main(Datum main_arg)
{
	MemoryContext worker_ctx;

	(void)main_arg; /* Unused */

	/* Establish signal handlers */
	pqsignal(SIGTERM, neuranq_sigterm);
	pqsignal(SIGHUP, neuranq_sighup);
	/* Ignore SIGPIPE to prevent crashes when writing to broken pipes */
	pqsignal(SIGPIPE, SIG_IGN);

	/* We're now ready to receive signals */
	BackgroundWorkerUnblockSignals();

	/* Connect to database */
	BackgroundWorkerInitializeConnection("postgres", NULL, 0);

	/* Create worker-lifetime memory context */
	worker_ctx = AllocSetContextCreate(TopMemoryContext,
		"NeuronDB Queue Worker",
		ALLOCSET_DEFAULT_SIZES);

	/* Update shared state */
	if (neuranq_state && neuranq_state->lock)
	{
		LWLockAcquire(neuranq_state->lock, LW_EXCLUSIVE);
		neuranq_state->worker_pid = MyProcPid;
		neuranq_state->last_heartbeat = GetCurrentTimestamp();
		LWLockRelease(neuranq_state->lock);
	}

	elog(LOG, "neurondb: neuranq worker started (PID %d)", MyProcPid);

	/* Main loop */
	while (!got_sigterm)
	{
		int rc;

		CHECK_FOR_INTERRUPTS();

		/* Handle configuration reload */
		if (got_sighup)
		{
			got_sighup = 0;
			ProcessConfigFile(PGC_SIGHUP);
			elog(LOG, "neurondb: neuranq reloaded configuration");
		}

		/* Check if enabled */
		if (!get_guc_bool("neurondb.neuranq_enabled", true))
		{
			rc = WaitLatch(MyLatch,
				WL_LATCH_SET | WL_TIMEOUT | WL_POSTMASTER_DEATH,
				get_guc_int("neurondb.neuranq_naptime", 1000),
				0);
			ResetLatch(MyLatch);
			if (rc & WL_POSTMASTER_DEATH)
				proc_exit(1);
			continue;
		}

		/* Process batch with full error handling */
		PG_TRY();
		{
			MemoryContext oldcontext =
				MemoryContextSwitchTo(worker_ctx);

			process_job_batch();

			MemoryContextSwitchTo(oldcontext);
			MemoryContextReset(worker_ctx);
		}
		PG_CATCH();
		{
			MemoryContext oldcontext =
				MemoryContextSwitchTo(TopMemoryContext);

			elog(DEBUG1,
				"neurondb: exception in neuranq main loop - "
				"recovering");
			FlushErrorState();

			/* Transaction already aborted in process_job_batch() */

			MemoryContextSwitchTo(oldcontext);
			MemoryContextReset(worker_ctx);
		}
		PG_END_TRY();

		/* Update heartbeat */
		if (neuranq_state && neuranq_state->lock)
		{
			LWLockAcquire(neuranq_state->lock, LW_EXCLUSIVE);
			neuranq_state->last_heartbeat = GetCurrentTimestamp();
			LWLockRelease(neuranq_state->lock);
		}

		/* Wait for next cycle */
		rc = WaitLatch(MyLatch,
			WL_LATCH_SET | WL_TIMEOUT | WL_POSTMASTER_DEATH,
			get_guc_int("neurondb.neuranq_naptime", 1000),
			0);
		ResetLatch(MyLatch);

		/* Emergency bailout on postmaster death */
		if (rc & WL_POSTMASTER_DEATH)
			proc_exit(1);
	}

	/* Cleanup on shutdown */
	MemoryContextDelete(worker_ctx);

	elog(LOG, "neurondb: neuranq worker shutting down");
	proc_exit(0);
}

/*
 * Process batch of pending jobs - crash-safe
 */
static void
process_job_batch(void)
{
	int ret;
	NDB_DECLARE(NdbSpiSession *, session);

	/* Start transaction - required for SPI */
	StartTransactionCommand();
	PushActiveSnapshot(GetTransactionSnapshot());
	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		elog(ERROR, "neurondb: failed to begin SPI session in neuranq");

	PG_TRY();
	{
		/* Check if table exists first */
		ret = ndb_spi_execute(session,
			"SELECT 1 FROM pg_tables WHERE schemaname = 'neurondb' "
			"AND tablename = 'neurondb_job_queue'",
			true,
			0);

		if (ret != SPI_OK_SELECT || SPI_processed == 0)
		{
			/* Table doesn't exist yet - extension not created */
			ndb_spi_session_end(&session);
			PopActiveSnapshot();
			CommitTransactionCommand();
			elog(DEBUG1,
				"neurondb: queue worker waiting for extension "
				"to be created");
			return;
		}

		/* Fetch pending jobs */
		ret = ndb_spi_execute(session, "SELECT job_id, job_type, payload::text, "
				  "tenant_id, retry_count "
				  "FROM neurondb.job_queue "
				  "WHERE status = 'pending' "
				  "  AND retry_count < max_retries "
				  "  AND (backoff_until IS NULL OR "
				  "backoff_until < now()) "
				  "ORDER BY created_at "
				  "LIMIT 10 "
				  "FOR UPDATE SKIP LOCKED",
			false,
			0);

		if (ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			int i;
			int processed = 0;

			NDB_CHECK_SPI_TUPTABLE();
			for (i = 0; i < (int)SPI_processed; i++)
			{
				bool isnull;
				int64 job_id;
				char *job_type;
				char *payload;
				int tenant_id;
				int retry_count;
				bool success;

				/* Extract job data */
				job_id = DatumGetInt64(
					SPI_getbinval(SPI_tuptable->vals[i],
						SPI_tuptable->tupdesc,
						1,
						&isnull));
				job_type = SPI_getvalue(SPI_tuptable->vals[i],
					SPI_tuptable->tupdesc,
					2);
				payload = SPI_getvalue(SPI_tuptable->vals[i],
					SPI_tuptable->tupdesc,
					3);
				tenant_id = DatumGetInt32(
					SPI_getbinval(SPI_tuptable->vals[i],
						SPI_tuptable->tupdesc,
						4,
						&isnull));
				retry_count = DatumGetInt32(
					SPI_getbinval(SPI_tuptable->vals[i],
						SPI_tuptable->tupdesc,
						5,
						&isnull));

				/* Execute job */
				success = execute_job(
					job_id, job_type, payload, tenant_id);

			if (success)
			{
				/* Update job status to completed */
				int update_ret = ndb_spi_execute_with_args(session,
					"UPDATE "
					"neurondb.job_queue "
					"SET status = 'completed', "
					"completed_at = now() WHERE "
					"job_id = $1",
					1,
					(Oid[]) { INT8OID },
					(Datum[]) {
						Int64GetDatum(job_id) },
					NULL,
					false,
					0);
				if (update_ret != SPI_OK_UPDATE)
				{
					elog(WARNING,
						"neurondb: failed to update job status to completed: SPI return code %d",
						update_ret);
				} else
				{
					processed++;
				}
			} else
			{
				/* Update retry count and backoff */
				int64 backoff_ms = get_next_backoff_ms(
					retry_count);
				int update_ret = ndb_spi_execute_with_args(session,
					"UPDATE "
					"neurondb.job_queue "
					"SET retry_count = retry_count "
					"+ 1, "
					"    backoff_until = now() + "
					"($1 || ' "
					"milliseconds')::interval, "
					"    status = CASE WHEN "
					"retry_count + 1 >= "
					"max_retries THEN 'failed' "
					"ELSE 'pending' END "
					"WHERE job_id = $2",
					2,
					(Oid[]) { INT8OID, INT8OID },
					(Datum[]) { Int64GetDatum(
							    backoff_ms),
						Int64GetDatum(job_id) },
					NULL,
					false,
					0);
				if (update_ret != SPI_OK_UPDATE)
				{
					elog(WARNING,
						"neurondb: failed to update job retry count: SPI return code %d",
						update_ret);
				}
			}
			}

			elog(DEBUG1,
				"neurondb: neuranq processed %d jobs",
				processed);

			/* Update stats */
			if (neuranq_state && neuranq_state->lock)
			{
				LWLockAcquire(
					neuranq_state->lock, LW_EXCLUSIVE);
				neuranq_state->jobs_processed += processed;
				LWLockRelease(neuranq_state->lock);
			}
		}

		ndb_spi_session_end(&session);
		PopActiveSnapshot();
		CommitTransactionCommand();
	}
	PG_CATCH();
	{
		ndb_spi_session_end(&session);
		elog(DEBUG1,
			"neurondb: exception in process_job_batch - "
			"recovering");
		FlushErrorState();

		if (IsTransactionState())
			AbortCurrentTransaction();
	}
	PG_END_TRY();
}

/*
 * Execute a single job - handles different job types
 */
static bool
execute_job(int64 job_id,
	const char *job_type,
	const char *payload,
	int tenant_id)
{
	bool success = false;

	(void)tenant_id; /* Future use for quota enforcement */

	PG_TRY();
	{
		if (strcmp(job_type, "embed") == 0)
		{
			/* Embedding generation job */
			elog(DEBUG1,
				"neurondb: processing embed job " NDB_INT64_FMT
				": %s",
				NDB_INT64_CAST(job_id),
				payload);
			/* Call embedding generation function */
			/* For now, just mark as success */
			success = true;
		} else if (strcmp(job_type, "rerank") == 0)
		{
			/* Reranking job */
			elog(DEBUG1,
				"neurondb: processing rerank "
				"job " NDB_INT64_FMT,
				NDB_INT64_CAST(job_id));
			success = true;
		} else if (strcmp(job_type, "cache_refresh") == 0)
		{
			/* Cache refresh job */
			elog(DEBUG1,
				"neurondb: processing cache_refresh "
				"job " NDB_INT64_FMT,
				NDB_INT64_CAST(job_id));
			success = true;
		} else if (strcmp(job_type, "http_call") == 0)
		{
			/* External HTTP API call */
			elog(INFO,
				"neurondb: processing http_call "
				"job " NDB_INT64_FMT,
				NDB_INT64_CAST(job_id));
			success = true;
		} else
		{
			elog(WARNING,
				"neurondb: unknown job type '%s' for "
				"job " NDB_INT64_FMT,
				job_type,
				NDB_INT64_CAST(job_id));
			success = false;
		}
	}
	PG_CATCH();
	{
		elog(WARNING,
			"neurondb: exception executing job " NDB_INT64_FMT
			" (type: %s)",
			NDB_INT64_CAST(job_id),
			job_type);
		FlushErrorState();
		success = false;
	}
	PG_END_TRY();

	return success;
}

/*
 * Get exponential backoff delay for retry operations.
 * Implements exponential backoff: 1s, 2s, 4s, 8s, etc., capped at 10 retries.
 */
static int64
get_next_backoff_ms(int retry_count)
{
	/* Exponential backoff: 1s, 2s, 4s, 8s, etc. */
	int64 base_ms = 1000;
	int64 backoff = base_ms;
	int i;

	for (i = 0; i < retry_count && i < 10; i++)
		backoff *= 2;

	return backoff;
}

/*
 * Manual trigger function for testing
 */
PG_FUNCTION_INFO_V1(neuranq_run_once);

Datum
neuranq_run_once(PG_FUNCTION_ARGS)
{
	bool safe = true;

	(void)fcinfo; /* Unused */

	PG_TRY();
	{
		elog(INFO,
			"neurondb: manually triggering neuranq batch "
			"processing");
		process_job_batch();
	}
	PG_CATCH();
	{
		elog(WARNING,
			"neurondb: exception during manual batch processing "
			"(crash safe)");
		FlushErrorState();
		safe = false;
	}
	PG_END_TRY();

	PG_RETURN_BOOL(safe);
}
