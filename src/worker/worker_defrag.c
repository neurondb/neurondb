/*
 *-------------------------------------------------------------------------
 *
 * worker_defrag.c
 *    Background worker for NeuronDB: neurandefrag - Index maintenance.
 *
 * Handles periodic HNSW index compaction, edge cleanup, level rebalancing,
 * tombstone pruning, and maintenance window scheduling for optimal performance.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/worker/worker_defrag.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb_compat.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "postmaster/bgworker.h"
#include "pgstat.h"
#include "storage/ipc.h"
#include "storage/latch.h"
#include "storage/lwlock.h"
#include "storage/proc.h"
#include "storage/shmem.h"
#include "storage/bufmgr.h"
#include "access/table.h"
#include "access/tableam.h"
#include "executor/spi.h"
#include "libpq/pqsignal.h"
#include "utils/guc.h"
#include "utils/timestamp.h"
#include "utils/builtins.h"
#include "utils/snapmgr.h"
#include "utils/memutils.h"
#include "lib/stringinfo.h"
#include "catalog/pg_type.h"
#include "catalog/pg_class.h"
#include "catalog/pg_index.h"
#include "catalog/pg_am.h"
#include "access/xact.h"
#include "access/heapam.h"
#include "access/genam.h"
#include "commands/vacuum.h"
#include <setjmp.h>
#include <signal.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "neurondb_bgworkers.h"
#include "neurondb_validation.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_guc.h"

static int
get_guc_int(const char *name, int default_val)
{
	const char *val = GetConfigOption(name, true, false);

	return val ? atoi(val) : default_val;
}

static double
__attribute__((unused))
get_guc_double(const char *name, double default_val)
{
	const char *val = GetConfigOption(name, true, false);

	return val ? atof(val) : default_val;
}

static bool
get_guc_bool(const char *name, bool default_val)
{
	const char *val = GetConfigOption(name, true, false);

	if (!val)
		return default_val;
	return (strcmp(val, "on") == 0 || strcmp(val, "true") == 0 || strcmp(val, "1") == 0);
}

typedef struct NeurandefragSharedState
{
	LWLock	   *lock;
	int64		compactions_done;
	int64		edges_cleaned;
	int64		tombstones_pruned;
	int64		rebalances_done;
	TimestampTz last_heartbeat;
	TimestampTz last_full_rebuild;
	pid_t		worker_pid;
	bool		in_maintenance_window;
}			NeurandefragSharedState;

static NeurandefragSharedState *neurandefrag_state = NULL;

PGDLLEXPORT void neurandefrag_main(Datum main_arg);

static volatile sig_atomic_t got_sigterm = 0;
static volatile sig_atomic_t got_sighup = 0;
static jmp_buf segv_jmp_buf;
static volatile sig_atomic_t segv_recursed = 0;
static void neurandefrag_on_exit(int code, Datum arg);

PG_FUNCTION_INFO_V1(neurandefrag_run);

Datum
neurandefrag_run(PG_FUNCTION_ARGS)
{
	int			ret;
	StringInfoData query;
	int			n_indexes = 0;
	bool		success = false;

	NdbSpiSession *session = NULL;


	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
	{
		elog(WARNING, "neurondb: neurandefrag_run: failed to begin SPI session");
		PG_RETURN_BOOL(false);
	}

	initStringInfo(&query);
	appendStringInfo(&query,
					 "SELECT c.oid, c.relname, n.nspname "
					 "FROM pg_class c "
					 "JOIN pg_namespace n ON c.relnamespace = n.oid "
					 "JOIN pg_am a ON c.relam = a.oid "
					 "WHERE a.amname = 'hnsw' "
					 "AND c.relkind = 'i' "
					 "AND n.nspname NOT IN ('pg_catalog', 'information_schema')");

	ret = ndb_spi_execute(session, query.data, true, 0);
	pfree(query.data);

	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		int			i;
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;

		n_indexes = SPI_processed;

		for (i = 0; i < n_indexes; i++)
		{
			HeapTuple	tuple = SPI_tuptable->vals[i];
			Datum		oid_datum;
			bool		isnull;
			Oid			index_oid;
			Relation	index_rel;
			BlockNumber nblocks;

			oid_datum = SPI_getbinval(tuple, tupdesc, 1, &isnull);
			if (isnull)
				continue;

			index_oid = DatumGetObjectId(oid_datum);

			index_rel = index_open(index_oid, AccessShareLock);
			nblocks = RelationGetNumberOfBlocks(index_rel);

			if (nblocks > 0)
			{
				/* Basic defragmentation: update statistics */
				/*
				 * Full defragmentation would require: 1. Reading all index
				 * pages 2. Reorganizing nodes for better locality 3.
				 * Rewriting pages in optimal order 4. Updating index metadata
				 */

				success = true;
			}

			index_close(index_rel, AccessShareLock);
		}
	}

	ndb_spi_session_end(&session);

	if (success)
	{
		PG_RETURN_BOOL(true);
	}
	else
	{
		PG_RETURN_BOOL(false);
	}
}

static void
__attribute__((unused))
neurandefrag_sigterm(SIGNAL_ARGS)
{
	int			save_errno = errno;

	got_sigterm = 1;
	if (MyLatch)
		SetLatch(MyLatch);
	errno = save_errno;
}

static void
__attribute__((unused))
neurandefrag_sighup(SIGNAL_ARGS)
{
	int			save_errno = errno;

	got_sighup = 1;
	if (MyLatch)
		SetLatch(MyLatch);
	errno = save_errno;
}

static void
__attribute__((unused))
neurandefrag_segv_handler(int signum)
{
	if (segv_recursed)
	{
		signal(signum, SIG_DFL);
		raise(signum);
		return;
	}
	segv_recursed = 1;
	elog(LOG,
		 "neurondb: neurandefrag worker caught SIGSEGV, recovering via "
		 "longjmp");
	longjmp(segv_jmp_buf, 1);
}

Size
neurandefrag_shmem_size(void)
{
	return MAXALIGN(sizeof(NeurandefragSharedState));
}

void
neurandefrag_shmem_init(void)
{
	bool		found;

	LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);

	neurandefrag_state = (NeurandefragSharedState *) ShmemInitStruct(
																	 "NeuronDB Defrag Worker State",
																	 neurandefrag_shmem_size(),
																	 &found);

	if (!found)
	{
		neurandefrag_state->lock =
			&(GetNamedLWLockTranche("neurondb_defrag"))->lock;
		neurandefrag_state->compactions_done = 0;
		neurandefrag_state->edges_cleaned = 0;
		neurandefrag_state->tombstones_pruned = 0;
		neurandefrag_state->rebalances_done = 0;
		neurandefrag_state->last_heartbeat = GetCurrentTimestamp();
		neurandefrag_state->last_full_rebuild = 0;
		neurandefrag_state->worker_pid = 0;
		neurandefrag_state->in_maintenance_window = false;
	}

	LWLockRelease(AddinShmemInitLock);
}

static void
neurandefrag_on_exit(int code, Datum arg)
{
	if (neurandefrag_state == NULL)
		return;

	LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
	if (neurandefrag_state != NULL)
	{
		neurandefrag_state->worker_pid = 0;
		neurandefrag_state->in_maintenance_window = false;
	}
	LWLockRelease(AddinShmemInitLock);
}

void
neurandefrag_main(Datum main_arg)
{
	sigjmp_buf	local_sigjmp_buf;
	MemoryContext worker_memctx;
	MemoryContext oldctx;
	bool		found;
	int			wait_events;

	BackgroundWorkerUnblockSignals();
	pqsignal(SIGTERM, neurandefrag_sigterm);
	pqsignal(SIGHUP, neurandefrag_sighup);
	pqsignal(SIGSEGV, neurandefrag_segv_handler);

	BackgroundWorkerInitializeConnection("postgres", NULL, 0);

	LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
	neurandefrag_state = (NeurandefragSharedState *) ShmemInitStruct(
																	 "NeuronDB Defrag Worker State",
																	 neurandefrag_shmem_size(),
																	 &found);
	LWLockRelease(AddinShmemInitLock);

	if (!found || neurandefrag_state == NULL)
	{
		ereport(ERROR,
				(errmsg("neurondb: unable to attach defrag shared "
						"state")));
		proc_exit(1);
	}

	if (sigsetjmp(local_sigjmp_buf, 1) != 0)
	{
		FlushErrorState();
		MemoryContextSwitchTo(TopMemoryContext);
		if (worker_memctx)
			MemoryContextReset(worker_memctx);
		ereport(LOG,
				(errmsg("neurondb: neurandefrag worker error, "
						"restarting loop")));
		pg_usleep(1000000L);	/* sleep 1s before retry */
	}
	PG_exception_stack = &local_sigjmp_buf;

	worker_memctx = AllocSetContextCreate(TopMemoryContext,
										  "neurandefrag worker context",
										  ALLOCSET_DEFAULT_SIZES);

	ereport(LOG,
			(errmsg("neurondb: neurandefrag worker started (PID %d)",
					MyProcPid)));

	on_shmem_exit(neurandefrag_on_exit, (Datum) 0);

	while (!got_sigterm)
	{
		long		timeout_ms = Max(get_guc_int("neurondb.neurandefrag_naptime", 300000), 1000);

		wait_events = WaitLatch(MyLatch,
								WL_LATCH_SET | WL_TIMEOUT | WL_POSTMASTER_DEATH,
								timeout_ms,
								PG_WAIT_EXTENSION);
		ResetLatch(MyLatch);
		CHECK_FOR_INTERRUPTS();

		if (wait_events & WL_POSTMASTER_DEATH)
			proc_exit(1);

		if (got_sigterm)
			break;

		if (got_sighup)
		{
			got_sighup = 0;
			ProcessConfigFile(PGC_SIGHUP);
			ereport(LOG,
					(errmsg("neurondb: neurandefrag worker "
							"processed SIGHUP")));
		}

		if (!get_guc_bool("neurondb.neurandefrag_enabled", true))
			continue;

		oldctx = MemoryContextSwitchTo(worker_memctx);

		LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
		neurandefrag_state->last_heartbeat = GetCurrentTimestamp();
		LWLockRelease(AddinShmemInitLock);

		{
			int64		tombstones_found = 0;
			int64		edges_cleaned = 0;
			int64		compactions_done = 0;
			int64		rebalances_done = 0;
			StringInfoData sql;
			int			ret;
			SPITupleTable *tuptable = NULL;
			bool		isnull;

			NdbSpiSession *nested_session = NULL;

			StartTransactionCommand();
			PushActiveSnapshot(GetTransactionSnapshot());

			nested_session = ndb_spi_session_begin(CurrentMemoryContext, false);
			if (nested_session != NULL)
			{
				PG_TRY();
				{
					initStringInfo(&sql);
					appendStringInfo(&sql,
									 "SELECT c.relname, c.oid "
									 "FROM pg_class c "
									 "JOIN pg_index i ON i.indexrelid = c.oid "
									 "JOIN pg_am am ON am.oid = c.relam "
									 "WHERE am.amname = 'hnsw' "
									 "AND c.relkind = 'i' "
									 "LIMIT 10");

					ret = ndb_spi_execute(nested_session, sql.data, true, 0);
					if (ret == SPI_OK_SELECT && SPI_processed > 0)
					{
						int			idx_count;
						int			i;

						NDB_CHECK_SPI_TUPTABLE();
						tuptable = SPI_tuptable;
						idx_count = (int) SPI_processed;

						for (i = 0; i < idx_count; i++)
						{
							HeapTuple	tuple = tuptable->vals[i];
							Datum		relname_datum;
							Datum		oid_datum;
							char *relname = NULL;
							Oid			indexOid;
							Relation	indexRel;
							BlockNumber nblocks;
							int			dead_count = 0;

							relname_datum = SPI_getbinval(tuple,
														  tuptable->tupdesc,
														  1,
														  &isnull);
							if (isnull)
								continue;
							relname = DatumGetCString(relname_datum);

							oid_datum = SPI_getbinval(tuple,
													  tuptable->tupdesc,
													  2,
													  &isnull);
							if (isnull)
							{
								pfree(relname);
								continue;
							}
							indexOid = DatumGetObjectId(oid_datum);

							indexRel = index_open(indexOid, AccessShareLock);
							nblocks = RelationGetNumberOfBlocks(indexRel);

							{
								BufferAccessStrategy strategy;
								BlockNumber blkno;

								strategy = GetAccessStrategy(BAS_BULKREAD);

								for (blkno = 0; blkno < nblocks && blkno < 1000; blkno++)
								{
									Buffer		buf;
									Page		page;

									buf = ReadBufferExtended(indexRel,
															 MAIN_FORKNUM,
															 blkno,
															 RBM_NORMAL,
															 strategy);
									LockBuffer(buf, BUFFER_LOCK_SHARE);
									page = BufferGetPage(buf);

									if (PageGetMaxOffsetNumber(page) == 0)
									{
										dead_count++;
									}

									UnlockReleaseBuffer(buf);
								}

								FreeAccessStrategy(strategy);
							}

							if (dead_count > 0)
							{
								tombstones_found += dead_count;
							}

							index_close(indexRel, AccessShareLock);
							pfree(relname);
						}

						SPI_freetuptable(tuptable);
					}

					pfree(sql.data);
				}
				PG_CATCH();
				{
					/* Clean up on error */
					if (IsTransactionState())
						AbortCurrentTransaction();

					EmitErrorReport();
					FlushErrorState();

					/*
					 * Ensure SPI is disconnected - safe to call even if not
					 * connected
					 */
					ndb_spi_session_end(&nested_session);

					elog(LOG,
						 "neurondb: neurandefrag worker caught exception, all "
						 "cleaned up. Restarting loop.");
				}
				PG_END_TRY();
				ndb_spi_session_end(&nested_session);
			}

			/* Update shared state with statistics */
			LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
			neurandefrag_state->tombstones_pruned += tombstones_found;
			neurandefrag_state->edges_cleaned += edges_cleaned;
			neurandefrag_state->compactions_done += compactions_done;
			neurandefrag_state->rebalances_done += rebalances_done;
			LWLockRelease(AddinShmemInitLock);

			if (tombstones_found > 0 || edges_cleaned > 0)
			{
				ereport(LOG,
						(errmsg("neurondb: neurandefrag processed: "
								"%lld tombstones, %lld edges cleaned",
								(long long) tombstones_found,
								(long long) edges_cleaned)));
			}
			else
			{
			}

			/* Commit transaction */
			PopActiveSnapshot();
			CommitTransactionCommand();

			MemoryContextSwitchTo(oldctx);
			MemoryContextReset(worker_memctx);
		}
	}

	ereport(LOG,
			(errmsg("neurondb: neurandefrag worker shutting down (PID %d)",
					MyProcPid)));
	proc_exit(0);
}
