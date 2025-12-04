/*-------------------------------------------------------------------------
 *
 * bgworker_init.c
 *		Background worker initialization and registration
 *
 * This module handles registration of all NeurondB background workers
 * via shared_preload_libraries mechanism.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/bgworker_init.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "postmaster/bgworker.h"
#include "storage/ipc.h"
#include "storage/lwlock.h"
#include "storage/shmem.h"
#include "utils/guc.h"
#include "access/reloptions.h"
#include "libpq/pqsignal.h"
#include <signal.h>

/* Include ONNX Runtime header */
#include "neurondb_onnx.h"
#include "neurondb_gpu_backend.h"
#include "ml_gpu_registry.h"
#include "neurondb_config.h"
#include "neurondb_automl.h"
#include "neurondb_index.h"
#include "neurondb_guc.h"

/* Reloption kinds for HNSW and IVF indexes */
int			relopt_kind_hnsw;
int			relopt_kind_ivf;

/* Forward declarations from background worker modules */
extern void neuranq_main(Datum main_arg);
extern Size neuranq_shmem_size(void);
extern void neuranq_shmem_init(void);

extern void neuranmon_main(Datum main_arg);
extern Size neuranmon_shmem_size(void);
extern void neuranmon_shmem_init(void);

extern void neurandefrag_main(Datum main_arg);
extern Size neurandefrag_shmem_size(void);
extern void neurandefrag_shmem_init(void);

/* GPU module declarations */
extern void neurondb_gpu_init(void);
extern void neurondb_gpu_register_models(void);
extern Size neurondb_llm_shmem_size(void);
extern void neurondb_llm_shmem_init(void);
extern void neuranllm_main(Datum main_arg);

/* Prometheus and cache declarations */
extern Size entrypoint_cache_shmem_size(void);
extern void entrypoint_cache_shmem_init(void);

/* Scan provider declarations */
extern void register_hybrid_scan_provider(void);

/* Module initialization */
void		neurondb_worker_fini(void);

/* Shared memory hooks */
#if PG_VERSION_NUM >= 150000
static shmem_request_hook_type prev_shmem_request_hook = NULL;
static void neurondb_shmem_request(void);
#endif
static shmem_startup_hook_type prev_shmem_startup_hook = NULL;
static void neurondb_shmem_startup(void);

/*
 * Module initialization - PostgreSQL extension entry point
 */
void
_PG_init(void)
{
	BackgroundWorker worker;

	if (!process_shared_preload_libraries_in_progress)
	{
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("neurondb must be loaded via shared_preload_libraries"),
				 errhint("Add \"neurondb\" to shared_preload_libraries in postgresql.conf, then restart the server.")));
		return;
	}

	elog(LOG, "neurondb: initializing background workers");

	/* Ignore SIGPIPE to prevent crashes when writing to broken pipes */
	/* This makes error reporting crash-proof when clients disconnect */
	pqsignal(SIGPIPE, SIG_IGN);

	/* Register reloption kinds for HNSW and IVF indexes */
	relopt_kind_hnsw = add_reloption_kind();
	relopt_kind_ivf = add_reloption_kind();
	elog(LOG, "neurondb: registered reloption kinds (HNSW=%d, IVF=%d)",
		 relopt_kind_hnsw, relopt_kind_ivf);

	/* Register HNSW index options */
	add_int_reloption(relopt_kind_hnsw, "m",
					  "Maximum number of connections per node",
					  16, 4, 200, AccessExclusiveLock);
	add_int_reloption(relopt_kind_hnsw, "ef_construction",
					  "Size of dynamic candidate list during construction",
					  200, 10, 1000, AccessExclusiveLock);
	add_int_reloption(relopt_kind_hnsw, "ef_search",
					  "Size of dynamic candidate list during search",
					  64, 10, 1000, AccessExclusiveLock);

	/* Register IVF index options */
	add_int_reloption(relopt_kind_ivf, "lists",
					  "Number of inverted lists",
					  100, 1, 10000, AccessExclusiveLock);

	/* Initialize all GUC variables (centralized) */
	neurondb_init_all_gucs();

	/* Initialize ONNX Runtime (after GUCs are set up) */
	neurondb_onnx_init();

	/* Register custom scan providers */
	register_hybrid_scan_provider();

	/* Install shared memory request hook */
	prev_shmem_request_hook = shmem_request_hook;
	shmem_request_hook = neurondb_shmem_request;

	/* Install shared memory startup hook */
	prev_shmem_startup_hook = shmem_startup_hook;
	shmem_startup_hook = neurondb_shmem_startup;

	/*
	 * Register background worker: neuranq (Queue Executor)
	 */
	memset(&worker, 0, sizeof(worker));
	worker.bgw_flags =
		BGWORKER_SHMEM_ACCESS | BGWORKER_BACKEND_DATABASE_CONNECTION;
	worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
	snprintf(worker.bgw_library_name, BGW_MAXLEN, "neurondb");
	snprintf(worker.bgw_function_name, BGW_MAXLEN, "neuranq_main");
	snprintf(worker.bgw_name, BGW_MAXLEN, "neurondb: queue worker");
	snprintf(worker.bgw_type, BGW_MAXLEN, "neurondb_queue");
	worker.bgw_restart_time = BGW_DEFAULT_RESTART_INTERVAL;
	worker.bgw_notify_pid = 0;
	worker.bgw_main_arg = (Datum) 0;

	RegisterBackgroundWorker(&worker);

	elog(LOG, "neurondb: registered neuranq background worker");

	/*
	 * Register background worker: neuranmon (Auto-Tuner)
	 */
	memset(&worker, 0, sizeof(worker));
	worker.bgw_flags =
		BGWORKER_SHMEM_ACCESS | BGWORKER_BACKEND_DATABASE_CONNECTION;
	worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
	snprintf(worker.bgw_library_name, BGW_MAXLEN, "neurondb");
	snprintf(worker.bgw_function_name, BGW_MAXLEN, "neuranmon_main");
	snprintf(worker.bgw_name, BGW_MAXLEN, "neurondb: tuner worker");
	snprintf(worker.bgw_type, BGW_MAXLEN, "neurondb_tuner");
	worker.bgw_restart_time = BGW_DEFAULT_RESTART_INTERVAL;
	worker.bgw_notify_pid = 0;
	worker.bgw_main_arg = (Datum) 0;

	RegisterBackgroundWorker(&worker);

	elog(LOG, "neurondb: registered neuranmon background worker");

	/*
	 * Register background worker: neurandefrag (Index Maintenance)
	 * Temporarily disabled due to assertion failure during extension loading
	 */

	/*
	 * memset(&worker, 0, sizeof(worker)); worker.bgw_flags =
	 * BGWORKER_SHMEM_ACCESS | BGWORKER_BACKEND_DATABASE_CONNECTION;
	 * worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
	 * snprintf(worker.bgw_library_name, BGW_MAXLEN, "neurondb");
	 * snprintf(worker.bgw_function_name, BGW_MAXLEN, "neurandefrag_main");
	 * snprintf(worker.bgw_name, BGW_MAXLEN, "neurondb: defrag worker");
	 * snprintf(worker.bgw_type, BGW_MAXLEN, "neurondb_defrag");
	 * worker.bgw_restart_time = BGW_DEFAULT_RESTART_INTERVAL;
	 * worker.bgw_notify_pid = 0; worker.bgw_main_arg = (Datum)0;
	 *
	 * RegisterBackgroundWorker(&worker);
	 *
	 * elog(LOG, "neurondb: registered neurandefrag background worker");
	 */

	/* Register background worker: neuranllm (LLM jobs) */
	memset(&worker, 0, sizeof(worker));
	worker.bgw_flags =
		BGWORKER_SHMEM_ACCESS | BGWORKER_BACKEND_DATABASE_CONNECTION;
	worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
	snprintf(worker.bgw_library_name, BGW_MAXLEN, "neurondb");
	snprintf(worker.bgw_function_name, BGW_MAXLEN, "neuranllm_main");
	snprintf(worker.bgw_name, BGW_MAXLEN, "neurondb: llm worker");
	snprintf(worker.bgw_type, BGW_MAXLEN, "neurondb_llm");
	worker.bgw_restart_time = BGW_DEFAULT_RESTART_INTERVAL;
	worker.bgw_notify_pid = 0;
	worker.bgw_main_arg = (Datum) 0;
	RegisterBackgroundWorker(&worker);
	elog(LOG, "neurondb: registered neuranllm background worker");
	elog(LOG, "neurondb: all background workers registered successfully");

#ifdef NDB_GPU_CUDA
	neurondb_gpu_register_cuda_backend();
#endif
#ifdef NDB_GPU_ROCM
	neurondb_gpu_register_rocm_backend();
#endif
#ifdef NDB_GPU_METAL
	neurondb_gpu_register_metal_backend();
#endif

	neurondb_gpu_register_models();
}

#ifndef NDB_GPU_CUDA
/* Intentional conditional compilation stub: CUDA backend not compiled in */
void
neurondb_gpu_register_cuda_backend(void)
{
	/* CUDA backend not available; noop stub to satisfy loader. */
	/* This is intentional - stub exists when CUDA is not compiled. */
}
#endif

#ifndef NDB_GPU_ROCM
/* Intentional conditional compilation stub: ROCm backend not compiled in */
void
neurondb_gpu_register_rocm_backend(void)
{
	/* ROCm backend not available; noop stub to satisfy loader. */
	/* This is intentional - stub exists when ROCm is not compiled. */
}
#endif

#ifndef NDB_GPU_METAL
/* Intentional conditional compilation stub: Metal backend not compiled in */
void
neurondb_gpu_register_metal_backend(void)
{
	/* Metal backend not available; noop stub to satisfy loader. */
	/* This is intentional - stub exists when Metal is not compiled. */
}
#endif

/*
 * Worker cleanup - called from _PG_fini in neurondb.c
 */
void
neurondb_worker_fini(void)
{
	/* Cleanup ONNX Runtime */
	neurondb_onnx_cleanup();

	/* Restore previous hooks */
#if PG_VERSION_NUM >= 150000
	shmem_request_hook = prev_shmem_request_hook;
#endif
	shmem_startup_hook = prev_shmem_startup_hook;

	elog(LOG, "neurondb: background workers shutting down");
}

#if PG_VERSION_NUM >= 150000
/*
 * Shared memory request hook (PG15+)
 */
static void
neurondb_shmem_request(void)
{
	/* Call previous hook if any */
	if (prev_shmem_request_hook)
		prev_shmem_request_hook();

	/* Request shared memory */
	RequestAddinShmemSpace(neuranq_shmem_size() + neuranmon_shmem_size()
						   + neurandefrag_shmem_size() + neurondb_llm_shmem_size()
						   + entrypoint_cache_shmem_size()
						   + 8192); /* prometheus metrics */

	/* Request LWLocks */
	RequestNamedLWLockTranche("neurondb_queue", 1);
	RequestNamedLWLockTranche("neurondb_tuner", 1);
	RequestNamedLWLockTranche("neurondb_defrag", 1);
	RequestNamedLWLockTranche("neurondb_llm", 1);
	RequestNamedLWLockTranche("neurondb_prometheus", 1);
	RequestNamedLWLockTranche("neurondb_entrypoint_cache", 1);
}
#endif

/*
 * Shared memory startup hook
 */
static void
neurondb_shmem_startup(void)
{
	/* Call previous hook if any */
	if (prev_shmem_startup_hook)
		prev_shmem_startup_hook();

	/* Initialize shared memory for all workers */
	neuranq_shmem_init();
	neuranmon_shmem_init();
	neurandefrag_shmem_init();
	neurondb_llm_shmem_init();
	entrypoint_cache_shmem_init();

	elog(LOG, "neurondb: shared memory initialized for all workers");
}
