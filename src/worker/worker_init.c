/*-------------------------------------------------------------------------
 *
 * bgworker_init.c
 *		Background worker initialization and registration
 *
 * This module handles registration of all NeurondB background workers
 * via shared_preload_libraries mechanism.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
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
#include "neurondb_replication.h"
#include <signal.h>

#include "neurondb_onnx.h"
#include "neurondb_gpu_backend.h"

#include "ml_gpu_registry.h"
#include "neurondb_config.h"
#include "neurondb_automl.h"
#include "neurondb_index.h"
#include "neurondb_guc.h"

int			relopt_kind_hnsw;
int			relopt_kind_ivf;
int			relopt_kind_pq;

extern void neuranq_main(Datum main_arg);
extern Size neuranq_shmem_size(void);
extern void neuranq_shmem_init(void);

extern void neuranmon_main(Datum main_arg);
extern Size neuranmon_shmem_size(void);
extern void neuranmon_shmem_init(void);

extern void neurandefrag_main(Datum main_arg);
extern Size neurandefrag_shmem_size(void);
extern void neurandefrag_shmem_init(void);

extern void neurondb_gpu_init(void);
extern void neurondb_gpu_register_models(void);
extern Size neurondb_llm_shmem_size(void);
extern void neurondb_llm_shmem_init(void);
extern void neuranllm_main(Datum main_arg);

extern Size entrypoint_cache_shmem_size(void);
extern void entrypoint_cache_shmem_init(void);

extern void register_hybrid_scan_provider(void);

void		neurondb_worker_fini(void);

#if PG_VERSION_NUM >= 150000
static shmem_request_hook_type prev_shmem_request_hook = NULL;
static void neurondb_shmem_request(void);
#endif
static shmem_startup_hook_type prev_shmem_startup_hook = NULL;
static void neurondb_shmem_startup(void);

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

	pqsignal(SIGPIPE, SIG_IGN);

	relopt_kind_hnsw = add_reloption_kind();
	relopt_kind_ivf = add_reloption_kind();
	relopt_kind_pq = add_reloption_kind();
	elog(LOG, "neurondb: registered reloption kinds (HNSW=%d, IVF=%d, PQ=%d)",
		 relopt_kind_hnsw, relopt_kind_ivf, relopt_kind_pq);

	/* Register all HNSW options - PostgreSQL requires registration for recognition */
	add_int_reloption(relopt_kind_hnsw, "m",
					  "Maximum number of connections per node",
					  16, 4, 200, AccessExclusiveLock);
	add_int_reloption(relopt_kind_hnsw, "ef_construction",
					  "Size of dynamic candidate list during construction",
					  200, 10, 1000, AccessExclusiveLock);
	add_int_reloption(relopt_kind_hnsw, "ef_search",
					  "Size of dynamic candidate list during search",
					  64, 10, 1000, AccessExclusiveLock);

	add_int_reloption(relopt_kind_ivf, "lists",
					  "Number of inverted lists",
					  100, 1, 10000, AccessExclusiveLock);
	add_int_reloption(relopt_kind_ivf, "probes",
					  "Number of lists to probe",
					  10, 1, 1000, AccessExclusiveLock);

	add_int_reloption(relopt_kind_pq, "m",
					  "Number of PQ subspaces",
					  8, 1, 32, AccessExclusiveLock);
	add_int_reloption(relopt_kind_pq, "ks",
					  "Codebook size per subspace",
					  256, 16, 65536, AccessExclusiveLock);
	add_int_reloption(relopt_kind_pq, "rerank_k",
					  "Number of candidates for reranking",
					  100, 1, 10000, AccessExclusiveLock);

	/* Register custom WAL resource manager for index replication */
	neurondb_replication_register_rmgr();

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

/* GPU backend registration stubs are now in src/gpu/gpu_stubs.c */
/* This ensures they're always available, even in CPU-only builds */

void
neurondb_worker_fini(void)
{
	neurondb_onnx_cleanup();

#if PG_VERSION_NUM >= 150000
	shmem_request_hook = prev_shmem_request_hook;
#endif
	shmem_startup_hook = prev_shmem_startup_hook;

	elog(LOG, "neurondb: background workers shutting down");
}

#if PG_VERSION_NUM >= 150000
static void
neurondb_shmem_request(void)
{
	if (prev_shmem_request_hook)
		prev_shmem_request_hook();

	RequestAddinShmemSpace(neuranq_shmem_size() + neuranmon_shmem_size()
						   + neurandefrag_shmem_size() + neurondb_llm_shmem_size()
						   + entrypoint_cache_shmem_size()
						   + 8192);

	RequestNamedLWLockTranche("neurondb_queue", 1);
	RequestNamedLWLockTranche("neurondb_tuner", 1);
	RequestNamedLWLockTranche("neurondb_defrag", 1);
	RequestNamedLWLockTranche("neurondb_llm", 1);
	RequestNamedLWLockTranche("neurondb_prometheus", 1);
	RequestNamedLWLockTranche("neurondb_entrypoint_cache", 1);
}
#endif

static void
neurondb_shmem_startup(void)
{
	if (prev_shmem_startup_hook)
		prev_shmem_startup_hook();

	neuranq_shmem_init();
	neuranmon_shmem_init();
	neurandefrag_shmem_init();
	neurondb_llm_shmem_init();
	entrypoint_cache_shmem_init();

	elog(LOG, "neurondb: shared memory initialized for all workers");
}
