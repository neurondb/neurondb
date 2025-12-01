/*-------------------------------------------------------------------------
 * neurondb_guc.c
 *   Centralized GUC (Grand Unified Configuration) handling for NeuronDB
 *
 * This module consolidates all GUC variable definitions and provides
 * a unified NeuronDBConfig structure for accessing configuration values.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *-------------------------------------------------------------------------*/

#include "postgres.h"
#include "fmgr.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include <limits.h>
#include "neurondb_guc.h"
#include "neurondb_macros.h"
#include "neurondb_constants.h"

/* ========================================================================
 * GUC Variable Definitions
 * ======================================================================== */

/* Core/index GUC variables */
int			neurondb_hnsw_ef_search = 64;
int			neurondb_ivf_probes = 10;
int			neurondb_ef_construction = 200;

/* GPU GUC variables */
int			neurondb_compute_mode = 2;  /* Default: AUTO (0=CPU, 1=GPU, 2=AUTO) */
int			neurondb_gpu_backend_type = 0;  /* Default: CUDA (0=CUDA, 1=ROCm, 2=Metal) */
int			neurondb_gpu_device = 0;
int			neurondb_gpu_batch_size = 8192;
int			neurondb_gpu_streams = 2;
double		neurondb_gpu_memory_pool_mb = 512.0;
char	   *neurondb_gpu_kernels = NULL;  /* Will be set by GUC system */
int			neurondb_gpu_timeout_ms = 30000;

/* LLM GUC variables */
char	   *neurondb_llm_provider = NULL;
char	   *neurondb_llm_model = NULL;
char	   *neurondb_llm_endpoint = NULL;
char	   *neurondb_llm_api_key = NULL;
int			neurondb_llm_timeout_ms = 30000;
int			neurondb_llm_cache_ttl = 600;
int			neurondb_llm_rate_limiter_qps = 5;
bool		neurondb_llm_fail_open = true;

/* Worker: neuranq GUC variables */
static int	neuranq_naptime = 1000;
static int	neuranq_queue_depth = 10000;
static int	neuranq_batch_size = 100;
static int	neuranq_timeout = 30000;
static int	neuranq_max_retries = 3;
static bool neuranq_enabled = true;

/* Worker: neuranmon GUC variables */
static int	neuranmon_naptime = 60000;
static int	neuranmon_sample_size = 1000;
static double neuranmon_target_latency = 100.0;
static double neuranmon_target_recall = 0.95;
static bool neuranmon_enabled = true;

/* Worker: neurandefrag GUC variables */
static int	neurandefrag_naptime = 300000;
static int	neurandefrag_compact_threshold = 10000;
static double neurandefrag_fragmentation_threshold = 0.3;
static char *neurandefrag_maintenance_window = "02:00-04:00";
static bool neurandefrag_enabled = true;

/* ONNX Runtime GUC variables */
char	   *neurondb_onnx_model_path = NULL;
bool		neurondb_onnx_use_gpu = true;
int			neurondb_onnx_threads = 4;
int			neurondb_onnx_cache_size = 10;

/* Quota GUC variables */
static int64 default_max_vectors = 1000000;
static int64 default_max_storage_mb = 10240;
static int default_max_qps = 1000;
static bool enforce_quotas = true;

/* AutoML GUC variables */
bool		neurondb_automl_use_gpu = false;

/* ========================================================================
 * Config Structure Instance
 * ======================================================================== */

/* Global config structure - allocated in TopMemoryContext */
NeuronDBConfig *neurondb_config = NULL;

/* ========================================================================
 * GUC Hook Functions
 * ======================================================================== */

/*
 * Validation hook for neurondb.gpu_backend_type
 * Only valid when compute_mode is GPU or AUTO
 */
static bool
neurondb_check_gpu_backend_type(int *newval, void **extra, GucSource source)
{
	if (neurondb_compute_mode == NDB_COMPUTE_MODE_CPU)
	{
		elog(WARNING,
			 "neurondb.gpu_backend_type is ignored when neurondb.compute_mode is 'cpu'");
		/* Accept but warn - the value will be stored but not used */
		return true;
	}
	/* Always valid when compute_mode is GPU or AUTO */
	return true;
}


/*
 * Sync function to update config structure from GUC variables.
 * Called after GUC changes and during initialization.
 */
void
neurondb_sync_config_from_gucs(void)
{
	if (neurondb_config == NULL)
		return;

	/* Core settings */
	neurondb_config->core.hnsw_ef_search = neurondb_hnsw_ef_search;
	neurondb_config->core.ivf_probes = neurondb_ivf_probes;
	neurondb_config->core.ef_construction = neurondb_ef_construction;

	/* GPU settings */
	neurondb_config->gpu.compute_mode = neurondb_compute_mode;
	neurondb_config->gpu.backend_type = neurondb_gpu_backend_type;
	neurondb_config->gpu.device = neurondb_gpu_device;
	neurondb_config->gpu.batch_size = neurondb_gpu_batch_size;
	neurondb_config->gpu.streams = neurondb_gpu_streams;
	neurondb_config->gpu.memory_pool_mb = neurondb_gpu_memory_pool_mb;
	neurondb_config->gpu.kernels = neurondb_gpu_kernels;
	neurondb_config->gpu.timeout_ms = neurondb_gpu_timeout_ms;

	/* LLM settings */
	neurondb_config->llm.provider = neurondb_llm_provider;
	neurondb_config->llm.model = neurondb_llm_model;
	neurondb_config->llm.endpoint = neurondb_llm_endpoint;
	neurondb_config->llm.api_key = neurondb_llm_api_key;
	neurondb_config->llm.timeout_ms = neurondb_llm_timeout_ms;
	neurondb_config->llm.cache_ttl = neurondb_llm_cache_ttl;
	neurondb_config->llm.rate_limiter_qps = neurondb_llm_rate_limiter_qps;
	neurondb_config->llm.fail_open = neurondb_llm_fail_open;

	/* Worker: neuranq settings */
	neurondb_config->neuranq.naptime = neuranq_naptime;
	neurondb_config->neuranq.queue_depth = neuranq_queue_depth;
	neurondb_config->neuranq.batch_size = neuranq_batch_size;
	neurondb_config->neuranq.timeout = neuranq_timeout;
	neurondb_config->neuranq.max_retries = neuranq_max_retries;
	neurondb_config->neuranq.enabled = neuranq_enabled;

	/* Worker: neuranmon settings */
	neurondb_config->neuranmon.naptime = neuranmon_naptime;
	neurondb_config->neuranmon.sample_size = neuranmon_sample_size;
	neurondb_config->neuranmon.target_latency = neuranmon_target_latency;
	neurondb_config->neuranmon.target_recall = neuranmon_target_recall;
	neurondb_config->neuranmon.enabled = neuranmon_enabled;

	/* Worker: neurandefrag settings */
	neurondb_config->neurandefrag.naptime = neurandefrag_naptime;
	neurondb_config->neurandefrag.compact_threshold = neurandefrag_compact_threshold;
	neurondb_config->neurandefrag.fragmentation_threshold = neurandefrag_fragmentation_threshold;
	neurondb_config->neurandefrag.maintenance_window = neurandefrag_maintenance_window;
	neurondb_config->neurandefrag.enabled = neurandefrag_enabled;

	/* ONNX settings */
	neurondb_config->onnx.model_path = neurondb_onnx_model_path;
	neurondb_config->onnx.use_gpu = neurondb_onnx_use_gpu;
	neurondb_config->onnx.threads = neurondb_onnx_threads;
	neurondb_config->onnx.cache_size = neurondb_onnx_cache_size;

	/* Quota settings */
	neurondb_config->quota.default_max_vectors = default_max_vectors;
	neurondb_config->quota.default_max_storage_mb = default_max_storage_mb;
	neurondb_config->quota.default_max_qps = default_max_qps;
	neurondb_config->quota.enforce_quotas = enforce_quotas;

	/* AutoML settings */
	neurondb_config->automl.use_gpu = neurondb_automl_use_gpu;
}

/* ========================================================================
 * GUC Initialization
 * ======================================================================== */

/*
 * Initialize all GUC variables and the config structure.
 * This function should be called from _PG_init() in worker_init.c
 */
void
neurondb_init_all_gucs(void)
{
	MemoryContext oldcontext;
	NDB_DECLARE(NeuronDBConfig *, config);

	/* Allocate config structure in TopMemoryContext */
	oldcontext = MemoryContextSwitchTo(TopMemoryContext);
	NDB_ALLOC(config, NeuronDBConfig, 1);
	neurondb_config = config;
	MemoryContextSwitchTo(oldcontext);

	/* ====================================================================
	 * Core/Index GUCs
	 * ==================================================================== */

	DefineCustomIntVariable("neurondb.hnsw_ef_search",
							"Sets the ef_search parameter for HNSW index scans",
							"Higher values improve recall but increase search time. Default is 64.",
							&neurondb_hnsw_ef_search,
							64,
							1,
							10000,
							PGC_USERSET,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomIntVariable("neurondb.ivf_probes",
							"Sets the number of probes for IVF index scans",
							"Higher values improve recall but increase search time. Default is 10.",
							&neurondb_ivf_probes,
							10,
							1,
							1000,
							PGC_USERSET,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomIntVariable("neurondb.ef_construction",
							"Sets the ef_construction parameter for HNSW index builds",
							"Higher values improve index quality but increase build time. Default is 200.",
							&neurondb_ef_construction,
							200,
							4,
							2000,
							PGC_USERSET,
							0,
							NULL,
							NULL,
							NULL);

	/* ====================================================================
	 * GPU GUCs
	 * ==================================================================== */

	/* Compute mode parameter (0=CPU, 1=GPU, 2=AUTO) */
	DefineCustomIntVariable("neurondb.compute_mode",
							"Compute execution mode",
							"Controls whether ML operations run on CPU, GPU, or auto-select. "
							"Values: 0 (cpu) - CPU only, don't initialize GPU; "
							"1 (gpu) - GPU required, error if unavailable; "
							"2 (auto) - Try GPU first, fallback to CPU. Default is 2 (auto).",
							&neurondb_compute_mode,
							2,  /* Default: AUTO */
							0,  /* Min: CPU */
							2,  /* Max: AUTO */
							PGC_USERSET,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomIntVariable("neurondb.gpu_device",
							"GPU device ID to use (0-based)",
							NULL,
							&neurondb_gpu_device,
							0,
							0,
							16,
							PGC_USERSET,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomIntVariable("neurondb.gpu_batch_size",
							"Batch size for GPU operations",
							NULL,
							&neurondb_gpu_batch_size,
							8192,
							64,
							65536,
							PGC_USERSET,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomIntVariable("neurondb.gpu_streams",
							"Number of CUDA/HIP streams for parallel operations",
							NULL,
							&neurondb_gpu_streams,
							2,
							1,
							8,
							PGC_USERSET,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomRealVariable("neurondb.gpu_memory_pool_mb",
							 "GPU memory pool size in MB",
							 NULL,
							 &neurondb_gpu_memory_pool_mb,
							 512.0,
							 64.0,
							 32768.0,
							 PGC_USERSET,
							 0,
							 NULL,
							 NULL,
							 NULL);

	/* GPU backend type enum parameter (0=CUDA, 1=ROCm, 2=Metal) */
	DefineCustomIntVariable("neurondb.gpu_backend_type",
							"GPU backend type",
							"Selects GPU backend implementation. Only valid when compute_mode is 'gpu' or 'auto'. "
							"Values: 0 (cuda) - NVIDIA CUDA; 1 (rocm) - AMD ROCm; 2 (metal) - Apple Metal. "
							"Default is 0 (cuda). Ignored when compute_mode is 'cpu'.",
							&neurondb_gpu_backend_type,
							0,  /* Default: CUDA */
							0,  /* Min: CUDA */
							2,  /* Max: Metal */
							PGC_USERSET,
							0,
							neurondb_check_gpu_backend_type,
							NULL,
							NULL);

	DefineCustomStringVariable("neurondb.gpu_kernels",
							   "List of GPU-accelerated kernels (comma-separated: "
							   "l2,cosine,ip)",
							   NULL,
							   &neurondb_gpu_kernels,
							   "l2,cosine,ip,rf_split,rf_predict",
							   PGC_USERSET,
							   0,
							   NULL,
							   NULL,
							   NULL);

	DefineCustomIntVariable("neurondb.gpu_timeout_ms",
							"GPU kernel execution timeout in milliseconds",
							NULL,
							&neurondb_gpu_timeout_ms,
							30000,
							1000,
							300000,
							PGC_USERSET,
							0,
							NULL,
							NULL,
							NULL);

	/* ====================================================================
	 * LLM GUCs
	 * ==================================================================== */

	DefineCustomStringVariable("neurondb.llm_provider",
							   "LLM provider",
							   NULL,
							   &neurondb_llm_provider,
							   "huggingface",
							   PGC_USERSET,
							   0,
							   NULL,
							   NULL,
							   NULL);

	DefineCustomStringVariable("neurondb.llm_model",
							   "Default LLM model id",
							   NULL,
							   &neurondb_llm_model,
							   "sentence-transformers/all-MiniLM-L6-v2",
							   PGC_USERSET,
							   0,
							   NULL,
							   NULL,
							   NULL);

	DefineCustomStringVariable("neurondb.llm_endpoint",
							   "LLM endpoint base URL",
							   NULL,
							   &neurondb_llm_endpoint,
							   "https://router.huggingface.co",
							   PGC_USERSET,
							   0,
							   NULL,
							   NULL,
							   NULL);

	DefineCustomStringVariable("neurondb.llm_api_key",
							   "LLM API key (set via ALTER SYSTEM or env)",
							   NULL,
							   &neurondb_llm_api_key,
							   "",
							   PGC_SUSET,
							   GUC_SUPERUSER_ONLY,
							   NULL,
							   NULL,
							   NULL);

	DefineCustomIntVariable("neurondb.llm_timeout_ms",
							"HTTP timeout (ms)",
							NULL,
							&neurondb_llm_timeout_ms,
							30000,
							1000,
							600000,
							PGC_USERSET,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomIntVariable("neurondb.llm_cache_ttl",
							"Cache TTL seconds",
							NULL,
							&neurondb_llm_cache_ttl,
							600,
							0,
							86400,
							PGC_USERSET,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomIntVariable("neurondb.llm_rate_limiter_qps",
							"Rate limiter QPS",
							NULL,
							&neurondb_llm_rate_limiter_qps,
							5,
							1,
							10000,
							PGC_USERSET,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomBoolVariable("neurondb.llm_fail_open",
							 "Fail open on provider errors",
							 NULL,
							 &neurondb_llm_fail_open,
							 true,
							 PGC_USERSET,
							 0,
							 NULL,
							 NULL,
							 NULL);

	/* ====================================================================
	 * Worker: neuranq GUCs
	 * ==================================================================== */

	DefineCustomIntVariable("neurondb.neuranq_naptime",
							"Duration between job processing cycles (ms)",
							NULL,
							&neuranq_naptime,
							1000,
							100,
							60000,
							PGC_SIGHUP,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomIntVariable("neurondb.neuranq_queue_depth",
							"Maximum job queue size",
							NULL,
							&neuranq_queue_depth,
							10000,
							100,
							1000000,
							PGC_SIGHUP,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomIntVariable("neurondb.neuranq_batch_size",
							"Jobs to process per cycle",
							NULL,
							&neuranq_batch_size,
							100,
							1,
							10000,
							PGC_SIGHUP,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomIntVariable("neurondb.neuranq_timeout",
							"Job execution timeout (ms)",
							NULL,
							&neuranq_timeout,
							30000,
							1000,
							300000,
							PGC_SIGHUP,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomIntVariable("neurondb.neuranq_max_retries",
							"Maximum retry attempts per job",
							NULL,
							&neuranq_max_retries,
							3,
							0,
							10,
							PGC_SIGHUP,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomBoolVariable("neurondb.neuranq_enabled",
							 "Enable queue worker",
							 NULL,
							 &neuranq_enabled,
							 true,
							 PGC_SIGHUP,
							 0,
							 NULL,
							 NULL,
							 NULL);

	/* ====================================================================
	 * Worker: neuranmon GUCs
	 * ==================================================================== */

	DefineCustomIntVariable("neurondb.neuranmon_naptime",
							"Duration between tuning cycles (ms)",
							NULL,
							&neuranmon_naptime,
							60000,
							10000,
							600000,
							PGC_SIGHUP,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomIntVariable("neurondb.neuranmon_sample_size",
							"Number of queries to sample",
							NULL,
							&neuranmon_sample_size,
							1000,
							100,
							100000,
							PGC_SIGHUP,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomRealVariable("neurondb.neuranmon_target_latency",
							 "Target query latency (ms)",
							 NULL,
							 &neuranmon_target_latency,
							 100.0,
							 1.0,
							 10000.0,
							 PGC_SIGHUP,
							 0,
							 NULL,
							 NULL,
							 NULL);

	DefineCustomRealVariable("neurondb.neuranmon_target_recall",
							 "Target recall@k threshold",
							 NULL,
							 &neuranmon_target_recall,
							 0.95,
							 0.5,
							 1.0,
							 PGC_SIGHUP,
							 0,
							 NULL,
							 NULL,
							 NULL);

	DefineCustomBoolVariable("neurondb.neuranmon_enabled",
							 "Enable tuner worker",
							 NULL,
							 &neuranmon_enabled,
							 true,
							 PGC_SIGHUP,
							 0,
							 NULL,
							 NULL,
							 NULL);

	/* ====================================================================
	 * Worker: neurandefrag GUCs
	 * ==================================================================== */

	DefineCustomIntVariable("neurondb.neurandefrag_naptime",
							"Duration between maintenance cycles (ms)",
							NULL,
							&neurandefrag_naptime,
							300000,
							60000,
							3600000,
							PGC_SIGHUP,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomIntVariable("neurondb.neurandefrag_compact_threshold",
							"Edge count threshold for compaction trigger",
							NULL,
							&neurandefrag_compact_threshold,
							10000,
							1000,
							1000000,
							PGC_SIGHUP,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomRealVariable("neurondb.neurandefrag_fragmentation_threshold",
							 "Fragmentation ratio necessary to trigger a full rebuild",
							 NULL,
							 &neurandefrag_fragmentation_threshold,
							 0.3,
							 0.1,
							 0.9,
							 PGC_SIGHUP,
							 0,
							 NULL,
							 NULL,
							 NULL);

	DefineCustomStringVariable("neurondb.neurandefrag_maintenance_window",
							   "Maintenance window in HH:MM-HH:MM format",
							   NULL,
							   &neurandefrag_maintenance_window,
							   "02:00-04:00",
							   PGC_SIGHUP,
							   0,
							   NULL,
							   NULL,
							   NULL);

	DefineCustomBoolVariable("neurondb.neurandefrag_enabled",
							 "Enable/disable the Neurandefrag background worker",
							 NULL,
							 &neurandefrag_enabled,
							 true,
							 PGC_SIGHUP,
							 0,
							 NULL,
							 NULL,
							 NULL);

	/* ====================================================================
	 * ONNX Runtime GUCs
	 * ==================================================================== */

	DefineCustomStringVariable("neurondb.onnx_model_path",
							   "Directory with ONNX model files",
							   "Files exported from HuggingFace transformers in ONNX format "
							   "must be placed under this directory.",
							   &neurondb_onnx_model_path,
							   "/var/lib/neurondb/models",
							   PGC_SUSET,
							   0,
							   NULL,
							   NULL,
							   NULL);

	DefineCustomBoolVariable("neurondb.onnx_use_gpu",
							 "Attempt to use GPU acceleration for ONNX inference.",
							 "If enabled, CUDA (NVIDIA) or CoreML (macOS) execution will be "
							 "tried before falling back to CPU.",
							 &neurondb_onnx_use_gpu,
							 true,
							 PGC_SUSET,
							 0,
							 NULL,
							 NULL,
							 NULL);

	DefineCustomIntVariable("neurondb.onnx_threads",
							"Number of ONNX Runtime intra-operator threads.",
							"Controls the intra-op-thread pool for ONNX inference.",
							&neurondb_onnx_threads,
							4,
							1,
							64,
							PGC_SUSET,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomIntVariable("neurondb.onnx_cache_size",
							"ONNX model LRU cache size (number of sessions)",
							"When this limit is reached, the least recently used session "
							"will be evicted.",
							&neurondb_onnx_cache_size,
							10,
							1,
							100,
							PGC_SUSET,
							0,
							NULL,
							NULL,
							NULL);

	/* ====================================================================
	 * Quota GUCs
	 * ==================================================================== */

	DefineCustomIntVariable("neurondb.default_max_vectors",
							"Default maximum vectors per tenant (thousands)",
							NULL,
							(int *) &default_max_vectors,
							1000000,
							1000,
							INT_MAX,
							PGC_SIGHUP,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomIntVariable("neurondb.default_max_storage_mb",
							"Default maximum storage (MB) per tenant",
							NULL,
							(int *) &default_max_storage_mb,
							10240,
							100,
							INT_MAX,
							PGC_SIGHUP,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomIntVariable("neurondb.default_max_qps",
							"Default maximum queries per second per tenant",
							NULL,
							&default_max_qps,
							1000,
							1,
							INT_MAX,
							PGC_SIGHUP,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomBoolVariable("neurondb.enforce_quotas",
							 "Enable hard quota enforcement",
							 NULL,
							 &enforce_quotas,
							 true,
							 PGC_SUSET,
							 0,
							 NULL,
							 NULL,
							 NULL);

	/* ====================================================================
	 * AutoML GUCs
	 * ==================================================================== */

	DefineCustomBoolVariable("neurondb.automl.use_gpu",
							 "Enable GPU acceleration for AutoML training",
							 "When enabled, AutoML will prefer GPU training for supported algorithms.",
							 &neurondb_automl_use_gpu,
							 false,
							 PGC_USERSET,
							 0,
							 NULL,
							 NULL,
							 NULL);

	/* Sync initial values to config structure */
	neurondb_sync_config_from_gucs();
}


