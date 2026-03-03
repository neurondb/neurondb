/*-------------------------------------------------------------------------
 * neurondb_guc.c
 *   Centralized GUC (Grand Unified Configuration) handling for NeuronDB
 *
 * This module consolidates all GUC variable definitions and provides
 * a unified NeuronDBConfig structure for accessing configuration values.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *-------------------------------------------------------------------------*/

#include "postgres.h"
#include "fmgr.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include <limits.h>
#include "neurondb_guc.h"
#include "neurondb_macros.h"
#include "neurondb_constants.h"

/* Enum options for iterative_scan */
static const struct config_enum_entry iterative_scan_options[] = {
	{"off", 0, false},
	{"strict_order", 1, false},
	{"relaxed_order", 2, false},
	{NULL, 0, false}
};

int			neurondb_hnsw_ef_search = 64;
int			neurondb_hnsw_k = 10;
int			neurondb_ivf_probes = 10;
int			neurondb_ef_construction = 200;

/* Iterative scan parameters */
int			hnsw_iterative_scan = 0;		/* 0=off, 1=strict_order, 2=relaxed_order */
int			hnsw_max_scan_tuples = 20000;
double		hnsw_scan_mem_multiplier = 1.0;
int			ivf_iterative_scan = 0;	/* 0=off, 1=strict_order, 2=relaxed_order */
int			ivf_max_probes = 100;

int			neurondb_compute_mode = 0;
/* Default GPU backend type: set at compile time based on available backend */
#if defined(NDB_GPU_METAL)
#define NDB_GPU_BACKEND_TYPE_DEFAULT 2	/* Metal on macOS */
#elif defined(NDB_GPU_ROCM)
#define NDB_GPU_BACKEND_TYPE_DEFAULT 1	/* ROCm on AMD */
#else
#define NDB_GPU_BACKEND_TYPE_DEFAULT 0	/* CUDA by default */
#endif

int			neurondb_gpu_backend_type = NDB_GPU_BACKEND_TYPE_DEFAULT;
int			neurondb_gpu_device = 0;
int			neurondb_gpu_batch_size = 8192;
int			neurondb_gpu_streams = 2;
double		neurondb_gpu_memory_pool_mb = 512.0;
char	   *neurondb_gpu_kernels = NULL;
int			neurondb_gpu_timeout_ms = 30000;

char	   *neurondb_llm_provider = NULL;
char	   *neurondb_llm_model = NULL;
char	   *neurondb_llm_endpoint = NULL;
char	   *neurondb_llm_api_key = NULL;
int			neurondb_llm_timeout_ms = 30000;
int			neurondb_llm_cache_ttl = 600;
int			neurondb_llm_rate_limiter_qps = 5;
bool		neurondb_llm_fail_open = true;

static int	neuranq_naptime = 1000;
static int	neuranq_queue_depth = 10000;
static int	neuranq_batch_size = 100;
static int	neuranq_timeout = 30000;
static int	neuranq_max_retries = 3;
static bool neuranq_enabled = true;

static int	neuranmon_naptime = 60000;
static int	neuranmon_sample_size = 1000;
static double neuranmon_target_latency = 100.0;
static double neuranmon_target_recall = 0.95;
static bool neuranmon_enabled = true;

static int	neurandefrag_naptime = 300000;
static int	neurandefrag_compact_threshold = 10000;
static double neurandefrag_fragmentation_threshold = 0.3;
static char *neurandefrag_maintenance_window = "02:00-04:00";
static bool neurandefrag_enabled = true;

char	   *neurondb_onnx_model_path = NULL;
bool		neurondb_onnx_use_gpu = true;
int			neurondb_onnx_threads = 4;
int			neurondb_onnx_cache_size = 10;

static int64 default_max_vectors = 1000000;
static int64 default_max_storage_mb = 10240;
static int	default_max_qps = 1000;
static bool enforce_quotas = true;

bool		neurondb_automl_use_gpu = false;
bool		neurondb_vector_capsule_enabled = false;

/* Security and governance GUCs */
bool		neurondb_confidential_compute = false;
bool		neurondb_rls_embeddings_enabled = false;
bool		neurondb_encryption_enabled = false;
bool		neurondb_audit_ml_enabled = false;
bool		neurondb_audit_rag_enabled = false;
int			neurondb_audit_retention_days = 365;

/* ML training/prediction limits (configurable) */
int			neurondb_ml_max_samples = 200000;
int			neurondb_ml_max_feature_elements = 100000;

/* Replication GUCs */
bool		neurondb_enable_replication = false;

NeuronDBConfig *neurondb_config = NULL;

static bool
neurondb_check_gpu_backend_type(int *newval, void **extra, GucSource source)
{
	if (neurondb_compute_mode == NDB_COMPUTE_MODE_CPU)
	{
		elog(WARNING,
			 "neurondb.gpu_backend_type is ignored when neurondb.compute_mode is 'cpu'");
		return true;
	}
	return true;
}

void
neurondb_sync_config_from_gucs(void)
{
	if (neurondb_config == NULL)
		return;

	neurondb_config->core.hnsw_ef_search = neurondb_hnsw_ef_search;
	neurondb_config->core.ivf_probes = neurondb_ivf_probes;
	neurondb_config->core.ef_construction = neurondb_ef_construction;

	neurondb_config->gpu.compute_mode = neurondb_compute_mode;
	neurondb_config->gpu.backend_type = neurondb_gpu_backend_type;
	neurondb_config->gpu.device = neurondb_gpu_device;
	neurondb_config->gpu.batch_size = neurondb_gpu_batch_size;
	neurondb_config->gpu.streams = neurondb_gpu_streams;
	neurondb_config->gpu.memory_pool_mb = neurondb_gpu_memory_pool_mb;
	neurondb_config->gpu.kernels = neurondb_gpu_kernels;
	neurondb_config->gpu.timeout_ms = neurondb_gpu_timeout_ms;

	neurondb_config->llm.provider = neurondb_llm_provider;
	neurondb_config->llm.model = neurondb_llm_model;
	neurondb_config->llm.endpoint = neurondb_llm_endpoint;
	neurondb_config->llm.api_key = neurondb_llm_api_key;
	neurondb_config->llm.timeout_ms = neurondb_llm_timeout_ms;
	neurondb_config->llm.cache_ttl = neurondb_llm_cache_ttl;
	neurondb_config->llm.rate_limiter_qps = neurondb_llm_rate_limiter_qps;
	neurondb_config->llm.fail_open = neurondb_llm_fail_open;

	neurondb_config->neuranq.naptime = neuranq_naptime;
	neurondb_config->neuranq.queue_depth = neuranq_queue_depth;
	neurondb_config->neuranq.batch_size = neuranq_batch_size;
	neurondb_config->neuranq.timeout = neuranq_timeout;
	neurondb_config->neuranq.max_retries = neuranq_max_retries;
	neurondb_config->neuranq.enabled = neuranq_enabled;

	neurondb_config->neuranmon.naptime = neuranmon_naptime;
	neurondb_config->neuranmon.sample_size = neuranmon_sample_size;
	neurondb_config->neuranmon.target_latency = neuranmon_target_latency;
	neurondb_config->neuranmon.target_recall = neuranmon_target_recall;
	neurondb_config->neuranmon.enabled = neuranmon_enabled;

	neurondb_config->neurandefrag.naptime = neurandefrag_naptime;
	neurondb_config->neurandefrag.compact_threshold = neurandefrag_compact_threshold;
	neurondb_config->neurandefrag.fragmentation_threshold = neurandefrag_fragmentation_threshold;
	neurondb_config->neurandefrag.maintenance_window = neurandefrag_maintenance_window;
	neurondb_config->neurandefrag.enabled = neurandefrag_enabled;

	neurondb_config->onnx.model_path = neurondb_onnx_model_path;
	neurondb_config->onnx.use_gpu = neurondb_onnx_use_gpu;
	neurondb_config->onnx.threads = neurondb_onnx_threads;
	neurondb_config->onnx.cache_size = neurondb_onnx_cache_size;

	neurondb_config->quota.default_max_vectors = default_max_vectors;
	neurondb_config->quota.default_max_storage_mb = default_max_storage_mb;
	neurondb_config->quota.default_max_qps = default_max_qps;
	neurondb_config->quota.enforce_quotas = enforce_quotas;

	neurondb_config->automl.use_gpu = neurondb_automl_use_gpu;
}

void
neurondb_init_all_gucs(void)
{
	MemoryContext oldcontext;

	NeuronDBConfig *config = NULL;

	oldcontext = MemoryContextSwitchTo(TopMemoryContext);
	nalloc(config, NeuronDBConfig, 1);
	neurondb_config = config;
	MemoryContextSwitchTo(oldcontext);

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


	DefineCustomIntVariable("neurondb.hnsw_k",
							"Sets the k parameter for HNSW index scans",
							"Number of nearest neighbors to return. Default is 10.",
							&neurondb_hnsw_k,
							10,
							1,
							1000,
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

	/* HNSW iterative scan parameters */
	DefineCustomEnumVariable("neurondb.hnsw_iterative_scan",
							 "Sets the mode for iterative scans",
							 "Valid values: 'off' (default), 'strict_order', 'relaxed_order'. "
							 "Iterative scans automatically extend index searches when filtering reduces results.",
							 &hnsw_iterative_scan,
							 0, /* HNSW_ITERATIVE_SCAN_OFF */
							 iterative_scan_options,
							 PGC_USERSET,
							 0,
							 NULL,
							 NULL,
							 NULL);

	DefineCustomIntVariable("neurondb.hnsw_max_scan_tuples",
							"Sets the max number of tuples to visit for iterative scans",
							"This is approximate and does not affect the initial scan. Default is 20000.",
							&hnsw_max_scan_tuples,
							20000,
							1,
							INT_MAX,
							PGC_USERSET,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomRealVariable("neurondb.hnsw_scan_mem_multiplier",
							 "Sets the multiple of work_mem to use for iterative scans",
							 "Default is 1.0. Try increasing this if increasing neurondb.hnsw_max_scan_tuples does not improve recall.",
							 &hnsw_scan_mem_multiplier,
							 1.0,
							 1.0,
							 1000.0,
							 PGC_USERSET,
							 0,
							 NULL,
							 NULL,
							 NULL);

	/* IVF iterative scan parameters */
	DefineCustomEnumVariable("neurondb.ivf_iterative_scan",
							 "Sets the mode for iterative scans",
							 "Valid values: 'off' (default), 'relaxed_order'. "
							 "Iterative scans automatically extend index searches when filtering reduces results.",
							 &ivf_iterative_scan,
							 0, /* IVF_ITERATIVE_SCAN_OFF */
							 iterative_scan_options,
							 PGC_USERSET,
							 0,
							 NULL,
							 NULL,
							 NULL);

	DefineCustomIntVariable("neurondb.ivf_max_probes",
							"Sets the max number of probes for iterative scans",
							"If this is lower than neurondb.ivf_probes, neurondb.ivf_probes will be used. Default is 100.",
							&ivf_max_probes,
							100,
							1,
							INT_MAX,
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

	DefineCustomIntVariable("neurondb.compute_mode",
							"Compute execution mode",
							"Controls whether ML operations run on CPU, GPU, or auto-select. "
							"Values: 0 (cpu) - CPU only, don't initialize GPU; "
							"1 (gpu) - GPU required, error if unavailable; "
							"2 (auto) - Try GPU first, fallback to CPU. Default is 0 (cpu).",
							&neurondb_compute_mode,
							0,
							0,
							2,
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

	DefineCustomIntVariable("neurondb.gpu_backend_type",
							"GPU backend type",
							"Selects GPU backend implementation. Only valid when compute_mode is 'gpu' or 'auto'. "
							"Values: 0 (cuda) - NVIDIA CUDA; 1 (rocm) - AMD ROCm; 2 (metal) - Apple Metal. "
							"Default is platform-specific (cuda=0, rocm=1, metal=2). Ignored when compute_mode is 'cpu'.",
							&neurondb_gpu_backend_type,
							NDB_GPU_BACKEND_TYPE_DEFAULT,
							0,
							2,
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

	DefineCustomBoolVariable("neurondb.vector_capsule_enabled",
							 "Enable VectorCapsule features (multi-representation vectors with metadata).",
							 "When enabled, allows creation of VectorCapsule types with adaptive representation selection, integrity checking, and provenance tracking.",
							 &neurondb_vector_capsule_enabled,
							 false,
							 PGC_USERSET,
							 0,
							 NULL,
							 NULL,
							 NULL);

	DefineCustomIntVariable("neurondb.ml_max_samples",
							"Maximum number of samples allowed for ML training",
							"Training requests exceeding this limit will fail. Default is 200000.",
							&neurondb_ml_max_samples,
							200000,
							100,
							INT_MAX,
							PGC_SUSET,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomIntVariable("neurondb.ml_max_feature_elements",
							"Maximum number of feature elements for prediction input",
							"Prediction feature arrays exceeding this dimension will be rejected. Default is 100000.",
							&neurondb_ml_max_feature_elements,
							100000,
							1,
							INT_MAX,
							PGC_SUSET,
							0,
							NULL,
							NULL,
							NULL);

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

	/* Security and governance GUCs */
	DefineCustomBoolVariable("neurondb.rls_embeddings_enabled",
							 "Enable Row-Level Security for embeddings",
							 "When enabled, RLS policies are enforced during vector index scans and ANN searches.",
							 &neurondb_rls_embeddings_enabled,
							 false,
							 PGC_USERSET,
							 0,
							 NULL,
							 NULL,
							 NULL);

	DefineCustomBoolVariable("neurondb.confidential_compute",
							 "Enable confidential compute mode (e.g. SGX/SEV)",
							 "When enabled, other functions may enforce encryption and audit logging.",
							 &neurondb_confidential_compute,
							 false,
							 PGC_USERSET,
							 0,
							 NULL,
							 NULL,
							 NULL);

	DefineCustomBoolVariable("neurondb.encryption_enabled",
							 "Enable field-level encryption for vectors",
							 "When enabled, supports encryption/decryption of sensitive vector data and metadata.",
							 &neurondb_encryption_enabled,
							 false,
							 PGC_SUSET,
							 0,
							 NULL,
							 NULL,
							 NULL);

	DefineCustomBoolVariable("neurondb.audit_ml_enabled",
							 "Enable audit logging for ML inference operations",
							 "When enabled, logs all ML model inference calls for compliance and security.",
							 &neurondb_audit_ml_enabled,
							 false,
							 PGC_USERSET,
							 0,
							 NULL,
							 NULL,
							 NULL);

	DefineCustomBoolVariable("neurondb.audit_rag_enabled",
							 "Enable audit logging for RAG operations",
							 "When enabled, logs all RAG retrieve and generate operations for compliance.",
							 &neurondb_audit_rag_enabled,
							 false,
							 PGC_USERSET,
							 0,
							 NULL,
							 NULL,
							 NULL);

	DefineCustomIntVariable("neurondb.audit_retention_days",
							"Audit log retention period in days",
							"Audit logs older than this many days will be eligible for archival or deletion. Default is 365.",
							&neurondb_audit_retention_days,
							365,
							1,
							3650,
							PGC_SUSET,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomBoolVariable("neurondb.enable_replication",
							 "Enable replication support for vector indexes",
							 "When enabled, supports streaming replication for HNSW and IVF indexes with consistency guarantees.",
							 &neurondb_enable_replication,
							 false,
							 PGC_SIGHUP,
							 0,
							 NULL,
							 NULL,
							 NULL);

	neurondb_sync_config_from_gucs();
}
