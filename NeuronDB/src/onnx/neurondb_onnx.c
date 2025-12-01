/*
 * neurondb_onnx.c
 * 
 * ONNX Runtime Integration Layer for HuggingFace Models in NeuronDB
 *
 * This module serves as the main implementation for wrapping ONNX Runtime C API
 * functionality, allowing NeuronDB to manage, cache, execute, and introspect 
 * inference sessions for models exported from HuggingFace (transformers etc.) in ONNX format.
 *
 * Core Architecture:
 * 
 *     HuggingFace Model (PyTorch/Transformers)
 *         |
 *     (ONNX Export)
 *         |
 *     ONNX Model (.onnx file)
 *         |
 *     [ONNX Runtime C API]
 *         |
 *     This Module (neurondb_onnx.c)
 *         |
 *     PostgreSQL (NeuronDB Extension)  
 * 
 * Supported Execution Providers (EPs):
 *     - "CPUExecutionProvider": Always available on all platforms.
 *     - "CUDAExecutionProvider": NVIDIA GPU inference (requires ORT compiled with CUDA).
 *     - "TensorRTExecutionProvider": NVIDIA TensorRT acceleration (if supported).
 *     - "CoreMLExecutionProvider": Apple Silicon GPU acceleration (on macOS only).
 *     - (Expandable: DirectML, OpenVINO, etc.)
 *
 * Model session caching is handled in a least-recently-used (LRU) fashion to avoid 
 * excessive memory pressure and ensure quick re-use of hot models.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 * SPDX-License-Identifier: PostgreSQL
 */

#include "postgres.h"

#include "fmgr.h"
#include "funcapi.h"
#include "miscadmin.h"
#include "access/htup_details.h"
#include "catalog/pg_type.h"
#include "storage/lwlock.h"
#include "storage/shmem.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "neurondb_onnx.h"

/* GUC variables are now defined in neurondb_guc.c */

#ifdef HAVE_ONNX_RUNTIME

#include <onnxruntime_c_api.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/* ---- Global ONNX API objects ---- */
static const OrtApi *g_ort_api = NULL; /* ONNX Runtime C API Instance */
static OrtEnv *g_ort_env = NULL; /* ONNX Runtime Environment   */
static bool g_onnx_initialized = false; /* ONNX env state            */

/* ---- LRU Model Session Cache ---- */
typedef struct ONNXCacheEntry
{
	char *model_name; /* Key: Model identifier (string)           */
	ONNXModelSession
		*session; /* Loaded ONNX session (opaque struct)      */
	time_t last_used; /* Unix time for LRU eviction scoring       */
	struct ONNXCacheEntry
		*next; /* Linked-list pointer                      */
} ONNXCacheEntry;

static ONNXCacheEntry *g_model_cache_head =
	NULL; /* Start of cache linked list */
static int g_model_cache_count = 0; /* Number of currently cached sessions */

/* ---- Function Forward Declarations ---- */
static void onnx_check_status(OrtStatus *status, const char *operation);
static void onnx_cache_add(const char *model_name, ONNXModelSession *session);
static ONNXModelSession *onnx_cache_get(const char *model_name);
static void onnx_cache_evict_lru(void);

/*
 * onnx_check_status
 * 
 * Centralized helper to check the result of ONNX Runtime API operations.
 * If an error (status != NULL) is encountered, elog/ereport ERROR immediately,
 * including the ONNX error message and operation.
 */
static void
onnx_check_status(OrtStatus *status, const char *operation)
{
	const char *error_message;

	if (status == NULL)
		return;

	error_message = g_ort_api->GetErrorMessage(status);
	g_ort_api->ReleaseStatus(status);

	ereport(ERROR,
		(errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
			errmsg("ONNX Runtime error during %s: %s",
				operation,
				error_message ? error_message
					      : "(no message)")));
}

/*
 * neurondb_onnx_init
 * 
 * Initializes ONNX Runtime for the process. Must be called before any ONNX ops.
 * Handles idempotency -- it is safe for multiple calls in the same backend.
 */
void
neurondb_onnx_init(void)
{
	OrtStatus *status;

	if (g_onnx_initialized)
		return;

	g_ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
	if (!g_ort_api)
		ereport(ERROR,
			(errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
				errmsg("Failed to initialize ONNX Runtime API"),
				errhint("Ensure the ONNX Runtime library is "
					"installed and accessible")));

	status = g_ort_api->CreateEnv(
		ORT_LOGGING_LEVEL_WARNING, "NeuronDB", &g_ort_env);
	onnx_check_status(status, "CreateEnv");

	g_onnx_initialized = true;

	ereport(LOG,
		(errmsg("ONNX Runtime initialized for NeuronDB"),
			errdetail("ONNX Runtime version: %s",
				neurondb_onnx_version())));
}

/*
 * neurondb_onnx_cleanup
 * 
 * Release all ONNX Runtime resources, including the global ONNX Runtime environment
 * and all cached model sessions.
 */
void
neurondb_onnx_cleanup(void)
{
	ONNXCacheEntry *entry, *next;

	if (!g_onnx_initialized)
		return;

	/* Free all model sessions from the cache. */
	for (entry = g_model_cache_head; entry != NULL; entry = next)
	{
		next = entry->next;
		if (entry->session)
			neurondb_onnx_unload_model(entry->session);
		if (entry->model_name)
			NDB_FREE(entry->model_name);
		NDB_FREE(entry);
	}

	g_model_cache_head = NULL;
	g_model_cache_count = 0;

	if (g_ort_env)
	{
		g_ort_api->ReleaseEnv(g_ort_env);
		g_ort_env = NULL;
	}

	g_onnx_initialized = false;
}

/*
 * neurondb_onnx_load_model
 *
 * Loads a specified ONNX model into memory, configures inference options and EP, and returns a
 * pointer to a persistent ONNXModelSession object. Used internally and by the cache loader.
 *
 * Returns pointer to allocated ONNXModelSession, or ERROR on failure.
 */
ONNXModelSession *
neurondb_onnx_load_model(const char *model_path,
	ONNXModelType model_type,
	ONNXProvider provider)
{
	ONNXModelSession *session;
	OrtSessionOptions *session_options = NULL;
	OrtStatus *status = NULL;

	Assert(model_path != NULL);

	if (!g_onnx_initialized)
		neurondb_onnx_init();

	/* Allocate ONNXModelSession (tracked by TopMemoryContext for session lifetime) */
	session = (ONNXModelSession *)MemoryContextAllocZero(
		TopMemoryContext, sizeof(ONNXModelSession));
	session->model_path = MemoryContextStrdup(TopMemoryContext, model_path);
	session->model_type = model_type;
	session->provider = provider;
	session->env = g_ort_env;
	session->is_loaded = false;

	/* Create ONNX session options */
	status = g_ort_api->CreateSessionOptions(&session_options);
	onnx_check_status(status, "CreateSessionOptions");

	/* Set number of intra-op threads (parallelism inside individual ops) */
	status = g_ort_api->SetIntraOpNumThreads(
		session_options, neurondb_onnx_threads);
	onnx_check_status(status, "SetIntraOpNumThreads");

	/* Use maximum graph optimization */
	status = g_ort_api->SetSessionGraphOptimizationLevel(
		session_options, ORT_ENABLE_ALL);
	onnx_check_status(status, "SetSessionGraphOptimizationLevel");

	/* Select/initialize execution provider, attempting to satisfy 'provider' */
	if (neurondb_onnx_use_gpu)
	{
		switch (provider)
		{
		case ONNX_PROVIDER_CUDA:
#ifdef ORT_ENABLE_CUDA
			/* Attempt CUDA provider (requires ORT built with CUDA) */
			{
				OrtCUDAProviderOptions cuda_options;
				memset(&cuda_options, 0, sizeof(cuda_options));
				cuda_options.device_id = 0;
				cuda_options.cudnn_conv_algo_search =
					OrtCudnnConvAlgoSearchHeuristic;
				cuda_options.gpu_mem_limit = SIZE_MAX;
				cuda_options.arena_extend_strategy = 0;

				status =
					g_ort_api
						->SessionOptionsAppendExecutionProvider_CUDA(
							session_options,
							&cuda_options);

				if (status != NULL)
				{
						"CUDA provider initialization "
						"failed; falling back to CPU. "
						"Model: %s",
						model_path);
					g_ort_api->ReleaseStatus(status);
					provider = ONNX_PROVIDER_CPU;
				} else
				{
					elog(LOG,
						"ONNX model will use CUDA "
						"execution provider");
				}
			}
#else
			elog(DEBUG1,
				"ONNX Runtime CUDA provider not compiled in; "
				"falling back to CPU for model: %s",
				model_path);
			provider = ONNX_PROVIDER_CPU;
#endif
			break;

		case ONNX_PROVIDER_COREML:
#ifdef __APPLE__
			/* CoreML provider is available on macOS but API varies by ONNX Runtime version */
			/* For ONNX Runtime 1.17+, CoreML is added via session options differently */
			elog(LOG,
				"CoreML provider requested - using CPU "
				"provider (CoreML support TBD for ORT 1.17+)");
			provider = ONNX_PROVIDER_CPU;
#else
			elog(DEBUG1,
				"CoreML execution provider not supported "
				"(non-macOS); falling back to CPU for model: "
				"%s",
				model_path);
			provider = ONNX_PROVIDER_CPU;
#endif
			break;

		case ONNX_PROVIDER_CPU:
		default:
			/* CPU is always supported. */
			break;
		}
	}

	session->session_options = session_options;
	session->provider = provider; /* May have changed based on fallbacks */

	/* Actually load the session from file */
#ifdef _WIN32
	{
		wchar_t wmodel_path[1024];
		MultiByteToWideChar(
			CP_UTF8, 0, model_path, -1, wmodel_path, 1024);
		status = g_ort_api->CreateSession(g_ort_env,
			wmodel_path,
			session_options,
			&session->session);
	}
#else
	status = g_ort_api->CreateSession(
		g_ort_env, model_path, session_options, &session->session);
#endif
	onnx_check_status(status, "CreateSession");

	session->is_loaded = true;

	ereport(DEBUG1,
		(errmsg("Loaded ONNX model: %s", model_path),
			errdetail("Type: %s; Execution Provider: %s",
				neurondb_onnx_model_type_name(model_type),
				neurondb_onnx_provider_name(provider))));

	return session;
}

/*
 * neurondb_onnx_unload_model
 * 
 * Releases all resources for a model session. This involves ONNX session objects 
 * and all model/session-path memory, and frees the ONNXModelSession struct.
 */
void
neurondb_onnx_unload_model(ONNXModelSession *session)
{
	if (!session)
		return;

	if (session->session)
		g_ort_api->ReleaseSession(session->session);

	if (session->session_options)
		g_ort_api->ReleaseSessionOptions(session->session_options);

	if (session->model_path)
		NDB_FREE(session->model_path);

	NDB_FREE(session);
}

/*
 * neurondb_onnx_create_tensor
 * 
 * Allocates and copies a new ONNXTensor structure from provided flat float array of data,
 * shape, and dimension count.
 */
ONNXTensor *
neurondb_onnx_create_tensor(float *data, int64 *shape, int32 ndim)
{
	ONNXTensor *tensor;
	size_t total_size = 1;
	int i;

	Assert(data != NULL);
	Assert(shape != NULL);
	Assert(ndim > 0);

	for (i = 0; i < ndim; i++)
		total_size *= (size_t)shape[i];

	tensor = (ONNXTensor *)palloc0(sizeof(ONNXTensor));
	tensor->data = (float *)palloc(total_size * sizeof(float));
	memcpy(tensor->data, data, total_size * sizeof(float));

	/* Allocate shape as int64_t for ONNX API compatibility */
	tensor->shape = (int64 *)palloc(ndim * sizeof(int64));
	for (i = 0; i < ndim; i++)
		tensor->shape[i] = shape[i];

	tensor->ndim = ndim;
	tensor->size = total_size;

	return tensor;
}

/*
 * neurondb_onnx_run_inference
 * 
 * Executes inference for the specified ONNXModelSession, using an input ONNXTensor.
 * Allocates and returns a new ONNXTensor (caller takes ownership).
 */
ONNXTensor *
neurondb_onnx_run_inference(ONNXModelSession *session, ONNXTensor *input)
{
	OrtStatus *status;
	OrtValue *input_tensor = NULL;
	OrtValue *output_tensor = NULL;
	OrtMemoryInfo *memory_info = NULL;
	OrtTensorTypeAndShapeInfo *output_info = NULL;
	ONNXTensor *output;
	float *output_data;
	int64 *output_dims;
	size_t output_size, num_dims, i;
	const char *input_names[] = { "input_ids" };
	const char *output_names_embed[] = { "last_hidden_state" };
	const char *output_names_gen[] = { "logits" };
	const char **output_names;

	if (!session || !session->is_loaded)
		ereport(ERROR,
			(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				errmsg("ONNX model session not loaded")));

	if (!input || !input->data)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("Invalid input tensor")));

	/* Setup memory info indicating CPU-allocated memory */
	status = g_ort_api->CreateCpuMemoryInfo(
		OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
	onnx_check_status(status, "CreateCpuMemoryInfo");

	/* Create ONNX input tensor - need to cast int64* to int64_t* */
	{
		int64_t *shape_cast =
			(int64_t *)palloc(input->ndim * sizeof(int64_t));
		int i;

		for (i = 0; i < input->ndim; i++)
			shape_cast[i] = (int64_t)input->shape[i];

		status = g_ort_api->CreateTensorWithDataAsOrtValue(memory_info,
			input->data,
			input->size * sizeof(float),
			shape_cast,
			input->ndim,
			ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
			&input_tensor);

		NDB_FREE(shape_cast);
	}
	onnx_check_status(status, "CreateTensorWithDataAsOrtValue");

	/* The default assumption: input name "input_ids", output name "last_hidden_state".
	   These can be generalized if the model schema is passed in future.
	   For generation models, use "logits" as output name. */
	/* For generation models, use "logits" as output name */
	if (session->model_type == ONNX_MODEL_GENERATION)
	{
		output_names = output_names_gen;
	} else
	{
		output_names = output_names_embed;
	}

	status = g_ort_api->Run(session->session,
		NULL, /* No run options */
		input_names,
		(const OrtValue *const *)&input_tensor,
		1,
		output_names,
		1,
		&output_tensor);
	onnx_check_status(status, "Run");

	status = g_ort_api->GetTensorMutableData(
		output_tensor, (void **)&output_data);
	onnx_check_status(status, "GetTensorMutableData");

	status = g_ort_api->GetTensorTypeAndShape(output_tensor, &output_info);
	onnx_check_status(status, "GetTensorTypeAndShape");

	status = g_ort_api->GetDimensionsCount(output_info, &num_dims);
	onnx_check_status(status, "GetDimensionsCount");

	/* GetDimensions requires int64_t*, allocate temp array and copy to int64* */
	{
		int64_t *dims_temp =
			(int64_t *)palloc(num_dims * sizeof(int64_t));

		status = g_ort_api->GetDimensions(
			output_info, dims_temp, num_dims);
		onnx_check_status(status, "GetDimensions");

		/* Copy to PostgreSQL int64 */
		output_dims = (int64 *)palloc(num_dims * sizeof(int64));
		for (i = 0; i < num_dims; i++)
			output_dims[i] = (int64)dims_temp[i];

		NDB_FREE(dims_temp);
	}

	output_size = 1;
	for (i = 0; i < num_dims; i++)
		output_size *= output_dims[i];

	output = (ONNXTensor *)palloc0(sizeof(ONNXTensor));
	output->data = (float *)palloc(output_size * sizeof(float));
	memcpy(output->data, output_data, output_size * sizeof(float));
	output->shape = output_dims;
	output->ndim = num_dims;
	output->size = output_size;

	g_ort_api->ReleaseTensorTypeAndShapeInfo(output_info);
	g_ort_api->ReleaseValue(output_tensor);
	g_ort_api->ReleaseValue(input_tensor);
	g_ort_api->ReleaseMemoryInfo(memory_info);

	return output;
}

/*
 * neurondb_onnx_free_tensor
 * 
 * Frees an ONNXTensor structure created by neurondb_onnx_create_tensor or
 * neurondb_onnx_run_inference, including its associated data and shape arrays.
 */
void
neurondb_onnx_free_tensor(ONNXTensor *tensor)
{
	if (!tensor)
		return;
	if (tensor->data)
		NDB_FREE(tensor->data);
	if (tensor->shape)
		NDB_FREE(tensor->shape);
	NDB_FREE(tensor);
}

/*
 * onnx_cache_add
 * 
 * Adds a new ONNX model session to the LRU cache, evicting the least recently used entry
 * if at maximum capacity.
 */
static void
onnx_cache_add(const char *model_name, ONNXModelSession *session)
{
	ONNXCacheEntry *entry;

	/* Evict oldest entry if cache at or above capacity. */
	if (g_model_cache_count >= neurondb_onnx_cache_size)
		onnx_cache_evict_lru();

	entry = (ONNXCacheEntry *)MemoryContextAllocZero(
		TopMemoryContext, sizeof(ONNXCacheEntry));
	entry->model_name = MemoryContextStrdup(TopMemoryContext, model_name);
	entry->session = session;
	entry->last_used = time(NULL);
	entry->next = g_model_cache_head;

	g_model_cache_head = entry;
	g_model_cache_count++;

	ereport(DEBUG2,
		(errmsg("ONNX model cache insert: %s", model_name),
			errdetail("Cache entries: %d/%d",
				g_model_cache_count,
				neurondb_onnx_cache_size)));
}

/*
 * onnx_cache_get
 * 
 * Looks up a model name in the model session cache.
 * Moves the entry to the "most recently used" position by updating its last_used.
 */
static ONNXModelSession *
onnx_cache_get(const char *model_name)
{
	ONNXCacheEntry *entry;

	for (entry = g_model_cache_head; entry; entry = entry->next)
	{
		if (strcmp(entry->model_name, model_name) == 0)
		{
			entry->last_used = time(NULL);
			ereport(DEBUG2,
				(errmsg("ONNX cache hit: %s", model_name)));
			return entry->session;
		}
	}
	return NULL;
}

/*
 * onnx_cache_evict_lru
 * 
 * Evicts the least-recently-used cache entry, used when cache is full.
 */
static void
onnx_cache_evict_lru(void)
{
	ONNXCacheEntry *entry, *prev, *lru_entry, *lru_prev;
	time_t oldest;

	if (!g_model_cache_head)
		return;

	/* Find the LRU entry (next-to-last loop for lru_prev) */
	lru_entry = g_model_cache_head;
	lru_prev = NULL;
	oldest = lru_entry->last_used;
	prev = NULL;
	for (entry = g_model_cache_head; entry;
		prev = entry, entry = entry->next)
	{
		if (entry->last_used < oldest)
		{
			oldest = entry->last_used;
			lru_entry = entry;
			lru_prev = prev;
		}
	}
	if (lru_prev)
		lru_prev->next = lru_entry->next;
	else
		g_model_cache_head = lru_entry->next;

	ereport(DEBUG1,
		(errmsg("Evicting ONNX model from cache: %s",
			lru_entry->model_name)));

	neurondb_onnx_unload_model(lru_entry->session);
	NDB_FREE(lru_entry->model_name);
	NDB_FREE(lru_entry);

	g_model_cache_count--;
}

/*
 * neurondb_onnx_get_or_load_model
 * 
 * Gets a model session from the cache, or loads and caches it if not present.
 * Builds on the session LRU cache and directory config logic.
 */
ONNXModelSession *
neurondb_onnx_get_or_load_model(const char *model_name,
	ONNXModelType model_type)
{
	ONNXModelSession *session;
	char model_path[MAXPGPATH];
	ONNXProvider provider;

	/* Check for a cache hit first */
	session = onnx_cache_get(model_name);
	if (session)
		return session;

	if (!neurondb_onnx_model_path)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb.onnx_model_path is not set"),
				errhint("Set neurondb.onnx_model_path in "
					"postgresql.conf to use ONNX "
					"models.")));

	snprintf(model_path,
		MAXPGPATH,
		"%s/%s/model.onnx",
		neurondb_onnx_model_path,
		model_name);

	/* Establish preferred provider: try CUDA/CoreML if enabled and supported, else CPU */
	provider = ONNX_PROVIDER_CPU;
	if (neurondb_onnx_use_gpu)
	{
#ifdef ORT_ENABLE_CUDA
		provider = ONNX_PROVIDER_CUDA;
#elif defined(__APPLE__)
		provider = ONNX_PROVIDER_COREML;
#endif
	}

	session = neurondb_onnx_load_model(model_path, model_type, provider);
	onnx_cache_add(model_name, session);

	return session;
}

/*
 * neurondb_onnx_version
 *
 * Return a non-null pointer to a string representing the ONNX Runtime library version,
 * or "not initialized" if the API instance isn't ready.
 */
const char *
neurondb_onnx_version(void)
{
	if (g_ort_api == NULL)
		return "not initialized";

	/* ORT_API_VERSION provides version info */
	return "ONNX Runtime 1.17.0+";
}

/*
 * neurondb_onnx_get_providers
 * 
 * Returns an array of strings describing available execution providers in this
 * build of neurondb_onnx (CPU, CUDA, TensorRT, CoreML, etc.)
 * Populates num_providers with the count of returned providers.
 */
char **
neurondb_onnx_get_providers(int *num_providers)
{
	char **providers;
	int count = 0;

	Assert(num_providers != NULL);

	/* Allocate extra entries for possible providers */
	providers = (char **)palloc(5 * sizeof(char *));

	providers[count++] = pstrdup("CPUExecutionProvider");
#ifdef ORT_ENABLE_CUDA
	providers[count++] = pstrdup("CUDAExecutionProvider");
	providers[count++] = pstrdup("TensorRTExecutionProvider");
#endif
#ifdef __APPLE__
	providers[count++] = pstrdup("CoreMLExecutionProvider");
#endif

	*num_providers = count;
	return providers;
}

/*
 * neurondb_onnx_available
 * 
 * Returns true if ONNX Runtime has been initialized/available in this backend.
 */
bool
neurondb_onnx_available(void)
{
	return g_onnx_initialized;
}

#else /* !HAVE_ONNX_RUNTIME */

/*
 * Intentional conditional compilation stub: no ONNX Runtime support.
 * This stub allows compilation without ONNX Runtime - functions return errors at runtime.
 * All API entry points ereport ERROR or return nothing as appropriate.
 */

void
neurondb_onnx_init(void)
{
	ereport(ERROR,
		(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			errmsg("ONNX Runtime support not compiled in"),
			errhint("Rebuild NeuronDB with --enable-onnx flag")));
}

void
neurondb_onnx_cleanup(void)
{
	/* Intentionally empty: never called */
}

ONNXModelSession *
neurondb_onnx_load_model(const char *model_path,
	ONNXModelType model_type,
	ONNXProvider provider)
{
	ereport(ERROR,
		(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			errmsg("ONNX Runtime support not compiled in")));
	return NULL;
}

void
neurondb_onnx_unload_model(ONNXModelSession *session)
{
	/* Intentionally empty: nothing to do */
}

ONNXTensor *
neurondb_onnx_create_tensor(float *data, int64 *shape, int32 ndim)
{
	ereport(ERROR,
		(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			errmsg("ONNX Runtime support not compiled in")));
	return NULL;
}

ONNXTensor *
neurondb_onnx_run_inference(ONNXModelSession *session, ONNXTensor *input)
{
	ereport(ERROR,
		(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			errmsg("ONNX Runtime support not compiled in")));
	return NULL;
}

void
neurondb_onnx_free_tensor(ONNXTensor *tensor)
{
	/* Intentionally empty */
}

ONNXModelSession *
neurondb_onnx_get_or_load_model(const char *model_name,
	ONNXModelType model_type)
{
	ereport(ERROR,
		(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			errmsg("ONNX Runtime support not compiled in")));
	return NULL;
}

const char *
neurondb_onnx_version(void)
{
	return "ONNX Runtime not available";
}

char **
neurondb_onnx_get_providers(int *num_providers)
{
	*num_providers = 0;
	return NULL;
}

bool
neurondb_onnx_available(void)
{
	return false;
}

#endif /* HAVE_ONNX_RUNTIME */

/*
 * Utility Functions - Always Compiled
 */

/* 
 * neurondb_onnx_provider_name
 * 
 * Convert ONNXProvider enum to user-friendly string.
 */
const char *
neurondb_onnx_provider_name(ONNXProvider provider)
{
	switch (provider)
	{
	case ONNX_PROVIDER_CPU:
		return "CPU";
	case ONNX_PROVIDER_CUDA:
		return "CUDA";
	case ONNX_PROVIDER_TENSORRT:
		return "TensorRT";
	case ONNX_PROVIDER_COREML:
		return "CoreML";
	case ONNX_PROVIDER_DIRECTML:
		return "DirectML";
	default:
		return "Unknown";
	}
}

/*
 * neurondb_onnx_model_type_name
 * 
 * Convert ONNXModelType enum to user-friendly description.
 */
const char *
neurondb_onnx_model_type_name(ONNXModelType type)
{
	switch (type)
	{
	case ONNX_MODEL_EMBEDDING:
		return "Embedding";
	case ONNX_MODEL_CLASSIFICATION:
		return "Classification";
	case ONNX_MODEL_NER:
		return "NER";
	case ONNX_MODEL_QA:
		return "QuestionAnswering";
	case ONNX_MODEL_GENERATION:
		return "Generation";
	case ONNX_MODEL_CUSTOM:
		return "Custom";
	default:
		return "Unknown";
	}
}

/* GUC initialization is now centralized in neurondb_guc.c */
