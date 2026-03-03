/*-------------------------------------------------------------------------
 *
 * gpu_model_bridge.c
 *    SQL ML entry points bridge.
 *
 * This module connects SQL ML entry points with the model registry,
 * providing helper routines for training and prediction.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/common/gpu_model_bridge.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include "ml_catalog.h"
#include "neurondb_gpu.h"
#include "neurondb_gpu_backend.h"
#include "ml_gpu_random_forest.h"
#include "ml_gpu_logistic_regression.h"
#include "ml_gpu_linear_regression.h"
#include "neurondb_gpu_bridge.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_constants.h"
#ifdef NDB_GPU_CUDA
#include "neurondb_cuda_linreg.h"
#endif

extern int	ndb_gpu_dt_train(const float *features,
							 const double *labels,
							 int n_samples,
							 int feature_dim,
							 const Jsonb * hyperparams,
							 bytea * *model_data,
							 Jsonb * *metrics,
							 char **errstr);
extern int	ndb_gpu_ridge_train(const float *features,
								const double *targets,
								int n_samples,
								int feature_dim,
								const Jsonb * hyperparams,
								bytea * *model_data,
								Jsonb * *metrics,
								char **errstr);
extern int	ndb_gpu_lasso_train(const float *features,
								const double *targets,
								int n_samples,
								int feature_dim,
								const Jsonb * hyperparams,
								bytea * *model_data,
								Jsonb * *metrics,
								char **errstr);

#include "miscadmin.h"
#include "utils/builtins.h"
#include "utils/memutils.h"
#include "utils/jsonb.h"
#include "lib/stringinfo.h"
#include <string.h>
#include <math.h>

static void
ndb_gpu_init_train_result(MLGpuTrainResult *result)
{
	if (result == NULL)
		return;
	memset(result, 0, sizeof(MLGpuTrainResult));
}

static char *
ndb_gpu_strdup_or_null(const char *src)
{
	if (src == NULL)
		return NULL;
	return MemoryContextStrdup(TopMemoryContext, src);
}

/* Duplicate errstr to TopMemoryContext so it doesn't point to transient memory */
static void
ndb_gpu_ensure_errstr_top(char **errstr)
{
	char *dup;

	if (errstr == NULL || *errstr == NULL)
		return;
	dup = MemoryContextStrdup(TopMemoryContext, *errstr);
	*errstr = dup;
}

bool
ndb_gpu_try_train_model(const char *algorithm,
						const char *project_name,
						const char *model_name,
						const char *training_table,
						const char *training_column,
						const char *const *feature_columns,
						int feature_count,
						Jsonb * hyperparameters,
						const float *feature_matrix,
						const double *label_vector,
						int sample_count,
						int feature_dim,
						int class_count,
						MLGpuTrainResult *result,
						char **errstr)
{
	const MLGpuModelOps *ops;
	MLGpuModel	model;
	MLGpuTrainSpec spec;
	MLGpuContext ctx;
	bytea	   *payload = NULL;
	Jsonb	   *metadata = NULL;
	bool		trained = false;
	volatile bool retval = false;
	volatile bool ops_failed_with_exception = false;
	bool		ops_trained = false;
	bool		is_unsupervised;

	/* Initialize all local variables to safe defaults */
	memset(&model, 0, sizeof(MLGpuModel));
	memset(&spec, 0, sizeof(MLGpuTrainSpec));
	memset(&ctx, 0, sizeof(MLGpuContext));

	if (errstr)
		*errstr = NULL;
	if (result)
		ndb_gpu_init_train_result(result);

	if (NDB_COMPUTE_MODE_IS_CPU())
	{
		if (errstr)
			*errstr = ndb_gpu_strdup_or_null("ndb_gpu_try_train_model: CPU mode - GPU code should not be called");
		ndb_gpu_ensure_errstr_top(errstr);
		return false;
	}

	/* Check if algorithm is unsupervised (doesn't require label_vector) */

	is_unsupervised = (algorithm != NULL && (
		strcmp(algorithm, "gmm") == 0 ||
		strcmp(algorithm, "kmeans") == 0 ||
		strcmp(algorithm, "minibatch_kmeans") == 0 ||
		strcmp(algorithm, "hierarchical") == 0 ||
		strcmp(algorithm, "dbscan") == 0 ||
		strcmp(algorithm, "pca") == 0));

	if (feature_matrix == NULL || sample_count <= 0 || feature_dim <= 0)
	{
		if (errstr)
			*errstr = ndb_gpu_strdup_or_null(psprintf("ndb_gpu_try_train_model: invalid parameters (feature_matrix=%p, sample_count=%d, feature_dim=%d)",
							   (void *) feature_matrix, sample_count, feature_dim));
		ndb_gpu_ensure_errstr_top(errstr);
		return false;
	}

	/* label_vector is required for supervised algorithms only */
	if (!is_unsupervised && label_vector == NULL)
	{
		if (errstr)
			*errstr = ndb_gpu_strdup_or_null(psprintf("ndb_gpu_try_train_model: label_vector is NULL for supervised algorithm '%s'",
							   algorithm ? algorithm : "unknown"));
		ndb_gpu_ensure_errstr_top(errstr);
		return false;
	}

	if (!neurondb_gpu_is_available())
	{
		if (errstr)
			*errstr = ndb_gpu_strdup_or_null("ndb_gpu_try_train_model: GPU is not available");
		/* GPU mode: error if GPU is not available */
		if (!NDB_COMPUTE_MODE_IS_CPU() && NDB_REQUIRE_GPU())
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("%s: GPU is not available - GPU mode requires GPU to be available",
							algorithm ? algorithm : "unknown"),
					 errdetail("GPU backend could not be initialized or is not available"),
					 errhint("Set compute_mode='auto' for automatic CPU fallback.")));
		}
		/* AUTO/CPU mode: return false for fallback */
		ndb_gpu_ensure_errstr_top(errstr);
		return false;
	}

	ops = ndb_gpu_lookup_model_ops(algorithm);
	if (ops == NULL || ops->train == NULL || ops->serialize == NULL)
	{
		/* For algorithms with direct paths (linear_regression, logistic_regression, ridge, lasso), ops might be NULL */
		/* This is OK - we'll use the direct path instead */
		if (algorithm != NULL && strcmp(algorithm, "linear_regression") != 0 && strcmp(algorithm, "logistic_regression") != 0
			&& strcmp(algorithm, "ridge") != 0 && strcmp(algorithm, "lasso") != 0)
		{
			if (errstr)
				*errstr = ndb_gpu_strdup_or_null(psprintf("ndb_gpu_try_train_model: no GPU ops for algorithm '%s'", algorithm));
			/* GPU mode: error if no GPU ops for algorithm */
			if (!NDB_COMPUTE_MODE_IS_CPU() && NDB_REQUIRE_GPU())
			{
				ereport(ERROR,
						(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
						 errmsg("%s: GPU training not supported - no GPU ops available",
								algorithm ? algorithm : "unknown"),
						 errdetail("Algorithm '%s' does not have GPU implementation registered",
								   algorithm ? algorithm : "unknown"),
						 errhint("Set compute_mode='auto' for automatic CPU fallback.")));
			}
			/* AUTO/CPU mode: return false for fallback */
			ndb_gpu_ensure_errstr_top(errstr);
			return false;
		}
	}

	memset(&ctx, 0, sizeof(MLGpuContext));
	ctx.backend = ndb_gpu_get_active_backend();
	if (ctx.backend == NULL)
	{
		if (errstr)
			*errstr = ndb_gpu_strdup_or_null("ndb_gpu_try_train_model: active_backend is NULL");
		/* GPU mode: error if active backend is NULL */
		if (!NDB_COMPUTE_MODE_IS_CPU() && NDB_REQUIRE_GPU())
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("%s: GPU backend is NULL - GPU mode requires GPU backend to be initialized",
							algorithm ? algorithm : "unknown"),
					 errdetail("GPU backend could not be initialized"),
					 errhint("Set compute_mode='auto' for automatic CPU fallback.")));
		}
		/* AUTO/CPU mode: return false for fallback */
		ndb_gpu_ensure_errstr_top(errstr);
		return false;
	}


	ctx.backend_name = (ctx.backend->name) ? ctx.backend->name : "unknown";
	
	/* Check if Metal backend and algorithm is unsupported - skip GPU training and use CPU */
	if (algorithm != NULL && strcmp(ctx.backend_name, "Metal") == 0)
	{
		/* List of algorithms unsupported by Metal GPU */
		if (strcmp(algorithm, "random_forest") == 0 ||
			strcmp(algorithm, "logistic_regression") == 0 ||
			strcmp(algorithm, "svm") == 0 ||
			strcmp(algorithm, "decision_tree") == 0 ||
			strcmp(algorithm, "ridge") == 0 ||
			strcmp(algorithm, "lasso") == 0 ||
			strcmp(algorithm, "knn") == 0 ||
			strcmp(algorithm, "xgboost") == 0 ||
			strcmp(algorithm, "catboost") == 0)
		{
			/* Metal backend doesn't support this algorithm - skip GPU and use CPU */
			if (errstr)
				*errstr = ndb_gpu_strdup_or_null(psprintf("Metal backend: %s is unsupported (known limitations) - using CPU", algorithm));
			/* GPU mode: allow CPU fallback for Metal unsupported algorithms */
			/* AUTO/CPU mode: return false for CPU fallback */
			ndb_gpu_ensure_errstr_top(errstr);
			return false;
		}
	}
	
	ctx.device_id = neurondb_gpu_device;
	ctx.stream_handle = NULL;
	ctx.scratch_pool = NULL;
	ctx.memory_ctx = CurrentMemoryContext;


	memset(&model, 0, sizeof(MLGpuModel));
	model.ops = ops;
	model.backend_state = NULL;
	model.catalog_id = InvalidOid;
	model.model_name = pstrdup(model_name ? model_name : algorithm);
	if (model.model_name == NULL)
	{
		/* GPU mode: error if memory allocation fails */
		if (!NDB_COMPUTE_MODE_IS_CPU() && NDB_REQUIRE_GPU())
		{
			ereport(ERROR,
					(errcode(ERRCODE_OUT_OF_MEMORY),
					 errmsg("%s: GPU training failed - memory allocation error",
							algorithm ? algorithm : "unknown"),
					 errdetail("Failed to allocate memory for model name"),
					 errhint("Set compute_mode='auto' for automatic CPU fallback.")));
		}
		/* AUTO/CPU mode: return false for fallback */
		elog(WARNING,
			 "ndb_gpu_try_train_model: failed to allocate model_name");
		return false;
	}
	model.is_gpu_resident = true;
	model.gpu_ready = false;


	memset(&spec, 0, sizeof(MLGpuTrainSpec));
	spec.algorithm = algorithm;
	spec.project_name = project_name;
	spec.model_name = model_name;
	spec.training_table = training_table;
	spec.training_column = training_column;
	spec.feature_columns = feature_columns;
	spec.feature_count = feature_count;
	spec.hyperparameters = hyperparameters;
	spec.context = &ctx;
	spec.expected_features = -1;
	spec.expected_classes = -1;
	spec.feature_matrix = feature_matrix;
	spec.label_vector = label_vector;
	spec.sample_count = sample_count;
	spec.feature_dim = feature_dim;
	spec.class_count = class_count;

	ereport(DEBUG2,
			(errmsg("ndb_gpu_try_train_model: training spec initialized"),
			 errdetail("spec.feature_matrix=%p, spec.label_vector=%p, spec.sample_count=%d, spec.feature_dim=%d",
					   (void *) spec.feature_matrix, (void *) spec.label_vector, spec.sample_count, spec.feature_dim)));

	/* Skip ops->train path for linear_regression - use direct path instead */
	/* Also skip if feature_matrix is NULL - data needs to be loaded first */

	/*
	 * The direct algorithm-specific paths below will handle loading data from
	 * table
	 */
	ereport(DEBUG2,
			(errmsg("ndb_gpu_try_train_model: checking if should use ops->train path"),
			 errdetail("ops=%p, ops->train=%p, ops->serialize=%p, algorithm=%s, feature_matrix=%p",
					   (void *) ops,
					   ops ? (void *) ops->train : NULL,
					   ops ? (void *) ops->serialize : NULL,
					   algorithm ? algorithm : "NULL",
					   (void *) feature_matrix)));

	if (ops != NULL && ops->train != NULL && ops->serialize != NULL
		&& feature_matrix != NULL
		&& (is_unsupervised || label_vector != NULL)
		&& sample_count > 0 && feature_dim > 0
		&& (algorithm == NULL || (strcmp(algorithm, "linear_regression") != 0 && strcmp(algorithm, "logistic_regression") != 0 && strcmp(algorithm, "ridge") != 0 && strcmp(algorithm, "lasso") != 0)))
	{
		TimestampTz train_start;
		TimestampTz train_end;
		bool		ops_serialized;
		long		secs;
		int			usecs = 0;
		double		elapsed_ms;


		train_start = GetCurrentTimestamp();
		ops_trained = false;
		ops_serialized = false;
		secs = 0;

		ereport(DEBUG2,
				(errmsg("ndb_gpu_try_train_model: about to call ops->train"),
				 errdetail("model=%p, spec=%p, model.ops=%p", (void *) &model, (void *) &spec, (void *) model.ops)));

		/* Defensive: Wrap ops->train call in error handling */
		PG_TRY();
		{
			ereport(DEBUG2,
					(errmsg("ndb_gpu_try_train_model: inside PG_TRY, calling ops->train")));

			if (ops->train(&model, &spec, errstr))
			{

				model.gpu_ready = true;
				ops_trained = true;


				if (ops->serialize(&model, &payload, &metadata, errstr))
				{
					ops_serialized = true;
					trained = true;
				}
				else
				{
					/* Ensure caller gets an error string when ops->serialize fails */
					if (errstr && *errstr == NULL)
							*errstr = ndb_gpu_strdup_or_null(psprintf("%s: ops->serialize returned false without error", algorithm ? algorithm : "unknown"));
						ndb_gpu_ensure_errstr_top(errstr);
				}
			}
			else
			{
				/* Ensure caller gets an error string when ops->train fails without throwing */
				if (errstr && *errstr == NULL)
					*errstr = ndb_gpu_strdup_or_null(psprintf("%s: ops->train returned false without error", algorithm ? algorithm : "unknown"));
				ndb_gpu_ensure_errstr_top(errstr);
			}
		}
		PG_CATCH();
		{
			/* Catch any PostgreSQL-level errors from ops->train */
			ErrorData *edata = NULL;
			char *error_msg = NULL;
			
			/* In GPU mode, re-throw the error immediately - no fallback allowed */
			/* Don't try to capture error data here as it may be corrupted */
			if (!NDB_COMPUTE_MODE_IS_CPU() && NDB_REQUIRE_GPU())
			{
				/* Don't call ops->destroy here - memory contexts may have been reset during exception */
				model.backend_state = NULL;
				/* Re-throw the error for GPU mode - let it propagate with original details */
				PG_RE_THROW();
			}
			
			/* AUTO mode: log warning and fall back */
			elog(WARNING,
				 "%s: exception caught during ops->train, falling back to direct path",
				 algorithm ? algorithm : "unknown");
			
			/* Try to capture error details for logging */
			if (CurrentMemoryContext != ErrorContext)
			{
				edata = CopyErrorData();
				if (edata)
				{
					if (edata->detail && strlen(edata->detail) > 0)
						error_msg = pstrdup(edata->detail);
					else if (edata->message && strlen(edata->message) > 0)
						error_msg = pstrdup(edata->message);
				}
			}
			
			FlushErrorState();
			ops_trained = false;
			trained = false;
			ops_failed_with_exception = true;
				if (errstr && *errstr == NULL)
				{
					if (error_msg)
						*errstr = ndb_gpu_strdup_or_null(error_msg);
					else
						*errstr = ndb_gpu_strdup_or_null("Exception during ops->train");
				}
				ndb_gpu_ensure_errstr_top(errstr);
			/* Don't call ops->destroy here - memory contexts may have been reset during exception */
			/* Just NULL out the pointer - memory context cleanup will handle freeing */
			model.backend_state = NULL;
			
			if (edata)
				FreeErrorData(edata);
			if (error_msg)
				pfree(error_msg);
			/* Continue to direct path fallback in AUTO mode */
		}
		PG_END_TRY();

		train_end = GetCurrentTimestamp();
		TimestampDifference(train_start, train_end, &secs, &usecs);
		elapsed_ms = ((double) secs * 1000.0) + ((double) usecs / 1000.0);

		if (trained)
		{
			ndb_gpu_stats_record(true, elapsed_ms, 0.0, false);
		}
		else if (ops_trained || ops_serialized)
		{
			/* GPU training failed - determine fallback behavior */
			/* GPU mode + CUDA/ROCm: error out (GPU must work) */
			/* GPU mode + Metal: allow CPU fallback (Metal has limitations) */
			/* AUTO mode: always allow CPU fallback */
			/* CPU mode: never error, just return false for fallback */
			GPUBackend backend = neurondb_gpu_get_backend();
			bool is_metal_backend = (backend == GPU_BACKEND_METAL);
			bool allow_fallback = true;
			
			if (NDB_REQUIRE_GPU() && !is_metal_backend)
			{
				/* GPU mode with CUDA/ROCm: GPU must work, error out */
				allow_fallback = false;
			}
			
			if (!allow_fallback)
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("%s: GPU training failed - GPU mode requires GPU to be available",
								algorithm ? algorithm : "unknown"),
						 errdetail("GPU stage elapsed %.3f ms before failure",
								   elapsed_ms),
						 errhint("Set compute_mode='auto' for automatic CPU fallback.")));
			}
			/* AUTO/CPU mode OR Metal backend: fall back to CPU */
			ndb_gpu_stats_record(false, 0.0, elapsed_ms, true);
		}
	}

	if (!trained && !ops_failed_with_exception && algorithm != NULL
		&& strcmp(algorithm, "random_forest") == 0
		&& feature_matrix != NULL && label_vector != NULL
		&& sample_count > 0 && feature_dim > 0)
	{
		TimestampTz train_start = GetCurrentTimestamp();
		TimestampTz train_end;
		int			gpu_rc = ndb_gpu_rf_train(feature_matrix,
											  label_vector,
											  sample_count,
											  feature_dim,
											  class_count,
											  hyperparameters,
											  &payload,
											  &metadata,
											  errstr);
		long		secs = 0;
		int			usecs = 0;
		double		elapsed_ms;

		train_end = GetCurrentTimestamp();
		TimestampDifference(train_start, train_end, &secs, &usecs);
		elapsed_ms = ((double) secs * 1000.0) + ((double) usecs / 1000.0);

		if (gpu_rc == 0)
		{
			trained = true;
			ndb_gpu_stats_record(true, elapsed_ms, 0.0, false);
		}
		else
		{
			/* GPU training failed - determine fallback behavior */
			/* GPU mode + CUDA/ROCm: error out (GPU must work) */
			/* GPU mode + Metal: allow CPU fallback (Metal has limitations) */
			/* AUTO mode: always allow CPU fallback */
			GPUBackend backend = neurondb_gpu_get_backend();
			bool is_metal_backend = (backend == GPU_BACKEND_METAL);
			bool allow_fallback = true;
			
			if (NDB_REQUIRE_GPU() && !is_metal_backend)
			{
				/* GPU mode with CUDA/ROCm: GPU must work, error out */
				allow_fallback = false;
			}
			
			if (!allow_fallback)
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("random_forest: GPU training failed - GPU mode requires GPU to be available"),
						 errdetail("GPU attempt elapsed %.3f ms (%s)",
								   elapsed_ms,
								   (errstr && *errstr) ? *errstr : "no error"),
						 errhint("Set compute_mode='auto' for automatic CPU fallback.")));
			}
			/* AUTO mode OR Metal backend: fall back to CPU */
			ndb_gpu_stats_record(false, 0.0, elapsed_ms, true);
		}
	}

	if (!trained && algorithm != NULL
		&& strcmp(algorithm, "logistic_regression") == 0
		&& feature_matrix != NULL && label_vector != NULL
		&& sample_count > 0 && feature_dim > 0)
	{
		TimestampTz train_start = GetCurrentTimestamp();
		TimestampTz train_end;
		const ndb_gpu_backend *backend = ndb_gpu_get_active_backend();
		int			gpu_rc;
		long		secs = 0;
		int			usecs = 0;
		double		elapsed_ms;

		
		/* Set errstr if backend is NULL so we have a specific error message */
		if (backend == NULL)
		{
			if (errstr && *errstr == NULL)
				*errstr = ndb_gpu_strdup_or_null("logistic_regression: ndb_gpu_get_active_backend() returned NULL in direct path");
		}
		else if (backend->lr_train == NULL)
		{
			if (errstr && *errstr == NULL)
				*errstr = ndb_gpu_strdup_or_null(psprintf("logistic_regression: backend->lr_train is NULL (backend=%s)", backend->name ? backend->name : "unknown"));
		}

		/* GPU mode: ensure we're in GPU mode - no fallback allowed */
		if (NDB_REQUIRE_GPU() && (backend == NULL || backend->lr_train == NULL))
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("logistic_regression: GPU mode requires GPU backend but backend or lr_train is NULL"),
					 errdetail("backend=%p, lr_train=%p",
							   (void *) backend,
							   backend ? (void *) backend->lr_train : NULL),
					 errhint("GPU mode requires GPU to be available - cannot fall back to CPU")));
		}

		if (backend == NULL || backend->lr_train == NULL)
		{
			/* Ensure errstr is set before returning */
			if (errstr && *errstr == NULL)
			{
				if (backend == NULL)
					*errstr = ndb_gpu_strdup_or_null("logistic_regression: ndb_gpu_get_active_backend() returned NULL");
				else
					*errstr = ndb_gpu_strdup_or_null(psprintf("logistic_regression: backend->lr_train is NULL (backend=%s)", backend->name ? backend->name : "unknown"));
			}
			ndb_gpu_ensure_errstr_top(errstr);
			goto lr_fallback;
		}

		/* Defensive: Validate all parameters before calling CUDA function */
		if (feature_matrix == NULL)
		{
			if (NDB_REQUIRE_GPU())
			{
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("logistic_regression: feature_matrix is NULL - GPU mode requires valid input")));
			}
			if (errstr)
				*errstr = ndb_gpu_strdup_or_null("logistic_regression: feature_matrix is NULL");
			ndb_gpu_ensure_errstr_top(errstr);
			elog(WARNING,
				 "logistic_regression: feature_matrix is NULL, skipping GPU");
			goto lr_fallback;
		}

		if (label_vector == NULL)
		{
			if (NDB_REQUIRE_GPU())
			{
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("logistic_regression: label_vector is NULL - GPU mode requires valid input")));
			}
			if (errstr)
				*errstr = ndb_gpu_strdup_or_null("logistic_regression: label_vector is NULL");
			ndb_gpu_ensure_errstr_top(errstr);
			elog(WARNING,
				 "logistic_regression: label_vector is NULL, skipping GPU");
			goto lr_fallback;
		}

		if (sample_count <= 0 || sample_count > 10000000)
		{
			if (NDB_REQUIRE_GPU())
			{
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("logistic_regression: invalid sample_count %d - GPU mode requires valid input",
								sample_count)));
			}
			if (errstr)
				*errstr = ndb_gpu_strdup_or_null(psprintf("logistic_regression: invalid sample_count %d",
								   sample_count));
			ndb_gpu_ensure_errstr_top(errstr);
			elog(WARNING,
				 "logistic_regression: invalid sample_count %d, skipping GPU",
				 sample_count);
			goto lr_fallback;
		}

		if (feature_dim <= 0 || feature_dim > 10000)
		{
			if (NDB_REQUIRE_GPU())
			{
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("logistic_regression: invalid feature_dim %d - GPU mode requires valid input",
								feature_dim)));
			}
			if (errstr)
				*errstr = ndb_gpu_strdup_or_null(psprintf("logistic_regression: invalid feature_dim %d",
								   feature_dim));
			ndb_gpu_ensure_errstr_top(errstr);
			elog(WARNING,
				 "logistic_regression: invalid feature_dim %d, skipping GPU",
				 feature_dim);
			goto lr_fallback;
		}

		/* Defensive: Wrap CUDA call in error handling */
		PG_TRY();
		{
			gpu_rc = ndb_gpu_lr_train(feature_matrix,
									  label_vector,
									  sample_count,
									  feature_dim,
									  hyperparameters,
									  &payload,
									  &metadata,
									  errstr);
		}
		PG_CATCH();
		{
			/* Catch any PostgreSQL-level errors from CUDA code */
			/* CRITICAL: After exception, DO NOT free payload/metadata - memory context may be corrupted */
			ErrorData *edata;
			char *error_msg;
			
			if (CurrentMemoryContext != ErrorContext)
			{
				edata = CopyErrorData();
				if (edata)
				{
					/* Prefer detail message over main message for more specific errors */
					if (edata->detail && strlen(edata->detail) > 0)
					{
						error_msg = pstrdup(edata->detail);
					}
					else if (edata->message && strlen(edata->message) > 0)
					{
						error_msg = pstrdup(edata->message);
					}
					/* Set errstr so it's available after re-throw */
					if (errstr && *errstr == NULL && error_msg != NULL)
						*errstr = ndb_gpu_strdup_or_null(error_msg);
				}
			}
			
			/* GPU mode: re-raise error, no fallback */
			if (NDB_REQUIRE_GPU())
			{
				/* Skip payload/metadata cleanup - memory context may be corrupted */
				payload = NULL;
				metadata = NULL;
				if (edata)
					FreeErrorData(edata);
				if (error_msg)
					pfree(error_msg);
				PG_RE_THROW();
			}
			/* AUTO mode: fall back to CPU */
			elog(WARNING,
				 "logistic_regression: exception caught during GPU training, falling back to CPU");
			gpu_rc = -1;
			if (errstr && *errstr == NULL)
			{
				if (error_msg)
					*errstr = ndb_gpu_strdup_or_null(error_msg);
				else
					*errstr = ndb_gpu_strdup_or_null("Exception during GPU training");
			}
			ndb_gpu_ensure_errstr_top(errstr);
			/* Skip payload/metadata cleanup - memory context may be corrupted */
			payload = NULL;
			metadata = NULL;
			if (edata)
				FreeErrorData(edata);
			if (error_msg)
				pfree(error_msg);
			FlushErrorState();
		}
		PG_END_TRY();
		train_end = GetCurrentTimestamp();
		TimestampDifference(train_start, train_end, &secs, &usecs);
		elapsed_ms = ((double) secs * 1000.0) + ((double) usecs / 1000.0);

		if (gpu_rc == 0)
		{
			trained = true;
			ndb_gpu_stats_record(true, elapsed_ms, 0.0, false);
			if (metadata != NULL)
			{
				char	   *meta_txt = DatumGetCString(
													   DirectFunctionCall1(jsonb_out,
																		   JsonbPGetDatum(metadata)));

				pfree(meta_txt);
			}
			else
			{
				elog(WARNING,
					 "gpu_model_bridge: LR direct path "
					 "metadata is NULL!");
			}
			/* GPU training succeeded */
		}
		else
		{
			/* GPU training failed - determine fallback behavior */
			/* GPU mode + CUDA/ROCm: error out (GPU must work) */
			/* GPU mode + Metal: allow CPU fallback (Metal has limitations) */
			/* AUTO mode: always allow CPU fallback */
			GPUBackend backend = neurondb_gpu_get_backend();
			bool is_metal_backend = (backend == GPU_BACKEND_METAL);
			bool allow_fallback = true;
			
			if (NDB_REQUIRE_GPU() && !is_metal_backend)
			{
				/* GPU mode with CUDA/ROCm: GPU must work, error out */
				allow_fallback = false;
			}
			
			if (!allow_fallback)
			{
				/* Capture error message before raising ERROR */
				/* Also ensure errstr is set so it's available to the caller */
				char *error_detail = NULL;
				if (errstr && *errstr)
				{
					error_detail = ndb_gpu_strdup_or_null(*errstr);
				}
				else
				{
					/* If errstr is not set, try to get a meaningful error message */
					error_detail = ndb_gpu_strdup_or_null(psprintf("ndb_gpu_lr_train returned %d (gpu_rc=%d) - no error details available", gpu_rc, gpu_rc));
					/* Set errstr so it's available to the caller */
					if (errstr && *errstr == NULL)
						*errstr = ndb_gpu_strdup_or_null(error_detail);
					ndb_gpu_ensure_errstr_top(errstr);
				}
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("logistic_regression: GPU training failed - GPU mode requires GPU to be available"),
						 errdetail("GPU attempt elapsed %.3f ms. Error: %s",
								   elapsed_ms,
								   error_detail),
						 errhint("Set compute_mode='auto' for automatic CPU fallback.")));
				/* Should not reach here, but included for safety */
				if (error_detail)
					pfree(error_detail);
			}
			/* AUTO/CPU mode OR Metal backend: fall back to CPU */
			ndb_gpu_stats_record(false, 0.0, elapsed_ms, true);
		}
lr_fallback:;
	}

	if (!trained && !ops_failed_with_exception && algorithm != NULL
		&& strcmp(algorithm, "linear_regression") == 0
		&& feature_matrix != NULL && label_vector != NULL
		&& sample_count > 0 && feature_dim > 0)
	{
		TimestampTz train_start = GetCurrentTimestamp();
		TimestampTz train_end;
		const ndb_gpu_backend *backend = ndb_gpu_get_active_backend();
		int			gpu_rc;
		long		secs = 0;
		int			usecs = 0;
		double		elapsed_ms;

		ereport(DEBUG2,
				(errmsg("linear_regression: attempting direct GPU training"),
				 errdetail("backend=%s, linreg_train=%p, samples=%d, dim=%d, feature_matrix=%p, label_vector=%p",
						   backend ? (backend->name ? backend->name : "unknown")
						   : "NULL",
						   backend ? (void *) backend->linreg_train : NULL,
						   sample_count,
						   feature_dim,
						   (void *) feature_matrix,
						   (void *) label_vector)));

		/* GPU mode: ensure we're in GPU mode - no fallback allowed */
		if (NDB_REQUIRE_GPU() && (backend == NULL || backend->linreg_train == NULL))
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("linear_regression: GPU mode requires GPU backend but backend or linreg_train is NULL"),
					 errdetail("backend=%p, linreg_train=%p",
							   (void *) backend,
							   backend ? (void *) backend->linreg_train : NULL),
					 errhint("GPU mode requires GPU to be available - cannot fall back to CPU")));
		}

		if (backend == NULL || backend->linreg_train == NULL)
		{
			goto linreg_fallback;
		}

		/* Defensive: Validate all parameters before calling GPU function */
		if (feature_matrix == NULL)
		{
			if (NDB_REQUIRE_GPU())
			{
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("linear_regression: feature_matrix is NULL - GPU mode requires valid input")));
			}
			if (errstr)
				*errstr = ndb_gpu_strdup_or_null("linear_regression: feature_matrix is NULL");
			ndb_gpu_ensure_errstr_top(errstr);
			elog(WARNING,
				 "linear_regression: feature_matrix is NULL, skipping GPU");
			goto linreg_fallback;
		}

		if (label_vector == NULL)
		{
			if (NDB_REQUIRE_GPU())
			{
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("linear_regression: label_vector is NULL - GPU mode requires valid input")));
			}
			if (errstr)
				*errstr = ndb_gpu_strdup_or_null("linear_regression: label_vector is NULL");
			ndb_gpu_ensure_errstr_top(errstr);
			elog(WARNING,
				 "linear_regression: label_vector is NULL, skipping GPU");
			goto linreg_fallback;
		}

		if (sample_count <= 0 || sample_count > 10000000)
		{
			if (NDB_REQUIRE_GPU())
			{
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("linear_regression: invalid sample_count %d - GPU mode requires valid input",
								sample_count)));
			}
			if (errstr)
				*errstr = ndb_gpu_strdup_or_null(psprintf("linear_regression: invalid sample_count %d",
								   sample_count));
			ndb_gpu_ensure_errstr_top(errstr);
			elog(WARNING,
				 "linear_regression: invalid sample_count %d, skipping GPU",
				 sample_count);
			goto linreg_fallback;
		}

		if (feature_dim <= 0 || feature_dim > 10000)
		{
			if (NDB_REQUIRE_GPU())
			{
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("linear_regression: invalid feature_dim %d - GPU mode requires valid input",
								feature_dim)));
			}
			if (errstr)
				*errstr = ndb_gpu_strdup_or_null(psprintf("linear_regression: invalid feature_dim %d",
								   feature_dim));
			ndb_gpu_ensure_errstr_top(errstr);
			elog(WARNING,
				 "linear_regression: invalid feature_dim %d, skipping GPU",
				 feature_dim);
			goto linreg_fallback;
		}

		/* Defensive: Wrap GPU call in error handling */
		ereport(DEBUG2,
				(errmsg("linear_regression: about to call ndb_gpu_linreg_train"),
				 errdetail("feature_matrix=%p, label_vector=%p, sample_count=%d, feature_dim=%d",
						   (void *) feature_matrix,
						   (void *) label_vector,
						   sample_count,
						   feature_dim)));

		PG_TRY();
		{
			ereport(DEBUG2,
					(errmsg("linear_regression: inside PG_TRY, calling ndb_gpu_linreg_train")));

			gpu_rc = ndb_gpu_linreg_train(feature_matrix,
										  label_vector,
										  sample_count,
										  feature_dim,
										  hyperparameters,
										  &payload,
										  &metadata,
										  errstr);

			ereport(DEBUG2,
					(errmsg("linear_regression: ndb_gpu_linreg_train returned"),
					 errdetail("gpu_rc=%d, payload=%p, metadata=%p",
							   gpu_rc,
							   (void *) payload,
							   (void *) metadata)));
		}
		PG_CATCH();
		{
			/* Catch any PostgreSQL-level errors from GPU code */
			/* CRITICAL: After exception, DO NOT free payload/metadata - memory context may be corrupted */
			if (NDB_COMPUTE_MODE_IS_CPU())
			{
				elog(WARNING,
					 "linear_regression: exception caught during GPU training attempt in CPU mode, falling back to CPU");
				FlushErrorState();
				gpu_rc = -1;
				if (errstr && *errstr == NULL)
				{
					*errstr = ndb_gpu_strdup_or_null("Exception during GPU training (CPU mode)");
					ndb_gpu_ensure_errstr_top(errstr);
				}
				/* Skip payload/metadata cleanup - memory context may be corrupted */
				payload = NULL;
				metadata = NULL;
			}
			/* GPU mode: re-raise error, no fallback */
			else if (!NDB_COMPUTE_MODE_IS_CPU() && NDB_REQUIRE_GPU())
			{
				/* Skip payload/metadata cleanup - memory context may be corrupted */
				payload = NULL;
				metadata = NULL;
				PG_RE_THROW();
			}
			/* AUTO mode: fall back to CPU */
			elog(WARNING,
				 "linear_regression: exception caught during GPU training, falling back to CPU");
			gpu_rc = -1;
			if (errstr && *errstr == NULL)
			{
				*errstr = ndb_gpu_strdup_or_null("Exception during GPU training");
				ndb_gpu_ensure_errstr_top(errstr);
			}
			/* Skip payload/metadata cleanup - memory context may be corrupted */
			payload = NULL;
			metadata = NULL;
			FlushErrorState();
		}
		PG_END_TRY();
		train_end = GetCurrentTimestamp();
		TimestampDifference(train_start, train_end, &secs, &usecs);
		elapsed_ms = ((double) secs * 1000.0) + ((double) usecs / 1000.0);

		if (gpu_rc == 0)
		{
			char *meta_txt = NULL;

			trained = true;
			ndb_gpu_stats_record(true, elapsed_ms, 0.0, false);

			/* Populate result structure with payload and metadata */
			if (result != NULL)
			{
				result->spec.model_data = payload;
				payload = NULL; /* Transfer ownership to result */

				if (metadata != NULL)
				{
					/* Set metrics for model registration */
					result->spec.metrics = metadata;
					result->metadata = metadata;
					/* Don't transfer ownership - metadata will be copied during registration */

					meta_txt = DatumGetCString(
											   DirectFunctionCall1(jsonb_out,
																   JsonbPGetDatum(metadata)));

					pfree(meta_txt);
				}
				else
				{
					/* metadata is NULL - pack_model should have created it */
					/* This is a critical error in GPU mode - model cannot be registered without metrics */
					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg("neurondb: GPU training succeeded but metrics JSON is NULL"),
							 errdetail("pack_model should have created metrics JSON with training_backend=1 and storage=gpu"),
							 errhint("This indicates a bug in ndb_cuda_linreg_pack_model - metrics JSON creation failed or was skipped")));
				}
			}
			else
			{
				/* result is NULL, free payload and metadata */
				if (payload != NULL)
				{
					pfree(payload);
					payload = NULL;
				}
				if (metadata != NULL)
				{
					pfree(metadata);
					metadata = NULL;
				}
			}

		}
		else
		{
			/* GPU mode: error if GPU training fails, unless Metal requests CPU fallback */
			/* CPU mode: never error, just return false for fallback */
			if (!NDB_COMPUTE_MODE_IS_CPU() && NDB_REQUIRE_GPU())
			{
				/* Capture error message before raising ERROR */
				char *error_detail = NULL;
				bool allow_cpu_fallback = false;
				
				if (errstr && *errstr)
				{
					error_detail = ndb_gpu_strdup_or_null(*errstr);
					/* Check if Metal backend requested CPU fallback */
					if (error_detail && (strstr(error_detail, "using CPU fallback") != NULL ||
										 strstr(error_detail, "Metal") != NULL))
					{
						allow_cpu_fallback = true;
					}
				}
				else
				{
					error_detail = pstrdup("GPU training failed - no error details available");
				}
				
				if (allow_cpu_fallback)
				{
					/* Metal backend requested CPU fallback - allow it without logging INFO message */
					/* Don't free error_detail here - it's a copy and will be cleaned up by memory context */
					/* Don't free *errstr either - it's managed by the caller */
					/* Just return false to allow CPU fallback */
					return false;
				}
				else
				{
					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg("linear_regression: GPU training failed - GPU mode requires GPU to be available"),
							 errdetail("GPU attempt elapsed %.3f ms. Error: %s",
									   elapsed_ms,
									   error_detail),
							 errhint("Set compute_mode='auto' for automatic CPU fallback.")));
					/* Should not reach here, but included for safety */
					if (error_detail)
						pfree(error_detail);
				}
			}
			/* AUTO mode: fall back to CPU */
			ndb_gpu_stats_record(false, 0.0, elapsed_ms, true);
		}
linreg_fallback:;
	}

	if (!trained && !ops_trained && algorithm != NULL
		&& strcmp(algorithm, "decision_tree") == 0
		&& feature_matrix != NULL && label_vector != NULL
		&& sample_count > 0 && feature_dim > 0)
	{
		TimestampTz train_start = GetCurrentTimestamp();
		TimestampTz train_end;
		int			gpu_rc = ndb_gpu_dt_train(feature_matrix,
											  label_vector,
											  sample_count,
											  feature_dim,
											  hyperparameters,
											  &payload,
											  &metadata,
											  errstr);
		long		secs = 0;
		int			usecs = 0;
		double		elapsed_ms;

		train_end = GetCurrentTimestamp();
		TimestampDifference(train_start, train_end, &secs, &usecs);
		elapsed_ms = ((double) secs * 1000.0) + ((double) usecs / 1000.0);

		if (gpu_rc == 0)
		{
			trained = true;
			ndb_gpu_stats_record(true, elapsed_ms, 0.0, false);
		}
		else
		{
			/* GPU mode: error if GPU training fails */
			/* CPU mode: never error, just return false for fallback */
			if (!NDB_COMPUTE_MODE_IS_CPU() && NDB_REQUIRE_GPU())
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("decision_tree: GPU training failed - GPU mode requires GPU to be available"),
						 errdetail("GPU attempt elapsed %.3f ms (%s)",
								   elapsed_ms,
								   (errstr && *errstr) ? *errstr : "no error"),
						 errhint("Set compute_mode='auto' for automatic CPU fallback.")));
			}
			/* AUTO mode: fall back to CPU */
			ndb_gpu_stats_record(false, 0.0, elapsed_ms, true);
		}
	}

	if (!trained && !ops_trained && algorithm != NULL && strcmp(algorithm, "ridge") == 0
		&& feature_matrix != NULL && label_vector != NULL
		&& sample_count > 0 && feature_dim > 0)
	{
		TimestampTz train_start = GetCurrentTimestamp();
		TimestampTz train_end;
		int			gpu_rc = ndb_gpu_ridge_train(feature_matrix,
												 label_vector,
												 sample_count,
												 feature_dim,
												 hyperparameters,
												 &payload,
												 &metadata,
												 errstr);
		long		secs = 0;
		int			usecs = 0;
		double		elapsed_ms;

		train_end = GetCurrentTimestamp();
		TimestampDifference(train_start, train_end, &secs, &usecs);
		elapsed_ms = ((double) secs * 1000.0) + ((double) usecs / 1000.0);

		if (gpu_rc == 0)
		{
			trained = true;
			ndb_gpu_stats_record(true, elapsed_ms, 0.0, false);
			/* Populate result structure with payload and metadata */
			if (result != NULL)
			{
				result->spec.model_data = payload;
				payload = NULL; /* Transfer ownership to result */

				if (metadata != NULL)
				{
					result->spec.metrics = metadata;
					result->metadata = metadata;
				}
			}
		}
		else
		{
			/* GPU mode: error if GPU training fails */
			/* CPU mode: never error, just return false for fallback */
			if (!NDB_COMPUTE_MODE_IS_CPU() && NDB_REQUIRE_GPU())
			{
				/* Capture error message safely before raising ERROR */
				char *error_detail = NULL;
				if (errstr && *errstr)
				{
					/* Copy error string to ensure it persists */
					error_detail = pstrdup(*errstr);
				}
				else
				{
					error_detail = psprintf("ndb_gpu_ridge_train returned %d (elapsed %.3f ms) - no error details available", gpu_rc, elapsed_ms);
				}
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("ridge: GPU training failed - GPU mode requires GPU to be available"),
						 errdetail("GPU attempt elapsed %.3f ms. Return code: %d. Error: %s",
								   elapsed_ms, gpu_rc, error_detail ? error_detail : "unknown error"),
						 errhint("Set compute_mode='auto' for automatic CPU fallback.")));
				/* Should not reach here, but included for safety */
				if (error_detail)
					pfree(error_detail);
			}
			/* AUTO mode: fall back to CPU */
			/* Ensure error string is set for caller */
			if (errstr && *errstr == NULL)
			{
				*errstr = psprintf("ndb_gpu_ridge_train returned %d (elapsed %.3f ms) - no specific error available", gpu_rc, elapsed_ms);
			}
			ndb_gpu_ensure_errstr_top(errstr);
			ndb_gpu_stats_record(false, 0.0, elapsed_ms, true);
		}
	}

	if (!trained && !ops_trained && algorithm != NULL && strcmp(algorithm, "lasso") == 0
		&& feature_matrix != NULL && label_vector != NULL
		&& sample_count > 0 && feature_dim > 0)
	{
		TimestampTz train_start = GetCurrentTimestamp();
		TimestampTz train_end;
		int			gpu_rc = ndb_gpu_lasso_train(feature_matrix,
												 label_vector,
												 sample_count,
												 feature_dim,
												 hyperparameters,
												 &payload,
												 &metadata,
												 errstr);
		long		secs = 0;
		int			usecs = 0;
		double		elapsed_ms;

		train_end = GetCurrentTimestamp();
		TimestampDifference(train_start, train_end, &secs, &usecs);
		elapsed_ms = ((double) secs * 1000.0) + ((double) usecs / 1000.0);

		if (gpu_rc == 0)
		{
			trained = true;
			ndb_gpu_stats_record(true, elapsed_ms, 0.0, false);
		}
		else
		{
			/* GPU mode: error if GPU training fails */
			/* CPU mode: never error, just return false for fallback */
			if (!NDB_COMPUTE_MODE_IS_CPU() && NDB_REQUIRE_GPU())
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("lasso: GPU training failed - GPU mode requires GPU to be available"),
						 errdetail("GPU attempt elapsed %.3f ms (%s)",
								   elapsed_ms,
								   (errstr && *errstr) ? *errstr : "no error"),
						 errhint("Set compute_mode='auto' for automatic CPU fallback.")));
			}
			/* AUTO mode: fall back to CPU */
			ndb_gpu_stats_record(false, 0.0, elapsed_ms, true);
		}
	}

	if (trained && result != NULL)
	{
		result->spec.algorithm = ndb_gpu_strdup_or_null(algorithm);
		result->spec.model_type = NULL;
		result->spec.training_table =
			ndb_gpu_strdup_or_null(training_table);
		result->spec.training_column =
			ndb_gpu_strdup_or_null(training_column);
		result->spec.project_name =
			ndb_gpu_strdup_or_null(project_name);
		result->spec.model_name = ndb_gpu_strdup_or_null(model_name);

		/*
		 * Note: parameters are handled by caller, don't set here to avoid
		 * ownership confusion
		 */

		/* Ensure metrics Jsonb is in the correct memory context */
		if (metadata != NULL)
		{
			Jsonb	   *metrics_copy = (Jsonb *) PG_DETOAST_DATUM_COPY(
																	   PointerGetDatum(metadata));

			result->spec.metrics = metrics_copy;
			result->metadata = metrics_copy;
			result->metrics = metrics_copy;
		}
		else
		{
			result->spec.metrics = NULL;
			result->metadata = NULL;
			result->metrics = NULL;
			elog(WARNING,
				 "gpu_model_bridge: metadata is NULL, cannot "
				 "set result->spec.metrics!");
		}

		ereport(DEBUG2, (errmsg("gpu_model_bridge: about to set model_data, result->spec.model_data=%p, payload=%p", (void *) result->spec.model_data, (void *) payload)));
		/* Only set model_data if it's not already set (e.g., by direct path) */
		if (result->spec.model_data == NULL)
		{
			/*
			 * IMPORTANT: Detach returned payload from any GPU backend state.
			 *
			 * Some GPU backends may allocate the returned model payload in memory
			 * tied to backend_state and/or free it during ops->destroy().  The
			 * payload must survive beyond ops->destroy() because the caller will
			 * persist/serialize it and later free it via ndb_gpu_free_train_result().
			 *
			 * So always take a deep copy of the varlena payload into PostgreSQL
			 * managed memory before we destroy backend_state.
			 */
			bytea *payload_copy = NULL;
			if (payload != NULL)
				payload_copy = (bytea *) PG_DETOAST_DATUM_COPY(PointerGetDatum(payload));

			ereport(DEBUG2, (errmsg("gpu_model_bridge: setting model_data from payload")));
			result->spec.model_data = payload_copy;
			result->payload = payload_copy;
			ereport(DEBUG2, (errmsg("gpu_model_bridge: model_data set successfully")));
		}
		else
		{
			ereport(DEBUG2, (errmsg("gpu_model_bridge: model_data already set, just setting payload pointer")));
			/* model_data already set by direct path, just set payload pointer */
			result->payload = result->spec.model_data;
			/* Free the payload we received since we're not using it */
			if (payload != NULL)
			{
				ereport(DEBUG2, (errmsg("gpu_model_bridge: freeing unused payload")));
				pfree(payload);
				ereport(DEBUG2, (errmsg("gpu_model_bridge: unused payload freed")));
			}
			ereport(DEBUG2, (errmsg("gpu_model_bridge: payload pointer set")));
		}
		ereport(DEBUG2, (errmsg("gpu_model_bridge: model_data assignment completed")));
	}

	/*
	 * DO NOT call ops->destroy here!
	 * The model->backend_state owns the payload memory that result->spec.model_data points to.
	 * If we destroy the backend_state now, we'll free the payload, but the caller (neurondb_train)
	 * still needs to write spec.model_data to the catalog via ml_catalog_register_model.
	 * The caller is responsible for cleanup after the catalog write completes.
	 */
	ereport(DEBUG2, (errmsg("gpu_model_bridge: skipping ops->destroy to preserve model_data lifetime")));

	ereport(DEBUG2, (errmsg("gpu_model_bridge: about to check if (!trained), trained=%d", trained)));
	if (!trained)
	{
		/* If errstr is not set and we have an algorithm, set a default error message */
		/* This ensures the caller always gets a meaningful error message */
		/* Copy error string to current memory context to ensure it persists */
		if (errstr)
		{
			if (*errstr == NULL)
			{
				if (algorithm != NULL)
				{
					*errstr = psprintf("ndb_gpu_try_train_model: GPU training failed for algorithm '%s' - no specific error available. Check GPU availability, backend registration, and that the algorithm is supported.", algorithm);
				}
				else
				{
					*errstr = pstrdup("ndb_gpu_try_train_model: GPU training failed - no specific error available. Check GPU availability and backend registration.");
				}
			}
			else
			{
				/* Copy error string to current memory context to ensure it persists */
				char *old_err = *errstr;
				*errstr = pstrdup(old_err);
				if (old_err)
					pfree(old_err);
			}
		}
		/* GPU mode: error if GPU training failed */
		if (!NDB_COMPUTE_MODE_IS_CPU() && NDB_REQUIRE_GPU())
		{
			char *error_detail = NULL;
			if (errstr && *errstr)
				error_detail = pstrdup(*errstr);
			else
				error_detail = pstrdup("GPU training failed - no specific error available");
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("%s: GPU training failed - GPU mode requires GPU to be available",
							algorithm ? algorithm : "unknown"),
					 errdetail("%s", error_detail),
					 errhint("Set compute_mode='auto' for automatic CPU fallback.")));
			/* Should not reach here, but included for safety */
			if (error_detail)
				pfree(error_detail);
		}
		/* AUTO/CPU mode: Do not free payload/metadata here - they are either NULL or owned by GPU backend */
		/* The GPU backend is responsible for cleaning up its own allocations */
		/* Do not call ndb_gpu_free_train_result since training never completed successfully */
		/* The result structure was never populated, so nothing to free */
	}
	else
	{
		ereport(DEBUG2, (errmsg("gpu_model_bridge: trained is true, skipping cleanup")));
	}

	ereport(DEBUG2, (errmsg("gpu_model_bridge: about to return, trained=%d", trained)));
	ereport(DEBUG2, (errmsg("gpu_model_bridge: CurrentMemoryContext=%p", (void *) CurrentMemoryContext)));
	ereport(DEBUG2, (errmsg("gpu_model_bridge: result=%p, result->spec.model_data=%p", (void *) result, result ? (void *) result->spec.model_data : NULL)));

	/*
	 * Store return value in volatile to ensure it's properly stored before
	 * return
	 */
	retval = trained;
	ereport(DEBUG2, (errmsg("gpu_model_bridge: retval=%d, about to execute return", retval)));

	/* Force compiler to not optimize away the return value */
__asm__ __volatile__("":::"memory");
	ereport(DEBUG2, (errmsg("gpu_model_bridge: memory barrier complete, executing return")));

	return (bool) retval;
}

void
ndb_gpu_free_train_result(MLGpuTrainResult *result)
{
	char *tmp = NULL;

	if (result == NULL)
		return;

	if (result->spec.algorithm)
	{
		tmp = (char *) result->spec.algorithm;
		pfree(tmp);
		result->spec.algorithm = NULL;
	}
	if (result->spec.training_table)
	{
		tmp = (char *) result->spec.training_table;
		pfree(tmp);
		result->spec.training_table = NULL;
	}
	if (result->spec.training_column)
	{
		tmp = (char *) result->spec.training_column;
		pfree(tmp);
		result->spec.training_column = NULL;
	}
	if (result->spec.project_name)
	{
		tmp = (char *) result->spec.project_name;
		pfree(tmp);
		result->spec.project_name = NULL;
	}
	if (result->spec.model_name)
	{
		tmp = (char *) result->spec.model_name;
		pfree(tmp);
		result->spec.model_name = NULL;
	}
	/* 
	 * DO NOT free result->spec.model_data here!
	 * After ml_catalog_register_model writes the payload to the catalog,
	 * the payload memory is either:
	 * 1. Still owned by the GPU backend_state (which will be freed later), OR
	 * 2. Already written to the catalog and no longer needed.
	 * Either way, freeing it here causes a double-free crash.
	 */
	if (result->spec.metrics)
		pfree(result->spec.metrics);
	if (result->metadata && result->metadata != result->spec.metrics)
		pfree(result->metadata);
	if (result->metrics && result->metrics != result->spec.metrics)
		pfree(result->metrics);

	memset(result, 0, sizeof(MLGpuTrainResult));
}
