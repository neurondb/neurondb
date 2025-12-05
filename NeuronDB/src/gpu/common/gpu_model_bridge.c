/*-------------------------------------------------------------------------
 *
 * gpu_model_bridge.c
 *    SQL ML entry points bridge.
 *
 * This module connects SQL ML entry points with the model registry,
 * providing helper routines for training and prediction.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
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

/* Forward declarations for GPU training functions */
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
#include <string.h>

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
	return pstrdup(src);
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

	/* Initialize all local variables to safe defaults */
	memset(&model, 0, sizeof(MLGpuModel));
	memset(&spec, 0, sizeof(MLGpuTrainSpec));
	memset(&ctx, 0, sizeof(MLGpuContext));

	if (errstr)
		*errstr = NULL;
	if (result)
		ndb_gpu_init_train_result(result);

	/* CPU mode: never attempt GPU code */
	if (NDB_COMPUTE_MODE_IS_CPU())
	{
		return false;
	}

	/* Early return if feature_matrix is NULL - data needs to be loaded first */

	/*
	 * The caller (neurondb_train) will load data and call again, or fall back
	 * to CPU
	 */
	if (feature_matrix == NULL || label_vector == NULL || sample_count <= 0 || feature_dim <= 0)
	{
		return false;
	}

	if (!neurondb_gpu_is_available())
	{
		return false;
	}

	ops = ndb_gpu_lookup_model_ops(algorithm);
	if (ops == NULL || ops->train == NULL || ops->serialize == NULL)
	{
		return false;
	}

	memset(&ctx, 0, sizeof(MLGpuContext));
	ctx.backend = ndb_gpu_get_active_backend();
	if (ctx.backend == NULL)
	{
		return false;
	}

	ereport(DEBUG1,
			(errmsg("ndb_gpu_try_train_model: active backend obtained"),
			 errdetail("backend_name=%s", ctx.backend->name ? ctx.backend->name : "NULL")));
	ereport(DEBUG1,
			(errmsg("ndb_gpu_try_train_model: setting up context"),
			 errdetail("backend_name=%s, device_id=%d", ctx.backend->name ? ctx.backend->name : "NULL", neurondb_gpu_device)));

	ctx.backend_name = (ctx.backend->name) ? ctx.backend->name : "unknown";
	ctx.device_id = neurondb_gpu_device;
	ctx.stream_handle = NULL;
	ctx.scratch_pool = NULL;
	ctx.memory_ctx = CurrentMemoryContext;

	ereport(DEBUG1,
			(errmsg("ndb_gpu_try_train_model: initializing model structure")));

	memset(&model, 0, sizeof(MLGpuModel));
	model.ops = ops;
	model.backend_state = NULL;
	model.catalog_id = InvalidOid;
	model.model_name = pstrdup(model_name ? model_name : algorithm);
	if (model.model_name == NULL)
	{
		elog(WARNING,
			 "ndb_gpu_try_train_model: failed to allocate model_name");
		return false;
	}
	model.is_gpu_resident = true;
	model.gpu_ready = false;

	ereport(DEBUG1,
			(errmsg("ndb_gpu_try_train_model: initializing training spec")));

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
		&& feature_matrix != NULL && label_vector != NULL
		&& sample_count > 0 && feature_dim > 0
		&& (algorithm == NULL || strcmp(algorithm, "linear_regression") != 0))
	{
		TimestampTz train_start;
		TimestampTz train_end;
		bool		ops_trained;
		bool		ops_serialized;
		long		secs;
		int			usecs = 0;
		double		elapsed_ms;

		ereport(DEBUG1,
				(errmsg("ndb_gpu_try_train_model: using ops->train path"),
				 errdetail("algorithm=%s", algorithm ? algorithm : "NULL")));

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
				ereport(DEBUG1,
						(errmsg("ndb_gpu_try_train_model: ops->train returned true")));

				model.gpu_ready = true;
				ops_trained = true;

				ereport(DEBUG1,
						(errmsg("ndb_gpu_try_train_model: about to call ops->serialize")));

				if (ops->serialize(&model, &payload, &metadata, errstr))
				{
					ops_serialized = true;
					trained = true;
				}
				else
				{
					ereport(DEBUG1,
							(errmsg("%s: GPU serialize failed: %s",
									algorithm ? algorithm : "unknown",
									(errstr && *errstr) ? *errstr : "no error message")));
				}
			}
			else
			{
				ereport(DEBUG1,
						(errmsg("%s: GPU train failed: %s",
								algorithm ? algorithm : "unknown",
								(errstr && *errstr) ? *errstr : "no error message")));
			}
		}
		PG_CATCH();
		{
			/* Catch any PostgreSQL-level errors from ops->train */
			elog(WARNING,
				 "%s: exception caught during ops->train, falling back to direct path or CPU",
				 algorithm ? algorithm : "unknown");
			ops_trained = false;
			trained = false;
			if (errstr && *errstr == NULL)
				*errstr = pstrdup("Exception during ops->train");
			PG_RE_THROW();
		}
		PG_END_TRY();

		train_end = GetCurrentTimestamp();
		TimestampDifference(train_start, train_end, &secs, &usecs);
		elapsed_ms = ((double) secs * 1000.0) + ((double) usecs / 1000.0);

		if (trained)
		{
			ndb_gpu_stats_record(true, elapsed_ms, 0.0, false);
			ereport(DEBUG1,
					(errmsg("%s: GPU training succeeded",
							algorithm ? algorithm : "unknown"),
					 errdetail(
							   "Elapsed %.3f ms on backend %s",
							   elapsed_ms,
							   ctx.backend_name
							   ? ctx.backend_name
							   : "unknown"),
					 errhidestmt(true)));
		}
		else if (ops_trained || ops_serialized)
		{
			/* GPU mode: error if GPU training fails */
			/* CPU mode: never error, just return false for fallback */
			if (!NDB_COMPUTE_MODE_IS_CPU() && NDB_REQUIRE_GPU())
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("%s: GPU training failed - GPU mode requires GPU to be available",
								algorithm ? algorithm : "unknown"),
						 errdetail("GPU stage elapsed %.3f ms before failure",
								   elapsed_ms),
						 errhint("Set compute_mode='auto' for automatic CPU fallback.")));
			}
			/* AUTO mode: fall back to CPU */
			ndb_gpu_stats_record(false, 0.0, elapsed_ms, true);
			ereport(DEBUG1,
					(errmsg("%s: GPU training fell back to CPU",
							algorithm ? algorithm : "unknown"),
					 errdetail("GPU stage elapsed %.3f ms "
							   "before fallback",
							   elapsed_ms),
					 errhidestmt(true)));
		}
	}

	if (!trained && algorithm != NULL
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
			ereport(DEBUG1,
					(errmsg("random_forest: GPU training succeeded "
							"(direct)"),
					 errdetail(
							   "Elapsed %.3f ms on backend %s",
							   elapsed_ms,
							   ctx.backend_name
							   ? ctx.backend_name
							   : "unknown"),
					 errhidestmt(true)));
		}
		else
		{
			/* GPU mode: error if GPU training fails */
			/* CPU mode: never error, just return false for fallback */
			if (!NDB_COMPUTE_MODE_IS_CPU() && NDB_REQUIRE_GPU())
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("random_forest: GPU training failed - GPU mode requires GPU to be available"),
						 errdetail("GPU attempt elapsed %.3f ms (%s)",
								   elapsed_ms,
								   (errstr && *errstr) ? *errstr : "no error"),
						 errhint("Set compute_mode='auto' for automatic CPU fallback.")));
			}
			/* AUTO mode: fall back to CPU */
			ndb_gpu_stats_record(false, 0.0, elapsed_ms, true);
			ereport(INFO,
					(errmsg("random_forest: GPU training "
							"unavailable, using CPU"),
					 errdetail("GPU attempt elapsed %.3f ms "
							   "(%s)",
							   elapsed_ms,
							   (errstr && *errstr)
							   ? *errstr
							   : "no error"),
					 errhidestmt(true)));
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

		elog(DEBUG1,
			 "logistic_regression: attempting direct GPU training: "
			 "backend=%s, lr_train=%p, samples=%d, dim=%d",
			 backend ? (backend->name ? backend->name : "unknown")
			 : "NULL",
			 backend ? (void *) backend->lr_train : NULL,
			 sample_count,
			 feature_dim);

		if (backend == NULL || backend->lr_train == NULL)
		{
			elog(DEBUG1,
				 "logistic_regression: backend or lr_train is "
				 "NULL, skipping GPU");
			goto lr_fallback;
		}

		/* Defensive: Validate all parameters before calling CUDA function */
		if (feature_matrix == NULL)
		{
			if (errstr)
				*errstr = pstrdup("logistic_regression: feature_matrix is NULL");
			elog(WARNING,
				 "logistic_regression: feature_matrix is NULL, skipping GPU");
			goto lr_fallback;
		}

		if (label_vector == NULL)
		{
			if (errstr)
				*errstr = pstrdup("logistic_regression: label_vector is NULL");
			elog(WARNING,
				 "logistic_regression: label_vector is NULL, skipping GPU");
			goto lr_fallback;
		}

		if (sample_count <= 0 || sample_count > 10000000)
		{
			if (errstr)
				*errstr = psprintf("logistic_regression: invalid sample_count %d",
								   sample_count);
			elog(WARNING,
				 "logistic_regression: invalid sample_count %d, skipping GPU",
				 sample_count);
			goto lr_fallback;
		}

		if (feature_dim <= 0 || feature_dim > 10000)
		{
			if (errstr)
				*errstr = psprintf("logistic_regression: invalid feature_dim %d",
								   feature_dim);
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
			/* GPU mode: re-raise error, no fallback */
			if (NDB_REQUIRE_GPU())
			{
				if (payload != NULL)
				{
					NDB_FREE(payload);
					payload = NULL;
				}
				if (metadata != NULL)
				{
					NDB_FREE(metadata);
					metadata = NULL;
				}
				PG_RE_THROW();
			}
			/* AUTO mode: fall back to CPU */
			elog(WARNING,
				 "logistic_regression: exception caught during GPU training, falling back to CPU");
			gpu_rc = -1;
			if (errstr && *errstr == NULL)
				*errstr = pstrdup("Exception during GPU training");
			if (payload != NULL)
			{
				NDB_FREE(payload);
				payload = NULL;
			}
			if (metadata != NULL)
			{
				NDB_FREE(metadata);
				metadata = NULL;
			}
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

				elog(DEBUG1,
					 "gpu_model_bridge: LR direct path "
					 "metadata: %s",
					 meta_txt);
				NDB_FREE(meta_txt);
			}
			else
			{
				elog(WARNING,
					 "gpu_model_bridge: LR direct path "
					 "metadata is NULL!");
			}
			ereport(INFO,
					(errmsg("logistic_regression: GPU training "
							"succeeded (direct)"),
					 errdetail(
							   "Elapsed %.3f ms on backend %s",
							   elapsed_ms,
							   ctx.backend_name
							   ? ctx.backend_name
							   : "unknown"),
					 errhidestmt(true)));
		}
		else
		{
			/* GPU mode: error if GPU training fails */
			/* CPU mode: never error, just return false for fallback */
			if (!NDB_COMPUTE_MODE_IS_CPU() && NDB_REQUIRE_GPU())
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("logistic_regression: GPU training failed - GPU mode requires GPU to be available"),
						 errdetail("GPU attempt elapsed %.3f ms (%s)",
								   elapsed_ms,
								   (errstr && *errstr) ? *errstr : "no error"),
						 errhint("Set compute_mode='auto' for automatic CPU fallback.")));
			}
			/* AUTO mode: fall back to CPU */
			ndb_gpu_stats_record(false, 0.0, elapsed_ms, true);
			ereport(INFO,
					(errmsg("logistic_regression: GPU training "
							"unavailable, using CPU"),
					 errdetail("GPU attempt elapsed %.3f ms "
							   "(%s)",
							   elapsed_ms,
							   (errstr && *errstr)
							   ? *errstr
							   : "no error"),
					 errhidestmt(true)));
		}
lr_fallback:;
	}

	if (!trained && algorithm != NULL
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

		if (backend == NULL || backend->linreg_train == NULL)
		{
			elog(DEBUG1,
				 "linear_regression: backend or linreg_train is "
				 "NULL, skipping GPU");
			goto linreg_fallback;
		}

		/* Defensive: Validate all parameters before calling GPU function */
		if (feature_matrix == NULL)
		{
			if (errstr)
				*errstr = pstrdup("linear_regression: feature_matrix is NULL");
			elog(WARNING,
				 "linear_regression: feature_matrix is NULL, skipping GPU");
			goto linreg_fallback;
		}

		if (label_vector == NULL)
		{
			if (errstr)
				*errstr = pstrdup("linear_regression: label_vector is NULL");
			elog(WARNING,
				 "linear_regression: label_vector is NULL, skipping GPU");
			goto linreg_fallback;
		}

		if (sample_count <= 0 || sample_count > 10000000)
		{
			if (errstr)
				*errstr = psprintf("linear_regression: invalid sample_count %d",
								   sample_count);
			elog(WARNING,
				 "linear_regression: invalid sample_count %d, skipping GPU",
				 sample_count);
			goto linreg_fallback;
		}

		if (feature_dim <= 0 || feature_dim > 10000)
		{
			if (errstr)
				*errstr = psprintf("linear_regression: invalid feature_dim %d",
								   feature_dim);
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
			/* CPU mode: never error, just fall back to CPU */
			if (NDB_COMPUTE_MODE_IS_CPU())
			{
				elog(WARNING,
					 "linear_regression: exception caught during GPU training attempt in CPU mode, falling back to CPU");
				FlushErrorState();
				gpu_rc = -1;
				if (errstr && *errstr == NULL)
					*errstr = pstrdup("Exception during GPU training (CPU mode)");
				if (payload != NULL)
				{
					NDB_FREE(payload);
					payload = NULL;
				}
				if (metadata != NULL)
				{
					NDB_FREE(metadata);
					metadata = NULL;
				}
			}
			/* GPU mode: re-raise error, no fallback */
			else if (!NDB_COMPUTE_MODE_IS_CPU() && NDB_REQUIRE_GPU())
			{
				if (payload != NULL)
				{
					NDB_FREE(payload);
					payload = NULL;
				}
				if (metadata != NULL)
				{
					NDB_FREE(metadata);
					metadata = NULL;
				}
				PG_RE_THROW();
			}
			/* AUTO mode: fall back to CPU */
			elog(WARNING,
				 "linear_regression: exception caught during GPU training, falling back to CPU");
			gpu_rc = -1;
			if (errstr && *errstr == NULL)
				*errstr = pstrdup("Exception during GPU training");
			if (payload != NULL)
			{
				NDB_FREE(payload);
				payload = NULL;
			}
			if (metadata != NULL)
			{
				NDB_FREE(metadata);
				metadata = NULL;
			}
			FlushErrorState();
		}
		PG_END_TRY();
		train_end = GetCurrentTimestamp();
		TimestampDifference(train_start, train_end, &secs, &usecs);
		elapsed_ms = ((double) secs * 1000.0) + ((double) usecs / 1000.0);

		if (gpu_rc == 0)
		{
			char	   *meta_txt = NULL;

			trained = true;
			ndb_gpu_stats_record(true, elapsed_ms, 0.0, false);

			/* Populate result structure with payload and metadata */
			if (result != NULL)
			{
				result->spec.model_data = payload;
				payload = NULL; /* Transfer ownership to result */

				if (metadata != NULL)
				{
					/*
					 * Skip metrics copying to avoid DirectFunctionCall1
					 * issues
					 */
					result->spec.metrics = NULL;
					result->metadata = NULL;

					meta_txt = DatumGetCString(
											   DirectFunctionCall1(jsonb_out,
																   JsonbPGetDatum(metadata)));

					elog(DEBUG1,
						 "gpu_model_bridge: linear_regression direct path "
						 "metadata: %s",
						 meta_txt);
					NDB_FREE(meta_txt);
				}
				else
				{
					elog(WARNING,
						 "gpu_model_bridge: linear_regression direct path "
						 "metadata is NULL!");
					result->spec.metrics = NULL;
				}
			}
			else
			{
				/* result is NULL, free payload and metadata */
				if (payload != NULL)
				{
					NDB_FREE(payload);
					payload = NULL;
				}
				if (metadata != NULL)
				{
					NDB_FREE(metadata);
					metadata = NULL;
				}
			}

			ereport(INFO,
					(errmsg("linear_regression: GPU training "
							"succeeded (direct)"),
					 errdetail(
							   "Elapsed %.3f ms on backend %s",
							   elapsed_ms,
							   backend && backend->name
							   ? backend->name
							   : "unknown"),
					 errhidestmt(true)));
		}
		else
		{
			/* GPU mode: error if GPU training fails */
			/* CPU mode: never error, just return false for fallback */
			if (!NDB_COMPUTE_MODE_IS_CPU() && NDB_REQUIRE_GPU())
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("linear_regression: GPU training failed - GPU mode requires GPU to be available"),
						 errdetail("GPU attempt elapsed %.3f ms (%s)",
								   elapsed_ms,
								   (errstr && *errstr) ? *errstr : "no error"),
						 errhint("Set compute_mode='auto' for automatic CPU fallback.")));
			}
			/* AUTO mode: fall back to CPU */
			ndb_gpu_stats_record(false, 0.0, elapsed_ms, true);
			ereport(INFO,
					(errmsg("linear_regression: GPU training "
							"unavailable, using CPU"),
					 errdetail("GPU attempt elapsed %.3f ms "
							   "(%s)",
							   elapsed_ms,
							   (errstr && *errstr)
							   ? *errstr
							   : "no error"),
					 errhidestmt(true)));
		}
linreg_fallback:;
	}

	if (!trained && algorithm != NULL
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
			ereport(DEBUG1,
					(errmsg("decision_tree: GPU training succeeded "
							"(direct)"),
					 errdetail(
							   "Elapsed %.3f ms on backend %s",
							   elapsed_ms,
							   ctx.backend_name
							   ? ctx.backend_name
							   : "unknown"),
					 errhidestmt(true)));
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
			ereport(INFO,
					(errmsg("decision_tree: GPU training "
							"unavailable, using CPU"),
					 errdetail("GPU attempt elapsed %.3f ms "
							   "(%s)",
							   elapsed_ms,
							   (errstr && *errstr)
							   ? *errstr
							   : "no error"),
					 errhidestmt(true)));
		}
	}

	if (!trained && algorithm != NULL && strcmp(algorithm, "ridge") == 0
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
			ereport(DEBUG1,
					(errmsg("ridge: GPU training succeeded "
							"(direct)"),
					 errdetail(
							   "Elapsed %.3f ms on backend %s",
							   elapsed_ms,
							   ctx.backend_name
							   ? ctx.backend_name
							   : "unknown"),
					 errhidestmt(true)));
		}
		else
		{
			/* GPU mode: error if GPU training fails */
			/* CPU mode: never error, just return false for fallback */
			if (!NDB_COMPUTE_MODE_IS_CPU() && NDB_REQUIRE_GPU())
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("ridge: GPU training failed - GPU mode requires GPU to be available"),
						 errdetail("GPU attempt elapsed %.3f ms (%s)",
								   elapsed_ms,
								   (errstr && *errstr) ? *errstr : "no error"),
						 errhint("Set compute_mode='auto' for automatic CPU fallback.")));
			}
			/* AUTO mode: fall back to CPU */
			ndb_gpu_stats_record(false, 0.0, elapsed_ms, true);
			ereport(INFO,
					(errmsg("ridge: GPU training unavailable, "
							"using CPU"),
					 errdetail("GPU attempt elapsed %.3f ms "
							   "(%s)",
							   elapsed_ms,
							   (errstr && *errstr)
							   ? *errstr
							   : "no error"),
					 errhidestmt(true)));
		}
	}

	if (!trained && algorithm != NULL && strcmp(algorithm, "lasso") == 0
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
			ereport(DEBUG1,
					(errmsg("lasso: GPU training succeeded "
							"(direct)"),
					 errdetail(
							   "Elapsed %.3f ms on backend %s",
							   elapsed_ms,
							   ctx.backend_name
							   ? ctx.backend_name
							   : "unknown"),
					 errhidestmt(true)));
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
			ereport(INFO,
					(errmsg("lasso: GPU training unavailable, "
							"using CPU"),
					 errdetail("GPU attempt elapsed %.3f ms "
							   "(%s)",
							   elapsed_ms,
							   (errstr && *errstr)
							   ? *errstr
							   : "no error"),
					 errhidestmt(true)));
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
			elog(DEBUG1,
				 "gpu_model_bridge: copied metrics to "
				 "result->spec.metrics: %p",
				 (void *) result->spec.metrics);
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
			ereport(DEBUG2, (errmsg("gpu_model_bridge: setting model_data from payload")));
			result->spec.model_data = payload;
			result->payload = payload;
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
				NDB_FREE(payload);
				ereport(DEBUG2, (errmsg("gpu_model_bridge: unused payload freed")));
			}
			ereport(DEBUG2, (errmsg("gpu_model_bridge: payload pointer set")));
		}
		ereport(DEBUG2, (errmsg("gpu_model_bridge: model_data assignment completed")));
	}

	ereport(DEBUG2, (errmsg("gpu_model_bridge: about to check ops->destroy, ops=%p", (void *) ops)));
	if (ops != NULL && ops->destroy != NULL)
	{
		ereport(DEBUG2, (errmsg("gpu_model_bridge: calling ops->destroy")));
		ops->destroy(&model);
		ereport(DEBUG2, (errmsg("gpu_model_bridge: ops->destroy completed")));
	}
	else
	{
		ereport(DEBUG2, (errmsg("gpu_model_bridge: ops is NULL or destroy is NULL, skipping")));
	}

	ereport(DEBUG2, (errmsg("gpu_model_bridge: about to check if (!trained), trained=%d", trained)));
	if (!trained)
	{
		ereport(DEBUG1, (errmsg("gpu_model_bridge: trained is false, cleaning up")));
		NDB_FREE(payload);
		NDB_FREE(metadata);
		if (result != NULL)
		{
			ereport(DEBUG2, (errmsg("gpu_model_bridge: calling ndb_gpu_free_train_result")));
			ndb_gpu_free_train_result(result);
			ereport(DEBUG2, (errmsg("gpu_model_bridge: ndb_gpu_free_train_result completed")));
		}
		ereport(DEBUG1, (errmsg("gpu_model_bridge: cleanup completed")));
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
	if (result == NULL)
		return;

	if (result->spec.algorithm)
		NDB_FREE(result->spec.algorithm);
	if (result->spec.training_table)
		NDB_FREE(result->spec.training_table);
	if (result->spec.training_column)
		NDB_FREE(result->spec.training_column);
	if (result->spec.project_name)
		NDB_FREE(result->spec.project_name);
	if (result->spec.model_name)
		NDB_FREE(result->spec.model_name);
	NDB_FREE(result->spec.model_data);
	NDB_FREE(result->spec.metrics);
	if (result->metadata && result->metadata != result->spec.metrics)
		NDB_FREE(result->metadata);
	if (result->metrics && result->metrics != result->spec.metrics)
		NDB_FREE(result->metrics);

	memset(result, 0, sizeof(MLGpuTrainResult));
}
