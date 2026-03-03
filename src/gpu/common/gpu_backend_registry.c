/*-------------------------------------------------------------------------
 *
 * gpu_backend_registry.c
 *    Backend registration and selection system.
 *
 * This module manages the registry of available backends and provides
 * automatic selection based on system capabilities.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/common/gpu_backend_registry.c
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#include "utils/jsonb.h"
#include "utils/builtins.h"
#include "utils/elog.h"
#include "funcapi.h"
#include "access/htup_details.h"
#include "catalog/pg_type.h"

#include <signal.h>

#include "neurondb_gpu_backend.h"
#include "ml_gpu_naive_bayes.h"
#include "ml_gpu_knn.h"
#include "neurondb_constants.h"
#include "ml_gpu_random_forest.h"
#include "ml_random_forest_internal.h"
#include "ml_gpu_logistic_regression.h"
#include "ml_logistic_regression_internal.h"
#include "ml_gpu_linear_regression.h"
#include "ml_linear_regression_internal.h"
#include "ml_gpu_svm.h"
#include "ml_svm_internal.h"
#include "ml_gpu_decision_tree.h"
#include "ml_decision_tree_internal.h"
#include "ml_gpu_ridge_regression.h"
#include "ml_ridge_regression_internal.h"
#include "ml_gpu_lasso_regression.h"
#include "ml_lasso_regression_internal.h"
#include "ml_gpu_xgboost.h"
#include "ml_gpu_catboost.h"
#include "neurondb_macros.h"

#include <string.h>

typedef struct NDBGpuBackendRegistry
{
	const ndb_gpu_backend *backends[NDB_GPU_MAX_BACKENDS];
	int			count;
}			NDBGpuBackendRegistry;

static NDBGpuBackendRegistry registry =
{
	.backends =
	{
		NULL
	},
		.count = 0,
};

static const ndb_gpu_backend *active_backend = NULL;

static int
ndb_backend_priority(NDBGpuBackendKind kind)
{
	switch (kind)
	{
		case NDB_GPU_BACKEND_METAL:
			return 100;
		case NDB_GPU_BACKEND_CUDA:
			return 90;
		case NDB_GPU_BACKEND_ROCM:
			return 80;
		default:
			return 0;
	}
}

static int
ndb_backend_is_available(const ndb_gpu_backend *backend)
{
	if (backend == NULL)
		return 0;

	if (backend->is_available == NULL)
		return 1;

	return backend->is_available();
}

int
ndb_gpu_register_backend(const ndb_gpu_backend *backend)
{
	int			i;

	if (backend == NULL)
	{
		return -1;
	}

	if (registry.count >= NDB_GPU_MAX_BACKENDS)
	{
		return -1;
	}

	for (i = 0; i < registry.count; i++)
	{
		if (registry.backends[i]->kind == backend->kind)
		{
			return 0;
		}
	}

	registry.backends[registry.count++] = backend;


	return 0;
}

static const ndb_gpu_backend *
ndb_gpu_select_best_internal(void)
{
	const ndb_gpu_backend *best = NULL;
	int			best_priority = -1;
	int			i;

	for (i = 0; i < registry.count; i++)
	{
		const ndb_gpu_backend *candidate = registry.backends[i];
		int			priority;

		if (!ndb_backend_is_available(candidate))
			continue;

		priority = candidate->priority != 0
			? candidate->priority
			: ndb_backend_priority(candidate->kind);

		if (priority > best_priority)
		{
			best = candidate;
			best_priority = priority;
		}
	}

	return best;
}

int
ndb_gpu_set_active_backend(const ndb_gpu_backend *backend)
{
	int			rc;

	/* CPU mode: never set or initialize a backend */
	if (NDB_COMPUTE_MODE_IS_CPU())
	{
		if (active_backend && active_backend->shutdown)
			active_backend->shutdown();
		active_backend = NULL;
		return 0;
	}

	if (active_backend == backend)
		return 0;

	if (active_backend && active_backend->shutdown)
		active_backend->shutdown();

	active_backend = backend;

	if (backend && backend->init)
	{
		/* Initialize GPU backend - temporarily ignore SIGPIPE */
		sigset_t	old_sigset,
					new_sigset;

		sigemptyset(&new_sigset);
		sigaddset(&new_sigset, SIGPIPE);
		if (sigprocmask(SIG_BLOCK, &new_sigset, &old_sigset) == 0)
		{
			elog(LOG, "neurondb: Initializing GPU backend '%s'",
				 backend->name ? backend->name : "unknown");

			rc = backend->init();

			/* Restore signal mask */
			sigprocmask(SIG_SETMASK, &old_sigset, NULL);
		}
		else
		{
			/* Fallback if we can't block SIGPIPE */
			rc = backend->init();
		}

		if (rc != 0)
		{
			elog(WARNING,
				 "neurondb: failed to initialise GPU backend "
				 "'%s' (rc=%d)",
				 backend->name ? backend->name : "unknown",
				 rc);

			if (backend->shutdown)
				backend->shutdown();

			active_backend = NULL;
			return rc;
		}
		elog(LOG, "neurondb: GPU backend '%s' initialized successfully",
			 backend->name ? backend->name : "unknown");
	}

	return 0;
}

const ndb_gpu_backend *
ndb_gpu_get_active_backend(void)
{
	/* CPU mode: never return a backend */
	if (NDB_COMPUTE_MODE_IS_CPU())
		return NULL;
	return active_backend;
}

int
ndb_gpu_rf_train(const float *features,
				 const double *labels,
				 int n_samples,
				 int feature_dim,
				 int class_count,
				 const Jsonb * hyperparams,
				 bytea * *model_data,
				 Jsonb * *metrics,
				 char **errstr)
{
	int			rc;

	/* CPU mode: never run GPU code */
	if (NDB_COMPUTE_MODE_IS_CPU())
	{
		if (errstr)
			*errstr = NULL;
		return -1;
	}


	if (errstr)
		*errstr = NULL;


	if (!active_backend || active_backend->rf_train == NULL)
	{
		return -1;
	}

	ereport(DEBUG2,
			(errmsg("ndb_gpu_rf_train: about to call active_backend->rf_train"),
			 errdetail("backend_name=%s", active_backend->name ? active_backend->name : "NULL")));

	/* Log hyperparameters value before calling backend */
	if (hyperparams != NULL)
	{
		PG_TRY();
		{
			char	   *hyperparams_text = DatumGetCString(
											DirectFunctionCall1(jsonb_out,
																JsonbPGetDatum(hyperparams)));
			if (hyperparams_text)
				pfree(hyperparams_text);
		}
		PG_CATCH();
		{
			FlushErrorState();
		}
		PG_END_TRY();
	}

	PG_TRY();
	{
		rc = active_backend->rf_train(features,
									  labels,
									  n_samples,
									  feature_dim,
									  class_count,
									  hyperparams,
									  model_data,
									  metrics,
									  errstr);
	}
	PG_CATCH();
	{
		PG_RE_THROW();
	}
	PG_END_TRY();

	ereport(DEBUG2,
			(errmsg("ndb_gpu_rf_train: active_backend->rf_train returned"),
			 errdetail("rc=%d, model_data=%p, metrics=%p",
					   rc,
					   (void *) (model_data ? *model_data : NULL),
					   (void *) (metrics ? *metrics : NULL))));

	return rc;
}

int
ndb_gpu_rf_predict(const bytea * model_data,
				   const float *input,
				   int feature_dim,
				   int *class_out,
				   char **errstr)
{
	/* CPU mode: never run GPU code */
	if (NDB_COMPUTE_MODE_IS_CPU())
	{
		if (errstr)
			*errstr = NULL;
		return -1;
	}

	if (errstr)
		*errstr = NULL;
	if (!active_backend || active_backend->rf_predict == NULL)
		return -1;
	return active_backend->rf_predict(
									  model_data, input, feature_dim, class_out, errstr);
}

int
ndb_gpu_rf_pack_model(const RFModel *model,
					  bytea * *model_data,
					  Jsonb * *metrics,
					  char **errstr)
{
	if (errstr)
		*errstr = NULL;
	if (!active_backend || active_backend->rf_pack == NULL)
		return -1;
	return active_backend->rf_pack(model, model_data, metrics, errstr);
}

int
ndb_gpu_lr_train(const float *features,
				 const double *labels,
				 int n_samples,
				 int feature_dim,
				 const Jsonb * hyperparams,
				 bytea * *model_data,
				 Jsonb * *metrics,
				 char **errstr)
{
	int			rc;

	/* CPU mode: never run GPU code */
	if (NDB_COMPUTE_MODE_IS_CPU())
	{
		if (errstr)
			*errstr = NULL;
		return -1;
	}

	if (errstr)
		*errstr = NULL;
	if (!active_backend)
	{
		if (errstr)
			*errstr = pstrdup("ndb_gpu_lr_train: active_backend is NULL");
		return -1;
	}
	if (active_backend->lr_train == NULL)
	{
		if (errstr)
			*errstr = psprintf("ndb_gpu_lr_train: backend->lr_train is NULL (backend=%s)", active_backend->name ? active_backend->name : "unknown");
		return -1;
	}

	rc = active_backend->lr_train(features,
								  labels,
								  n_samples,
								  feature_dim,
								  hyperparams,
								  model_data,
								  metrics,
								  errstr);
	return rc;
}

int
ndb_gpu_lr_predict(const bytea * model_data,
				   const float *input,
				   int feature_dim,
				   double *probability_out,
				   char **errstr)
{
	if (errstr)
		*errstr = NULL;
	if (!active_backend || active_backend->lr_predict == NULL)
		return -1;
	return active_backend->lr_predict(
									  model_data, input, feature_dim, probability_out, errstr);
}

int
ndb_gpu_lr_pack_model(const LRModel *model,
					  bytea * *model_data,
					  Jsonb * *metrics,
					  char **errstr)
{
	if (errstr)
		*errstr = NULL;
	if (!active_backend || active_backend->lr_pack == NULL)
		return -1;
	return active_backend->lr_pack(model, model_data, metrics, errstr);
}

int
ndb_gpu_linreg_train(const float *features,
					 const double *targets,
					 int n_samples,
					 int feature_dim,
					 const Jsonb * hyperparams,
					 bytea * *model_data,
					 Jsonb * *metrics,
					 char **errstr)
{
	int			rc;

	/* CPU mode: never run GPU code */
	if (NDB_COMPUTE_MODE_IS_CPU())
	{
		if (errstr)
			*errstr = NULL;
		return -1;
	}

	if (errstr)
		*errstr = NULL;
	if (!active_backend)
	{
		if (errstr)
			*errstr = pstrdup("ndb_gpu_linreg_train: active_backend is NULL");
		return -1;
	}
	if (active_backend->linreg_train == NULL)
	{
		if (errstr)
			*errstr = psprintf("ndb_gpu_linreg_train: backend->linreg_train is NULL (backend=%s)", active_backend->name ? active_backend->name : "unknown");
		return -1;
	}

	rc = active_backend->linreg_train(features,
									  targets,
									  n_samples,
									  feature_dim,
									  hyperparams,
									  model_data,
									  metrics,
									  errstr);
	return rc;
}

int
ndb_gpu_linreg_predict(const bytea * model_data,
					   const float *input,
					   int feature_dim,
					   double *prediction_out,
					   char **errstr)
{
	if (errstr)
		*errstr = NULL;
	if (!active_backend || active_backend->linreg_predict == NULL)
		return -1;
	return active_backend->linreg_predict(
										  model_data, input, feature_dim, prediction_out, errstr);
}

int
ndb_gpu_linreg_pack_model(const LinRegModel *model,
						  bytea * *model_data,
						  Jsonb * *metrics,
						  char **errstr)
{
	if (errstr)
		*errstr = NULL;
	if (!active_backend || active_backend->linreg_pack == NULL)
		return -1;
	return active_backend->linreg_pack(model, model_data, metrics, errstr);
}

int
ndb_gpu_svm_train(const float *features,
				  const double *labels,
				  int n_samples,
				  int feature_dim,
				  const Jsonb * hyperparams,
				  bytea * *model_data,
				  Jsonb * *metrics,
				  char **errstr)
{
	if (errstr)
		*errstr = NULL;
	if (!active_backend || active_backend->svm_train == NULL)
		return -1;
	return active_backend->svm_train(features,
									 labels,
									 n_samples,
									 feature_dim,
									 hyperparams,
									 model_data,
									 metrics,
									 errstr);
}

int
ndb_gpu_svm_predict(const bytea * model_data,
					const float *input,
					int feature_dim,
					int *class_out,
					double *confidence_out,
					char **errstr)
{
	if (errstr)
		*errstr = NULL;
	if (!active_backend || active_backend->svm_predict == NULL)
		return -1;
	return active_backend->svm_predict(model_data,
									   input,
									   feature_dim,
									   class_out,
									   confidence_out,
									   errstr);
}

int
ndb_gpu_svm_pack_model(const SVMModel * model,
					   bytea * *model_data,
					   Jsonb * *metrics,
					   char **errstr)
{
	if (errstr)
		*errstr = NULL;
	if (!active_backend || active_backend->svm_pack == NULL)
		return -1;
	return active_backend->svm_pack(model, model_data, metrics, errstr);
}

int
ndb_gpu_svm_predict_double(const bytea * model_data,
						   const float *input,
						   int feature_dim,
						   double *prediction_out,
						   char **errstr)
{
	int			class_out;
	double		confidence_out;
	int			rc;

	if (errstr)
		*errstr = NULL;
	if (prediction_out == NULL)
		return -1;

	rc = ndb_gpu_svm_predict(model_data,
							 input,
							 feature_dim,
							 &class_out,
							 &confidence_out,
							 errstr);
	if (rc != 0)
		return rc;

	/* Return class as double (0.0 or 1.0) */
	*prediction_out = (double) class_out;
	return 0;
}

int
ndb_gpu_dt_train(const float *features,
				 const double *labels,
				 int n_samples,
				 int feature_dim,
				 const Jsonb * hyperparams,
				 bytea * *model_data,
				 Jsonb * *metrics,
				 char **errstr)
{
	if (errstr)
		*errstr = NULL;
	if (!active_backend || active_backend->dt_train == NULL)
		return -1;
	return active_backend->dt_train(features,
									labels,
									n_samples,
									feature_dim,
									hyperparams,
									model_data,
									metrics,
									errstr);
}

int
ndb_gpu_dt_predict(const bytea * model_data,
				   const float *input,
				   int feature_dim,
				   double *prediction_out,
				   char **errstr)
{
	if (errstr)
		*errstr = NULL;
	if (!active_backend || active_backend->dt_predict == NULL)
		return -1;
	return active_backend->dt_predict(
									  model_data, input, feature_dim, prediction_out, errstr);
}

int
ndb_gpu_dt_pack_model(const struct DTModel *model,
					  bytea * *model_data,
					  Jsonb * *metrics,
					  char **errstr)
{
	if (errstr)
		*errstr = NULL;
	if (!active_backend || active_backend->dt_pack == NULL)
		return -1;
	return active_backend->dt_pack(model, model_data, metrics, errstr);
}

int
ndb_gpu_ridge_train(const float *features,
					const double *targets,
					int n_samples,
					int feature_dim,
					const Jsonb * hyperparams,
					bytea * *model_data,
					Jsonb * *metrics,
					char **errstr)
{
	/* CPU mode: never run GPU code */
	if (NDB_COMPUTE_MODE_IS_CPU())
	{
		if (errstr)
			*errstr = NULL;
		return -1;
	}

	if (errstr)
		*errstr = NULL;
	if (!active_backend)
	{
		if (errstr)
			*errstr = pstrdup("GPU backend not available");
		return -1;
	}
	if (active_backend->ridge_train == NULL)
	{
		if (errstr)
			*errstr = psprintf("GPU backend '%s' does not support ridge_train",
							   active_backend->name ? active_backend->name : "unknown");
		return -1;
	}
	{
		int rc = active_backend->ridge_train(features,
											  targets,
											  n_samples,
											  feature_dim,
											  hyperparams,
											  model_data,
											  metrics,
											  errstr);
		/* Copy error string to current memory context if it exists */
		if (rc != 0 && errstr && *errstr != NULL)
		{
			char *old_err = *errstr;
			*errstr = pstrdup(old_err);
			if (old_err)
				pfree(old_err);
		}
		return rc;
	}
}

int
ndb_gpu_ridge_predict(const bytea * model_data,
					  const float *input,
					  int feature_dim,
					  double *prediction_out,
					  char **errstr)
{
	if (errstr)
		*errstr = NULL;
	if (!active_backend || active_backend->ridge_predict == NULL)
		return -1;
	return active_backend->ridge_predict(
										 model_data, input, feature_dim, prediction_out, errstr);
}

int
ndb_gpu_ridge_pack_model(const struct RidgeModel *model,
						 bytea * *model_data,
						 Jsonb * *metrics,
						 char **errstr)
{
	if (errstr)
		*errstr = NULL;
	if (!active_backend || active_backend->ridge_pack == NULL)
		return -1;
	return active_backend->ridge_pack(model, model_data, metrics, errstr);
}

int
ndb_gpu_lasso_train(const float *features,
					const double *targets,
					int n_samples,
					int feature_dim,
					const Jsonb * hyperparams,
					bytea * *model_data,
					Jsonb * *metrics,
					char **errstr)
{
	/* CPU mode: never run GPU code */
	if (NDB_COMPUTE_MODE_IS_CPU())
	{
		if (errstr)
			*errstr = NULL;
		return -1;
	}

	if (errstr)
		*errstr = NULL;
	if (!active_backend)
	{
		if (errstr)
			*errstr = pstrdup("GPU backend not initialized");
		return -1;
	}
	if (active_backend->lasso_train == NULL)
	{
		if (errstr)
			*errstr = psprintf("lasso_train not implemented for backend '%s'",
							   active_backend->name ? active_backend->name : "unknown");
		return -1;
	}
	return active_backend->lasso_train(features,
									   targets,
									   n_samples,
									   feature_dim,
									   hyperparams,
									   model_data,
									   metrics,
									   errstr);
}

int
ndb_gpu_lasso_predict(const bytea * model_data,
					  const float *input,
					  int feature_dim,
					  double *prediction_out,
					  char **errstr)
{
	if (errstr)
		*errstr = NULL;
	if (!active_backend || active_backend->lasso_predict == NULL)
		return -1;
	return active_backend->lasso_predict(
										 model_data, input, feature_dim, prediction_out, errstr);
}

int
ndb_gpu_lasso_pack_model(const struct LassoModel *model,
						 bytea * *model_data,
						 Jsonb * *metrics,
						 char **errstr)
{
	if (errstr)
		*errstr = NULL;
	if (!active_backend || active_backend->lasso_pack == NULL)
		return -1;
	return active_backend->lasso_pack(model, model_data, metrics, errstr);
}

int
ndb_gpu_nb_train(const float *features,
				 const double *labels,
				 int n_samples,
				 int feature_dim,
				 int class_count,
				 const Jsonb * hyperparams,
				 bytea * *model_data,
				 Jsonb * *metrics,
				 char **errstr)
{
	/* CPU mode: never run GPU code */
	if (NDB_COMPUTE_MODE_IS_CPU())
	{
		if (errstr)
			*errstr = NULL;
		return -1;
	}

	if (errstr)
		*errstr = NULL;
	if (!active_backend)
	{
		if (errstr)
			*errstr = pstrdup("GPU backend not available");
		return -1;
	}
	if (active_backend->nb_train == NULL)
	{
		if (errstr)
			*errstr = psprintf("GPU backend '%s' does not support nb_train",
							   active_backend->name ? active_backend->name : "unknown");
		return -1;
	}
	return active_backend->nb_train(features,
									 labels,
									 n_samples,
									 feature_dim,
									 class_count,
									 hyperparams,
									 model_data,
									 metrics,
									 errstr);
}

int
ndb_gpu_nb_predict(const bytea * model_data,
				   const float *input,
				   int feature_dim,
				   int *class_out,
				   double *probability_out,
				   char **errstr)
{
	if (errstr)
		*errstr = NULL;
	if (!active_backend || active_backend->nb_predict == NULL)
		return -1;
	return active_backend->nb_predict(model_data,
									   input,
									   feature_dim,
									   class_out,
									   probability_out,
									   errstr);
}

int
ndb_gpu_nb_pack_model(const struct GaussianNBModel *model,
					  bytea * *model_data,
					  Jsonb * *metrics,
					  char **errstr)
{
	if (errstr)
		*errstr = NULL;
	if (!active_backend || active_backend->nb_pack == NULL)
		return -1;
	return active_backend->nb_pack(model, model_data, metrics, errstr);
}

int
ndb_gpu_knn_train(const float *features,
				  const double *labels,
				  int n_samples,
				  int feature_dim,
				  int k,
				  int task_type,
				  const Jsonb * hyperparams,
				  bytea * *model_data,
				  Jsonb * *metrics,
				  char **errstr)
{
	/* CPU mode: never run GPU code */
	if (NDB_COMPUTE_MODE_IS_CPU())
	{
		if (errstr)
			*errstr = NULL;
		return -1;
	}

	if (errstr)
		*errstr = NULL;
	if (!active_backend)
	{
		if (errstr)
			*errstr = pstrdup("GPU backend not available");
		return -1;
	}
	if (active_backend->knn_train == NULL)
	{
		if (errstr)
			*errstr = psprintf("GPU backend '%s' does not support knn_train",
							   active_backend->name ? active_backend->name : "unknown");
		return -1;
	}
	return active_backend->knn_train(features,
									  labels,
									  n_samples,
									  feature_dim,
									  k,
									  task_type,
									  hyperparams,
									  model_data,
									  metrics,
									  errstr);
}

int
ndb_gpu_knn_predict(const bytea * model_data,
					const float *input,
					int feature_dim,
					double *prediction_out,
					char **errstr)
{
	if (errstr)
		*errstr = NULL;
	if (!active_backend || active_backend->knn_predict == NULL)
		return -1;
	return active_backend->knn_predict(model_data,
										input,
										feature_dim,
										prediction_out,
										errstr);
}

int
ndb_gpu_knn_pack(const struct KNNModel *model,
				 bytea * *model_data,
				 Jsonb * *metrics,
				 char **errstr)
{
	if (errstr)
		*errstr = NULL;
	if (!active_backend || active_backend->knn_pack == NULL)
		return -1;
	return active_backend->knn_pack(model, model_data, metrics, errstr);
}

int
ndb_gpu_xgboost_train(const float *features,
					  const double *labels,
					  int n_samples,
					  int feature_dim,
					  const Jsonb * hyperparams,
					  bytea * *model_data,
					  Jsonb * *metrics,
					  char **errstr)
{
	/* CPU mode: never run GPU code */
	if (NDB_COMPUTE_MODE_IS_CPU())
	{
		if (errstr)
			*errstr = NULL;
		return -1;
	}

	if (errstr)
		*errstr = NULL;
	if (!active_backend)
	{
		if (errstr)
			*errstr = pstrdup("GPU backend not available");
		return -1;
	}
	if (active_backend->xgboost_train == NULL)
	{
		if (errstr)
			*errstr = psprintf("GPU backend '%s' does not support xgboost_train",
							   active_backend->name ? active_backend->name : "unknown");
		return -1;
	}
	return active_backend->xgboost_train(features,
										  labels,
										  n_samples,
										  feature_dim,
										  hyperparams,
										  model_data,
										  metrics,
										  errstr);
}

int
ndb_gpu_xgboost_predict(const bytea * model_data,
						 const float *input,
						 int feature_dim,
						 double *prediction_out,
						 char **errstr)
{
	if (errstr)
		*errstr = NULL;
	if (!active_backend || active_backend->xgboost_predict == NULL)
		return -1;
	return active_backend->xgboost_predict(model_data,
											input,
											feature_dim,
											prediction_out,
											errstr);
}

int
ndb_gpu_xgboost_pack_model(const struct XGBoostModel *model,
							 bytea * *model_data,
							 Jsonb * *metrics,
							 char **errstr)
{
	if (errstr)
		*errstr = NULL;
	if (!active_backend || active_backend->xgboost_pack == NULL)
		return -1;
	return active_backend->xgboost_pack(model, model_data, metrics, errstr);
}

int
ndb_gpu_catboost_train(const float *features,
					   const double *labels,
					   int n_samples,
					   int feature_dim,
					   const Jsonb * hyperparams,
					   bytea * *model_data,
					   Jsonb * *metrics,
					   char **errstr)
{
	/* CPU mode: never run GPU code */
	if (NDB_COMPUTE_MODE_IS_CPU())
	{
		if (errstr)
			*errstr = NULL;
		return -1;
	}

	if (errstr)
		*errstr = NULL;
	if (!active_backend)
	{
		if (errstr)
			*errstr = pstrdup("GPU backend not available");
		return -1;
	}
	if (active_backend->catboost_train == NULL)
	{
		if (errstr)
			*errstr = psprintf("GPU backend '%s' does not support catboost_train",
							   active_backend->name ? active_backend->name : "unknown");
		return -1;
	}
	return active_backend->catboost_train(features,
										  labels,
										  n_samples,
										  feature_dim,
										  hyperparams,
										  model_data,
										  metrics,
										  errstr);
}

int
ndb_gpu_catboost_predict(const bytea * model_data,
						  const float *input,
						  int feature_dim,
						  double *prediction_out,
						  char **errstr)
{
	if (errstr)
		*errstr = NULL;
	if (!active_backend || active_backend->catboost_predict == NULL)
		return -1;
	return active_backend->catboost_predict(model_data,
											input,
											feature_dim,
											prediction_out,
											errstr);
}

int
ndb_gpu_catboost_pack_model(const struct CatBoostModel *model,
							 bytea * *model_data,
							 Jsonb * *metrics,
							 char **errstr)
{
	if (errstr)
		*errstr = NULL;
	if (!active_backend || active_backend->catboost_pack == NULL)
		return -1;
	return active_backend->catboost_pack(model, model_data, metrics, errstr);
}

const ndb_gpu_backend *
ndb_gpu_select_backend(const char *name)
{
	const ndb_gpu_backend *chosen = NULL;
	int			i;

	if (registry.count == 0)
	{
		return NULL;
	}

	if (name == NULL || name[0] == '\0' || pg_strcasecmp(name, "auto") == 0)
	{
		chosen = ndb_gpu_select_best_internal();
		if (!chosen)
		{
			return NULL;
		}
	}
	else
	{
		for (i = 0; i < registry.count; i++)
		{
			const ndb_gpu_backend *candidate = registry.backends[i];

			if (pg_strcasecmp(candidate->name, name) != 0)
				continue;

			if (!ndb_backend_is_available(candidate))
			{
				return NULL;
			}

			chosen = candidate;
			break;
		}

		if (chosen == NULL)
		{
			return NULL;
		}
	}

	if (ndb_gpu_set_active_backend(chosen) != 0)
		return NULL;

	elog(LOG, "neurondb: selected GPU backend: %s", chosen->name);

	return active_backend;
}

void
ndb_gpu_list_backends(void)
{
	int			i;

	elog(LOG, "neurondb: GPU backends registered: %d", registry.count);

	for (i = 0; i < registry.count; i++)
	{
		const ndb_gpu_backend *backend = registry.backends[i];
		const char *name = backend->name ? backend->name : "unknown";
		const char *provider =
			backend->provider ? backend->provider : "unknown";
		bool		available = ndb_backend_is_available(backend) != 0;

		elog(LOG,
			 "  [%d] %s (%s) - %s",
			 i + 1,
			 name,
			 provider,
			 available ? "available" : "unavailable");
	}
}

/* -------------------------------------------------------------------------
 * SQL-visible listing of registered GPU backends
 * Returns one row per backend with name, provider, availability, priority, kind
 * ------------------------------------------------------------------------- */
PG_FUNCTION_INFO_V1(neurondb_gpu_backends);
Datum
neurondb_gpu_backends(PG_FUNCTION_ARGS)
{
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	TupleDesc	tupdesc;
	int			i;

	if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo))
		ereport(ERROR,
				(errmsg("neurondb_gpu_backends: invalid resultinfo")));

	if (rsinfo->expectedDesc != NULL)
		tupdesc = rsinfo->expectedDesc;
	else
	{
		tupdesc = CreateTemplateTupleDesc(5);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "name", TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc,
						   (AttrNumber) 2,
						   "provider",
						   TEXTOID,
						   -1,
						   0);
		TupleDescInitEntry(tupdesc,
						   (AttrNumber) 3,
						   "available",
						   BOOLOID,
						   -1,
						   0);
		TupleDescInitEntry(tupdesc,
						   (AttrNumber) 4,
						   "priority",
						   INT4OID,
						   -1,
						   0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 5, "kind", TEXTOID, -1, 0);
		BlessTupleDesc(tupdesc);
		rsinfo->expectedDesc = tupdesc;
	}
	tupdesc = rsinfo->expectedDesc;

	/* Initialize materialized SRF */
	{
		MemoryContext per_query_ctx = rsinfo->econtext->ecxt_per_query_memory;
		MemoryContext oldcontext = MemoryContextSwitchTo(per_query_ctx);
		Tuplestorestate *tupstore = tuplestore_begin_heap(true, false, 1024);

		rsinfo->returnMode = SFRM_Materialize;
		rsinfo->setResult = tupstore;
		rsinfo->setDesc = tupdesc;

		MemoryContextSwitchTo(oldcontext);
	}

	for (i = 0; i < registry.count; i++)
	{
		const ndb_gpu_backend *backend = registry.backends[i];
		Datum		values[5];
		bool		nulls[5] = {false, false, false, false, false};
		const char *name =
			backend && backend->name ? backend->name : "unknown";
		const char *provider = backend && backend->provider
			? backend->provider
			: "unknown";
		bool		available =
			backend ? (ndb_backend_is_available(backend) != 0) : false;
		int32		priority = backend ? ndb_backend_priority(backend->kind) : 0;
		const char *kind_str = "unknown";

		if (backend)
		{
			switch (backend->kind)
			{
				case NDB_GPU_BACKEND_METAL:
					kind_str = "metal";
					break;
				case NDB_GPU_BACKEND_CUDA:
					kind_str = "cuda";
					break;
				case NDB_GPU_BACKEND_ROCM:
					kind_str = "rocm";
					break;
				default:
					kind_str = "none";
					break;
			}
		}

		values[0] = CStringGetTextDatum(name);
		values[1] = CStringGetTextDatum(provider);
		values[2] = BoolGetDatum(available);
		values[3] = Int32GetDatum(priority);
		values[4] = CStringGetTextDatum(kind_str);

		tuplestore_putvalues(rsinfo->setResult,
							 rsinfo->setDesc ? rsinfo->setDesc : tupdesc,
							 values,
							 nulls);
	}

	return (Datum) 0;
}
