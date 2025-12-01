/*-------------------------------------------------------------------------
 *
 * neurondb_rocm_lasso.h
 *    ROCm-specific data structures and API for Lasso Regression
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/neurondb_rocm_lasso.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_ROCM_LASSO_H
#define NEURONDB_ROCM_LASSO_H

#ifndef __HIPCC__
#include "postgres.h"
#include "utils/jsonb.h"
#include "ml_lasso_regression_internal.h"
#else
struct varlena;
typedef struct varlena bytea;
struct Jsonb;
struct LassoModel;
#endif

/* ROCm-specific Lasso Regression model header */
typedef struct NdbCudaLassoModelHeader
{
	int32 feature_dim;
	int32 n_samples;
	float intercept;
	float *coefficients; /* Array of feature_dim floats */
	double lambda;
	int32 max_iters;
	double r_squared;
	double mse;
	double mae;
} NdbCudaLassoModelHeader;

#ifdef __cplusplus
extern "C" {
#endif

extern int ndb_rocm_lasso_pack_model(const struct LassoModel *model,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr);

extern int ndb_rocm_lasso_train(const float *features,
	const double *targets,
	int n_samples,
	int feature_dim,
	const Jsonb *hyperparams,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr);

extern int ndb_rocm_lasso_predict(const bytea *model_data,
	const float *input,
	int feature_dim,
	double *prediction_out,
	char **errstr);

extern int ndb_rocm_lasso_evaluate(const bytea *model_data,
	const float *features,
	const double *targets,
	int n_samples,
	int feature_dim,
	double *mse_out,
	double *mae_out,
	double *rmse_out,
	double *r_squared_out,
	char **errstr);

#ifdef __cplusplus
}
#endif

#endif /* NEURONDB_ROCM_LASSO_H */
