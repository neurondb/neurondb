/*-------------------------------------------------------------------------
 *
 * neurondb_cuda_linreg.h
 *    CUDA-specific data structures and API for Linear Regression
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/neurondb_cuda_linreg.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_CUDA_LINREG_H
#define NEURONDB_CUDA_LINREG_H

#include "postgres.h"

#ifndef __CUDACC__
#include "utils/jsonb.h"
#include "utils/bytea.h"
#else
#include <stdint.h>
typedef int32_t int32;
/* Forward declarations for CUDA compilation */
struct Jsonb;
typedef struct bytea bytea;
#endif

#include "ml_linear_regression_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* CUDA-specific Linear Regression model header */
typedef struct NdbCudaLinRegModelHeader
{
	int32 feature_dim;
	int32 n_samples;
	float intercept;
	float *coefficients; /* Array of feature_dim floats */
	double r_squared;
	double mse;
	double mae;
} NdbCudaLinRegModelHeader;

/* Host-side API */
extern int ndb_cuda_linreg_pack_model(const LinRegModel *model,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr);

extern int ndb_cuda_linreg_train(const float *features,
	const double *targets,
	int n_samples,
	int feature_dim,
	const Jsonb *hyperparams,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr);

extern int ndb_cuda_linreg_predict(const bytea *model_data,
	const float *input,
	int feature_dim,
	double *prediction_out,
	char **errstr);

extern int ndb_cuda_linreg_evaluate(const bytea *model_data,
	const float *features,
	const double *targets,
	int n_samples,
	int feature_dim,
	double *mse_out,
	double *mae_out,
	double *rmse_out,
	double *r_squared_out,
	char **errstr);

/* CUDA kernel declarations */
extern void ndb_cuda_linreg_compute_xtx_kernel(const float *features,
	const double *targets,
	int n_samples,
	int feature_dim,
	int dim_with_intercept,
	double *XtX,
	double *Xty);

extern void ndb_cuda_linreg_predict_kernel(const float *input,
	const float *coefficients,
	float intercept,
	int feature_dim,
	float *prediction);

#ifdef __cplusplus
}
#endif

#endif /* NEURONDB_CUDA_LINREG_H */
