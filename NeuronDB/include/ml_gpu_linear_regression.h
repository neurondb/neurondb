/*-------------------------------------------------------------------------
 *
 * ml_gpu_linear_regression.h
 *    GPU support for Linear Regression
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/ml_gpu_linear_regression.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_ML_GPU_LINEAR_REGRESSION_H
#define NEURONDB_ML_GPU_LINEAR_REGRESSION_H

#include "postgres.h"
#include "utils/jsonb.h"

extern int	ndb_gpu_linreg_train(const float *features,
								 const double *targets,
								 int n_samples,
								 int feature_dim,
								 const Jsonb * hyperparams,
								 bytea * *model_data,
								 Jsonb * *metrics,
								 char **errstr);

extern int	ndb_gpu_linreg_predict(const bytea * model_data,
								   const float *input,
								   int feature_dim,
								   double *prediction_out,
								   char **errstr);

extern int	ndb_gpu_linreg_pack_model(const struct LinRegModel *model,
									  bytea * *model_data,
									  Jsonb * *metrics,
									  char **errstr);

#endif							/* NEURONDB_ML_GPU_LINEAR_REGRESSION_H */
