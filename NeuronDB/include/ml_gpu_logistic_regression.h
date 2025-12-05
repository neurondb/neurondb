/*-------------------------------------------------------------------------
 *
 * ml_gpu_logistic_regression.h
 *    GPU support for Logistic Regression
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/ml_gpu_logistic_regression.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_ML_GPU_LOGISTIC_REGRESSION_H
#define NEURONDB_ML_GPU_LOGISTIC_REGRESSION_H

#include "postgres.h"
#include "utils/jsonb.h"

extern int	ndb_gpu_lr_train(const float *features,
							 const double *labels,
							 int n_samples,
							 int feature_dim,
							 const Jsonb * hyperparams,
							 bytea * *model_data,
							 Jsonb * *metrics,
							 char **errstr);

extern int	ndb_gpu_lr_predict(const bytea * model_data,
							   const float *input,
							   int feature_dim,
							   double *probability_out,
							   char **errstr);

extern int	ndb_gpu_lr_pack_model(const struct LRModel *model,
								  bytea * *model_data,
								  Jsonb * *metrics,
								  char **errstr);

#endif							/* NEURONDB_ML_GPU_LOGISTIC_REGRESSION_H */
