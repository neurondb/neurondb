/*-------------------------------------------------------------------------
 *
 * ml_gpu_xgboost.h
 *    GPU support for XGBoost
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/ml_gpu_xgboost.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_ML_GPU_XGBOOST_H
#define NEURONDB_ML_GPU_XGBOOST_H

#include "postgres.h"
#include "utils/jsonb.h"

struct XGBoostModel;

extern int	ndb_gpu_xgboost_train(const float *features,
								  const double *labels,
								  int n_samples,
								  int feature_dim,
								  const Jsonb * hyperparams,
								  bytea * *model_data,
								  Jsonb * *metrics,
								  char **errstr);

extern int	ndb_gpu_xgboost_predict(const bytea * model_data,
									const float *input,
									int feature_dim,
									double *prediction_out,
									char **errstr);

extern int	ndb_gpu_xgboost_pack_model(const struct XGBoostModel *model,
									   bytea * *model_data,
									   Jsonb * *metrics,
									   char **errstr);

#endif							/* NEURONDB_ML_GPU_XGBOOST_H */





