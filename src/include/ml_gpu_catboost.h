/*-------------------------------------------------------------------------
 *
 * ml_gpu_catboost.h
 *    GPU support for CatBoost
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/ml_gpu_catboost.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_ML_GPU_CATBOOST_H
#define NEURONDB_ML_GPU_CATBOOST_H

#include "postgres.h"
#include "utils/jsonb.h"

struct CatBoostModel;

extern int	ndb_gpu_catboost_train(const float *features,
								   const double *labels,
								   int n_samples,
								   int feature_dim,
								   const Jsonb * hyperparams,
								   bytea * *model_data,
								   Jsonb * *metrics,
								   char **errstr);

extern int	ndb_gpu_catboost_predict(const bytea * model_data,
									 const float *input,
									 int feature_dim,
									 double *prediction_out,
									 char **errstr);

extern int	ndb_gpu_catboost_pack_model(const struct CatBoostModel *model,
										bytea * *model_data,
										Jsonb * *metrics,
										char **errstr);

#endif							/* NEURONDB_ML_GPU_CATBOOST_H */





