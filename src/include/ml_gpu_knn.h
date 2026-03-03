/*-------------------------------------------------------------------------
 *
 * ml_gpu_knn.h
 *    K-Nearest Neighbors GPU helper interfaces.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/ml_gpu_knn.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_ML_GPU_KNN_H
#define NEURONDB_ML_GPU_KNN_H

#include "postgres.h"
#include "utils/jsonb.h"

struct KNNModel;

extern int	ndb_gpu_knn_train(const float *features,
							  const double *labels,
							  int n_samples,
							  int feature_dim,
							  int k,
							  int task_type,
							  const Jsonb * hyperparams,
							  bytea * *model_data,
							  Jsonb * *metrics,
							  char **errstr);

extern int	ndb_gpu_knn_predict(const bytea * model_data,
								const float *input,
								int feature_dim,
								double *prediction_out,
								char **errstr);

extern int	ndb_gpu_knn_pack(const struct KNNModel *model,
							 bytea * *model_data,
							 Jsonb * *metrics,
							 char **errstr);

#endif							/* NEURONDB_ML_GPU_KNN_H */






