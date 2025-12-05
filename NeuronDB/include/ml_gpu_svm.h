/*-------------------------------------------------------------------------
 *
 * ml_gpu_svm.h
 *    GPU support for Support Vector Machine
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/ml_gpu_svm.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_ML_GPU_SVM_H
#define NEURONDB_ML_GPU_SVM_H

#include "postgres.h"
#include "utils/jsonb.h"

extern int	ndb_gpu_svm_train(const float *features,
							  const double *labels,
							  int n_samples,
							  int feature_dim,
							  const Jsonb * hyperparams,
							  bytea * *model_data,
							  Jsonb * *metrics,
							  char **errstr);

extern int	ndb_gpu_svm_predict(const bytea * model_data,
								const float *input,
								int feature_dim,
								int *class_out,
								double *confidence_out,
								char **errstr);

/* Convenience wrapper that returns prediction as double (for compatibility) */
extern int	ndb_gpu_svm_predict_double(const bytea * model_data,
									   const float *input,
									   int feature_dim,
									   double *prediction_out,
									   char **errstr);

extern int	ndb_gpu_svm_pack_model(const struct SVMModel *model,
								   bytea * *model_data,
								   Jsonb * *metrics,
								   char **errstr);

#endif							/* NEURONDB_ML_GPU_SVM_H */
