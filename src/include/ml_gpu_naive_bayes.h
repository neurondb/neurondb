/*-------------------------------------------------------------------------
 *
 * ml_gpu_naive_bayes.h
 *    Naive Bayes GPU helper interfaces.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/ml_gpu_naive_bayes.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_ML_GPU_NAIVE_BAYES_H
#define NEURONDB_ML_GPU_NAIVE_BAYES_H

#include "postgres.h"
#include "utils/jsonb.h"

struct GaussianNBModel;

extern int	ndb_gpu_nb_train(const float *features,
							 const double *labels,
							 int n_samples,
							 int feature_dim,
							 int class_count,
							 const Jsonb * hyperparams,
							 bytea * *model_data,
							 Jsonb * *metrics,
							 char **errstr);

extern int	ndb_gpu_nb_predict(const bytea * model_data,
							   const float *input,
							   int feature_dim,
							   int *class_out,
							   double *probability_out,
							   char **errstr);

extern int	ndb_gpu_nb_pack_model(const struct GaussianNBModel *model,
								  bytea * *model_data,
								  Jsonb * *metrics,
								  char **errstr);

#endif							/* NEURONDB_ML_GPU_NAIVE_BAYES_H */






