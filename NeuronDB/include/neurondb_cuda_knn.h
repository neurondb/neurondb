/*-------------------------------------------------------------------------
 *
 * neurondb_cuda_knn.h
 *    CUDA-specific data structures and API for K-Nearest Neighbors
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/neurondb_cuda_knn.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_CUDA_KNN_H
#define NEURONDB_CUDA_KNN_H

#ifndef __CUDACC__
#include "postgres.h"
#include "utils/jsonb.h"
struct KNNModel;
#else
#include <stdint.h>
typedef int32_t int32;
struct varlena;
typedef struct varlena bytea;
struct Jsonb;
struct KNNModel;
#endif

/* CUDA-specific KNN model header */
typedef struct NdbCudaKnnModelHeader
{
	int32 n_samples;
	int32 n_features;
	int32 k;
	int32 task_type;  /* 0=classification, 1=regression */
} NdbCudaKnnModelHeader;

#ifdef __cplusplus
extern "C" {
#endif

extern int ndb_cuda_knn_pack(const struct KNNModel *model,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr);

extern int ndb_cuda_knn_train(const float *features,
	const double *labels,
	int n_samples,
	int feature_dim,
	int k,
	int task_type,
	const Jsonb *hyperparams,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr);

extern int ndb_cuda_knn_predict(const bytea *model_data,
	const float *input,
	int feature_dim,
	double *prediction_out,
	char **errstr);

/* CUDA kernel functions for prediction */
extern int ndb_cuda_knn_compute_distances(const float *query,
	const float *training_features,
	int n_samples,
	int feature_dim,
	float *distances);

extern int ndb_cuda_knn_find_top_k(const float *distances,
	const double *labels,
	int n_samples,
	int k,
	int task_type,
	double *prediction_out);

extern int ndb_cuda_knn_predict_batch(const bytea *model_data,
	const float *features,
	int n_samples,
	int feature_dim,
	int *predictions_out,
	char **errstr);

extern int ndb_cuda_knn_evaluate_batch(const bytea *model_data,
	const float *features,
	const double *labels,
	int n_samples,
	int feature_dim,
	double *accuracy_out,
	double *precision_out,
	double *recall_out,
	double *f1_out,
	char **errstr);

#ifdef __cplusplus
}
#endif

#endif /* NEURONDB_CUDA_KNN_H */

