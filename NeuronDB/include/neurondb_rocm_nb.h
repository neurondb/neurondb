/*-------------------------------------------------------------------------
 *
 * neurondb_rocm_nb.h
 *    ROCm-specific data structures and API for Naive Bayes
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/neurondb_rocm_nb.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_ROCM_NB_H
#define NEURONDB_ROCM_NB_H

#ifndef __HIPCC__
#include "postgres.h"
#include "utils/jsonb.h"
struct GaussianNBModel;
#else
#include <stdint.h>
typedef int32_t int32;
struct varlena;
typedef struct varlena bytea;
struct Jsonb;
struct GaussianNBModel;
#endif

/* ROCm-specific Naive Bayes model header */
typedef struct NdbCudaNbModelHeader
{
	int32 n_classes;
	int32 n_features;
	int32 n_samples;
} NdbCudaNbModelHeader;

#ifdef __cplusplus
extern "C" {
#endif

extern int ndb_rocm_nb_pack_model(const struct GaussianNBModel *model,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr);

extern int ndb_rocm_nb_train(const float *features,
	const double *labels,
	int n_samples,
	int feature_dim,
	int class_count,
	const Jsonb *hyperparams,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr);

extern int ndb_rocm_nb_predict(const bytea *model_data,
	const float *input,
	int feature_dim,
	int *class_out,
	double *probability_out,
	char **errstr);

extern int ndb_rocm_nb_predict_batch(const bytea *model_data,
	const float *features,
	int n_samples,
	int feature_dim,
	int *predictions_out,
	char **errstr);

extern int ndb_rocm_nb_evaluate_batch(const bytea *model_data,
	const float *features,
	const int *labels,
	int n_samples,
	int feature_dim,
	double *accuracy_out,
	double *precision_out,
	double *recall_out,
	double *f1_out,
	char **errstr);

/* CUDA kernel functions for training */
extern int ndb_rocm_nb_count_classes(const double *labels,
	int n_samples,
	int n_classes,
	int *class_counts);

extern int ndb_rocm_nb_compute_means(const float *features,
	const double *labels,
	int n_samples,
	int feature_dim,
	int n_classes,
	double *means,
	const int *class_counts);

extern int ndb_rocm_nb_compute_variances(const float *features,
	const double *labels,
	const double *means,
	int n_samples,
	int feature_dim,
	int n_classes,
	double *variances,
	const int *class_counts);

#ifdef __cplusplus
}
#endif

#endif /* NEURONDB_ROCM_NB_H */

