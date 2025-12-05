/*-------------------------------------------------------------------------
 *
 * neurondb_rocm_svm.h
 *    ROCm-specific data structures and API for Support Vector Machine
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/neurondb_rocm_svm.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_ROCM_SVM_H
#define NEURONDB_ROCM_SVM_H

#include "postgres.h"

#ifndef __HIPCC__
#include "utils/jsonb.h"
#include "utils/bytea.h"
#else
/* Forward declarations for CUDA compilation */
struct Jsonb;
typedef struct bytea bytea;
#endif

#include "ml_svm_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ROCm-specific SVM model header */
typedef struct NdbCudaSvmModelHeader
{
	int32 feature_dim;
	int32 n_samples;
	int32 n_support_vectors;
	float bias;
	float *alphas; /* Array of n_support_vectors floats */
	float *support_vectors; /* Array of n_support_vectors * feature_dim floats */
	int32 *support_vector_indices; /* Array of n_support_vectors ints */
	double C;
	int32 max_iters;
} NdbCudaSvmModelHeader;

/* Host-side API */
extern int ndb_rocm_svm_pack_model(const SVMModel *model,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr);

extern int ndb_rocm_svm_train(const float *features,
	const double *labels,
	int n_samples,
	int feature_dim,
	const Jsonb *hyperparams,
	bytea **model_data,
	Jsonb **metrics,
	char **errstr);

extern int ndb_rocm_svm_predict(const bytea *model_data,
	const float *input,
	int feature_dim,
	int *class_out,
	double *confidence_out,
	char **errstr);

extern int ndb_rocm_svm_predict_batch(const bytea *model_data,
	const float *features,
	int n_samples,
	int feature_dim,
	int *predictions_out,
	char **errstr);

extern int ndb_rocm_svm_evaluate_batch(const bytea *model_data,
	const float *features,
	const int *labels,
	int n_samples,
	int feature_dim,
	double *accuracy_out,
	double *precision_out,
	double *recall_out,
	double *f1_out,
	char **errstr);

/* CUDA kernel declarations */
extern void ndb_rocm_svm_linear_kernel_kernel(const float *x,
	const float *y,
	int feature_dim,
	float *result);

extern void ndb_rocm_svm_predict_kernel(const float *input,
	const float *support_vectors,
	const float *alphas,
	const double *labels,
	float bias,
	int n_support_vectors,
	int feature_dim,
	float *prediction);

/* CUDA kernel launcher functions for SMO training */
extern int ndb_rocm_svm_launch_compute_kernel_row(const float *features,
	int n_samples,
	int feature_dim,
	int row_idx,
	float *kernel_row);

extern int ndb_rocm_svm_launch_compute_errors(const float *alphas,
	const double *labels,
	const float *kernel_matrix,
	float bias,
	int n_samples,
	float *errors);

extern int ndb_rocm_svm_launch_update_errors(const float *kernel_row,
	float delta_alpha,
	float label_i,
	int n_samples,
	float *errors);

#ifdef __cplusplus
}
#endif

#endif /* NEURONDB_ROCM_SVM_H */

