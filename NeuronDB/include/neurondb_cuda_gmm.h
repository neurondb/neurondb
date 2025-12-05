/*-------------------------------------------------------------------------
 *
 * neurondb_cuda_gmm.h
 *    CUDA-specific data structures and API for Gaussian Mixture Model
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/neurondb_cuda_gmm.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_CUDA_GMM_H
#define NEURONDB_CUDA_GMM_H

#ifndef __CUDACC__
#include "postgres.h"
#include "utils/jsonb.h"
struct GMMModel;
#else
#include <stdint.h>
typedef int32_t int32;
struct varlena;
typedef struct varlena bytea;
struct Jsonb;
struct GMMModel;
#endif

/* CUDA-specific GMM model header */
typedef struct NdbCudaGmmModelHeader
{
	int32		n_components;
	int32		n_features;
	int32		n_samples;
	int32		max_iters;
	double		tolerance;
} NdbCudaGmmModelHeader;

#ifdef __cplusplus
extern "C"
{
#endif

extern int	ndb_cuda_gmm_pack_model(const struct GMMModel *model,
									bytea * *model_data,
									Jsonb * *metrics,
									char **errstr);

extern int	ndb_cuda_gmm_train(const float *features,
							   int n_samples,
							   int feature_dim,
							   int n_components,
							   const Jsonb * hyperparams,
							   bytea * *model_data,
							   Jsonb * *metrics,
							   char **errstr);

extern int	ndb_cuda_gmm_predict(const bytea * model_data,
								 const float *input,
								 int feature_dim,
								 int *cluster_out,
								 double *probability_out,
								 char **errstr);

/* CUDA kernel functions for training */
extern int	ndb_cuda_gmm_estep(const float *features,
							   const double *mixing_coeffs,
							   const double *means,
							   const double *variances,
							   int n_samples,
							   int feature_dim,
							   int n_components,
							   double *responsibilities);

extern int	ndb_cuda_gmm_mstep(const float *features,
							   const double *responsibilities,
							   int n_samples,
							   int feature_dim,
							   int n_components,
							   double *mixing_coeffs,
							   double *means,
							   double *variances);

#ifdef __cplusplus
}
#endif

#endif							/* NEURONDB_CUDA_GMM_H */
