/*-------------------------------------------------------------------------
 *
 * neurondb_cuda_xgboost.h
 *    CUDA-specific data structures and API for XGBoost
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/neurondb_cuda_xgboost.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_CUDA_XGBOOST_H
#define NEURONDB_CUDA_XGBOOST_H

#ifndef __CUDACC__
#include "postgres.h"
#include "utils/jsonb.h"
#else
struct varlena;
typedef struct varlena bytea;
struct Jsonb;
struct XGBoostModel;
#endif

/* CUDA-specific XGBoost model header */
typedef struct NdbCudaXGBoostModelHeader
{
	int32		feature_dim;
	int32		n_samples;
	int32		n_estimators;
	int32		max_depth;
	float		learning_rate;
	char		objective[32];
} NdbCudaXGBoostModelHeader;

#ifdef __cplusplus
extern "C"
{
#endif

extern int	ndb_cuda_xgboost_pack_model(const void *model,
									   bytea * *model_data,
									   Jsonb * *metrics,
									   char **errstr);

extern int	ndb_cuda_xgboost_train(const float *features,
									const double *labels,
									int n_samples,
									int feature_dim,
									const Jsonb * hyperparams,
									bytea * *model_data,
									Jsonb * *metrics,
									char **errstr);

extern int	ndb_cuda_xgboost_predict(const bytea * model_data,
									 const float *input,
									 int feature_dim,
									 double *prediction_out,
									 char **errstr);

#ifdef __cplusplus
}
#endif

#endif							/* NEURONDB_CUDA_XGBOOST_H */

