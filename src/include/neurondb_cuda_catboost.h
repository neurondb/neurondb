/*-------------------------------------------------------------------------
 *
 * neurondb_cuda_catboost.h
 *    CUDA-specific data structures and API for CatBoost
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/neurondb_cuda_catboost.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_CUDA_CATBOOST_H
#define NEURONDB_CUDA_CATBOOST_H

#ifndef __CUDACC__
#include "postgres.h"
#include "utils/jsonb.h"
#else
struct varlena;
typedef struct varlena bytea;
struct Jsonb;
struct CatBoostModel;
#endif

/* CUDA-specific CatBoost model header */
typedef struct NdbCudaCatBoostModelHeader
{
	int32		feature_dim;
	int32		n_samples;
	int32		iterations;
	int32		depth;
	float		learning_rate;
	char		loss_function[32];
} NdbCudaCatBoostModelHeader;

#ifdef __cplusplus
extern "C"
{
#endif

extern int	ndb_cuda_catboost_pack_model(const void *model,
										bytea * *model_data,
										Jsonb * *metrics,
										char **errstr);

extern int	ndb_cuda_catboost_train(const float *features,
									 const double *labels,
									 int n_samples,
									 int feature_dim,
									 const Jsonb * hyperparams,
									 bytea * *model_data,
									 Jsonb * *metrics,
									 char **errstr);

extern int	ndb_cuda_catboost_predict(const bytea * model_data,
									  const float *input,
									  int feature_dim,
									  double *prediction_out,
									  char **errstr);

#ifdef __cplusplus
}
#endif

#endif							/* NEURONDB_CUDA_CATBOOST_H */

