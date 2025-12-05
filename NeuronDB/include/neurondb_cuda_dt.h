/*-------------------------------------------------------------------------
 *
 * neurondb_cuda_dt.h
 *    CUDA-specific data structures and API for Decision Tree
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/neurondb_cuda_dt.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_CUDA_DT_H
#define NEURONDB_CUDA_DT_H

#ifndef __CUDACC__
#include "postgres.h"
#include "utils/jsonb.h"
#include "ml_decision_tree_internal.h"
#else
struct varlena;
typedef struct varlena bytea;
struct Jsonb;
struct DTModel;
#endif

/* CUDA-specific Decision Tree node structure */
typedef struct NdbCudaDtNode
{
	int			feature_idx;
	float		threshold;
	int			left_child;
	int			right_child;
	float		value;
	bool		is_leaf;
} NdbCudaDtNode;

/* CUDA-specific Decision Tree model header */
typedef struct NdbCudaDtModelHeader
{
	int32		feature_dim;
	int32		n_samples;
	int32		max_depth;
	int32		min_samples_split;
	int32		node_count;
} NdbCudaDtModelHeader;

#ifdef __cplusplus
extern "C"
{
#endif

extern int	ndb_cuda_dt_pack_model(const struct DTModel *model,
								   bytea * *model_data,
								   Jsonb * *metrics,
								   char **errstr);

extern int	ndb_cuda_dt_train(const float *features,
							  const double *labels,
							  int n_samples,
							  int feature_dim,
							  const Jsonb * hyperparams,
							  bytea * *model_data,
							  Jsonb * *metrics,
							  char **errstr);

extern int	ndb_cuda_dt_predict(const bytea * model_data,
								const float *input,
								int feature_dim,
								double *prediction_out,
								char **errstr);

extern int	ndb_cuda_dt_predict_batch(const bytea * model_data,
									  const float *features,
									  int n_samples,
									  int feature_dim,
									  int *predictions_out,
									  char **errstr);

extern int	ndb_cuda_dt_evaluate_batch(const bytea * model_data,
									   const float *features,
									   const int *labels,
									   int n_samples,
									   int feature_dim,
									   double *accuracy_out,
									   double *precision_out,
									   double *recall_out,
									   double *f1_out,
									   char **errstr);

/* CUDA kernel launchers */
extern int	ndb_cuda_dt_launch_feature_stats(const float *features,
											 const int *indices,
											 int n_samples,
											 int feature_dim,
											 int feature_idx,
											 float *min_val,
											 float *max_val,
											 double *sum,
											 double *sumsq);

extern int	ndb_cuda_dt_launch_split_counts_classification(const float *features,
														   const int *labels,
														   const int *indices,
														   int n_samples,
														   int feature_dim,
														   int feature_idx,
														   float threshold,
														   int class_count,
														   int *left_counts,
														   int *right_counts);

extern int	ndb_cuda_dt_launch_split_stats_regression(const float *features,
													  const double *labels,
													  const int *indices,
													  int n_samples,
													  int feature_dim,
													  int feature_idx,
													  float threshold,
													  double *left_sum,
													  double *left_sumsq,
													  int *left_count,
													  double *right_sum,
													  double *right_sumsq,
													  int *right_count);

#ifdef __cplusplus
}
#endif

#endif							/* NEURONDB_CUDA_DT_H */
