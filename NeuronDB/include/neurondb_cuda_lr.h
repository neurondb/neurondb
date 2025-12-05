/*-------------------------------------------------------------------------
 *
 * neurondb_cuda_lr.h
 *    CUDA-specific data structures and API for Logistic Regression
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/neurondb_cuda_lr.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_CUDA_LR_H
#define NEURONDB_CUDA_LR_H

#ifndef __CUDACC__
#include "postgres.h"
#include "utils/jsonb.h"
#include "ml_logistic_regression_internal.h"
#ifdef NDB_GPU_CUDA
#include "neurondb_cuda_runtime.h"
#endif
#else
struct varlena;
typedef struct varlena bytea;
struct Jsonb;
struct LRModel;
#endif

typedef struct NdbCudaLrModelHeader
{
	int			feature_dim;
	int			n_samples;
	int			max_iters;
	double		learning_rate;
	double		lambda;
	double		bias;
} NdbCudaLrModelHeader;

#ifdef __cplusplus
extern "C"
{
#endif

extern int	ndb_cuda_lr_pack_model(const struct LRModel *model,
								   bytea * *model_data,
								   Jsonb * *metrics,
								   char **errstr);

extern int	ndb_cuda_lr_train(const float *features,
							  const double *labels,
							  int n_samples,
							  int feature_dim,
							  const Jsonb * hyperparams,
							  bytea * *model_data,
							  Jsonb * *metrics,
							  char **errstr);

extern int	ndb_cuda_lr_predict(const bytea * model_data,
								const float *input,
								int feature_dim,
								double *probability_out,
								char **errstr);

extern int	ndb_cuda_lr_forward_pass(const float *features,
									 const double *weights,
									 double bias,
									 int n_samples,
									 int feature_dim,
									 double *outputs);

extern int	ndb_cuda_lr_forward_pass_gpu(const float *d_features,
										 const float *d_weights,
										 float bias,
										 int n_samples,
										 int feature_dim,
										 double *d_outputs);

extern int	ndb_cuda_lr_compute_gradients(const float *features,
										  const double *labels,
										  const double *predictions,
										  int n_samples,
										  int feature_dim,
										  double *grad_weights,
										  double *grad_bias);

extern int	ndb_cuda_lr_sigmoid(const double *inputs, int n, double *outputs);

/* GPU wrapper functions return CUDA error code (int) */
extern int	ndb_cuda_lr_sigmoid_gpu(const double *d_in, int n, double *d_out);

extern int	ndb_cuda_lr_compute_errors_gpu(const double *d_predictions,
										   const double *d_labels,
										   float *d_errors,
										   int n);

extern int	ndb_cuda_lr_convert_z_add_bias_gpu(const float *d_z_float,
											   double bias,
											   int n,
											   double *d_z_double);

extern int	ndb_cuda_lr_reduce_errors_bias_gpu(const float *d_errors,
											   int n,
											   double *d_error_sum);

extern int	ndb_cuda_lr_update_weights_gpu(float *d_weights,
										   const float *d_grad_weights,
										   float learning_rate,
										   float lambda,
										   int feature_dim,
										   double *d_bias,
										   double grad_bias);

/* Kernel launcher from gpu_lr_kernels.cu */
#ifndef __CUDACC__
extern cudaError_t launch_lr_eval_kernel(const float *features,
										 const double *labels,
										 const double *weights,
										 double bias,
										 double threshold,
										 int n_samples,
										 int feature_dim,
										 long long *tp_out,
										 long long *tn_out,
										 long long *fp_out,
										 long long *fn_out,
										 double *log_loss_out,
										 long long *count_out);
#endif

extern int	ndb_cuda_lr_evaluate(const bytea * model_data,
								 const float *features,
								 const double *labels,
								 int n_samples,
								 int feature_dim,
								 double threshold,
								 double *accuracy_out,
								 double *precision_out,
								 double *recall_out,
								 double *f1_out,
								 double *log_loss_out,
								 char **errstr);

#ifdef __cplusplus
}
#endif

#endif							/* NEURONDB_CUDA_LR_H */
