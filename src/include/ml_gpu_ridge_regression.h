/*-------------------------------------------------------------------------
 *
 * ml_gpu_ridge_regression.h
 *    Ridge Regression GPU helper interfaces.
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_ML_GPU_RIDGE_REGRESSION_H
#define NEURONDB_ML_GPU_RIDGE_REGRESSION_H

#include "postgres.h"
#include "utils/jsonb.h"

struct RidgeModel;

extern int	ndb_gpu_ridge_train(const float *features,
								const double *targets,
								int n_samples,
								int feature_dim,
								const Jsonb * hyperparams,
								bytea * *model_data,
								Jsonb * *metrics,
								char **errstr);

extern int	ndb_gpu_ridge_predict(const bytea * model_data,
								  const float *input,
								  int feature_dim,
								  double *prediction_out,
								  char **errstr);

extern int	ndb_gpu_ridge_pack_model(const struct RidgeModel *model,
									 bytea * *model_data,
									 Jsonb * *metrics,
									 char **errstr);

#endif							/* NEURONDB_ML_GPU_RIDGE_REGRESSION_H */
