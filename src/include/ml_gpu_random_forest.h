/*-------------------------------------------------------------------------
 *
 * ml_gpu_random_forest.h
 *    Random Forest GPU helper interfaces.
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_ML_GPU_RANDOM_FOREST_H
#define NEURONDB_ML_GPU_RANDOM_FOREST_H

#include "postgres.h"
#include "utils/jsonb.h"

struct RFModel;

extern int	ndb_gpu_rf_train(const float *features,
							 const double *labels,
							 int n_samples,
							 int feature_dim,
							 int class_count,
							 const Jsonb * hyperparams,
							 bytea * *model_data,
							 Jsonb * *metrics,
							 char **errstr);

extern int	ndb_gpu_rf_predict(const bytea * model_data,
							   const float *input,
							   int feature_dim,
							   int *class_out,
							   char **errstr);

extern int	ndb_gpu_rf_pack_model(const struct RFModel *model,
								  bytea * *model_data,
								  Jsonb * *metrics,
								  char **errstr);

#endif							/* NEURONDB_ML_GPU_RANDOM_FOREST_H */
