/*-------------------------------------------------------------------------
 *
 * ml_gpu_decision_tree.h
 *    Decision Tree GPU helper interfaces.
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_ML_GPU_DECISION_TREE_H
#define NEURONDB_ML_GPU_DECISION_TREE_H

#include "postgres.h"
#include "utils/jsonb.h"

struct DTModel;

extern int	ndb_gpu_dt_train(const float *features,
							 const double *labels,
							 int n_samples,
							 int feature_dim,
							 const Jsonb * hyperparams,
							 bytea * *model_data,
							 Jsonb * *metrics,
							 char **errstr);

extern int	ndb_gpu_dt_predict(const bytea * model_data,
							   const float *input,
							   int feature_dim,
							   double *prediction_out,
							   char **errstr);

extern int	ndb_gpu_dt_pack_model(const struct DTModel *model,
								  bytea * *model_data,
								  Jsonb * *metrics,
								  char **errstr);

#endif							/* NEURONDB_ML_GPU_DECISION_TREE_H */
