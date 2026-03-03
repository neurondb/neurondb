/*-------------------------------------------------------------------------
 *
 * neurondb_gpu_bridge.h
 *    Helpers that connect SQL-visible ML APIs with GPU model registry.
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_GPU_BRIDGE_H
#define NEURONDB_GPU_BRIDGE_H

#include "ml_catalog.h"
#include "neurondb_gpu_model.h"

typedef struct MLGpuTrainResult
{
	MLCatalogModelSpec spec;
	bytea	   *payload;
	Jsonb	   *metadata;
	Jsonb	   *metrics;
	int32		model_id;
} MLGpuTrainResult;

extern bool ndb_gpu_try_train_model(const char *algorithm,
									const char *project_name,
									const char *model_name,
									const char *training_table,
									const char *training_column,
									const char *const *feature_columns,
									int feature_count,
									Jsonb * hyperparameters,
									const float *feature_matrix,
									const double *label_vector,
									int sample_count,
									int feature_dim,
									int class_count,
									MLGpuTrainResult *result,
									char **errstr);

extern void ndb_gpu_free_train_result(MLGpuTrainResult *result);

#endif							/* NEURONDB_GPU_BRIDGE_H */
