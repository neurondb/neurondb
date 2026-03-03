/*-------------------------------------------------------------------------
 *
 * neurondb_gpu_model.h
 *    GPU-resident model abstractions and registry.
 *
 * Defines a backend-agnostic interface that allows ML algorithms to expose
 * native GPU training, prediction, serialization, and teardown.
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_GPU_MODEL_H
#define NEURONDB_GPU_MODEL_H

#include "postgres.h"
#include "utils/jsonb.h"

struct MLGpuModel;
struct MLGpuTrainSpec;
struct MLGpuEvalSpec;
struct MLGpuContext;
struct MLGpuMetrics;
struct ndb_gpu_backend;

typedef bool (*MLGpuTrainFn) (struct MLGpuModel *model,
							  const struct MLGpuTrainSpec *spec,
							  char **errstr);
typedef bool (*MLGpuPredictFn) (const struct MLGpuModel *model,
								const float *input,
								int input_dim,
								float *output,
								int output_dim,
								char **errstr);
typedef bool (*MLGpuEvaluateFn) (const struct MLGpuModel *model,
								 const struct MLGpuEvalSpec *spec,
								 struct MLGpuMetrics *out,
								 char **errstr);
typedef bool (*MLGpuSerializeFn) (const struct MLGpuModel *model,
								  bytea * *payload_out,
								  Jsonb * *metadata_out,
								  char **errstr);
typedef bool (*MLGpuDeserializeFn) (struct MLGpuModel *model,
									const bytea * payload,
									const Jsonb * metadata,
									char **errstr);
typedef void (*MLGpuDestroyFn) (struct MLGpuModel *model);

typedef struct MLGpuModelOps
{
	const char *algorithm;
	MLGpuTrainFn train;
	MLGpuPredictFn predict;
	MLGpuEvaluateFn evaluate;
	MLGpuSerializeFn serialize;
	MLGpuDeserializeFn deserialize;
	MLGpuDestroyFn destroy;
} MLGpuModelOps;

typedef struct MLGpuModel
{
	const MLGpuModelOps *ops;
	void	   *backend_state;
	int32		catalog_id;
	char	   *model_name;
	bool		is_gpu_resident;
	bool		gpu_ready;
} MLGpuModel;

typedef struct MLGpuTrainSpec
{
	const char *algorithm;
	const char *training_table;
	const char *training_column;
	const char *const *feature_columns;
	int			feature_count;
	const char *project_name;
	const char *model_name;
	Jsonb	   *hyperparameters;
	struct MLGpuContext *context;
	int32		expected_features;
	int32		expected_classes;
	const float *feature_matrix;
	const double *label_vector;
	int32		sample_count;
	int32		feature_dim;
	int32		class_count;
} MLGpuTrainSpec;

typedef struct MLGpuEvalSpec
{
	const char *evaluation_table;
	const char *label_column;
	struct MLGpuContext *context;
	Jsonb	   *options;
} MLGpuEvalSpec;

typedef struct MLGpuMetrics
{
	Jsonb	   *payload;
} MLGpuMetrics;

typedef struct MLGpuContext
{
	const char *backend_name;
	const struct ndb_gpu_backend *backend;
	int			device_id;
	void	   *stream_handle;
	void	   *scratch_pool;
	MemoryContext memory_ctx;
} MLGpuContext;

extern bool ndb_gpu_register_model_ops(const MLGpuModelOps *ops);
extern const MLGpuModelOps *ndb_gpu_lookup_model_ops(const char *algorithm);
extern void ndb_gpu_clear_model_registry(void);

#endif							/* NEURONDB_GPU_MODEL_H */
