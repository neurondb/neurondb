/*-------------------------------------------------------------------------
 *
 * neurondb_gpu.h
 *      GPU acceleration - core structures, API, and integration for NeurondB.
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 * IDENTIFICATION
 *      include/neurondb_gpu.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_GPU_H
#define NEURONDB_GPU_H

#include "postgres.h"
#include "utils/timestamp.h"
#include "utils/jsonb.h"
#include <stdint.h>
#include <stdbool.h>
#include "neurondb_gpu_types.h"
#include "neurondb_gpu_backend.h"

/* Backwards-compatibility aliases for legacy enum names */
typedef NDBGpuBackendKind GPUBackend;

#define GPU_BACKEND_NONE NDB_GPU_BACKEND_NONE
#define GPU_BACKEND_CUDA NDB_GPU_BACKEND_CUDA
#define GPU_BACKEND_ROCM NDB_GPU_BACKEND_ROCM
#define GPU_BACKEND_OPENCL NDB_GPU_BACKEND_NONE
#define GPU_BACKEND_METAL NDB_GPU_BACKEND_METAL

/* GPU statistics for monitoring */
typedef struct GPUStats
{
	int64		queries_executed;
	int64		fallback_count;
	float8		total_gpu_time_ms;
	float8		total_cpu_time_ms;
	float8		avg_latency_ms;
	TimestampTz last_reset;
} GPUStats;

/* GPU device information struct */
typedef struct GPUDeviceInfo
{
	int			device_id;
	char		name[256];
	int64		total_memory_mb;
	int64		free_memory_mb;
	int			compute_major;
	int			compute_minor;
	bool		is_available;
} GPUDeviceInfo;

/* GUC variables are now in neurondb_guc.h */
#include "neurondb_guc.h"

/*==================*/
/* Initialization, shutdown, status, config */
extern void neurondb_gpu_init(void);
extern void neurondb_gpu_shutdown(void);
extern bool neurondb_gpu_is_available(void);
extern bool ndb_gpu_kernel_enabled(const char *kernel_name);
extern void ndb_gpu_init_if_needed(void);
extern int	ndb_gpu_runtime_init(int *device_id);
extern void ndb_gpu_mem_pool_init(int size_mb);
extern void ndb_gpu_streams_init(int num_streams);

/* Backend and device info */
extern GPUBackend neurondb_gpu_get_backend(void);
extern int	neurondb_gpu_get_device_count(void);
extern GPUDeviceInfo *neurondb_gpu_get_device_info(int device_id);
extern void neurondb_gpu_set_device(int device_id);

/* Stats */
extern GPUStats *neurondb_gpu_get_stats(void);
extern void neurondb_gpu_reset_stats(void);
extern void ndb_gpu_stats_record(bool used_gpu,
								 double gpu_ms,
								 double cpu_ms,
								 bool fallback);

/*==================*/
/* Vector math: distance & similarity */
extern float
			neurondb_gpu_l2_distance(const float *vec1, const float *vec2, int dim);
extern float
			neurondb_gpu_cosine_distance(const float *vec1, const float *vec2, int dim);
extern float
			neurondb_gpu_inner_product(const float *vec1, const float *vec2, int dim);

/* Batch operations */
extern void neurondb_gpu_batch_l2_distance(const float *queries,
										   const float *vectors,
										   float *results,
										   int num_queries,
										   int num_vectors,
										   int dim);

extern void neurondb_gpu_batch_cosine_distance(const float *queries,
											   const float *vectors,
											   float *results,
											   int num_queries,
											   int num_vectors,
											   int dim);

/*==================*/
/* Quantization (for compression & faster search) */
extern void
			neurondb_gpu_quantize_int8(const float *input, int8 * output, int count);
extern void
			neurondb_gpu_quantize_int4(const float *input, unsigned char *output, int count);
extern void
			neurondb_gpu_quantize_fp16(const float *input, void *output, int count);
extern void
			neurondb_gpu_quantize_fp8_e4m3(const float *input, unsigned char *output, int count);
extern void
			neurondb_gpu_quantize_fp8_e5m2(const float *input, unsigned char *output, int count);
extern void
			neurondb_gpu_quantize_binary(const float *input, uint8 * output, int count);

/*==================*/
/* Clustering and ML algorithms */
extern void neurondb_gpu_kmeans(const float *vectors,
								int num_vectors,
								int dim,
								int k,
								int max_iters,
								float *centroids,
								int *assignments);
extern void neurondb_gpu_dbscan(const float *vectors,
								int num_vectors,
								int dim,
								float eps,
								int min_points,
								int *cluster_ids);

/* Neural network inference */
extern void neurondb_gpu_onnx_inference(void *model_handle,
										const float *input,
										int input_size,
										float *output,
										int output_size);

/*==================*/
/* Random Forest (Phase-1: GPU-assisted best split for binary labels) */
extern bool neurondb_gpu_rf_best_split_binary(const float *feature_values,
											  const uint8_t * labels01,
											  int n,
											  double *best_threshold,
											  double *best_gini,
											  int *left_count,
											  int *right_count);

extern bool neurondb_gpu_rf_predict(const void *rf_hdr,
									const void *trees,
									const void *nodes,
									int node_capacity,
									const float *x,
									int n_features,
									int *class_out,
									char **errstr);

/*==================*/
/* Hugging Face / LLM (GPU-accelerated inference) */
extern int	neurondb_gpu_hf_embed(const char *model_name,
								  const char *text,
								  float **vec_out,
								  int *dim_out,
								  char **errstr);
extern int	neurondb_gpu_hf_complete(const char *model_name,
									 const char *prompt,
									 const char *params_json,
									 char **text_out,
									 char **errstr);
extern int	neurondb_gpu_hf_rerank(const char *model_name,
								   const char *query,
								   const char **docs,
								   int ndocs,
								   float **scores_out,
								   char **errstr);

/* Batch operations */
#ifdef NDB_GPU_CUDA
#include "neurondb_cuda_hf.h"
extern int	neurondb_gpu_hf_complete_batch(const char *model_name,
										   const char **prompts,
										   int num_prompts,
										   const char *params_json,
										   NdbCudaHfBatchResult *results,
										   char **errstr);
#endif
extern int	neurondb_gpu_hf_rerank_batch(const char *model_name,
										 const char **queries,
										 const char ***docs_array,
										 int *ndocs_array,
										 int num_queries,
										 float ***scores_out,
										 int **nscores_out,
										 char **errstr);

/*==================*/
/* Implementation for device-level management (optional: not part of minimal prototype) */
#ifdef NDB_GPU_INTERNAL
/* Memory pool and stream management functions for internal use */
void	   *ndb_gpu_allocate(size_t bytes);
void		ndb_gpu_free(void *ptr);
void		ndb_gpu_memcpy_to_device(void *dst, const void *src, size_t bytes);
void		ndb_gpu_memcpy_from_device(void *dst, const void *src, size_t bytes);
void		ndb_gpu_synchronize(void);
#endif

extern void neurondb_gpu_matmul(const float *A,
								const float *B,
								float *C,
								int m,
								int n,
								int k,
								bool use_gpu);
extern void neurondb_gpu_vector_add(const float *a,
									const float *b,
									float *result,
									int n,
									bool use_gpu);
extern void neurondb_gpu_activation_relu(const float *input,
										 float *output,
										 int n,
										 bool use_gpu);
extern void neurondb_gpu_kmeans_update(const float *data,
									   const float *centroids,
									   int *assignments,
									   float *new_centroids,
									   int n_samples,
									   int n_features,
									   int k,
									   bool use_gpu);
extern void neurondb_gpu_compute_gradient(const float *weights,
										  const float *X,
										  const float *y,
										  float *gradient,
										  int n_samples,
										  int n_features,
										  bool use_gpu);
extern void
			neurondb_gpu_softmax(const float *input, float *output, int n, bool use_gpu);

#endif							/* NEURONDB_GPU_H */
