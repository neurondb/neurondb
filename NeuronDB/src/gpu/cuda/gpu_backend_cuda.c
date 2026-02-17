/*-------------------------------------------------------------------------
 *
 * gpu_backend_cuda.c
 *    Backend implementation.
 *
 * This module bridges the generic backend interface with runtime primitives
 * and launch wrappers for distances, clustering, and quantization.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/cuda/gpu_backend_cuda.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include "utils/elog.h"

#include "neurondb_gpu_backend.h"
#include "neurondb_gpu_types.h"
#include "neurondb_gpu.h"
#include "neurondb_cuda_runtime.h"
#include "neurondb_cuda_launchers.h"
#include "neurondb_cuda_rf.h"
#include "neurondb_cuda_lr.h"
#include "neurondb_cuda_linreg.h"
#include "neurondb_cuda_svm.h"
#include "neurondb_cuda_dt.h"
#include "neurondb_cuda_ridge.h"
#include "neurondb_cuda_lasso.h"
#include "neurondb_cuda_nb.h"
#include "neurondb_cuda_gmm.h"
#include "neurondb_cuda_knn.h"
#include "neurondb_cuda_hf.h"
#include "neurondb_cuda_xgboost.h"
#include "neurondb_cuda_catboost.h"
#ifdef HAVE_ONNX_RUNTIME
#include "neurondb_onnx.h"
#endif

#include <stdint.h>
#include <float.h>
#include <unistd.h>				/* for getpid() */

#ifdef NDB_GPU_CUDA

#include <cublas_v2.h>
#include <string.h>
#include <math.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

#else

void
neurondb_gpu_register_cuda_backend(void)
{
	/* No-op for CPU-only builds */
}

#endif							/* NDB_GPU_CUDA */

#ifdef NDB_GPU_CUDA

typedef struct
{
	int			device_id;
	bool		initialized;
	cublasHandle_t handle;
	pid_t		init_pid;
}			NdbcCudaContext;

static NdbcCudaContext cuda_ctx =
{
	.device_id = 0,
		.initialized = false,
		.handle = NULL,
		.init_pid = 0
};

static int	ndb_cuda_init(void);
static void ndb_cuda_shutdown(void);
static int	ndb_cuda_is_available(void);
static int	ndb_cuda_device_count(void);
static int	ndb_cuda_device_info(int device_id, NDBGpuDeviceInfo *info);
static int	ndb_cuda_set_device(int device_id);
static int	ndb_cuda_mem_alloc(void **ptr, size_t bytes);
static int	ndb_cuda_mem_free(void *ptr);
static int	ndb_cuda_memcpy_h2d(void *dst, const void *src, size_t bytes);
static int	ndb_cuda_memcpy_d2h(void *dst, const void *src, size_t bytes);
static int	ndb_cuda_stream_create(ndb_stream_t * stream);
static int	ndb_cuda_stream_destroy(ndb_stream_t stream);
static int	ndb_cuda_stream_synchronize(ndb_stream_t stream);
static int	ndb_cuda_launch_l2_distance(const float *A,
										const float *B,
										float *out,
										int n,
										int d,
										ndb_stream_t stream);
static int	ndb_cuda_launch_cosine(const float *A,
								   const float *B,
								   float *out,
								   int n,
								   int d,
								   ndb_stream_t stream);
static int	ndb_cuda_launch_kmeans_assign(const float *vectors,
										  const float *centroids,
										  int *assignments,
										  int num_vectors,
										  int dim,
										  int k,
										  ndb_stream_t stream);
static int	ndb_cuda_launch_kmeans_update(const float *vectors,
										  const int *assignments,
										  float *centroids,
										  int num_vectors,
										  int dim,
										  int k,
										  ndb_stream_t stream);
static int	ndb_cuda_launch_quant_fp16(const float *input,
									   void *output,
									   int count,
									   ndb_stream_t stream);
static int	ndb_cuda_launch_quant_int8(const float *input,
									   int8_t * output,
									   int count,
									   float scale,
									   ndb_stream_t stream);
static int	ndb_cuda_launch_quant_int4(const float *input,
									   unsigned char *output,
									   int count,
									   float scale,
									   ndb_stream_t stream);
static int	ndb_cuda_launch_quant_fp8_e4m3(const float *input,
										   unsigned char *output,
										   int count,
										   ndb_stream_t stream);
static int	ndb_cuda_launch_quant_fp8_e5m2(const float *input,
										   unsigned char *output,
										   int count,
										   ndb_stream_t stream);
static int	ndb_cuda_launch_quant_binary(const float *input,
										 uint8_t * output,
										 int count,
										 ndb_stream_t stream);
static int	ndb_cuda_launch_pq_encode(const float *vectors,
									  const float *codebooks,
									  uint8_t * codes,
									  int nvec,
									  int dim,
									  int m,
									  int ks,
									  ndb_stream_t stream);
static int	ndb_cuda_launch_pq_asymmetric_distance_batch(const float *query,
														  const uint8_t *codes,
														  const float *codebooks,
														  float *distances,
														  int nvec,
														  int dim,
														  int m,
														  int ks,
														  ndb_stream_t stream);
static int	ndb_cuda_launch_hnsw_build(const float *vectors,
									   int num_vectors,
									   int dim,
									   int m,
									   int ef_construction,
									   uint32_t **result_nodes,
									   uint32_t **result_neighbors,
									   int32_t **result_neighbor_counts,
									   int32_t **result_node_levels,
									   uint32_t *entry_point,
									   int *entry_level,
									   ndb_stream_t stream);
static int	ndb_cuda_hnsw_search(const float *query,
					 const float *nodes,
					 const uint32_t *neighbors,
					 const int32_t *neighbor_counts,
					 const int32_t *node_levels,
					 uint32_t entry_point,
					 int entry_level,
					 int dim,
					 int m,
					 int ef_search,
					 int k,
					 uint32_t *result_blocks,
					 float *result_distances,
					 ndb_stream_t stream);
static int	ndb_cuda_hnsw_search_filtered(const float *query,
										   const float *nodes,
										   const uint32_t *neighbors,
										   const int32_t *neighbor_counts,
										   const int32_t *node_levels,
										   uint32_t entry_point,
										   int entry_level,
										   int dim,
										   int m,
										   int ef_search,
										   int k,
										   const uint32_t *filter_blocks,
										   int filter_block_count,
										   uint32_t *result_blocks,
										   float *result_distances,
										   int *result_count,
										   ndb_stream_t stream);
static int	ndb_cuda_hnsw_search_batch(const float *queries,
										const float *nodes,
										const uint32_t *neighbors,
										const int32_t *neighbor_counts,
										const int32_t *node_levels,
										uint32_t entry_point,
										int entry_level,
										int num_queries,
										int dim,
										int m,
										int ef_search,
										int k,
										uint32_t *result_blocks,
										float *result_distances,
										ndb_stream_t stream);
static int	ndb_cuda_ivf_search(const float *query,
								 const float *centroids,
								 const float *vectors,
								 const int32_t *list_offsets,
								 const int32_t *list_sizes,
								 int nlists,
								 int nprobe,
								 int dim,
								 int k,
								 uint32_t *result_indices,
								 float *result_distances,
								 ndb_stream_t stream);
static int	ndb_cuda_ivf_search_batch(const float *queries,
									  const float *centroids,
									  const float *vectors,
									  const int32_t *list_offsets,
									  const int32_t *list_sizes,
									  int num_queries,
									  int nlists,
									  int nprobe,
									  int dim,
									  int k,
									  uint32_t *result_indices,
									  float *result_distances,
									  ndb_stream_t stream);

static int
ndb_cuda_init(void)
{
	int			device_count = 0;
	cudaError_t err;
	cublasStatus_t status;
	pid_t		current_pid = getpid();

	/*
	 * Fork detection: If CUDA was initialized in a different process, we're
	 * in a forked backend and must reset/reinitialize CUDA. CUDA contexts are
	 * not fork-safe and must be created per-process.
	 */
	if (cuda_ctx.initialized && cuda_ctx.init_pid != current_pid)
	{

		cudaDeviceReset();

		if (cuda_ctx.handle)
		{
			cublasDestroy(cuda_ctx.handle);
			cuda_ctx.handle = NULL;
		}

		cuda_ctx.initialized = false;
		cuda_ctx.init_pid = 0;
	}

	if (cuda_ctx.initialized)
		return 0;

	cudaGetLastError();

	err = cudaGetDeviceCount(&device_count);
	if (err != cudaSuccess || device_count <= 0)
	{
		elog(WARNING,
			 "neurondb: cudaGetDeviceCount failed: %s (devices=%d)",
			 cudaGetErrorString(err),
			 device_count);
		return -1;
	}

	err = cudaSetDevice(cuda_ctx.device_id);
	if (err != cudaSuccess)
	{
		elog(WARNING,
			 "neurondb: cudaSetDevice(%d) failed: %s",
			 cuda_ctx.device_id,
			 cudaGetErrorString(err));
		return -1;
	}

	err = cudaFree(0);
	if (err != cudaSuccess)
	{
		elog(WARNING,
			 "neurondb: cudaFree(0) warm-up failed: %s",
			 cudaGetErrorString(err));
		return -1;
	}

	status = cublasCreate(&cuda_ctx.handle);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		elog(WARNING,
			 "neurondb: cublasCreate failed with status %d",
			 status);
		return -1;
	}

	cuda_ctx.initialized = true;
	cuda_ctx.init_pid = current_pid;


	return 0;
}

static void
ndb_cuda_shutdown(void)
{
	if (!cuda_ctx.initialized)
		return;

	cublasDestroy(cuda_ctx.handle);
	cuda_ctx.handle = NULL;
	cuda_ctx.initialized = false;
	cuda_ctx.init_pid = 0;
}

static int
ndb_cuda_is_available(void)
{
	int			device_count = 0;

	return (cudaGetDeviceCount(&device_count) == cudaSuccess
			&& device_count > 0)
		? 1
		: 0;
}

static int
ndb_cuda_device_count(void)
{
	int			device_count = 0;

	if (cudaGetDeviceCount(&device_count) != cudaSuccess)
		return 0;
	return device_count;
}

static int
ndb_cuda_device_info(int device_id, NDBGpuDeviceInfo *info)
{
	struct cudaDeviceProp prop;
	size_t		free_mem = 0;
	size_t		total_mem = 0;

	if (info == NULL)
		return -1;

	if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess)
		return -1;

	if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess)
		free_mem = 0;

	memset(info, 0, sizeof(NDBGpuDeviceInfo));
	info->device_id = device_id;
	strncpy(info->name, prop.name, sizeof(info->name) - 1);
	info->name[sizeof(info->name) - 1] = '\0';
	info->total_memory_bytes = total_mem;
	info->free_memory_bytes = free_mem;
	info->compute_major = prop.major;
	info->compute_minor = prop.minor;
	info->is_available = true;

	return 0;
}

static int
ndb_cuda_set_device(int device_id)
{
	if (cudaSetDevice(device_id) != cudaSuccess)
		return -1;
	cuda_ctx.device_id = device_id;
	return 0;
}

cublasHandle_t
ndb_cuda_get_cublas_handle(void)
{
	if (!cuda_ctx.initialized)
		return NULL;
	return cuda_ctx.handle;
}

static int
ndb_cuda_mem_alloc(void **ptr, size_t bytes)
{
	if (ptr == NULL)
		return -1;
	if (cudaMalloc(ptr, bytes) != cudaSuccess)
		return -1;
	return 0;
}

static int
ndb_cuda_mem_free(void *ptr)
{
	if (ptr == NULL)
		return 0;
	return (cudaFree(ptr) == cudaSuccess) ? 0 : -1;
}

static int
ndb_cuda_memcpy_h2d(void *dst, const void *src, size_t bytes)
{
	return (cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice)
			== cudaSuccess)
		? 0
		: -1;
}

static int
ndb_cuda_memcpy_d2h(void *dst, const void *src, size_t bytes)
{
	return (cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost)
			== cudaSuccess)
		? 0
		: -1;
}

static int
ndb_cuda_stream_create(ndb_stream_t * stream)
{
	cudaStream_t native;

	if (cudaStreamCreate(&native) != cudaSuccess)
		return -1;
	if (stream)
		*stream = (ndb_stream_t) native;
	return 0;
}

static int
ndb_cuda_stream_destroy(ndb_stream_t stream)
{
	cudaStream_t native = (cudaStream_t) stream;

	if (native == NULL)
		return 0;
	return (cudaStreamDestroy(native) == cudaSuccess) ? 0 : -1;
}

static int
ndb_cuda_stream_synchronize(ndb_stream_t stream)
{
	cudaStream_t native = (cudaStream_t) stream;

	if (native == NULL)
		return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -1;
	return (cudaStreamSynchronize(native) == cudaSuccess) ? 0 : -1;
}

static int
ndb_cuda_launch_l2_distance(const float *A,
							const float *B,
							float *out,
							int n,
							int d,
							ndb_stream_t stream)
{
	cudaStream_t native = stream ? (cudaStream_t) stream : 0;
	float	   *d_A = NULL;
	float	   *d_B = NULL;
	float	   *d_diff = NULL;
	size_t		bytes;
	int			i;

	if (!cuda_ctx.initialized || A == NULL || B == NULL || out == NULL
		|| n <= 0 || d <= 0)
		return -1;

	bytes = (size_t) n * d * sizeof(float);
	if (cudaMalloc((void **) &d_A, bytes) != cudaSuccess)
		goto fail;
	if (cudaMalloc((void **) &d_B, bytes) != cudaSuccess)
		goto fail;
	if (cudaMalloc((void **) &d_diff, d * sizeof(float)) != cudaSuccess)
		goto fail;

	if (cudaMemcpyAsync(d_A, A, bytes, cudaMemcpyHostToDevice, native)
		!= cudaSuccess)
		goto fail;
	if (cudaMemcpyAsync(d_B, B, bytes, cudaMemcpyHostToDevice, native)
		!= cudaSuccess)
		goto fail;

	if (cublasSetStream(cuda_ctx.handle, native) != CUBLAS_STATUS_SUCCESS)
		goto fail;

	for (i = 0; i < n; i++)
	{
		const float *d_Ai = d_A + ((size_t) i * d);
		const float *d_Bi = d_B + ((size_t) i * d);
		float		alpha = -1.0f;

		if (cublasScopy(cuda_ctx.handle, d, d_Ai, 1, d_diff, 1)
			!= CUBLAS_STATUS_SUCCESS)
			goto fail;
		if (cublasSaxpy(cuda_ctx.handle, d, &alpha, d_Bi, 1, d_diff, 1)
			!= CUBLAS_STATUS_SUCCESS)
			goto fail;
		if (cublasSnrm2(cuda_ctx.handle, d, d_diff, 1, &out[i])
			!= CUBLAS_STATUS_SUCCESS)
			goto fail;
	}

	cublasSetStream(cuda_ctx.handle, NULL);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_diff);
	return 0;

fail:
	if (d_A)
		cudaFree(d_A);
	if (d_B)
		cudaFree(d_B);
	if (d_diff)
		cudaFree(d_diff);
	cublasSetStream(cuda_ctx.handle, NULL);
	return -1;
}

static int
ndb_cuda_launch_cosine(const float *A,
					   const float *B,
					   float *out,
					   int n,
					   int d,
					   ndb_stream_t stream)
{
	cudaStream_t native = stream ? (cudaStream_t) stream : 0;
	float	   *d_A = NULL;
	float	   *d_B = NULL;
	size_t		bytes;
	int			i;

	if (!cuda_ctx.initialized || A == NULL || B == NULL || out == NULL
		|| n <= 0 || d <= 0)
		return -1;

	bytes = (size_t) n * d * sizeof(float);
	if (cudaMalloc((void **) &d_A, bytes) != cudaSuccess)
		goto fail;
	if (cudaMalloc((void **) &d_B, bytes) != cudaSuccess)
		goto fail;

	if (cudaMemcpyAsync(d_A, A, bytes, cudaMemcpyHostToDevice, native)
		!= cudaSuccess)
		goto fail;
	if (cudaMemcpyAsync(d_B, B, bytes, cudaMemcpyHostToDevice, native)
		!= cudaSuccess)
		goto fail;

	if (cublasSetStream(cuda_ctx.handle, native) != CUBLAS_STATUS_SUCCESS)
		goto fail;

	for (i = 0; i < n; i++)
	{
		const float *d_Ai = d_A + ((size_t) i * d);
		const float *d_Bi = d_B + ((size_t) i * d);
		float		dot,
					norm_a,
					norm_b;

		if (cublasSdot(cuda_ctx.handle, d, d_Ai, 1, d_Bi, 1, &dot)
			!= CUBLAS_STATUS_SUCCESS)
			goto fail;
		if (cublasSnrm2(cuda_ctx.handle, d, d_Ai, 1, &norm_a)
			!= CUBLAS_STATUS_SUCCESS)
			goto fail;
		if (cublasSnrm2(cuda_ctx.handle, d, d_Bi, 1, &norm_b)
			!= CUBLAS_STATUS_SUCCESS)
			goto fail;

		if (norm_a <= 0.0f || norm_b <= 0.0f)
			out[i] = 1.0f;
		else
		{
			float		cosine = dot / (norm_a * norm_b);

			if (cosine < -1.0f)
				cosine = -1.0f;
			else if (cosine > 1.0f)
				cosine = 1.0f;
			out[i] = 1.0f - cosine;
		}
	}

	cublasSetStream(cuda_ctx.handle, NULL);
	cudaFree(d_A);
	cudaFree(d_B);
	return 0;

fail:
	if (d_A)
		cudaFree(d_A);
	if (d_B)
		cudaFree(d_B);
	cublasSetStream(cuda_ctx.handle, NULL);
	return -1;
}

static int
ndb_cuda_launch_quant_fp16(const float *input,
						   void *output,
						   int count,
						   ndb_stream_t stream)
{
	cudaStream_t native = stream ? (cudaStream_t) stream : 0;

	if (!cuda_ctx.initialized || input == NULL || output == NULL
		|| count <= 0)
		return -1;

	return (launch_quantize_fp32_to_fp16(input, output, count, native)
			== cudaSuccess)
		? 0
		: -1;
}

static int
ndb_cuda_launch_quant_int8(const float *input,
						   int8_t * output,
						   int count,
						   float scale,
						   ndb_stream_t stream)
{
	cudaStream_t native = stream ? (cudaStream_t) stream : 0;

	if (!cuda_ctx.initialized || input == NULL || output == NULL
		|| count <= 0)
		return -1;

	return (launch_quantize_fp32_to_int8(
										 input, (signed char *) output, count, scale, native)
			== cudaSuccess)
		? 0
		: -1;
}

static int
ndb_cuda_launch_quant_int4(const float *input,
						   unsigned char *output,
						   int count,
						   float scale,
						   ndb_stream_t stream)
{
	cudaStream_t native = stream ? (cudaStream_t) stream : 0;

	if (!cuda_ctx.initialized || input == NULL || output == NULL
		|| count <= 0)
		return -1;

	return (launch_quantize_fp32_to_int4(
										 input, output, count, scale, native) == cudaSuccess)
		? 0
		: -1;
}

static int
ndb_cuda_launch_quant_fp8_e4m3(const float *input,
							   unsigned char *output,
							   int count,
							   ndb_stream_t stream)
{
	cudaStream_t native = stream ? (cudaStream_t) stream : 0;

	if (!cuda_ctx.initialized || input == NULL || output == NULL
		|| count <= 0)
		return -1;

	return (launch_quantize_fp32_to_fp8_e4m3(
											 input, output, count, native) == cudaSuccess)
		? 0
		: -1;
}

static int
ndb_cuda_launch_quant_fp8_e5m2(const float *input,
							   unsigned char *output,
							   int count,
							   ndb_stream_t stream)
{
	cudaStream_t native = stream ? (cudaStream_t) stream : 0;

	if (!cuda_ctx.initialized || input == NULL || output == NULL
		|| count <= 0)
		return -1;

	return (launch_quantize_fp32_to_fp8_e5m2(
											 input, output, count, native) == cudaSuccess)
		? 0
		: -1;
}

static int
ndb_cuda_launch_quant_binary(const float *input,
							 uint8_t * output,
							 int count,
							 ndb_stream_t stream)
{
	cudaStream_t native = stream ? (cudaStream_t) stream : 0;

	if (!cuda_ctx.initialized || input == NULL || output == NULL
		|| count <= 0)
		return -1;

	return (launch_quantize_fp32_to_binary(
										   input, (unsigned char *) output, count, native)
			== cudaSuccess)
		? 0
		: -1;
}

static int
ndb_cuda_launch_kmeans_assign(const float *vectors,
							  const float *centroids,
							  int *assignments,
							  int num_vectors,
							  int dim,
							  int k,
							  ndb_stream_t stream)
{
	(void) stream;

	if (!cuda_ctx.initialized || vectors == NULL || centroids == NULL
		|| assignments == NULL)
		return -1;

	return (gpu_kmeans_assign(vectors,
							  centroids,
							  (int32_t *) assignments,
							  num_vectors,
							  k,
							  dim)
			== 0)
		? 0
		: -1;
}

static int
ndb_cuda_launch_kmeans_update(const float *vectors,
							  const int *assignments,
							  float *centroids,
							  int num_vectors,
							  int dim,
							  int k,
							  ndb_stream_t stream)
{
	int32_t    *assign32 = NULL;
	int32_t    *counts = NULL;
	int			i;
	int			rc;

	(void) stream;

	if (!cuda_ctx.initialized || vectors == NULL || assignments == NULL
		|| centroids == NULL)
		return -1;

	nalloc(assign32, int32_t, num_vectors);
	nalloc(counts, int32_t, k);
	MemSet(counts, 0, sizeof(int32_t) * k);

	for (i = 0; i < num_vectors; i++)
		assign32[i] = (int32_t) assignments[i];

	rc = gpu_kmeans_update(
						   vectors, assign32, centroids, counts, num_vectors, k, dim);

	pfree(assign32);
	pfree(counts);

	return rc == 0 ? 0 : -1;
}

static int
ndb_cuda_launch_pq_encode(const float *vectors,
						  const float *codebooks,
						  uint8_t * codes,
						  int nvec,
						  int dim,
						  int m,
						  int ks,
						  ndb_stream_t stream)
{
	(void) stream;

	if (!cuda_ctx.initialized || vectors == NULL || codebooks == NULL
		|| codes == NULL)
		return -1;

	return gpu_pq_encode_batch(vectors, codebooks, codes, nvec, dim, m, ks)
		== 0
		? 0
		: -1;
}

static int
ndb_cuda_launch_pq_asymmetric_distance_batch(const float *query,
											  const uint8_t *codes,
											  const float *codebooks,
											  float *distances,
											  int nvec,
											  int dim,
											  int m,
											  int ks,
											  ndb_stream_t stream)
{
	(void) stream;

	if (!cuda_ctx.initialized || query == NULL || codes == NULL
		|| codebooks == NULL || distances == NULL)
		return -1;

	return gpu_pq_asymmetric_distance_batch(query, codes, codebooks, distances,
											nvec, dim, m, ks) == 0
		? 0
		: -1;
}

/* Forward declaration for PQ kernel functions */
extern int gpu_pq_encode_batch(const float *h_vectors,
								const float *h_codebooks,
								uint8_t *h_codes,
								int nvec,
								int dim,
								int m,
								int ks);
extern int gpu_pq_asymmetric_distance_batch(const float *h_query,
											 const uint8_t *h_codes,
											 const float *h_codebooks,
											 float *h_distances,
											 int nvec,
											 int dim,
											 int m,
											 int ks);

/* Forward declaration for HNSW kernel function */
extern int gpu_hnsw_search_cuda(const float *h_query,
								 const float *h_nodes,
								 const uint32_t *h_neighbors,
								 const int32_t *h_neighbor_counts,
								 const int32_t *h_node_levels,
								 uint32_t entry_point,
								 int entry_level,
								 int dim,
								 int m,
								 int ef_search,
								 int k,
								 uint32_t *h_result_blocks,
								 float *h_result_distances);

/* Forward declaration for IVF kernel function */
extern int gpu_ivf_search_cuda(const float *h_query,
								const float *h_centroids,
								const float *h_vectors,
								const int32_t *h_list_offsets,
								const int32_t *h_list_sizes,
								int nlists,
								int nprobe,
								int dim,
								int k,
								uint32_t *h_result_indices,
								float *h_result_distances);

/* Forward declaration for batch HNSW kernel function */
extern int gpu_hnsw_search_batch_cuda(const float *h_queries,
									  const float *h_nodes,
									  const uint32_t *h_neighbors,
									  const int32_t *h_neighbor_counts,
									  const int32_t *h_node_levels,
									  uint32_t entry_point,
									  int entry_level,
									  int num_queries,
									  int dim,
									  int m,
									  int ef_search,
									  int k,
									  uint32_t *h_result_blocks,
									  float *h_result_distances);

/* Forward declaration for batch IVF kernel function */
extern int gpu_ivf_search_batch_cuda(const float *h_queries,
									 const float *h_centroids,
									 const float *h_vectors,
									 const int32_t *h_list_offsets,
									 const int32_t *h_list_sizes,
									 int num_queries,
									 int nlists,
									 int nprobe,
									 int dim,
									 int k,
									 uint32_t *h_result_indices,
									 float *h_result_distances);

static int
ndb_cuda_hnsw_search(const float *query,
					 const float *nodes,
					 const uint32_t *neighbors,
					 const int32_t *neighbor_counts,
					 const int32_t *node_levels,
					 uint32_t entry_point,
					 int entry_level,
					 int dim,
					 int m,
					 int ef_search,
					 int k,
					 uint32_t *result_blocks,
					 float *result_distances,
					 ndb_stream_t stream)
{
	/*
	 * TODO: Use stream for async execution.
	 * The stream parameter should be used to enqueue CUDA kernel launches
	 * asynchronously, allowing overlap of computation with data transfers
	 * and other operations. This requires modifying the kernel launch calls
	 * to use the stream parameter and ensuring proper synchronization when
	 * results are needed.
	 */
	(void) stream;

	if (!cuda_ctx.initialized)
		return -1;

	return gpu_hnsw_search_cuda(query,
								nodes,
								neighbors,
								neighbor_counts,
								node_levels,
								entry_point,
								entry_level,
								dim,
								m,
								ef_search,
								k,
								result_blocks,
								result_distances);
}

static int
ndb_cuda_hnsw_search_filtered(const float *query,
							   const float *nodes,
							   const uint32_t *neighbors,
							   const int32_t *neighbor_counts,
							   const int32_t *node_levels,
							   uint32_t entry_point,
							   int entry_level,
							   int dim,
							   int m,
							   int ef_search,
							   int k,
							   const uint32_t *filter_blocks,
							   int filter_block_count,
							   uint32_t *result_blocks,
							   float *result_distances,
							   int *result_count,
							   ndb_stream_t stream)
{
	/* GPU-integrated filtered HNSW search */
	/* For now, use fallback: regular search + CPU-side filtering */
	/* Full GPU implementation would require modifying the CUDA kernel to:
	 * - Accept filter_blocks array in device memory
	 * - Check filter during neighbor exploration
	 * - Early termination when enough filtered results found
	 * - Reduce data transfer by filtering on GPU
	 */
	(void)stream;

	if (!cuda_ctx.initialized)
		return -1;

	/* Use regular search and filter on CPU for now */
	/* TODO: Implement full GPU-integrated filtering kernel */
	{
		uint32_t   *candidates = NULL;
		float	   *candidate_dists = NULL;
		int			candidate_count = 0;
		int			filtered = 0;
		int			i, j;
		int			rc;

		nalloc(candidates, uint32_t, ef_search);
		nalloc(candidate_dists, float, ef_search);

		rc = gpu_hnsw_search_cuda(query,
								  nodes,
								  neighbors,
								  neighbor_counts,
								  node_levels,
								  entry_point,
								  entry_level,
								  dim,
								  m,
								  ef_search,
								  ef_search,
								  candidates,
								  candidate_dists);

		if (rc != 0)
		{
			pfree(candidates);
			pfree(candidate_dists);
			return -1;
		}

		/* Count valid candidates */
		for (i = 0; i < ef_search; i++)
		{
			if (candidates[i] == 0xFFFFFFFF)
				break;
			candidate_count++;
		}

		/* Filter using filter_blocks set */
		for (i = 0; i < candidate_count && filtered < k; i++)
		{
			bool		passes = false;

			if (filter_blocks != NULL && filter_block_count > 0)
			{
				for (j = 0; j < filter_block_count; j++)
				{
					if (filter_blocks[j] == candidates[i])
					{
						passes = true;
						break;
					}
				}
			}
			else
			{
				passes = true;
			}

			if (passes)
			{
				result_blocks[filtered] = candidates[i];
				result_distances[filtered] = candidate_dists[i];
				filtered++;
			}
		}

		/* Fill remaining with invalid */
		for (i = filtered; i < k; i++)
		{
			result_blocks[i] = 0xFFFFFFFF;
			result_distances[i] = FLT_MAX;
		}

		if (result_count != NULL)
			*result_count = filtered;

		pfree(candidates);
		pfree(candidate_dists);

		return 0;
	}
}

static int
ndb_cuda_hnsw_search_batch(const float *queries,
							const float *nodes,
							const uint32_t *neighbors,
							const int32_t *neighbor_counts,
							const int32_t *node_levels,
							uint32_t entry_point,
							int entry_level,
							int num_queries,
							int dim,
							int m,
							int ef_search,
							int k,
							uint32_t *result_blocks,
							float *result_distances,
							ndb_stream_t stream)
{
	/*
	 * TODO: Use stream for async execution.
	 * The stream parameter should be used to enqueue CUDA kernel launches
	 * asynchronously, allowing overlap of computation with data transfers
	 * and other operations. This requires modifying the kernel launch calls
	 * to use the stream parameter and ensuring proper synchronization when
	 * results are needed.
	 */
	(void) stream;

	if (!cuda_ctx.initialized)
		return -1;

	return gpu_hnsw_search_batch_cuda(queries,
									 nodes,
									 neighbors,
									 neighbor_counts,
									 node_levels,
									 entry_point,
									 entry_level,
									 num_queries,
									 dim,
									 m,
									 ef_search,
									 k,
									 result_blocks,
									 result_distances);
}

static int
ndb_cuda_ivf_search(const float *query,
					const float *centroids,
					const float *vectors,
					const int32_t *list_offsets,
					const int32_t *list_sizes,
					int nlists,
					int nprobe,
					int dim,
					int k,
					uint32_t *result_indices,
					float *result_distances,
					ndb_stream_t stream)
{
	/*
	 * TODO: Use stream for async execution.
	 * The stream parameter should be used to enqueue CUDA kernel launches
	 * asynchronously, allowing overlap of computation with data transfers
	 * and other operations. This requires modifying the kernel launch calls
	 * to use the stream parameter and ensuring proper synchronization when
	 * results are needed.
	 */
	(void) stream;

	if (!cuda_ctx.initialized)
		return -1;

	return gpu_ivf_search_cuda(query,
							   centroids,
							   vectors,
							   list_offsets,
							   list_sizes,
							   nlists,
							   nprobe,
							   dim,
							   k,
							   result_indices,
							   result_distances);
}

static int
ndb_cuda_ivf_search_batch(const float *queries,
						   const float *centroids,
						   const float *vectors,
						   const int32_t *list_offsets,
						   const int32_t *list_sizes,
						   int num_queries,
						   int nlists,
						   int nprobe,
						   int dim,
						   int k,
						   uint32_t *result_indices,
						   float *result_distances,
						   ndb_stream_t stream)
{
	/*
	 * TODO: Use stream for async execution.
	 * The stream parameter should be used to enqueue CUDA kernel launches
	 * asynchronously, allowing overlap of computation with data transfers
	 * and other operations. This requires modifying the kernel launch calls
	 * to use the stream parameter and ensuring proper synchronization when
	 * results are needed.
	 */
	(void) stream;

	if (!cuda_ctx.initialized)
		return -1;

	return gpu_ivf_search_batch_cuda(queries,
									 centroids,
									 vectors,
									 list_offsets,
									 list_sizes,
									 num_queries,
									 nlists,
									 nprobe,
									 dim,
									 k,
									 result_indices,
									 result_distances);
}

static const ndb_gpu_backend ndb_cuda_backend = {
	.name = "CUDA",
	.provider = "NVIDIA",
	.kind = NDB_GPU_BACKEND_CUDA,
	.features = NDB_GPU_FEATURE_DISTANCE | NDB_GPU_FEATURE_QUANTIZE
	| NDB_GPU_FEATURE_CLUSTERING,
	.priority = 90,

	.init = ndb_cuda_init,
	.shutdown = ndb_cuda_shutdown,
	.is_available = ndb_cuda_is_available,

	.device_count = ndb_cuda_device_count,
	.device_info = ndb_cuda_device_info,
	.set_device = ndb_cuda_set_device,

	.mem_alloc = ndb_cuda_mem_alloc,
	.mem_free = ndb_cuda_mem_free,
	.memcpy_h2d = ndb_cuda_memcpy_h2d,
	.memcpy_d2h = ndb_cuda_memcpy_d2h,

	.launch_l2_distance = ndb_cuda_launch_l2_distance,
	.launch_cosine = ndb_cuda_launch_cosine,
	.launch_kmeans_assign = ndb_cuda_launch_kmeans_assign,
	.launch_kmeans_update = ndb_cuda_launch_kmeans_update,
	.launch_quant_fp16 = ndb_cuda_launch_quant_fp16,
	.launch_quant_int8 = ndb_cuda_launch_quant_int8,
	.launch_quant_int4 = ndb_cuda_launch_quant_int4,
	.launch_quant_fp8_e4m3 = ndb_cuda_launch_quant_fp8_e4m3,
	.launch_quant_fp8_e5m2 = ndb_cuda_launch_quant_fp8_e5m2,
	.launch_quant_binary = ndb_cuda_launch_quant_binary,
	.launch_pq_encode = ndb_cuda_launch_pq_encode,
	.launch_pq_asymmetric_distance_batch = ndb_cuda_launch_pq_asymmetric_distance_batch,
	.launch_hnsw_build = ndb_cuda_launch_hnsw_build,

	.rf_train = ndb_cuda_rf_train,
	.rf_predict = ndb_cuda_rf_predict,
	.rf_pack = ndb_cuda_rf_pack_model,

	.lr_train = ndb_cuda_lr_train,
	.lr_predict = ndb_cuda_lr_predict,
	.lr_pack = ndb_cuda_lr_pack_model,

	.linreg_train = ndb_cuda_linreg_train,
	.linreg_predict = ndb_cuda_linreg_predict,
	.linreg_pack = ndb_cuda_linreg_pack_model,

	.svm_train = ndb_cuda_svm_train,
	.svm_predict = ndb_cuda_svm_predict,
	.svm_pack = ndb_cuda_svm_pack_model,

	.dt_train = ndb_cuda_dt_train,
	.dt_predict = ndb_cuda_dt_predict,
	.dt_pack = ndb_cuda_dt_pack_model,

	.ridge_train = ndb_cuda_ridge_train,
	.ridge_predict = ndb_cuda_ridge_predict,
	.ridge_pack = ndb_cuda_ridge_pack_model,

	.lasso_train = ndb_cuda_lasso_train,
	.lasso_predict = ndb_cuda_lasso_predict,
	.lasso_pack = ndb_cuda_lasso_pack_model,

	.nb_train = ndb_cuda_nb_train,
	.nb_predict = ndb_cuda_nb_predict,
	.nb_pack = ndb_cuda_nb_pack_model,

	.gmm_train = ndb_cuda_gmm_train,
	.gmm_predict = ndb_cuda_gmm_predict,
	.gmm_pack = ndb_cuda_gmm_pack_model,

	.knn_train = ndb_cuda_knn_train,
	.knn_predict = ndb_cuda_knn_predict,
	.knn_pack = ndb_cuda_knn_pack,

	.xgboost_train = ndb_cuda_xgboost_train,
	.xgboost_predict = ndb_cuda_xgboost_predict,
	.xgboost_pack = (int (*)(const struct XGBoostModel *, bytea **, Jsonb **, char **)) ndb_cuda_xgboost_pack_model,

	.catboost_train = ndb_cuda_catboost_train,
	.catboost_predict = ndb_cuda_catboost_predict,
	.catboost_pack = (int (*)(const struct CatBoostModel *, bytea **, Jsonb **, char **)) ndb_cuda_catboost_pack_model,

	.hf_embed = ndb_cuda_hf_embed,
#ifdef HAVE_ONNX_RUNTIME
	.hf_image_embed = ndb_onnx_hf_image_embed,
	.hf_multimodal_embed = ndb_onnx_hf_multimodal_embed,
#else
	.hf_image_embed = NULL,
	.hf_multimodal_embed = NULL,
#endif
	.hf_complete = ndb_cuda_hf_complete,
	.hf_rerank = ndb_cuda_hf_rerank,

	.hnsw_search = ndb_cuda_hnsw_search,
	.hnsw_search_filtered = ndb_cuda_hnsw_search_filtered,
	.hnsw_search_batch = ndb_cuda_hnsw_search_batch,
	.ivf_search = ndb_cuda_ivf_search,
	.ivf_search_batch = ndb_cuda_ivf_search_batch,

	.stream_create = ndb_cuda_stream_create,
	.stream_destroy = ndb_cuda_stream_destroy,
	.stream_synchronize = ndb_cuda_stream_synchronize,
};

void
neurondb_gpu_register_cuda_backend(void)
{
	if (ndb_gpu_register_backend(&ndb_cuda_backend) != 0)
		elog(WARNING, "neurondb: failed to register CUDA backend");
}

#endif							/* NDB_GPU_CUDA */
