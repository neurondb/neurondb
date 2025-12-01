/*-------------------------------------------------------------------------
 *
 * gpu_backend_cuda.c
 *    Backend implementation.
 *
 * This module bridges the generic backend interface with runtime primitives
 * and launch wrappers for distances, clustering, and quantization.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
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
#ifdef HAVE_ONNX_RUNTIME
#include "neurondb_onnx.h"
#endif

#include <stdint.h>
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
static int	ndb_cuda_device_info(int device_id, NDBGpuDeviceInfo * info);
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
		elog(DEBUG1,
			 "neurondb: Detected fork (parent PID %d, current PID %d) - resetting CUDA",
			 cuda_ctx.init_pid,
			 current_pid);

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

	elog(DEBUG1,
		 "neurondb: CUDA initialized successfully in process %d (device %d)",
		 current_pid,
		 cuda_ctx.device_id);

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
ndb_cuda_device_info(int device_id, NDBGpuDeviceInfo * info)
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
	int32_t    *assign32;
	int32_t    *counts;
	int			i;
	int			rc;

	(void) stream;

	if (!cuda_ctx.initialized || vectors == NULL || assignments == NULL
		|| centroids == NULL)
		return -1;

	assign32 = (int32_t *) palloc(sizeof(int32_t) * num_vectors);
	counts = (int32_t *) palloc0(sizeof(int32_t) * k);

	for (i = 0; i < num_vectors; i++)
		assign32[i] = (int32_t) assignments[i];

	rc = gpu_kmeans_update(
						   vectors, assign32, centroids, counts, num_vectors, k, dim);

	NDB_FREE(assign32);
	NDB_FREE(counts);

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
