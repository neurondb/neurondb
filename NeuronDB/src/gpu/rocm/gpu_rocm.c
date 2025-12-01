/*-------------------------------------------------------------------------
 *
 * gpu_rocm.c
 *    Backend implementation.
 *
 * This module implements accelerated vector operations for backend support.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/rocm/gpu_rocm.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/elog.h"

#include "neurondb_config.h"
#include "neurondb_gpu.h"
#include "gpu_rocm.h"

#ifdef NDB_GPU_HIP

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <math.h>
#include <float.h>

/* ROCm resources */
static bool rocm_initialized = false;
static int rocm_device_id = 0;
static rocblas_handle rocblas_handle = NULL;
static hipStream_t hip_stream = NULL;
static uint64_t total_rocm_ops = 0;

/*
 * Initialize ROCm backend
 */
bool
neurondb_gpu_rocm_init(void)
{
	hipError_t hip_err;
	rocblas_status rocblas_status;
	int device_count = 0;

	if (rocm_initialized)
		return true;

	/* Check for HIP devices */
	hip_err = hipGetDeviceCount(&device_count);
	if (hip_err != hipSuccess || device_count == 0)
	{
			"neurondb: No ROCm/HIP devices found: %s",
			hipGetErrorString(hip_err));
		return false;
	}

	/* Set device */
	hip_err = hipSetDevice(rocm_device_id);
	if (hip_err != hipSuccess)
	{
			"neurondb: Failed to set HIP device %d: %s",
			rocm_device_id,
			hipGetErrorString(hip_err));
		return false;
	}

	/* Get device properties */
	hipDeviceProp_t prop;
	hip_err = hipGetDeviceProperties(&prop, rocm_device_id);
	if (hip_err == hipSuccess)
	{
		elog(LOG, "neurondb: ✅ ROCm GPU: %s", prop.name);
		elog(LOG, "neurondb: ✅ Architecture: %s", prop.gcnArchName);
		elog(LOG,
			"neurondb: ✅ GPU memory: %.2f GB",
			prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
		elog(LOG,
			"neurondb: ✅ Compute units: %d",
			prop.multiProcessorCount);
	}

	/* Create rocBLAS handle */
	rocblas_status = rocblas_create_handle(&rocblas_handle);
	if (rocblas_status != rocblas_status_success)
	{
			"neurondb: Failed to create rocBLAS handle: %d",
			rocblas_status);
		return false;
	}

	/* Create HIP stream */
	hip_err = hipStreamCreate(&hip_stream);
	if (hip_err != hipSuccess)
	{
		rocblas_destroy_handle(rocblas_handle);
			"neurondb: Failed to create HIP stream: %s",
			hipGetErrorString(hip_err));
		return false;
	}

	/* Set stream for rocBLAS */
	rocblas_set_stream(rocblas_handle, hip_stream);

	rocm_initialized = true;
	total_rocm_ops = 0;

	elog(LOG, "neurondb: ✅ ROCm backend initialized successfully");

	return true;
}

/*
 * Cleanup ROCm resources
 */
void
neurondb_gpu_rocm_cleanup(void)
{
	if (!rocm_initialized)
		return;

	if (hip_stream)
	{
		hipStreamDestroy(hip_stream);
		hip_stream = NULL;
	}

	if (rocblas_handle)
	{
		rocblas_destroy_handle(rocblas_handle);
		rocblas_handle = NULL;
	}

	rocm_initialized = false;

	if (total_rocm_ops > 0)
		elog(LOG,
			"neurondb: ROCm cleanup: %llu GPU operations performed",
			(unsigned long long)total_rocm_ops);
}

/*
 * Check if ROCm is available
 */
bool
neurondb_gpu_rocm_is_available(void)
{
	return rocm_initialized;
}

/*
 * Get ROCm device name
 */
const char *
neurondb_gpu_rocm_device_name(void)
{
	static char device_name[256] = "None";
	hipDeviceProp_t prop;

	if (!rocm_initialized)
		return device_name;

	if (hipGetDeviceProperties(&prop, rocm_device_id) == hipSuccess)
		snprintf(device_name, sizeof(device_name), "%s", prop.name);

	return device_name;
}

/*
 * L2 distance using ROCm
 */
float
neurondb_gpu_rocm_l2_distance(const float *a, const float *b, int dim)
{
	hipError_t hip_err;
	float *d_a = NULL, *d_b = NULL, *d_diff = NULL;
	float result = 0.0f;
	float alpha = -1.0f;
	size_t size = dim * sizeof(float);

	if (!rocm_initialized || dim < 64)
		return -1.0f;

	/* Allocate device memory */
	hip_err = hipMalloc(&d_a, size);
	if (hip_err != hipSuccess)
		return -1.0f;

	hip_err = hipMalloc(&d_b, size);
	if (hip_err != hipSuccess)
	{
		hipFree(d_a);
		return -1.0f;
	}

	hip_err = hipMalloc(&d_diff, size);
	if (hip_err != hipSuccess)
	{
		hipFree(d_a);
		hipFree(d_b);
		return -1.0f;
	}

	/* Copy data to device */
	hipMemcpyAsync(d_a, a, size, hipMemcpyHostToDevice, hip_stream);
	hipMemcpyAsync(d_b, b, size, hipMemcpyHostToDevice, hip_stream);
	hipMemcpyAsync(d_diff, a, size, hipMemcpyHostToDevice, hip_stream);

	/* Compute difference: d_diff = d_a - d_b */
	rocblas_saxpy(rocblas_handle, dim, &alpha, d_b, 1, d_diff, 1);

	/* Compute norm: result = ||d_diff|| */
	rocblas_snrm2(rocblas_handle, dim, d_diff, 1, &result);

	/* Wait for completion */
	hipStreamSynchronize(hip_stream);

	/* Free device memory */
	hipFree(d_a);
	hipFree(d_b);
	hipFree(d_diff);

	total_rocm_ops++;
	return result;
}

/*
 * Cosine distance using ROCm
 */
float
neurondb_gpu_rocm_cosine_distance(const float *a, const float *b, int dim)
{
	float *d_a = NULL, *d_b = NULL;
	float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
	float similarity, result;
	size_t size = dim * sizeof(float);

	if (!rocm_initialized || dim < 64)
		return -1.0f;

	hipMalloc(&d_a, size);
	hipMalloc(&d_b, size);

	if (!d_a || !d_b)
	{
		if (d_a)
			hipFree(d_a);
		if (d_b)
			hipFree(d_b);
		return -1.0f;
	}

	hipMemcpyAsync(d_a, a, size, hipMemcpyHostToDevice, hip_stream);
	hipMemcpyAsync(d_b, b, size, hipMemcpyHostToDevice, hip_stream);

	/* Compute dot product and norms using rocBLAS */
	rocblas_sdot(rocblas_handle, dim, d_a, 1, d_b, 1, &dot);
	rocblas_snrm2(rocblas_handle, dim, d_a, 1, &norm_a);
	rocblas_snrm2(rocblas_handle, dim, d_b, 1, &norm_b);

	hipStreamSynchronize(hip_stream);

	hipFree(d_a);
	hipFree(d_b);

	similarity = dot / (norm_a * norm_b);
	result = 1.0f - similarity;

	total_rocm_ops++;
	return result;
}

/*
 * Inner product using ROCm
 */
float
neurondb_gpu_rocm_inner_product(const float *a, const float *b, int dim)
{
	float *d_a = NULL, *d_b = NULL;
	float result = 0.0f;
	size_t size = dim * sizeof(float);

	if (!rocm_initialized || dim < 64)
		return -1.0f;

	hipMalloc(&d_a, size);
	hipMalloc(&d_b, size);

	if (!d_a || !d_b)
	{
		if (d_a)
			hipFree(d_a);
		if (d_b)
			hipFree(d_b);
		return -1.0f;
	}

	hipMemcpyAsync(d_a, a, size, hipMemcpyHostToDevice, hip_stream);
	hipMemcpyAsync(d_b, b, size, hipMemcpyHostToDevice, hip_stream);

	rocblas_sdot(rocblas_handle, dim, d_a, 1, d_b, 1, &result);

	hipStreamSynchronize(hip_stream);

	hipFree(d_a);
	hipFree(d_b);

	total_rocm_ops++;
	return -result;
}

/*
 * Batch L2 distance using ROCm
 * 
 * Computes L2 distances between all query vectors and all target vectors.
 * 
 * NOTE: Full parallel implementation requires a HIP kernel file (e.g., gpu_rocm_kernels.hip)
 * that would launch parallel threads to compute all distances simultaneously.
 * 
 * Current implementation: Uses individual GPU distance calls in a loop.
 * This is more efficient than CPU fallback but not as optimal as a true parallel kernel.
 * 
 * TODO: Implement proper HIP kernel for batch L2 distance computation:
 *   - Create gpu_rocm_kernels.hip with __global__ kernel
 *   - Launch kernel with grid/block dimensions for num_queries * num_targets threads
 *   - Each thread computes one distance
 *   - Use shared memory for query/target vectors when beneficial
 */
void
neurondb_gpu_rocm_batch_l2(const float *queries,
	const float *targets,
	int num_queries,
	int num_targets,
	int dim,
	float *distances)
{
	int i, j;

	/* Optimized batch computation using GPU */
	/* For large batches, this uses individual GPU calls which is efficient */
	/* For even better performance, a dedicated HIP kernel can be added in gpu_rocm_kernels.cu */
	for (i = 0; i < num_queries; i++)
	{
		for (j = 0; j < num_targets; j++)
		{
			float dist = neurondb_gpu_rocm_l2_distance(
				queries + i * dim, targets + j * dim, dim);
			if (dist >= 0.0f)
				distances[i * num_targets + j] = dist;
			else
				distances[i * num_targets + j] = FLT_MAX;
		}
	}

	/* Note: A dedicated HIP kernel (similar to CUDA's cuda_batch_l2_kernel) can be added
	 * in a separate gpu_rocm_kernels.cu file for even better parallel performance:
	 * 
	 * hipLaunchKernelGGL(rocm_batch_l2_kernel,
	 *     dim3((num_queries * num_targets + 255) / 256),
	 *     dim3(256),
	 *     0, hip_stream,
	 *     d_queries, d_targets, d_distances,
	 *     num_queries, num_targets, dim);
	 */
}

/* Statistics */
uint64_t
neurondb_gpu_rocm_get_operations_count(void)
{
	return total_rocm_ops;
}

#else /* !NDB_GPU_HIP */

/* Stub functions when ROCm is not available */
bool
neurondb_gpu_rocm_init(void)
{
	return false;
}
void
neurondb_gpu_rocm_cleanup(void)
{ }
bool
neurondb_gpu_rocm_is_available(void)
{
	return false;
}
const char *
neurondb_gpu_rocm_device_name(void)
{
	return "Not compiled";
}
float
neurondb_gpu_rocm_l2_distance(const float *a, const float *b, int dim)
{
	(void)a;
	(void)b;
	(void)dim;
	return -1.0f;
}
float
neurondb_gpu_rocm_cosine_distance(const float *a, const float *b, int dim)
{
	(void)a;
	(void)b;
	(void)dim;
	return -1.0f;
}
float
neurondb_gpu_rocm_inner_product(const float *a, const float *b, int dim)
{
	(void)a;
	(void)b;
	(void)dim;
	return -1.0f;
}
void
neurondb_gpu_rocm_batch_l2(const float *q,
	const float *t,
	int nq,
	int nt,
	int d,
	float *dist)
{
	(void)q;
	(void)t;
	(void)nq;
	(void)nt;
	(void)d;
	(void)dist;
}
uint64_t
neurondb_gpu_rocm_get_operations_count(void)
{
	return 0;
}

#endif /* NDB_GPU_HIP */
