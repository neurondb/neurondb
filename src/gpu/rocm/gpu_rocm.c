/*-------------------------------------------------------------------------
 *
 * gpu_rocm.c
 *    Backend implementation.
 *
 * This module implements accelerated vector operations for backend support.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
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

static bool rocm_initialized = false;
static int rocm_device_id = 0;
static rocblas_handle rocblas_handle = NULL;
static hipStream_t hip_stream = NULL;
static uint64_t total_rocm_ops = 0;

bool
neurondb_gpu_rocm_init(void)
{
	hipError_t hip_err;
	rocblas_status rocblas_status;
	int device_count = 0;

	if (rocm_initialized)
		return true;

	hip_err = hipGetDeviceCount(&device_count);
	if (hip_err != hipSuccess || device_count == 0)
	{
			"neurondb: No ROCm/HIP devices found: %s",
			hipGetErrorString(hip_err));
		return false;
	}

	hip_err = hipSetDevice(rocm_device_id);
	if (hip_err != hipSuccess)
	{
			"neurondb: Failed to set HIP device %d: %s",
			rocm_device_id,
			hipGetErrorString(hip_err));
		return false;
	}

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

	rocblas_status = rocblas_create_handle(&rocblas_handle);
	if (rocblas_status != rocblas_status_success)
	{
			"neurondb: Failed to create rocBLAS handle: %d",
			rocblas_status);
		return false;
	}

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

	hipMemcpyAsync(d_a, a, size, hipMemcpyHostToDevice, hip_stream);
	hipMemcpyAsync(d_b, b, size, hipMemcpyHostToDevice, hip_stream);
	hipMemcpyAsync(d_diff, a, size, hipMemcpyHostToDevice, hip_stream);

	/* Compute difference: d_diff = d_a - d_b */
	rocblas_saxpy(rocblas_handle, dim, &alpha, d_b, 1, d_diff, 1);

	/* Compute norm: result = ||d_diff|| */
	rocblas_snrm2(rocblas_handle, dim, d_diff, 1, &result);

	hipStreamSynchronize(hip_stream);

	hipFree(d_a);
	hipFree(d_b);
	hipFree(d_diff);

	total_rocm_ops++;
	return result;
}

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

void
neurondb_gpu_rocm_batch_l2(const float *queries,
	const float *targets,
	int num_queries,
	int num_targets,
	int dim,
	float *distances)
{
	int i, j;

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
