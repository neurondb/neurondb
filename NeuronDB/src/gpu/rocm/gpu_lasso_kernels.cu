/*-------------------------------------------------------------------------
 *
 * gpu_lasso_kernels.cu
 *    CUDA kernels for Lasso Regression coordinate descent
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/cuda/gpu_lasso_kernels.cu
 *
 *-------------------------------------------------------------------------
 */

#include "neurondb_hip/hip_runtime.h"

#ifdef NDB_GPU_HIP

#include <hip/hip_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

/*
 * Helper function for atomic addition of double values
 */
static __device__ inline void
ndb_atomicAdd_double(double *addr, double val)
{
#if __CUDA_ARCH__ >= 600
	atomicAdd(addr, val);
#else
	unsigned long long int *uaddr;
	unsigned long long int old;
	unsigned long long int assumed;
	double sum;

	uaddr = (unsigned long long int *) addr;
	old = *uaddr;

	do
	{
		assumed = old;
		sum = __longlong_as_double(assumed) + val;
		old = atomicCAS(uaddr,
				assumed,
				__double_as_longlong(sum));
	} while (assumed != old);
#endif
}

/*
 * CUDA kernel to compute rho (dot product of feature column j with residuals)
 * Each thread processes multiple samples and accumulates in registers
 */
__global__ void
ndb_rocm_lasso_compute_rho_kernel(const float *features,
	const double *residuals,
	int n_samples,
	int feature_dim,
	int feature_idx,
	double *rho_out)
{
	double local_sum = 0.0;
	int stride = blockDim.x * gridDim.x;
	int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

	/* Grid-stride loop */
	for (; sample_idx < n_samples; sample_idx += stride)
	{
		if (feature_idx < feature_dim)
		{
			const float *feature_col_j = features + (sample_idx * feature_dim + feature_idx);
			local_sum += (double)(*feature_col_j) * residuals[sample_idx];
		}
	}

	/* Single atomic add per thread */
	if (local_sum != 0.0)
		ndb_atomicAdd_double(rho_out, local_sum);
}

/*
 * CUDA kernel to compute z (sum of squares of feature column j)
 * Each thread processes multiple samples and accumulates in registers
 */
__global__ void
ndb_rocm_lasso_compute_z_kernel(const float *features,
	int n_samples,
	int feature_dim,
	int feature_idx,
	double *z_out)
{
	double local_sum = 0.0;
	int stride = blockDim.x * gridDim.x;
	int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

	/* Grid-stride loop */
	for (; sample_idx < n_samples; sample_idx += stride)
	{
		if (feature_idx < feature_dim)
		{
			const float *feature_col_j = features + (sample_idx * feature_dim + feature_idx);
			double val = (double)(*feature_col_j);
			local_sum += val * val;
		}
	}

	/* Single atomic add per thread */
	if (local_sum != 0.0)
		ndb_atomicAdd_double(z_out, local_sum);
}

/*
 * CUDA kernel to update residuals after weight update
 * residuals[i] -= feature[i][j] * weight_diff
 */
__global__ void
ndb_rocm_lasso_update_residuals_kernel(const float *features,
	double *residuals,
	int n_samples,
	int feature_dim,
	int feature_idx,
	double weight_diff)
{
	int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (sample_idx < n_samples && feature_idx < feature_dim)
	{
		const float *feature_col_j = features + (sample_idx * feature_dim + feature_idx);
		residuals[sample_idx] -= (double)(*feature_col_j) * weight_diff;
	}
}

/*
 * Host wrapper to launch rho computation kernel
 */
extern "C" hipError_t
launch_lasso_compute_rho_kernel(const float *features,
	const double *residuals,
	int n_samples,
	int feature_dim,
	int feature_idx,
	double *rho_out)
{
	if (features == NULL || residuals == NULL || rho_out == NULL)
		return hipErrorInvalidValue;

	if (n_samples <= 0 || feature_dim <= 0 || feature_idx < 0 || feature_idx >= feature_dim)
		return hipErrorInvalidValue;

	/* Clear output */
	cudaMemset(rho_out, 0, sizeof(double));

	int threads_per_block = 256;
	int blocks = (n_samples + threads_per_block - 1) / threads_per_block;

	if (blocks > 65535)
		blocks = 65535;

	hipLaunchKernelGGL(ndb_rocm_lasso_compute_rho_kernel,
		dim3(blocks),
		dim3(threads_per_block),
		0,
		0,
		features,
		residuals,
		n_samples,
		feature_dim,
		feature_idx,
		rho_out);

	return hipGetLastError();
}

/*
 * Host wrapper to launch z computation kernel
 */
extern "C" hipError_t
launch_lasso_compute_z_kernel(const float *features,
	int n_samples,
	int feature_dim,
	int feature_idx,
	double *z_out)
{
	if (features == NULL || z_out == NULL)
		return hipErrorInvalidValue;

	if (n_samples <= 0 || feature_dim <= 0 || feature_idx < 0 || feature_idx >= feature_dim)
		return hipErrorInvalidValue;

	/* Clear output */
	cudaMemset(z_out, 0, sizeof(double));

	int threads_per_block = 256;
	int blocks = (n_samples + threads_per_block - 1) / threads_per_block;

	if (blocks > 65535)
		blocks = 65535;

	hipLaunchKernelGGL(ndb_rocm_lasso_compute_z_kernel,
		dim3(blocks),
		dim3(threads_per_block),
		0,
		0,
		features,
		n_samples,
		feature_dim,
		feature_idx,
		z_out);

	return hipGetLastError();
}

/*
 * Host wrapper to launch residual update kernel
 */
extern "C" hipError_t
launch_lasso_update_residuals_kernel(const float *features,
	double *residuals,
	int n_samples,
	int feature_dim,
	int feature_idx,
	double weight_diff)
{
	if (features == NULL || residuals == NULL)
		return hipErrorInvalidValue;

	if (n_samples <= 0 || feature_dim <= 0 || feature_idx < 0 || feature_idx >= feature_dim)
		return hipErrorInvalidValue;

	int threads_per_block = 256;
	int blocks = (n_samples + threads_per_block - 1) / threads_per_block;

	if (blocks > 65535)
		blocks = 65535;

	hipLaunchKernelGGL(ndb_rocm_lasso_update_residuals_kernel,
		dim3(blocks),
		dim3(threads_per_block),
		0,
		0,
		features,
		residuals,
		n_samples,
		feature_dim,
		feature_idx,
		weight_diff);

	return hipGetLastError();
}

#endif /* NDB_GPU_HIP */

