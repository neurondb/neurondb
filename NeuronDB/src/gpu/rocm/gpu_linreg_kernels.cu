/*-------------------------------------------------------------------------
 *
 * gpu_linreg_kernels.cu
 *    ROCm kernels for Linear Regression operations
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/rocm/gpu_linreg_kernels.cu
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
 * Uses native atomicAdd on compute capability 6.0+ (Pascal and newer)
 * Falls back to CAS loop for older architectures
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
 * CUDA kernel to compute X'X and X'y for normal equations
 * Each thread processes multiple samples and accumulates in registers, then does one atomic add
 * This reduces atomic contention significantly compared to per-sample atomics
 */
__global__ void
ndb_rocm_linreg_compute_xtx_kernel(const float *features,
	const double *targets,
	int n_samples,
	int feature_dim,
	int dim_with_intercept,
	double *XtX,
	double *Xty)
{
	/* Per-thread accumulators in registers (faster than shared/global memory) */
	/* Use dynamic indexing based on dim_with_intercept */
	double local_Xty[32];       /* Max 32 (safety margin) */
	double local_XtX[32 * 32];  /* Max 32x32 (safety margin) */
	int i, j;
	
	/* Validate dimension */
	if (dim_with_intercept > 32 || dim_with_intercept <= 0)
		return;  /* Safety check: dimension too large for local arrays */
	
	/* Initialize local accumulators */
	for (i = 0; i < dim_with_intercept; i++)
	{
		local_Xty[i] = 0.0;
		for (j = 0; j < dim_with_intercept; j++)
		{
			/* Use dim_with_intercept stride to match global array */
			local_XtX[i * dim_with_intercept + j] = 0.0;
		}
	}
	
	/* Grid-stride loop: each thread processes every stride-th sample */
	/* This ensures all samples are covered exactly once, regardless of grid size */
	int stride = blockDim.x * gridDim.x;
	int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	for (; sample_idx < n_samples; sample_idx += stride)
	{
		double xi[32];  /* Max 32 features + intercept (safety margin) */
		double yi = targets[sample_idx];
		
		/* Build xi vector with intercept */
		xi[0] = 1.0;
		for (i = 1; i < dim_with_intercept; i++)
		{
			if (i - 1 < feature_dim)
				xi[i] = (double)features[sample_idx * feature_dim + (i - 1)];
			else
				xi[i] = 0.0;
		}
		
		/* Accumulate X'X and X'y in registers (no atomics here) */
		for (j = 0; j < dim_with_intercept; j++)
		{
			/* X'y accumulation */
			local_Xty[j] += xi[j] * yi;
			
			/* X'X accumulation - use dim_with_intercept stride to match global array */
			for (i = 0; i < dim_with_intercept; i++)
			{
				local_XtX[j * dim_with_intercept + i] += xi[j] * xi[i];
			}
		}
	}
	
	/* Single atomic add per thread (much less contention) */
	/* Use helper function for proper double atomic addition */
	for (j = 0; j < dim_with_intercept; j++)
	{
		ndb_atomicAdd_double(&Xty[j], local_Xty[j]);
		for (i = 0; i < dim_with_intercept; i++)
		{
			/* Use matching stride: local array uses dim_with_intercept, global uses dim_with_intercept */
			ndb_atomicAdd_double(&XtX[j * dim_with_intercept + i],
				local_XtX[j * dim_with_intercept + i]);
		}
	}
}

/*
 * CUDA kernel to build X matrix with intercept column
 * Each thread processes one row: sets intercept=1.0, copies features
 */
__global__ void
ndb_rocm_build_X_matrix_kernel(const float *features,
	float *X_with_intercept,
	int n_samples,
	int feature_dim,
	int dim_with_intercept)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < n_samples)
	{
		/* Set intercept column to 1.0 */
		X_with_intercept[idx * dim_with_intercept] = 1.0f;

		/* Copy features */
		const float *src = features + idx * feature_dim;
		float *dst = X_with_intercept + idx * dim_with_intercept + 1;
		int i;

		for (i = 0; i < feature_dim; i++)
			dst[i] = src[i];
	}
}

/*
 * CUDA kernel for prediction: y = intercept + coef1*x1 + coef2*x2 + ...
 */
__global__ void
ndb_rocm_linreg_predict_kernel(const float *input,
	const float *coefficients,
	float intercept,
	int feature_dim,
	float *prediction)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx == 0)
	{
		float result = intercept;
		int i;

		for (i = 0; i < feature_dim; i++)
			result += coefficients[i] * input[i];

		*prediction = result;
	}
}

/*
 * Host launch wrapper for build_X_matrix_kernel
 */
extern "C" hipError_t
launch_build_X_matrix_kernel(const float *features,
	float *X_with_intercept,
	int n_samples,
	int feature_dim,
	int dim_with_intercept)
{
	int build_threads = 256;
	int build_blocks = (n_samples + build_threads - 1) / build_threads;

	if (n_samples <= 0 || features == NULL || X_with_intercept == NULL)
		return hipSuccess;

	hipLaunchKernelGGL(ndb_rocm_build_X_matrix_kernel,
		dim3(build_blocks),
		dim3(build_threads),
		0,
		0,
		features,
		X_with_intercept,
		n_samples,
		feature_dim,
		dim_with_intercept);

	return hipGetLastError();
}

/*
 * Host launch wrapper for linreg_compute_xtx_kernel
 */
extern "C" hipError_t
launch_linreg_compute_xtx_kernel(const float *features,
	const double *targets,
	int n_samples,
	int feature_dim,
	int dim_with_intercept,
	double *XtX,
	double *Xty)
{
	/* Validate parameters */
	if (dim_with_intercept <= 0 || dim_with_intercept > 32)
		return hipErrorInvalidValue;
	
	if (n_samples <= 0 || features == NULL || targets == NULL || XtX == NULL || Xty == NULL)
		return hipSuccess;
	
	int threads_per_block = 256;
	int blocks = (n_samples + threads_per_block - 1) / threads_per_block;
	
	/* Cap blocks to maximum grid dimension */
	if (blocks > 65535)
		blocks = 65535;

	hipLaunchKernelGGL(ndb_rocm_linreg_compute_xtx_kernel,
		dim3(blocks),
		dim3(threads_per_block),
		0,
		0,
		features,
		targets,
		n_samples,
		feature_dim,
		dim_with_intercept,
		XtX,
		Xty);

	return hipGetLastError();
}

/*
 * CUDA kernel for batch evaluation: computes predictions and accumulates SSE, SAE, count
 * Each thread processes multiple samples, accumulates in registers, then does one atomic add per metric
 * 
 * Inputs:
 *   features: [n_samples, feature_dim] - feature matrix (row-major, float)
 *   targets: [n_samples] - true target values (double)
 *   coefficients: [feature_dim] - model coefficients (double)
 *   intercept: scalar intercept term (double)
 * 
 * Outputs (atomic accumulators):
 *   sse_out: sum of squared errors (double)
 *   sae_out: sum of absolute errors (double)
 *   count_out: number of samples processed (int)
 */
__global__ void
ndb_rocm_linreg_eval_kernel(const float *features,
	const double *targets,
	const double *coefficients,
	double intercept,
	int n_samples,
	int feature_dim,
	double *sse_out,
	double *sae_out,
	long long *count_out)
{
	/* Per-thread local accumulators in registers (fastest) */
	double local_sse = 0.0;
	double local_sae = 0.0;
	long long local_count = 0;
	
	/* Grid-stride loop: each thread processes every stride-th sample */
	/* This ensures all samples are covered exactly once, regardless of grid size */
	int stride = blockDim.x * gridDim.x;
	int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	/* Process samples assigned to this thread */
	for (; sample_idx < n_samples; sample_idx += stride)
	{
		/* Compute prediction: y_pred = intercept + sum(coefficients[i] * features[i]) */
		double y_pred = intercept;
		const float *feat_row = features + (sample_idx * feature_dim);
		
		/* Dot product: coefficients * features */
		for (int i = 0; i < feature_dim; i++)
		{
			y_pred += coefficients[i] * (double)feat_row[i];
		}
		
		/* Get true target value */
		double y_true = targets[sample_idx];
		
		/* Compute error */
		double error = y_true - y_pred;
		
		/* Accumulate metrics in registers (no atomics here) */
		local_sse += error * error;  /* Sum of squared errors */
		local_sae += fabs(error);    /* Sum of absolute errors */
		local_count += 1;            /* Sample count */
	}
	
	/* Single atomic add per thread per metric (much less contention) */
	if (local_sse != 0.0)
	{
		ndb_atomicAdd_double(sse_out, local_sse);
	}
	
	if (local_sae != 0.0)
	{
		ndb_atomicAdd_double(sae_out, local_sae);
	}
	
	if (local_count > 0)
	{
		atomicAdd((unsigned long long *)count_out, (unsigned long long)local_count);
	}
}

/*
 * Host launch wrapper for linreg_eval_kernel
 * 
 * Returns hipSuccess on success, or hipError_* on failure
 */
extern "C" hipError_t
launch_linreg_eval_kernel(const float *features,
	const double *targets,
	const double *coefficients,
	double intercept,
	int n_samples,
	int feature_dim,
	double *sse_out,
	double *sae_out,
	long long *count_out)
{
	hipError_t err;
	
	/* Parameter validation */
	if (n_samples <= 0)
	{
		/* No samples to process - initialize outputs to zero */
		if (sse_out != NULL)
		{
			err = cudaMemset(sse_out, 0, sizeof(double));
			if (err != hipSuccess)
				return err;
		}
		if (sae_out != NULL)
		{
			err = cudaMemset(sae_out, 0, sizeof(double));
			if (err != hipSuccess)
				return err;
		}
		if (count_out != NULL)
		{
			err = cudaMemset(count_out, 0, sizeof(long long));
			if (err != hipSuccess)
				return err;
		}
		return hipSuccess;
	}
	
	if (features == NULL || targets == NULL || coefficients == NULL)
		return hipErrorInvalidValue;
	
	if (sse_out == NULL || sae_out == NULL || count_out == NULL)
		return hipErrorInvalidValue;
	
	if (feature_dim <= 0 || feature_dim > 10000)
		return hipErrorInvalidValue;
	
	/* Initialize output accumulators to zero */
	err = cudaMemset(sse_out, 0, sizeof(double));
	if (err != hipSuccess)
		return err;
	
	err = cudaMemset(sae_out, 0, sizeof(double));
	if (err != hipSuccess)
		return err;
	
	err = cudaMemset(count_out, 0, sizeof(long long));
	if (err != hipSuccess)
		return err;
	
	/* Configure kernel launch parameters - use grid-stride loop */
	int threads_per_block = 256;
	int blocks = (n_samples + threads_per_block - 1) / threads_per_block;
	
	/* Cap blocks to maximum grid dimension */
	if (blocks > 65535)
		blocks = 65535;
	
	/* Launch kernel */
	hipLaunchKernelGGL(ndb_rocm_linreg_eval_kernel,
		dim3(blocks),
		dim3(threads_per_block),
		0,
		0,
		features,
		targets,
		coefficients,
		intercept,
		n_samples,
		feature_dim,
		sse_out,
		sae_out,
		count_out);
	
	/* Check for launch errors */
	err = hipGetLastError();
	if (err != hipSuccess)
		return err;
	
	/* Synchronize to ensure kernel completes */
	err = hipDeviceSynchronize();
	return err;
}

/*
 * HIP kernel to add lambda to diagonal of matrix (for Ridge regression)
 * Adds lambda to diagonal elements starting from index 1 (skipping intercept)
 */
__global__ void
ndb_rocm_ridge_add_lambda_diagonal_kernel(double *matrix,
										  double lambda,
										  int dim)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	/* Add lambda to diagonal elements (excluding first element, which is intercept) */
	if (idx > 0 && idx < dim)
	{
		matrix[idx * dim + idx] += lambda;
	}
}

/*
 * Host launch wrapper for ridge_add_lambda_diagonal_kernel
 */
extern "C" hipError_t
launch_ridge_add_lambda_diagonal_kernel(double *matrix,
										 double lambda,
										 int dim)
{
	if (dim <= 0 || matrix == NULL)
		return hipErrorInvalidValue;

	int threads_per_block = 256;
	int blocks = (dim + threads_per_block - 1) / threads_per_block;

	if (blocks > 65535)
		blocks = 65535;

	hipLaunchKernelGGL(ndb_rocm_ridge_add_lambda_diagonal_kernel,
					   dim3(blocks),
					   dim3(threads_per_block),
					   0,
					   0,
					   matrix,
					   lambda,
					   dim);

	return hipGetLastError();
}

#endif /* NDB_GPU_HIP */
