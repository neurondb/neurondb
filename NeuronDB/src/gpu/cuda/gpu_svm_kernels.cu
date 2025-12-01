/*-------------------------------------------------------------------------
 *
 * gpu_svm_kernels.cu
 *    CUDA kernels for Support Vector Machine operations
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/cuda/gpu_svm_kernels.cu
 *
 *-------------------------------------------------------------------------
 */

#include "neurondb_cuda_runtime.h"

#ifdef NDB_GPU_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

/*
 * CUDA kernel for linear kernel computation: K(x, y) = x · y
 */
__global__ void
ndb_cuda_svm_linear_kernel_kernel(const float *x,
	const float *y,
	int feature_dim,
	float *result)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx == 0)
	{
		float dot = 0.0f;
		int i;

		for (i = 0; i < feature_dim; i++)
			dot += x[i] * y[i];

		*result = dot;
	}
}

/*
 * CUDA kernel for SVM prediction: f(x) = Σ(alpha_i * y_i * K(x_i, x)) + bias
 */
__global__ void
ndb_cuda_svm_predict_kernel(const float *input,
	const float *support_vectors,
	const float *alphas,
	const double *labels,
	float bias,
	int n_support_vectors,
	int feature_dim,
	float *prediction)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx == 0)
	{
		float result = bias;
		int i, j;

		for (i = 0; i < n_support_vectors; i++)
		{
			float kernel_val = 0.0f;
			const float *sv = support_vectors + (i * feature_dim);

			/* Linear kernel: K(x_i, x) = x_i · x */
			for (j = 0; j < feature_dim; j++)
				kernel_val += sv[j] * input[j];

			result += alphas[i] * (float)labels[i] * kernel_val;
		}

		*prediction = result;
	}
}

/*
 * CUDA kernel to compute kernel matrix row: K[i][j] = K(x_i, x_j) for all j
 */
__global__ void
ndb_cuda_svm_compute_kernel_row_kernel(const float *features,
	int n_samples,
	int feature_dim,
	int row_idx,
	float *kernel_row)
{
	int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (col_idx >= n_samples)
		return;
	
	const float *x_i = features + row_idx * feature_dim;
	const float *x_j = features + col_idx * feature_dim;
	float dot = 0.0f;
	int k;
	
	for (k = 0; k < feature_dim; k++)
		dot += x_i[k] * x_j[k];
	
	kernel_row[col_idx] = dot;
}

/*
 * CUDA kernel to compute errors: E_i = f(x_i) - y_i
 */
__global__ void
ndb_cuda_svm_compute_errors_kernel(const float *alphas,
	const double *labels,
	const float *kernel_matrix,
	float bias,
	int n_samples,
	float *errors)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i >= n_samples)
		return;
	
	float f_x = bias;
	int j;
	
	/* Compute f(x_i) = Σ(alpha_j * y_j * K(x_j, x_i)) + bias */
	for (j = 0; j < n_samples; j++)
	{
		if (alphas[j] > 1e-6f)
		{
			float k_val = kernel_matrix[j * n_samples + i];
			f_x += alphas[j] * (float)labels[j] * k_val;
		}
	}
	
	errors[i] = f_x - (float)labels[i];
}

/*
 * CUDA kernel to update errors after alpha change: E_j -= delta_alpha * y_i * K(x_i, x_j)
 */
__global__ void
ndb_cuda_svm_update_errors_kernel(const float *kernel_row,
	float delta_alpha,
	float label_i,
	int n_samples,
	float *errors)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j >= n_samples)
		return;
	
	errors[j] -= delta_alpha * label_i * kernel_row[j];
}

/*
 * Host-side launcher for compute_kernel_row kernel
 */
extern "C" int
ndb_cuda_svm_launch_compute_kernel_row(const float *features,
	int n_samples,
	int feature_dim,
	int row_idx,
	float *kernel_row)
{
	float *d_features = NULL;
	float *d_kernel_row = NULL;
	int threads_per_block = 256;
	int blocks = (n_samples + threads_per_block - 1) / threads_per_block;
	cudaError_t err;

	if (features == NULL || kernel_row == NULL)
		return -1;
	if (row_idx < 0 || row_idx >= n_samples)
		return -1;
	if (n_samples <= 0 || feature_dim <= 0)
		return -1;

	/* Allocate device memory */
	size_t features_bytes = sizeof(float) * (size_t)n_samples * (size_t)feature_dim;
	size_t row_bytes = sizeof(float) * (size_t)n_samples;

	err = cudaMalloc((void **)&d_features, features_bytes);
	if (err != cudaSuccess)
		return -1;

	err = cudaMalloc((void **)&d_kernel_row, row_bytes);
	if (err != cudaSuccess)
	{
		cudaFree(d_features);
		return -1;
	}

	/* Copy features to device */
	err = cudaMemcpy(d_features, features, features_bytes, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		cudaFree(d_features);
		cudaFree(d_kernel_row);
		return -1;
	}

	/* Launch kernel */
	ndb_cuda_svm_compute_kernel_row_kernel<<<blocks, threads_per_block>>>(
		d_features, n_samples, feature_dim, row_idx, d_kernel_row);

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		cudaFree(d_features);
		cudaFree(d_kernel_row);
		return -1;
	}

	/* Wait for kernel to complete */
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		cudaFree(d_features);
		cudaFree(d_kernel_row);
		return -1;
	}

	/* Copy result back to host */
	err = cudaMemcpy(kernel_row, d_kernel_row, row_bytes, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		cudaFree(d_features);
		cudaFree(d_kernel_row);
		return -1;
	}

	cudaFree(d_features);
	cudaFree(d_kernel_row);
	return 0;
}

/*
 * Host-side launcher for compute_errors kernel
 */
extern "C" int
ndb_cuda_svm_launch_compute_errors(const float *alphas,
	const double *labels,
	const float *kernel_matrix,
	float bias,
	int n_samples,
	float *errors)
{
	float *d_alphas = NULL;
	double *d_labels = NULL;
	float *d_kernel_matrix = NULL;
	float *d_errors = NULL;
	int threads_per_block = 256;
	int blocks = (n_samples + threads_per_block - 1) / threads_per_block;
	cudaError_t err;

	if (alphas == NULL || labels == NULL || kernel_matrix == NULL || errors == NULL)
		return -1;
	if (n_samples <= 0)
		return -1;

	/* Allocate device memory */
	size_t alphas_bytes = sizeof(float) * (size_t)n_samples;
	size_t labels_bytes = sizeof(double) * (size_t)n_samples;
	size_t kernel_matrix_bytes = sizeof(float) * (size_t)n_samples * (size_t)n_samples;
	size_t errors_bytes = sizeof(float) * (size_t)n_samples;

	err = cudaMalloc((void **)&d_alphas, alphas_bytes);
	if (err != cudaSuccess)
		return -1;

	err = cudaMalloc((void **)&d_labels, labels_bytes);
	if (err != cudaSuccess)
	{
		cudaFree(d_alphas);
		return -1;
	}

	err = cudaMalloc((void **)&d_kernel_matrix, kernel_matrix_bytes);
	if (err != cudaSuccess)
	{
		cudaFree(d_alphas);
		cudaFree(d_labels);
		return -1;
	}

	err = cudaMalloc((void **)&d_errors, errors_bytes);
	if (err != cudaSuccess)
	{
		cudaFree(d_alphas);
		cudaFree(d_labels);
		cudaFree(d_kernel_matrix);
		return -1;
	}

	/* Copy data to device */
	err = cudaMemcpy(d_alphas, alphas, alphas_bytes, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
		goto cleanup;

	err = cudaMemcpy(d_labels, labels, labels_bytes, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
		goto cleanup;

	err = cudaMemcpy(d_kernel_matrix, kernel_matrix, kernel_matrix_bytes, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
		goto cleanup;

	/* Launch kernel */
	ndb_cuda_svm_compute_errors_kernel<<<blocks, threads_per_block>>>(
		d_alphas, d_labels, d_kernel_matrix, bias,
		n_samples, d_errors);

	err = cudaGetLastError();
	if (err != cudaSuccess)
		goto cleanup;

	/* Wait for kernel to complete */
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
		goto cleanup;

	/* Copy result back to host */
	err = cudaMemcpy(errors, d_errors, errors_bytes, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
		goto cleanup;

	cudaFree(d_alphas);
	cudaFree(d_labels);
	cudaFree(d_kernel_matrix);
	cudaFree(d_errors);
	return 0;

cleanup:
	if (d_alphas)
		cudaFree(d_alphas);
	if (d_labels)
		cudaFree(d_labels);
	if (d_kernel_matrix)
		cudaFree(d_kernel_matrix);
	if (d_errors)
		cudaFree(d_errors);
	return -1;
}

/*
 * Host-side launcher for update_errors kernel
 */
extern "C" int
ndb_cuda_svm_launch_update_errors(const float *kernel_row,
	float delta_alpha,
	float label_i,
	int n_samples,
	float *errors)
{
	float *d_kernel_row = NULL;
	float *d_errors = NULL;
	int threads_per_block = 256;
	int blocks = (n_samples + threads_per_block - 1) / threads_per_block;
	cudaError_t err;

	if (kernel_row == NULL || errors == NULL)
		return -1;
	if (n_samples <= 0)
		return -1;

	/* Allocate device memory */
	size_t kernel_row_bytes = sizeof(float) * (size_t)n_samples;
	size_t errors_bytes = sizeof(float) * (size_t)n_samples;

	err = cudaMalloc((void **)&d_kernel_row, kernel_row_bytes);
	if (err != cudaSuccess)
		return -1;

	err = cudaMalloc((void **)&d_errors, errors_bytes);
	if (err != cudaSuccess)
	{
		cudaFree(d_kernel_row);
		return -1;
	}

	/* Copy data to device */
	err = cudaMemcpy(d_kernel_row, kernel_row, kernel_row_bytes, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		cudaFree(d_kernel_row);
		cudaFree(d_errors);
		return -1;
	}

	err = cudaMemcpy(d_errors, errors, errors_bytes, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		cudaFree(d_kernel_row);
		cudaFree(d_errors);
		return -1;
	}

	/* Launch kernel */
	ndb_cuda_svm_update_errors_kernel<<<blocks, threads_per_block>>>(
		d_kernel_row, delta_alpha, label_i, n_samples, d_errors);

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		cudaFree(d_kernel_row);
		cudaFree(d_errors);
		return -1;
	}

	/* Wait for kernel to complete */
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		cudaFree(d_kernel_row);
		cudaFree(d_errors);
		return -1;
	}

	/* Copy result back to host */
	err = cudaMemcpy(errors, d_errors, errors_bytes, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		cudaFree(d_kernel_row);
		cudaFree(d_errors);
		return -1;
	}

	cudaFree(d_kernel_row);
	cudaFree(d_errors);
	return 0;
}

#endif /* NDB_GPU_CUDA */
