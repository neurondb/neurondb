/*
 * gpu_nb_kernels.cu
 *    CUDA kernels for Naive Bayes training and inference.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/cuda/gpu_nb_kernels.cu
 *
 *-------------------------------------------------------------------------
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#include "neurondb_cuda_nb.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*-------------------------------------------------------------------------
 * Kernel: Count samples per class
 *-------------------------------------------------------------------------
 */
__global__ static void
ndb_cuda_nb_count_classes_kernel(const double *labels,
	int n_samples,
	int n_classes,
	int *class_counts)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n_samples)
		return;

	int class_id = (int)labels[idx];
	if (class_id >= 0 && class_id < n_classes)
		atomicAdd(&class_counts[class_id], 1);
}

/*-------------------------------------------------------------------------
 * Kernel: Compute means for each class and feature
 *-------------------------------------------------------------------------
 */
__global__ static void
ndb_cuda_nb_compute_means_kernel(const float *features,
	const double *labels,
	int n_samples,
	int feature_dim,
	int n_classes,
	double *means,
	const int *class_counts /* unused - kept for API compatibility */)
{
	int class_id = blockIdx.x;
	int feature_id = threadIdx.x;

	if (class_id >= n_classes || feature_id >= feature_dim)
		return;

	double sum = 0.0;
	int count = 0;

	for (int i = 0; i < n_samples; i++)
	{
		int label = (int)labels[i];
		if (label == class_id)
		{
			sum += (double)features[i * feature_dim + feature_id];
			count++;
		}
	}

	if (count > 0)
		means[class_id * feature_dim + feature_id] = sum / count;
}

/*-------------------------------------------------------------------------
 * Kernel: Compute variances for each class and feature
 *-------------------------------------------------------------------------
 */
__global__ static void
ndb_cuda_nb_compute_variances_kernel(const float *features,
	const double *labels,
	const double *means,
	int n_samples,
	int feature_dim,
	int n_classes,
	double *variances,
	const int *class_counts /* unused - kept for API compatibility */)
{
	int class_id = blockIdx.x;
	int feature_id = threadIdx.x;

	if (class_id >= n_classes || feature_id >= feature_dim)
		return;

	double mean = means[class_id * feature_dim + feature_id];
	double sum_sq = 0.0;
	int count = 0;

	for (int i = 0; i < n_samples; i++)
	{
		int label = (int)labels[i];
		if (label == class_id)
		{
			double diff = (double)features[i * feature_dim + feature_id] - mean;
			sum_sq += diff * diff;
			count++;
		}
	}

	if (count > 0)
		variances[class_id * feature_dim + feature_id] = sum_sq / count;
}

/*-------------------------------------------------------------------------
 * Host function: Count classes
 *-------------------------------------------------------------------------
 */
extern "C" int
ndb_cuda_nb_count_classes(const double *labels,
	int n_samples,
	int n_classes,
	int *class_counts)
{
	double *d_labels = NULL;
	int *d_class_counts = NULL;
	size_t label_bytes;
	size_t count_bytes;
	cudaError_t status;
	int threads = 256;
	int blocks;

	if (labels == NULL || class_counts == NULL || n_samples <= 0 || n_classes <= 0)
		return -1;

	/* NOTE: CUDA backend is already initialized by ndb_gpu_cuda_init()
	 * No need for redundant cudaFree(0) or cudaSetDevice() calls here.
	 * The redundant initialization was causing crashes in forked processes.
	 */

	label_bytes = sizeof(double) * (size_t)n_samples;
	count_bytes = sizeof(int) * (size_t)n_classes;

	/* DEBUG: Check CUDA state before malloc */
	{
		int device;
		void *test_ptr = NULL;
		cudaError_t check_status = cudaGetDevice(&device);
		if (check_status != cudaSuccess)
		{
			fprintf(stderr, "ndb_cuda_nb_count_classes: cudaGetDevice failed: %s\n", 
				cudaGetErrorString(check_status));
			return -1;
		}
		fprintf(stderr, "ndb_cuda_nb_count_classes: Using CUDA device %d\n", device);
		
		/* Try a test allocation */
		fprintf(stderr, "ndb_cuda_nb_count_classes: Testing cudaMalloc with 1024 bytes\n");
		check_status = cudaMalloc(&test_ptr, 1024);
		fprintf(stderr, "ndb_cuda_nb_count_classes: cudaMalloc test result: %s (ptr=%p)\n",
			cudaGetErrorString(check_status), test_ptr);
		if (check_status == cudaSuccess && test_ptr != NULL)
		{
			cudaFree(test_ptr);
			fprintf(stderr, "ndb_cuda_nb_count_classes: Test allocation succeeded\n");
		}
		else
		{
			fprintf(stderr, "ndb_cuda_nb_count_classes: Test allocation FAILED\n");
			return -1;
		}
	}

	fprintf(stderr, "ndb_cuda_nb_count_classes: Allocating %zu bytes for labels\n", label_bytes);
	status = cudaMalloc((void **)&d_labels, label_bytes);
	fprintf(stderr, "ndb_cuda_nb_count_classes: cudaMalloc for labels returned: %s\n",
		cudaGetErrorString(status));
	if (status != cudaSuccess)
		return -1;

	fprintf(stderr, "ndb_cuda_nb_count_classes: Allocating %zu bytes for class_counts\n", count_bytes);
	status = cudaMalloc((void **)&d_class_counts, count_bytes);
	fprintf(stderr, "ndb_cuda_nb_count_classes: cudaMalloc for class_counts returned: %s\n",
		cudaGetErrorString(status));
	if (status != cudaSuccess)
	{
		cudaFree(d_labels);
		return -1;
	}

	fprintf(stderr, "ndb_cuda_nb_count_classes: Calling cudaMemset\n");
	status = cudaMemset(d_class_counts, 0, count_bytes);
	fprintf(stderr, "ndb_cuda_nb_count_classes: cudaMemset returned: %s\n",
		cudaGetErrorString(status));
	if (status != cudaSuccess)
	{
		cudaFree(d_class_counts);
		cudaFree(d_labels);
		return -1;
	}

	fprintf(stderr, "ndb_cuda_nb_count_classes: Calling cudaMemcpy H2D\n");
	status = cudaMemcpy(d_labels, labels, label_bytes, cudaMemcpyHostToDevice);
	fprintf(stderr, "ndb_cuda_nb_count_classes: cudaMemcpy H2D returned: %s\n",
		cudaGetErrorString(status));
	if (status != cudaSuccess)
	{
		cudaFree(d_class_counts);
		cudaFree(d_labels);
		return -1;
	}

	blocks = (n_samples + threads - 1) / threads;
	fprintf(stderr, "ndb_cuda_nb_count_classes: Launching kernel with blocks=%d, threads=%d\n", blocks, threads);
	ndb_cuda_nb_count_classes_kernel<<<blocks, threads>>>(d_labels, n_samples, n_classes, d_class_counts);
	fprintf(stderr, "ndb_cuda_nb_count_classes: Kernel launched\n");

	status = cudaGetLastError();
	fprintf(stderr, "ndb_cuda_nb_count_classes: cudaGetLastError returned: %s\n",
		cudaGetErrorString(status));
	if (status != cudaSuccess)
	{
		cudaFree(d_class_counts);
		cudaFree(d_labels);
		return -1;
	}

	fprintf(stderr, "ndb_cuda_nb_count_classes: Calling cudaDeviceSynchronize\n");
	status = cudaDeviceSynchronize();
	fprintf(stderr, "ndb_cuda_nb_count_classes: cudaDeviceSynchronize returned: %s\n",
		cudaGetErrorString(status));
	if (status != cudaSuccess)
	{
		cudaFree(d_class_counts);
		cudaFree(d_labels);
		return -1;
	}

	fprintf(stderr, "ndb_cuda_nb_count_classes: Calling cudaMemcpy D2H\n");
	status = cudaMemcpy(class_counts, d_class_counts, count_bytes, cudaMemcpyDeviceToHost);
	fprintf(stderr, "ndb_cuda_nb_count_classes: cudaMemcpy D2H returned: %s\n",
		cudaGetErrorString(status));
	if (status != cudaSuccess)
	{
		cudaFree(d_class_counts);
		cudaFree(d_labels);
		return -1;
	}

	cudaFree(d_class_counts);
	cudaFree(d_labels);
	return 0;
}

/*-------------------------------------------------------------------------
 * Host function: Compute means
 *-------------------------------------------------------------------------
 */
extern "C" int
ndb_cuda_nb_compute_means(const float *features,
	const double *labels,
	int n_samples,
	int feature_dim,
	int n_classes,
	double *means,
	const int *class_counts)
{
	float *d_features = NULL;
	double *d_labels = NULL;
	double *d_means = NULL;
	int *d_class_counts = NULL;
	size_t feature_bytes;
	size_t label_bytes;
	size_t mean_bytes;
	size_t count_bytes;
	cudaError_t status;

	if (features == NULL || labels == NULL || means == NULL || class_counts == NULL
		|| n_samples <= 0 || feature_dim <= 0 || n_classes <= 0)
		return -1;

	/* Check feature_dim does not exceed maxThreadsPerBlock (typically 1024) */
	if (feature_dim > 1024)
		return -1;

	/* Initialize CUDA runtime if needed (for consistency with count_classes) */
	status = cudaFree(0);
	if (status != cudaSuccess && status != cudaErrorInitializationError)
		return -1;

	/* Check CUDA device availability */
	int device_count = 0;
	status = cudaGetDeviceCount(&device_count);
	if (status != cudaSuccess || device_count <= 0)
		return -1;

	/* Ensure device is set */
	status = cudaSetDevice(0);
	if (status != cudaSuccess)
		return -1;

	feature_bytes = sizeof(float) * (size_t)n_samples * (size_t)feature_dim;
	label_bytes = sizeof(double) * (size_t)n_samples;
	mean_bytes = sizeof(double) * (size_t)n_classes * (size_t)feature_dim;
	count_bytes = sizeof(int) * (size_t)n_classes;

	status = cudaMalloc((void **)&d_features, feature_bytes);
	if (status != cudaSuccess)
		return -1;

	status = cudaMalloc((void **)&d_labels, label_bytes);
	if (status != cudaSuccess)
	{
		cudaFree(d_features);
		return -1;
	}

	status = cudaMalloc((void **)&d_means, mean_bytes);
	if (status != cudaSuccess)
	{
		cudaFree(d_labels);
		cudaFree(d_features);
		return -1;
	}

	status = cudaMalloc((void **)&d_class_counts, count_bytes);
	if (status != cudaSuccess)
	{
		cudaFree(d_means);
		cudaFree(d_labels);
		cudaFree(d_features);
		return -1;
	}

	status = cudaMemset(d_means, 0, mean_bytes);
	if (status != cudaSuccess)
	{
		cudaFree(d_class_counts);
		cudaFree(d_means);
		cudaFree(d_labels);
		cudaFree(d_features);
		return -1;
	}

	status = cudaMemcpy(d_features, features, feature_bytes, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		cudaFree(d_class_counts);
		cudaFree(d_means);
		cudaFree(d_labels);
		cudaFree(d_features);
		return -1;
	}

	status = cudaMemcpy(d_labels, labels, label_bytes, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		cudaFree(d_class_counts);
		cudaFree(d_means);
		cudaFree(d_labels);
		cudaFree(d_features);
		return -1;
	}

	status = cudaMemcpy(d_class_counts, class_counts, count_bytes, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		cudaFree(d_class_counts);
		cudaFree(d_means);
		cudaFree(d_labels);
		cudaFree(d_features);
		return -1;
	}

	ndb_cuda_nb_compute_means_kernel<<<n_classes, feature_dim>>>(d_features, d_labels, n_samples, feature_dim, n_classes, d_means, d_class_counts);

	status = cudaGetLastError();
	if (status != cudaSuccess)
	{
		cudaFree(d_class_counts);
		cudaFree(d_means);
		cudaFree(d_labels);
		cudaFree(d_features);
		return -1;
	}

	status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
	{
		cudaFree(d_class_counts);
		cudaFree(d_means);
		cudaFree(d_labels);
		cudaFree(d_features);
		return -1;
	}

	status = cudaMemcpy(means, d_means, mean_bytes, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
	{
		cudaFree(d_class_counts);
		cudaFree(d_means);
		cudaFree(d_labels);
		cudaFree(d_features);
		return -1;
	}

	cudaFree(d_class_counts);
	cudaFree(d_means);
	cudaFree(d_labels);
	cudaFree(d_features);
	return 0;
}

/*-------------------------------------------------------------------------
 * Host function: Compute variances
 *-------------------------------------------------------------------------
 */
extern "C" int
ndb_cuda_nb_compute_variances(const float *features,
	const double *labels,
	const double *means,
	int n_samples,
	int feature_dim,
	int n_classes,
	double *variances,
	const int *class_counts)
{
	float *d_features = NULL;
	double *d_labels = NULL;
	double *d_means = NULL;
	double *d_variances = NULL;
	int *d_class_counts = NULL;
	size_t feature_bytes;
	size_t label_bytes;
	size_t mean_bytes;
	size_t variance_bytes;
	size_t count_bytes;
	cudaError_t status;

	if (features == NULL || labels == NULL || means == NULL || variances == NULL || class_counts == NULL
		|| n_samples <= 0 || feature_dim <= 0 || n_classes <= 0)
		return -1;

	/* Check feature_dim does not exceed maxThreadsPerBlock (typically 1024) */
	if (feature_dim > 1024)
		return -1;

	/* Initialize CUDA runtime if needed (for consistency with count_classes) */
	status = cudaFree(0);
	if (status != cudaSuccess && status != cudaErrorInitializationError)
		return -1;

	/* Check CUDA device availability */
	int device_count = 0;
	status = cudaGetDeviceCount(&device_count);
	if (status != cudaSuccess || device_count <= 0)
		return -1;

	/* Ensure device is set */
	status = cudaSetDevice(0);
	if (status != cudaSuccess)
		return -1;

	feature_bytes = sizeof(float) * (size_t)n_samples * (size_t)feature_dim;
	label_bytes = sizeof(double) * (size_t)n_samples;
	mean_bytes = sizeof(double) * (size_t)n_classes * (size_t)feature_dim;
	variance_bytes = sizeof(double) * (size_t)n_classes * (size_t)feature_dim;
	count_bytes = sizeof(int) * (size_t)n_classes;

	status = cudaMalloc((void **)&d_features, feature_bytes);
	if (status != cudaSuccess)
		return -1;

	status = cudaMalloc((void **)&d_labels, label_bytes);
	if (status != cudaSuccess)
	{
		cudaFree(d_features);
		return -1;
	}

	status = cudaMalloc((void **)&d_means, mean_bytes);
	if (status != cudaSuccess)
	{
		cudaFree(d_labels);
		cudaFree(d_features);
		return -1;
	}

	status = cudaMalloc((void **)&d_variances, variance_bytes);
	if (status != cudaSuccess)
	{
		cudaFree(d_means);
		cudaFree(d_labels);
		cudaFree(d_features);
		return -1;
	}

	status = cudaMalloc((void **)&d_class_counts, count_bytes);
	if (status != cudaSuccess)
	{
		cudaFree(d_variances);
		cudaFree(d_means);
		cudaFree(d_labels);
		cudaFree(d_features);
		return -1;
	}

	status = cudaMemset(d_variances, 0, variance_bytes);
	if (status != cudaSuccess)
	{
		cudaFree(d_class_counts);
		cudaFree(d_variances);
		cudaFree(d_means);
		cudaFree(d_labels);
		cudaFree(d_features);
		return -1;
	}

	status = cudaMemcpy(d_features, features, feature_bytes, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		cudaFree(d_class_counts);
		cudaFree(d_variances);
		cudaFree(d_means);
		cudaFree(d_labels);
		cudaFree(d_features);
		return -1;
	}

	status = cudaMemcpy(d_labels, labels, label_bytes, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		cudaFree(d_class_counts);
		cudaFree(d_variances);
		cudaFree(d_means);
		cudaFree(d_labels);
		cudaFree(d_features);
		return -1;
	}

	status = cudaMemcpy(d_means, means, mean_bytes, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		cudaFree(d_class_counts);
		cudaFree(d_variances);
		cudaFree(d_means);
		cudaFree(d_labels);
		cudaFree(d_features);
		return -1;
	}

	status = cudaMemcpy(d_class_counts, class_counts, count_bytes, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		cudaFree(d_class_counts);
		cudaFree(d_variances);
		cudaFree(d_means);
		cudaFree(d_labels);
		cudaFree(d_features);
		return -1;
	}

	ndb_cuda_nb_compute_variances_kernel<<<n_classes, feature_dim>>>(d_features, d_labels, d_means, n_samples, feature_dim, n_classes, d_variances, d_class_counts);

	status = cudaGetLastError();
	if (status != cudaSuccess)
	{
		cudaFree(d_class_counts);
		cudaFree(d_variances);
		cudaFree(d_means);
		cudaFree(d_labels);
		cudaFree(d_features);
		return -1;
	}

	status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
	{
		cudaFree(d_class_counts);
		cudaFree(d_variances);
		cudaFree(d_means);
		cudaFree(d_labels);
		cudaFree(d_features);
		return -1;
	}

	status = cudaMemcpy(variances, d_variances, variance_bytes, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
	{
		cudaFree(d_class_counts);
		cudaFree(d_variances);
		cudaFree(d_means);
		cudaFree(d_labels);
		cudaFree(d_features);
		return -1;
	}

	cudaFree(d_class_counts);
	cudaFree(d_variances);
	cudaFree(d_means);
	cudaFree(d_labels);
	cudaFree(d_features);
	return 0;
}

