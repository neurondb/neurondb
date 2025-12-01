/*-------------------------------------------------------------------------
 *
 * gpu_dt_kernels.cu
 *    CUDA kernels for Decision Tree training
 *
 * Implements GPU-accelerated split finding for both classification
 * (Gini impurity) and regression (variance reduction).
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/cuda/gpu_dt_kernels.cu
 *
 *-------------------------------------------------------------------------
 */

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Kernel: Compute feature statistics (min, max, mean, variance)
 * for a single feature across all samples
 */
__global__ void
ndb_cuda_dt_feature_stats_kernel(const float *features,
	const int *indices,
	int n_samples,
	int feature_dim,
	int feature_idx,
	float *min_val,
	float *max_val,
	double *sum,
	double *sumsq)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ float s_min[256];
	__shared__ float s_max[256];
	__shared__ double s_sum[256];
	__shared__ double s_sumsq[256];
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int bdim = blockDim.x;
	int i;
	float local_min = FLT_MAX;
	float local_max = -FLT_MAX;
	double local_sum = 0.0;
	double local_sumsq = 0.0;

	/* Each thread processes multiple samples */
	for (i = idx; i < n_samples; i += gridDim.x * blockDim.x)
	{
		int sample_idx = indices[i];
		float val = features[sample_idx * feature_dim + feature_idx];

		if (isfinite(val))
		{
			if (val < local_min)
				local_min = val;
			if (val > local_max)
				local_max = val;
			local_sum += (double)val;
			local_sumsq += (double)val * (double)val;
		}
	}

	/* Store in shared memory */
	s_min[tid] = local_min;
	s_max[tid] = local_max;
	s_sum[tid] = local_sum;
	s_sumsq[tid] = local_sumsq;
	__syncthreads();

	/* Reduce within block */
	for (int stride = bdim / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			if (s_min[tid + stride] < s_min[tid])
				s_min[tid] = s_min[tid + stride];
			if (s_max[tid + stride] > s_max[tid])
				s_max[tid] = s_max[tid + stride];
			s_sum[tid] += s_sum[tid + stride];
			s_sumsq[tid] += s_sumsq[tid + stride];
		}
		__syncthreads();
	}

	/* Write block results to global memory */
	if (tid == 0)
	{
		min_val[bid] = s_min[0];
		max_val[bid] = s_max[0];
		sum[bid] = s_sum[0];
		sumsq[bid] = s_sumsq[0];
	}
}

/*
 * Kernel: Compute split statistics for classification (Gini impurity)
 * Counts classes on left and right side of threshold
 */
__global__ void
ndb_cuda_dt_split_counts_classification_kernel(const float *features,
	const int *labels,
	const int *indices,
	int n_samples,
	int feature_dim,
	int feature_idx,
	float threshold,
	int class_count,
	int *left_counts,
	int *right_counts)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n_samples)
		return;

	int sample_idx = indices[idx];
	float val = features[sample_idx * feature_dim + feature_idx];
	int label = labels[sample_idx];

	if (label < 0 || label >= class_count)
		return;

	if (isfinite(val))
	{
		if (val <= threshold)
		{
			atomicAdd(&left_counts[label], 1);
		}
		else
		{
			atomicAdd(&right_counts[label], 1);
		}
	}
}

/*
 * Kernel: Compute split statistics for regression (variance)
 * Uses shared memory reduction within blocks, then atomic operations
 */
__global__ void
ndb_cuda_dt_split_stats_regression_kernel(const float *features,
	const double *labels,
	const int *indices,
	int n_samples,
	int feature_dim,
	int feature_idx,
	float threshold,
	double *left_sum,
	double *left_sumsq,
	int *left_count,
	double *right_sum,
	double *right_sumsq,
	int *right_count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ double s_left_sum[256];
	__shared__ double s_left_sumsq[256];
	__shared__ int s_left_count[256];
	__shared__ double s_right_sum[256];
	__shared__ double s_right_sumsq[256];
	__shared__ int s_right_count[256];
	int tid = threadIdx.x;
	int bid __attribute__((unused)) = blockIdx.x;
	int bdim = blockDim.x;
	int i;
	double local_left_sum = 0.0;
	double local_left_sumsq = 0.0;
	int local_left_count = 0;
	double local_right_sum = 0.0;
	double local_right_sumsq = 0.0;
	int local_right_count = 0;

	/* Initialize shared memory */
	if (tid < 256)
	{
		s_left_sum[tid] = 0.0;
		s_left_sumsq[tid] = 0.0;
		s_left_count[tid] = 0;
		s_right_sum[tid] = 0.0;
		s_right_sumsq[tid] = 0.0;
		s_right_count[tid] = 0;
	}
	__syncthreads();

	/* Each thread processes multiple samples */
	for (i = idx; i < n_samples; i += gridDim.x * blockDim.x)
	{
		int sample_idx = indices[i];
		float val = features[sample_idx * feature_dim + feature_idx];
		double label_val = labels[sample_idx];

		if (!isfinite(val) || !isfinite(label_val))
			continue;

		if (val <= threshold)
		{
			local_left_sum += label_val;
			local_left_sumsq += label_val * label_val;
			local_left_count++;
		}
		else
		{
			local_right_sum += label_val;
			local_right_sumsq += label_val * label_val;
			local_right_count++;
		}
	}

	/* Store in shared memory */
	s_left_sum[tid] = local_left_sum;
	s_left_sumsq[tid] = local_left_sumsq;
	s_left_count[tid] = local_left_count;
	s_right_sum[tid] = local_right_sum;
	s_right_sumsq[tid] = local_right_sumsq;
	s_right_count[tid] = local_right_count;
	__syncthreads();

	/* Reduce within block */
	for (int stride = bdim / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			s_left_sum[tid] += s_left_sum[tid + stride];
			s_left_sumsq[tid] += s_left_sumsq[tid + stride];
			s_left_count[tid] += s_left_count[tid + stride];
			s_right_sum[tid] += s_right_sum[tid + stride];
			s_right_sumsq[tid] += s_right_sumsq[tid + stride];
			s_right_count[tid] += s_right_count[tid + stride];
		}
		__syncthreads();
	}

	/* Write block results using atomic operations */
	if (tid == 0)
	{
		atomicAdd((unsigned long long *)left_sum, __double_as_longlong(s_left_sum[0]));
		atomicAdd((unsigned long long *)left_sumsq, __double_as_longlong(s_left_sumsq[0]));
		atomicAdd(left_count, s_left_count[0]);
		atomicAdd((unsigned long long *)right_sum, __double_as_longlong(s_right_sum[0]));
		atomicAdd((unsigned long long *)right_sumsq, __double_as_longlong(s_right_sumsq[0]));
		atomicAdd(right_count, s_right_count[0]);
	}
}

/*
 * Host function: Launch feature statistics kernel
 */
extern "C" int
ndb_cuda_dt_launch_feature_stats(const float *features,
	const int *indices,
	int n_samples,
	int feature_dim,
	int feature_idx,
	float *min_val,
	float *max_val,
	double *sum,
	double *sumsq)
{
	float *d_features = NULL;
	int *d_indices = NULL;
	float *d_min = NULL;
	float *d_max = NULL;
	double *d_sum = NULL;
	double *d_sumsq = NULL;
	cudaError_t status;
	int threads = 256;
	int blocks = (n_samples + threads - 1) / threads;
	float final_min = FLT_MAX;
	float final_max = -FLT_MAX;
	double final_sum = 0.0;
	double final_sumsq = 0.0;
	float *h_min = NULL;
	float *h_max = NULL;
	double *h_sum = NULL;
	double *h_sumsq = NULL;

	if (blocks <= 0)
		blocks = 1;
	if (blocks > 1024)
		blocks = 1024;

	if (features == NULL || indices == NULL || n_samples <= 0
		|| feature_dim <= 0 || feature_idx < 0 || feature_idx >= feature_dim)
		return -1;

	status = cudaMalloc((void **)&d_features, sizeof(float) * n_samples * feature_dim);
	if (status != cudaSuccess)
		goto error;
	status = cudaMalloc((void **)&d_indices, sizeof(int) * n_samples);
	if (status != cudaSuccess)
		goto error;
	status = cudaMalloc((void **)&d_min, sizeof(float) * blocks);
	if (status != cudaSuccess)
		goto error;
	status = cudaMalloc((void **)&d_max, sizeof(float) * blocks);
	if (status != cudaSuccess)
		goto error;
	status = cudaMalloc((void **)&d_sum, sizeof(double) * blocks);
	if (status != cudaSuccess)
		goto error;
	status = cudaMalloc((void **)&d_sumsq, sizeof(double) * blocks);
	if (status != cudaSuccess)
		goto error;

	status = cudaMemcpy(d_features, features, sizeof(float) * n_samples * feature_dim, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		goto error;
	status = cudaMemcpy(d_indices, indices, sizeof(int) * n_samples, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		goto error;

	/* Initialize reduction arrays */
	for (int i = 0; i < blocks; i++)
	{
		float h_min[1] = {FLT_MAX};
		float h_max[1] = {-FLT_MAX};
		double h_sum[1] = {0.0};
		double h_sumsq[1] = {0.0};
		cudaMemcpy(&d_min[i], h_min, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(&d_max[i], h_max, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(&d_sum[i], h_sum, sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(&d_sumsq[i], h_sumsq, sizeof(double), cudaMemcpyHostToDevice);
	}

	ndb_cuda_dt_feature_stats_kernel<<<blocks, threads>>>(
		d_features, d_indices, n_samples, feature_dim, feature_idx,
		d_min, d_max, d_sum, d_sumsq);
	status = cudaGetLastError();
	if (status != cudaSuccess)
		goto error;
	status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
		goto error;

	/* Reduce results on host */
	final_min = FLT_MAX;
	final_max = -FLT_MAX;
	final_sum = 0.0;
	final_sumsq = 0.0;
	h_min = (float *)malloc(sizeof(float) * blocks);
	h_max = (float *)malloc(sizeof(float) * blocks);
	h_sum = (double *)malloc(sizeof(double) * blocks);
	h_sumsq = (double *)malloc(sizeof(double) * blocks);

	if (h_min == NULL || h_max == NULL || h_sum == NULL || h_sumsq == NULL)
	{
		if (h_min)
			free(h_min);
		if (h_max)
			free(h_max);
		if (h_sum)
			free(h_sum);
		if (h_sumsq)
			free(h_sumsq);
		goto error;
	}

	status = cudaMemcpy(h_min, d_min, sizeof(float) * blocks, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
	{
		free(h_min);
		free(h_max);
		free(h_sum);
		free(h_sumsq);
		goto error;
	}
	status = cudaMemcpy(h_max, d_max, sizeof(float) * blocks, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
	{
		free(h_min);
		free(h_max);
		free(h_sum);
		free(h_sumsq);
		goto error;
	}
	status = cudaMemcpy(h_sum, d_sum, sizeof(double) * blocks, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
	{
		free(h_min);
		free(h_max);
		free(h_sum);
		free(h_sumsq);
		goto error;
	}
	status = cudaMemcpy(h_sumsq, d_sumsq, sizeof(double) * blocks, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
	{
		free(h_min);
		free(h_max);
		free(h_sum);
		free(h_sumsq);
		goto error;
	}

	for (int i = 0; i < blocks; i++)
	{
		if (h_min[i] < final_min)
			final_min = h_min[i];
		if (h_max[i] > final_max)
			final_max = h_max[i];
		final_sum += h_sum[i];
		final_sumsq += h_sumsq[i];
	}

	*min_val = final_min;
	*max_val = final_max;
	*sum = final_sum;
	*sumsq = final_sumsq;

	free(h_min);
	free(h_max);
	free(h_sum);
	free(h_sumsq);

	cudaFree(d_features);
	cudaFree(d_indices);
	cudaFree(d_min);
	cudaFree(d_max);
	cudaFree(d_sum);
	cudaFree(d_sumsq);
	return 0;

error:
	if (d_features)
		cudaFree(d_features);
	if (d_indices)
		cudaFree(d_indices);
	if (d_min)
		cudaFree(d_min);
	if (d_max)
		cudaFree(d_max);
	if (d_sum)
		cudaFree(d_sum);
	if (d_sumsq)
		cudaFree(d_sumsq);
	return -1;
}

/*
 * Host function: Launch split counts kernel for classification
 */
extern "C" int
ndb_cuda_dt_launch_split_counts_classification(const float *features,
	const int *labels,
	const int *indices,
	int n_samples,
	int feature_dim,
	int feature_idx,
	float threshold,
	int class_count,
	int *left_counts,
	int *right_counts)
{
	float *d_features = NULL;
	int *d_labels = NULL;
	int *d_indices = NULL;
	int *d_left_counts = NULL;
	int *d_right_counts = NULL;
	cudaError_t status;
	int threads = 256;
	int blocks = (n_samples + threads - 1) / threads;
	if (blocks <= 0)
		blocks = 1;

	if (features == NULL || labels == NULL || indices == NULL
		|| n_samples <= 0 || feature_dim <= 0 || feature_idx < 0
		|| feature_idx >= feature_dim || class_count <= 0
		|| left_counts == NULL || right_counts == NULL)
		return -1;

	status = cudaMalloc((void **)&d_features, sizeof(float) * n_samples * feature_dim);
	if (status != cudaSuccess)
		goto error;
	status = cudaMalloc((void **)&d_labels, sizeof(int) * n_samples);
	if (status != cudaSuccess)
		goto error;
	status = cudaMalloc((void **)&d_indices, sizeof(int) * n_samples);
	if (status != cudaSuccess)
		goto error;
	status = cudaMalloc((void **)&d_left_counts, sizeof(int) * class_count);
	if (status != cudaSuccess)
		goto error;
	status = cudaMalloc((void **)&d_right_counts, sizeof(int) * class_count);
	if (status != cudaSuccess)
		goto error;

	status = cudaMemcpy(d_features, features, sizeof(float) * n_samples * feature_dim, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		goto error;
	status = cudaMemcpy(d_labels, labels, sizeof(int) * n_samples, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		goto error;
	status = cudaMemcpy(d_indices, indices, sizeof(int) * n_samples, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		goto error;
	status = cudaMemset(d_left_counts, 0, sizeof(int) * class_count);
	if (status != cudaSuccess)
		goto error;
	status = cudaMemset(d_right_counts, 0, sizeof(int) * class_count);
	if (status != cudaSuccess)
		goto error;

	ndb_cuda_dt_split_counts_classification_kernel<<<blocks, threads>>>(
		d_features, d_labels, d_indices, n_samples, feature_dim,
		feature_idx, threshold, class_count, d_left_counts, d_right_counts);
	status = cudaGetLastError();
	if (status != cudaSuccess)
		goto error;
	status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
		goto error;

	status = cudaMemcpy(left_counts, d_left_counts, sizeof(int) * class_count, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
		goto error;
	status = cudaMemcpy(right_counts, d_right_counts, sizeof(int) * class_count, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
		goto error;

	cudaFree(d_features);
	cudaFree(d_labels);
	cudaFree(d_indices);
	cudaFree(d_left_counts);
	cudaFree(d_right_counts);
	return 0;

error:
	if (d_features)
		cudaFree(d_features);
	if (d_labels)
		cudaFree(d_labels);
	if (d_indices)
		cudaFree(d_indices);
	if (d_left_counts)
		cudaFree(d_left_counts);
	if (d_right_counts)
		cudaFree(d_right_counts);
	return -1;
}

/*
 * Host function: Launch split statistics kernel for regression
 */
extern "C" int
ndb_cuda_dt_launch_split_stats_regression(const float *features,
	const double *labels,
	const int *indices,
	int n_samples,
	int feature_dim,
	int feature_idx,
	float threshold,
	double *left_sum,
	double *left_sumsq,
	int *left_count,
	double *right_sum,
	double *right_sumsq,
	int *right_count)
{
	float *d_features = NULL;
	double *d_labels = NULL;
	int *d_indices = NULL;
	double *d_left_sum = NULL;
	double *d_left_sumsq = NULL;
	int *d_left_count = NULL;
	double *d_right_sum = NULL;
	double *d_right_sumsq = NULL;
	int *d_right_count = NULL;
	cudaError_t status;
	int threads = 256;
	int blocks = (n_samples + threads - 1) / threads;
	double zero_d = 0.0;
	int zero_i = 0;

	if (blocks <= 0)
		blocks = 1;

	if (features == NULL || labels == NULL || indices == NULL
		|| n_samples <= 0 || feature_dim <= 0 || feature_idx < 0
		|| feature_idx >= feature_dim
		|| left_sum == NULL || left_sumsq == NULL || left_count == NULL
		|| right_sum == NULL || right_sumsq == NULL || right_count == NULL)
		return -1;

	status = cudaMalloc((void **)&d_features, sizeof(float) * n_samples * feature_dim);
	if (status != cudaSuccess)
		goto error;
	status = cudaMalloc((void **)&d_labels, sizeof(double) * n_samples);
	if (status != cudaSuccess)
		goto error;
	status = cudaMalloc((void **)&d_indices, sizeof(int) * n_samples);
	if (status != cudaSuccess)
		goto error;
	status = cudaMalloc((void **)&d_left_sum, sizeof(double));
	if (status != cudaSuccess)
		goto error;
	status = cudaMalloc((void **)&d_left_sumsq, sizeof(double));
	if (status != cudaSuccess)
		goto error;
	status = cudaMalloc((void **)&d_left_count, sizeof(int));
	if (status != cudaSuccess)
		goto error;
	status = cudaMalloc((void **)&d_right_sum, sizeof(double));
	if (status != cudaSuccess)
		goto error;
	status = cudaMalloc((void **)&d_right_sumsq, sizeof(double));
	if (status != cudaSuccess)
		goto error;
	status = cudaMalloc((void **)&d_right_count, sizeof(int));
	if (status != cudaSuccess)
		goto error;

	status = cudaMemcpy(d_features, features, sizeof(float) * n_samples * feature_dim, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		goto error;
	status = cudaMemcpy(d_labels, labels, sizeof(double) * n_samples, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		goto error;
	status = cudaMemcpy(d_indices, indices, sizeof(int) * n_samples, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		goto error;

	/* Initialize reduction variables */
	cudaMemcpy(d_left_sum, &zero_d, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_left_sumsq, &zero_d, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_left_count, &zero_i, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_right_sum, &zero_d, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_right_sumsq, &zero_d, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_right_count, &zero_i, sizeof(int), cudaMemcpyHostToDevice);

	ndb_cuda_dt_split_stats_regression_kernel<<<blocks, threads>>>(
		d_features, d_labels, d_indices, n_samples, feature_dim,
		feature_idx, threshold, d_left_sum, d_left_sumsq, d_left_count,
		d_right_sum, d_right_sumsq, d_right_count);
	status = cudaGetLastError();
	if (status != cudaSuccess)
		goto error;
	status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
		goto error;

	status = cudaMemcpy(left_sum, d_left_sum, sizeof(double), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
		goto error;
	status = cudaMemcpy(left_sumsq, d_left_sumsq, sizeof(double), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
		goto error;
	status = cudaMemcpy(left_count, d_left_count, sizeof(int), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
		goto error;
	status = cudaMemcpy(right_sum, d_right_sum, sizeof(double), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
		goto error;
	status = cudaMemcpy(right_sumsq, d_right_sumsq, sizeof(double), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
		goto error;
	status = cudaMemcpy(right_count, d_right_count, sizeof(int), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
		goto error;

	cudaFree(d_features);
	cudaFree(d_labels);
	cudaFree(d_indices);
	cudaFree(d_left_sum);
	cudaFree(d_left_sumsq);
	cudaFree(d_left_count);
	cudaFree(d_right_sum);
	cudaFree(d_right_sumsq);
	cudaFree(d_right_count);
	return 0;

error:
	if (d_features)
		cudaFree(d_features);
	if (d_labels)
		cudaFree(d_labels);
	if (d_indices)
		cudaFree(d_indices);
	if (d_left_sum)
		cudaFree(d_left_sum);
	if (d_left_sumsq)
		cudaFree(d_left_sumsq);
	if (d_left_count)
		cudaFree(d_left_count);
	if (d_right_sum)
		cudaFree(d_right_sum);
	if (d_right_sumsq)
		cudaFree(d_right_sumsq);
	if (d_right_count)
		cudaFree(d_right_count);
	return -1;
}

#ifdef __cplusplus
}
#endif

