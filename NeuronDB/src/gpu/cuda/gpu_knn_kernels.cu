/*
 * gpu_knn_kernels.cu
 *    CUDA kernels for K-Nearest Neighbors prediction.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/cuda/gpu_knn_kernels.cu
 *
 *-------------------------------------------------------------------------
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#include "neurondb_cuda_knn.h"

/*-------------------------------------------------------------------------
 * Kernel: Compute distances from query to all training samples
 *-------------------------------------------------------------------------
 */
__global__ static void
ndb_cuda_knn_distance_kernel(const float *query,
	const float *training_features,
	int n_samples,
	int feature_dim,
	float *distances)
{
	int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (sample_idx >= n_samples)
		return;

	const float *sample = training_features + sample_idx * feature_dim;
	double sum = 0.0;

	for (int d = 0; d < feature_dim; d++)
	{
		double diff = (double)query[d] - (double)sample[d];
		sum += diff * diff;
	}

	distances[sample_idx] = (float)sqrt(sum);
}

/*-------------------------------------------------------------------------
 * Kernel: Find top-k indices using partial sort
 * Supports any k value by using global memory
 *-------------------------------------------------------------------------
 */
__global__ static void
ndb_cuda_knn_top_k_kernel(const float *distances,
	const double *labels,
	int n_samples,
	int k,
	int task_type,
	int *top_k_indices,
	double *prediction_out)
{
	/* This kernel runs on a single thread to find top-k */
	if (threadIdx.x != 0 || blockIdx.x != 0)
		return;

	int i, j;
	float max_dist_in_topk;
	int max_idx_in_topk;

	/* Initialize top-k with first k samples */
	for (i = 0; i < k && i < n_samples; i++)
	{
		top_k_indices[i] = i;
	}

	/* Sort first k elements by distance (simple bubble sort) */
	for (i = 0; i < k - 1 && i < n_samples - 1; i++)
	{
		for (j = i + 1; j < k && j < n_samples; j++)
		{
			if (distances[top_k_indices[i]] > distances[top_k_indices[j]])
			{
				int temp = top_k_indices[i];
				top_k_indices[i] = top_k_indices[j];
				top_k_indices[j] = temp;
			}
		}
	}

	/* Find maximum distance in current top-k */
	max_dist_in_topk = distances[top_k_indices[k - 1]];
	max_idx_in_topk = k - 1;

	/* Process remaining samples */
	for (i = k; i < n_samples; i++)
	{
		if (distances[i] < max_dist_in_topk)
		{
			/* Replace the maximum element in top-k */
			top_k_indices[max_idx_in_topk] = i;

			/* Re-sort to find new maximum */
			max_dist_in_topk = distances[top_k_indices[0]];
			max_idx_in_topk = 0;
			for (j = 1; j < k; j++)
			{
				if (distances[top_k_indices[j]] > max_dist_in_topk)
				{
					max_dist_in_topk = distances[top_k_indices[j]];
					max_idx_in_topk = j;
				}
			}
		}
	}

	/* Final sort of top-k indices by distance */
	for (i = 0; i < k - 1; i++)
	{
		for (j = i + 1; j < k; j++)
		{
			if (distances[top_k_indices[i]] > distances[top_k_indices[j]])
			{
				int temp = top_k_indices[i];
				top_k_indices[i] = top_k_indices[j];
				top_k_indices[j] = temp;
			}
		}
	}

	/* Compute prediction */
	if (task_type == 0)
	{
		/* Classification: majority vote */
		int class_votes[2] = {0, 0};
		for (i = 0; i < k; i++)
		{
			int label = (int)labels[top_k_indices[i]];
			if (label >= 0 && label < 2)
				class_votes[label]++;
		}
		/* Match CPU semantics: class 1 only if strictly greater, tie goes to class 0 */
		*prediction_out = (class_votes[1] > class_votes[0]) ? 1.0 : 0.0;
	}
	else
	{
		/* Regression: average */
		double sum = 0.0;
		for (i = 0; i < k; i++)
			sum += labels[top_k_indices[i]];
		*prediction_out = sum / k;
	}
}

/*-------------------------------------------------------------------------
 * Host function: Compute distances
 *-------------------------------------------------------------------------
 */
extern "C" int
ndb_cuda_knn_compute_distances(const float *query,
	const float *training_features,
	int n_samples,
	int feature_dim,
	float *distances)
{
	float *d_query = NULL;
	float *d_training = NULL;
	float *d_distances = NULL;
	size_t query_bytes;
	size_t training_bytes;
	size_t distance_bytes;
	cudaError_t status;
	int threads = 256;
	int blocks;

	if (query == NULL || training_features == NULL || distances == NULL
		|| n_samples <= 0 || feature_dim <= 0)
		return -1;

	query_bytes = sizeof(float) * (size_t)feature_dim;
	training_bytes = sizeof(float) * (size_t)n_samples * (size_t)feature_dim;
	distance_bytes = sizeof(float) * (size_t)n_samples;

	cudaGetLastError();

	status = cudaMalloc((void **)&d_query, query_bytes);
	if (status != cudaSuccess)
		return -1;

	status = cudaMalloc((void **)&d_training, training_bytes);
	if (status != cudaSuccess)
	{
		cudaFree(d_query);
		return -1;
	}

	status = cudaMalloc((void **)&d_distances, distance_bytes);
	if (status != cudaSuccess)
	{
		cudaFree(d_training);
		cudaFree(d_query);
		return -1;
	}

	status = cudaMemcpy(d_query, query, query_bytes, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		cudaFree(d_distances);
		cudaFree(d_training);
		cudaFree(d_query);
		return -1;
	}

	status = cudaMemcpy(d_training, training_features, training_bytes, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		cudaFree(d_distances);
		cudaFree(d_training);
		cudaFree(d_query);
		return -1;
	}

	blocks = (n_samples + threads - 1) / threads;
	ndb_cuda_knn_distance_kernel<<<blocks, threads>>>(d_query, d_training, n_samples, feature_dim, d_distances);

	status = cudaGetLastError();
	if (status != cudaSuccess)
	{
		cudaFree(d_distances);
		cudaFree(d_training);
		cudaFree(d_query);
		return -1;
	}

	status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
	{
		cudaFree(d_distances);
		cudaFree(d_training);
		cudaFree(d_query);
		return -1;
	}

	status = cudaMemcpy(distances, d_distances, distance_bytes, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
	{
		cudaFree(d_distances);
		cudaFree(d_training);
		cudaFree(d_query);
		return -1;
	}

	cudaFree(d_distances);
	cudaFree(d_training);
	cudaFree(d_query);
	return 0;
}

/*-------------------------------------------------------------------------
 * Host function: Find top-k and compute prediction
 *-------------------------------------------------------------------------
 */
extern "C" int
ndb_cuda_knn_find_top_k(const float *distances,
	const double *labels,
	int n_samples,
	int k,
	int task_type,
	double *prediction_out)
{
	float *d_distances = NULL;
	double *d_labels = NULL;
	int *d_top_k_indices = NULL;
	double *d_prediction = NULL;
	size_t distance_bytes;
	size_t label_bytes;
	size_t indices_bytes;
	size_t pred_bytes;
	cudaError_t status;

	if (distances == NULL || labels == NULL || prediction_out == NULL
		|| n_samples <= 0 || k <= 0 || k > n_samples)
		return -1;

	/* Validate task_type: 0 = classification, 1 = regression */
	if (task_type != 0 && task_type != 1)
		return -1;

	/* Sanity check: cap k to prevent excessive memory allocation */
	if (k > 1000000)
		return -1;

	distance_bytes = sizeof(float) * (size_t)n_samples;
	label_bytes = sizeof(double) * (size_t)n_samples;
	indices_bytes = sizeof(int) * (size_t)k;
	pred_bytes = sizeof(double);

	cudaGetLastError();

	status = cudaMalloc((void **)&d_distances, distance_bytes);
	if (status != cudaSuccess)
		return -1;

	status = cudaMalloc((void **)&d_labels, label_bytes);
	if (status != cudaSuccess)
	{
		cudaFree(d_distances);
		return -1;
	}

	status = cudaMalloc((void **)&d_top_k_indices, indices_bytes);
	if (status != cudaSuccess)
	{
		cudaFree(d_labels);
		cudaFree(d_distances);
		return -1;
	}

	status = cudaMalloc((void **)&d_prediction, pred_bytes);
	if (status != cudaSuccess)
	{
		cudaFree(d_top_k_indices);
		cudaFree(d_labels);
		cudaFree(d_distances);
		return -1;
	}

	status = cudaMemcpy(d_distances, distances, distance_bytes, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		cudaFree(d_prediction);
		cudaFree(d_top_k_indices);
		cudaFree(d_labels);
		cudaFree(d_distances);
		return -1;
	}

	status = cudaMemcpy(d_labels, labels, label_bytes, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		cudaFree(d_prediction);
		cudaFree(d_top_k_indices);
		cudaFree(d_labels);
		cudaFree(d_distances);
		return -1;
	}

	ndb_cuda_knn_top_k_kernel<<<1, 1>>>(d_distances, d_labels, n_samples, k, task_type, d_top_k_indices, d_prediction);

	status = cudaGetLastError();
	if (status != cudaSuccess)
	{
		cudaFree(d_prediction);
		cudaFree(d_top_k_indices);
		cudaFree(d_labels);
		cudaFree(d_distances);
		return -1;
	}

	status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
	{
		cudaFree(d_prediction);
		cudaFree(d_top_k_indices);
		cudaFree(d_labels);
		cudaFree(d_distances);
		return -1;
	}

	status = cudaMemcpy(prediction_out, d_prediction, pred_bytes, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
	{
		cudaFree(d_prediction);
		cudaFree(d_top_k_indices);
		cudaFree(d_labels);
		cudaFree(d_distances);
		return -1;
	}

	cudaFree(d_prediction);
	cudaFree(d_top_k_indices);
	cudaFree(d_labels);
	cudaFree(d_distances);
	return 0;
}

