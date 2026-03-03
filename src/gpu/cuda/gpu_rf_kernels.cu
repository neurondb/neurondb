/*
 * gpu_rf_kernels.cu
 *	  CUDA kernels for Random Forest inference.
 *
 * Note: This code uses atomicAdd on double precision values in the feature
 * statistics kernel, which requires CUDA compute capability sm_60+ (Pascal
 * and later). On older GPUs, this may result in compilation errors or
 * runtime failures.
 */

#include <cuda_runtime.h>
#include <math.h>

#include "neurondb_cuda_rf.h"

__global__ static void
ndb_cuda_rf_predict_kernel(const NdbCudaRfNode *nodes,
	const NdbCudaRfTreeHeader *trees,
	int tree_count,
	const float *input,
	int feature_dim,
	int class_count,
	int *votes)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= tree_count)
		return;

	const NdbCudaRfTreeHeader *tree = &trees[tid];

	/* Guard against zero-node trees */
	if (tree->node_count <= 0)
		return;

	const NdbCudaRfNode *tree_nodes = nodes + tree->nodes_start;
	int idx = (tree->root_index >= 0) ? tree->root_index : 0;
	int steps = 0;
	int max_steps = tree->node_count + 1;
	int cls = 0;

	while (steps < max_steps && idx >= 0 && idx < tree->node_count)
	{
		const NdbCudaRfNode *node = &tree_nodes[idx];

		if (node->feature_idx < 0)
		{
			cls = (int)rintf(node->value);
			break;
		}
		if (node->feature_idx >= feature_dim)
		{
			cls = (int)rintf(node->value);
			break;
		}

		float val = input[node->feature_idx];

		if (val <= node->threshold)
			idx = node->left_child;
		else
			idx = node->right_child;

		steps++;
	}

	if (steps >= max_steps || idx < 0 || idx >= tree->node_count)
		cls = (int)rintf(tree_nodes[0].value);

	if (cls >= 0 && cls < class_count)
		atomicAdd(&votes[cls], 1);
}

__global__ static void
ndb_cuda_rf_hist_kernel(const int *labels,
	int n_samples,
	int class_count,
	int *counts)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < n_samples)
	{
		int cls = labels[idx];

		if (cls >= 0 && cls < class_count)
			atomicAdd(&counts[cls], 1);
	}
}

extern "C" int
ndb_cuda_rf_histogram(const int *labels,
	int n_samples,
	int class_count,
	int *counts_out)
{
	int *d_labels = NULL;
	int *d_counts = NULL;
	size_t label_bytes;
	size_t count_bytes;
	cudaError_t status;
	int threads = 256;
	int blocks;

	if (labels == NULL || counts_out == NULL || n_samples <= 0
		|| class_count <= 0)
		return -1;

	label_bytes = sizeof(int) * n_samples;
	count_bytes = sizeof(int) * class_count;

	status = cudaMalloc((void **)&d_labels, label_bytes);
	if (status != cudaSuccess)
		goto error;
	status = cudaMalloc((void **)&d_counts, count_bytes);
	if (status != cudaSuccess)
		goto error;

	status = cudaMemcpy(
		d_labels, labels, label_bytes, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		goto error;
	status = cudaMemset(d_counts, 0, count_bytes);
	if (status != cudaSuccess)
		goto error;

	blocks = (n_samples + threads - 1) / threads;
	if (blocks <= 0)
		blocks = 1;

	ndb_cuda_rf_hist_kernel<<<blocks, threads>>>(
		d_labels, n_samples, class_count, d_counts);
	status = cudaGetLastError();
	if (status != cudaSuccess)
		goto error;
	status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
		goto error;

	status = cudaMemcpy(
		counts_out, d_counts, count_bytes, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
		goto error;

	cudaFree(d_labels);
	cudaFree(d_counts);
	return 0;

error:
	if (d_labels)
		cudaFree(d_labels);
	if (d_counts)
		cudaFree(d_counts);
	return -1;
}

__global__ static void
ndb_cuda_rf_feature_stats_kernel(const float *features,
	int n_samples,
	int feature_dim,
	int feature_idx,
	double *sum_out,
	double *sumsq_out)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < n_samples)
	{
		float val = features[idx * feature_dim + feature_idx];

		atomicAdd(sum_out, (double)val);
		atomicAdd(sumsq_out, (double)val * (double)val);
	}
}

__global__ static void
ndb_cuda_rf_split_kernel(const float *features,
	const int *labels,
	int n_samples,
	int feature_dim,
	int feature_idx,
	float threshold,
	int class_count,
	int *left_counts,
	int *right_counts)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < n_samples)
	{
		int cls = labels[idx];
		float val;

		if (cls < 0 || cls >= class_count)
			return;

		val = features[idx * feature_dim + feature_idx];
		if (val <= threshold)
			atomicAdd(&left_counts[cls], 1);
		else
			atomicAdd(&right_counts[cls], 1);
	}
}

extern "C" int
ndb_cuda_rf_launch_feature_stats(const float *features,
	int n_samples,
	int feature_dim,
	int feature_idx,
	double *sum_dev,
	double *sumsq_dev)
{
	int threads = 256;
	int blocks;
	cudaError_t status;

	if (features == NULL || sum_dev == NULL || sumsq_dev == NULL
		|| n_samples <= 0 || feature_dim <= 0 || feature_idx < 0
		|| feature_idx >= feature_dim)
		return -1;

	status = cudaMemset(sum_dev, 0, sizeof(double));
	if (status != cudaSuccess)
		return -1;
	status = cudaMemset(sumsq_dev, 0, sizeof(double));
	if (status != cudaSuccess)
		return -1;

	threads = 256;
	blocks = (n_samples + threads - 1) / threads;
	if (blocks <= 0)
		blocks = 1;

	ndb_cuda_rf_feature_stats_kernel<<<blocks, threads>>>(features,
		n_samples,
		feature_dim,
		feature_idx,
		sum_dev,
		sumsq_dev);
	status = cudaGetLastError();
	if (status != cudaSuccess)
		return -1;
	status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
		return -1;

	return 0;
}

extern "C" int
ndb_cuda_rf_launch_split_counts(const float *features,
	const int *labels,
	int n_samples,
	int feature_dim,
	int feature_idx,
	float threshold,
	int class_count,
	int *left_counts_dev,
	int *right_counts_dev)
{
	int threads = 256;
	int blocks;
	cudaError_t status;

	if (features == NULL || labels == NULL || left_counts_dev == NULL
		|| right_counts_dev == NULL || n_samples <= 0
		|| feature_dim <= 0 || class_count <= 0 || feature_idx < 0
		|| feature_idx >= feature_dim)
		return -1;

	status = cudaMemset(left_counts_dev, 0, sizeof(int) * class_count);
	if (status != cudaSuccess)
		return -1;
	status = cudaMemset(right_counts_dev, 0, sizeof(int) * class_count);
	if (status != cudaSuccess)
		return -1;

	blocks = (n_samples + threads - 1) / threads;
	if (blocks <= 0)
		blocks = 1;

	ndb_cuda_rf_split_kernel<<<blocks, threads>>>(features,
		labels,
		n_samples,
		feature_dim,
		feature_idx,
		threshold,
		class_count,
		left_counts_dev,
		right_counts_dev);
	status = cudaGetLastError();
	if (status != cudaSuccess)
		return -1;
	status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
		return -1;

	return 0;
}

extern "C" int
ndb_cuda_rf_infer(const NdbCudaRfNode *nodes,
	const NdbCudaRfTreeHeader *trees,
	int tree_count,
	const float *input,
	int feature_dim,
	int class_count,
	int *votes)
{
	NdbCudaRfNode *d_nodes = NULL;
	NdbCudaRfTreeHeader *d_trees = NULL;
	float *d_input = NULL;
	int *d_votes = NULL;
	cudaError_t status;
	int total_nodes = 0;
	int i;
	int threads = 256;
	int blocks;

	if (tree_count <= 0 || class_count <= 0)
		return -1;
	if (nodes == NULL || trees == NULL || input == NULL || votes == NULL)
		return -1;

	for (i = 0; i < tree_count; i++)
		total_nodes += trees[i].node_count;

	status = cudaMalloc(
		(void **)&d_nodes, sizeof(NdbCudaRfNode) * total_nodes);
	if (status != cudaSuccess)
		goto error;
	status = cudaMalloc(
		(void **)&d_trees, sizeof(NdbCudaRfTreeHeader) * tree_count);
	if (status != cudaSuccess)
		goto error;
	status = cudaMalloc((void **)&d_input, sizeof(float) * feature_dim);
	if (status != cudaSuccess)
		goto error;
	status = cudaMalloc((void **)&d_votes, sizeof(int) * class_count);
	if (status != cudaSuccess)
		goto error;

	status = cudaMemcpy(d_nodes,
		nodes,
		sizeof(NdbCudaRfNode) * total_nodes,
		cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		goto error;
	status = cudaMemcpy(d_trees,
		trees,
		sizeof(NdbCudaRfTreeHeader) * tree_count,
		cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		goto error;
	status = cudaMemcpy(d_input,
		input,
		sizeof(float) * feature_dim,
		cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		goto error;
	status = cudaMemset(d_votes, 0, sizeof(int) * class_count);
	if (status != cudaSuccess)
		goto error;

	blocks = (tree_count + threads - 1) / threads;
	if (blocks <= 0)
		blocks = 1;

	ndb_cuda_rf_predict_kernel<<<blocks, threads>>>(d_nodes,
		d_trees,
		tree_count,
		d_input,
		feature_dim,
		class_count,
		d_votes);
	status = cudaGetLastError();
	if (status != cudaSuccess)
		goto error;
	status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
		goto error;

	status = cudaMemcpy(votes,
		d_votes,
		sizeof(int) * class_count,
		cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
		goto error;

	cudaFree(d_nodes);
	cudaFree(d_trees);
	cudaFree(d_input);
	cudaFree(d_votes);
	return 0;

error:
	if (d_nodes)
		cudaFree(d_nodes);
	if (d_trees)
		cudaFree(d_trees);
	if (d_input)
		cudaFree(d_input);
	if (d_votes)
		cudaFree(d_votes);
	return -1;
}

/*
 * Batch prediction kernel: 2D grid
 * grid.x = samples, grid.y = trees
 * Each thread processes one (sample, tree) pair
 */
__global__ static void
ndb_cuda_rf_predict_batch_kernel(const NdbCudaRfNode *nodes,
	const NdbCudaRfTreeHeader *trees,
	int tree_count,
	const float *features,	/* n_samples x feature_dim */
	int n_samples,
	int feature_dim,
	int class_count,
	int *votes)		/* n_samples x class_count */
{
	int sample_stride = blockDim.x * gridDim.x;
	int tree_stride = blockDim.y * gridDim.y;
	int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int tree_idx = blockIdx.y * blockDim.y + threadIdx.y;
	int t;

	/* Grid stride loops: each thread processes multiple (sample, tree) pairs */
	for (; sample_idx < n_samples; sample_idx += sample_stride)
	{
		for (t = tree_idx; t < tree_count; t += tree_stride)
		{
			const NdbCudaRfTreeHeader *tree = &trees[t];

			/* Guard against zero-node trees */
			if (tree->node_count <= 0)
				continue;

			const NdbCudaRfNode *tree_nodes = nodes + tree->nodes_start;
			const float *input = features + (sample_idx * feature_dim);
			int idx = (tree->root_index >= 0) ? tree->root_index : 0;
			int steps = 0;
			int max_steps = tree->node_count + 1;
			int cls = 0;

			/* Pre-check max_feature_index if available */
			if (tree->max_feature_index >= feature_dim)
			{
				/* Tree requires features beyond input dimension, use root prediction */
				if (tree->node_count > 0)
					cls = (int)rintf(tree_nodes[0].value);
				else
					cls = 0; /* Default fallback */

				if (cls >= 0 && cls < class_count)
				{
					int *sample_votes = votes + (sample_idx * class_count);
					atomicAdd(&sample_votes[cls], 1);
				}
				continue;
			}

			/* Walk tree for this sample */
			while (steps < max_steps && idx >= 0 && idx < tree->node_count)
			{
				const NdbCudaRfNode *node = &tree_nodes[idx];

				if (node->feature_idx < 0
					|| node->feature_idx >= feature_dim)
				{
					cls = (int)rintf(node->value);
					break;
				}

				float val = input[node->feature_idx];

				if (val <= node->threshold)
					idx = node->left_child;
				else
					idx = node->right_child;

				steps++;
			}

			if (steps >= max_steps || idx < 0 || idx >= tree->node_count)
				cls = (int)rintf(tree_nodes[0].value);

			/* Atomic add vote for this sample's class */
			if (cls >= 0 && cls < class_count)
			{
				int *sample_votes = votes + (sample_idx * class_count);
				atomicAdd(&sample_votes[cls], 1);
			}
		}
	}
}

/*
 * Host wrapper for batch prediction kernel
 * Note: d_votes must be cleared by caller before calling this function
 */
extern "C" int
launch_rf_predict_batch_kernel(const NdbCudaRfNode *d_nodes,
	const NdbCudaRfTreeHeader *d_trees,
	int tree_count,
	const float *d_features,
	int n_samples,
	int feature_dim,
	int class_count,
	int *d_votes)
{
	dim3 threads_per_block(16, 16);	/* 16x16 = 256 threads per block */
	dim3 blocks;
	cudaError_t status;

	if (d_nodes == NULL || d_trees == NULL || d_features == NULL
		|| d_votes == NULL)
		return -1;

	if (n_samples <= 0 || tree_count <= 0 || feature_dim <= 0
		|| class_count <= 0)
		return -1;

	/* Calculate grid dimensions - no truncation needed with grid stride loops */
	blocks.x = (n_samples + threads_per_block.x - 1) / threads_per_block.x;
	blocks.y = (tree_count + threads_per_block.y - 1) / threads_per_block.y;

	if (blocks.x == 0)
		blocks.x = 1;
	if (blocks.y == 0)
		blocks.y = 1;

	/* Clear old error before launch */
	cudaGetLastError();

	ndb_cuda_rf_predict_batch_kernel<<<blocks, threads_per_block>>>(
		d_nodes,
		d_trees,
		tree_count,
		d_features,
		n_samples,
		feature_dim,
		class_count,
		d_votes);

	status = cudaGetLastError();
	if (status != cudaSuccess)
		return -1;

	status = cudaDeviceSynchronize();
	return (status == cudaSuccess) ? 0 : -1;
}
