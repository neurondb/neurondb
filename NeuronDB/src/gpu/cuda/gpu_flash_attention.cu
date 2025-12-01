/*-------------------------------------------------------------------------
 *
 * gpu_flash_attention.cu
 *    Flash Attention 2 CUDA kernels for efficient reranking
 *
 * Implements memory-efficient attention mechanism (O(N) vs O(N²))
 * for cross-encoder reranking models. Supports long context windows (8K+).
 *
 * Based on Flash Attention 2 paper:
 * "Flash Attention 2: Faster Attention with Better Parallelism and Work Partitioning"
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/cuda/gpu_flash_attention.cu
 *
 *-------------------------------------------------------------------------
 */

#ifdef NDB_GPU_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#endif

#ifdef NDB_GPU_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#endif

#include <float.h>
#include <math.h>
#include "neurondb_cuda_launchers.h"

/*
 * Flash Attention 2 kernel for cross-encoder reranking
 *
 * Computes attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
 * using tiling to reduce memory from O(N²) to O(N).
 *
 * Args:
 *   Q: Query matrix [batch, seq_len, head_dim]
 *   K: Key matrix [batch, seq_len, head_dim]
 *   V: Value matrix [batch, seq_len, head_dim]
 *   output: Output matrix [batch, seq_len, head_dim]
 *   batch_size: Number of sequences
 *   seq_len: Sequence length
 *   head_dim: Attention head dimension
 *   scale: Scaling factor (1.0 / sqrt(head_dim))
 */
extern "C" __global__ void
flash_attention_kernel(const float *__restrict__ Q,
	const float *__restrict__ K,
	const float *__restrict__ V,
	float *__restrict__ output,
	int batch_size,
	int seq_len,
	int head_dim,
	float scale)
{
	/* Tile size for block-wise computation - reduced to fit 48KB shared memory limit */
	const int TILE_M = 32; /* Tile size for query (reduced from 64 to fit shared memory) */
	const int TILE_N = 32; /* Tile size for key/value (reduced from 64 to fit shared memory) */

	int batch_idx = blockIdx.z;
	int tile_m = blockIdx.y;
	/* tile_n reserved for future use - suppress unused warning */
	(void)blockIdx.x;

	/* Shared memory for tiles - reduced sizes to fit 48KB limit */
	/* Q_tile: 32*64*4 = 8KB, K_tile: 32*64*4 = 8KB, V_tile: 32*64*4 = 8KB, S_tile: 32*32*4 = 4KB = 28KB total */
	__shared__ float Q_tile[TILE_M][64];
	__shared__ float K_tile[TILE_N][64];
	__shared__ float V_tile[TILE_N][64];
	__shared__ float S_tile[TILE_M][TILE_N];

	int tid = threadIdx.x;
	int tid_y = threadIdx.y;

	/* Initialize output accumulator */
	float acc[64] = {0.0f};
	float max_val = -FLT_MAX;
	float sum_exp = 0.0f;

	/* Process tiles */
	for (int n = 0; n < (seq_len + TILE_N - 1) / TILE_N; n++)
	{
		/* Load Q tile */
		if (tile_m * TILE_M + tid_y < seq_len && tid < head_dim)
		{
			int q_idx = batch_idx * seq_len * head_dim +
				(tile_m * TILE_M + tid_y) * head_dim + tid;
			Q_tile[tid_y][tid] = Q[q_idx];
		}
		else
		{
			Q_tile[tid_y][tid] = 0.0f;
		}

		/* Load K and V tiles */
		if (n * TILE_N + tid_y < seq_len && tid < head_dim)
		{
			int kv_idx = batch_idx * seq_len * head_dim +
				(n * TILE_N + tid_y) * head_dim + tid;
			K_tile[tid_y][tid] = K[kv_idx];
			V_tile[tid_y][tid] = V[kv_idx];
		}
		else
		{
			K_tile[tid_y][tid] = 0.0f;
			V_tile[tid_y][tid] = 0.0f;
		}

		__syncthreads();

		/* Compute attention scores S = QK^T * scale */
		if (tile_m * TILE_M + tid_y < seq_len && n * TILE_N + tid < seq_len)
		{
			float score = 0.0f;

			for (int d = 0; d < head_dim; d++)
			{
				score += Q_tile[tid_y][d] * K_tile[tid][d];
			}

			S_tile[tid_y][tid] = score * scale;
		}

		__syncthreads();

		/* Online softmax and accumulate */
		if (tile_m * TILE_M + tid_y < seq_len)
		{
			float local_max = -FLT_MAX;

			/* Find max in this tile */
			for (int i = 0; i < TILE_N && n * TILE_N + i < seq_len; i++)
			{
				if (S_tile[tid_y][i] > local_max)
					local_max = S_tile[tid_y][i];
			}

			/* Update global max */
			if (local_max > max_val)
			{
				sum_exp *= expf(max_val - local_max);
				max_val = local_max;
			}
			else
			{
				sum_exp *= expf(local_max - max_val);
			}

			/* Compute exp and accumulate */
			for (int i = 0; i < TILE_N && n * TILE_N + i < seq_len; i++)
			{
				float exp_val = expf(S_tile[tid_y][i] - max_val);
				sum_exp += exp_val;

				/* Accumulate weighted values */
				for (int d = 0; d < head_dim; d++)
				{
					acc[d] += exp_val * V_tile[i][d];
				}
			}
		}

		__syncthreads();
	}

	/* Write output */
	if (tile_m * TILE_M + tid_y < seq_len && tid < head_dim)
	{
		int out_idx = batch_idx * seq_len * head_dim +
			(tile_m * TILE_M + tid_y) * head_dim + tid;

		if (sum_exp > 1e-10f)
			output[out_idx] = acc[tid] / sum_exp;
		else
			output[out_idx] = 0.0f;
	}
}

#if defined(NDB_GPU_CUDA)

/*
 * Host launch wrapper for Flash Attention kernel
 */
extern "C" cudaError_t
launch_flash_attention(const float *Q,
	const float *K,
	const float *V,
	float *output,
	int batch_size,
	int seq_len,
	int head_dim,
	cudaStream_t stream)
{
	if (batch_size <= 0 || seq_len <= 0 || head_dim <= 0 ||
		Q == NULL || K == NULL || V == NULL || output == NULL)
		return cudaSuccess;

	float scale = 1.0f / sqrtf((float)head_dim);

	/* Grid dimensions: [num_key_tiles, num_query_tiles, batch] */
	/* Updated tile sizes to match kernel tile size (32) */
	int tile_n = (seq_len + 31) / 32;
	int tile_m = (seq_len + 31) / 32;

	dim3 grid(tile_n, tile_m, batch_size);
	dim3 block(64, 32); /* Match TILE_M = 32 for y-dimension */

	flash_attention_kernel<<<grid, block, 0, stream>>>(
		Q, K, V, output, batch_size, seq_len, head_dim, scale);

	return cudaGetLastError();
}

#endif /* NDB_GPU_CUDA */

