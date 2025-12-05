/*-------------------------------------------------------------------------
 *
 * neurondb_flash_attention.h
 *    Flash Attention 2 API declarations
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/neurondb_flash_attention.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_FLASH_ATTENTION_H
#define NEURONDB_FLASH_ATTENTION_H

#include "postgres.h"

#ifdef NDB_GPU_CUDA
#include <cuda_runtime.h>

/* Flash Attention kernel launcher */
extern cudaError_t launch_flash_attention(const float *Q,
										  const float *K,
										  const float *V,
										  float *output,
										  int batch_size,
										  int seq_len,
										  int head_dim,
										  cudaStream_t stream);
#endif

#endif							/* NEURONDB_FLASH_ATTENTION_H */
