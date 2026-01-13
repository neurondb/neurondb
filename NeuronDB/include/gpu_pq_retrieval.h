/*-------------------------------------------------------------------------
 *
 * gpu_pq_retrieval.h
 *    GPU-accelerated Product Quantization retrieval interface
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 *-------------------------------------------------------------------------
 */

#ifndef GPU_PQ_RETRIEVAL_H
#define GPU_PQ_RETRIEVAL_H

#include <stdint.h>

/* GPU-accelerated PQ asymmetric search */
extern int neurondb_gpu_pq_asymmetric_search(const float *query,
											  const uint8_t *pq_codes,
											  const float *codebooks,
											  int num_vectors,
											  int dim,
											  int m,
											  int ks,
											  int k,
											  uint32_t *result_indices,
											  float *result_distances);

#endif /* GPU_PQ_RETRIEVAL_H */




