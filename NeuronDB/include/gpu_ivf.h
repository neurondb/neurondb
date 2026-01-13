/*-------------------------------------------------------------------------
 *
 * gpu_ivf.h
 *    GPU-accelerated IVF search interface
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 *-------------------------------------------------------------------------
 */

#ifndef GPU_IVF_H
#define GPU_IVF_H

#include <stdint.h>

/* GPU-accelerated IVF search */
extern int neurondb_gpu_ivf_search(const float *query,
									const float *centroids,
									const float *vectors,
									const int32_t *list_offsets,
									const int32_t *list_sizes,
									int nlists,
									int nprobe,
									int dim,
									int k,
									uint32_t *result_indices,
									float *result_distances);

/* GPU-accelerated batch IVF search */
extern int neurondb_gpu_ivf_search_batch(const float *queries,
										 const float *centroids,
										 const float *vectors,
										 const int32_t *list_offsets,
										 const int32_t *list_sizes,
										 int num_queries,
										 int nlists,
										 int nprobe,
										 int dim,
										 int k,
										 uint32_t *result_indices,
										 float *result_distances);

#endif /* GPU_IVF_H */




