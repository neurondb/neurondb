/*-------------------------------------------------------------------------
 *
 * gpu_hnsw.h
 *    GPU-accelerated HNSW search interface
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 *-------------------------------------------------------------------------
 */

#ifndef GPU_HNSW_H
#define GPU_HNSW_H

#include <stdint.h>

/* GPU-accelerated HNSW search */
extern int neurondb_gpu_hnsw_search(const float *query,
									 const float *nodes,
									 const uint32_t *neighbors,
									 const int32_t *neighbor_counts,
									 const int32_t *node_levels,
									 uint32_t entry_point,
									 int entry_level,
									 int dim,
									 int m,
									 int ef_search,
									 int k,
									 uint32_t *result_blocks,
									 float *result_distances);

/* GPU-accelerated batch HNSW search */
extern int neurondb_gpu_hnsw_search_batch(const float *queries,
										   const float *nodes,
										   const uint32_t *neighbors,
										   const int32_t *neighbor_counts,
										   const int32_t *node_levels,
										   uint32_t entry_point,
										   int entry_level,
										   int num_queries,
										   int dim,
										   int m,
										   int ef_search,
										   int k,
										   uint32_t *result_blocks,
										   float *result_distances);

#endif /* GPU_HNSW_H */




