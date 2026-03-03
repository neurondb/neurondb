/*-------------------------------------------------------------------------
 *
 * neurondb_rocm_launchers.h
 *     Host-callable HIP launcher prototypes for ROCm backend.
 *
 * These functions are implemented in HIP compilation units and exposed with
 * C linkage so C files compiled with GCC can invoke them without pulling in
 * HIP-specific headers directly.
 *
 *-------------------------------------------------------------------------*/

#ifndef NEURONDB_ROCM_LAUNCHERS_H
#define NEURONDB_ROCM_LAUNCHERS_H

#include <stdint.h>

#ifdef NDB_GPU_HIP
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef NDB_GPU_HIP
/* rocBLAS handle accessor */
extern rocblas_handle ndb_rocm_get_rocblas_handle(void);

hipError_t launch_quantize_fp32_to_fp16_hip(const float *input,
											 void *output,
											 int count,
											 hipStream_t stream);

hipError_t launch_quantize_fp32_to_int8_hip(const float *input,
											 signed char *output,
											 int count,
											 float scale,
											 hipStream_t stream);

hipError_t launch_quantize_fp32_to_int4_hip(const float *input,
											 unsigned char *output,
											 int count,
											 float scale,
											 hipStream_t stream);

hipError_t launch_quantize_fp32_to_fp8_e4m3_hip(const float *input,
												 unsigned char *output,
												 int count,
												 hipStream_t stream);

hipError_t launch_quantize_fp32_to_fp8_e5m2_hip(const float *input,
												 unsigned char *output,
												 int count,
												 hipStream_t stream);

hipError_t launch_quantize_fp32_to_binary_hip(const float *input,
											   unsigned char *output,
											   int count,
											   hipStream_t stream);

int			gpu_kmeans_assign_hip(const float *h_vectors,
								  const float *h_centroids,
								  int32_t *h_assignments,
								  int nvec,
								  int k,
								  int dim);

int			gpu_kmeans_update_hip(const float *h_vectors,
								  const int32_t *h_assignments,
								  float *h_centroids,
								  int32_t *h_counts,
								  int nvec,
								  int k,
								  int dim);

int			gpu_pq_encode_batch_hip(const float *h_vectors,
									const float *h_codebooks,
									uint8_t *h_codes,
									int nvec,
									int dim,
									int m,
									int ks);
int			gpu_pq_asymmetric_distance_batch_hip(const float *h_query,
												  const uint8_t *h_codes,
												  const float *h_codebooks,
												  float *h_distances,
												  int nvec,
												  int dim,
												  int m,
												  int ks);
int			gpu_hnsw_search_hip(const float *h_query,
								 const float *h_nodes,
								 const uint32_t *h_neighbors,
								 const int32_t *h_neighbor_counts,
								 const int32_t *h_node_levels,
								 uint32_t entry_point,
								 int entry_level,
								 int dim,
								 int m,
								 int ef_search,
								 int k,
								 uint32_t *h_result_blocks,
								 float *h_result_distances);
int			gpu_hnsw_search_batch_hip(const float *h_queries,
										const float *h_nodes,
										const uint32_t *h_neighbors,
										const int32_t *h_neighbor_counts,
										const int32_t *h_node_levels,
										uint32_t entry_point,
										int entry_level,
										int num_queries,
										int dim,
										int m,
										int ef_search,
										int k,
										uint32_t *h_result_blocks,
										float *h_result_distances);
int			gpu_ivf_search_hip(const float *h_query,
								const float *h_centroids,
								const float *h_vectors,
								const int32_t *h_list_offsets,
								const int32_t *h_list_sizes,
								int nlists,
								int nprobe,
								int dim,
								int k,
								uint32_t *h_result_indices,
								float *h_result_distances);
int			gpu_ivf_search_batch_hip(const float *h_queries,
									   const float *h_centroids,
									   const float *h_vectors,
									   const int32_t *h_list_offsets,
									   const int32_t *h_list_sizes,
									   int num_queries,
									   int nlists,
									   int nprobe,
									   int dim,
									   int k,
									   uint32_t *h_result_indices,
									   float *h_result_distances);
#endif

#ifdef __cplusplus
}								/* extern "C" */
#endif

#endif							/* NEURONDB_ROCM_LAUNCHERS_H */
