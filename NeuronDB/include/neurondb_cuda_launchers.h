/*-------------------------------------------------------------------------
 *
 * neurondb_cuda_launchers.h
 *     Host-callable CUDA launcher prototypes.
 *
 * These functions are implemented in CUDA compilation units and exposed with
 * C linkage so C files compiled with GCC can invoke them without pulling in
 * CUDA-specific headers directly.
 *
 *-------------------------------------------------------------------------*/

#ifndef NEURONDB_CUDA_LAUNCHERS_H
#define NEURONDB_CUDA_LAUNCHERS_H

#include "neurondb_cuda_runtime.h"
#include <stdint.h>

#ifdef NDB_GPU_CUDA
#include <cublas_v2.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef NDB_GPU_CUDA
/* cuBLAS handle accessor */
extern cublasHandle_t ndb_cuda_get_cublas_handle(void);
#endif

cudaError_t launch_quantize_fp32_to_fp16(const float *input,
										 void *output,
										 int count,
										 cudaStream_t stream);

cudaError_t launch_quantize_fp32_to_int8(const float *input,
										 signed char *output,
										 int count,
										 float scale,
										 cudaStream_t stream);

cudaError_t launch_quantize_fp32_to_int4(const float *input,
										 unsigned char *output,
										 int count,
										 float scale,
										 cudaStream_t stream);

cudaError_t launch_quantize_fp32_to_fp8_e4m3(const float *input,
											 unsigned char *output,
											 int count,
											 cudaStream_t stream);

cudaError_t launch_quantize_fp32_to_fp8_e5m2(const float *input,
											 unsigned char *output,
											 int count,
											 cudaStream_t stream);

cudaError_t launch_quantize_fp32_to_binary(const float *input,
										   unsigned char *output,
										   int count,
										   cudaStream_t stream);

int			gpu_kmeans_assign(const float *h_vectors,
							  const float *h_centroids,
							  int32_t *h_assignments,
							  int nvec,
							  int k,
							  int dim);

int			gpu_kmeans_update(const float *h_vectors,
							  const int32_t *h_assignments,
							  float *h_centroids,
							  int32_t *h_counts,
							  int nvec,
							  int k,
							  int dim);

int			gpu_pq_encode_batch(const float *h_vectors,
								const float *h_codebooks,
								uint8_t *h_codes,
								int nvec,
								int dim,
								int m,
								int ks);

int			gpu_pq_asymmetric_distance_batch(const float *h_query,
											 const uint8_t *h_codes,
											 const float *h_codebooks,
											 float *h_distances,
											 int nvec,
											 int dim,
											 int m,
											 int ks);

#ifdef __cplusplus
}								/* extern "C" */
#endif

#endif							/* NEURONDB_CUDA_LAUNCHERS_H */
