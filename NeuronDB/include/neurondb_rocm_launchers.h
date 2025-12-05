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
#endif

#ifdef __cplusplus
}								/* extern "C" */
#endif

#endif							/* NEURONDB_ROCM_LAUNCHERS_H */
