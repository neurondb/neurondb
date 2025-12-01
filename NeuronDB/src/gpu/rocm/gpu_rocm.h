/*
 * gpu_rocm.h
 *     AMD ROCm GPU backend header
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 */

#ifndef NEURONDB_GPU_ROCM_H
#define NEURONDB_GPU_ROCM_H

#include <stdbool.h>
#include <stdint.h>

/* ROCm initialization and cleanup */
extern bool neurondb_gpu_rocm_init(void);
extern void neurondb_gpu_rocm_cleanup(void);
extern bool neurondb_gpu_rocm_is_available(void);
extern const char *neurondb_gpu_rocm_device_name(void);

/* ROCm distance operations */
extern float
			neurondb_gpu_rocm_l2_distance(const float *a, const float *b, int dim);
extern float
			neurondb_gpu_rocm_cosine_distance(const float *a, const float *b, int dim);
extern float
			neurondb_gpu_rocm_inner_product(const float *a, const float *b, int dim);

/* ROCm batch operations */
extern void neurondb_gpu_rocm_batch_l2(const float *queries,
									   const float *targets,
									   int num_queries,
									   int num_targets,
									   int dim,
									   float *distances);

/* Statistics */
extern uint64_t neurondb_gpu_rocm_get_operations_count(void);

#endif							/* NEURONDB_GPU_ROCM_H */
