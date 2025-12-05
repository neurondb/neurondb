/*
 * gpu_metal.h
 *     Metal backend API for NeurondB GPU acceleration
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *     src/gpu/gpu_metal.h
 */

#ifndef NEURONDB_GPU_METAL_H
#define NEURONDB_GPU_METAL_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

/* Metal backend initialization */
extern bool neurondb_gpu_metal_init(void);
extern void neurondb_gpu_metal_cleanup(void);
extern bool neurondb_gpu_metal_is_available(void);
extern const char *neurondb_gpu_metal_device_name(void);

/* Metal distance operations */
extern float
			neurondb_gpu_metal_l2_distance(const float *a, const float *b, int dim);
extern float
			neurondb_gpu_metal_cosine_distance(const float *a, const float *b, int dim);
extern float
			neurondb_gpu_metal_inner_product(const float *a, const float *b, int dim);

/* Metal batch operations */
extern void neurondb_gpu_metal_batch_l2(const float *queries,
										const float *targets,
										int num_queries,
										int num_targets,
										int dim,
										float *distances);

/* Metal device information */
extern void neurondb_gpu_metal_device_info(char *name,
										   size_t name_len,
										   uint64_t * total_memory,
										   uint64_t * free_memory);

#endif							/* NEURONDB_GPU_METAL_H */
