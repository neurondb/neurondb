/*
 * gpu_metal_wrapper.h
 *     C header for Metal Objective-C implementation
 *
 * Provides C-compatible interface to Metal GPU acceleration.
 * NO include of PostgreSQL headers to avoid conflicts.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 */

#ifndef GPU_METAL_WRAPPER_H
#define GPU_METAL_WRAPPER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Metal backend lifecycle */
extern bool metal_backend_init(void);
extern void metal_backend_cleanup(void);
extern bool metal_backend_is_available(void);
extern const char *metal_backend_device_name(void);

/* Metal GPU distance operations - ACTUAL GPU ACCELERATION */
extern float metal_backend_l2_distance(const float *a, const float *b, int dim);
extern float
metal_backend_cosine_distance(const float *a, const float *b, int dim);
extern float
metal_backend_inner_product(const float *a, const float *b, int dim);

/* Metal GPU batch operations - PARALLEL GPU EXECUTION */
extern void metal_backend_batch_l2(const float *queries,
	const float *targets,
	int num_queries,
	int num_targets,
	int dim,
	float *distances);

/* Metal device information */
extern void metal_backend_device_info(char *name,
	size_t name_len,
	uint64_t *total_memory,
	uint64_t *free_memory);

/* Metal GPU statistics */
extern uint64_t metal_backend_get_operations_count(void);
extern double metal_backend_get_avg_time_us(void);
extern void metal_backend_reset_statistics(void);
extern void metal_backend_get_capabilities(char *caps, size_t caps_len);

#ifdef __cplusplus
}
#endif

#endif /* GPU_METAL_WRAPPER_H */
