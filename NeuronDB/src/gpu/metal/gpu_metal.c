/*-------------------------------------------------------------------------
 *
 * gpu_metal.c
 *    Backend C wrapper.
 *
 * This module provides PostgreSQL integration for backend operations.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/metal/gpu_metal.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/elog.h"

#include "neurondb_config.h"
#include "neurondb_gpu.h"
#include "gpu_metal.h"

#ifdef NDB_GPU_METAL

#include "gpu_metal_wrapper.h"
#include <string.h>

/* Initialize Metal backend */
bool
neurondb_gpu_metal_init(void)
{

	if (metal_backend_init())
	{
		char		caps[512];

		metal_backend_get_capabilities(caps, sizeof(caps));

		elog(LOG, "neurondb: ✅ Metal GPU initialized: %s", caps);
		elog(LOG,
			 "neurondb: Metal device: %s",
			 metal_backend_device_name());
		return true;
	}

	return false;
}

/* Cleanup Metal resources */
void
neurondb_gpu_metal_cleanup(void)
{
	metal_backend_cleanup();
}

/* Check if Metal is available */
bool
neurondb_gpu_metal_is_available(void)
{
	return metal_backend_is_available();
}

/* Get Metal device name */
const char *
neurondb_gpu_metal_device_name(void)
{
	return metal_backend_device_name();
}

/* L2 distance calculation using Metal */
float
neurondb_gpu_metal_l2_distance(const float *a, const float *b, int dim)
{
	return metal_backend_l2_distance(a, b, dim);
}

/* Cosine distance calculation using Metal */
float
neurondb_gpu_metal_cosine_distance(const float *a, const float *b, int dim)
{
	return metal_backend_cosine_distance(a, b, dim);
}

/* Inner product calculation using Metal */
float
neurondb_gpu_metal_inner_product(const float *a, const float *b, int dim)
{
	return metal_backend_inner_product(a, b, dim);
}

/* Batch L2 distance calculation using Metal */
void
neurondb_gpu_metal_batch_l2(const float *queries,
							const float *targets,
							int num_queries,
							int num_targets,
							int dim,
							float *distances)
{
	/* For small batches, not worth Metal overhead */
	if (num_queries * num_targets < 1000)
		return;

	/* Use individual distance calculations */
	for (int i = 0; i < num_queries; i++)
	{
		for (int j = 0; j < num_targets; j++)
		{
			float		dist = metal_backend_l2_distance(
														 queries + i * dim, targets + j * dim, dim);

			if (dist >= 0.0f)
				distances[i * num_targets + j] = dist;
		}
	}
}

/* Get Metal device information */
void
neurondb_gpu_metal_device_info(char *name,
							   size_t name_len,
							   uint64_t * total_memory,
							   uint64_t * free_memory)
{
	metal_backend_device_info(name, name_len, total_memory, free_memory);
}

#else							/* !NDB_GPU_METAL */

/* Stub functions when Metal is not available */
bool
neurondb_gpu_metal_init(void)
{
	return false;
}
void
neurondb_gpu_metal_cleanup(void)
{
}
bool
neurondb_gpu_metal_is_available(void)
{
	return false;
}
const char *
neurondb_gpu_metal_device_name(void)
{
	return "Not compiled";
}
float
neurondb_gpu_metal_l2_distance(const float *a, const float *b, int dim)
{
	return -1.0f;
}
float
neurondb_gpu_metal_cosine_distance(const float *a, const float *b, int dim)
{
	return -1.0f;
}
float
neurondb_gpu_metal_inner_product(const float *a, const float *b, int dim)
{
	return -1.0f;
}
void
neurondb_gpu_metal_batch_l2(const float *q,
							const float *t,
							int nq,
							int nt,
							int d,
							float *dist)
{
}
void
neurondb_gpu_metal_device_info(char *n, size_t nl, uint64_t * tm, uint64_t * fm)
{
}

#endif							/* NDB_GPU_METAL */
