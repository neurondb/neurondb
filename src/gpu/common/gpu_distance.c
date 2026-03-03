/*-------------------------------------------------------------------------
 *
 * gpu_distance.c
 *    Accelerated distance operations.
 *
 * This module implements L2, cosine, and inner product distance metrics
 * for high-performance vector operations.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/common/gpu_distance.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/timestamp.h"

#include "neurondb_gpu.h"
#include "neurondb_gpu_backend.h"
#include "neurondb_constants.h"

#include <math.h>
#include <string.h>

float
neurondb_gpu_l2_distance(const float *vec1, const float *vec2, int dim)
{
	const ndb_gpu_backend *backend;
	float		result = -1.0f;

	if (NDB_COMPUTE_MODE_IS_CPU())
		return -1.0f;

	if (!neurondb_gpu_is_available())
		return -1.0f;

	backend = ndb_gpu_get_active_backend();
	if (!backend || !backend->launch_l2_distance)
	{
		return -1.0f;
	}

	if (backend->launch_l2_distance(vec1, vec2, &result, 1, dim, NULL) != 0)
		return -1.0f;

	return result;
}

float
neurondb_gpu_cosine_distance(const float *vec1, const float *vec2, int dim)
{
	const ndb_gpu_backend *backend;
	float		result = -1.0f;

	if (NDB_COMPUTE_MODE_IS_CPU())
		return -1.0f;

	if (!neurondb_gpu_is_available())
		return -1.0f;

	backend = ndb_gpu_get_active_backend();
	if (!backend || !backend->launch_cosine)
	{
		return -1.0f;
	}

	if (backend->launch_cosine(vec1, vec2, &result, 1, dim, NULL) != 0)
		return -1.0f;

	return result;
}

float
neurondb_gpu_inner_product(const float *vec1, const float *vec2, int dim)
{
	const ndb_gpu_backend *backend;
	int			i;
	float		dot = 0.0f;

	if (NDB_COMPUTE_MODE_IS_CPU())
	{
		/* CPU fallback */
		for (i = 0; i < dim; i++)
			dot += vec1[i] * vec2[i];
		return -dot;
	}

	if (!neurondb_gpu_is_available())
	{
		for (i = 0; i < dim; i++)
			dot += vec1[i] * vec2[i];
		return -dot;			/* Negative for maximum inner product search */
	}

	backend = ndb_gpu_get_active_backend();
	if (!backend)
	{
		/* CPU fallback */
		for (i = 0; i < dim; i++)
			dot += vec1[i] * vec2[i];
		return -dot;
	}

	/* Try to use backend if it supports inner product via cosine */
	/* For now, use CPU fallback */

	for (i = 0; i < dim; i++)
		dot += vec1[i] * vec2[i];

	return -dot;
}
