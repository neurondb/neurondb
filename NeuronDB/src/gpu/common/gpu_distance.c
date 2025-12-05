/*-------------------------------------------------------------------------
 *
 * gpu_distance.c
 *    Accelerated distance operations.
 *
 * This module implements L2, cosine, and inner product distance metrics
 * for high-performance vector operations.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
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

/*
 * GPU L2 distance (Euclidean)
 */
float
neurondb_gpu_l2_distance(const float *vec1, const float *vec2, int dim)
{
	const ndb_gpu_backend *backend;
	float		result = -1.0f;

	/* CPU mode: never run GPU code */
	if (NDB_COMPUTE_MODE_IS_CPU())
		return -1.0f;

	if (!neurondb_gpu_is_available())
		return -1.0f;

	backend = ndb_gpu_get_active_backend();
	if (!backend || !backend->launch_l2_distance)
	{
		elog(DEBUG1,
			 "neurondb: GPU l2 distance not implemented for backend "
			 "%s; using CPU fallback",
			 backend && backend->name ? backend->name : "unknown");
		return -1.0f;
	}

	if (backend->launch_l2_distance(vec1, vec2, &result, 1, dim, NULL) != 0)
		return -1.0f;

	return result;
}

/*
 * GPU cosine distance
 */
float
neurondb_gpu_cosine_distance(const float *vec1, const float *vec2, int dim)
{
	const ndb_gpu_backend *backend;
	float		result = -1.0f;

	/* CPU mode: never run GPU code */
	if (NDB_COMPUTE_MODE_IS_CPU())
		return -1.0f;

	if (!neurondb_gpu_is_available())
		return -1.0f;

	backend = ndb_gpu_get_active_backend();
	if (!backend || !backend->launch_cosine)
	{
		elog(DEBUG1,
			 "neurondb: GPU cosine distance not implemented for "
			 "backend %s; using CPU fallback",
			 backend && backend->name ? backend->name : "unknown");
		return -1.0f;
	}

	if (backend->launch_cosine(vec1, vec2, &result, 1, dim, NULL) != 0)
		return -1.0f;

	return result;
}

/*
 * GPU inner product (negative for maximum inner product search)
 */
float
neurondb_gpu_inner_product(const float *vec1, const float *vec2, int dim)
{
	const ndb_gpu_backend *backend;
	int			i;
	float		dot = 0.0f;

	/* CPU mode: never run GPU code */
	if (NDB_COMPUTE_MODE_IS_CPU())
	{
		/* CPU fallback */
		for (i = 0; i < dim; i++)
			dot += vec1[i] * vec2[i];
		return -dot;			/* Negative for maximum inner product search */
	}

	if (!neurondb_gpu_is_available())
	{
		/* CPU fallback */
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
	elog(DEBUG1,
		 "neurondb: GPU inner product using CPU fallback (backend: %s)",
		 backend->name ? backend->name : "unknown");

	for (i = 0; i < dim; i++)
		dot += vec1[i] * vec2[i];

	return -dot;
}
