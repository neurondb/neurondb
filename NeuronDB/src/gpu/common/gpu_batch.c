/*-------------------------------------------------------------------------
 *
 * gpu_batch.c
 *    Accelerated batch distance operations.
 *
 * This module computes distance matrices between query vectors and database
 * vectors efficiently using batch operations.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/common/gpu_batch.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include <math.h>
#include <string.h>

#include "neurondb_config.h"
#include "neurondb_gpu.h"
#include "neurondb_gpu_backend.h"
#include "neurondb_constants.h"

void
neurondb_gpu_batch_l2_distance(const float *queries,
							   const float *vectors,
							   float *results,
							   int num_queries,
							   int num_vectors,
							   int dim)
{
	const		ndb_gpu_backend *backend;
	int			q,
				v,
				d;
	float		diff,
				sum;

	/* CPU mode: never attempt GPU code, go straight to CPU fallback */
	if (NDB_COMPUTE_MODE_IS_CPU())
		goto cpu_fallback;

	if (!neurondb_gpu_is_available())
	{
		/* CPU fallback */
		goto cpu_fallback;
	}

	backend = ndb_gpu_get_active_backend();
	if (!backend)
	{
		goto cpu_fallback;
	}

	/* Try to use backend batch operation if available */
	/* For now, use CPU fallback until backend batch functions are added */
	elog(DEBUG1,
		 "neurondb: GPU batch L2 distance using CPU fallback (backend: %s)",
		 backend->name ? backend->name : "unknown");
	goto cpu_fallback;

cpu_fallback:
	/* Efficient CPU implementation using matrix operations */
	for (q = 0; q < num_queries; q++)
	{
		const float *query = queries + q * dim;

		for (v = 0; v < num_vectors; v++)
		{
			const float *vector = vectors + v * dim;

			sum = 0.0f;

			for (d = 0; d < dim; d++)
			{
				diff = query[d] - vector[d];
				sum += diff * diff;
			}

			results[q * num_vectors + v] = sqrtf(sum);
		}
	}
}

void
neurondb_gpu_batch_cosine_distance(const float *queries,
								   const float *vectors,
								   float *results,
								   int num_queries,
								   int num_vectors,
								   int dim)
{
	const		ndb_gpu_backend *backend;
	int			q,
				v,
				d;
	float		dot_product,
				norm_query,
				norm_vector,
				similarity;

	/* CPU mode: never attempt GPU code, go straight to CPU fallback */
	if (NDB_COMPUTE_MODE_IS_CPU())
		goto cpu_fallback;

	if (!neurondb_gpu_is_available())
	{
		/* CPU fallback */
		goto cpu_fallback;
	}

	backend = ndb_gpu_get_active_backend();
	if (!backend)
	{
		goto cpu_fallback;
	}

	/* Try to use backend batch operation if available */
	/* For now, use CPU fallback until backend batch functions are added */
	elog(DEBUG1,
		 "neurondb: GPU batch cosine distance using CPU fallback (backend: %s)",
		 backend->name ? backend->name : "unknown");
	goto cpu_fallback;

cpu_fallback:
	/* Efficient CPU implementation */
	for (q = 0; q < num_queries; q++)
	{
		const float *query = queries + q * dim;

		/* Precompute query norm */
		norm_query = 0.0f;
		for (d = 0; d < dim; d++)
			norm_query += query[d] * query[d];
		norm_query = sqrtf(norm_query);

		if (norm_query == 0.0f)
		{
			/*
			 * Zero vector - set all distances to 1.0 (maximum cosine
			 * distance)
			 */
			for (v = 0; v < num_vectors; v++)
				results[q * num_vectors + v] = 1.0f;
			continue;
		}

		for (v = 0; v < num_vectors; v++)
		{
			const float *vector = vectors + v * dim;

			/* Compute dot product */
			dot_product = 0.0f;
			for (d = 0; d < dim; d++)
				dot_product += query[d] * vector[d];

			/* Compute vector norm */
			norm_vector = 0.0f;
			for (d = 0; d < dim; d++)
				norm_vector += vector[d] * vector[d];
			norm_vector = sqrtf(norm_vector);

			if (norm_vector == 0.0f)
			{
				results[q * num_vectors + v] = 1.0f;
			}
			else
			{
				similarity = dot_product / (norm_query * norm_vector);
				results[q * num_vectors + v] = 1.0f - similarity;
			}
		}
	}
}
