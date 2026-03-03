/*-------------------------------------------------------------------------
 *
 * gpu_index_build.c
 *    GPU-accelerated index building
 *
 * Implements GPU-accelerated HNSW graph construction and IVF K-means
 * clustering for faster index builds.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/common/gpu_index_build.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb_gpu_backend.h"
#include "neurondb_gpu.h"
#include "neurondb_constants.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include <math.h>
#include <stdlib.h>

#define HNSW_MAX_LEVEL 16

/*
 * GPU-accelerated HNSW graph construction
 *
 * Builds HNSW graph structure on GPU for faster index creation.
 */
int
neurondb_gpu_hnsw_build(const float *vectors,
						int num_vectors,
						int dim,
						int m,
						int ef_construction,
						uint32_t **result_nodes,
						uint32_t **result_neighbors,
						int32_t **result_neighbor_counts,
						int32_t **result_node_levels,
						uint32_t *entry_point,
						int *entry_level)
{
	const ndb_gpu_backend *backend;

	if (NDB_COMPUTE_MODE_IS_CPU())
		return -1;

	if (!neurondb_gpu_is_available())
		return -1;

	backend = ndb_gpu_get_active_backend();
	if (!backend)
		return -1;

	/* Use GPU-accelerated HNSW build if backend supports it */
	if (backend->launch_hnsw_build != NULL)
	{
		ndb_stream_t stream = NULL;

		/* Create stream for async execution if supported */
		if (backend->stream_create)
			backend->stream_create(&stream);

		int rc = backend->launch_hnsw_build(vectors,
											num_vectors,
											dim,
											m,
											ef_construction,
											result_nodes,
											result_neighbors,
											result_neighbor_counts,
											result_node_levels,
											entry_point,
											entry_level,
											stream);

		/* Synchronize stream if created */
		if (stream && backend->stream_synchronize)
			backend->stream_synchronize(stream);
		if (stream && backend->stream_destroy)
			backend->stream_destroy(stream);

		return rc;
	}

	/* Fallback: Use GPU-accelerated distance computation for neighbor selection */
	/* This hybrid approach uses GPU for expensive distance computations while
	 * keeping graph management on CPU for simplicity and correctness */
	if (backend->launch_l2_distance != NULL)
	{
		/* Allocate result arrays */
		int			i, j, l;
		float	   *distances = NULL;
		uint32_t   *indices = NULL;
		int			max_level = 0;
		MemoryContext oldctx;

		oldctx = MemoryContextSwitchTo(CurrentMemoryContext);

		/* Allocate arrays for results */
		nalloc(*result_nodes, uint32_t, num_vectors);
		nalloc(*result_node_levels, int32_t, num_vectors);

		/* Calculate levels for each vector (exponential distribution) */
		{
			float		ml = 1.0f / log(2.0f);	/* Default ML */
			int			level;

			for (i = 0; i < num_vectors; i++)
			{
				level = 0;
				while (level < HNSW_MAX_LEVEL - 1 && ((float) random() / (float) RAND_MAX) < exp(-level / ml))
					level++;
				(*result_node_levels)[i] = level;
				if (level > max_level)
					max_level = level;
			}
		}

		/* Set entry point (highest level node) */
		*entry_level = max_level;
		for (i = 0; i < num_vectors; i++)
		{
			if ((*result_node_levels)[i] == max_level)
			{
				*entry_point = (uint32_t) i;
				break;
			}
		}

		/* Allocate neighbor arrays (worst case: m*2 per level) */
		{
			int			total_neighbors = 0;

			for (i = 0; i < num_vectors; i++)
			{
				int			level = (*result_node_levels)[i];

				total_neighbors += (level + 1) * m * 2;
			}

			nalloc(*result_neighbors, uint32_t, total_neighbors);
			nalloc(*result_neighbor_counts, int32_t, num_vectors * (max_level + 1));
		}

		/* Build graph level by level, using GPU for distance computation */
		{
			int			neighbor_offset = 0;
			int			count_offset = 0;

			/* Process from highest level to level 0 */
			for (l = max_level; l >= 0; l--)
			{
				int			lm = (l == 0 ? m * 2 : m);
				int			level_vector_count = 0;
				int			*level_indices = NULL;

				/* Collect vectors at this level or higher */
				for (i = 0; i < num_vectors; i++)
				{
					if ((*result_node_levels)[i] >= l)
						level_vector_count++;
				}

				if (level_vector_count == 0)
					continue;

				nalloc(level_indices, int, level_vector_count);
				{
					int			idx = 0;

					for (i = 0; i < num_vectors; i++)
					{
						if ((*result_node_levels)[i] >= l)
							level_indices[idx++] = i;
					}
				}

				/* For each vector at this level, find neighbors */
				for (i = 0; i < num_vectors; i++)
				{
					if ((*result_node_levels)[i] < l)
					{
						(*result_neighbor_counts)[count_offset + l] = 0;
						continue;
					}

					/* Use GPU to compute distances to all candidates at this level */
					nalloc(distances, float, level_vector_count);
					nalloc(indices, uint32_t, level_vector_count);

					/* Build candidate matrix (all vectors at level >= l) */
					{
						float	   *candidate_matrix = NULL;
						ndb_stream_t dist_stream = NULL;
						int			rc;

						nalloc(candidate_matrix, float, level_vector_count * dim);

						for (j = 0; j < level_vector_count; j++)
						{
							memcpy(candidate_matrix + j * dim,
								   vectors + level_indices[j] * dim,
								   dim * sizeof(float));
						}

						/* Compute distances using GPU */
						if (backend->stream_create)
							backend->stream_create(&dist_stream);

						rc = backend->launch_l2_distance(vectors + i * dim,
														 candidate_matrix,
														 distances,
														 level_vector_count,
														 dim,
														 dist_stream);

						if (dist_stream && backend->stream_synchronize)
							backend->stream_synchronize(dist_stream);
						if (dist_stream && backend->stream_destroy)
							backend->stream_destroy(dist_stream);

						pfree(candidate_matrix);

						if (rc != 0)
						{
							pfree(distances);
							pfree(indices);
							pfree(level_indices);
							MemoryContextSwitchTo(oldctx);
							return -1;
						}
					}

					/* Initialize indices */
					for (j = 0; j < level_vector_count; j++)
						indices[j] = (uint32_t) level_indices[j];

					/* Select top m neighbors (exclude self) */
					{
						int			selected = 0;
						int			k;

						/* Partial sort to get top m (excluding self) */
						for (k = 0; k < level_vector_count && selected < lm; k++)
						{
							if (indices[k] == (uint32_t) i)
								continue;	/* Skip self */

							/* Insert into sorted position */
							int			pos = selected;

							while (pos > 0 && distances[indices[pos - 1]] > distances[indices[k]])
								pos--;

							/* Shift and insert */
							if (pos < selected)
							{
								for (int p = selected; p > pos; p--)
								{
									indices[p] = indices[p - 1];
									distances[p] = distances[p - 1];
								}
							}
							indices[pos] = indices[k];
							distances[pos] = distances[k];
							selected++;
						}

						(*result_neighbor_counts)[count_offset + l] = selected;

						/* Copy selected neighbors */
						for (j = 0; j < selected; j++)
						{
							(*result_neighbors)[neighbor_offset++] = indices[j];
						}
					}

					pfree(distances);
					pfree(indices);
				}

				count_offset += num_vectors;
				pfree(level_indices);
			}
		}

		/* Set node indices */
		for (i = 0; i < num_vectors; i++)
			(*result_nodes)[i] = (uint32_t) i;

		MemoryContextSwitchTo(oldctx);
		return 0;
	}

	/* No GPU support available */
	return -1;
}

/*
 * GPU-accelerated K-means for IVF
 *
 * Computes centroids for IVF index using GPU-accelerated K-means.
 */
int
neurondb_gpu_ivf_kmeans(const float *vectors,
						int num_vectors,
						int dim,
						int k,
						int max_iterations,
						float *centroids,
						int *assignments)
{
	const ndb_gpu_backend *backend;

	if (NDB_COMPUTE_MODE_IS_CPU())
		return -1;

	if (!neurondb_gpu_is_available())
		return -1;

	backend = ndb_gpu_get_active_backend();
	if (!backend)
		return -1;

	/* Use existing GPU K-means implementation if available */
	/* The backend already has launch_kmeans_assign and launch_kmeans_update */
	if (backend->launch_kmeans_assign != NULL && backend->launch_kmeans_update != NULL)
	{
		/* Initialize centroids (random or k-means++) */
		/* For now, use random initialization */
		for (int i = 0; i < k; i++)
		{
			int			rand_idx = random() % num_vectors;
			memcpy(centroids + i * dim, vectors + rand_idx * dim, dim * sizeof(float));
		}

		/* Run K-means iterations */
		for (int iter = 0; iter < max_iterations; iter++)
		{
			/* Assign vectors to nearest centroids */
			backend->launch_kmeans_assign(vectors,
										  centroids,
										  assignments,
										  num_vectors,
										  dim,
										  k,
										  NULL); /* stream */

			/* Update centroids */
			backend->launch_kmeans_update(vectors,
										  assignments,
										  centroids,
										  num_vectors,
										  dim,
										  k,
										  NULL); /* stream */
		}

		return 0;
	}

	return -1;
}

