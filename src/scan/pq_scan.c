/*-------------------------------------------------------------------------
 *
 * pq_scan.c
 *    PQ index scan implementation with two-stage retrieval
 *
 * Implements:
 * - Stage 1: Coarse search using PQ-encoded vectors (GPU-accelerated)
 * - Stage 2: Fine rerank with full-precision vectors
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/scan/pq_scan.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_types.h"
#include "fmgr.h"
#include "access/relscan.h"
#include "utils/rel.h"
#include "storage/bufmgr.h"
#include "utils/builtins.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_gpu.h"
#include "gpu_pq_retrieval.h"
#include <math.h>
#include <float.h>

#ifdef __AVX2__
#include <immintrin.h>
#define HAVE_AVX2 1
#else
#define HAVE_AVX2 0
#endif

/*
 * PQ scan state
 */
typedef struct PqScanState
{
	/* Query parameters */
	const float *query;
	int			dim;
	int			k;
	int			rerank_k;

	/* PQ parameters */
	int			m;				/* Number of subspaces */
	int			ks;				/* Codebook size */
	int			subspace_dim;

	/* Stage 1: Coarse search results */
	uint32_t   *coarse_indices;
	float	   *coarse_distances;
	int			coarse_count;

	/* Stage 2: Fine rerank results */
	uint32_t   *fine_indices;
	float	   *fine_distances;
	int			fine_count;

	/* Current result position */
	int			current_pos;
}			PqScanState;

/*
 * Initialize PQ scan
 */
static PqScanState *
pq_scan_init(const float *query, int dim, int k, int rerank_k, int m, int ks)
{
	PqScanState *state = NULL;

	nalloc(state, PqScanState, 1);
	NDB_CHECK_ALLOC(state, "state");

	state->query = query;
	state->dim = dim;
	state->k = k;
	state->rerank_k = rerank_k;
	state->m = m;
	state->ks = ks;
	state->subspace_dim = dim / m;

	nalloc(state->coarse_indices, uint32_t, rerank_k);
	nalloc(state->coarse_distances, float, rerank_k);
	nalloc(state->fine_indices, uint32_t, k);
	nalloc(state->fine_distances, float, k);

	state->coarse_count = 0;
	state->fine_count = 0;
	state->current_pos = 0;

	return state;
}

/*
 * Free PQ scan state
 */
static void
pq_scan_free(PqScanState * state)
{
	if (state)
	{
		pfree(state->coarse_indices);
		pfree(state->coarse_distances);
		pfree(state->fine_indices);
		pfree(state->fine_distances);
		pfree(state);
	}
}

/*
 * Stage 1: Coarse search using PQ codes
 * Uses GPU-accelerated asymmetric distance computation
 */
static int
pq_scan_coarse_search(PqScanState * state,
					  const uint8_t *pq_codes,
					  const float *codebooks,
					  int num_vectors)
{
	/* Use GPU for fast PQ distance computation */
	if (neurondb_gpu_is_available())
	{
		return neurondb_gpu_pq_asymmetric_search(state->query,
												 pq_codes,
												 codebooks,
												 num_vectors,
												 state->dim,
												 state->m,
												 state->ks,
												 state->rerank_k,
												 state->coarse_indices,
												 state->coarse_distances);
	}

	/*
	 * CPU fallback: Compute PQ asymmetric distances on CPU
	 * PQ asymmetric distance = sum over m subspaces of (distance from query subspace
	 * to codebook centroid indexed by PQ code)
	 */
	{
		float	   *pq_dists = NULL;
		int			i, j;
		int			subspace_dim = state->dim / state->m;

		/* Precompute distances from query subspaces to all codebook centroids */
		nalloc(pq_dists, float, state->m * state->ks);
		NDB_CHECK_ALLOC(pq_dists, "pq_dists");

		for (i = 0; i < state->m; i++)
		{
			int			query_start = i * subspace_dim;
			float	   *sub_dists = pq_dists + i * state->ks;

			/* Compute distances from query subspace to all centroids in this subspace */
			for (j = 0; j < state->ks; j++)
			{
				const float *centroid = codebooks + (i * state->ks + j) * subspace_dim;
				float		dist = 0.0f;
				int			d;

#if HAVE_AVX2
				/* SIMD-optimized distance computation for AVX2 */
				if (subspace_dim >= 8)
				{
					__m256 sum_vec = _mm256_setzero_ps();
					int		simd_end = subspace_dim & ~7;

					/* Process 8 elements at a time */
					for (d = 0; d < simd_end; d += 8)
					{
						__m256 q_vec = _mm256_loadu_ps(&state->query[query_start + d]);
						__m256 c_vec = _mm256_loadu_ps(&centroid[d]);
						__m256 diff = _mm256_sub_ps(q_vec, c_vec);
						__m256 sq = _mm256_mul_ps(diff, diff);
						sum_vec = _mm256_add_ps(sum_vec, sq);
					}

					/* Horizontal sum of 8 floats */
					__m128 v_low = _mm256_castps256_ps128(sum_vec);
					__m128 v_high = _mm256_extractf128_ps(sum_vec, 1);
					__m128 sum128 = _mm_add_ps(v_low, v_high);
					sum128 = _mm_hadd_ps(sum128, sum128);
					sum128 = _mm_hadd_ps(sum128, sum128);
					dist = _mm_cvtss_f32(sum128);

					/* Handle remaining elements */
					for (d = simd_end; d < subspace_dim; d++)
					{
						float		diff = state->query[query_start + d] - centroid[d];
						dist += diff * diff;
					}
				}
				else
#endif
				{
					/* Scalar fallback for small dimensions */
					for (d = 0; d < subspace_dim; d++)
					{
						float		diff = state->query[query_start + d] - centroid[d];
						dist += diff * diff;
					}
				}

				sub_dists[j] = dist;
			}
		}

		/* Compute PQ distances for all vectors and find top rerank_k */
		{
			struct
			{
				uint32_t	idx;
				float		dist;
			}		   *candidates = NULL;
			int			candidate_count = 0;
			int			capacity = state->rerank_k * 2;

			nalloc(candidates, struct { uint32_t idx; float dist; }, capacity);

			for (i = 0; i < num_vectors; i++)
			{
				float		pq_dist = 0.0f;
				int			sub;

				/* Sum distances across subspaces */
				for (sub = 0; sub < state->m; sub++)
				{
					uint8_t		code = pq_codes[i * state->m + sub];
					pq_dist += pq_dists[sub * state->ks + code];
				}

				/* Add to candidates if within capacity or better than worst */
				if (candidate_count < capacity)
				{
					candidates[candidate_count].idx = i;
					candidates[candidate_count].dist = pq_dist;
					candidate_count++;
				}
				else
				{
					/* Find worst candidate */
					int			worst_idx = 0;
					float		worst_dist = candidates[0].dist;

					for (j = 1; j < candidate_count; j++)
					{
						if (candidates[j].dist > worst_dist)
						{
							worst_dist = candidates[j].dist;
							worst_idx = j;
						}
					}

					if (pq_dist < worst_dist)
					{
						candidates[worst_idx].idx = i;
						candidates[worst_idx].dist = pq_dist;
					}
				}
			}

			/* Sort candidates and take top rerank_k */
			{
				int			sort_count = Min(candidate_count, state->rerank_k);
				int			s, t;

				/* Simple selection sort for top-k */
				for (s = 0; s < sort_count; s++)
				{
					int			best_idx = s;
					float		best_dist = candidates[s].dist;

					for (t = s + 1; t < candidate_count; t++)
					{
						if (candidates[t].dist < best_dist)
						{
							best_dist = candidates[t].dist;
							best_idx = t;
						}
					}

					if (best_idx != s)
					{
						uint32_t	temp_idx = candidates[s].idx;
						float		temp_dist = candidates[s].dist;

						candidates[s].idx = candidates[best_idx].idx;
						candidates[s].dist = candidates[best_idx].dist;
						candidates[best_idx].idx = temp_idx;
						candidates[best_idx].dist = temp_dist;
					}

					state->coarse_indices[s] = candidates[s].idx;
					state->coarse_distances[s] = candidates[s].dist;
				}

				state->coarse_count = sort_count;
			}

			nfree(candidates);
		}

		nfree(pq_dists);
		return 0;
	}
}

/*
 * Stage 2: Fine rerank with full-precision vectors
 */
static int
pq_scan_fine_rerank(PqScanState * state, const float *full_vectors)
{
	int			i;
	float	   *distances = NULL;

	nalloc(distances, float, state->coarse_count);

	/* Compute distances to full-precision vectors */
	for (i = 0; i < state->coarse_count; i++)
	{
		uint32_t	idx = state->coarse_indices[i];
		const float *vec = full_vectors + idx * state->dim;
		float		dist = 0.0f;
		int			d;

		for (d = 0; d < state->dim; d++)
		{
			float		diff = state->query[d] - vec[d];

			dist += diff * diff;
		}
		distances[i] = sqrtf(dist);
	}

	/*
	 * Use max-heap for efficient top-k selection (O(n log k) complexity).
	 * The heap maintains the k smallest distances. When the heap is full,
	 * we compare new elements with the maximum (root), and if smaller,
	 * replace the root and heapify down.
	 */
	{
		int			heap_size = 0;
		int			max_heap_size = (state->k < state->coarse_count) ? state->k : state->coarse_count;
		int			j;

		/* Build max-heap of size k */
		for (i = 0; i < state->coarse_count; i++)
		{
			if (heap_size < max_heap_size)
			{
				/* Insert into heap */
				int			idx = heap_size++;
				state->fine_indices[idx] = state->coarse_indices[i];
				state->fine_distances[idx] = distances[i];

				/* Bubble up to maintain max-heap property */
				while (idx > 0)
				{
					int			parent = (idx - 1) / 2;

					if (state->fine_distances[parent] >= state->fine_distances[idx])
						break;

					/* Swap with parent */
					{
						float		temp_dist = state->fine_distances[idx];
						uint32_t	temp_idx = state->fine_indices[idx];

						state->fine_distances[idx] = state->fine_distances[parent];
						state->fine_indices[idx] = state->fine_indices[parent];
						state->fine_distances[parent] = temp_dist;
						state->fine_indices[parent] = temp_idx;
					}
					idx = parent;
				}
			}
			else if (distances[i] < state->fine_distances[0])
			{
				/* Replace root (max element) with smaller element */
				state->fine_indices[0] = state->coarse_indices[i];
				state->fine_distances[0] = distances[i];

				/* Heapify down */
				idx = 0;
				while (true)
				{
					int			left = 2 * idx + 1;
					int			right = 2 * idx + 2;
					int			largest = idx;

					if (left < heap_size && state->fine_distances[left] > state->fine_distances[largest])
						largest = left;
					if (right < heap_size && state->fine_distances[right] > state->fine_distances[largest])
						largest = right;

					if (largest == idx)
						break;

					/* Swap with largest child */
					{
						float		temp_dist = state->fine_distances[idx];
						uint32_t	temp_idx = state->fine_indices[idx];

						state->fine_distances[idx] = state->fine_distances[largest];
						state->fine_indices[idx] = state->fine_indices[largest];
						state->fine_distances[largest] = temp_dist;
						state->fine_indices[largest] = temp_idx;
					}
					idx = largest;
				}
			}
		}

		/* Extract elements from max-heap in sorted order (smallest first) */
		/* We'll extract by repeatedly removing the max and placing it at the end */
		state->fine_count = heap_size;
		for (i = heap_size - 1; i > 0; i--)
		{
			/* Swap root with last element */
			{
				float		temp_dist = state->fine_distances[0];
				uint32_t	temp_idx = state->fine_indices[0];

				state->fine_distances[0] = state->fine_distances[i];
				state->fine_indices[0] = state->fine_indices[i];
				state->fine_distances[i] = temp_dist;
				state->fine_indices[i] = temp_idx;
			}

			/* Heapify down on reduced heap */
			idx = 0;
			while (true)
			{
				int			left = 2 * idx + 1;
				int			right = 2 * idx + 2;
				int			largest = idx;

				if (left < i && state->fine_distances[left] > state->fine_distances[largest])
					largest = left;
				if (right < i && state->fine_distances[right] > state->fine_distances[largest])
					largest = right;

				if (largest == idx)
					break;

				/* Swap with largest child */
				{
					float		temp_dist = state->fine_distances[idx];
					uint32_t	temp_idx = state->fine_indices[idx];

					state->fine_distances[idx] = state->fine_distances[largest];
					state->fine_indices[idx] = state->fine_indices[largest];
					state->fine_distances[largest] = temp_dist;
					state->fine_indices[largest] = temp_idx;
				}
				idx = largest;
			}
		}
	}
	pfree(distances);

	return 0;
}

