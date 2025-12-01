/*-------------------------------------------------------------------------
 *
 * ml_dimensionality_reduction.c
 *    Dimensionality reduction algorithms.
 *
 * This module implements t-SNE, UMAP, and autoencoder-based dimensionality
 * reduction for data visualization and feature compression.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_dimensionality_reduction.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "utils/lsyscache.h"
#include "executor/spi.h"
#include "utils/array.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_simd.h"
#include "ml_catalog.h"
#include "ml_utils.h"

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"
#include "lib/stringinfo.h"
#include "libpq/pqformat.h"
#include "utils/jsonb.h"

/*
 * reduce_tsne
 * -----------
 * t-SNE dimensionality reduction.
 *
 * SQL Arguments:
 *   table_name    - Source table with vectors
 *   vector_column - Vector column name
 *   n_components  - Target dimensionality (default: 2)
 *   perplexity    - Perplexity parameter (default: 30.0)
 *   learning_rate - Learning rate (default: 200.0)
 *   iterations    - Number of iterations (default: 1000)
 *
 * Returns:
 *   Table with reduced vectors
 */
PG_FUNCTION_INFO_V1(reduce_tsne);

Datum
reduce_tsne(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *vector_column;
	int			n_components;
	double		perplexity;
	double		learning_rate;
	int			iterations;
	char	   *tbl_str;
	char	   *vec_col_str;
	float	  **vectors;
	int			nvec,
				dim;
	float	  **reduced;
	int			i,
				j,
				k;
	FuncCallContext *funcctx;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;

		table_name = PG_GETARG_TEXT_PP(0);
		vector_column = PG_GETARG_TEXT_PP(1);
		n_components = PG_ARGISNULL(2) ? 2 : PG_GETARG_INT32(2);
		perplexity = PG_ARGISNULL(3) ? 30.0 : PG_GETARG_FLOAT8(3);
		learning_rate = PG_ARGISNULL(4) ? 200.0 : PG_GETARG_FLOAT8(4);
		iterations = PG_ARGISNULL(5) ? 1000 : PG_GETARG_INT32(5);

		if (n_components < 1)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("n_components must be positive")));

		tbl_str = text_to_cstring(table_name);
		vec_col_str = text_to_cstring(vector_column);

		vectors = neurondb_fetch_vectors_from_table(
													tbl_str, vec_col_str, &nvec, &dim);
		if (nvec < 2)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("Need at least 2 vectors")));

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		/* Initialize reduced vectors randomly */
		NDB_DECLARE(float **, reduced);
		NDB_ALLOC(reduced, float *, nvec);
		for (i = 0; i < nvec; i++)
		{
			NDB_ALLOC(reduced[i], float, n_components);
			for (j = 0; j < n_components; j++)
				reduced[i][j] = ((float) rand() / (float) RAND_MAX - 0.5) * 0.0001;
		}

		/* Compute pairwise squared distances */
		{
			double	  **distances_sq = NULL;
			double	  **P = NULL;	/* High-dimensional probabilities */
			double	  **Q = NULL;	/* Low-dimensional probabilities */
			double	   *sum_P = NULL;
			int			iter;
			int			k;

			/* Allocate distance matrix */
			NDB_DECLARE(double **, distances_sq);
			NDB_ALLOC(distances_sq, double *, nvec);
			for (i = 0; i < nvec; i++)
			{
				NDB_ALLOC(distances_sq[i], double, nvec);
			}

			/* Compute squared Euclidean distances */
			for (i = 0; i < nvec; i++)
			{
				for (j = i + 1; j < nvec; j++)
				{
					double		dist_sq = 0.0;

					for (k = 0; k < dim; k++)
					{
						double		diff = (double) vectors[i][k] - (double) vectors[j][k];

						dist_sq += diff * diff;
					}
					distances_sq[i][j] = dist_sq;
					distances_sq[j][i] = dist_sq;
				}
			}

			/* Compute P matrix (high-dimensional probabilities) */

			/*
			 * Use binary search to find sigma for each point to achieve
			 * target perplexity
			 */
			NDB_DECLARE(double **, P);
			NDB_DECLARE(double *, sum_P);
			NDB_ALLOC(P, double *, nvec);
			NDB_ALLOC(sum_P, double, nvec);
			for (i = 0; i < nvec; i++)
			{
				double		sigma = 1.0;
				double		target_perplexity = perplexity;
				double		current_perplexity = 0.0;
				int			binary_search_iter;
				double		sigma_min = 1e-10;
				double		sigma_max = 1000.0;

				NDB_ALLOC(P[i], double, nvec);

				/* Binary search for sigma */
				for (binary_search_iter = 0; binary_search_iter < 50; binary_search_iter++)
				{
					double		sum = 0.0;
					double		log_perplexity = 0.0;

					/* Compute conditional probabilities p_{j|i} */
					for (j = 0; j < nvec; j++)
					{
						if (i == j)
							P[i][j] = 0.0;
						else
						{
							double		dist = distances_sq[i][j] / (2.0 * sigma * sigma);

							P[i][j] = exp(-dist);
							sum += P[i][j];
						}
					}

					/* Normalize and compute perplexity */
					if (sum > 1e-10)
					{
						for (j = 0; j < nvec; j++)
						{
							if (i != j)
							{
								P[i][j] /= sum;
								if (P[i][j] > 1e-10)
									log_perplexity -= P[i][j] * log(P[i][j]);
							}
						}
						current_perplexity = exp(log_perplexity);
					}

					/* Adjust sigma */
					if (current_perplexity < target_perplexity)
						sigma_min = sigma;
					else
						sigma_max = sigma;

					sigma = (sigma_min + sigma_max) / 2.0;

					if (fabs(current_perplexity - target_perplexity) < 0.1 || sigma_max - sigma_min < 1e-10)
						break;
				}

			}

			/* Symmetrize P matrix: P_{ij} = (p_{j|i} + p_{i|j}) / (2*N) */
			for (i = 0; i < nvec; i++)
			{
				for (j = i + 1; j < nvec; j++)
				{
					double		sym_p = (P[i][j] + P[j][i]) / (2.0 * (double) nvec);

					P[i][j] = sym_p;
					P[j][i] = sym_p;
					sum_P[i] += sym_p;
					sum_P[j] += sym_p;
				}
			}

			/* t-SNE optimization loop */
			for (iter = 0; iter < iterations; iter++)
			{
				double	  **gradient = NULL;
				double		momentum = (iter < 250) ? 0.5 : 0.8;
				double	  **Y_prev = NULL;
				int			d;

				/* Allocate gradient and previous Y */
				NDB_DECLARE(double **, gradient);
				NDB_DECLARE(double **, Y_prev);
				NDB_ALLOC(gradient, double *, nvec);
				NDB_ALLOC(Y_prev, double *, nvec);
				for (i = 0; i < nvec; i++)
				{
					NDB_ALLOC(gradient[i], double, n_components);
					NDB_ALLOC(Y_prev[i], double, n_components);
					for (d = 0; d < n_components; d++)
						Y_prev[i][d] = (double) reduced[i][d];
				}

				/* Compute Q matrix (low-dimensional probabilities) */
				NDB_DECLARE(double **, Q);
				NDB_ALLOC(Q, double *, nvec);
				for (i = 0; i < nvec; i++)
					NDB_ALLOC(Q[i], double, nvec);

				for (i = 0; i < nvec; i++)
				{
					double		sum_q = 0.0;

					for (j = 0; j < nvec; j++)
					{
						if (i != j)
						{
							double		dist_sq = 0.0;

							for (d = 0; d < n_components; d++)
							{
								double		diff = (double) reduced[i][d] - (double) reduced[j][d];

								dist_sq += diff * diff;
							}
							Q[i][j] = 1.0 / (1.0 + dist_sq);	/* t-distribution with
																 * df=1 */
							sum_q += Q[i][j];
						}
					}

					/* Normalize Q */
					if (sum_q > 1e-10)
					{
						for (j = 0; j < nvec; j++)
							if (i != j)
								Q[i][j] /= sum_q;
					}
				}

				/* Compute gradient */
				for (i = 0; i < nvec; i++)
				{
					for (j = 0; j < nvec; j++)
					{
						if (i != j)
						{
							double		pq_diff = P[i][j] - Q[i][j];

							for (d = 0; d < n_components; d++)
							{
								double		diff = (double) reduced[i][d] - (double) reduced[j][d];

								gradient[i][d] += 4.0 * pq_diff * diff * Q[i][j];
							}
						}
					}
				}

				/* Update positions with momentum */
				for (i = 0; i < nvec; i++)
				{
					for (d = 0; d < n_components; d++)
					{
						double		update = momentum * (Y_prev[i][d] - (double) reduced[i][d]) + learning_rate * gradient[i][d];

						reduced[i][d] = (float) ((double) reduced[i][d] + update);
					}
				}

				/* Cleanup gradient and Q for this iteration */
				for (i = 0; i < nvec; i++)
				{
					NDB_FREE(gradient[i]);
					NDB_FREE(Y_prev[i]);
					NDB_FREE(Q[i]);
				}
				NDB_FREE(gradient);
				NDB_FREE(Y_prev);
				NDB_FREE(Q);
			}

			/* Cleanup */
			for (i = 0; i < nvec; i++)
			{
				NDB_FREE(distances_sq[i]);
				NDB_FREE(P[i]);
			}
			NDB_FREE(distances_sq);
			NDB_FREE(P);
			NDB_FREE(sum_P);
		}
		funcctx->user_fctx = reduced;
		funcctx->max_calls = nvec;
		MemoryContextSwitchTo(oldcontext);

		for (i = 0; i < nvec; i++)
			NDB_FREE(vectors[i]);
		NDB_FREE(vectors);
		NDB_FREE(tbl_str);
		NDB_FREE(vec_col_str);
	}

	funcctx = SRF_PERCALL_SETUP();
	if (funcctx->call_cntr < funcctx->max_calls)
	{
		float	  **reduced = (float **) funcctx->user_fctx;
		int			idx = funcctx->call_cntr;
		ArrayType  *result;
		Datum	   *result_datums;

		NDB_DECLARE(Datum *, result_datums);
		NDB_ALLOC(result_datums, Datum, n_components);
		for (j = 0; j < n_components; j++)
			result_datums[j] = Float4GetDatum(reduced[idx][j]);

		result = construct_array(result_datums,
								 n_components,
								 FLOAT4OID,
								 sizeof(float4),
								 FLOAT4PASSBYVAL,
								 'i');

		NDB_FREE(result_datums);
		SRF_RETURN_NEXT(funcctx, PointerGetDatum(result));
	}

	SRF_RETURN_DONE(funcctx);
}

/*
 * reduce_umap
 * ----------
 * UMAP dimensionality reduction.
 *
 * SQL Arguments:
 *   table_name    - Source table with vectors
 *   vector_column - Vector column name
 *   n_components  - Target dimensionality (default: 2)
 *   n_neighbors   - Number of neighbors (default: 15)
 *   min_dist      - Minimum distance (default: 0.1)
 *   learning_rate - Learning rate (default: 1.0)
 *   iterations    - Number of iterations (default: 500)
 *
 * Returns:
 *   Table with reduced vectors
 */
PG_FUNCTION_INFO_V1(reduce_umap);

Datum
reduce_umap(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *vector_column;
	int			n_components;
	int			n_neighbors;
	double		min_dist;
	double		learning_rate;
	int			iterations;
	char	   *tbl_str;
	char	   *vec_col_str;
	float	  **vectors;
	int			nvec,
				dim;
	float	  **reduced;
	int			i,
				j,
				k;
	FuncCallContext *funcctx;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;

		table_name = PG_GETARG_TEXT_PP(0);
		vector_column = PG_GETARG_TEXT_PP(1);
		n_components = PG_ARGISNULL(2) ? 2 : PG_GETARG_INT32(2);
		n_neighbors = PG_ARGISNULL(3) ? 15 : PG_GETARG_INT32(3);
		min_dist = PG_ARGISNULL(4) ? 0.1 : PG_GETARG_FLOAT8(4);
		learning_rate = PG_ARGISNULL(5) ? 1.0 : PG_GETARG_FLOAT8(5);
		iterations = PG_ARGISNULL(6) ? 500 : PG_GETARG_INT32(6);

		if (n_components < 1)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("n_components must be positive")));

		tbl_str = text_to_cstring(table_name);
		vec_col_str = text_to_cstring(vector_column);

		vectors = neurondb_fetch_vectors_from_table(
													tbl_str, vec_col_str, &nvec, &dim);
		if (nvec < 2)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("Need at least 2 vectors")));

		if (n_neighbors < 2 || n_neighbors >= nvec)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("n_neighbors must be between 2 and number of vectors")));

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		/* Initialize reduced vectors randomly */
		NDB_DECLARE(float **, reduced);
		NDB_ALLOC(reduced, float *, nvec);
		for (i = 0; i < nvec; i++)
		{
			NDB_ALLOC(reduced[i], float, n_components);
			for (j = 0; j < n_components; j++)
				reduced[i][j] = ((float) rand() / (float) RAND_MAX - 0.5) * 0.0001;
		}

		/* UMAP algorithm implementation */
		{
			double	  **distances = NULL;
			int		  **neighbors = NULL;
			double	  **high_prob = NULL;	/* High-dimensional probabilities */
			int			iter;
			int			d;

			/* Compute pairwise distances */
			NDB_DECLARE(double **, distances);
			NDB_ALLOC(distances, double *, nvec);
			for (i = 0; i < nvec; i++)
			{
				NDB_ALLOC(distances[i], double, nvec);
				for (j = 0; j < nvec; j++)
				{
					if (i == j)
						distances[i][j] = 0.0;
					else
					{
						double		dist_sq = 0.0;

						for (k = 0; k < dim; k++)
						{
							double		diff = (double) vectors[i][k] - (double) vectors[j][k];

							dist_sq += diff * diff;
						}
						distances[i][j] = sqrt(dist_sq);
					}
				}
			}

			/* Find k-nearest neighbors for each point */
			NDB_DECLARE(int **, neighbors);
			NDB_ALLOC(neighbors, int *, nvec);
			for (i = 0; i < nvec; i++)
			{
				int		   *neighbor_indices = (int *) palloc(sizeof(int) * n_neighbors);
				double	   *neighbor_dists = (double *) palloc(sizeof(double) * n_neighbors);
				int			found = 0;

				/* Initialize with first n_neighbors */
				for (j = 0; j < nvec && found < n_neighbors; j++)
				{
					if (i != j)
					{
						neighbor_indices[found] = j;
						neighbor_dists[found] = distances[i][j];
						found++;
					}
				}

				/* Sort and keep only k nearest */
				for (j = 0; j < found - 1; j++)
				{
					for (k = j + 1; k < found; k++)
					{
						if (neighbor_dists[j] > neighbor_dists[k])
						{
							double		tmp_dist = neighbor_dists[j];
							int			tmp_idx = neighbor_indices[j];

							neighbor_dists[j] = neighbor_dists[k];
							neighbor_indices[j] = neighbor_indices[k];
							neighbor_dists[k] = tmp_dist;
							neighbor_indices[k] = tmp_idx;
						}
					}
				}

				/* Check remaining points and update if closer */
				for (j = n_neighbors; j < nvec; j++)
				{
					if (i != j)
					{
						double		dist = distances[i][j];

						if (dist < neighbor_dists[n_neighbors - 1])
						{
							/* Insert in sorted order */
							int			pos = n_neighbors - 1;

							while (pos > 0 && dist < neighbor_dists[pos - 1])
								pos--;

							/* Shift and insert */
							for (k = n_neighbors - 1; k > pos; k--)
							{
								neighbor_dists[k] = neighbor_dists[k - 1];
								neighbor_indices[k] = neighbor_indices[k - 1];
							}
							neighbor_dists[pos] = dist;
							neighbor_indices[pos] = j;
						}
					}
				}

				neighbors[i] = neighbor_indices;
				NDB_FREE(neighbor_dists);
			}

			/* Compute high-dimensional probabilities (fuzzy simplicial set) */
			high_prob = (double **) palloc(sizeof(double *) * nvec);
			for (i = 0; i < nvec; i++)
			{
				double		sigma = 0.0;
				double		rho = distances[i][neighbors[i][0]];	/* Distance to nearest
																	 * neighbor */

				/*
				 * Compute sigma using binary search to achieve target
				 * perplexity
				 */
				{
					double		sigma_min = 0.0;
					double		sigma_max = 1000.0;
					int			binary_iter;

					for (binary_iter = 0; binary_iter < 50; binary_iter++)
					{
						double		sum = 0.0;

						sigma = (sigma_min + sigma_max) / 2.0;

						for (j = 0; j < n_neighbors; j++)
						{
							int			neighbor_idx = neighbors[i][j];
							double		dist = distances[i][neighbor_idx] - rho;

							if (dist < 0.0)
								dist = 0.0;
							sum += exp(-dist / sigma);
						}

						if (sum < log((double) n_neighbors))
							sigma_min = sigma;
						else
							sigma_max = sigma;

						if (sigma_max - sigma_min < 1e-10)
							break;
					}
				}

				high_prob[i] = (double *) palloc0(sizeof(double) * nvec);
				for (j = 0; j < n_neighbors; j++)
				{
					int			neighbor_idx = neighbors[i][j];
					double		dist = distances[i][neighbor_idx] - rho;

					if (dist < 0.0)
						dist = 0.0;
					high_prob[i][neighbor_idx] = exp(-dist / sigma);
				}

				/* Symmetrize */
				for (j = 0; j < n_neighbors; j++)
				{
					int			neighbor_idx = neighbors[i][j];
					double		prob = high_prob[i][neighbor_idx];

					if (high_prob[neighbor_idx][i] < prob)
						high_prob[neighbor_idx][i] = prob;
					high_prob[i][neighbor_idx] = high_prob[neighbor_idx][i];
				}
			}

			/* UMAP optimization loop */
			for (iter = 0; iter < iterations; iter++)
			{
				double	  **gradient = NULL;
				int			d;

				gradient = (double **) palloc(sizeof(double *) * nvec);
				for (i = 0; i < nvec; i++)
					gradient[i] = (double *) palloc0(sizeof(double) * n_components);

				/* Compute gradient */
				for (i = 0; i < nvec; i++)
				{
					for (j = 0; j < nvec; j++)
					{
						if (i != j)
						{
							/* Compute low-dimensional distance */
							double		low_dist_sq = 0.0;

							for (d = 0; d < n_components; d++)
							{
								double		diff = (double) reduced[i][d] - (double) reduced[j][d];

								low_dist_sq += diff * diff;
							}
							double		low_dist = sqrt(low_dist_sq + 1e-10);

							/* Compute low-dimensional probability */
							double		a = pow(1.0 + low_dist_sq / (min_dist * min_dist), -1.0);
							double		b = high_prob[i][j];

							/* Gradient */
							double		grad_coeff = -2.0 * a * b / (low_dist + 1e-10);

							for (d = 0; d < n_components; d++)
							{
								double		diff = (double) reduced[i][d] - (double) reduced[j][d];

								gradient[i][d] += grad_coeff * diff;
							}
						}
					}
				}

				/* Update positions */
				for (i = 0; i < nvec; i++)
				{
					for (d = 0; d < n_components; d++)
					{
						reduced[i][d] = (float) ((double) reduced[i][d] + learning_rate * gradient[i][d]);
					}
				}

				/* Cleanup gradient */
				for (i = 0; i < nvec; i++)
					NDB_FREE(gradient[i]);
				NDB_FREE(gradient);
			}

			/* Cleanup */
			for (i = 0; i < nvec; i++)
			{
				NDB_FREE(distances[i]);
				NDB_FREE(neighbors[i]);
				NDB_FREE(high_prob[i]);
			}
			NDB_FREE(distances);
			NDB_FREE(neighbors);
			NDB_FREE(high_prob);
		}

		funcctx->user_fctx = reduced;
		funcctx->max_calls = nvec;
		MemoryContextSwitchTo(oldcontext);

		for (i = 0; i < nvec; i++)
			NDB_FREE(vectors[i]);
		NDB_FREE(vectors);
		NDB_FREE(tbl_str);
		NDB_FREE(vec_col_str);
	}

	funcctx = SRF_PERCALL_SETUP();
	if (funcctx->call_cntr < funcctx->max_calls)
	{
		float	  **reduced = (float **) funcctx->user_fctx;
		int			idx = funcctx->call_cntr;
		ArrayType  *result;
		Datum	   *result_datums;

		NDB_DECLARE(Datum *, result_datums);
		NDB_ALLOC(result_datums, Datum, n_components);
		for (j = 0; j < n_components; j++)
			result_datums[j] = Float4GetDatum(reduced[idx][j]);

		result = construct_array(result_datums,
								 n_components,
								 FLOAT4OID,
								 sizeof(float4),
								 FLOAT4PASSBYVAL,
								 'i');

		NDB_FREE(result_datums);
		SRF_RETURN_NEXT(funcctx, PointerGetDatum(result));
	}

	SRF_RETURN_DONE(funcctx);
}

/*
 * Autoencoder structures (simplified neural network for encoder-decoder)
 */
typedef struct AutoencoderLayer
{
	int			n_inputs;
	int			n_outputs;
	float	  **weights;		/* [n_outputs][n_inputs+1] (includes bias) */
	float	   *activations;
	float	   *deltas;
}			AutoencoderLayer;

typedef struct Autoencoder
{
	int			n_layers;
	int			n_inputs;
	int			n_outputs;		/* Same as inputs for reconstruction */
	int			bottleneck_dim;
	AutoencoderLayer *layers;
	char	   *activation_func;
	float		learning_rate;
}			Autoencoder;

/* Activation functions */
static float
ae_activation_relu(float x)
{
	return (x > 0.0f) ? x : 0.0f;
}

static float
ae_activation_sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

static float
ae_activation_tanh(float x)
{
	return tanhf(x);
}

static float
ae_activation_derivative_relu(float x)
{
	return (x > 0.0f) ? 1.0f : 0.0f;
}

static float
ae_activation_derivative_sigmoid(float x)
{
	float		s = ae_activation_sigmoid(x);

	return s * (1.0f - s);
}

static float
ae_activation_derivative_tanh(float x)
{
	float		t = tanhf(x);

	return 1.0f - t * t;
}

/* Forward pass */
static void
autoencoder_forward(Autoencoder * ae, float *input, float *output)
{
	int			i,
				j,
				k;
	float	   *prev_activations = input;

	for (i = 0; i < ae->n_layers; i++)
	{
		AutoencoderLayer *layer = &ae->layers[i];
		float	   *curr_activations = layer->activations;

		for (j = 0; j < layer->n_outputs; j++)
		{
			float		sum = layer->weights[j][layer->n_inputs];	/* bias */

			for (k = 0; k < layer->n_inputs; k++)
				sum += layer->weights[j][k] * prev_activations[k];

			/* Apply activation */
			if (ae->activation_func != NULL)
			{
				if (strcmp(ae->activation_func, "relu") == 0)
					curr_activations[j] = ae_activation_relu(sum);
				else if (strcmp(ae->activation_func, "sigmoid") == 0)
					curr_activations[j] = ae_activation_sigmoid(sum);
				else if (strcmp(ae->activation_func, "tanh") == 0)
					curr_activations[j] = ae_activation_tanh(sum);
				else
					curr_activations[j] = sum;	/* linear */
			}
			else
				curr_activations[j] = sum;
		}

		prev_activations = curr_activations;
	}

	/* Copy final layer to output */
	memcpy(output, ae->layers[ae->n_layers - 1].activations, ae->n_outputs * sizeof(float));
}

/* Backward pass (backpropagation) */
static void
autoencoder_backward(Autoencoder * ae, float *input, float *target, float *predicted)
{
	int			i,
				j,
				k;
	AutoencoderLayer *output_layer = &ae->layers[ae->n_layers - 1];

	/* Compute output layer deltas (reconstruction error) */
	for (j = 0; j < output_layer->n_outputs; j++)
	{
		float		error = target[j] - predicted[j];
		float		activation = output_layer->activations[j];
		float		derivative;

		if (strcmp(ae->activation_func, "relu") == 0)
			derivative = ae_activation_derivative_relu(activation);
		else if (strcmp(ae->activation_func, "sigmoid") == 0)
			derivative = ae_activation_derivative_sigmoid(activation);
		else if (strcmp(ae->activation_func, "tanh") == 0)
			derivative = ae_activation_derivative_tanh(activation);
		else
			derivative = 1.0f;

		output_layer->deltas[j] = error * derivative;
	}

	/* Backpropagate through hidden layers */
	for (i = ae->n_layers - 2; i >= 0; i--)
	{
		AutoencoderLayer *curr_layer = &ae->layers[i];
		AutoencoderLayer *next_layer = &ae->layers[i + 1];

		for (j = 0; j < curr_layer->n_outputs; j++)
		{
			float		sum = 0.0f;
			float		activation;
			float		derivative;

			for (k = 0; k < next_layer->n_outputs; k++)
				sum += next_layer->weights[k][j] * next_layer->deltas[k];

			activation = curr_layer->activations[j];

			if (strcmp(ae->activation_func, "relu") == 0)
				derivative = ae_activation_derivative_relu(activation);
			else if (strcmp(ae->activation_func, "sigmoid") == 0)
				derivative = ae_activation_derivative_sigmoid(activation);
			else if (strcmp(ae->activation_func, "tanh") == 0)
				derivative = ae_activation_derivative_tanh(activation);
			else
				derivative = 1.0f;

			curr_layer->deltas[j] = sum * derivative;
		}
	}
}

/* Update weights */
static void
autoencoder_update_weights(Autoencoder * ae, float *input)
{
	int			i,
				j,
				k;
	float	   *prev_activations = input;

	for (i = 0; i < ae->n_layers; i++)
	{
		AutoencoderLayer *layer = &ae->layers[i];

		for (j = 0; j < layer->n_outputs; j++)
		{
			/* Update bias */
			layer->weights[j][layer->n_inputs] +=
				ae->learning_rate * layer->deltas[j];

			/* Update input weights */
			for (k = 0; k < layer->n_inputs; k++)
				layer->weights[j][k] += ae->learning_rate
					* layer->deltas[j]
					* prev_activations[k];
		}

		prev_activations = layer->activations;
	}
}

/* Initialize autoencoder */
static Autoencoder *
autoencoder_init(int n_inputs,
				 int bottleneck_dim,
				 int *encoder_layers,
				 int n_encoder_layers,
				 int *decoder_layers,
				 int n_decoder_layers,
				 const char *activation,
				 float learning_rate)
{
	int			i,
				j,
				k;
	int			prev_size;
	int			total_layers = n_encoder_layers + n_decoder_layers;
	Autoencoder *ae = (Autoencoder *) palloc0(sizeof(Autoencoder));

	ae->n_inputs = n_inputs;
	ae->n_outputs = n_inputs;	/* Reconstruction target */
	ae->bottleneck_dim = bottleneck_dim;
	ae->n_layers = total_layers;
	ae->activation_func = pstrdup(activation);
	ae->learning_rate = learning_rate;

	ae->layers = (AutoencoderLayer *) palloc(ae->n_layers * sizeof(AutoencoderLayer));

	/* Initialize encoder layers */
	prev_size = n_inputs;
	for (i = 0; i < n_encoder_layers; i++)
	{
		AutoencoderLayer *layer = &ae->layers[i];

		layer->n_inputs = prev_size;
		layer->n_outputs = encoder_layers[i];
		layer->weights = (float **) palloc(layer->n_outputs * sizeof(float *));
		layer->activations = (float *) palloc(layer->n_outputs * sizeof(float));
		layer->deltas = (float *) palloc(layer->n_outputs * sizeof(float));

		for (j = 0; j < layer->n_outputs; j++)
		{
			layer->weights[j] = (float *) palloc((layer->n_inputs + 1) * sizeof(float));
			/* Initialize weights randomly */
			for (k = 0; k <= layer->n_inputs; k++)
				layer->weights[j][k] = ((float) rand() / (float) RAND_MAX) * 0.1f - 0.05f;
		}

		prev_size = layer->n_outputs;
	}

	/* Ensure bottleneck dimension matches last encoder layer */
	if (prev_size != bottleneck_dim)
	{
		/* Add bottleneck layer if needed */
		AutoencoderLayer *bottleneck = &ae->layers[n_encoder_layers];

		bottleneck->n_inputs = prev_size;
		bottleneck->n_outputs = bottleneck_dim;
		bottleneck->weights = (float **) palloc(bottleneck->n_outputs * sizeof(float *));
		bottleneck->activations = (float *) palloc(bottleneck->n_outputs * sizeof(float));
		bottleneck->deltas = (float *) palloc(bottleneck->n_outputs * sizeof(float));

		for (j = 0; j < bottleneck->n_outputs; j++)
		{
			bottleneck->weights[j] = (float *) palloc((bottleneck->n_inputs + 1) * sizeof(float));
			for (k = 0; k <= bottleneck->n_inputs; k++)
				bottleneck->weights[j][k] = ((float) rand() / (float) RAND_MAX) * 0.1f - 0.05f;
		}

		prev_size = bottleneck_dim;
		i = n_encoder_layers + 1;
	}
	else
	{
		i = n_encoder_layers;
	}

	/* Initialize decoder layers */
	for (; i < ae->n_layers; i++)
	{
		int			decoder_idx = i - n_encoder_layers - (prev_size == bottleneck_dim ? 1 : 0);
		AutoencoderLayer *layer = &ae->layers[i];

		layer->n_inputs = prev_size;
		layer->n_outputs = decoder_idx < n_decoder_layers ? decoder_layers[decoder_idx] : n_inputs;
		layer->weights = (float **) palloc(layer->n_outputs * sizeof(float *));
		layer->activations = (float *) palloc(layer->n_outputs * sizeof(float));
		layer->deltas = (float *) palloc(layer->n_outputs * sizeof(float));

		for (j = 0; j < layer->n_outputs; j++)
		{
			layer->weights[j] = (float *) palloc((layer->n_inputs + 1) * sizeof(float));
			for (k = 0; k <= layer->n_inputs; k++)
				layer->weights[j][k] = ((float) rand() / (float) RAND_MAX) * 0.1f - 0.05f;
		}

		prev_size = layer->n_outputs;
	}

	return ae;
}

/* Serialize autoencoder */
static bytea *
autoencoder_serialize(const Autoencoder * ae)
{
	StringInfoData buf;
	int			i,
				j,
				k;
	int			activation_len;

	if (ae == NULL)
		return NULL;

	initStringInfo(&buf);

	/* Write header */
	pq_sendint32(&buf, ae->n_layers);
	pq_sendint32(&buf, ae->n_inputs);
	pq_sendint32(&buf, ae->n_outputs);
	pq_sendint32(&buf, ae->bottleneck_dim);
	pq_sendfloat8(&buf, ae->learning_rate);

	/* Write activation function */
	activation_len = ae->activation_func ? strlen(ae->activation_func) : 0;
	pq_sendint32(&buf, activation_len);
	if (activation_len > 0)
		pq_sendbytes(&buf, ae->activation_func, activation_len);

	/* Write layers */
	for (i = 0; i < ae->n_layers; i++)
	{
		AutoencoderLayer *layer = &ae->layers[i];

		pq_sendint32(&buf, layer->n_inputs);
		pq_sendint32(&buf, layer->n_outputs);

		for (j = 0; j < layer->n_outputs; j++)
		{
			for (k = 0; k <= layer->n_inputs; k++)
				pq_sendfloat8(&buf, layer->weights[j][k]);
		}
	}

	return (bytea *) pq_endtypsend(&buf);
}

/*
 * train_autoencoder
 * -----------------
 * Train autoencoder for dimensionality reduction.
 *
 * Arguments:
 *   table_name - Source table with vectors
 *   vector_column - Vector column name
 *   bottleneck_dim - Dimension of compressed representation
 *   encoder_layers - Array of encoder layer sizes (optional)
 *   decoder_layers - Array of decoder layer sizes (optional)
 *   activation - Activation function ('relu', 'sigmoid', 'tanh')
 *   learning_rate - Learning rate (default: 0.001)
 *   epochs - Number of training epochs (default: 100)
 */
PG_FUNCTION_INFO_V1(train_autoencoder);

Datum
train_autoencoder(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_PP(0);
	text	   *vector_column = PG_GETARG_TEXT_PP(1);
	int32		bottleneck_dim = PG_GETARG_INT32(2);
	ArrayType  *encoder_layers_array = PG_ARGISNULL(3) ? NULL : PG_GETARG_ARRAYTYPE_P(3);
	ArrayType  *decoder_layers_array = PG_ARGISNULL(4) ? NULL : PG_GETARG_ARRAYTYPE_P(4);
	text	   *activation_text = PG_ARGISNULL(5) ? NULL : PG_GETARG_TEXT_PP(5);
	float8		learning_rate = PG_ARGISNULL(6) ? 0.001 : PG_GETARG_FLOAT8(6);
	int32		epochs = PG_ARGISNULL(7) ? 100 : PG_GETARG_INT32(7);

	char	   *table_name_str;
	char	   *vector_col_str;
	char	   *activation;
	float	  **vectors = NULL;
	int			nvec,
				dim;
	Autoencoder *ae = NULL;
	int		   *encoder_layers = NULL;
	int		   *decoder_layers = NULL;
	int			n_encoder_layers = 0;
	int			n_decoder_layers = 0;
	int			epoch,
				sample;
	float		loss;
	int			i,
				j;
	bytea	   *serialized = NULL;
	MLCatalogModelSpec spec;
	int32		model_id = 0;
	StringInfoData metricsbuf;
	StringInfoData paramsbuf;
	Jsonb	   *params_jsonb = NULL;
	Jsonb	   *metrics_jsonb = NULL;

	/* Validate inputs */
	if (bottleneck_dim <= 0 || bottleneck_dim > 10000)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("bottleneck_dim must be between 1 and 10000")));

	if (epochs <= 0 || epochs > 100000)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("epochs must be between 1 and 100000")));

	table_name_str = text_to_cstring(table_name);
	vector_col_str = text_to_cstring(vector_column);
	activation = activation_text ? text_to_cstring(activation_text) : pstrdup("relu");

	/* Validate activation function */
	if (strcmp(activation, "relu") != 0
		&& strcmp(activation, "sigmoid") != 0
		&& strcmp(activation, "tanh") != 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("activation must be 'relu', 'sigmoid', or 'tanh'")));

	/* Extract encoder/decoder layers */
	if (encoder_layers_array != NULL)
	{
		n_encoder_layers = ArrayGetNItems(ARR_NDIM(encoder_layers_array), ARR_DIMS(encoder_layers_array));
		encoder_layers = (int *) palloc(n_encoder_layers * sizeof(int));
		for (i = 0; i < n_encoder_layers; i++)
		{
			bool		isnull;
			Datum		elem = array_ref(encoder_layers_array, 1, &i, -1, -1, false, 'i', &isnull);

			if (isnull)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("encoder_layers cannot contain NULL")));
			encoder_layers[i] = DatumGetInt32(elem);
			if (encoder_layers[i] <= 0)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("encoder layer sizes must be positive")));
		}
	}
	else
	{
		/* Default: single encoder layer to bottleneck */
		n_encoder_layers = 1;
		encoder_layers = (int *) palloc(sizeof(int));
		encoder_layers[0] = bottleneck_dim;
	}

	if (decoder_layers_array != NULL)
	{
		n_decoder_layers = ArrayGetNItems(ARR_NDIM(decoder_layers_array), ARR_DIMS(decoder_layers_array));
		decoder_layers = (int *) palloc(n_decoder_layers * sizeof(int));
		for (i = 0; i < n_decoder_layers; i++)
		{
			bool		isnull;
			Datum		elem = array_ref(decoder_layers_array, 1, &i, -1, -1, false, 'i', &isnull);

			if (isnull)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("decoder_layers cannot contain NULL")));
			decoder_layers[i] = DatumGetInt32(elem);
			if (decoder_layers[i] <= 0)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("decoder layer sizes must be positive")));
		}
	}
	else
	{
		/* Default: single decoder layer from bottleneck */
		n_decoder_layers = 0;	/* Will be handled in init */
		decoder_layers = NULL;
	}

	/* Fetch vectors from table */
	vectors = neurondb_fetch_vectors_from_table(table_name_str, vector_col_str, &nvec, &dim);
	if (nvec < 1)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Need at least 1 vector for training")));

	if (bottleneck_dim >= dim)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("bottleneck_dim (%d) must be less than input dimension (%d)",
						bottleneck_dim, dim)));

	/* Initialize autoencoder */
	ae = autoencoder_init(dim,
						  bottleneck_dim,
						  encoder_layers,
						  n_encoder_layers,
						  decoder_layers,
						  n_decoder_layers,
						  activation,
						  (float) learning_rate);

	/* Training loop */
	for (epoch = 0; epoch < epochs; epoch++)
	{
		loss = 0.0f;

		for (sample = 0; sample < nvec; sample++)
		{
			float	   *reconstructed = (float *) palloc(dim * sizeof(float));
			float	   *input = vectors[sample];

			/* Forward pass */
			autoencoder_forward(ae, input, reconstructed);

			/* Compute reconstruction loss (MSE) */
			for (j = 0; j < dim; j++)
			{
				float		error = input[j] - reconstructed[j];

				loss += error * error;
			}

			/* Backward pass (use input as target) */
			autoencoder_backward(ae, input, input, reconstructed);

			/* Update weights */
			autoencoder_update_weights(ae, input);

			NDB_FREE(reconstructed);
		}

		loss /= (nvec * dim);

		if (epoch % 10 == 0 || epoch == epochs - 1)
		{
			elog(DEBUG1,
				 "Autoencoder epoch %d: reconstruction loss = %.6f",
				 epoch, loss);
		}
	}

	/* Serialize model */
	serialized = autoencoder_serialize(ae);

	/* Build parameters JSON */
	initStringInfo(&paramsbuf);
	appendStringInfo(&paramsbuf,
					 "{\"bottleneck_dim\":%d,"
					 "\"input_dim\":%d,"
					 "\"activation\":\"%s\","
					 "\"learning_rate\":%.6f,"
					 "\"epochs\":%d}",
					 bottleneck_dim,
					 dim,
					 activation,
					 learning_rate,
					 epochs);
	params_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetTextDatum(paramsbuf.data)));
	NDB_FREE(paramsbuf.data);

	/* Build metrics JSON */
	initStringInfo(&metricsbuf);
	appendStringInfo(&metricsbuf,
					 "{\"final_loss\":%.6f,"
					 "\"n_samples\":%d,"
					 "\"compression_ratio\":%.2f}",
					 loss,
					 nvec,
					 (double) dim / (double) bottleneck_dim);
	metrics_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetTextDatum(metricsbuf.data)));
	NDB_FREE(metricsbuf.data);

	/* Register model in catalog */
	memset(&spec, 0, sizeof(MLCatalogModelSpec));
	spec.algorithm = "autoencoder";
	spec.training_table = table_name_str;
	spec.model_data = serialized;
	spec.parameters = params_jsonb;
	spec.metrics = metrics_jsonb;

	model_id = ml_catalog_register_model(&spec);

	/* Cleanup */
	for (i = 0; i < nvec; i++)
		NDB_FREE(vectors[i]);
	NDB_FREE(vectors);

	if (ae != NULL)
	{
		for (i = 0; i < ae->n_layers; i++)
		{
			AutoencoderLayer *layer = &ae->layers[i];

			if (layer->weights != NULL)
			{
				for (j = 0; j < layer->n_outputs; j++)
					NDB_FREE(layer->weights[j]);
				NDB_FREE(layer->weights);
			}
			NDB_FREE(layer->activations);
			NDB_FREE(layer->deltas);
		}
		NDB_FREE(ae->layers);
		NDB_FREE(ae->activation_func);
		NDB_FREE(ae);
	}

	NDB_FREE(table_name_str);
	NDB_FREE(vector_col_str);
	NDB_FREE(activation);
	if (encoder_layers)
		NDB_FREE(encoder_layers);
	if (decoder_layers)
		NDB_FREE(decoder_layers);

	PG_RETURN_INT32(model_id);
}
