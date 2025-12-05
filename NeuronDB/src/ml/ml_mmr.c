/*-------------------------------------------------------------------------
 *
 * ml_mmr.c
 *    Maximal marginal relevance for diverse reranking.
 *
 * This module implements MMR to balance relevance and diversity in search
 * results by iteratively selecting relevant but dissimilar documents.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_mmr.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "catalog/pg_type.h"
#include "utils/lsyscache.h"

#include "neurondb.h"

#include <math.h>
#include <float.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/*
 * Compute cosine similarity between two vectors
 */
static inline double
cosine_similarity(const float *a, const float *b, int dim)
{
	double		dot = 0.0;
	double		norm_a = 0.0;
	double		norm_b = 0.0;
	int			i;

	for (i = 0; i < dim; i++)
	{
		dot += a[i] * b[i];
		norm_a += a[i] * a[i];
		norm_b += b[i] * b[i];
	}

	if (norm_a < 1e-10 || norm_b < 1e-10)
		return 0.0;

	return dot / (sqrt(norm_a) * sqrt(norm_b));
}

/*
 * mmr_rerank
 *    Rerank search results using Maximal Marginal Relevance algorithm.
 *
 * This function implements the MMR algorithm to balance relevance and
 * diversity in search results. The algorithm iteratively selects documents
 * that maximize a combination of relevance to the query and dissimilarity
 * to previously selected documents. The MMR score for each candidate
 * document is computed as lambda times its similarity to the query minus
 * one minus lambda times its maximum similarity to any document already
 * in the result set. Lambda controls the trade-off between relevance and
 * diversity, with values near 1.0 favoring relevance and values near 0.0
 * favoring diversity. The algorithm begins by selecting the most relevant
 * document, then repeatedly selects the document with the highest MMR score
 * until the desired number of results is obtained. This approach prevents
 * redundancy in results while maintaining high relevance, making it
 * particularly useful for information retrieval scenarios where diverse
 * perspectives on a topic are valuable.
 */
PG_FUNCTION_INFO_V1(mmr_rerank);

Datum
mmr_rerank(PG_FUNCTION_ARGS)
{
	ArrayType  *query_array;
	ArrayType  *candidates_array;
	float		lambda;
	int			top_k;
	float	   *query;
	int			query_dim;
	NDB_DECLARE(float **, candidates);
	int			n_candidates;
	int			dim;
	NDB_DECLARE(bool *, selected);
	NDB_DECLARE(int *, result_indices);
	NDB_DECLARE(double *, query_scores);
	ArrayType  *result;
	NDB_DECLARE(Datum *, result_datums);
	int			selected_count;
	int			i,
				j;
	int16		typlen;
	bool		typbyval;
	char		typalign;

	query_array = PG_GETARG_ARRAYTYPE_P(0);
	candidates_array = PG_GETARG_ARRAYTYPE_P(1);
	lambda = PG_GETARG_FLOAT4(2);
	top_k = PG_GETARG_INT32(3);

	if (lambda < 0.0 || lambda > 1.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("lambda must be between 0.0 and 1.0")));

	/* Extract query vector */
	query_dim = ARR_DIMS(query_array)[0];
	query = (float *) ARR_DATA_PTR(query_array);

	/* Extract candidate vectors */
	if (ARR_NDIM(candidates_array) != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("candidates must be 2-dimensional "
						"array")));

	n_candidates = ARR_DIMS(candidates_array)[0];
	dim = ARR_DIMS(candidates_array)[1];

	if (dim != query_dim)
	{
		elog(DEBUG1,
			 "Query dimension %d != candidate dimension %d",
			 query_dim,
			 dim);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Query dimension %d != candidate dimension %d",
						query_dim,
						dim)));
	}

	if (top_k < 1)
		top_k = n_candidates;
	else if (top_k > n_candidates)
		top_k = n_candidates;

	/* Parse candidate vectors */
	NDB_ALLOC(candidates, float *, n_candidates);
	{
		float	   *data = (float *) ARR_DATA_PTR(candidates_array);

		for (i = 0; i < n_candidates; i++)
			candidates[i] = &data[i * dim];
	}

	/* Compute query-candidate similarities */
	NDB_ALLOC(query_scores, double, n_candidates);
	for (i = 0; i < n_candidates; i++)
		query_scores[i] = cosine_similarity(query, candidates[i], dim);

	/* MMR algorithm */
	NDB_ALLOC(selected, bool, n_candidates);
	NDB_ALLOC(result_indices, int, top_k);
	selected_count = 0;

	while (selected_count < top_k)
	{
		double		max_mmr = -1e30;
		int			best_idx = -1;

		/* Find candidate with maximum MMR score */
		for (i = 0; i < n_candidates; i++)
		{
			double		mmr_score;
			double		max_sim_to_selected;

			if (selected[i])
				continue;

			/* Compute max similarity to already-selected documents */
			max_sim_to_selected = 0.0;
			for (j = 0; j < selected_count; j++)
			{
				int			selected_idx = result_indices[j];
				double		sim = cosine_similarity(candidates[i],
													candidates[selected_idx],
													dim);

				if (sim > max_sim_to_selected)
					max_sim_to_selected = sim;
			}

			/* MMR = λ × Sim(D_i, Q) - (1-λ) × max Sim(D_i, S) */
			mmr_score = lambda * query_scores[i]
				- (1.0 - lambda) * max_sim_to_selected;

			if (mmr_score > max_mmr)
			{
				max_mmr = mmr_score;
				best_idx = i;
			}
		}

		if (best_idx == -1)
			break;

		result_indices[selected_count] = best_idx;
		selected[best_idx] = true;
		selected_count++;
	}

	/* Build result array (1-based indices) */
	NDB_ALLOC(result_datums, Datum, selected_count);
	for (i = 0; i < selected_count; i++)
		result_datums[i] = Int32GetDatum(result_indices[i] + 1);

	get_typlenbyvalalign(INT4OID, &typlen, &typbyval, &typalign);
	result = construct_array(result_datums,
							 selected_count,
							 INT4OID,
							 typlen,
							 typbyval,
							 typalign);

	NDB_FREE(candidates);
	NDB_FREE(query_scores);
	NDB_FREE(selected);
	NDB_FREE(result_indices);
	NDB_FREE(result_datums);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * mmr_rerank_with_scores
 *    MMR reranking that also returns MMR scores
 *
 * Returns: array of (index, mmr_score) tuples
 */
PG_FUNCTION_INFO_V1(mmr_rerank_with_scores);

Datum
mmr_rerank_with_scores(PG_FUNCTION_ARGS)
{
	ArrayType  *query_array;
	ArrayType  *candidates_array;
	float		lambda;
	int			top_k;
	float	   *query;
	int			query_dim;
	NDB_DECLARE(float **, candidates);
	int			n_candidates;
	int			dim;
	NDB_DECLARE(bool *, selected);
	NDB_DECLARE(int *, result_indices);
	NDB_DECLARE(double *, result_scores);
	NDB_DECLARE(double *, query_scores);
	ArrayType  *result;
	NDB_DECLARE(Datum *, result_datums);
	int			selected_count;
	int			i,
				j;

	/* Parse arguments (same as mmr_rerank) */
	query_array = PG_GETARG_ARRAYTYPE_P(0);
	candidates_array = PG_GETARG_ARRAYTYPE_P(1);
	lambda = PG_GETARG_FLOAT4(2);
	top_k = PG_GETARG_INT32(3);

	if (lambda < 0.0 || lambda > 1.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("lambda must be between 0.0 and 1.0")));

	query_dim = ARR_DIMS(query_array)[0];
	query = (float *) ARR_DATA_PTR(query_array);

	if (ARR_NDIM(candidates_array) != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("candidates must be 2-dimensional "
						"array")));

	n_candidates = ARR_DIMS(candidates_array)[0];
	dim = ARR_DIMS(candidates_array)[1];

	if (dim != query_dim)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Dimension mismatch")));

	if (top_k < 1 || top_k > n_candidates)
		top_k = n_candidates;

	NDB_ALLOC(candidates, float *, n_candidates);
	{
		float	   *data = (float *) ARR_DATA_PTR(candidates_array);

		for (i = 0; i < n_candidates; i++)
			candidates[i] = &data[i * dim];
	}

	NDB_ALLOC(query_scores, double, n_candidates);
	for (i = 0; i < n_candidates; i++)
		query_scores[i] = cosine_similarity(query, candidates[i], dim);

	NDB_ALLOC(selected, bool, n_candidates);
	NDB_ALLOC(result_indices, int, top_k);
	NDB_ALLOC(result_scores, double, top_k);
	selected_count = 0;

	/* MMR algorithm with score tracking */
	while (selected_count < top_k)
	{
		double		max_mmr = -1e30;
		int			best_idx = -1;

		for (i = 0; i < n_candidates; i++)
		{
			double		mmr_score;
			double		max_sim_to_selected;

			if (selected[i])
				continue;

			max_sim_to_selected = 0.0;
			for (j = 0; j < selected_count; j++)
			{
				int			selected_idx = result_indices[j];
				double		sim = cosine_similarity(candidates[i],
													candidates[selected_idx],
													dim);

				if (sim > max_sim_to_selected)
					max_sim_to_selected = sim;
			}

			mmr_score = lambda * query_scores[i]
				- (1.0 - lambda) * max_sim_to_selected;

			if (mmr_score > max_mmr)
			{
				max_mmr = mmr_score;
				best_idx = i;
			}
		}

		if (best_idx == -1)
			break;

		result_indices[selected_count] = best_idx;
		result_scores[selected_count] = max_mmr;
		selected[best_idx] = true;
		selected_count++;
	}

	/* Build result array: flat array [idx1, score1, idx2, score2, ...] */
	NDB_ALLOC(result_datums, Datum, selected_count * 2);
	for (i = 0; i < selected_count; i++)
	{
		result_datums[i * 2] = Int32GetDatum(result_indices[i] + 1);
		result_datums[i * 2 + 1] =
			Float4GetDatum((float) result_scores[i]);
	}

	result = construct_array(result_datums,
							 selected_count * 2,
							 FLOAT4OID,
							 sizeof(float4),
							 true,
							 'i');

	NDB_FREE(candidates);
	NDB_FREE(query_scores);
	NDB_FREE(selected);
	NDB_FREE(result_indices);
	NDB_FREE(result_scores);
	NDB_FREE(result_datums);

	PG_RETURN_ARRAYTYPE_P(result);
}
