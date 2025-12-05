/*-------------------------------------------------------------------------
 *
 * ml_rerank_ensemble.c
 *    Ensemble reranking methods.
 *
 * This module combines multiple ranking models using weighted sum, rank-based
 * fusion, and Borda count methods for improved search relevance.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_rerank_ensemble.c
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
 * Document score structure
 */
typedef struct DocScore
{
	int32		doc_id;
	double		ensemble_score;
}			DocScore;

/*
 * Comparison for sorting by score (descending)
 */
static int
docscore_cmp(const void *a, const void *b)
{
	const		DocScore *da = (const DocScore *) a;
	const		DocScore *db = (const DocScore *) b;

	if (da->ensemble_score > db->ensemble_score)
		return -1;
	if (da->ensemble_score < db->ensemble_score)
		return 1;
	return 0;
}

/*
 * rerank_ensemble_weighted
 * -------------------------
 * Combine multiple ranking systems using weighted sum.
 *
 * SQL Arguments:
 *   doc_ids       - Array of document IDs (all systems must use same IDs)
 *   score_matrix  - 2D array: [num_systems][num_docs]
 *                   Each row is scores from one ranking system
 *   weights       - Weights for each system (default: equal weights)
 *   normalize     - Apply min-max normalization (default: true)
 *
 * Returns:
 *   Array of document IDs sorted by ensemble score (descending)
 *
 * Example Usage:
 *   -- Combine semantic (0.6) + BM25 (0.4):
 *   SELECT rerank_ensemble_weighted(
 *     ARRAY[1, 2, 3, 4, 5],                    -- Doc IDs
 *     ARRAY[
 *       ARRAY[0.9, 0.8, 0.6, 0.4, 0.2],       -- Semantic scores
 *       ARRAY[0.5, 0.9, 0.3, 0.7, 0.1]        -- BM25 scores
 *     ],
 *     ARRAY[0.6, 0.4],                         -- Weights
 *     true                                      -- Normalize
 *   );
 *
 * Notes:
 *   - Weights should sum to 1.0 (will be normalized if not)
 *   - Min-max normalization recommended for heterogeneous scores
 *   - All systems must provide scores for all documents
 */
PG_FUNCTION_INFO_V1(rerank_ensemble_weighted);

Datum
rerank_ensemble_weighted(PG_FUNCTION_ARGS)
{
	ArrayType  *doc_ids_array;
	ArrayType  *score_matrix_array;
	ArrayType  *weights_array;
	bool		normalize;
	int32	   *doc_ids;
	float8	   *score_matrix;
	float8	   *weights;
	int			num_docs;
	int			num_systems;
	DocScore   *doc_scores;
	double	   *normalized_scores;
	double		weight_sum;
	int			i,
				s;
	ArrayType  *result;
	Datum	   *result_datums;
	int16		typlen;
	bool		typbyval;
	char		typalign;

	doc_ids_array = PG_GETARG_ARRAYTYPE_P(0);
	score_matrix_array = PG_GETARG_ARRAYTYPE_P(1);

	if (!PG_ARGISNULL(2))
		weights_array = PG_GETARG_ARRAYTYPE_P(2);
	else
		weights_array = NULL;

	normalize = PG_ARGISNULL(3) ? true : PG_GETARG_BOOL(3);

	/* Validate and extract arrays */
	if (ARR_NDIM(doc_ids_array) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("doc_ids must be 1-dimensional array")));

	if (ARR_NDIM(score_matrix_array) != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("score_matrix must be 2-dimensional "
						"array")));

	num_docs = ARR_DIMS(doc_ids_array)[0];
	num_systems = ARR_DIMS(score_matrix_array)[0];

	if (ARR_DIMS(score_matrix_array)[1] != num_docs)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("score_matrix must have %d columns to match doc_ids",
						num_docs)));

	doc_ids = (int32 *) ARR_DATA_PTR(doc_ids_array);
	score_matrix = (float8 *) ARR_DATA_PTR(score_matrix_array);

	/* Handle weights */
	if (weights_array != NULL)
	{
		if (ARR_DIMS(weights_array)[0] != num_systems)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("weights array must have %d elements",
							num_systems)));

		weights = (float8 *) ARR_DATA_PTR(weights_array);

		/* Normalize weights to sum to 1.0 */
		weight_sum = 0.0;
		for (s = 0; s < num_systems; s++)
			weight_sum += weights[s];

		if (weight_sum < 1e-10)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("Weights must sum to positive "
							"value")));
	}
	else
	{
		/* Equal weights */
		NDB_DECLARE(float8 *, weights);
		NDB_ALLOC(weights, float8, num_systems);
		NDB_CHECK_ALLOC(weights, "weights");
		for (s = 0; s < num_systems; s++)
			weights[s] = 1.0 / num_systems;
		weight_sum = 1.0;
	}

	elog(DEBUG1,
		 "neurondb: Ensemble reranking: %d systems, %d docs",
		 num_systems,
		 num_docs);

	/* Normalize scores if requested */
	NDB_ALLOC(normalized_scores, double, num_systems * num_docs);

	if (normalize)
	{
		/* Min-max normalization per system */
		for (s = 0; s < num_systems; s++)
		{
			double		min_score = DBL_MAX;
			double		max_score = -DBL_MAX;

			/* Find min/max */
			for (i = 0; i < num_docs; i++)
			{
				double		score = score_matrix[s * num_docs + i];

				if (score < min_score)
					min_score = score;
				if (score > max_score)
					max_score = score;
			}

			/* Normalize to [0, 1] */
			{
				double		range = max_score - min_score;

				if (range < 1e-10)
					range = 1.0;	/* All scores equal */

				for (i = 0; i < num_docs; i++)
				{
					double		score =
						score_matrix[s * num_docs + i];

					normalized_scores[s * num_docs + i] =
						(score - min_score) / range;
				}
			}
		}
	}
	else
	{
		/* Use raw scores */
		for (s = 0; s < num_systems; s++)
			for (i = 0; i < num_docs; i++)
				normalized_scores[s * num_docs + i] =
					score_matrix[s * num_docs + i];
	}

	/* Compute ensemble scores (weighted sum) */
	NDB_ALLOC(doc_scores, DocScore, num_docs);
	NDB_CHECK_ALLOC(doc_scores, "doc_scores");

	for (i = 0; i < num_docs; i++)
	{
		double		ensemble_score = 0.0;

		for (s = 0; s < num_systems; s++)
			ensemble_score += (weights[s] / weight_sum)
				* normalized_scores[s * num_docs + i];

		doc_scores[i].doc_id = doc_ids[i];
		doc_scores[i].ensemble_score = ensemble_score;
	}

	/* Sort by ensemble score */
	qsort(doc_scores, num_docs, sizeof(DocScore), docscore_cmp);

	/* Build result array */
	NDB_ALLOC(result_datums, Datum, num_docs);
	NDB_CHECK_ALLOC(result_datums, "result_datums");
	for (i = 0; i < num_docs; i++)
		result_datums[i] = Int32GetDatum(doc_scores[i].doc_id);

	get_typlenbyvalalign(INT4OID, &typlen, &typbyval, &typalign);
	result = construct_array(
							 result_datums, num_docs, INT4OID, typlen, typbyval, typalign);

	NDB_FREE(normalized_scores);
	NDB_FREE(doc_scores);
	NDB_FREE(result_datums);
	if (weights_array == NULL)
		NDB_FREE(weights);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * rerank_ensemble_borda
 * ---------------------
 * Borda count: rank-based voting for ensemble.
 *
 * Each system assigns points based on rank:
 *   - Top rank: n points
 *   - Second: n-1 points
 *   - ...
 *   - Last: 1 point
 *
 * Returns documents sorted by total Borda points.
 */
PG_FUNCTION_INFO_V1(rerank_ensemble_borda);

Datum
rerank_ensemble_borda(PG_FUNCTION_ARGS)
{
	ArrayType  *ranked_lists_array;
	int			ndim;
	int		   *dims;
	int			num_systems;
	int			max_docs;
	int32	   *ranked_lists;
	DocScore   *doc_scores;
	int		   *doc_id_map;
	int			num_unique_docs;
	int			s,
				i,
				rank;
	ArrayType  *result;
	Datum	   *result_datums;
	int16		typlen;
	bool		typbyval;
	char		typalign;

	ranked_lists_array = PG_GETARG_ARRAYTYPE_P(0);

	ndim = ARR_NDIM(ranked_lists_array);
	if (ndim != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("ranked_lists must be 2-dimensional")));

	dims = ARR_DIMS(ranked_lists_array);
	num_systems = dims[0];
	max_docs = dims[1];
	ranked_lists = (int32 *) ARR_DATA_PTR(ranked_lists_array);

	elog(DEBUG1,
		 "neurondb: Borda count ensemble: %d systems, max %d docs",
		 num_systems,
		 max_docs);

	/* Collect unique doc IDs */
	NDB_ALLOC(doc_id_map, int, num_systems * max_docs);
	NDB_CHECK_ALLOC(doc_id_map, "doc_id_map");
	num_unique_docs = 0;

	for (s = 0; s < num_systems; s++)
	{
		for (i = 0; i < max_docs; i++)
		{
			int32		doc_id = ranked_lists[s * max_docs + i];
			int			found = 0;
			int			j;

			if (doc_id <= 0)
				continue;

			for (j = 0; j < num_unique_docs; j++)
			{
				if (doc_id_map[j] == doc_id)
				{
					found = 1;
					break;
				}
			}

			if (!found)
			{
				doc_id_map[num_unique_docs] = doc_id;
				num_unique_docs++;
			}
		}
	}

	/* Initialize scores */
	NDB_ALLOC(doc_scores, DocScore, num_unique_docs);
	NDB_CHECK_ALLOC(doc_scores, "doc_scores");
	for (i = 0; i < num_unique_docs; i++)
	{
		doc_scores[i].doc_id = doc_id_map[i];
		doc_scores[i].ensemble_score = 0.0;
	}

	/* Compute Borda scores */
	for (s = 0; s < num_systems; s++)
	{
		for (rank = 0; rank < max_docs; rank++)
		{
			int32		doc_id = ranked_lists[s * max_docs + rank];
			int			points =
				max_docs - rank;	/* Higher rank = more points */
			int			j;

			if (doc_id <= 0)
				continue;

			for (j = 0; j < num_unique_docs; j++)
			{
				if (doc_scores[j].doc_id == doc_id)
				{
					doc_scores[j].ensemble_score += points;
					break;
				}
			}
		}
	}

	/* Sort by Borda score */
	qsort(doc_scores, num_unique_docs, sizeof(DocScore), docscore_cmp);

	NDB_ALLOC(result_datums, Datum, num_unique_docs);
	NDB_CHECK_ALLOC(result_datums, "result_datums");
	for (i = 0; i < num_unique_docs; i++)
		result_datums[i] = Int32GetDatum(doc_scores[i].doc_id);

	get_typlenbyvalalign(INT4OID, &typlen, &typbyval, &typalign);
	result = construct_array(result_datums,
							 num_unique_docs,
							 INT4OID,
							 typlen,
							 typbyval,
							 typalign);

	NDB_FREE(doc_id_map);
	NDB_FREE(doc_scores);
	NDB_FREE(result_datums);

	PG_RETURN_ARRAYTYPE_P(result);
}
