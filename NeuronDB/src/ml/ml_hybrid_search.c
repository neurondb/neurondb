/*-------------------------------------------------------------------------
 *
 * ml_hybrid_search.c
 *    Lexical-semantic hybrid search.
 *
 * This module combines lexical matching and semantic matching for improved
 * search results using weighted fusion methods.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_hybrid_search.c
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
 * Document score with hybrid components
 */
typedef struct HybridScore
{
	int32		doc_id;
	double		lexical_score;
	double		semantic_score;
	double		hybrid_score;
}			HybridScore;

/*
 * Comparison for sorting (descending by hybrid score)
 */
static int
hybrid_score_cmp(const void *a, const void *b)
{
	const		HybridScore *ha = (const HybridScore *) a;
	const		HybridScore *hb = (const HybridScore *) b;

	if (ha->hybrid_score > hb->hybrid_score)
		return -1;
	if (ha->hybrid_score < hb->hybrid_score)
		return 1;
	return 0;
}

/*
 * hybrid_search_fusion
 * --------------------
 * Combine lexical and semantic scores with weighted fusion.
 *
 * SQL Arguments:
 *   doc_ids         - Document IDs
 *   semantic_scores - Vector similarity scores (e.g., cosine)
 *   lexical_scores  - Lexical scores (e.g., BM25, ts_rank)
 *   semantic_weight - Weight for semantic component (default: 0.5)
 *   normalize       - Apply min-max normalization (default: true)
 *
 * Returns:
 *   Array of document IDs sorted by hybrid score
 *
 * Example Usage:
 *   -- Equal weight hybrid search:
 *   WITH semantic AS (
 *     SELECT id, embedding <=> query_vec AS sim_score
 *     FROM documents, (SELECT embedding FROM queries WHERE id = 1) q(query_vec)
 *   ),
 *   lexical AS (
 *     SELECT id, ts_rank(tsv, query) AS lex_score
 *     FROM documents, (SELECT to_tsquery('search query') AS query) q
 *   )
 *   SELECT hybrid_search_fusion(
 *     array_agg(semantic.id),
 *     array_agg(semantic.sim_score),
 *     array_agg(lexical.lex_score),
 *     0.6,  -- 60% semantic, 40% lexical
 *     true  -- normalize
 *   )
 *   FROM semantic
 *   JOIN lexical ON semantic.id = lexical.id;
 *
 * Tuning Guidelines:
 *   - 0.5: Balanced (good starting point)
 *   - 0.7-0.8: Favor semantic (conceptual search)
 *   - 0.2-0.3: Favor lexical (keyword-heavy domains)
 *   - Tune based on metrics (Recall@K, NDCG)
 *
 * Notes:
 *   - Normalization recommended when score scales differ
 *   - Consider query type routing (short=lexical, long=semantic)
 *   - Monitor hit rates for each component
 */
PG_FUNCTION_INFO_V1(hybrid_search_fusion);

Datum
hybrid_search_fusion(PG_FUNCTION_ARGS)
{
	ArrayType  *doc_ids_array;
	ArrayType  *semantic_scores_array;
	ArrayType  *lexical_scores_array;
	double		semantic_weight;
	bool		normalize;
	int32	   *doc_ids;
	float8	   *semantic_scores;
	float8	   *lexical_scores;
	int			num_docs;
	HybridScore *scores;
	double		sem_min,
				sem_max,
				lex_min,
				lex_max;
	int			i;
	ArrayType  *result;
	Datum	   *result_datums;
	int16		typlen;
	bool		typbyval;
	char		typalign;

	/* Parse arguments */
	doc_ids_array = PG_GETARG_ARRAYTYPE_P(0);
	semantic_scores_array = PG_GETARG_ARRAYTYPE_P(1);
	lexical_scores_array = PG_GETARG_ARRAYTYPE_P(2);
	semantic_weight = PG_ARGISNULL(3) ? 0.5 : PG_GETARG_FLOAT8(3);
	normalize = PG_ARGISNULL(4) ? true : PG_GETARG_BOOL(4);

	/* Validate */
	if (ARR_NDIM(doc_ids_array) != 1 || ARR_NDIM(semantic_scores_array) != 1
		|| ARR_NDIM(lexical_scores_array) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("All arrays must be 1-dimensional")));

	num_docs = ARR_DIMS(doc_ids_array)[0];

	if (ARR_DIMS(semantic_scores_array)[0] != num_docs
		|| ARR_DIMS(lexical_scores_array)[0] != num_docs)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("All arrays must have same length")));

	if (semantic_weight < 0.0 || semantic_weight > 1.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("semantic_weight must be between 0 and "
						"1")));

	doc_ids = (int32 *) ARR_DATA_PTR(doc_ids_array);
	semantic_scores = (float8 *) ARR_DATA_PTR(semantic_scores_array);
	lexical_scores = (float8 *) ARR_DATA_PTR(lexical_scores_array);

	elog(DEBUG1,
		 "neurondb: Hybrid search fusion (%d docs, weight=%.2f)",
		 num_docs,
		 semantic_weight);

	scores = (HybridScore *) palloc(sizeof(HybridScore) * num_docs);

	/* Initialize scores */
	for (i = 0; i < num_docs; i++)
	{
		scores[i].doc_id = doc_ids[i];
		scores[i].semantic_score = semantic_scores[i];
		scores[i].lexical_score = lexical_scores[i];
	}

	/* Normalize if requested */
	if (normalize)
	{
		/* Find min/max for semantic */
		sem_min = DBL_MAX;
		sem_max = -DBL_MAX;
		for (i = 0; i < num_docs; i++)
		{
			if (scores[i].semantic_score < sem_min)
				sem_min = scores[i].semantic_score;
			if (scores[i].semantic_score > sem_max)
				sem_max = scores[i].semantic_score;
		}

		/* Find min/max for lexical */
		lex_min = DBL_MAX;
		lex_max = -DBL_MAX;
		for (i = 0; i < num_docs; i++)
		{
			if (scores[i].lexical_score < lex_min)
				lex_min = scores[i].lexical_score;
			if (scores[i].lexical_score > lex_max)
				lex_max = scores[i].lexical_score;
		}

		/* Normalize to [0, 1] */
		{
			double		sem_range = (sem_max - sem_min) < 1e-10
				? 1.0
				: (sem_max - sem_min);
			double		lex_range = (lex_max - lex_min) < 1e-10
				? 1.0
				: (lex_max - lex_min);

			for (i = 0; i < num_docs; i++)
			{
				scores[i].semantic_score =
					(scores[i].semantic_score - sem_min)
					/ sem_range;
				scores[i].lexical_score =
					(scores[i].lexical_score - lex_min)
					/ lex_range;
			}
		}
	}

	/* Compute hybrid scores (weighted fusion) */
	{
		double		lexical_weight = 1.0 - semantic_weight;

		for (i = 0; i < num_docs; i++)
		{
			scores[i].hybrid_score =
				semantic_weight * scores[i].semantic_score
				+ lexical_weight * scores[i].lexical_score;
		}
	}

	/* Sort by hybrid score */
	qsort(scores, num_docs, sizeof(HybridScore), hybrid_score_cmp);

	/* Build result */
	result_datums = (Datum *) palloc(sizeof(Datum) * num_docs);
	for (i = 0; i < num_docs; i++)
		result_datums[i] = Int32GetDatum(scores[i].doc_id);

	get_typlenbyvalalign(INT4OID, &typlen, &typbyval, &typalign);
	result = construct_array(
							 result_datums, num_docs, INT4OID, typlen, typbyval, typalign);

	NDB_FREE(scores);
	NDB_FREE(result_datums);

	PG_RETURN_ARRAYTYPE_P(result);
}
