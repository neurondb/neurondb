/*-------------------------------------------------------------------------
 *
 * ml_ltr.c
 *    Learning to rank implementation.
 *
 * This module implements pointwise linear models for reranking search
 * results based on multiple features.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_ltr.c
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
#include "neurondb_simd.h"

#include <math.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/*
 * Document score structure
 */
typedef struct DocLTRScore
{
	int32		doc_id;
	double		ltr_score;
}			DocLTRScore;

/*
 * Comparison for sorting (descending)
 */
static int
doc_ltr_score_cmp(const void *a, const void *b)
{
	const		DocLTRScore *da = (const DocLTRScore *) a;
	const		DocLTRScore *db = (const DocLTRScore *) b;

	if (da->ltr_score > db->ltr_score)
		return -1;
	if (da->ltr_score < db->ltr_score)
		return 1;
	return 0;
}

/*
 * ltr_rerank_pointwise
 * --------------------
 * Apply linear LTR model to rerank documents.
 *
 * SQL Arguments:
 *   doc_ids        - Array of document IDs
 *   feature_matrix - 2D array [num_docs][num_features]
 *   weights        - Feature weights (trained externally)
 *   bias           - Model bias/intercept (default: 0.0)
 *
 * Returns:
 *   Array of document IDs sorted by LTR score (descending)
 *
 * Example Usage:
 *   -- Combine semantic (w=0.5), BM25 (w=0.3), recency (w=0.2):
 *   SELECT ltr_rerank_pointwise(
 *     ARRAY[1, 2, 3, 4, 5],                       -- Doc IDs
 *     ARRAY[
 *       ARRAY[0.9, 0.7, 0.2],                     -- Doc 1 features
 *       ARRAY[0.8, 0.9, 0.1],                     -- Doc 2 features
 *       ARRAY[0.6, 0.5, 0.8],                     -- Doc 3 features
 *       ARRAY[0.4, 0.6, 0.9],                     -- Doc 4 features
 *       ARRAY[0.2, 0.3, 0.5]                      -- Doc 5 features
 *     ],
 *     ARRAY[0.5, 0.3, 0.2],                       -- Feature weights
 *     0.0                                          -- Bias
 *   );
 *
 * Training Workflow:
 *   1. Collect features for training data
 *   2. Label with relevance (e.g., clicks, ratings)
 *   3. Train model externally (LightGBM, sklearn)
 *   4. Deploy weights to this function
 *   5. Monitor and retrain periodically
 *
 * Notes:
 *   - Uses SIMD-optimized dot product
 *   - Normalize features for best results
 *   - Weights typically sum to ~1.0
 *   - Consider feature scaling before training
 */
PG_FUNCTION_INFO_V1(ltr_rerank_pointwise);

Datum
ltr_rerank_pointwise(PG_FUNCTION_ARGS)
{
	ArrayType  *doc_ids_array;
	ArrayType  *feature_matrix_array;
	ArrayType  *weights_array;
	double		bias;
	int32	   *doc_ids;
	float8	   *feature_matrix;
	float8	   *weights;
	int			num_docs;
	int			num_features;
	DocLTRScore *doc_scores;
	int			i,
				f;
#if defined(NEURONDB_HAS_AVX2) || defined(NEURONDB_HAS_NEON)
	float	   *features_f;
	float	   *weights_f;
#endif
	ArrayType  *result;
	Datum	   *result_datums;
	int16		typlen;
	bool		typbyval;
	char		typalign;

	/* Parse arguments */
	doc_ids_array = PG_GETARG_ARRAYTYPE_P(0);
	feature_matrix_array = PG_GETARG_ARRAYTYPE_P(1);
	weights_array = PG_GETARG_ARRAYTYPE_P(2);
	bias = PG_ARGISNULL(3) ? 0.0 : PG_GETARG_FLOAT8(3);

	/* Validate arrays */
	if (ARR_NDIM(doc_ids_array) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("doc_ids must be 1-dimensional")));

	if (ARR_NDIM(feature_matrix_array) != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("feature_matrix must be "
						"2-dimensional")));

	if (ARR_NDIM(weights_array) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("weights must be 1-dimensional")));

	num_docs = ARR_DIMS(doc_ids_array)[0];

	if (ARR_DIMS(feature_matrix_array)[0] != num_docs)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("feature_matrix must have %d rows to match doc_ids",
						num_docs)));

	num_features = ARR_DIMS(feature_matrix_array)[1];

	if (ARR_DIMS(weights_array)[0] != num_features)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("weights must have %d elements to match features",
						num_features)));

	doc_ids = (int32 *) ARR_DATA_PTR(doc_ids_array);
	feature_matrix = (float8 *) ARR_DATA_PTR(feature_matrix_array);
	weights = (float8 *) ARR_DATA_PTR(weights_array);

	elog(DEBUG1,
		 "neurondb: LTR reranking %d docs with %d features",
		 num_docs,
		 num_features);

	/* Compute LTR scores using SIMD-optimized dot product */
	doc_scores = (DocLTRScore *) palloc(sizeof(DocLTRScore) * num_docs);

#if defined(NEURONDB_HAS_AVX2) || defined(NEURONDB_HAS_NEON)
	features_f = (float *) palloc(sizeof(float) * num_features);
	weights_f = (float *) palloc(sizeof(float) * num_features);

	/* Convert weights once */
	for (f = 0; f < num_features; f++)
		weights_f[f] = (float) weights[f];
#endif

	for (i = 0; i < num_docs; i++)
	{
		double		score = bias;

/* Use SIMD dot product for feature · weight */
#if defined(NEURONDB_HAS_AVX2) || defined(NEURONDB_HAS_NEON)
		/* Convert features to floats for SIMD */
		for (f = 0; f < num_features; f++)
			features_f[f] =
				(float) feature_matrix[i * num_features + f];

		score += neurondb_dot_product(
									  features_f, weights_f, num_features);
#else
		/* Scalar fallback with double accumulator */
		for (f = 0; f < num_features; f++)
			score += feature_matrix[i * num_features + f]
				* weights[f];
#endif

		doc_scores[i].doc_id = doc_ids[i];
		doc_scores[i].ltr_score = score;
	}

#if defined(NEURONDB_HAS_AVX2) || defined(NEURONDB_HAS_NEON)
	NDB_FREE(features_f);
	NDB_FREE(weights_f);
#endif

	/* Sort by LTR score (descending) */
	qsort(doc_scores, num_docs, sizeof(DocLTRScore), doc_ltr_score_cmp);

	/* Build result array */
	result_datums = (Datum *) palloc(sizeof(Datum) * num_docs);
	for (i = 0; i < num_docs; i++)
		result_datums[i] = Int32GetDatum(doc_scores[i].doc_id);

	get_typlenbyvalalign(INT4OID, &typlen, &typbyval, &typalign);
	result = construct_array(
							 result_datums, num_docs, INT4OID, typlen, typbyval, typalign);

	/* Cleanup */
	NDB_FREE(doc_scores);
	NDB_FREE(result_datums);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * ltr_score_features
 * ------------------
 * Compute LTR scores without reranking (for debugging/analysis).
 *
 * Returns: Array of LTR scores (same order as input)
 */
PG_FUNCTION_INFO_V1(ltr_score_features);

Datum
ltr_score_features(PG_FUNCTION_ARGS)
{
	ArrayType  *feature_matrix_array;
	ArrayType  *weights_array;
	double		bias;
	float8	   *feature_matrix;
	float8	   *weights;
	int			num_docs;
	int			num_features;
	double	   *scores;
	int			i,
				f;
	ArrayType  *result;
	Datum	   *result_datums;

	/* Parse arguments */
	feature_matrix_array = PG_GETARG_ARRAYTYPE_P(0);
	weights_array = PG_GETARG_ARRAYTYPE_P(1);
	bias = PG_ARGISNULL(2) ? 0.0 : PG_GETARG_FLOAT8(2);

	if (ARR_NDIM(feature_matrix_array) != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("feature_matrix must be "
						"2-dimensional")));

	num_docs = ARR_DIMS(feature_matrix_array)[0];
	num_features = ARR_DIMS(feature_matrix_array)[1];

	if (ARR_DIMS(weights_array)[0] != num_features)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("weights dimension mismatch")));

	feature_matrix = (float8 *) ARR_DATA_PTR(feature_matrix_array);
	weights = (float8 *) ARR_DATA_PTR(weights_array);

	scores = (double *) palloc(sizeof(double) * num_docs);

	/* Compute scores */
	for (i = 0; i < num_docs; i++)
	{
		scores[i] = bias;
		for (f = 0; f < num_features; f++)
			scores[i] += feature_matrix[i * num_features + f]
				* weights[f];
	}

	/* Build result */
	result_datums = (Datum *) palloc(sizeof(Datum) * num_docs);
	for (i = 0; i < num_docs; i++)
		result_datums[i] = Float8GetDatum(scores[i]);

	result = construct_array(result_datums,
							 num_docs,
							 FLOAT8OID,
							 sizeof(float8),
							 FLOAT8PASSBYVAL,
							 'd');

	NDB_FREE(scores);
	NDB_FREE(result_datums);

	PG_RETURN_ARRAYTYPE_P(result);
}
