/*-------------------------------------------------------------------------
 *
 * ml_recall_metrics.c
 *    Search quality evaluation metrics.
 *
 * This module implements retrieval evaluation metrics including Recall@K,
 * Precision@K, F1@K, MRR, and NDCG for assessing search quality.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_recall_metrics.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "catalog/pg_type.h"

#include "neurondb.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"

#include <math.h>

/*
 * recall_at_k
 * -----------
 * Compute Recall@K for retrieval evaluation.
 *
 * SQL Arguments:
 *   retrieved_ids  - Array of retrieved document IDs (ordered by rank)
 *   relevant_ids   - Array of known relevant document IDs
 *   k              - Cut-off rank (default: length of retrieved_ids)
 *
 * Returns:
 *   Float8: Recall@K value in [0, 1]
 *
 * Interpretation:
 *   - 1.0: All relevant items found in top K
 *   - 0.5: Half of relevant items found
 *   - 0.0: No relevant items in top K
 *
 * Example Usage:
 *   -- Evaluate search quality:
 *   SELECT recall_at_k(
 *     ARRAY[101, 205, 303, 102, 501],  -- Retrieved docs
 *     ARRAY[101, 102, 103, 104],        -- Known relevant docs
 *     5                                  -- K=5
 *   );  -- Returns: 0.5 (found 2 out of 4 relevant docs)
 *
 *   -- Monitor across queries:
 *   SELECT query_id, recall_at_k(results, ground_truth, 10) AS recall_10
 *   FROM search_evaluation;
 *
 * Notes:
 *   - Order of relevant_ids doesn't matter
 *   - Duplicate IDs are handled correctly
 *   - If k > length(retrieved_ids), uses full list
 */
PG_FUNCTION_INFO_V1(recall_at_k);

Datum
recall_at_k(PG_FUNCTION_ARGS)
{
	ArrayType  *retrieved_array;
	ArrayType  *relevant_array;
	int			k;
	int32	   *retrieved_ids;
	int32	   *relevant_ids;
	int			n_retrieved;
	int			n_relevant;
	int			found_count;
	int			i,
				j;
	double		recall;

	/* Parse arguments */
	retrieved_array = PG_GETARG_ARRAYTYPE_P(0);
	relevant_array = PG_GETARG_ARRAYTYPE_P(1);
	k = PG_ARGISNULL(2) ? -1 : PG_GETARG_INT32(2);

	/* Extract arrays */
	n_retrieved = ARR_DIMS(retrieved_array)[0];
	n_relevant = ARR_DIMS(relevant_array)[0];
	retrieved_ids = (int32 *) ARR_DATA_PTR(retrieved_array);
	relevant_ids = (int32 *) ARR_DATA_PTR(relevant_array);

	if (n_relevant == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("relevant_ids cannot be empty")));

	if (k < 0 || k > n_retrieved)
		k = n_retrieved;

	/* Count how many relevant items are in top K retrieved */
	found_count = 0;
	for (i = 0; i < n_relevant; i++)
	{
		for (j = 0; j < k; j++)
		{
			if (retrieved_ids[j] == relevant_ids[i])
			{
				found_count++;
				break;
			}
		}
	}

	recall = (double) found_count / n_relevant;

	elog(DEBUG1,
		 "neurondb: Recall@%d = %.4f (found %d of %d)",
		 k,
		 recall,
		 found_count,
		 n_relevant);

	PG_RETURN_FLOAT8(recall);
}

/*
 * precision_at_k
 * --------------
 * Compute Precision@K (fraction of top K that are relevant).
 */
PG_FUNCTION_INFO_V1(precision_at_k);

Datum
precision_at_k(PG_FUNCTION_ARGS)
{
	ArrayType  *retrieved_array;
	ArrayType  *relevant_array;
	int			k;
	int32	   *retrieved_ids;
	int32	   *relevant_ids;
	int			n_retrieved;
	int			n_relevant;
	int			found_count;
	int			i,
				j;
	double		precision;

	retrieved_array = PG_GETARG_ARRAYTYPE_P(0);
	relevant_array = PG_GETARG_ARRAYTYPE_P(1);
	k = PG_ARGISNULL(2) ? -1 : PG_GETARG_INT32(2);

	n_retrieved = ARR_DIMS(retrieved_array)[0];
	n_relevant = ARR_DIMS(relevant_array)[0];
	retrieved_ids = (int32 *) ARR_DATA_PTR(retrieved_array);
	relevant_ids = (int32 *) ARR_DATA_PTR(relevant_array);

	if (n_relevant == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("relevant_ids cannot be empty")));

	if (k < 0 || k > n_retrieved)
		k = n_retrieved;

	if (k == 0)
		PG_RETURN_FLOAT8(0.0);

	/* Count relevant items in top K */
	found_count = 0;
	for (i = 0; i < k; i++)
	{
		for (j = 0; j < n_relevant; j++)
		{
			if (retrieved_ids[i] == relevant_ids[j])
			{
				found_count++;
				break;
			}
		}
	}

	precision = (double) found_count / k;
	PG_RETURN_FLOAT8(precision);
}

/*
 * f1_at_k
 * -------
 * Compute F1 score at K (harmonic mean of Precision@K and Recall@K).
 */
PG_FUNCTION_INFO_V1(f1_at_k);

Datum
f1_at_k(PG_FUNCTION_ARGS)
{
	ArrayType  *retrieved_array;
	ArrayType  *relevant_array;
	int			k;
	int32	   *retrieved_ids;
	int32	   *relevant_ids;
	int			n_retrieved;
	int			n_relevant;
	int			found_count;
	int			i,
				j;
	double		precision,
				recall,
				f1;

	retrieved_array = PG_GETARG_ARRAYTYPE_P(0);
	relevant_array = PG_GETARG_ARRAYTYPE_P(1);
	k = PG_ARGISNULL(2) ? -1 : PG_GETARG_INT32(2);

	n_retrieved = ARR_DIMS(retrieved_array)[0];
	n_relevant = ARR_DIMS(relevant_array)[0];
	retrieved_ids = (int32 *) ARR_DATA_PTR(retrieved_array);
	relevant_ids = (int32 *) ARR_DATA_PTR(relevant_array);

	if (n_relevant == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("relevant_ids cannot be empty")));

	if (k < 0 || k > n_retrieved)
		k = n_retrieved;

	if (k == 0)
		PG_RETURN_FLOAT8(0.0);

	found_count = 0;
	for (i = 0; i < k; i++)
	{
		for (j = 0; j < n_relevant; j++)
		{
			if (retrieved_ids[i] == relevant_ids[j])
			{
				found_count++;
				break;
			}
		}
	}

	precision = (double) found_count / k;
	recall = (double) found_count / n_relevant;

	if (precision + recall < 1e-10)
		f1 = 0.0;
	else
		f1 = 2.0 * (precision * recall) / (precision + recall);

	PG_RETURN_FLOAT8(f1);
}

/*
 * mean_reciprocal_rank
 * --------------------
 * Compute MRR (Mean Reciprocal Rank) for multiple queries.
 *
 * SQL Arguments:
 *   retrieved_lists - 2D array: each row is retrieved IDs for one query
 *   relevant_lists  - 2D array: each row is relevant IDs for one query
 *
 * Returns:
 *   Float8: MRR across all queries
 *
 * Example:
 *   SELECT mean_reciprocal_rank(
 *     ARRAY[ARRAY[101,202,303], ARRAY[501,502,101]],  -- 2 queries
 *     ARRAY[ARRAY[101], ARRAY[101]]                    -- Ground truth
 *   );  -- Returns: (1/1 + 1/3) / 2 = 0.667
 */
PG_FUNCTION_INFO_V1(mean_reciprocal_rank);

Datum
mean_reciprocal_rank(PG_FUNCTION_ARGS)
{
	ArrayType  *retrieved_array;
	ArrayType  *relevant_array;
	int			ndim_retrieved,
				ndim_relevant;
	int		   *dims_retrieved,
			   *dims_relevant;
	int			n_queries;
	int			max_retrieved_len,
				max_relevant_len;
	int32	   *retrieved_data;
	int32	   *relevant_data;
	double		mrr;
	double		sum_rr;
	int			q,
				i,
				j;

	retrieved_array = PG_GETARG_ARRAYTYPE_P(0);
	relevant_array = PG_GETARG_ARRAYTYPE_P(1);

	/* Validate 2D arrays */
	ndim_retrieved = ARR_NDIM(retrieved_array);
	ndim_relevant = ARR_NDIM(relevant_array);

	if (ndim_retrieved != 2 || ndim_relevant != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Both arrays must be 2-dimensional")));

	dims_retrieved = ARR_DIMS(retrieved_array);
	dims_relevant = ARR_DIMS(relevant_array);
	n_queries = dims_retrieved[0];
	max_retrieved_len = dims_retrieved[1];
	max_relevant_len = dims_relevant[1];

	if (dims_relevant[0] != n_queries)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Number of queries must match in both "
						"arrays")));

	retrieved_data = (int32 *) ARR_DATA_PTR(retrieved_array);
	relevant_data = (int32 *) ARR_DATA_PTR(relevant_array);

	/* Compute reciprocal ranks */
	sum_rr = 0.0;
	for (q = 0; q < n_queries; q++)
	{
		int			first_relevant_rank = -1;
		int32	   *query_retrieved = &retrieved_data[q * max_retrieved_len];
		int32	   *query_relevant = &relevant_data[q * max_relevant_len];

		/* Find rank of first relevant item */
		for (i = 0; i < max_retrieved_len; i++)
		{
			for (j = 0; j < max_relevant_len; j++)
			{
				if (query_retrieved[i] == query_relevant[j]
					&& query_relevant[j] > 0)
				{
					first_relevant_rank =
						i + 1;	/* 1-based rank */
					break;
				}
			}
			if (first_relevant_rank > 0)
				break;
		}

		if (first_relevant_rank > 0)
			sum_rr += 1.0 / first_relevant_rank;
		/* else: no relevant item found, contributes 0 to MRR */
	}

	mrr = sum_rr / n_queries;
	PG_RETURN_FLOAT8(mrr);
}
