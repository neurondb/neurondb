/*-------------------------------------------------------------------------
 *
 * ml_outlier_detection.c
 *    Outlier detection for drift monitoring.
 *
 * This module implements Z-score, robust Z-score, and IQR methods for
 * detecting outliers in embedding data.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_outlier_detection.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "utils/lsyscache.h"

#include "neurondb.h"
#include "neurondb_ml.h"

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

static inline double
distance_from_mean(const float *vec, const double *mean, int dim)
{
	double		sum = 0.0;
	int			i;

	for (i = 0; i < dim; i++)
	{
		double		diff = (double) vec[i] - mean[i];

		sum += diff * diff;
	}
	return sqrt(sum);
}

static int
double_compare(const void *a, const void *b)
{
	double		da = *(const double *) a;
	double		db = *(const double *) b;

	if (da < db)
		return -1;
	if (da > db)
		return 1;
	return 0;
}

/*
 * detect_outliers_zscore
 * ----------------------
 * Z-score based outlier detection for embedding drift monitoring.
 *
 * SQL Arguments:
 *   table_name    - Source table with vectors
 *   vector_column - Vector column name
 *   threshold     - Z-score threshold (default: 3.0)
 *                   Typical values: 2.5 (more sensitive), 3.0 (standard), 3.5 (less sensitive)
 *   method        - Detection method (default: 'zscore')
 *                   Options: 'zscore', 'modified_zscore', 'iqr'
 *
 * Returns:
 *   Boolean array indicating outliers (true = outlier)
 *
 * Interpretation:
 *   - Z-score > 3.0: ~99.7% confidence outlier
 *   - Z-score > 2.5: ~98.8% confidence outlier
 *   - Z-score > 2.0: ~95.4% confidence outlier
 *
 * Example Usage:
 *   -- Flag outliers for review:
 *   SELECT ctid, embedding
 *   FROM documents
 *   WHERE (detect_outliers_zscore('documents', 'embedding', 3.0))[row_number() OVER ()];
 *
 *   -- Count outliers:
 *   SELECT COUNT(*) FILTER (WHERE is_outlier)
 *   FROM (
 *     SELECT unnest(detect_outliers_zscore('docs', 'vec', 2.5)) AS is_outlier
 *   ) t;
 *
 * Performance:
 *   - Time: O(n*d) for computation + O(n log n) for sorting (robust methods)
 *   - Space: O(n + d)
 *   - Fast enough for millions of vectors
 */
PG_FUNCTION_INFO_V1(detect_outliers_zscore);

Datum
detect_outliers_zscore(PG_FUNCTION_ARGS)
{
	text *table_name = NULL;
	text *vector_column = NULL;
	double		threshold;
	text *method_text = NULL;
	char *tbl_str = NULL;
	char *vec_col_str = NULL;
	char *method = NULL;
	float	  **vectors;
	int			nvec,
				dim;
	double *mean = NULL;
	double *distances = NULL;
	bool *outliers = NULL;
	int			i,
				d;
	ArrayType *result = NULL;
	Datum *result_datums = NULL;
	int16		typlen;
	bool		typbyval;
	char		typalign;

	table_name = PG_GETARG_TEXT_PP(0);
	vector_column = PG_GETARG_TEXT_PP(1);
	threshold = PG_ARGISNULL(2) ? 3.0 : PG_GETARG_FLOAT8(2);
	method_text = PG_ARGISNULL(3) ? cstring_to_text("zscore")
		: PG_GETARG_TEXT_PP(3);

	if (threshold < 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("threshold must be non-negative")));

	tbl_str = text_to_cstring(table_name);
	vec_col_str = text_to_cstring(vector_column);
	method = text_to_cstring(method_text);


	/* Fetch vectors */
	vectors = neurondb_fetch_vectors_from_table(
												tbl_str, vec_col_str, &nvec, &dim);
	if (vectors == NULL || nvec == 0)
	{
		nfree(tbl_str);
		nfree(vec_col_str);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found")));
	}

	if (dim <= 0)
	{
		nfree(tbl_str);
		nfree(vec_col_str);
		/* Free vectors array and rows if vectors is not NULL */
		if (vectors != NULL)
		{
			for (int idx = 0; idx < nvec; idx++)
			{
				if (vectors[idx] != NULL)
					nfree(vectors[idx]);
			}
			nfree(vectors);
		}
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Invalid vector dimension: %d", dim)));
	}

	if (nvec < 2)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Need at least 2 vectors for outlier "
						"detection")));

	/* Compute centroid (mean vector) */
	nalloc(mean, double, dim);
	for (i = 0; i < nvec; i++)
		for (d = 0; d < dim; d++)
			mean[d] += (double) vectors[i][d];

	for (d = 0; d < dim; d++)
		mean[d] /= nvec;

	/* Compute distances from mean */
	nalloc(distances, double, nvec);
	for (i = 0; i < nvec; i++)
		distances[i] = distance_from_mean(vectors[i], mean, dim);

	/* Allocate outlier flags */
	nalloc(outliers, bool, nvec);

	/* Apply detection method */
	if (strcmp(method, "zscore") == 0)
	{
		/* Standard Z-score method */
		double		mean_dist = 0.0;
		double		std_dist = 0.0;

		/* Compute mean distance */
		for (i = 0; i < nvec; i++)
			mean_dist += distances[i];
		mean_dist /= nvec;

		/* Compute standard deviation */
		for (i = 0; i < nvec; i++)
		{
			double		diff = distances[i] - mean_dist;

			std_dist += diff * diff;
		}
		std_dist = sqrt(std_dist / nvec);

		if (std_dist < 1e-10)
		{
			/* All points identical - no outliers */
		}
		else
		{
			/* Flag outliers: distance > mean + threshold * std */
			double		outlier_threshold =
				mean_dist + threshold * std_dist;

			for (i = 0; i < nvec; i++)
				outliers[i] =
					(distances[i] > outlier_threshold);
		}
	}
	else if (strcmp(method, "modified_zscore") == 0
			 || strcmp(method, "robust") == 0)
	{
		/* Modified Z-score using median and MAD (more robust) */
		double		median_dist;
		double		mad;		/* Median Absolute Deviation */
		double *sorted_distances = NULL;
		double *abs_deviations = NULL;

		/* Compute median distance */
		nalloc(sorted_distances, double, nvec);
		memcpy(sorted_distances, distances, sizeof(double) * nvec);
		qsort(sorted_distances, nvec, sizeof(double), double_compare);

		median_dist = (nvec % 2 == 0)
			? (sorted_distances[nvec / 2 - 1]
			   + sorted_distances[nvec / 2])
			/ 2.0
			: sorted_distances[nvec / 2];

		/* Compute MAD */
		nalloc(abs_deviations, double, nvec);
		for (i = 0; i < nvec; i++)
			abs_deviations[i] = fabs(distances[i] - median_dist);

		qsort(abs_deviations, nvec, sizeof(double), double_compare);
		mad = (nvec % 2 == 0) ? (abs_deviations[nvec / 2 - 1]
								 + abs_deviations[nvec / 2])
			/ 2.0
			: abs_deviations[nvec / 2];

		if (mad < 1e-10)
		{
			mad = 1.0;			/* Fallback to avoid division by zero */
		}

		/* Modified Z-score: M_i = 0.6745 * (x_i - median) / MAD */
		for (i = 0; i < nvec; i++)
		{
			double		modified_z =
				0.6745 * fabs(distances[i] - median_dist) / mad;

			outliers[i] = (modified_z > threshold);
		}

		nfree(sorted_distances);
		nfree(abs_deviations);
	}
	else if (strcmp(method, "iqr") == 0)
	{
		/* IQR (Interquartile Range) method */
		double		q1,
					q3,
					iqr;
		double		lower_bound,
					upper_bound;
		double *sorted_distances = NULL;
		int			q1_idx,
					q3_idx;

		nalloc(sorted_distances, double, nvec);
		memcpy(sorted_distances, distances, sizeof(double) * nvec);
		qsort(sorted_distances, nvec, sizeof(double), double_compare);

		/* Compute Q1 and Q3 */
		q1_idx = nvec / 4;
		q3_idx = (3 * nvec) / 4;
		q1 = sorted_distances[q1_idx];
		q3 = sorted_distances[q3_idx];
		iqr = q3 - q1;

		/* Outlier bounds: Q1 - 1.5*IQR and Q3 + 1.5*IQR */
		lower_bound = q1 - threshold * iqr;
		upper_bound = q3 + threshold * iqr;

		for (i = 0; i < nvec; i++)
		{
			outliers[i] = (distances[i] < lower_bound
						   || distances[i] > upper_bound);
		}

		nfree(sorted_distances);
	}
	else
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Unknown method '%s'. Use 'zscore', 'modified_zscore', or 'iqr'",
						method)));
	}

	/* Build result array */
	nalloc(result_datums, Datum, nvec);
	for (i = 0; i < nvec; i++)
		result_datums[i] = BoolGetDatum(outliers[i]);

	get_typlenbyvalalign(BOOLOID, &typlen, &typbyval, &typalign);
	result = construct_array(
							 result_datums, nvec, BOOLOID, typlen, typbyval, typalign);

	for (i = 0; i < nvec; i++)
		nfree(vectors[i]);
	nfree(vectors);
	nfree(mean);
	nfree(distances);
	nfree(outliers);
	nfree(result_datums);
	nfree(tbl_str);
	nfree(vec_col_str);
	nfree(method);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * compute_outlier_scores
 * ----------------------
 * Return numeric outlier scores instead of boolean flags.
 *
 * Returns:
 *   Float8 array of outlier scores (Z-scores or modified Z-scores)
 *   Higher values = more outlier-like
 *
 * Useful for:
 *   - Ranking by "outlierness"
 *   - Dynamic threshold adjustment
 *   - Visualization of outlier distribution
 */
PG_FUNCTION_INFO_V1(compute_outlier_scores);

Datum
compute_outlier_scores(PG_FUNCTION_ARGS)
{
	text *table_name = NULL;
	text *vector_column = NULL;
	text *method_text = NULL;
	char *tbl_str = NULL;
	char *vec_col_str = NULL;
	char *method = NULL;
	float	  **vectors;
	int			nvec,
				dim;
	double *mean = NULL;
	double *distances = NULL;
	double *scores = NULL;
	int			i,
				d;
	ArrayType *result = NULL;
	Datum *result_datums = NULL;

	table_name = PG_GETARG_TEXT_PP(0);
	vector_column = PG_GETARG_TEXT_PP(1);
	method_text = PG_ARGISNULL(2) ? cstring_to_text("zscore")
		: PG_GETARG_TEXT_PP(2);

	tbl_str = text_to_cstring(table_name);
	vec_col_str = text_to_cstring(vector_column);
	method = text_to_cstring(method_text);

	/* Fetch vectors */
	vectors = neurondb_fetch_vectors_from_table(
												tbl_str, vec_col_str, &nvec, &dim);
	if (vectors == NULL || nvec == 0)
	{
		nfree(tbl_str);
		nfree(vec_col_str);
		nfree(method);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found")));
	}

	if (dim <= 0)
	{
		nfree(tbl_str);
		nfree(vec_col_str);
		nfree(method);
		/* Free vectors array and rows if vectors is not NULL */
		if (vectors != NULL)
		{
			for (int idx = 0; idx < nvec; idx++)
			{
				if (vectors[idx] != NULL)
					nfree(vectors[idx]);
			}
			nfree(vectors);
		}
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Invalid vector dimension: %d", dim)));
	}

	if (nvec < 2)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Need at least 2 vectors")));

	/* Compute mean */
	nalloc(mean, double, dim);
	for (i = 0; i < nvec; i++)
		for (d = 0; d < dim; d++)
			mean[d] += (double) vectors[i][d];
	for (d = 0; d < dim; d++)
		mean[d] /= nvec;

	/* Compute distances */
	nalloc(distances, double, nvec);
	for (i = 0; i < nvec; i++)
		distances[i] = distance_from_mean(vectors[i], mean, dim);

	/* Compute scores */
	nalloc(scores, double, nvec);

	if (strcmp(method, "zscore") == 0)
	{
		double		mean_dist = 0.0;
		double		std_dist = 0.0;

		for (i = 0; i < nvec; i++)
			mean_dist += distances[i];
		mean_dist /= nvec;

		for (i = 0; i < nvec; i++)
		{
			double		diff = distances[i] - mean_dist;

			std_dist += diff * diff;
		}
		std_dist = sqrt(std_dist / nvec);

		if (std_dist < 1e-10)
		{
			for (i = 0; i < nvec; i++)
				scores[i] = 0.0;
		}
		else
		{
			for (i = 0; i < nvec; i++)
				scores[i] =
					(distances[i] - mean_dist) / std_dist;
		}
	}
	else if (strcmp(method, "modified_zscore") == 0)
	{
		double		median_dist;
		double		mad;
		double *sorted_distances = NULL;
		double *abs_deviations = NULL;

		nalloc(sorted_distances, double, nvec);
		memcpy(sorted_distances, distances, sizeof(double) * nvec);
		qsort(sorted_distances, nvec, sizeof(double), double_compare);

		median_dist = (nvec % 2 == 0)
			? (sorted_distances[nvec / 2 - 1]
			   + sorted_distances[nvec / 2])
			/ 2.0
			: sorted_distances[nvec / 2];

		nalloc(abs_deviations, double, nvec);
		for (i = 0; i < nvec; i++)
			abs_deviations[i] = fabs(distances[i] - median_dist);

		qsort(abs_deviations, nvec, sizeof(double), double_compare);
		mad = (nvec % 2 == 0) ? (abs_deviations[nvec / 2 - 1]
								 + abs_deviations[nvec / 2])
			/ 2.0
			: abs_deviations[nvec / 2];

		if (mad < 1e-10)
			mad = 1.0;

		for (i = 0; i < nvec; i++)
			scores[i] =
				0.6745 * fabs(distances[i] - median_dist) / mad;

		nfree(sorted_distances);
		nfree(abs_deviations);
	}
	else
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Unknown method '%s'", method)));
	}

	nalloc(result_datums, Datum, nvec);
	for (i = 0; i < nvec; i++)
		result_datums[i] = Float8GetDatum(scores[i]);

	result = construct_array(result_datums,
							 nvec,
							 FLOAT8OID,
							 sizeof(float8),
							 FLOAT8PASSBYVAL,
							 'd');

	for (i = 0; i < nvec; i++)
		nfree(vectors[i]);
	nfree(vectors);
	nfree(mean);
	nfree(distances);
	nfree(scores);
	nfree(result_datums);
	nfree(tbl_str);
	nfree(vec_col_str);
	nfree(method);

	PG_RETURN_ARRAYTYPE_P(result);
}
