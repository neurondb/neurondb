/*-------------------------------------------------------------------------
 *
 * ml_drift_detection.c
 *    Embedding drift detection.
 *
 * This module detects embedding drift by comparing distributions over time
 * using centroid shift, covariance change, and KL divergence methods.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_drift_detection.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"
#include "access/htup_details.h"

#include "neurondb.h"
#include "neurondb_ml.h"

#include <math.h>
#include <float.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/*
 * vector_distance
 *
 * Compute Euclidean distance between two vectors of dimension 'dim'.
 */
static inline double
vector_distance(const double *a, const double *b, int dim)
{
	double		sum = 0.0;
	int			i;

	for (i = 0; i < dim; i++)
	{
		double		diff = a[i] - b[i];

		sum += diff * diff;
	}
	return sqrt(sum);
}

/*
 * detect_centroid_drift
 * ---------------------
 * Detect embedding drift by comparing centroids between two datasets.
 *
 * Input parameters:
 *   baseline_table    - Baseline/reference table
 *   baseline_column   - Baseline vector column
 *   current_table     - Current/test table
 *   current_column    - Current vector column
 *
 * Returns:
 *   Record with (drift_distance FLOAT8, normalized_drift FLOAT8, is_significant BOOLEAN)
 *
 * Notes:
 *   - Requires at least 10 vectors in each dataset
 *   - Dimensions must match
 */
PG_FUNCTION_INFO_V1(detect_centroid_drift);

Datum
detect_centroid_drift(PG_FUNCTION_ARGS)
{
	text *baseline_table = NULL;
	text *baseline_column = NULL;
	text *current_table = NULL;
	text *current_column = NULL;
	char *baseline_tbl = NULL;
	char *baseline_col = NULL;
	char *current_tbl = NULL;
	char *current_col = NULL;
	float	  **baseline_vecs;
	float	  **current_vecs;
	int			n_baseline,
				n_current;
	int			dim_baseline,
				dim_current;
	double *baseline_mean = NULL;
	double *current_mean = NULL;
	double *baseline_std = NULL;
	double		drift_distance;
	double		normalized_drift;
	double		avg_std;
	bool		is_significant;
	int			i,
				d;
	TupleDesc	tupdesc;
	Datum		values[3];
	bool		nulls[3];
	HeapTuple	tuple;

	baseline_table = PG_GETARG_TEXT_PP(0);
	baseline_column = PG_GETARG_TEXT_PP(1);
	current_table = PG_GETARG_TEXT_PP(2);
	current_column = PG_GETARG_TEXT_PP(3);

	baseline_tbl = text_to_cstring(baseline_table);
	baseline_col = text_to_cstring(baseline_column);
	current_tbl = text_to_cstring(current_table);
	current_col = text_to_cstring(current_column);


	/* Fetch vectors */
	baseline_vecs = neurondb_fetch_vectors_from_table(
													  baseline_tbl, baseline_col, &n_baseline, &dim_baseline);
	current_vecs = neurondb_fetch_vectors_from_table(
													 current_tbl, current_col, &n_current, &dim_current);

	if (baseline_vecs == NULL || n_baseline == 0 || current_vecs == NULL || n_current == 0)
	{
		nfree(baseline_tbl);
		nfree(baseline_col);
		nfree(current_tbl);
		nfree(current_col);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found in one or both datasets")));
	}

	if (dim_baseline <= 0 || dim_current <= 0)
	{
		nfree(baseline_tbl);
		nfree(baseline_col);
		nfree(current_tbl);
		nfree(current_col);
		/* Free vectors arrays if not NULL */
		if (baseline_vecs != NULL)
		{
			for (int idx = 0; idx < n_baseline; idx++)
			{
				if (baseline_vecs[idx] != NULL)
					nfree(baseline_vecs[idx]);
			}
			nfree(baseline_vecs);
		}
		if (current_vecs != NULL)
		{
			for (int idx = 0; idx < n_current; idx++)
			{
				if (current_vecs[idx] != NULL)
					nfree(current_vecs[idx]);
			}
			nfree(current_vecs);
		}
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Invalid vector dimensions: baseline=%d, current=%d", dim_baseline, dim_current)));
	}

	if (n_baseline < 10 || n_current < 10)
	{
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Need at least 10 vectors in each dataset (baseline=%d, current=%d)",
						n_baseline, n_current)));
	}

	if (dim_baseline != dim_current)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Dimension mismatch: baseline=%d, current=%d",
						dim_baseline, dim_current)));

	/* Compute baseline centroid and standard deviation */
	baseline_mean = (double *) palloc0(sizeof(double) * dim_baseline);
	baseline_std = (double *) palloc0(sizeof(double) * dim_baseline);

	for (i = 0; i < n_baseline; i++)
	{
		for (d = 0; d < dim_baseline; d++)
			baseline_mean[d] += (double) baseline_vecs[i][d];
	}
	for (d = 0; d < dim_baseline; d++)
		baseline_mean[d] /= n_baseline;

	/* Compute per-dimension standard deviation for baseline */
	for (i = 0; i < n_baseline; i++)
	{
		for (d = 0; d < dim_baseline; d++)
		{
			double		diff = (double) baseline_vecs[i][d] - baseline_mean[d];

			baseline_std[d] += diff * diff;
		}
	}

	for (d = 0; d < dim_baseline; d++)
		baseline_std[d] = sqrt(baseline_std[d] / n_baseline);

	/* Compute average std deviation across dimensions */
	avg_std = 0.0;
	for (d = 0; d < dim_baseline; d++)
		avg_std += baseline_std[d];
	avg_std /= dim_baseline;

	if (avg_std < 1e-10)
		avg_std = 1.0;			/* Avoid division by zero */

	/* Compute current centroid */
	current_mean = (double *) palloc0(sizeof(double) * dim_current);

	for (i = 0; i < n_current; i++)
	{
		for (d = 0; d < dim_current; d++)
			current_mean[d] += (double) current_vecs[i][d];
	}
	for (d = 0; d < dim_current; d++)
		current_mean[d] /= n_current;

	/* Compute drift distance */
	drift_distance = vector_distance(baseline_mean, current_mean, dim_baseline);
	normalized_drift = drift_distance / avg_std;
	is_significant = (normalized_drift > 3.0);


	/* Build result tuple */
	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("Function returning record called in context that cannot accept type record")));

	tupdesc = BlessTupleDesc(tupdesc);

	values[0] = Float8GetDatum(drift_distance);
	values[1] = Float8GetDatum(normalized_drift);
	values[2] = BoolGetDatum(is_significant);
	nulls[0] = false;
	nulls[1] = false;
	nulls[2] = false;

	tuple = heap_form_tuple(tupdesc, values, nulls);

	for (i = 0; i < n_baseline; i++)
		nfree(baseline_vecs[i]);
	for (i = 0; i < n_current; i++)
		nfree(current_vecs[i]);
	nfree(baseline_vecs);
	nfree(current_vecs);
	nfree(baseline_mean);
	nfree(current_mean);
	nfree(baseline_std);
	nfree(baseline_tbl);
	nfree(baseline_col);
	nfree(current_tbl);
	nfree(current_col);

	PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}

/*
 * compute_distribution_divergence
 * ------------------------------
 * Approximate KL divergence between two embedding distributions.
 * Assumes multivariate Gaussian distributions and computes simplified divergence.
 * Returns positive value where higher = more divergence.
 */
PG_FUNCTION_INFO_V1(compute_distribution_divergence);

Datum
compute_distribution_divergence(PG_FUNCTION_ARGS)
{
	text *baseline_table = NULL;
	text *baseline_column = NULL;
	text *current_table = NULL;
	text *current_column = NULL;
	char *baseline_tbl = NULL;
	char *baseline_col = NULL;
	char *current_tbl = NULL;
	char *current_col = NULL;
	float	  **baseline_vecs;
	float	  **current_vecs;
	int			n_baseline,
				n_current;
	int			dim_baseline,
				dim_current;
	double *baseline_mean = NULL;
	double *current_mean = NULL;
	double *baseline_var = NULL;
	double *current_var = NULL;
	double		divergence;
	int			i,
				d;

	baseline_table = PG_GETARG_TEXT_PP(0);
	baseline_column = PG_GETARG_TEXT_PP(1);
	current_table = PG_GETARG_TEXT_PP(2);
	current_column = PG_GETARG_TEXT_PP(3);

	baseline_tbl = text_to_cstring(baseline_table);
	baseline_col = text_to_cstring(baseline_column);
	current_tbl = text_to_cstring(current_table);
	current_col = text_to_cstring(current_column);

	/* Fetch vectors */
	baseline_vecs = neurondb_fetch_vectors_from_table(
													  baseline_tbl, baseline_col, &n_baseline, &dim_baseline);
	current_vecs = neurondb_fetch_vectors_from_table(
													 current_tbl, current_col, &n_current, &dim_current);

	if (baseline_vecs == NULL || n_baseline == 0 || current_vecs == NULL || n_current == 0)
	{
		nfree(baseline_tbl);
		nfree(baseline_col);
		nfree(current_tbl);
		nfree(current_col);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found in one or both datasets")));
	}

	if (dim_baseline <= 0 || dim_current <= 0)
	{
		nfree(baseline_tbl);
		nfree(baseline_col);
		nfree(current_tbl);
		nfree(current_col);
		/* Free vectors arrays if not NULL */
		if (baseline_vecs != NULL)
		{
			for (int idx = 0; idx < n_baseline; idx++)
			{
				if (baseline_vecs[idx] != NULL)
					nfree(baseline_vecs[idx]);
			}
			nfree(baseline_vecs);
		}
		if (current_vecs != NULL)
		{
			for (int idx = 0; idx < n_current; idx++)
			{
				if (current_vecs[idx] != NULL)
					nfree(current_vecs[idx]);
			}
			nfree(current_vecs);
		}
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Invalid vector dimensions: baseline=%d, current=%d", dim_baseline, dim_current)));
	}

	if (n_baseline < 10 || n_current < 10)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Need at least 10 vectors in each dataset")));

	if (dim_baseline != dim_current)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Dimension mismatch")));

	/* Compute means */
	baseline_mean = (double *) palloc0(sizeof(double) * dim_baseline);
	current_mean = (double *) palloc0(sizeof(double) * dim_current);

	for (i = 0; i < n_baseline; i++)
	{
		for (d = 0; d < dim_baseline; d++)
			baseline_mean[d] += (double) baseline_vecs[i][d];
	}
	for (d = 0; d < dim_baseline; d++)
		baseline_mean[d] /= n_baseline;

	for (i = 0; i < n_current; i++)
	{
		for (d = 0; d < dim_current; d++)
			current_mean[d] += (double) current_vecs[i][d];
	}
	for (d = 0; d < dim_current; d++)
		current_mean[d] /= n_current;

	/* Compute variances */
	baseline_var = (double *) palloc0(sizeof(double) * dim_baseline);
	current_var = (double *) palloc0(sizeof(double) * dim_current);

	for (i = 0; i < n_baseline; i++)
	{
		for (d = 0; d < dim_baseline; d++)
		{
			double		diff = (double) baseline_vecs[i][d] - baseline_mean[d];

			baseline_var[d] += diff * diff;
		}
	}
	for (d = 0; d < dim_baseline; d++)
		baseline_var[d] /= n_baseline;

	for (i = 0; i < n_current; i++)
	{
		for (d = 0; d < dim_current; d++)
		{
			double		diff = (double) current_vecs[i][d] - current_mean[d];

			current_var[d] += diff * diff;
		}
	}
	for (d = 0; d < dim_current; d++)
		current_var[d] /= n_current;

	/* Approximate KL divergence (simplified, per-dimension) */
	divergence = 0.0;
	for (d = 0; d < dim_baseline; d++)
	{
		double		mean_diff = baseline_mean[d] - current_mean[d];
		double		var_ratio;

		if (baseline_var[d] < 1e-10 || current_var[d] < 1e-10)
			continue;

		var_ratio = current_var[d] / baseline_var[d];

		/*
		 * KL(P||Q) ≈ 0.5 * [log(σ_q²/σ_p²) + σ_p²/σ_q² +
		 * (μ_p-μ_q)²/σ_q² - 1]
		 */
		divergence += 0.5 *
			(log(var_ratio) + 1.0 / var_ratio +
			 mean_diff * mean_diff / current_var[d] - 1.0);
	}

	for (i = 0; i < n_baseline; i++)
		nfree(baseline_vecs[i]);
	for (i = 0; i < n_current; i++)
		nfree(current_vecs[i]);
	nfree(baseline_vecs);
	nfree(current_vecs);
	nfree(baseline_mean);
	nfree(current_mean);
	nfree(baseline_var);
	nfree(current_var);
	nfree(baseline_tbl);
	nfree(baseline_col);
	nfree(current_tbl);
	nfree(current_col);

	PG_RETURN_FLOAT8(divergence);
}
