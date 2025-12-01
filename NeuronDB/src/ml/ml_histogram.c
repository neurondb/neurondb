/*-------------------------------------------------------------------------
 *
 * ml_histogram.c
 *    Similarity histogram and distance distribution analysis.
 *
 * This module analyzes distance and similarity distributions in embedding
 * spaces to provide insights into data quality and cluster structure.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_histogram.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "access/htup_details.h"

#include "neurondb.h"
#include "neurondb_ml.h"

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

#define MAX_HISTOGRAM_BINS		100

/*
 * double_cmp
 *	 Comparison function for qsort for doubles.
 */
static int
double_cmp(const void *a, const void *b)
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
 * euclidean_distance
 *	 Compute the Euclidean distance between two float vectors.
 */
static inline double
euclidean_distance(const float *a, const float *b, int dim)
{
	double		sum = 0.0;
	int			i;

	for (i = 0; i < dim; i++)
	{
		double		diff = (double) a[i] - (double) b[i];

		sum += diff * diff;
	}

	return sqrt(sum);
}

PG_FUNCTION_INFO_V1(similarity_histogram);

Datum
similarity_histogram(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *vector_column;
	int			num_samples;
	char	   *tbl_str;
	char	   *vec_col_str;
	float	  **vectors;
	int			nvec;
	int			dim;
	double	   *distances;
	double		min_dist;
	double		max_dist;
	double		mean_dist;
	double		stddev_dist;
	double		p50;
	double		p90;
	double		p95;
	double		p99;
	int			i;
	TupleDesc	tupdesc;
	Datum		values[9];
	bool		nulls[9];
	HeapTuple	tuple;

	/* Parse arguments */
	table_name = PG_GETARG_TEXT_PP(0);
	vector_column = PG_GETARG_TEXT_PP(1);
	num_samples = PG_ARGISNULL(2) ? 1000 : PG_GETARG_INT32(2);

	if (num_samples < 10 || num_samples > 100000)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("num_samples must be between 10 and 100000")));

	tbl_str = text_to_cstring(table_name);
	vec_col_str = text_to_cstring(vector_column);

	elog(DEBUG1,
		 "neurondb: Computing similarity histogram (%d samples)",
		 num_samples);

	/* Fetch vectors */
	vectors = neurondb_fetch_vectors_from_table(tbl_str, vec_col_str, &nvec, &dim);
	if (nvec < 2)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Need at least 2 vectors")));

	/* Sample random pairs and compute distances */
	distances = (double *) palloc(sizeof(double) * num_samples);

	for (i = 0; i < num_samples; i++)
	{
		int			idx1 = rand() % nvec;
		int			idx2 = rand() % nvec;

		while (idx2 == idx1)
			idx2 = rand() % nvec;

		distances[i] = euclidean_distance(vectors[idx1], vectors[idx2], dim);
	}

	/* Sort distances for percentile calculation */
	qsort(distances, num_samples, sizeof(double), double_cmp);

	/* Compute statistics */
	min_dist = distances[0];
	max_dist = distances[num_samples - 1];

	mean_dist = 0.0;
	for (i = 0; i < num_samples; i++)
		mean_dist += distances[i];
	mean_dist /= (double) num_samples;

	stddev_dist = 0.0;
	for (i = 0; i < num_samples; i++)
	{
		double		diff = distances[i] - mean_dist;

		stddev_dist += diff * diff;
	}
	stddev_dist = sqrt(stddev_dist / (double) num_samples);

	/* Compute percentiles (simple empirical method) */
	p50 = distances[(int) (num_samples * 0.50)];
	p90 = distances[(int) (num_samples * 0.90)];
	p95 = distances[(int) (num_samples * 0.95)];
	p99 = distances[(int) (num_samples * 0.99)];

	elog(DEBUG1,
		 "neurondb: Distance stats: min=%.4f, max=%.4f, mean=%.4f, p50=%.4f",
		 min_dist, max_dist, mean_dist, p50);

	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("function returning record called in context that cannot "
						"accept type record")));

	tupdesc = BlessTupleDesc(tupdesc);

	values[0] = Float8GetDatum(min_dist);
	values[1] = Float8GetDatum(max_dist);
	values[2] = Float8GetDatum(mean_dist);
	values[3] = Float8GetDatum(stddev_dist);
	values[4] = Float8GetDatum(p50);
	values[5] = Float8GetDatum(p90);
	values[6] = Float8GetDatum(p95);
	values[7] = Float8GetDatum(p99);
	values[8] = Int32GetDatum(num_samples);

	for (i = 0; i < 9; i++)
		nulls[i] = false;

	tuple = heap_form_tuple(tupdesc, values, nulls);

	/* Cleanup */
	for (i = 0; i < nvec; i++)
		NDB_FREE(vectors[i]);
	NDB_FREE(vectors);
	NDB_FREE(distances);
	NDB_FREE(tbl_str);
	NDB_FREE(vec_col_str);

	PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}
