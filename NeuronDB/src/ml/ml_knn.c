/*-------------------------------------------------------------------------
 *
 * ml_knn.c
 *    K-nearest neighbors implementation.
 *
 * This module implements KNN for classification and regression using
 * instance-based learning with Euclidean distance.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_knn.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "utils/array.h"
#include "utils/jsonb.h"
#include "common/jsonapi.h"
#include "utils/memutils.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "ml_catalog.h"
#include "neurondb_cuda_knn.h"
#include "vector/vector_types.h"
#include "ml_gpu_registry.h"
#include "neurondb_gpu.h"
#include "neurondb_validation.h"
#include "neurondb_spi_safe.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"
#include "neurondb_sql.h"
#include "neurondb_json.h"
#include "utils/elog.h"
#ifdef NDB_GPU_CUDA
#include "neurondb_gpu_model.h"
#endif

#ifdef NDB_GPU_CUDA
#include "neurondb_cuda_runtime.h"
#include <cublas_v2.h>
extern cublasHandle_t ndb_cuda_get_cublas_handle(void);
#endif

#include <math.h>
#include <float.h>

typedef struct
{
	float	   *features;
	double		label;
	double		distance;
	int			dim;
}			KNNSample;

static int
compare_samples(const void *a, const void *b)
{
	const		KNNSample *sa = (const KNNSample *) a;
	const		KNNSample *sb = (const KNNSample *) b;

	if (sa->distance < sb->distance)
		return -1;
	if (sa->distance > sb->distance)
		return 1;
	return 0;
}

static double
euclidean_distance(const float *a, const float *b, int dim)
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
 * knn_classify
 *
 * Classifies a single sample using K-Nearest Neighbors
 * Returns class label (0 or 1 for binary classification)
 */
PG_FUNCTION_INFO_V1(knn_classify);

Datum
knn_classify(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *feature_col;
	text	   *label_col;
	Vector	   *query_vector;
	int			k;
	char	   *tbl_str;
	char	   *feat_str;
	char	   *label_str;
	StringInfoData query = {0};
	NDB_DECLARE (NdbSpiSession *, classify_spi_session);
	int			ret;
	int			nvec = 0;
	int			dim;
	KNNSample  *samples = NULL;
	double		class_votes[2] = {0.0, 0.0};
	int			predicted_class;
	int			i;
	MemoryContext oldcontext;

	/* Input validation */
	NDB_CHECK_NULL_ARG(0, "table_name");
	NDB_CHECK_NULL_ARG(1, "feature_col");
	NDB_CHECK_NULL_ARG(2, "label_col");
	NDB_CHECK_NULL_ARG(3, "query_vector");
	NDB_CHECK_NULL_ARG(4, "k");

	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	label_col = PG_GETARG_TEXT_PP(2);
	query_vector = PG_GETARG_VECTOR_P(3);
	k = PG_GETARG_INT32(4);

	/* Validate vector structure */
	NDB_CHECK_VECTOR_VALID(query_vector);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	label_str = text_to_cstring(label_col);

	dim = query_vector->dim;

	if (k < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: knn_classify: k must be at least 1, got %d", k),
				 errdetail("K parameter is %d, minimum allowed is 1", k),
				 errhint("Specify a positive integer for the number of nearest neighbors.")));

	oldcontext = CurrentMemoryContext;
	Assert(oldcontext != NULL);
	NDB_SPI_SESSION_BEGIN(classify_spi_session, oldcontext);

	initStringInfo(&query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 quote_identifier(feat_str),
					 quote_identifier(label_str),
					 quote_identifier(tbl_str),
					 quote_identifier(feat_str),
					 quote_identifier(label_str));
	elog(DEBUG1, "knn_classify: executing query: %s", query.data);

	ret = ndb_spi_execute(classify_spi_session, query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		NDB_FREE(tbl_str);
		tbl_str = NULL;
		NDB_FREE(feat_str);
		feat_str = NULL;
		NDB_FREE(label_str);
		label_str = NULL;
		NDB_FREE(query.data);
		query.data = NULL;
		NDB_SPI_SESSION_END(classify_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: knn_classify: query failed"),
				 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
				 errhint("Verify the table exists and contains valid feature and label columns.")));
	}

	NDB_CHECK_SPI_TUPTABLE();

	nvec = SPI_processed;

	if (nvec < k)
	{
		NDB_FREE(query.data);
		NDB_SPI_SESSION_END(classify_spi_session);
		NDB_FREE(tbl_str);
		tbl_str = NULL;
		NDB_FREE(feat_str);
		feat_str = NULL;
		NDB_FREE(label_str);
		label_str = NULL;
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: knn_classify: need at least %d samples, got %d", k, nvec),
				 errdetail("Dataset contains %d rows, but k=%d requires at least %d rows", nvec, k, k),
				 errhint("Reduce k or add more training data.")));
	}

	MemoryContextSwitchTo(oldcontext);
	NDB_ALLOC(samples, KNNSample, nvec);

	{
		int			nsamples = 0;

		for (i = 0; i < nvec; i++)
		{
			HeapTuple	tuple;
			TupleDesc	tupdesc;
			Datum		feat_datum;
			Datum		label_datum;
			bool		feat_null;
			bool		label_null;
			Vector	   *vec;
			/* Safe access to SPI_tuptable - validate before access */
			if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || 
				i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
			{
				continue;
			}
			tuple = SPI_tuptable->vals[i];
			tupdesc = SPI_tuptable->tupdesc;
			if (tupdesc == NULL)
			{
				continue;
			}

			feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
			/* Safe access for label - validate tupdesc has at least 2 columns */
			if (tupdesc->natts < 2)
			{
				continue;
			}
			label_datum = SPI_getbinval(tuple, tupdesc, 2, &label_null);

			if (feat_null || label_null)
				continue;

			vec = DatumGetVector(feat_datum);
			if (vec == NULL || vec->dim <= 0)
				continue;

			if (vec->dim != dim)
			{
				int			j;

				NDB_FREE(query.data);
				NDB_SPI_SESSION_END(classify_spi_session);
				for (j = 0; j < nsamples; j++)
				{
					NDB_FREE(samples[j].features);
					samples[j].features = NULL;
				}
				NDB_FREE(samples);
				samples = NULL;
				NDB_FREE(tbl_str);
				tbl_str = NULL;
				NDB_FREE(feat_str);
				feat_str = NULL;
				NDB_FREE(label_str);
				label_str = NULL;
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: knn_classify: dimension mismatch: expected %d, got %d",
								dim, vec->dim)));
			}

			/* Copy features */
			NDB_ALLOC(samples[nsamples].features, float, dim);
			memcpy(samples[nsamples].features, vec->data, sizeof(float) * dim);
			samples[nsamples].dim = dim;

			samples[nsamples].label = DatumGetFloat8(label_datum);

			/* Compute distance */
			samples[nsamples].distance = euclidean_distance(
															query_vector->data, samples[nsamples].features, dim);
			nsamples++;
		}

		if (nsamples < k)
		{
			NDB_FREE(query.data);
			NDB_SPI_SESSION_END(classify_spi_session);
			for (i = 0; i < nsamples; i++)
			{
				NDB_FREE(samples[i].features);
				samples[i].features = NULL;
			}
			NDB_FREE(samples);
			samples = NULL;
			NDB_FREE(tbl_str);
			tbl_str = NULL;
			NDB_FREE(feat_str);
			feat_str = NULL;
			NDB_FREE(label_str);
			label_str = NULL;
			ereport(ERROR,
					(errcode(ERRCODE_INSUFFICIENT_RESOURCES),
					 errmsg("neurondb: need at least k=%d valid samples, but found only %d",
							k,
							nsamples)));
		}

		NDB_FREE(query.data);
		NDB_SPI_SESSION_END(classify_spi_session);

		qsort(samples, nsamples, sizeof(KNNSample), compare_samples);

		for (i = 0; i < k && i < nsamples; i++)
		{
			int			label_class = (int) samples[i].label;

			if (label_class >= 0 && label_class < 2)
				class_votes[label_class] += 1.0;
		}

		predicted_class = (class_votes[1] > class_votes[0]) ? 1 : 0;

		for (i = 0; i < nsamples; i++)
		{
			NDB_FREE(samples[i].features);
			samples[i].features = NULL;
		}
		NDB_FREE(samples);
		samples = NULL;
	}
	NDB_FREE(tbl_str);
	tbl_str = NULL;
	NDB_FREE(feat_str);
	feat_str = NULL;
	NDB_FREE(label_str);
	label_str = NULL;

	PG_RETURN_INT32(predicted_class);
}

/*
 * knn_regress
 *
 * Predicts a continuous value using K-Nearest Neighbors
 * Returns average of k nearest neighbors' values
 */
PG_FUNCTION_INFO_V1(knn_regress);

Datum
knn_regress(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *feature_col;
	text	   *target_col;
	Vector	   *query_vector;
	int			k;
	char	   *tbl_str;
	char	   *feat_str;
	char	   *targ_str;
	StringInfoData query = {0};
	NDB_DECLARE (NdbSpiSession *, regress_spi_session);
	int			ret;
	int			nvec = 0;
	int			dim;
	KNNSample  *samples = NULL;
	double		prediction = 0.0;
	int			i;
	MemoryContext oldcontext;

	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	target_col = PG_GETARG_TEXT_PP(2);
	query_vector = PG_GETARG_VECTOR_P(3);
	NDB_CHECK_VECTOR_VALID(query_vector);
	k = PG_GETARG_INT32(4);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(target_col);

	dim = query_vector->dim;

	if (k < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: knn_regress: k must be at least 1, got %d", k),
				 errdetail("K parameter is %d, minimum allowed is 1", k),
				 errhint("Specify a positive integer for the number of nearest neighbors.")));

	oldcontext = CurrentMemoryContext;
	Assert(oldcontext != NULL);
	NDB_SPI_SESSION_BEGIN(regress_spi_session, oldcontext);

	initStringInfo(&query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 quote_identifier(feat_str),
					 quote_identifier(targ_str),
					 quote_identifier(tbl_str),
					 quote_identifier(feat_str),
					 quote_identifier(targ_str));
	elog(DEBUG1, "knn_regress: executing query: %s", query.data);

	ret = ndb_spi_execute(regress_spi_session, query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT)
	{
		NDB_FREE(query.data);
		NDB_FREE(tbl_str);
		tbl_str = NULL;
		NDB_FREE(feat_str);
		feat_str = NULL;
		NDB_FREE(targ_str);
		targ_str = NULL;
		NDB_SPI_SESSION_END(regress_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: knn_regress: query failed"),
				 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
				 errhint("Verify the table exists and contains valid feature and target columns.")));
	}

	nvec = SPI_processed;

	if (nvec < k)
	{
		NDB_FREE(query.data);
		NDB_SPI_SESSION_END(regress_spi_session);
		NDB_FREE(tbl_str);
		tbl_str = NULL;
		NDB_FREE(feat_str);
		feat_str = NULL;
		NDB_FREE(targ_str);
		targ_str = NULL;
		ereport(ERROR,
				(errcode(ERRCODE_INSUFFICIENT_RESOURCES),
				 errmsg("neurondb: need at least k=%d samples, but found only %d",
						k,
						nvec)));
	}

	MemoryContextSwitchTo(oldcontext);
	NDB_ALLOC(samples, KNNSample, nvec);

	{
		int			nsamples = 0;

		for (i = 0; i < nvec; i++)
		{
			HeapTuple	tuple;
			TupleDesc	tupdesc;
			Datum		feat_datum;
			Datum		targ_datum;
			bool		feat_null;
			bool		targ_null;
			Vector	   *vec;
			/* Safe access to SPI_tuptable - validate before access */
			if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || 
				i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
			{
				continue;
			}
			tuple = SPI_tuptable->vals[i];
			tupdesc = SPI_tuptable->tupdesc;
			if (tupdesc == NULL)
			{
				continue;
			}

			feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
			/* Safe access for target - validate tupdesc has at least 2 columns */
			if (tupdesc->natts < 2)
			{
				continue;
			}
			targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

			if (feat_null || targ_null)
				continue;

			vec = DatumGetVector(feat_datum);
			if (vec == NULL || vec->dim <= 0)
				continue;

			if (vec->dim != dim)
			{
				int			j;

				NDB_FREE(query.data);
				NDB_SPI_SESSION_END(regress_spi_session);
				for (j = 0; j < nsamples; j++)
					NDB_FREE(samples[j].features);
				NDB_FREE(samples);
				NDB_FREE(tbl_str);
				NDB_FREE(feat_str);
				NDB_FREE(targ_str);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: dimension mismatch: expected %d, got %d",
								dim, vec->dim)));
			}

			/* Copy features */
			NDB_ALLOC(samples[nsamples].features, float, dim);
			memcpy(samples[nsamples].features, vec->data, sizeof(float) * dim);
			samples[nsamples].dim = dim;

			samples[nsamples].label = DatumGetFloat8(targ_datum);

			/* Compute distance */
			samples[nsamples].distance = euclidean_distance(
															query_vector->data, samples[nsamples].features, dim);
			nsamples++;
		}

		if (nsamples < k)
		{
			NDB_FREE(query.data);
			NDB_SPI_SESSION_END(regress_spi_session);
			for (i = 0; i < nsamples; i++)
				NDB_FREE(samples[i].features);
			NDB_FREE(samples);
			NDB_FREE(tbl_str);
			NDB_FREE(feat_str);
			NDB_FREE(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INSUFFICIENT_RESOURCES),
					 errmsg("neurondb: need at least k=%d valid samples, but found only %d",
							k,
							nsamples)));
		}

		NDB_FREE(query.data);
		NDB_SPI_SESSION_END(regress_spi_session);

		/* Sort by distance */
		qsort(samples, nsamples, sizeof(KNNSample), compare_samples);

		for (i = 0; i < k && i < nsamples; i++)
			prediction += samples[i].label;
		prediction /= k;

		for (i = 0; i < nsamples; i++)
			NDB_FREE(samples[i].features);
		NDB_FREE(samples);
	}
	NDB_FREE(tbl_str);
	NDB_FREE(feat_str);
	NDB_FREE(targ_str);

	PG_RETURN_FLOAT8(prediction);
}

/*
 * evaluate_knn_classifier
 *
 * Evaluates KNN classifier performance on a test set
 * Returns: [accuracy, precision, recall, f1_score]
 */
PG_FUNCTION_INFO_V1(evaluate_knn_classifier);

Datum
evaluate_knn_classifier(PG_FUNCTION_ARGS)
{
	text	   *train_table;
	text	   *test_table;
	text	   *feature_col;
	text	   *label_col;
	int			k;
	char	   *train_str;
	char	   *test_str;
	char	   *feat_str;
	char	   *label_str;
	StringInfoData query = {0};
	NDB_DECLARE (NdbSpiSession *, eval_spi_session);
	int			ret;
	int			ntest = 0;
	int			tp = 0,
				tn = 0,
				fp = 0,
				fn = 0;
	double		accuracy,
				precision,
				recall,
				f1_score;
	int			i;
	Datum	   *result_datums;
	ArrayType  *result_array;
	MemoryContext oldcontext;

	train_table = PG_GETARG_TEXT_PP(0);
	test_table = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);
	label_col = PG_GETARG_TEXT_PP(3);
	k = PG_GETARG_INT32(4);

	train_str = text_to_cstring(train_table);
	test_str = text_to_cstring(test_table);
	feat_str = text_to_cstring(feature_col);
	label_str = text_to_cstring(label_col);

	oldcontext = CurrentMemoryContext;
	Assert(oldcontext != NULL);
	NDB_SPI_SESSION_BEGIN(eval_spi_session, oldcontext);

	initStringInfo(&query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 quote_identifier(feat_str),
					 quote_identifier(label_str),
					 quote_identifier(test_str),
					 quote_identifier(feat_str),
					 quote_identifier(label_str));
	elog(DEBUG1, "evaluate_knn_classifier: executing query: %s", query.data);

	ret = ndb_spi_execute(eval_spi_session, query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		NDB_FREE(query.data);
		NDB_FREE(train_str);
		NDB_FREE(test_str);
		NDB_FREE(feat_str);
		NDB_FREE(label_str);
		NDB_SPI_SESSION_END(eval_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_knn_classifier: query failed"),
				 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
				 errhint("Verify the test table exists and contains valid feature and label columns.")));
	}

	ntest = SPI_processed;

	/* For each test sample, classify using KNN */
	for (i = 0; i < ntest; i++)
	{
		HeapTuple	tuple;
		TupleDesc	tupdesc;
		Datum		feat_datum;
		Datum		label_datum;
		bool		feat_null;
		bool		label_null;
		int			y_true;
		int			y_pred;
		/* Safe access to SPI_tuptable - validate before access */
		if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || 
			i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
		{
			continue;
		}
		tuple = SPI_tuptable->vals[i];
		tupdesc = SPI_tuptable->tupdesc;
		if (tupdesc == NULL)
		{
			continue;
		}

		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		label_datum = SPI_getbinval(tuple, tupdesc, 2, &label_null);

		if (feat_null || label_null)
			continue;

		y_true = (int) DatumGetFloat8(label_datum);

		y_pred = DatumGetInt32(DirectFunctionCall5(knn_classify,
												   PointerGetDatum(train_table),
												   PointerGetDatum(feature_col),
												   PointerGetDatum(label_col),
												   feat_datum,
												   Int32GetDatum(k)));

		/* Update confusion matrix */
		if (y_true == 1)
		{
			if (y_pred == 1)
				tp++;
			else
				fn++;
		}
		else
		{
			if (y_pred == 1)
				fp++;
			else
				tn++;
		}
	}

	NDB_FREE(query.data);
	NDB_SPI_SESSION_END(eval_spi_session);

	/* Compute metrics */
	accuracy = (double) (tp + tn) / (tp + tn + fp + fn);
	precision = (tp + fp > 0) ? (double) tp / (tp + fp) : 0.0;
	recall = (tp + fn > 0) ? (double) tp / (tp + fn) : 0.0;
	f1_score = (precision + recall > 0)
		? 2.0 * precision * recall / (precision + recall)
		: 0.0;

	/* Build result array */
	MemoryContextSwitchTo(oldcontext);

	NDB_ALLOC(result_datums, Datum, 4);
	result_datums[0] = Float8GetDatum(accuracy);
	result_datums[1] = Float8GetDatum(precision);
	result_datums[2] = Float8GetDatum(recall);
	result_datums[3] = Float8GetDatum(f1_score);

	result_array = construct_array(result_datums,
								   4,
								   FLOAT8OID,
								   sizeof(float8),
								   FLOAT8PASSBYVAL,
								   'd');

	NDB_FREE(result_datums);
	NDB_FREE(train_str);
	NDB_FREE(test_str);
	NDB_FREE(feat_str);
	NDB_FREE(label_str);

	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * predict_knn_model_id
 *
 * Predicts using a KNN model stored in the catalog by model_id.
 * This is the CPU fallback implementation.
 *
 * Arguments:
 *   model_id INT - model ID from neurondb.ml_models
 *   features REAL[] - feature vector for prediction
 * Returns:
 *   prediction (FLOAT8)
 */
/*
 * train_knn_model_id
 *
 * Trains KNN (lazy learner) - stores metadata in catalog, returns model_id
 * KNN doesn't actually train, it just stores the training table reference
 */
PG_FUNCTION_INFO_V1(train_knn_model_id);

Datum
train_knn_model_id(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *feature_col;
	text	   *label_col;
	int			k_value;
	char	   *tbl_str;
	char	   *feat_str;
	char	   *label_str;
	int			nvec,
				dim;
	int			i;
	bytea	   *model_data = NULL;
	MLCatalogModelSpec spec;
	Jsonb	   *metrics;
	StringInfoData metrics_json;
	int32		model_id;
	StringInfoData model_buf;
	MemoryContext oldcontext;
	NDB_DECLARE (NdbSpiSession *, train_spi_session);

	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	label_col = PG_GETARG_TEXT_PP(2);
	k_value = PG_GETARG_INT32(3);

	if (k_value < 1 || k_value > 1000)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: train_knn_model_id: k must be between 1 and 1000, got %d", k_value),
				 errdetail("K parameter is %d, valid range is 1-1000", k_value),
				 errhint("Specify a positive integer between 1 and 1000 for the number of nearest neighbors.")));

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	label_str = text_to_cstring(label_col);

	oldcontext = CurrentMemoryContext;
	Assert(oldcontext != NULL);
	NDB_SPI_SESSION_BEGIN(train_spi_session, oldcontext);

	/* Load training data and labels together using SPI */
	{
		StringInfoData query = {0};
		int			ret;
		Vector	   *first_vec;
		bool		isnull;
		Datum		vec_datum;

		initStringInfo(&query);
		appendStringInfo(&query, "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
						 quote_identifier(feat_str),
						 quote_identifier(label_str),
						 quote_identifier(tbl_str),
						 quote_identifier(feat_str),
						 quote_identifier(label_str));

		ret = ndb_spi_execute(train_spi_session, query.data, true, 0);
		if (ret != SPI_OK_SELECT)
		{
			NDB_FREE(query.data);
			NDB_SPI_SESSION_END(train_spi_session);
			NDB_FREE(tbl_str);
			NDB_FREE(feat_str);
			NDB_FREE(label_str);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: train_knn_model_id: failed to fetch training data"),
					 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
					 errhint("Verify the table exists and contains valid feature and label columns.")));
		}

		nvec = SPI_processed;
		if (nvec < k_value)
		{
			NDB_FREE(query.data);
			NDB_SPI_SESSION_END(train_spi_session);
			NDB_FREE(tbl_str);
			NDB_FREE(feat_str);
			NDB_FREE(label_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: train_knn_model_id: need at least %d samples, got %d", k_value, nvec),
					 errdetail("Dataset contains %d rows, but k=%d requires at least %d rows", nvec, k_value, k_value),
					 errhint("Reduce k or add more training data.")));
		}

		/* Get dimension from first row - handle both array and vector types */
		{
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;
			Oid			feat_type_oid = SPI_gettypeid(tupdesc, 1);
			bool		feat_is_array = (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID);

			/* Safe access for complex types - validate before access */
			if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || 
				SPI_processed == 0 || SPI_tuptable->vals[0] == NULL)
			{
				NDB_FREE(query.data);
				NDB_SPI_SESSION_END(train_spi_session);
				NDB_FREE(tbl_str);
				NDB_FREE(feat_str);
				NDB_FREE(label_str);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: knn_train: first row is invalid")));
			}
			vec_datum = SPI_getbinval(SPI_tuptable->vals[0], tupdesc, 1, &isnull);
			if (isnull)
			{
				NDB_FREE(query.data);
				NDB_SPI_SESSION_END(train_spi_session);
				NDB_FREE(tbl_str);
				NDB_FREE(feat_str);
				NDB_FREE(label_str);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: NULL feature in first row")));
			}

			if (feat_is_array)
			{
				ArrayType  *arr = DatumGetArrayTypeP(vec_datum);

				if (ARR_NDIM(arr) != 1)
				{
					NDB_FREE(query.data);
					NDB_SPI_SESSION_END(train_spi_session);
					NDB_FREE(tbl_str);
					NDB_FREE(feat_str);
					NDB_FREE(label_str);
					ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							 errmsg("neurondb: train_knn_model_id: feature array must be 1-dimensional"),
							 errdetail("Array has %d dimensions, expected 1", ARR_NDIM(arr)),
							 errhint("Ensure the feature column contains 1-dimensional arrays.")));
				}
				dim = ARR_DIMS(arr)[0];
			}
			else
			{
				first_vec = DatumGetVector(vec_datum);
				if (first_vec == NULL)
				{
					NDB_FREE(query.data);
					NDB_SPI_SESSION_END(train_spi_session);
					NDB_FREE(tbl_str);
					NDB_FREE(feat_str);
					NDB_FREE(label_str);
					ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							 errmsg("neurondb: train_knn_model_id: invalid vector in first row"),
							 errdetail("First row feature vector is NULL"),
							 errhint("Ensure the feature column contains valid vector data.")));
				}
				dim = first_vec->dim;
			}
		}

		/*
		 * Verify all features have same dimension (we don't need to store the
		 * data)
		 */
		{
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;
			Oid			feat_type_oid = SPI_gettypeid(tupdesc, 1);
			bool		feat_is_array = (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID);

			for (i = 0; i < nvec; i++)
			{
				/* Safe access to SPI_tuptable - validate before access */
				if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || 
					i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
				{
					continue;
				}
				vec_datum = SPI_getbinval(SPI_tuptable->vals[i], tupdesc, 1, &isnull);
				if (isnull)
				{
					NDB_FREE(query.data);
					NDB_SPI_SESSION_END(train_spi_session);
					NDB_FREE(tbl_str);
					NDB_FREE(feat_str);
					NDB_FREE(label_str);
					ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							 errmsg("neurondb: train_knn_model_id: NULL feature at row %d", i),
							 errdetail("Row %d contains a NULL feature value", i),
							 errhint("Remove NULL values from the feature column or filter them out.")));
				}

				if (feat_is_array)
				{
					ArrayType  *arr = DatumGetArrayTypeP(vec_datum);

					if (ARR_NDIM(arr) != 1 || ARR_DIMS(arr)[0] != dim)
					{
						NDB_FREE(query.data);
						NDB_SPI_SESSION_END(train_spi_session);
						NDB_FREE(tbl_str);
						NDB_FREE(feat_str);
						NDB_FREE(label_str);
						ereport(ERROR,
								(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
								 errmsg("neurondb: train_knn_model_id: inconsistent feature array dimension at row %d", i),
								 errdetail("Expected dimension %d but got %d (ndims=%d)", dim, ARR_DIMS(arr)[0], ARR_NDIM(arr)),
								 errhint("Ensure all feature arrays have consistent dimensions.")));
					}
				}
				else
				{
					Vector	   *vec = DatumGetVector(vec_datum);

					if (vec == NULL || vec->dim != dim)
					{
						NDB_FREE(query.data);
						NDB_SPI_SESSION_END(train_spi_session);
						NDB_FREE(tbl_str);
						NDB_FREE(feat_str);
						NDB_FREE(label_str);
						ereport(ERROR,
								(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
								 errmsg("neurondb: train_knn_model_id: inconsistent vector dimension at row %d", i),
								 errdetail("Expected dimension %d but got %d", dim, vec ? vec->dim : -1),
								 errhint("Ensure all feature vectors have the same dimension.")));
					}
				}
			}
		}

		NDB_FREE(query.data);
		NDB_SPI_SESSION_END(train_spi_session);
	}

	/* For KNN (lazy learner), store minimal model data: just k and dimensions */
	/* The actual training data stays in the table */
	initStringInfo(&model_buf);
	appendBinaryStringInfo(&model_buf, (char *) &k_value, sizeof(int));
	appendBinaryStringInfo(&model_buf, (char *) &nvec, sizeof(int));
	appendBinaryStringInfo(&model_buf, (char *) &dim, sizeof(int));

	/* Store table name, feature col, label col as strings for CPU prediction */
	appendBinaryStringInfo(&model_buf, tbl_str, strlen(tbl_str) + 1);
	appendBinaryStringInfo(&model_buf, feat_str, strlen(feat_str) + 1);
	appendBinaryStringInfo(&model_buf, label_str, strlen(label_str) + 1);

	{
		int			total_size = VARHDRSZ + model_buf.len;

		{
			NDB_DECLARE(char *, model_bytes);
			NDB_ALLOC(model_bytes, char, total_size);
			model_data = (bytea *) model_bytes;
		}
		SET_VARSIZE(model_data, total_size);
		memcpy(VARDATA(model_data), model_buf.data, model_buf.len);
		NDB_FREE(model_buf.data);
	}

	/* Build metrics JSONB */
	initStringInfo(&metrics_json);
	appendStringInfo(&metrics_json, "{\"storage\": \"cpu\", \"k\": %d, \"n_samples\": %d, \"n_features\": %d}",
					 k_value, nvec, dim);
	/* Use ndb_jsonb_in_cstring like other ML algorithms fix */
	metrics = ndb_jsonb_in_cstring(metrics_json.data);
	if (metrics == NULL)
	{
		NDB_FREE(metrics_json.data);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("neurondb: train_knn_model_id: failed to parse metrics JSON")));
	}
	NDB_FREE(metrics_json.data);

	/* Store model in catalog */
	memset(&spec, 0, sizeof(MLCatalogModelSpec));
	spec.project_name = NULL;
	spec.algorithm = "knn";
	spec.training_table = tbl_str;
	spec.training_column = label_str;
	spec.model_data = model_data;
	spec.metrics = metrics;
	spec.num_samples = nvec;
	spec.num_features = dim;

	model_id = ml_catalog_register_model(&spec);

	/* Cleanup */
	NDB_FREE(tbl_str);
	NDB_FREE(feat_str);
	NDB_FREE(label_str);

	PG_RETURN_INT32(model_id);
}

PG_FUNCTION_INFO_V1(predict_knn_model_id);

Datum
predict_knn_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	ArrayType  *features_array;
	bytea	   *model_data = NULL;
	Jsonb	   *metrics = NULL;
	const char *base;
	NdbCudaKnnModelHeader *hdr;
	float	   *query_features = NULL;
	int			i;
	int			j;
	int			ndims;
	int		   *dims;
	int			nelems;
	float8	   *features = NULL;
	float8	   *features_allocated = NULL;	/* For float4->float8 conversion */
	double		prediction = 0.0;
	MemoryContext oldcontext;
	MemoryContext callcontext;

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("predict_knn_model_id: requires 2 arguments, got %d",
						PG_NARGS()),
				 errhint("Usage: predict_knn_model_id(model_id, features[])")));

	model_id = PG_GETARG_INT32(0);
	features_array = PG_GETARG_ARRAYTYPE_P(1);

	/* Validate features array */
	ndims = ARR_NDIM(features_array);
	dims = ARR_DIMS(features_array);
	nelems = ArrayGetNItems(ndims, dims);

	/* Handle both float4[] (real[]) and float8[] arrays */
	{
		Oid			elem_type = ARR_ELEMTYPE(features_array);

		if (elem_type == FLOAT4OID)
		{
			/* Convert float4[] to float8[] */
			float4	   *features_f4 = (float4 *) ARR_DATA_PTR(features_array);

			NDB_DECLARE(float8 *, features_allocated);
			NDB_ALLOC(features_allocated, float8, nelems);
			for (i = 0; i < nelems; i++)
				features_allocated[i] = (float8) features_f4[i];
			features = features_allocated;
		}
		else
		{
			/* Already float8[] */
			features = (float8 *) ARR_DATA_PTR(features_array);
		}
	}

	if (nelems == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("predict_knn_model_id: features array cannot be empty")));

	/* Fetch model data from catalog */
	elog(DEBUG1, "predict_knn_model_id: creating memory context");
	callcontext = AllocSetContextCreate(CurrentMemoryContext,
										"predict_knn_model_id context",
										ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(callcontext);

	if (!ml_catalog_fetch_model_payload(model_id, &model_data, NULL, &metrics))
	{
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(callcontext);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("predict_knn_model_id: model %d not found", model_id)));
	}

	if (model_data == NULL)
	{
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(callcontext);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("predict_knn_model_id: model %d has no model_data", model_id)));
	}

	/* Ensure bytea is in current function's memory context */
	{
		int			data_len = VARSIZE(model_data);
		NDB_DECLARE(bytea *, copy);
		{
			NDB_DECLARE(char *, copy_bytes);
			NDB_ALLOC(copy_bytes, char, data_len);
			copy = (bytea *) copy_bytes;
		}

		memcpy(copy, model_data, data_len);
		model_data = copy;
	}

	base = VARDATA(model_data);

	/*
	 * Check if this is CPU model (starts with k, n_samples, n_features) or
	 * GPU model
	 */

	/*
	 * CPU model: k (int), n_samples (int), n_features (int), table_name
	 * (string), feature_col (string), label_col (string)
	 */
	/* GPU model: NdbCudaKnnModelHeader */
	if (VARSIZE(model_data) - VARHDRSZ < sizeof(int) * 3)
	{
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(callcontext);
		if (metrics)
			NDB_FREE(metrics);
		if (model_data)
			NDB_FREE(model_data);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("predict_knn_model_id: model %d has invalid header (too small)", model_id)));
	}

	/* Try to detect CPU vs GPU format */

	/*
	 * CPU format: first int is k (usually 1-100), GPU format: first bytes are
	 * magic/version
	 */
	{
		int			first_int = *(int *) base;

		/*
		 * If first int is reasonable k value (1-1000) and model size matches
		 * CPU format, use CPU
		 */
		if (first_int >= 1 && first_int <= 1000)
		{
			/* CPU model format */
			int			k_cpu,
						n_samples_cpu,
						n_features_cpu;
			char	   *table_name_cpu,
					   *feature_col_cpu,
					   *label_col_cpu;
			const char *str_ptr;
			StringInfoData sql_query;
			StringInfoData tbl_lit,
						feat_lit,
						label_lit;
			int			ret;
			int			k;

			memcpy(&k_cpu, base, sizeof(int));
			memcpy(&n_samples_cpu, base + sizeof(int), sizeof(int));
			memcpy(&n_features_cpu, base + sizeof(int) * 2, sizeof(int));

			if (nelems != n_features_cpu)
			{
				MemoryContextSwitchTo(oldcontext);
				MemoryContextDelete(callcontext);
				if (metrics)
					NDB_FREE(metrics);
				if (model_data)
					NDB_FREE(model_data);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("Feature dimension mismatch: expected %d, got %d", n_features_cpu, nelems)));
			}

			/* Extract strings */
			str_ptr = base + sizeof(int) * 3;
			table_name_cpu = (char *) str_ptr;
			str_ptr += strlen(table_name_cpu) + 1;
			feature_col_cpu = (char *) str_ptr;
			str_ptr += strlen(feature_col_cpu) + 1;
			label_col_cpu = (char *) str_ptr;

			/* Call knn_classify via SPI SQL to avoid memory context issues */

			/*
			 * knn_classify uses SPI internally, so calling it via SQL is
			 * safer
			 */

			/* Quote strings for SQL */
			initStringInfo(&tbl_lit);
			appendStringInfoChar(&tbl_lit, '\'');
			for (k = 0; table_name_cpu[k]; k++)
			{
				if (table_name_cpu[k] == '\'')
					appendStringInfoChar(&tbl_lit, '\'');
				appendStringInfoChar(&tbl_lit, table_name_cpu[k]);
			}
			appendStringInfoChar(&tbl_lit, '\'');

			initStringInfo(&feat_lit);
			appendStringInfoChar(&feat_lit, '\'');
			for (k = 0; feature_col_cpu[k]; k++)
			{
				if (feature_col_cpu[k] == '\'')
					appendStringInfoChar(&feat_lit, '\'');
				appendStringInfoChar(&feat_lit, feature_col_cpu[k]);
			}
			appendStringInfoChar(&feat_lit, '\'');

			initStringInfo(&label_lit);
			appendStringInfoChar(&label_lit, '\'');
			for (k = 0; label_col_cpu[k]; k++)
			{
				if (label_col_cpu[k] == '\'')
					appendStringInfoChar(&label_lit, '\'');
				appendStringInfoChar(&label_lit, label_col_cpu[k]);
			}
			appendStringInfoChar(&label_lit, '\'');

			/* Build vector literal string */
			initStringInfo(&sql_query);
			appendStringInfo(&sql_query, "SELECT knn_classify(%s, %s, %s, '",
							 tbl_lit.data, feat_lit.data, label_lit.data);

			/* Build vector literal */
			appendStringInfoChar(&sql_query, '[');
			for (i = 0; i < nelems; i++)
			{
				if (i > 0)
					appendStringInfoString(&sql_query, ", ");
				/* Validate feature value before formatting */
				if (!isfinite(features[i]))
				{
					NDB_FREE(sql_query.data);
					NDB_FREE(tbl_lit.data);
					NDB_FREE(feat_lit.data);
					NDB_FREE(label_lit.data);
					if (features_allocated)
						NDB_FREE(features_allocated);
					MemoryContextSwitchTo(oldcontext);
					MemoryContextDelete(callcontext);
					if (metrics)
						NDB_FREE(metrics);
					if (model_data)
						NDB_FREE(model_data);
					ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							 errmsg("neurondb: non-finite value in features array at index %d", i)));
				}
				appendStringInfo(&sql_query, "%.6f", features[i]);
			}
			appendStringInfo(&sql_query, "]'::vector, %d)", k_cpu);

			/* Execute via SPI */
			{
				NDB_DECLARE (NdbSpiSession *, predict_spi_session);
				MemoryContext predict_oldcontext = CurrentMemoryContext;

				Assert(predict_oldcontext != NULL);
				NDB_SPI_SESSION_BEGIN(predict_spi_session, predict_oldcontext);

				ret = ndb_spi_execute(predict_spi_session, sql_query.data, true, 0);
				if (ret != SPI_OK_SELECT || SPI_processed == 0)
				{
					NDB_SPI_SESSION_END(predict_spi_session);
					NDB_FREE(sql_query.data);
					NDB_FREE(tbl_lit.data);
					NDB_FREE(feat_lit.data);
					NDB_FREE(label_lit.data);
					MemoryContextSwitchTo(oldcontext);
					MemoryContextDelete(callcontext);
					if (metrics)
						NDB_FREE(metrics);
					if (model_data)
						NDB_FREE(model_data);
					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg("neurondb: predict_knn_model_id: knn_classify failed"),
							 errdetail("SPI execution returned code %d (expected %d) or processed 0 rows", ret, SPI_OK_SELECT),
							 errhint("Verify the model exists and the training table is accessible.")));
				}

				/* Get result - use safe function for int32 */
				{
					int32		result_val;
					
					if (ndb_spi_get_int32(predict_spi_session, 0, 1, &result_val))
					{
						prediction = (double) result_val;
					}
					else
					{
						prediction = 0.0;
					}
				}

				NDB_SPI_SESSION_END(predict_spi_session);
			}
			NDB_FREE(sql_query.data);
			NDB_FREE(tbl_lit.data);
			NDB_FREE(feat_lit.data);
			NDB_FREE(label_lit.data);

			/* Free features_allocated before deleting callcontext */
			if (features_allocated)
				NDB_FREE(features_allocated);

			/* Free model_data and metrics before switching contexts */
			if (model_data)
				NDB_FREE(model_data);
			if (metrics)
				NDB_FREE(metrics);

			MemoryContextSwitchTo(oldcontext);
			MemoryContextDelete(callcontext);

			PG_RETURN_FLOAT8(prediction);
		}
	}

	/* GPU model format */
	if (VARSIZE(model_data) - VARHDRSZ < sizeof(NdbCudaKnnModelHeader))
	{
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(callcontext);
		if (metrics)
			NDB_FREE(metrics);
		if (model_data)
			NDB_FREE(model_data);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("predict_knn_model_id: model %d has invalid header",
						model_id)));
	}

	hdr = (NdbCudaKnnModelHeader *) base;

	/* Validate model header */
	if (hdr->n_samples <= 0 || hdr->n_samples > 100000000 ||
		hdr->n_features <= 0 || hdr->n_features > 1000000 ||
		hdr->k <= 0 || hdr->k > hdr->n_samples ||
		(hdr->task_type != 0 && hdr->task_type != 1))
	{
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(callcontext);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("predict_knn_model_id: model %d has invalid header",
						model_id)));
	}

	/* Validate feature dimension */
	if (nelems != hdr->n_features)
	{
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(callcontext);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("predict_knn_model_id: feature dimension mismatch (expected %d, got %d)",
						hdr->n_features, nelems)));
	}

	/* Convert query features to float */
	NDB_ALLOC(query_features, float, hdr->n_features);
	for (i = 0; i < hdr->n_features; i++)
	{
		if (!isfinite(features[i]))
		{
			NDB_FREE(query_features);
			MemoryContextSwitchTo(oldcontext);
			MemoryContextDelete(callcontext);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("predict_knn_model_id: non-finite value in features array")));
		}
		query_features[i] = (float) features[i];
	}

	/* Try GPU prediction if GPU is available */
#ifdef NDB_GPU_CUDA
	if (neurondb_gpu_is_available())
	{
		int			rc;
		char	   *gpu_errstr = NULL;

		PG_TRY();
		{
			rc = ndb_cuda_knn_predict(model_data,
									  query_features,
									  hdr->n_features,
									  &prediction,
									  &gpu_errstr);

			if (rc == 0)
			{
				/* GPU prediction succeeded */
				NDB_FREE(query_features);
				if (gpu_errstr)
					NDB_FREE(gpu_errstr);
				if (features_allocated)
					NDB_FREE(features_allocated);
				if (model_data)
					NDB_FREE(model_data);
				if (metrics)
					NDB_FREE(metrics);
				MemoryContextSwitchTo(oldcontext);
				MemoryContextDelete(callcontext);
				PG_RETURN_FLOAT8(prediction);
			}
			else
			{
				/* GPU prediction failed, fall through to CPU */
				elog(DEBUG1,
					 "neurondb: predict_knn_model_id: GPU prediction failed: %s, falling back to CPU",
					 gpu_errstr ? gpu_errstr : "unknown error");
				if (gpu_errstr)
					NDB_FREE(gpu_errstr);
			}
		}
		PG_CATCH();
		{
			if (gpu_errstr)
				NDB_FREE(gpu_errstr);
			/* Fall through to CPU path */
		}
		PG_END_TRY();
	}
#endif

	/* Fallback to CPU-based prediction when GPU unavailable */
	{
		const float *training_features;
		const double *training_labels;
		NDB_DECLARE(float *, distances);
		NDB_DECLARE(int *, indices);

		/* Extract training data from model */
		training_features = (const float *) (base + sizeof(NdbCudaKnnModelHeader));
		training_labels = (const double *) (base + sizeof(NdbCudaKnnModelHeader) +
											sizeof(float) * (size_t) hdr->n_samples * (size_t) hdr->n_features);

		/* Allocate distance array and indices */
		NDB_ALLOC(distances, float, hdr->n_samples);
		NDB_ALLOC(indices, int, hdr->n_samples);

		/* Compute distances to all training samples */
		for (i = 0; i < hdr->n_samples; i++)
		{
			double		sum = 0.0;
			const float *train_feat = training_features + (i * hdr->n_features);

			for (j = 0; j < hdr->n_features; j++)
			{
				double		diff = query_features[j] - train_feat[j];

				sum += diff * diff;
			}
			distances[i] = (float) sqrt(sum);
			indices[i] = i;
		}

		/* Sort by distance (simple selection sort for k nearest) */
		for (i = 0; i < hdr->k && i < hdr->n_samples; i++)
		{
			int			min_idx = i;

			for (j = i + 1; j < hdr->n_samples; j++)
			{
				if (distances[j] < distances[min_idx])
					min_idx = j;
			}
			if (min_idx != i)
			{
				float		tmp_dist = distances[i];
				int			tmp_idx = indices[i];

				distances[i] = distances[min_idx];
				indices[i] = indices[min_idx];
				distances[min_idx] = tmp_dist;
				indices[min_idx] = tmp_idx;
			}
		}

		/* Compute prediction from k nearest neighbors */
		if (hdr->task_type == 0)
		{
			/* Classification: majority vote */
			double		class_votes[2] = {0.0, 0.0};

			for (i = 0; i < hdr->k && i < hdr->n_samples; i++)
			{
				int			label = (int) training_labels[indices[i]];

				if (label >= 0 && label < 2)
					class_votes[label] += 1.0;
			}
			prediction = (class_votes[1] > class_votes[0]) ? 1.0 : 0.0;
		}
		else
		{
			/* Regression: average */
			for (i = 0; i < hdr->k && i < hdr->n_samples; i++)
				prediction += training_labels[indices[i]];
			prediction /= hdr->k;
		}

		/* Cleanup */
		NDB_FREE(distances);
		NDB_FREE(indices);
	}

	/* Cleanup */
	NDB_FREE(query_features);
	if (features_allocated)
		NDB_FREE(features_allocated);
	if (model_data)
		NDB_FREE(model_data);
	if (metrics)
		NDB_FREE(metrics);
	MemoryContextSwitchTo(oldcontext);
	MemoryContextDelete(callcontext);

	PG_RETURN_FLOAT8(prediction);
}

/*
 * knn_predict_batch
 *
 * Helper function to predict a batch of samples using KNN model.
 * Updates confusion matrix.
 */
static void
knn_predict_batch(int32 model_id,
				  const float *features,
				  const double *labels,
				  int n_samples,
				  int feature_dim,
				  int *tp_out,
				  int *tn_out,
				  int *fp_out,
				  int *fn_out)
{
	int			i;
	int			tp = 0;
	int			tn = 0;
	int			fp = 0;
	int			fn = 0;
	bytea	   *model_data = NULL;
	Jsonb	   *metrics = NULL;
	const char *base;
	int			k_value = 0;

	if (features == NULL || labels == NULL || n_samples <= 0)
	{
		if (tp_out)
			*tp_out = 0;
		if (tn_out)
			*tn_out = 0;
		if (fp_out)
			*fp_out = 0;
		if (fn_out)
			*fn_out = 0;
		return;
	}

	/* Load model to get k value */
	if (!ml_catalog_fetch_model_payload(model_id, &model_data, NULL, &metrics))
	{
		if (tp_out)
			*tp_out = 0;
		if (tn_out)
			*tn_out = 0;
		if (fp_out)
			*fp_out = 0;
		if (fn_out)
			*fn_out = 0;
		return;
	}

	if (model_data == NULL)
	{
		if (metrics)
			NDB_FREE(metrics);
		if (tp_out)
			*tp_out = 0;
		if (tn_out)
			*tn_out = 0;
		if (fp_out)
			*fp_out = 0;
		if (fn_out)
			*fn_out = 0;
		return;
	}

	base = VARDATA(model_data);

	/* Detect CPU vs GPU format and get k_value */
	{
		int			first_int = *(int *) base;

		if (first_int >= 1 && first_int <= 1000)
		{
			/* CPU format */
			memcpy(&k_value, base, sizeof(int));
		}
		else
		{
			/* GPU format */
			const		NdbCudaKnnModelHeader *gpu_hdr;

			if (VARSIZE(model_data) - VARHDRSZ < (int) sizeof(NdbCudaKnnModelHeader))
			{
				if (metrics)
					NDB_FREE(metrics);
				if (model_data)
					NDB_FREE(model_data);
				if (tp_out)
					*tp_out = 0;
				if (tn_out)
					*tn_out = 0;
				if (fp_out)
					*fp_out = 0;
				if (fn_out)
					*fn_out = 0;
				return;
			}
			gpu_hdr = (const NdbCudaKnnModelHeader *) base;
			k_value = gpu_hdr->k;
		}
	}

	if (k_value < 1)
	{
		if (metrics)
			NDB_FREE(metrics);
		if (model_data)
			NDB_FREE(model_data);
		if (tp_out)
			*tp_out = 0;
		if (tn_out)
			*tn_out = 0;
		if (fp_out)
			*fp_out = 0;
		if (fn_out)
			*fn_out = 0;
		return;
	}

	/* Predict directly - avoid calling predict_knn_model_id which uses SPI */
	{
		int			first_int = *(int *) base;
		bool		is_gpu = (first_int < 1 || first_int > 1000);

		if (is_gpu)
		{
			/* GPU model: use GPU-accelerated batch evaluation when available */
			const		NdbCudaKnnModelHeader *gpu_hdr;

			if (VARSIZE(model_data) - VARHDRSZ < (int) sizeof(NdbCudaKnnModelHeader))
			{
				if (metrics)
					NDB_FREE(metrics);
				if (model_data)
					NDB_FREE(model_data);
				if (tp_out)
					*tp_out = 0;
				if (tn_out)
					*tn_out = 0;
				if (fp_out)
					*fp_out = 0;
				if (fn_out)
					*fn_out = 0;
				return;
			}

			gpu_hdr = (const NdbCudaKnnModelHeader *) base;

			/* Try GPU batch prediction if GPU is available */
#ifdef NDB_GPU_CUDA
			if (neurondb_gpu_is_available() && gpu_hdr->task_type == 0)
			{
				int		   *predictions = NULL;
				int			rc;
				char	   *gpu_errstr = NULL;

				elog(DEBUG1,
					 "neurondb: knn_predict_batch: attempting GPU batch prediction for %d samples (model_id=%d, n_features=%d, k=%d)",
					 n_samples, model_id, feature_dim, k_value);

				/* Allocate predictions array */
				NDB_DECLARE(int *, predictions);
				NDB_ALLOC(predictions, int, (size_t) n_samples);
				if (predictions != NULL)
				{
					PG_TRY();
					{
						rc = ndb_cuda_knn_predict_batch(model_data,
														features,
														n_samples,
														feature_dim,
														predictions,
														&gpu_errstr);

						if (rc == 0)
						{
							elog(DEBUG1,
								 "neurondb: knn_predict_batch: GPU batch prediction succeeded for %d samples",
								 n_samples);
							/* Compute confusion matrix from GPU predictions */
							for (i = 0; i < n_samples; i++)
							{
								double		y_true = labels[i];
								int			true_class;
								int			pred_class;

								if (!isfinite(y_true))
									continue;

								true_class = (int) rint(y_true);
								if (true_class < 0)
									true_class = 0;
								if (true_class > 1)
									true_class = 1;

								pred_class = predictions[i];
								if (pred_class < 0)
									pred_class = 0;
								if (pred_class > 1)
									pred_class = 1;

								/* Update confusion matrix */
								if (true_class == 1 && pred_class == 1)
									tp++;
								else if (true_class == 0 && pred_class == 0)
									tn++;
								else if (true_class == 0 && pred_class == 1)
									fp++;
								else if (true_class == 1 && pred_class == 0)
									fn++;
							}

							NDB_FREE(predictions);
							if (gpu_errstr)
								NDB_FREE(gpu_errstr);
							if (metrics)
								NDB_FREE(metrics);
							if (model_data)
								NDB_FREE(model_data);
							if (tp_out)
								*tp_out = tp;
							if (tn_out)
								*tn_out = tn;
							if (fp_out)
								*fp_out = fp;
							if (fn_out)
								*fn_out = fn;
							return;
						}
						else
						{
							/* GPU prediction failed, fall through to CPU */
							elog(WARNING,
								 "neurondb: knn_predict_batch: GPU batch prediction failed for %d samples (model_id=%d): %s, falling back to CPU",
								 n_samples, model_id, gpu_errstr ? gpu_errstr : "unknown error");
							if (gpu_errstr)
								NDB_FREE(gpu_errstr);
							NDB_FREE(predictions);
						}
					}
					PG_CATCH();
					{
						if (predictions)
							NDB_FREE(predictions);
						if (gpu_errstr)
							NDB_FREE(gpu_errstr);
						/* Fall through to CPU path */
					}
					PG_END_TRY();
				}
			}
#endif

			/*
			 * Fallback to CPU-based evaluation for GPU models when GPU
			 * unavailable
			 */
			{
				const float *training_features;
				const double *training_labels;

				elog(WARNING,
					 "neurondb: knn_predict_batch: using slow CPU fallback for GPU model (model_id=%d, n_samples=%d, n_train=%d). This is slow - ensure GPU is available.",
					 model_id, n_samples, gpu_hdr->n_samples);

				training_features = (const float *) (base + sizeof(NdbCudaKnnModelHeader));
				training_labels = (const double *) (base + sizeof(NdbCudaKnnModelHeader) +
													sizeof(float) * (size_t) gpu_hdr->n_samples * (size_t) gpu_hdr->n_features);

				/* For each test sample, compute distances and predict */
				for (i = 0; i < n_samples; i++)
				{
					const float *row = features + (i * feature_dim);
					double		y_true = labels[i];
					int			true_class;
					double		prediction = 0.0;
					int			pred_class;
					float	   *distances_local = NULL;
					int		   *indices_local = NULL;
					int			j;

					if (!isfinite(y_true))
						continue;

					true_class = (int) rint(y_true);
					if (true_class < 0)
						true_class = 0;
					if (true_class > 1)
						true_class = 1;

					/* Compute distances to all training samples */
					NDB_ALLOC(distances_local, float, gpu_hdr->n_samples);
					NDB_ALLOC(indices_local, int, gpu_hdr->n_samples);

					for (j = 0; j < gpu_hdr->n_samples; j++)
					{
						double		sum = 0.0;
						const float *train_feat = training_features + (j * gpu_hdr->n_features);
						int			k;

						for (k = 0; k < feature_dim; k++)
						{
							double		diff = row[k] - train_feat[k];

							sum += diff * diff;
						}
						distances_local[j] = (float) sqrt(sum);
						indices_local[j] = j;
					}

					/* Selection sort for k nearest */
					for (j = 0; j < k_value && j < gpu_hdr->n_samples; j++)
					{
						int			min_idx = j;
						int			k;

						for (k = j + 1; k < gpu_hdr->n_samples; k++)
						{
							if (distances_local[k] < distances_local[min_idx])
								min_idx = k;
						}
						if (min_idx != j)
						{
							float		tmp_dist = distances_local[j];
							int			tmp_idx = indices_local[j];

							distances_local[j] = distances_local[min_idx];
							indices_local[j] = indices_local[min_idx];
							distances_local[min_idx] = tmp_dist;
							indices_local[min_idx] = tmp_idx;
						}
					}

					/* Majority vote */
					{
						double		class_votes[2] = {0.0, 0.0};
						int			k;

						for (k = 0; k < k_value && k < gpu_hdr->n_samples; k++)
						{
							int			label = (int) training_labels[indices_local[k]];

							if (label >= 0 && label < 2)
								class_votes[label] += 1.0;
						}
						prediction = (class_votes[1] > class_votes[0]) ? 1.0 : 0.0;
					}

					pred_class = (int) rint(prediction);
					if (pred_class < 0)
						pred_class = 0;
					if (pred_class > 1)
						pred_class = 1;

					NDB_FREE(distances_local);
					NDB_FREE(indices_local);

					/* Update confusion matrix */
					if (true_class == 1 && pred_class == 1)
						tp++;
					else if (true_class == 0 && pred_class == 0)
						tn++;
					else if (true_class == 0 && pred_class == 1)
						fp++;
					else if (true_class == 1 && pred_class == 0)
						fn++;
				}
			}
		}
		else
		{
			/* CPU model: load training data and predict */
			char	   *table_name_cpu,
					   *feature_col_cpu,
					   *label_col_cpu;
			const char *str_ptr;
			StringInfoData query;
			int			ret;
			int			n_train = 0;
			float	   *train_features = NULL;
			double	   *train_labels = NULL;
			int			train_valid = 0;
			int			j;
			MemoryContext outer_context;

			/* Extract strings from model */
			str_ptr = base + sizeof(int) * 3;	/* Skip k, n_samples,
												 * n_features */
			table_name_cpu = (char *) str_ptr;
			str_ptr += strlen(table_name_cpu) + 1;
			feature_col_cpu = (char *) str_ptr;
			str_ptr += strlen(feature_col_cpu) + 1;
			label_col_cpu = (char *) str_ptr;

			/* Save outer memory context before nested SPI */
			outer_context = CurrentMemoryContext;

			/* Build query in outer context before nested SPI */
			initStringInfo(&query);
			appendStringInfo(&query,
							 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
							 quote_identifier(feature_col_cpu),
							 quote_identifier(label_col_cpu),
							 quote_identifier(table_name_cpu),
							 quote_identifier(feature_col_cpu),
							 quote_identifier(label_col_cpu));

			/* Load training data using nested SPI (PostgreSQL supports this) */
			{
				NDB_DECLARE (NdbSpiSession *, eval_nested_spi_session);
				MemoryContext eval_nested_oldcontext = CurrentMemoryContext;

				Assert(eval_nested_oldcontext != NULL);
				NDB_SPI_SESSION_BEGIN(eval_nested_spi_session, eval_nested_oldcontext);

				ret = ndb_spi_execute(eval_nested_spi_session, query.data, true, 0);
				if (ret == SPI_OK_SELECT)
				{
					n_train = SPI_processed;
					if (n_train >= k_value)
					{
					TupleDesc	tupdesc;
					Oid			feat_type_oid;
					bool		feat_is_array;
					MemoryContext nested_context;
					NDB_DECLARE(float *, train_features);
					NDB_DECLARE(double *, train_labels);

					/* Save nested SPI context */
					nested_context = CurrentMemoryContext;

					/* Allocate in outer context so it survives SPI_finish() */
					MemoryContextSwitchTo(outer_context);
					NDB_ALLOC(train_features, float, (size_t) n_train * (size_t) feature_dim);
					NDB_ALLOC(train_labels, double, (size_t) n_train);
					MemoryContextSwitchTo(nested_context);

					tupdesc = SPI_tuptable->tupdesc;
					feat_type_oid = SPI_gettypeid(tupdesc, 1);
					feat_is_array = (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID);

					for (i = 0; i < n_train; i++)
					{
						HeapTuple	tuple;
						TupleDesc	tupdesc;
						Datum		feat_datum;
						Datum		label_datum;
						bool		feat_null;
						bool		label_null;
						Vector	   *vec;
						ArrayType  *arr;
						float	   *train_row;
						/* Safe access to SPI_tuptable - validate before access */
						if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || 
							i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
						{
							continue;
						}
						tuple = SPI_tuptable->vals[i];
						tupdesc = SPI_tuptable->tupdesc;
						if (tupdesc == NULL)
						{
							continue;
						}

						feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
						label_datum = SPI_getbinval(tuple, tupdesc, 2, &label_null);

						if (feat_null || label_null)
							continue;

						train_row = train_features + (train_valid * feature_dim);
						train_labels[train_valid] = DatumGetFloat8(label_datum);

						if (feat_is_array)
						{
							arr = DatumGetArrayTypeP(feat_datum);
							if (ARR_NDIM(arr) != 1 || ARR_DIMS(arr)[0] != feature_dim)
								continue;
							if (feat_type_oid == FLOAT8ARRAYOID)
							{
								float8	   *data = (float8 *) ARR_DATA_PTR(arr);

								for (j = 0; j < feature_dim; j++)
									train_row[j] = (float) data[j];
							}
							else
							{
								float4	   *data = (float4 *) ARR_DATA_PTR(arr);

								memcpy(train_row, data, sizeof(float) * feature_dim);
							}
						}
						else
						{
							vec = DatumGetVector(feat_datum);
							if (vec == NULL || vec->dim != feature_dim)
								continue;
							memcpy(train_row, vec->data, sizeof(float) * feature_dim);
						}

						train_valid++;
					}
					}
				}
				NDB_SPI_SESSION_END(eval_nested_spi_session);
			}

			/*
			 * Free query.data - it's allocated in outer context
			 */

			if (train_valid < k_value)
			{
				if (train_features)
					NDB_FREE(train_features);
				if (train_labels)
					NDB_FREE(train_labels);
				if (metrics)
					NDB_FREE(metrics);
				if (model_data)
					NDB_FREE(model_data);
				if (tp_out)
					*tp_out = 0;
				if (tn_out)
					*tn_out = 0;
				if (fp_out)
					*fp_out = 0;
				if (fn_out)
					*fn_out = 0;
				return;
			}

			/* For each test sample, compute distances and predict */
			for (i = 0; i < n_samples; i++)
			{
				const float *row = features + (i * feature_dim);
				double		y_true = labels[i];
				int			true_class;
				double		prediction = 0.0;
				int			pred_class;
				KNNSample  *samples_local = NULL;

				if (!isfinite(y_true))
					continue;

				true_class = (int) rint(y_true);
				if (true_class < 0)
					true_class = 0;
				if (true_class > 1)
					true_class = 1;

				/* Compute distances to all training samples */
				NDB_ALLOC(samples_local, KNNSample, train_valid);
				for (j = 0; j < train_valid; j++)
				{
					samples_local[j].features = train_features + (j * feature_dim);
					samples_local[j].label = train_labels[j];
					samples_local[j].distance = euclidean_distance(
																   row, samples_local[j].features, feature_dim);
					samples_local[j].dim = feature_dim;
				}

				/* Sort by distance */
				qsort(samples_local, train_valid, sizeof(KNNSample), compare_samples);

				/* Majority vote */
				{
					double		class_votes[2] = {0.0, 0.0};
					int			k;

					for (k = 0; k < k_value && k < train_valid; k++)
					{
						int			label = (int) samples_local[k].label;

						if (label >= 0 && label < 2)
							class_votes[label] += 1.0;
					}
					prediction = (class_votes[1] > class_votes[0]) ? 1.0 : 0.0;
				}

				pred_class = (int) rint(prediction);
				if (pred_class < 0)
					pred_class = 0;
				if (pred_class > 1)
					pred_class = 1;

				NDB_FREE(samples_local);

				/* Update confusion matrix */
				if (true_class == 1 && pred_class == 1)
					tp++;
				else if (true_class == 0 && pred_class == 0)
					tn++;
				else if (true_class == 0 && pred_class == 1)
					fp++;
				else if (true_class == 1 && pred_class == 0)
					fn++;
			}

			if (train_features)
				NDB_FREE(train_features);
			if (train_labels)
				NDB_FREE(train_labels);
		}
	}

	if (metrics)
		NDB_FREE(metrics);
	if (model_data)
		NDB_FREE(model_data);

	if (tp_out)
		*tp_out = tp;
	if (tn_out)
		*tn_out = tn;
	if (fp_out)
		*fp_out = fp;
	if (fn_out)
		*fn_out = fn;
}

/*
 * evaluate_knn_by_model_id
 *
 * Evaluates KNN model by model_id using optimized batch evaluation.
 * Supports both GPU and CPU models with GPU-accelerated batch evaluation when available.
 *
 * Returns jsonb with metrics: accuracy, precision, recall, f1_score, n_samples
 */
PG_FUNCTION_INFO_V1(evaluate_knn_by_model_id);

Datum
evaluate_knn_by_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text	   *table_name;
	text	   *feature_col;
	text	   *label_col;
	char	   *tbl_str;
	char	   *feat_str;
	char	   *targ_str;
	int			ret;
	int			nvec = 0;
	int			i;
	int			j;
	int			valid_rows = 0;
	Oid			feat_type_oid = InvalidOid;
	bool		feat_is_array = false;
	double		accuracy = 0.0;
	double		precision = 0.0;
	double		recall = 0.0;
	double		f1_score = 0.0;
	int			tp = 0;
	int			tn = 0;
	int			fp = 0;
	int			fn = 0;
	MemoryContext oldcontext;
	MemoryContext callcontext;
	StringInfoData query;
	Jsonb	   *result_jsonb = NULL;
	bytea	   *gpu_payload = NULL;
	Jsonb	   *gpu_metrics = NULL;
	bool		is_gpu_model = false;
	int			feat_dim = 0;
	NDB_DECLARE(NdbSpiSession *, spi_session);

	/*
	 * Copy strings immediately to preserve them for error messages. These
	 * strings may be needed after SPI operations complete, so we copy them to
	 * the current memory context.
	 */
	char	   *tbl_str_copy = NULL;
	char	   *feat_str_copy = NULL;
	char	   *targ_str_copy = NULL;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_knn_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_knn_by_model_id: table_name, feature_col, and label_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);
	label_col = PG_GETARG_TEXT_PP(3);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(label_col);

	oldcontext = CurrentMemoryContext;

	/*
	 * Copy strings immediately to preserve them for error messages. These
	 * strings may be needed after SPI operations complete, so we copy them to
	 * the current memory context.
	 */
	tbl_str_copy = pstrdup(tbl_str);
	feat_str_copy = pstrdup(feat_str);
	targ_str_copy = pstrdup(targ_str);

	/* Build query in caller context, before SPI */
	initStringInfo(&query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 quote_identifier(feat_str),
					 quote_identifier(targ_str),
					 quote_identifier(tbl_str),
					 quote_identifier(feat_str),
					 quote_identifier(targ_str));

	/* Load model from catalog to determine feature dimension */
	if (!ml_catalog_fetch_model_payload(model_id, &gpu_payload, NULL, &gpu_metrics))
	{
		NDB_FREE(tbl_str);
		NDB_FREE(feat_str);
		NDB_FREE(targ_str);
		NDB_FREE(tbl_str_copy);
		NDB_FREE(feat_str_copy);
		NDB_FREE(targ_str_copy);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_knn_by_model_id: model %d not found",
						model_id)));
	}

	/* Validate model payload */
	if (gpu_payload == NULL)
	{
		NDB_FREE(tbl_str);
		NDB_FREE(feat_str);
		NDB_FREE(targ_str);
		NDB_FREE(tbl_str_copy);
		NDB_FREE(feat_str_copy);
		NDB_FREE(targ_str_copy);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_knn_by_model_id: model %d has NULL payload",
						model_id)));
	}

	/* Check if GPU model and extract feature dimension */
	if (gpu_payload != NULL)
	{
		const char *base = VARDATA(gpu_payload);
		int			first_int = *(int *) base;

		if (first_int >= 1 && first_int <= 1000)
		{
			/* CPU format: k, n_samples, n_features */
			int			k_cpu,
						n_samples_cpu,
						n_features_cpu;

			memcpy(&k_cpu, base, sizeof(int));
			memcpy(&n_samples_cpu, base + sizeof(int), sizeof(int));
			memcpy(&n_features_cpu, base + sizeof(int) * 2, sizeof(int));
			feat_dim = n_features_cpu;
			is_gpu_model = false;
		}
		else
		{
			/* GPU format */
			const		NdbCudaKnnModelHeader *gpu_hdr;

			if (VARSIZE(gpu_payload) - VARHDRSZ >= (int) sizeof(NdbCudaKnnModelHeader))
			{
				gpu_hdr = (const NdbCudaKnnModelHeader *) base;
				feat_dim = gpu_hdr->n_features;
				is_gpu_model = true;
			}
			else
			{
				NDB_FREE(tbl_str);
				NDB_FREE(feat_str);
				NDB_FREE(targ_str);
				if (gpu_payload)
					NDB_FREE(gpu_payload);
				if (gpu_metrics)
					NDB_FREE(gpu_metrics);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: evaluate_knn_by_model_id: model %d has invalid header",
								model_id)));
			}
		}
	}

	if (feat_dim <= 0)
	{
		NDB_FREE(tbl_str);
		NDB_FREE(feat_str);
		NDB_FREE(targ_str);
		if (gpu_payload)
			NDB_FREE(gpu_payload);
		if (gpu_metrics)
			NDB_FREE(gpu_metrics);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_knn_by_model_id: could not determine feature dimension")));
	}

	/* Connect to SPI */
	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	/* Save SPI context for later cleanup (used in GPU path) */
	callcontext = CurrentMemoryContext;
	(void) callcontext;			/* Suppress unused warning in CPU-only paths */

	ret = ndb_spi_execute(spi_session, query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		NDB_FREE(tbl_str);
		NDB_FREE(feat_str);
		NDB_FREE(targ_str);
		if (gpu_payload)
			NDB_FREE(gpu_payload);
		if (gpu_metrics)
			NDB_FREE(gpu_metrics);
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_knn_by_model_id: query failed")));
	}

	nvec = SPI_processed;
	if (nvec < 1)
	{
		/* Check if table/view exists and has any rows at all */
		StringInfoData check_query;
		int			check_ret;
		int			total_rows = 0;

		initStringInfo(&check_query);
		appendStringInfo(&check_query,
						 "SELECT COUNT(*) FROM %s",
						 quote_identifier(tbl_str));
		check_ret = ndb_spi_execute(spi_session, check_query.data, true, 0);
		if (check_ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			int32		count_val;
			Datum		count_datum;
			bool		isnull;
			
			/* Use safe function to get int32 count (will be cast to int64) */
			if (ndb_spi_get_result_safe(0, 1, NULL, &count_datum, &isnull) && !isnull)
			{
				count_val = DatumGetInt32(count_datum);
				total_rows = (int64) count_val;
			}
		}
		NDB_FREE(check_query.data);

		/* Free original strings before SPI session end */
		NDB_FREE(tbl_str);
		NDB_FREE(feat_str);
		NDB_FREE(targ_str);
		if (gpu_payload)
			NDB_FREE(gpu_payload);
		if (gpu_metrics)
			NDB_FREE(gpu_metrics);
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);

		/* Now use copies for error messages */
		if (total_rows == 0)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_knn_by_model_id: table/view '%s' has no rows",
							tbl_str_copy),
					 errhint("Ensure the table/view exists and contains data")));
		}
		else
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_knn_by_model_id: no valid rows found in '%s' (query returned %d rows, but all were filtered out due to NULL values)",
							tbl_str_copy, total_rows),
					 errhint("Ensure columns '%s' and '%s' are not NULL",
							 feat_str_copy, targ_str_copy)));
		}
	}

	/* Determine feature column type */
	if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
		feat_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 1);
	if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
		feat_is_array = true;

	/*
	 * GPU batch evaluation path for GPU models - uses optimized evaluation
	 * kernel
	 */
	if (is_gpu_model && neurondb_gpu_is_available())
	{
#ifdef NDB_GPU_CUDA
		const		NdbCudaKnnModelHeader *gpu_hdr;
		double	   *h_labels = NULL;
		float	   *h_features = NULL;
		size_t		payload_size;

		valid_rows = 0;

		/* Defensive check: validate payload size */
		payload_size = VARSIZE(gpu_payload) - VARHDRSZ;
		if (payload_size < sizeof(NdbCudaKnnModelHeader))
		{
			elog(DEBUG1,
				 "neurondb: evaluate_knn_by_model_id: GPU payload too small (%zu bytes), falling back to CPU",
				 payload_size);
			goto cpu_evaluation_path;
		}

		/* Load GPU model header with defensive checks */
		gpu_hdr = (const NdbCudaKnnModelHeader *) VARDATA(gpu_payload);
		if (gpu_hdr == NULL)
		{
			elog(DEBUG1,
				 "neurondb: evaluate_knn_by_model_id: NULL GPU header, falling back to CPU");
			goto cpu_evaluation_path;
		}

		if (gpu_hdr->n_features != feat_dim)
		{
			elog(DEBUG1,
				 "neurondb: evaluate_knn_by_model_id: feature dimension mismatch (model has %d, expected %d), falling back to CPU",
				 gpu_hdr->n_features, feat_dim);
			goto cpu_evaluation_path;
		}

		if (gpu_hdr->n_features <= 0 || gpu_hdr->n_features > 100000)
		{
			elog(DEBUG1,
				 "evaluate_knn_by_model_id: invalid feature_dim (%d), falling back to CPU",
				 gpu_hdr->n_features);
			goto cpu_evaluation_path;
		}

		/* Allocate host buffers for features and labels with size checks */
		{
			size_t		features_size = sizeof(float) * (size_t) nvec * (size_t) feat_dim;
			size_t		labels_size = sizeof(double) * (size_t) nvec;

			if (features_size > MaxAllocSize || labels_size > MaxAllocSize)
			{
				elog(DEBUG1,
					 "neurondb: evaluate_knn_by_model_id: allocation size too large (features=%zu, labels=%zu), falling back to CPU",
					 features_size, labels_size);
				goto cpu_evaluation_path;
			}

			{
				size_t		feat_count = features_size / sizeof(float);
				size_t		label_count = labels_size / sizeof(double);
				NDB_DECLARE(float *, h_features);
				NDB_DECLARE(double *, h_labels);
				NDB_ALLOC(h_features, float, feat_count);
				NDB_ALLOC(h_labels, double, label_count);
			}

			if (h_features == NULL || h_labels == NULL)
			{
				elog(DEBUG1,
					 "neurondb: evaluate_knn_by_model_id: memory allocation failed, falling back to CPU");
				if (h_features)
					NDB_FREE(h_features);
				if (h_labels)
					NDB_FREE(h_labels);
				goto cpu_evaluation_path;
			}
		}

		/*
		 * Extract features and labels from SPI results - optimized batch
		 * extraction
		 */
		/* Cache TupleDesc to avoid repeated lookups */
		{
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;

			if (tupdesc == NULL)
			{
				elog(DEBUG1,
					 "neurondb: evaluate_knn_by_model_id: NULL TupleDesc, falling back to CPU");
				NDB_FREE(h_features);
				NDB_FREE(h_labels);
				goto cpu_evaluation_path;
			}

			for (i = 0; i < nvec; i++)
			{
				HeapTuple	tuple;
				Datum		feat_datum;
				Datum		targ_datum;
				bool		feat_null;
				bool		targ_null;
				Vector	   *vec;
				ArrayType  *arr;
				float	   *feat_row;

				if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || i >= SPI_processed)
					break;

				tuple = SPI_tuptable->vals[i];
				if (tuple == NULL)
					continue;

				feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
				targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

				if (feat_null || targ_null)
					continue;

				/* Bounds check */
				if (valid_rows >= nvec)
				{
					elog(DEBUG1,
						 "neurondb: evaluate_knn_by_model_id: valid_rows overflow, breaking");
					break;
				}

				feat_row = h_features + (valid_rows * feat_dim);
				if (feat_row == NULL || feat_row < h_features || feat_row >= h_features + (nvec * feat_dim))
				{
					elog(DEBUG1,
						 "neurondb: evaluate_knn_by_model_id: feat_row out of bounds, skipping row");
					continue;
				}

				h_labels[valid_rows] = DatumGetFloat8(targ_datum);

				/* Extract feature vector - optimized paths */
				if (feat_is_array)
				{
					arr = DatumGetArrayTypeP(feat_datum);
					if (ARR_NDIM(arr) != 1 || ARR_DIMS(arr)[0] != feat_dim)
						continue;
					if (feat_type_oid == FLOAT8ARRAYOID)
					{
						/* Optimized: bulk conversion with loop unrolling hint */
						float8	   *data = (float8 *) ARR_DATA_PTR(arr);
						int			j_remain = feat_dim % 4;
						int			j_end = feat_dim - j_remain;

						/*
						 * Process 4 elements at a time for better cache
						 * locality
						 */
						for (j = 0; j < j_end; j += 4)
						{
							feat_row[j] = (float) data[j];
							feat_row[j + 1] = (float) data[j + 1];
							feat_row[j + 2] = (float) data[j + 2];
							feat_row[j + 3] = (float) data[j + 3];
						}
						/* Handle remaining elements */
						for (j = j_end; j < feat_dim; j++)
							feat_row[j] = (float) data[j];
					}
					else
					{
						/* FLOAT4ARRAYOID: direct memcpy (already optimal) */
						float4	   *data = (float4 *) ARR_DATA_PTR(arr);

						memcpy(feat_row, data, sizeof(float) * feat_dim);
					}
				}
				else
				{
					/* Vector type: direct memcpy (already optimal) */
					vec = DatumGetVector(feat_datum);
					if (vec->dim != feat_dim)
						continue;
					memcpy(feat_row, vec->data, sizeof(float) * feat_dim);
				}

				valid_rows++;
			}
		}

		if (valid_rows == 0)
		{
			NDB_FREE(h_features);
			NDB_FREE(h_labels);
			if (gpu_payload)
				NDB_FREE(gpu_payload);
			if (gpu_metrics)
				NDB_FREE(gpu_metrics);
			NDB_FREE(tbl_str);
			NDB_FREE(feat_str);
			NDB_FREE(targ_str);
			ndb_spi_stringinfo_free(spi_session, &query);
			NDB_SPI_SESSION_END(spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_knn_by_model_id: no valid rows found")));
		}

		/* Use optimized GPU batch evaluation */
		{
			int			rc;
			char	   *gpu_errstr = NULL;

			/* Defensive checks before GPU call */
			if (h_features == NULL || h_labels == NULL || valid_rows <= 0 || feat_dim <= 0)
			{
				elog(DEBUG1,
					 "neurondb: evaluate_knn_by_model_id: invalid inputs for GPU evaluation (features=%p, labels=%p, rows=%d, dim=%d), falling back to CPU",
					 (void *) h_features, (void *) h_labels, valid_rows, feat_dim);
				NDB_FREE(h_features);
				NDB_FREE(h_labels);
				goto cpu_evaluation_path;
			}

			PG_TRY();
			{
				rc = ndb_cuda_knn_evaluate_batch(gpu_payload,
												 h_features,
												 h_labels,
												 valid_rows,
												 feat_dim,
												 &accuracy,
												 &precision,
												 &recall,
												 &f1_score,
												 &gpu_errstr);

				if (rc == 0)
				{
					/* Success - build result and return */
					StringInfoData jsonbuf;
					initStringInfo(&jsonbuf);
					appendStringInfo(&jsonbuf,
									 "{\"accuracy\":%.6f,\"precision\":%.6f,\"recall\":%.6f,\"f1_score\":%.6f,\"n_samples\":%d}",
									 accuracy,
									 precision,
									 recall,
									 f1_score,
									 valid_rows);

					/* Use ndb_jsonb_in_cstring like other ML algorithms fix */
					result_jsonb = ndb_jsonb_in_cstring(jsonbuf.data);
					if (result_jsonb == NULL)
					{
						NDB_FREE(jsonbuf.data);
						NDB_FREE(h_features);
						NDB_FREE(h_labels);
						if (gpu_payload)
							NDB_FREE(gpu_payload);
						if (gpu_metrics)
							NDB_FREE(gpu_metrics);
						if (gpu_errstr)
							NDB_FREE(gpu_errstr);
						NDB_FREE(tbl_str);
						NDB_FREE(feat_str);
						NDB_FREE(targ_str);
						ndb_spi_stringinfo_free(eval_spi_session, &query);
						NDB_SPI_SESSION_END(eval_spi_session);
						MemoryContextSwitchTo(oldcontext);
						ereport(ERROR,
								(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
								 errmsg("neurondb: evaluate_knn_by_model_id: failed to parse metrics JSON")));
					}
					NDB_FREE(jsonbuf.data);

					MemoryContextSwitchTo(oldcontext);
					result_jsonb = (Jsonb *) PG_DETOAST_DATUM_COPY(JsonbPGetDatum(result_jsonb));

					/* Now safe to finish SPI and free SPI-allocated memory */
					MemoryContextSwitchTo(callcontext);
					NDB_FREE(h_features);
					NDB_FREE(h_labels);
					if (gpu_payload)
						NDB_FREE(gpu_payload);
					if (gpu_metrics)
						NDB_FREE(gpu_metrics);

					/*
					 * Note: gpu_errstr is allocated by GPU function, don't
					 * free it
					 */
					ndb_spi_stringinfo_free(spi_session, &query);
					NDB_SPI_SESSION_END(spi_session);

					/*
					 * Switch back to oldcontext and free strings allocated
					 * before SPI session begin
					 */
					MemoryContextSwitchTo(oldcontext);
					NDB_FREE(tbl_str);
					NDB_FREE(feat_str);
					NDB_FREE(targ_str);

					PG_RETURN_JSONB_P(result_jsonb);
				}
				else
				{
					/* GPU evaluation failed - fall back to CPU */
					elog(DEBUG1,
						 "neurondb: evaluate_knn_by_model_id: GPU batch evaluation failed: %s, falling back to CPU",
						 gpu_errstr ? gpu_errstr : "unknown error");
					if (gpu_errstr)
						NDB_FREE(gpu_errstr);
					NDB_FREE(h_features);
					NDB_FREE(h_labels);
					goto cpu_evaluation_path;
				}
			}
			PG_CATCH();
			{
				elog(DEBUG1,
					 "neurondb: evaluate_knn_by_model_id: exception during GPU evaluation, falling back to CPU");
				if (h_features)
					NDB_FREE(h_features);
				if (h_labels)
					NDB_FREE(h_labels);
				goto cpu_evaluation_path;
			}
			PG_END_TRY();
		}
#endif							/* NDB_GPU_CUDA */
	}
#ifndef NDB_GPU_CUDA
	/* When CUDA is not available, always use CPU path */
	if (false)
	{
	}
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-label"
cpu_evaluation_path:
#pragma GCC diagnostic pop

	/* CPU evaluation path */
	/* Use optimized batch prediction */
	{
		float	   *h_features = NULL;
		double	   *h_labels = NULL;

		valid_rows = 0;

		/* Allocate host buffers for features and labels */
		NDB_ALLOC(h_features, float, (size_t) nvec * (size_t) feat_dim);
		NDB_ALLOC(h_labels, double, (size_t) nvec);

		/*
		 * Extract features and labels from SPI results - optimized batch
		 * extraction
		 */
		/* Cache TupleDesc to avoid repeated lookups */
		{
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;

			for (i = 0; i < nvec; i++)
			{
				HeapTuple	tuple = SPI_tuptable->vals[i];
				Datum		feat_datum;
				Datum		targ_datum;
				bool		feat_null;
				bool		targ_null;
				Vector	   *vec;
				ArrayType  *arr;
				float	   *feat_row;

				feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
				targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

				if (feat_null || targ_null)
					continue;

				feat_row = h_features + (valid_rows * feat_dim);
				h_labels[valid_rows] = DatumGetFloat8(targ_datum);

				/* Extract feature vector - optimized paths */
				if (feat_is_array)
				{
					arr = DatumGetArrayTypeP(feat_datum);
					if (ARR_NDIM(arr) != 1 || ARR_DIMS(arr)[0] != feat_dim)
						continue;
					if (feat_type_oid == FLOAT8ARRAYOID)
					{
						/* Optimized: bulk conversion with loop unrolling hint */
						float8	   *data = (float8 *) ARR_DATA_PTR(arr);
						int			j_remain = feat_dim % 4;
						int			j_end = feat_dim - j_remain;

						/*
						 * Process 4 elements at a time for better cache
						 * locality
						 */
						for (j = 0; j < j_end; j += 4)
						{
							feat_row[j] = (float) data[j];
							feat_row[j + 1] = (float) data[j + 1];
							feat_row[j + 2] = (float) data[j + 2];
							feat_row[j + 3] = (float) data[j + 3];
						}
						/* Handle remaining elements */
						for (j = j_end; j < feat_dim; j++)
							feat_row[j] = (float) data[j];
					}
					else
					{
						/* FLOAT4ARRAYOID: direct memcpy (already optimal) */
						float4	   *data = (float4 *) ARR_DATA_PTR(arr);

						memcpy(feat_row, data, sizeof(float) * feat_dim);
					}
				}
				else
				{
					/* Vector type: direct memcpy (already optimal) */
					vec = DatumGetVector(feat_datum);
					if (vec->dim != feat_dim)
						continue;
					memcpy(feat_row, vec->data, sizeof(float) * feat_dim);
				}

				valid_rows++;
			}
		}

		if (valid_rows == 0)
		{
			/* Check if we had rows but they were all filtered out */
			/* Safe access for complex types - validate before access */
			if (nvec > 0 && SPI_tuptable != NULL && SPI_tuptable->vals != NULL && 
				SPI_tuptable->vals[0] != NULL && SPI_tuptable->tupdesc != NULL)
			{
				/* Try to determine why rows were filtered */
				HeapTuple	tuple = SPI_tuptable->vals[0];
				TupleDesc	tupdesc = SPI_tuptable->tupdesc;
				Datum		feat_datum;
				bool		feat_null;
				Vector	   *vec;
				int			actual_dim = 0;

				feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
				if (!feat_null)
				{
					if (feat_is_array)
					{
						ArrayType  *arr = DatumGetArrayTypeP(feat_datum);

						if (ARR_NDIM(arr) == 1)
							actual_dim = ARR_DIMS(arr)[0];
					}
					else
					{
						vec = DatumGetVector(feat_datum);
						if (vec != NULL)
							actual_dim = vec->dim;
					}
				}

				NDB_FREE(h_features);
				NDB_FREE(h_labels);
				if (gpu_payload)
					NDB_FREE(gpu_payload);
				if (gpu_metrics)
					NDB_FREE(gpu_metrics);
				NDB_FREE(tbl_str);
				NDB_FREE(feat_str);
				NDB_FREE(targ_str);
				ndb_spi_stringinfo_free(spi_session, &query);
				NDB_SPI_SESSION_END(spi_session);
				if (actual_dim > 0 && actual_dim != feat_dim)
				{
					ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							 errmsg("neurondb: evaluate_knn_by_model_id: feature dimension mismatch"),
							 errdetail("Model expects %d features but test data has %d features. All %d rows were filtered out due to dimension mismatch.",
									   feat_dim,
									   actual_dim,
									   nvec)));
				}
				else
				{
					ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							 errmsg("neurondb: evaluate_knn_by_model_id: no valid rows found"),
							 errdetail("Query returned %d rows but all were filtered out (NULL values or dimension mismatches).",
									   nvec)));
				}
			}
			else
			{
				NDB_FREE(h_features);
				NDB_FREE(h_labels);
				if (gpu_payload)
					NDB_FREE(gpu_payload);
				if (gpu_metrics)
					NDB_FREE(gpu_metrics);
				NDB_FREE(tbl_str);
				NDB_FREE(feat_str);
				NDB_FREE(targ_str);
				ndb_spi_stringinfo_free(spi_session, &query);
				NDB_SPI_SESSION_END(spi_session);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: evaluate_knn_by_model_id: no valid rows found"),
						 errdetail("Query returned 0 rows from table '%s'.", tbl_str_copy)));
			}
		}

		/* Use batch prediction helper */
		knn_predict_batch(model_id,
						  h_features,
						  h_labels,
						  valid_rows,
						  feat_dim,
						  &tp,
						  &tn,
						  &fp,
						  &fn);

		/* Compute metrics */
		if (valid_rows > 0)
		{
			int			total_predictions = tp + tn + fp + fn;

			/* Defensive check: ensure we have predictions */
			if (total_predictions == 0)
			{
				elog(DEBUG1,
					 "neurondb: evaluate_knn_by_model_id: no valid predictions made for %d valid rows (model_id=%d, feature_dim=%d)",
					 valid_rows, model_id, feat_dim);
				accuracy = 0.0;
				precision = 0.0;
				recall = 0.0;
				f1_score = 0.0;
			}
			else
			{
				/* Use valid_rows for accuracy (original behavior) */
				accuracy = (double) (tp + tn) / (double) valid_rows;

				/* Warn if some rows were skipped during prediction */
				if (total_predictions < valid_rows)
				{
					elog(DEBUG1,
						 "neurondb: evaluate_knn_by_model_id: %d rows skipped during prediction (labels out of range?)",
						 valid_rows - total_predictions);
				}

				if ((tp + fp) > 0)
					precision = (double) tp / (double) (tp + fp);
				else
					precision = 0.0;

				if ((tp + fn) > 0)
					recall = (double) tp / (double) (tp + fn);
				else
					recall = 0.0;

				if ((precision + recall) > 0.0)
					f1_score = 2.0 * (precision * recall) / (precision + recall);
				else
					f1_score = 0.0;
			}
		}

		NDB_FREE(h_features);
		NDB_FREE(h_labels);
		if (gpu_payload)
			NDB_FREE(gpu_payload);
		if (gpu_metrics)
			NDB_FREE(gpu_metrics);
	}

	/* End SPI session BEFORE creating JSONB to avoid context conflicts */
	ndb_spi_stringinfo_free(spi_session, &query);
	NDB_SPI_SESSION_END(spi_session);

	/* Switch to old context and build JSONB directly using JSONB API */
	MemoryContextSwitchTo(oldcontext);
	{
		JsonbParseState *state = NULL;
		JsonbValue	jkey;
		JsonbValue	jval;
		JsonbValue *final_value = NULL;
		Numeric		accuracy_num, precision_num, recall_num, f1_score_num, n_samples_num;
		int			n_samples_val = valid_rows > 0 ? valid_rows : nvec;

		/* Start object */
		PG_TRY();
		{
			(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

			/* Add accuracy */
			jkey.type = jbvString;
			jkey.val.string.len = 9;
			jkey.val.string.val = "accuracy";
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			accuracy_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(accuracy)));
			jval.type = jbvNumeric;
			jval.val.numeric = accuracy_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			/* Add precision */
			jkey.val.string.val = "precision";
			jkey.val.string.len = 9;
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			precision_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(precision)));
			jval.type = jbvNumeric;
			jval.val.numeric = precision_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			/* Add recall */
			jkey.val.string.val = "recall";
			jkey.val.string.len = 6;
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			recall_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(recall)));
			jval.type = jbvNumeric;
			jval.val.numeric = recall_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			/* Add f1_score */
			jkey.val.string.val = "f1_score";
			jkey.val.string.len = 8;
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			f1_score_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(f1_score)));
			jval.type = jbvNumeric;
			jval.val.numeric = f1_score_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			/* Add n_samples */
			jkey.val.string.val = "n_samples";
			jkey.val.string.len = 9;
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			n_samples_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(n_samples_val)));
			jval.type = jbvNumeric;
			jval.val.numeric = n_samples_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			/* End object */
			final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);
			
			if (final_value == NULL)
			{
				elog(ERROR, "neurondb: evaluate_knn: pushJsonbValue(WJB_END_OBJECT) returned NULL");
			}
			
			result_jsonb = JsonbValueToJsonb(final_value);
		}
		PG_CATCH();
		{
			ErrorData *edata = CopyErrorData();
			elog(ERROR, "neurondb: evaluate_knn: JSONB construction failed: %s", edata->message);
			FlushErrorState();
			result_jsonb = NULL;
		}
		PG_END_TRY();
	}

	if (result_jsonb == NULL)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_knn_by_model_id: JSONB result is NULL")));
	}

	/* Free strings allocated before SPI session begin (in oldcontext) */
	NDB_FREE(tbl_str);
	NDB_FREE(feat_str);
	NDB_FREE(targ_str);

	/* Free string copies used for error messages */
	if (tbl_str_copy)
		NDB_FREE(tbl_str_copy);
	if (feat_str_copy)
		NDB_FREE(feat_str_copy);
	if (targ_str_copy)
		NDB_FREE(targ_str_copy);

	PG_RETURN_JSONB_P(result_jsonb);
}

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration for KNN
 *-------------------------------------------------------------------------
 */

#ifdef NDB_GPU_CUDA

/* GPU Model State */
typedef struct KnnGpuModelState
{
	bytea	   *model_blob;
	Jsonb	   *metrics;
	int			feature_dim;
	int			n_samples;
	int			k;
	int			task_type;
}			KnnGpuModelState;

static void
knn_gpu_release_state(KnnGpuModelState * state)
{
	if (state == NULL)
		return;
	if (state->model_blob != NULL)
		NDB_FREE(state->model_blob);
	if (state->metrics != NULL)
		NDB_FREE(state->metrics);
	NDB_FREE(state);
}

static bool
knn_gpu_train(MLGpuModel * model, const MLGpuTrainSpec * spec, char **errstr)
{
	KnnGpuModelState *state;
	bytea	   *payload;
	Jsonb	   *metrics;
	int			rc;
	int			k = 5;
	int			task_type = 0;	/* Default to classification */
	const		ndb_gpu_backend *backend;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || spec == NULL)
		return false;
	if (!neurondb_gpu_is_available())
		return false;
	if (spec->feature_matrix == NULL || spec->label_vector == NULL)
		return false;
	if (spec->sample_count <= 0 || spec->feature_dim <= 0)
		return false;

	backend = ndb_gpu_get_active_backend();
	if (backend == NULL || backend->knn_train == NULL)
		return false;

	/* Extract hyperparameters */
	if (spec->hyperparameters != NULL)
	{
		Datum		k_datum;
		Datum		task_type_datum;
		Datum		numeric_datum;
		Numeric		num;

		k_datum = DirectFunctionCall2(
									  jsonb_object_field,
									  JsonbPGetDatum(spec->hyperparameters),
									  CStringGetTextDatum("k"));
		if (DatumGetPointer(k_datum) != NULL)
		{
			numeric_datum = DirectFunctionCall1(
												jsonb_numeric, k_datum);
			if (DatumGetPointer(numeric_datum) != NULL)
			{
				num = DatumGetNumeric(numeric_datum);
				k = DatumGetInt32(
								  DirectFunctionCall1(numeric_int4,
													  NumericGetDatum(num)));
				if (k <= 0 || k > spec->sample_count)
					k = (spec->sample_count < 10) ? spec->sample_count : 10;
			}
		}

		task_type_datum = DirectFunctionCall2(
											  jsonb_object_field,
											  JsonbPGetDatum(spec->hyperparameters),
											  CStringGetTextDatum("task_type"));
		if (DatumGetPointer(task_type_datum) != NULL)
		{
			numeric_datum = DirectFunctionCall1(
												jsonb_numeric, task_type_datum);
			if (DatumGetPointer(numeric_datum) != NULL)
			{
				num = DatumGetNumeric(numeric_datum);
				task_type = DatumGetInt32(
										  DirectFunctionCall1(numeric_int4,
															  NumericGetDatum(num)));
				if (task_type != 0 && task_type != 1)
					task_type = 0;
			}
		}
	}

	payload = NULL;
	metrics = NULL;

	rc = backend->knn_train(spec->feature_matrix,
							spec->label_vector,
							spec->sample_count,
							spec->feature_dim,
							k,
							task_type,
							spec->hyperparameters,
							&payload,
							&metrics,
							errstr);
	if (rc != 0 || payload == NULL)
	{
		if (payload != NULL)
			NDB_FREE(payload);
		if (metrics != NULL)
			NDB_FREE(metrics);
		return false;
	}

	if (model->backend_state != NULL)
	{
		knn_gpu_release_state((KnnGpuModelState *) model->backend_state);
		model->backend_state = NULL;
	}

	NDB_DECLARE(KnnGpuModelState *, state);
	NDB_ALLOC(state, KnnGpuModelState, 1);
	state->model_blob = payload;
	state->feature_dim = spec->feature_dim;
	state->n_samples = spec->sample_count;
	state->k = k;
	state->task_type = task_type;

	if (metrics != NULL)
	{
		state->metrics = (Jsonb *) PG_DETOAST_DATUM_COPY(
														 PointerGetDatum(metrics));
	}
	else
	{
		state->metrics = NULL;
	}

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;

	return true;
}

static bool
knn_gpu_predict(const MLGpuModel * model,
				const float *input,
				int input_dim,
				float *output,
				int output_dim,
				char **errstr)
{
	const		KnnGpuModelState *state;
	double		prediction;
	int			rc;
	const		ndb_gpu_backend *backend;

	if (errstr != NULL)
		*errstr = NULL;
	if (output != NULL && output_dim > 0)
		output[0] = 0.0f;
	if (model == NULL || input == NULL || output == NULL)
		return false;
	if (output_dim <= 0)
		return false;
	if (!model->gpu_ready || model->backend_state == NULL)
		return false;

	state = (const KnnGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
		return false;

	backend = ndb_gpu_get_active_backend();
	if (backend == NULL || backend->knn_predict == NULL)
		return false;

	rc = backend->knn_predict(state->model_blob,
							  input,
							  state->feature_dim > 0 ? state->feature_dim : input_dim,
							  &prediction,
							  errstr);
	if (rc != 0)
		return false;

	output[0] = (float) prediction;

	return true;
}

static bool
knn_gpu_evaluate(const MLGpuModel * model,
				 const MLGpuEvalSpec * spec,
				 MLGpuMetrics * out,
				 char **errstr)
{
	const		KnnGpuModelState *state;
	Jsonb	   *metrics_json;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || out == NULL)
		return false;
	if (model->backend_state == NULL)
		return false;

	state = (const KnnGpuModelState *) model->backend_state;

	{
		StringInfoData buf;

		initStringInfo(&buf);
		appendStringInfo(&buf,
						 "{\"algorithm\":\"knn\","
						 "\"storage\":\"gpu\","
						 "\"n_features\":%d,"
						 "\"n_samples\":%d,"
						 "\"k\":%d,"
						 "\"task_type\":%d}",
						 state->feature_dim > 0 ? state->feature_dim : 0,
						 state->n_samples > 0 ? state->n_samples : 0,
						 state->k,
						 state->task_type);

		PG_TRY();
		{
			metrics_json = DatumGetJsonbP(DirectFunctionCall1(
															  jsonb_in, CStringGetTextDatum(buf.data)));
		}
		PG_CATCH();
		{
			FlushErrorState();
			metrics_json = NULL;
		}
		PG_END_TRY();
		NDB_FREE(buf.data);
	}

	if (out != NULL)
		out->payload = metrics_json;

	return true;
}

static bool
knn_gpu_serialize(const MLGpuModel * model,
				  bytea * *payload_out,
				  Jsonb * *metadata_out,
				  char **errstr)
{
	const		KnnGpuModelState *state;
	bytea	   *payload_copy;
	int			payload_size;

	if (errstr != NULL)
		*errstr = NULL;
	if (payload_out != NULL)
		*payload_out = NULL;
	if (metadata_out != NULL)
		*metadata_out = NULL;
	if (model == NULL || model->backend_state == NULL)
		return false;

	state = (const KnnGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
		return false;

	payload_size = VARSIZE(state->model_blob);
	{
		NDB_DECLARE(char *, payload_bytes);
		NDB_ALLOC(payload_bytes, char, payload_size);
		payload_copy = (bytea *) payload_bytes;
	}
	memcpy(payload_copy, state->model_blob, payload_size);

	if (payload_out != NULL)
		*payload_out = payload_copy;
	else
		NDB_FREE(payload_copy);

	if (metadata_out != NULL && state->metrics != NULL)
	{
		*metadata_out = (Jsonb *) PG_DETOAST_DATUM_COPY(
														PointerGetDatum(state->metrics));
	}
	else if (metadata_out != NULL)
	{
		*metadata_out = NULL;
	}

	return true;
}

static bool
knn_gpu_deserialize(MLGpuModel * model,
					const bytea * payload,
					const Jsonb * metadata,
					char **errstr)
{
	KnnGpuModelState *state;
	bytea	   *payload_copy;
	int			payload_size;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
		return false;

	payload_size = VARSIZE(payload);
	{
		NDB_DECLARE(char *, payload_bytes);
		NDB_ALLOC(payload_bytes, char, payload_size);
		payload_copy = (bytea *) payload_bytes;
	}
	memcpy(payload_copy, payload, payload_size);

	NDB_DECLARE(KnnGpuModelState *, state);
	NDB_ALLOC(state, KnnGpuModelState, 1);
	state->model_blob = payload_copy;
	state->feature_dim = -1;
	state->n_samples = -1;
	state->k = -1;
	state->task_type = -1;

	if (model->backend_state != NULL)
		knn_gpu_release_state((KnnGpuModelState *) model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;

	return true;
}

static void
knn_gpu_destroy(MLGpuModel * model)
{
	if (model == NULL)
		return;
	if (model->backend_state != NULL)
		knn_gpu_release_state((KnnGpuModelState *) model->backend_state);
	model->backend_state = NULL;
	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

static const MLGpuModelOps knn_gpu_model_ops = {
	.algorithm = "knn",
	.train = knn_gpu_train,
	.predict = knn_gpu_predict,
	.evaluate = knn_gpu_evaluate,
	.serialize = knn_gpu_serialize,
	.deserialize = knn_gpu_deserialize,
	.destroy = knn_gpu_destroy,
};

#endif							/* NDB_GPU_CUDA */

void
neurondb_gpu_register_knn_model(void)
{
#ifdef NDB_GPU_CUDA
	static bool registered = false;

	if (registered)
		return;

	ndb_gpu_register_model_ops(&knn_gpu_model_ops);
	registered = true;
#else
	/* GPU not available - registration is a no-op */
	return;
#endif
}
