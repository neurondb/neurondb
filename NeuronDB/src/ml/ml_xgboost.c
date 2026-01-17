/*-------------------------------------------------------------------------
 *
 * ml_xgboost.c
 *    XGBoost gradient boosting integration.
 *
 * This module provides XGBoost gradient boosting for classification and
 * regression with model serialization and catalog storage.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_xgboost.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"
#include "access/htup_details.h"
#include "utils/memutils.h"
#include "lib/stringinfo.h"

#include "neurondb_spi.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_json.h"
#include "ml_catalog.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Check if XGBoost C API is available */
#if __has_include(<xgboost/c_api.h>)
#include <xgboost/c_api.h>
#define HAVE_XGBOOST 1
#else
#define HAVE_XGBOOST 0
#endif

PG_FUNCTION_INFO_V1(train_xgboost_classifier);
PG_FUNCTION_INFO_V1(train_xgboost_regressor);
PG_FUNCTION_INFO_V1(predict_xgboost);
PG_FUNCTION_INFO_V1(evaluate_xgboost_by_model_id);

#if HAVE_XGBOOST

/*
 * Load feature matrix and label vector from table using SPI.
 */
static void
load_training_data(const char *table,
	const char *feature_col,
	const char *label_col,
	float **out_features,
	float **out_labels,
	int *out_nrows,
	int *out_ncols)
{
	ArrayType  *feat_arr = NULL;
	bool		isnull;
	Datum		feat_datum;
	float	   *features = NULL;
	float	   *labels = NULL;
	HeapTuple	tuple;
	int			i;
	int			j;
	int			ncols;
	int			nrows;
	int			ret;
	MemoryContext oldcontext;
	NdbSpiSession *spi_session = NULL;
	StringInfoData query;
	TupleDesc	tupdesc;

	initStringInfo(&query);

	/* Construct query to select feature and label columns */
	appendStringInfo(
		&query, "SELECT %s, %s FROM %s", feature_col, label_col, table);
	oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	ndb_spi_stringinfo_init(spi_session, &query);
	appendStringInfo(
		&query, "SELECT %s, %s FROM %s", feature_col, label_col, table);

	ret = ndb_spi_execute(spi_session, query.data, true, 0);

	if (ret != SPI_OK_SELECT)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
				errmsg("neurondb: SPI_execute failed for training data")));

	nrows = SPI_processed;
	if (nrows <= 0)
		ereport(ERROR,
			(errcode(ERRCODE_INSUFFICIENT_RESOURCES),
				errmsg("neurondb: no training rows found")));

	/*
	 * Determine dimension (features can be a vector column).
	 * We expect features to be either PostgreSQL array or a single float column.
	 * We'll support 1-D and N-D, check first row.
	 */
	/* Safe access for complex types - validate before access */
	if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || 
		SPI_processed == 0 || SPI_tuptable->vals[0] == NULL || SPI_tuptable->tupdesc == NULL)
	{
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: null feature vector found")));
	}
	tupdesc = SPI_tuptable->tupdesc;
	tuple = SPI_tuptable->vals[0];

	feat_datum = SPI_getbinval(tuple, tupdesc, 1, &isnull);
	if (isnull)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				errmsg("neurondb: null feature vector found")));

	if (TupleDescAttr(tupdesc, 0)->atttypid == FLOAT4ARRAYOID
		|| TupleDescAttr(tupdesc, 0)->atttypid == FLOAT8ARRAYOID)
	{
		feat_arr = DatumGetArrayTypeP(feat_datum);
		ncols = ArrayGetNItems(ARR_NDIM(feat_arr), ARR_DIMS(feat_arr));
	} else
	{
		ncols = 1;
	}

	nalloc(features, float, nrows * ncols);
	nalloc(labels, float, nrows);

	for (i = 0; i < nrows; i++)
	{
		ArrayType  *curr_arr = NULL;
		bool		isnull_feat;
		bool		isnull_label;
		Datum		featval;
		Datum		labelval;
		float8	   *fdat = NULL;
		HeapTuple	current_tuple;
		int			arr_len;

		/* Safe access to SPI_tuptable - validate before access */
		if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || 
			i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
		{
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: null feature vector in row %d", i)));
		}
		current_tuple = SPI_tuptable->vals[i];

		/* Features */
		featval = SPI_getbinval(current_tuple, tupdesc, 1, &isnull_feat);
		if (isnull_feat)
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: null feature vector in row %d", i)));

		if (feat_arr)
		{
			curr_arr = DatumGetArrayTypeP(featval);

			if (ARR_NDIM(curr_arr) == 1)
			{
				arr_len = ArrayGetNItems(
					ARR_NDIM(curr_arr), ARR_DIMS(curr_arr));
				if (arr_len != ncols)
					ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							errmsg("neurondb: unexpected dimension of feature array")));
				fdat = (float8 *)ARR_DATA_PTR(curr_arr);
				for (j = 0; j < ncols; j++)
					features[i * ncols + j] =
						(float)fdat[j];
			} else
			{
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("neurondb: feature arrays must be 1D")));
			}
		} else
		{
			if (TupleDescAttr(tupdesc, 0)->atttypid == FLOAT8OID)
				features[i * ncols] =
					(float)DatumGetFloat8(featval);
			else if (TupleDescAttr(tupdesc, 0)->atttypid == FLOAT4OID)
				features[i * ncols] =
					(float)DatumGetFloat4(featval);
			else
				elog(ERROR, "Unsupported feature column type");
		}

		/* Labels - safe access for label - validate tupdesc has at least 2 columns */
		if (tupdesc->natts < 2)
		{
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: null label/target in row %d", i)));
		}
		labelval = SPI_getbinval(current_tuple, tupdesc, 2, &isnull_label);
		if (isnull_label)
			ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					errmsg("neurondb: null label/target in row %d", i)));

		if (TupleDescAttr(tupdesc, 1)->atttypid == INT4OID)
			labels[i] = (float)DatumGetInt32(labelval);
		else if (TupleDescAttr(tupdesc, 1)->atttypid == FLOAT4OID)
			labels[i] = (float)DatumGetFloat4(labelval);
		else if (TupleDescAttr(tupdesc, 1)->atttypid == FLOAT8OID)
			labels[i] = (float)DatumGetFloat8(labelval);
		else
			elog(ERROR, "Unsupported label/target column type");
	}

	*out_features = features;
	*out_labels = labels;
	*out_nrows = nrows;
	*out_ncols = ncols;

	ndb_spi_stringinfo_free(spi_session, &query);
	NDB_SPI_SESSION_END(spi_session);
}

/*
 * Save XGBoost model binary to catalog using unified API.
 */
static int32
store_xgboost_model(const void *model_bytes, size_t model_len,
					const char *table_name, const char *feature_col,
					const char *label_col, int n_estimators, int max_depth,
					float learning_rate, int n_samples, int n_features,
					const char *objective)
{
	bytea *model_data = NULL;
	Jsonb *metrics = NULL;
	StringInfoData metricsbuf;
	MLCatalogModelSpec spec;
	int32 model_id;
	char *model_data_bytes = NULL;
	size_t total_size;

	/* Convert model_bytes to bytea */
	total_size = VARHDRSZ + model_len;
	nalloc(model_data_bytes, char, total_size);
	model_data = (bytea *) model_data_bytes;
	SET_VARSIZE(model_data, total_size);
	memcpy(VARDATA(model_data), model_bytes, model_len);

	/* Build metrics JSON */
	initStringInfo(&metricsbuf);
	appendStringInfo(&metricsbuf,
					 "{\"algorithm\":\"xgboost\","
					 "\"training_backend\":0,"
					 "\"n_estimators\":%d,"
					 "\"max_depth\":%d,"
					 "\"learning_rate\":%.6f,"
					 "\"n_samples\":%d,"
					 "\"n_features\":%d,"
					 "\"objective\":\"%s\"}",
					 n_estimators,
					 max_depth,
					 learning_rate,
					 n_samples,
					 n_features,
					 objective ? objective : "reg:squarederror");

	metrics = ndb_jsonb_in_cstring(metricsbuf.data);
	if (metrics == NULL)
	{
		nfree(metricsbuf.data);
		nfree(model_data);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("neurondb: failed to parse metrics JSON for XGBoost model")));
	}
	nfree(metricsbuf.data);

	/* Register in catalog */
	memset(&spec, 0, sizeof(MLCatalogModelSpec));
	spec.algorithm = "xgboost";
	spec.model_type = (strcmp(objective, "multi:softmax") == 0) ? "classification" : "regression";
	spec.training_table = table_name;
	spec.training_column = label_col;
	spec.model_data = model_data;
	spec.metrics = metrics;
	spec.num_samples = n_samples;
	spec.num_features = n_features;

	model_id = ml_catalog_register_model(&spec);

	return model_id;
}

/*
 * Retrieve XGBoost model binary from catalog using unified API.
 */
static void *
fetch_xgboost_model(int32 model_id, size_t *model_size)
{
	bytea *model_data = NULL;
	Jsonb *parameters = NULL;
	Jsonb *metrics = NULL;
	void *data = NULL;
	size_t len;

	if (model_size == NULL)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: fetch_xgboost_model: model_size is NULL")));
	}

	/* Fetch model from catalog */
	if (!ml_catalog_fetch_model_payload(model_id, &model_data, &parameters, &metrics))
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: XGBoost model with id %d not found in catalog",
						model_id)));
	}

	if (model_data == NULL)
	{
		if (metrics != NULL)
			nfree(metrics);
		if (parameters != NULL)
			nfree(parameters);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: XGBoost model with id %d has NULL model_data",
						model_id)));
	}

	len = VARSIZE(model_data) - VARHDRSZ;
	char *data_bytes = NULL;
	nalloc(data_bytes, char, len);
	data = data_bytes;
	memcpy(data, VARDATA(model_data), len);

	*model_size = len;

	/* Clean up */
	nfree(model_data);
	if (metrics != NULL)
		nfree(metrics);
	if (parameters != NULL)
		nfree(parameters);

	return data;
}

/*
 * train_xgboost_classifier() - Train XGBoost classifier
 * Parameters:
 *   table_name TEXT - Training data table
 *   feature_col TEXT - Feature column name
 *   label_col TEXT - Label column name
 *   n_estimators INTEGER - Number of trees (default 100)
 *   max_depth INTEGER - Maximum tree depth (default 6)
 *   learning_rate FLOAT - Learning rate (default 0.3)
 * Returns: model_id INTEGER (stored in ml_models)
 */
Datum
train_xgboost_classifier(PG_FUNCTION_ARGS)
{
	text *table_name = PG_GETARG_TEXT_PP(0);
	text *feature_col = PG_GETARG_TEXT_PP(1);
	text *label_col = PG_GETARG_TEXT_PP(2);
	int32 n_estimators = PG_ARGISNULL(3) ? 100 : PG_GETARG_INT32(3);
	int32 max_depth = PG_ARGISNULL(4) ? 6 : PG_GETARG_INT32(4);
	float8 learning_rate = PG_ARGISNULL(5) ? 0.3 : PG_GETARG_FLOAT8(5);

	char *table_str = text_to_cstring(table_name);
	char *feature_str = text_to_cstring(feature_col);
	char *label_str = text_to_cstring(label_col);

	float *features = NULL;
	float *labels = NULL;
	int nrows = 0;
	int ncols = 0;
	DMatrixHandle dtrain = NULL;
	BoosterHandle booster = NULL;
	char num_class_str[16];
	char eta_str[32];
	char md_str[16];
	const char *keys[6];
	const char *vals[6];
	int param_count = 6;
	int i, iter;
	float max_label = 0.0f;
	int num_class;
	bst_ulong out_len = 0;
	char *out_bytes = NULL;
	int32 model_id;

	load_training_data(table_str,
		feature_str,
		label_str,
		&features,
		&labels,
		&nrows,
		&ncols);

	if (XGDMatrixCreateFromMat(features, nrows, ncols, (float)NAN, &dtrain)
		!= 0)
		elog(ERROR, "Failed to create DMatrix");

	if (XGDMatrixSetFloatInfo(dtrain, "label", labels, nrows) != 0)
		elog(ERROR, "Failed to set DMatrix labels");

	/* Determine number of classes as max(label) + 1 */
	max_label = labels[0];
	for (i = 1; i < nrows; i++)
	{
		if (labels[i] > max_label)
			max_label = labels[i];
	}
	num_class = (int)max_label + 1;

	snprintf(num_class_str, sizeof(num_class_str), "%d", num_class);
	snprintf(eta_str, sizeof(eta_str), "%f", learning_rate);
	snprintf(md_str, sizeof(md_str), "%d", max_depth);

	keys[0] = "objective";
	vals[0] = "multi:softmax";
	keys[1] = "num_class";
	vals[1] = num_class_str;
	keys[2] = "booster";
	vals[2] = "gbtree";
	keys[3] = "eta";
	vals[3] = eta_str;
	keys[4] = "max_depth";
	vals[4] = md_str;
	keys[5] = "verbosity";
	vals[5] = "1";

	if (XGBoosterCreate(&dtrain, 1, &booster) != 0)
		elog(ERROR, "Failed to create XGBoost booster");
	for (i = 0; i < param_count; i++)
	{
		if (XGBoosterSetParam(booster, keys[i], vals[i]) != 0)
			elog(ERROR, "Failed to set XGBoost parameter");
	}

	for (iter = 0; iter < n_estimators; iter++)
	{
		if (XGBoosterUpdateOneIter(booster, iter, dtrain) != 0)
			elog(ERROR, "Failed to update XGBoost booster");
	}

	if (XGBoosterSaveModelToBuffer(booster, "json", (const char **)&out_bytes, &out_len) != 0)
		elog(ERROR, "Failed to serialize XGBoost model");

	model_id = store_xgboost_model(out_bytes, out_len,
								  table_str, feature_str, label_str,
								  n_estimators, max_depth, learning_rate,
								  nrows, ncols, "multi:softmax");

	(void)XGBoosterFree(booster);
	(void)XGDMatrixFree(dtrain);
	nfree(features);
	nfree(labels);
	nfree(table_str);
	nfree(feature_str);
	nfree(label_str);

	PG_RETURN_INT32(model_id);
}

/*
 * train_xgboost_regressor() - Train XGBoost regressor
 * Parameters:
 *   table_name TEXT - Training data table
 *   feature_col TEXT - Feature column name
 *   target_col TEXT - Regression target column name
 *   n_estimators INTEGER - #trees (default 100)
 *   max_depth INTEGER
 *   learning_rate FLOAT
 * Returns: model_id INTEGER (stored in ml_models)
 */
Datum
train_xgboost_regressor(PG_FUNCTION_ARGS)
{
	text *table_name = PG_GETARG_TEXT_PP(0);
	text *feature_col = PG_GETARG_TEXT_PP(1);
	text *target_col = PG_GETARG_TEXT_PP(2);
	int32 n_estimators = PG_ARGISNULL(3) ? 100 : PG_GETARG_INT32(3);
	int32 max_depth = PG_ARGISNULL(4) ? 6 : PG_GETARG_INT32(4);
	float8 learning_rate = PG_ARGISNULL(5) ? 0.3 : PG_GETARG_FLOAT8(5);

	char *table_str = text_to_cstring(table_name);
	char *feature_str = text_to_cstring(feature_col);
	char *target_str = text_to_cstring(target_col);

	float *features = NULL;
	float *labels = NULL;
	int nrows = 0;
	int ncols = 0;
	DMatrixHandle dtrain = NULL;
	BoosterHandle booster = NULL;
	char eta_str[32];
	char md_str[16];
	const char *keys[5];
	const char *vals[5];
	int param_count = 5;
	int i, iter;
	bst_ulong out_len = 0;
	char *out_bytes = NULL;
	int32 model_id;

	load_training_data(table_str,
		feature_str,
		target_str,
		&features,
		&labels,
		&nrows,
		&ncols);

	if (XGDMatrixCreateFromMat(features, nrows, ncols, (float)NAN, &dtrain)
		!= 0)
		elog(ERROR, "Failed to create DMatrix");

	if (XGDMatrixSetFloatInfo(dtrain, "label", labels, nrows) != 0)
		elog(ERROR, "Failed to set DMatrix regression targets");

	snprintf(eta_str, sizeof(eta_str), "%f", learning_rate);
	snprintf(md_str, sizeof(md_str), "%d", max_depth);

	keys[0] = "objective";
	vals[0] = "reg:squarederror";
	keys[1] = "booster";
	vals[1] = "gbtree";
	keys[2] = "eta";
	vals[2] = eta_str;
	keys[3] = "max_depth";
	vals[3] = md_str;
	keys[4] = "verbosity";
	vals[4] = "1";

	if (XGBoosterCreate(&dtrain, 1, &booster) != 0)
		elog(ERROR, "Failed to create XGBoost booster");
	for (i = 0; i < param_count; i++)
	{
		if (XGBoosterSetParam(booster, keys[i], vals[i]) != 0)
			elog(ERROR, "Failed to set XGBoost parameter");
	}
	for (iter = 0; iter < n_estimators; iter++)
	{
		if (XGBoosterUpdateOneIter(booster, iter, dtrain) != 0)
			elog(ERROR, "Failed to update XGBoost booster");
	}

	if (XGBoosterSaveModelToBuffer(booster, "json", (const char **)&out_bytes, &out_len) != 0)
		elog(ERROR, "Failed to serialize XGBoost model");

	model_id = store_xgboost_model(out_bytes, out_len,
								  table_str, feature_str, target_str,
								  n_estimators, max_depth, learning_rate,
								  nrows, ncols, "reg:squarederror");

	(void)XGBoosterFree(booster);
	(void)XGDMatrixFree(dtrain);
	nfree(features);
	nfree(labels);
	nfree(table_str);
	nfree(feature_str);
	nfree(target_str);

	PG_RETURN_INT32(model_id);
}

/*
 * predict_xgboost() - Predict with XGBoost model
 * Arguments:
 *   model_id INT
 *   features FLOAT8[]
 * Returns:
 *   prediction (FLOAT8)
 */
Datum
predict_xgboost(PG_FUNCTION_ARGS)
{
	int32 model_id = PG_GETARG_INT32(0);
	ArrayType *features_array = PG_GETARG_ARRAYTYPE_P(1);
	int n_dims;
	float8 *features = NULL;
	float *feat_f = NULL;
	DMatrixHandle dmat = NULL;
	BoosterHandle booster = NULL;
	size_t model_size;
	void *mod_bytes = NULL;
	bst_ulong out_len = 0;
	const float *out_result = NULL;
	int i;
	float8 pred;

	if (ARR_NDIM(features_array) != 1)
		elog(ERROR, "features must be a 1-dimensional array");

	n_dims = (int)ArrayGetNItems(
		ARR_NDIM(features_array), ARR_DIMS(features_array));
	features = (float8 *)ARR_DATA_PTR(features_array);

	nalloc(feat_f, float, n_dims);
	for (i = 0; i < n_dims; i++)
		feat_f[i] = (float)features[i];

	mod_bytes = fetch_xgboost_model(model_id, &model_size);

	if (XGBoosterCreate(NULL, 0, &booster) != 0)
		elog(ERROR, "Failed to create XGBoost booster");

	if (XGBoosterLoadModelFromBuffer(booster, mod_bytes, model_size) != 0)
		elog(ERROR, "Failed to load XGBoost model from buffer");

	if (XGDMatrixCreateFromMat(feat_f, 1, n_dims, (float)NAN, &dmat) != 0)
		elog(ERROR, "Failed to create DMatrix for prediction");

	if (XGBoosterPredict(booster, dmat, 0, 0, 0, &out_len, &out_result)
		!= 0)
		elog(ERROR, "XGBoost prediction failed");

	pred = (out_len > 0) ? (float8)out_result[0] : 0.0;

	(void)XGBoosterFree(booster);
	(void)XGDMatrixFree(dmat);
	nfree(feat_f);
	nfree(mod_bytes);

	PG_RETURN_FLOAT8(pred);
}

#else /* !HAVE_XGBOOST */

/* Intentional conditional compilation stubs when XGBoost library is not available */
/* These stubs allow compilation without XGBoost - functions return errors at runtime */

Datum
train_xgboost_classifier(PG_FUNCTION_ARGS)
{
	ereport(ERROR,
		(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			errmsg("XGBoost is not available"),
			errhint("XGBoost library was not found during compilation. "
				"Reason: <xgboost/c_api.h> header file not found. "
				"To enable XGBoost support:\n"
				"1. Install XGBoost development libraries:\n"
				"   Ubuntu/Debian: sudo apt-get install libxgboost-dev\n"
				"   RHEL/CentOS: sudo yum install xgboost-devel\n"
				"   macOS: brew install xgboost\n"
				"2. Ensure XGBoost headers are in standard include paths\n"
				"3. Recompile NeuronDB: make clean && make install")));
	PG_RETURN_INT32(-1);
}

Datum
train_xgboost_regressor(PG_FUNCTION_ARGS)
{
	ereport(ERROR,
		(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			errmsg("XGBoost is not available"),
			errhint("XGBoost library was not found during compilation. "
				"Reason: <xgboost/c_api.h> header file not found. "
				"To enable XGBoost support:\n"
				"1. Install XGBoost development libraries:\n"
				"   Ubuntu/Debian: sudo apt-get install libxgboost-dev\n"
				"   RHEL/CentOS: sudo yum install xgboost-devel\n"
				"   macOS: brew install xgboost\n"
				"2. Ensure XGBoost headers are in standard include paths\n"
				"3. Recompile NeuronDB: make clean && make install")));
	PG_RETURN_INT32(-1);
}

Datum
predict_xgboost(PG_FUNCTION_ARGS)
{
	ereport(ERROR,
		(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			errmsg("XGBoost is not available"),
			errhint("Install libxgboost and recompile NeuronDB to "
				"enable XGBoost support.")));
	PG_RETURN_FLOAT8(0.0);
}

/*
 * evaluate_xgboost_by_model_id
 *
 * Evaluates an XGBoost model on a dataset and returns performance metrics.
 * Arguments: int4 model_id, text table_name, text feature_col, text label_col
 * Returns: jsonb with metrics
 */
Datum
evaluate_xgboost_by_model_id(PG_FUNCTION_ARGS)
{
#if HAVE_XGBOOST
    int32 model_id;
    text *table_name = NULL;
    text *feature_col = NULL;
    text *label_col = NULL;
    char *tbl_str = NULL;
    char *feat_str = NULL;
    char *targ_str = NULL;
    StringInfoData query;
    int ret;
    int nvec = 0;
    double mse = 0.0;
    double mae = 0.0;
    double ss_tot = 0.0;
    double ss_res = 0.0;
    double y_mean = 0.0;
    double r_squared;
    double rmse;
    int i;
    StringInfoData jsonbuf;
    Jsonb *result = NULL;
    MemoryContext oldcontext;
    bool is_classification = false;
    Oid label_type_oid = InvalidOid;
    /* Classification metrics */
    int tp = 0, tn = 0, fp = 0, fn = 0;
    double accuracy = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double f1_score = 0.0;

    /* Validate arguments */
    if (PG_NARGS() != 4)
        ereport(ERROR,
            (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                errmsg("neurondb: evaluate_xgboost_by_model_id: 4 arguments are required")));

    if (PG_ARGISNULL(0))
        ereport(ERROR,
            (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                errmsg("neurondb: evaluate_xgboost_by_model_id: model_id is required")));

    model_id = PG_GETARG_INT32(0);

    if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
        ereport(ERROR,
            (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                errmsg("neurondb: evaluate_xgboost_by_model_id: table_name, feature_col, and label_col are required")));

    table_name = PG_GETARG_TEXT_PP(1);
    feature_col = PG_GETARG_TEXT_PP(2);
    label_col = PG_GETARG_TEXT_PP(3);

    tbl_str = text_to_cstring(table_name);
    feat_str = text_to_cstring(feature_col);
    targ_str = text_to_cstring(label_col);

    oldcontext = CurrentMemoryContext;

    /* Connect to SPI */
    NdbSpiSession *spi_session = NULL;
    MemoryContext oldcontext_spi = CurrentMemoryContext;

    NDB_SPI_SESSION_BEGIN(spi_session, oldcontext_spi);

    /* Build query */
    ndb_spi_stringinfo_init(spi_session, &query);
    appendStringInfo(&query,
        "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
        feat_str, targ_str, tbl_str, feat_str, targ_str);

    ret = ndb_spi_execute(spi_session, query.data, true, 0);
    if (ret != SPI_OK_SELECT)
    {
        ndb_spi_stringinfo_free(spi_session, &query);
        NDB_SPI_SESSION_END(spi_session);
        nfree(tbl_str);
        nfree(feat_str);
        nfree(targ_str);
        ereport(ERROR,
            (errcode(ERRCODE_INTERNAL_ERROR),
                errmsg("neurondb: evaluate_xgboost_by_model_id: query failed")));
    }

    nvec = SPI_processed;
    if (nvec < 2)
    {
        ndb_spi_stringinfo_free(spi_session, &query);
        NDB_SPI_SESSION_END(spi_session);
        nfree(tbl_str);
        nfree(feat_str);
        nfree(targ_str);
        ereport(ERROR,
            (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                errmsg("neurondb: evaluate_xgboost_by_model_id: need at least 2 samples, got %d",
                    nvec)));
    }

    /* First pass: determine label type and compute mean of y (for regression) */
    if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL && SPI_tuptable->tupdesc->natts >= 2)
    {
        label_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 2);
        /* Check if label is integer type (classification) or float type (regression) */
        if (label_type_oid == INT4OID || label_type_oid == INT2OID || label_type_oid == INT8OID)
        {
            is_classification = true;
        }
    }

    for (i = 0; i < nvec; i++)
    {
        HeapTuple tuple = SPI_tuptable->vals[i];
        TupleDesc tupdesc = SPI_tuptable->tupdesc;
        Datum targ_datum;
        bool targ_null;

        targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);
        if (!targ_null && !is_classification)
            y_mean += DatumGetFloat8(targ_datum);
    }
    if (!is_classification && nvec > 0)
        y_mean /= nvec;

    /* Determine feature type from first row */
    Oid feat_type_oid = InvalidOid;
    bool feat_is_array = false;
    if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
    {
        feat_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 1);
        if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
            feat_is_array = true;
    }

    /* Second pass: compute predictions and metrics */
    for (i = 0; i < nvec; i++)
    {
        HeapTuple tuple = SPI_tuptable->vals[i];
        TupleDesc tupdesc = SPI_tuptable->tupdesc;
        Datum feat_datum;
        Datum targ_datum;
        bool feat_null;
        bool targ_null;
        ArrayType *arr = NULL;
        Vector *vec = NULL;
        double y_true;
        double y_pred;
        double error;
        int actual_dim;
        int j;

        feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
        targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

        if (feat_null || targ_null)
            continue;

        /* Handle both integer and float label types */
        int y_true_int = 0;
        double y_true_float = 0.0;
        
        if (is_classification)
        {
            if (label_type_oid == INT4OID || label_type_oid == INT2OID || label_type_oid == INT8OID)
                y_true_int = DatumGetInt32(targ_datum);
            else
                y_true_int = (int) rint(DatumGetFloat8(targ_datum));
        }
        else
        {
            y_true_float = DatumGetFloat8(targ_datum);
        }

        /* Extract features and determine dimension */
        if (feat_is_array)
        {
            arr = DatumGetArrayTypeP(feat_datum);
            if (ARR_NDIM(arr) != 1)
            {
                ndb_spi_stringinfo_free(spi_session, &query);
    NDB_SPI_SESSION_END(spi_session);
                nfree(tbl_str);
                nfree(feat_str);
                nfree(targ_str);
                ereport(ERROR,
                    (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                        errmsg("xgboost: features array must be 1-D")));
            }
            actual_dim = ARR_DIMS(arr)[0];
        }
        else
        {
            vec = DatumGetVector(feat_datum);
            actual_dim = vec->dim;
        }

        /* Make prediction using XGBoost model */
        if (feat_is_array)
        {
            /* Create a temporary array for prediction */
            Datum features_datum = feat_datum;
            y_pred = DatumGetFloat8(DirectFunctionCall2(predict_xgboost,
                                                       Int32GetDatum(model_id),
                                                       features_datum));
        }
        else
        {
            /* Convert vector to array for prediction */
            int ndims = 1;
            int dims[1] = {actual_dim};
            int lbs[1] = {1};
            Datum *elems = NULL;
            nalloc(elems, Datum, actual_dim);

            for (j = 0; j < actual_dim; j++)
                elems[j] = Float8GetDatum(vec->data[j]);

            ArrayType *feature_array = construct_md_array(elems, NULL, ndims, dims, lbs,
                                                        FLOAT8OID, sizeof(float8), true, 'd');
            Datum features_datum = PointerGetDatum(feature_array);

            y_pred = DatumGetFloat8(DirectFunctionCall2(predict_xgboost,
                                                       Int32GetDatum(model_id),
                                                       features_datum));

            nfree(elems);
            nfree(feature_array);
        }

        /* Compute metrics based on task type */
        if (is_classification)
        {
            int y_pred_int = (int) rint(y_pred);
            
            /* Classification: compute confusion matrix */
            if (y_true_int == 1 && y_pred_int == 1)
                tp++;
            else if (y_true_int == 0 && y_pred_int == 0)
                tn++;
            else if (y_true_int == 0 && y_pred_int == 1)
                fp++;
            else if (y_true_int == 1 && y_pred_int == 0)
                fn++;
        }
        else
        {
            /* Regression: compute errors */
            error = y_true_float - y_pred;
            mse += error * error;
            mae += fabs(error);
            ss_res += error * error;
            ss_tot += (y_true_float - y_mean) * (y_true_float - y_mean);
        }
    }

    ndb_spi_stringinfo_free(spi_session, &query);
    NDB_SPI_SESSION_END(spi_session);

    /* Build result JSON based on task type */
    MemoryContextSwitchTo(oldcontext);
    initStringInfo(&jsonbuf);
    
    if (is_classification)
    {
        /* Calculate classification metrics */
        int total = tp + tn + fp + fn;
        
        if (total > 0)
        {
            accuracy = (double) (tp + tn) / (double) total;
            precision = (tp + fp > 0) ? (double) tp / (double) (tp + fp) : 0.0;
            recall = (tp + fn > 0) ? (double) tp / (double) (tp + fn) : 0.0;
            f1_score = (precision + recall > 0.0)
                ? 2.0 * precision * recall / (precision + recall)
                : 0.0;
        }
        
        appendStringInfo(&jsonbuf,
            "{\"accuracy\":%.6f,\"precision\":%.6f,\"recall\":%.6f,\"f1_score\":%.6f,\"n_samples\":%d}",
            accuracy, precision, recall, f1_score, nvec);
    }
    else
    {
        /* Calculate regression metrics */
        mse /= nvec;
        mae /= nvec;
        rmse = sqrt(mse);

        /* Handle R² calculation - if ss_tot is zero (no variance in y), R² is undefined */
        if (ss_tot == 0.0)
            r_squared = 0.0; /* Convention: set to 0 when there's no variance to explain */
        else
            r_squared = 1.0 - (ss_res / ss_tot);
        
        appendStringInfo(&jsonbuf,
            "{\"mse\":%.6f,\"mae\":%.6f,\"rmse\":%.6f,\"r_squared\":%.6f,\"n_samples\":%d}",
            mse, mae, rmse, r_squared, nvec);
    }

    result = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetTextDatum(jsonbuf.data)));
    nfree(jsonbuf.data);

    nfree(tbl_str);
    nfree(feat_str);
    nfree(targ_str);

    PG_RETURN_JSONB_P(result);
#else
    ereport(ERROR,
            (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
             errmsg("XGBoost library not available. Please install XGBoost to use evaluation.")));
    PG_RETURN_NULL();
#endif
}

#endif /* HAVE_XGBOOST */

#if HAVE_XGBOOST
#include "neurondb_gpu_model.h"
#include "ml_gpu_registry.h"
#include "ml_gpu_xgboost.h"
#include "neurondb_gpu.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"

typedef struct XGBoostGpuModelState
{
	bytea *model_blob;
	Jsonb *metrics;
	int feature_dim;
	int n_samples;
} XGBoostGpuModelState;

static void
xgboost_gpu_release_state(XGBoostGpuModelState *state)
{
	if (state == NULL)
		return;
	if (state->model_blob != NULL)
		nfree(state->model_blob);
	if (state->metrics != NULL)
		nfree(state->metrics);
	/* State itself could be allocated with either palloc or nalloc - use nfree which is safe for both */
	nfree(state);
}

static bytea * __attribute__((unused))
xgboost_model_serialize_to_bytea(int n_estimators, int max_depth, float learning_rate, int n_features, const char *objective, uint8 training_backend)
{
	StringInfoData buf;
	int total_size;
	bytea *result = NULL;
	int obj_len;
	char *result_bytes = NULL;

	/* Validate training_backend */
	if (training_backend > 1)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: xgboost_model_serialize_to_bytea: invalid training_backend %d (must be 0 or 1)",
						training_backend)));
	}

	initStringInfo(&buf);
	/* Write training_backend first (0=CPU, 1=GPU) - unified storage format */
	appendBinaryStringInfo(&buf, (char *)&training_backend, sizeof(uint8));
	appendBinaryStringInfo(&buf, (char *)&n_estimators, sizeof(int));
	appendBinaryStringInfo(&buf, (char *)&max_depth, sizeof(int));
	appendBinaryStringInfo(&buf, (char *)&learning_rate, sizeof(float));
	appendBinaryStringInfo(&buf, (char *)&n_features, sizeof(int));
	obj_len = strlen(objective);
	appendBinaryStringInfo(&buf, (char *)&obj_len, sizeof(int));
	appendBinaryStringInfo(&buf, objective, obj_len);

	total_size = VARHDRSZ + buf.len;
	nalloc(result_bytes, char, total_size);
	result = (bytea *) result_bytes;
	SET_VARSIZE(result, total_size);
	memcpy(VARDATA(result), buf.data, buf.len);
	nfree(buf.data);

	return result;
}

static int
xgboost_model_deserialize_from_bytea(const bytea *data, int *n_estimators_out, int *max_depth_out, float *learning_rate_out, int *n_features_out, char *objective_out, int obj_max, uint8 * training_backend_out)
{
	const char *buf;
	int offset = 0;
	int obj_len;
	uint8		training_backend = 0;

	if (data == NULL || VARSIZE(data) < VARHDRSZ + sizeof(uint8) + sizeof(int) * 3 + sizeof(float) + sizeof(int))
		return -1;

	buf = VARDATA(data);
	/* Read training_backend first (unified storage format) */
	training_backend = (uint8) buf[offset];
	offset += sizeof(uint8);
	if (training_backend_out != NULL)
		*training_backend_out = training_backend;
	memcpy(n_estimators_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(max_depth_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(learning_rate_out, buf + offset, sizeof(float));
	offset += sizeof(float);
	memcpy(n_features_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(&obj_len, buf + offset, sizeof(int));
	offset += sizeof(int);

	if (obj_len >= obj_max)
		return -1;
	memcpy(objective_out, buf + offset, obj_len);
	objective_out[obj_len] = '\0';

	return 0;
}

static bool
xgboost_gpu_train(MLGpuModel *model, const MLGpuTrainSpec *spec, char **errstr)
{
	XGBoostGpuModelState *state = NULL;
	bytea *payload = NULL;
	Jsonb *metrics = NULL;
	int rc;

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

	payload = NULL;
	metrics = NULL;

	PG_TRY();
	{
		rc = ndb_gpu_xgboost_train(spec->feature_matrix,
								   spec->label_vector,
								   spec->sample_count,
								   spec->feature_dim,
								   spec->hyperparameters,
								   &payload,
								   &metrics,
								   errstr);
	}
	PG_CATCH();
	{
		FlushErrorState();
		rc = -1;
		if (errstr && *errstr == NULL)
			*errstr = pstrdup("XGBoost GPU training failed with exception");
	}
	PG_END_TRY();

	if (rc != 0 || payload == NULL)
	{
		return false;
	}

	if (model->backend_state != NULL)
	{
		xgboost_gpu_release_state((XGBoostGpuModelState *) model->backend_state);
		model->backend_state = NULL;
	}

	/* Use nalloc consistently - not palloc */
	nalloc(state, XGBoostGpuModelState, 1);
	memset(state, 0, sizeof(XGBoostGpuModelState));
	state->model_blob = payload;
	state->feature_dim = spec->feature_dim;
	state->n_samples = spec->sample_count;
	state->metrics = metrics;

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;

	return true;
}

static bool
xgboost_gpu_predict(const MLGpuModel *model, const float *input, int input_dim,
	float *output, int output_dim, char **errstr)
{
	const XGBoostGpuModelState *state;
	double prediction;
	int rc;

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

	state = (const XGBoostGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
		return false;

	/* Validate input dimension matches model */
	if (input_dim != state->feature_dim)
	{
		if (errstr != NULL)
			*errstr = pstrdup("neurondb: xgboost: feature dimension mismatch");
		return false;
	}

	rc = ndb_gpu_xgboost_predict(state->model_blob,
								 input,
								 input_dim,
								 &prediction,
								 errstr);
	if (rc != 0)
		return false;

	output[0] = (float) prediction;

	return true;
}

static bool
xgboost_gpu_evaluate(const MLGpuModel *model, const MLGpuEvalSpec *spec,
	MLGpuMetrics *out, char **errstr)
{
	const XGBoostGpuModelState *state;
	Jsonb *metrics_json = NULL;

	if (errstr != NULL)
		*errstr = NULL;
	if (out != NULL)
		out->payload = NULL;
	if (model == NULL || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("xgboost_gpu_evaluate: invalid model");
		return false;
	}

	state = (const XGBoostGpuModelState *)model->backend_state;

	/* Return metrics from state if available, otherwise return a basic metrics object */
	if (state->metrics != NULL)
	{
		metrics_json = (Jsonb *) PG_DETOAST_DATUM_COPY(PointerGetDatum(state->metrics));
	}
	else
	{
		/* Create basic metrics from model blob if available */
		StringInfoData buf;
		initStringInfo(&buf);
		appendStringInfo(&buf,
			"{\"algorithm\":\"xgboost\",\"storage\":\"gpu\","
			"\"n_features\":%d,\"n_samples\":%d}",
			state->feature_dim > 0 ? state->feature_dim : 0,
			state->n_samples > 0 ? state->n_samples : 0);
		metrics_json = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
			CStringGetTextDatum(buf.data)));
		nfree(buf.data);
	}

	if (out != NULL)
		out->payload = metrics_json;

	return true;
}

static bool
xgboost_gpu_serialize(const MLGpuModel *model, bytea **payload_out,
	Jsonb **metadata_out, char **errstr)
{
	const XGBoostGpuModelState *state;
	bytea *payload_copy = NULL;
	int payload_size;
	char *payload_bytes = NULL;

	if (errstr != NULL)
		*errstr = NULL;
	if (payload_out != NULL)
		*payload_out = NULL;
	if (metadata_out != NULL)
		*metadata_out = NULL;
	if (model == NULL || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("xgboost_gpu_serialize: invalid model");
		return false;
	}

	state = (const XGBoostGpuModelState *)model->backend_state;
	if (state->model_blob == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("xgboost_gpu_serialize: model blob is NULL");
		return false;
	}
	payload_size = VARSIZE(state->model_blob);
	nalloc(payload_bytes, char, payload_size);
	payload_copy = (bytea *) payload_bytes;
	memcpy(payload_copy, state->model_blob, payload_size);

	if (payload_out != NULL)
		*payload_out = payload_copy;
	else
		nfree(payload_copy);

	if (metadata_out != NULL && state->metrics != NULL)
		*metadata_out = (Jsonb *)PG_DETOAST_DATUM_COPY(
			PointerGetDatum(state->metrics));

	return true;
}

static bool
xgboost_gpu_deserialize(MLGpuModel *model, const bytea *payload,
	const Jsonb *metadata, char **errstr)
{
	XGBoostGpuModelState *state = NULL;
	bytea *payload_copy = NULL;
	int payload_size;
	int n_estimators = 0;
	int max_depth = 0;
	float learning_rate = 0.0f;
	int n_features = 0;
	char objective[32];
	JsonbIterator *it = NULL;
	JsonbValue v;
	int r;
	char *payload_bytes = NULL;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("xgboost_gpu_deserialize: invalid parameters");
		return false;
	}
	payload_size = VARSIZE(payload);
	nalloc(payload_bytes, char, payload_size);
	payload_copy = (bytea *) payload_bytes;
	memcpy(payload_copy, payload, payload_size);

	{
		uint8		training_backend = 0;

		if (xgboost_model_deserialize_from_bytea(payload_copy, &n_estimators, &max_depth, &learning_rate, &n_features, objective, sizeof(objective), &training_backend) != 0)
		{
			nfree(payload_copy);
			if (errstr != NULL)
				*errstr = pstrdup("xgboost_gpu_deserialize: failed to deserialize");
			return false;
		}
	}

	nalloc(state, XGBoostGpuModelState, 1);
	state->model_blob = payload_copy;
	state->feature_dim = n_features;
	state->n_samples = 0;

	if (metadata != NULL)
	{
		int			metadata_size;
		char *metadata_bytes = NULL;
		Jsonb *metadata_copy = NULL;
		metadata_size = VARSIZE(metadata);
		nalloc(metadata_bytes, char, metadata_size);
		metadata_copy = (Jsonb *) metadata_bytes;
		memcpy(metadata_copy, metadata, metadata_size);
		state->metrics = metadata_copy;

		it = JsonbIteratorInit((JsonbContainer *)&metadata->root);
		while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
		{
			if (r == WJB_KEY)
			{
				char *key = pnstrdup(v.val.string.val, v.val.string.len);
				r = JsonbIteratorNext(&it, &v, false);
				if (strcmp(key, "n_samples") == 0 && v.type == jbvNumeric)
					state->n_samples = DatumGetInt32(DirectFunctionCall1(numeric_int4,
						NumericGetDatum(v.val.numeric)));
				nfree(key);
			}
		}
	} else
	{
		state->metrics = NULL;
	}

	if (model->backend_state != NULL)
	{
		/* Free old state - it may have been allocated with either palloc or nalloc */
		/* To be safe, use the release function which handles both cases */
		xgboost_gpu_release_state((XGBoostGpuModelState *) model->backend_state);
	}

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = false;

	return true;
}

static void
xgboost_gpu_destroy(MLGpuModel *model)
{
	XGBoostGpuModelState *state = NULL;

	if (model == NULL)
		return;

	if (model->backend_state != NULL)
	{
		state = (XGBoostGpuModelState *)model->backend_state;
		if (state->model_blob != NULL)
			nfree(state->model_blob);
		if (state->metrics != NULL)
			nfree(state->metrics);
		nfree(state);
		model->backend_state = NULL;
	}

	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

static const MLGpuModelOps xgboost_gpu_model_ops = {
	.algorithm = "xgboost",
	.train = xgboost_gpu_train,
	.predict = xgboost_gpu_predict,
	.evaluate = xgboost_gpu_evaluate,
	.serialize = xgboost_gpu_serialize,
	.deserialize = xgboost_gpu_deserialize,
	.destroy = xgboost_gpu_destroy,
};

void
neurondb_gpu_register_xgboost_model(void)
{
#if HAVE_XGBOOST
	static bool registered = false;
	if (registered)
		return;
	ndb_gpu_register_model_ops(&xgboost_gpu_model_ops);
	registered = true;
#endif /* HAVE_XGBOOST */
}
#endif /* HAVE_XGBOOST */

/* Stub implementation when XGBoost is not available - must be outside #if HAVE_XGBOOST block */
#if !HAVE_XGBOOST
/* Forward declaration to match header */
extern void neurondb_gpu_register_xgboost_model(void);

void
neurondb_gpu_register_xgboost_model(void)
{
	/* No-op when XGBoost is not available */
	return;
}
#endif /* !HAVE_XGBOOST */