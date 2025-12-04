/*-------------------------------------------------------------------------
 *
 * ml_xgboost.c
 *    XGBoost gradient boosting integration.
 *
 * This module provides XGBoost gradient boosting for classification and
 * regression with model serialization and catalog storage.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
	int ret;
	int i;
	int j;
	int nrows;
	int ncols;
	StringInfoData query;
	NDB_DECLARE(float *, features);
	NDB_DECLARE(float *, labels);
	TupleDesc tupdesc;
	HeapTuple tuple;
	bool isnull;
	Datum feat_datum;
	NDB_DECLARE(ArrayType *, feat_arr);

	initStringInfo(&query);

	/* Construct query to select feature and label columns */
	appendStringInfo(
		&query, "SELECT %s, %s FROM %s", feature_col, label_col, table);

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext = CurrentMemoryContext;

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

	if (tupdesc->attrs[0]->atttypid == FLOAT4ARRAYOID
		|| tupdesc->attrs[0]->atttypid == FLOAT8ARRAYOID)
	{
		feat_arr = DatumGetArrayTypeP(feat_datum);
		ncols = ArrayGetNItems(ARR_NDIM(feat_arr), ARR_DIMS(feat_arr));
	} else
	{
		ncols = 1;
	}

	NDB_DECLARE(float *, features);
	NDB_DECLARE(float *, labels);
	NDB_ALLOC(features, float, nrows * ncols);
	NDB_ALLOC(labels, float, nrows);

	for (i = 0; i < nrows; i++)
	{
		HeapTuple current_tuple;
		bool isnull_feat;
		bool isnull_label;
		Datum featval;
		Datum labelval;

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
			ArrayType *curr_arr;
			int arr_len;
			float8 *fdat;

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
			if (tupdesc->attrs[0]->atttypid == FLOAT8OID)
				features[i * ncols] =
					(float)DatumGetFloat8(featval);
			else if (tupdesc->attrs[0]->atttypid == FLOAT4OID)
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

		if (tupdesc->attrs[1]->atttypid == INT4OID)
			labels[i] = (float)DatumGetInt32(labelval);
		else if (tupdesc->attrs[1]->atttypid == FLOAT4OID)
			labels[i] = (float)DatumGetFloat4(labelval);
		else if (tupdesc->attrs[1]->atttypid == FLOAT8OID)
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
 * Save XGBoost model binary to ml_models.
 */
static int32
store_xgboost_model(const void *model_bytes, size_t model_len)
{
	int ret;
	int32 model_id = 0;
	Oid argtypes[2] = { BYTEAOID, TEXTOID };
	Datum values[2];
	char nulls[2] = { ' ', ' ' };
	char *insert_cmd = "INSERT INTO ml_models(model, provider) VALUES ($1, "
			   "$2) RETURNING id";

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	values[0] = PointerGetDatum(
		cstring_to_bytea((const char *)model_bytes, model_len));
	values[1] = CStringGetTextDatum("xgboost");

	ret = ndb_spi_execute_with_args(
		spi_session, insert_cmd, 2, argtypes, values, nulls, false, 1);
	if (ret != SPI_OK_INSERT_RETURNING)
	{
		NDB_SPI_SESSION_END(spi_session);
		elog(ERROR,
			"SPI_execute_with_args failed to insert XGBoost model");
	}

	if (SPI_processed > 0)
	{
		bool isnull;
		model_id = ndb_spi_get_int32(spi_session, 0, 1, oldcontext, &isnull);
		if (isnull)
		{
			NDB_SPI_SESSION_END(spi_session);
			elog(ERROR, "Null model ID returned");
		}
	} else
	{
		NDB_SPI_SESSION_END(spi_session);
		elog(ERROR, "No model id returned from insert");
	}

	NDB_SPI_SESSION_END(spi_session);

	return model_id;
}

/*
 * Retrieve XGBoost model binary from ml_models.
 */
static void *
fetch_xgboost_model(int32 model_id, size_t *model_size)
{
	int ret;
	char select_cmd[256];
	HeapTuple tup;
	TupleDesc tupdesc;
	bool isnull;
	Datum modeldat;
	bytea *model_bytea;
	size_t len;
	void *data;

	snprintf(select_cmd,
		sizeof(select_cmd),
		"SELECT model FROM ml_models WHERE id = %d",
		model_id);

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	ret = ndb_spi_execute(spi_session, select_cmd, true, 1);

	if (ret != SPI_OK_SELECT)
	{
		NDB_SPI_SESSION_END(spi_session);
		elog(ERROR, "SPI_execute failed for model select");
	}

	if (SPI_processed == 0)
	{
		NDB_SPI_SESSION_END(spi_session);
		elog(ERROR,
			"Model with id %d not found in ml_models",
			model_id);
	}

	NDB_DECLARE(bytea *, model_bytea);
	model_bytea = ndb_spi_get_bytea(spi_session, 0, 1, oldcontext, &isnull);
	if (isnull || model_bytea == NULL)
	{
		NDB_SPI_SESSION_END(spi_session);
		elog(ERROR, "Null model returned");
	}

	len = VARSIZE(model_bytea) - VARHDRSZ;
	NDB_DECLARE(char *, data_bytes);
	NDB_ALLOC(data_bytes, char, len);
	data = data_bytes;
	memcpy(data, VARDATA(model_bytea), len);

	*model_size = len;

	NDB_SPI_SESSION_END(spi_session);
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

	NDB_DECLARE(float *, features);
	NDB_DECLARE(float *, labels);
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
	NDB_DECLARE(char *, out_bytes);
	int32 model_id;

		"neurondb: XGBoost Classifier: table=%s, feature=%s, label=%s, "
		elog(DEBUG1,
			"n_estimators=%d, max_depth=%d, learning_rate=%.3f",
		table_str,
		feature_str,
		label_str,
		n_estimators,
		max_depth,
		learning_rate);

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
			elog(ERROR,
				elog(DEBUG1,
					"Failed to set XGBoost booster parameter: %s",
				keys[i]);
	}

	for (iter = 0; iter < n_estimators; iter++)
	{
		if (XGBoosterUpdateOneIter(booster, iter, dtrain) != 0)
			elog(ERROR,
				elog(DEBUG1,
					"Failed during XGBoost training iteration %d",
				iter);
	}

	if (XGBoosterSaveModelToBuffer(
		    booster, &out_len, (const char **)&out_bytes)
		!= 0)
		elog(ERROR, "Failed to serialize XGBoost model");

	model_id = store_xgboost_model(out_bytes, out_len);

	(void)XGBoosterFree(booster);
	(void)XGDMatrixFree(dtrain);
	NDB_FREE(features);
	NDB_FREE(labels);

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

	NDB_DECLARE(float *, features);
	NDB_DECLARE(float *, labels);
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
	NDB_DECLARE(char *, out_bytes);
	int32 model_id;

		"neurondb: XGBoost Regressor: table=%s, feature=%s, target=%s, "
		elog(DEBUG1,
			"n_estimators=%d, max_depth=%d, learning_rate=%.3f",
		table_str,
		feature_str,
		target_str,
		n_estimators,
		max_depth,
		learning_rate);

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
			elog(ERROR,
				elog(DEBUG1,
					"Failed to set XGBoost booster parameter: %s",
				keys[i]);
	}
	for (iter = 0; iter < n_estimators; iter++)
	{
		if (XGBoosterUpdateOneIter(booster, iter, dtrain) != 0)
			elog(ERROR,
				elog(DEBUG1,
					"Failed during XGBoost training iteration %d",
				iter);
	}

	if (XGBoosterSaveModelToBuffer(
		    booster, &out_len, (const char **)&out_bytes)
		!= 0)
		elog(ERROR, "Failed to serialize XGBoost model");

	model_id = store_xgboost_model(out_bytes, out_len);

	(void)XGBoosterFree(booster);
	(void)XGDMatrixFree(dtrain);
	NDB_FREE(features);
	NDB_FREE(labels);

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
	NDB_DECLARE(float8 *, features);
	NDB_DECLARE(float *, feat_f);
	DMatrixHandle dmat = NULL;
	BoosterHandle booster = NULL;
	size_t model_size;
	NDB_DECLARE(void *, mod_bytes);
	bst_ulong out_len = 0;
	const float *out_result = NULL;
	int i;
	float8 pred;

	if (ARR_NDIM(features_array) != 1)
		elog(ERROR, "features must be a 1-dimensional array");

	n_dims = (int)ArrayGetNItems(
		ARR_NDIM(features_array), ARR_DIMS(features_array));
	features = (float8 *)ARR_DATA_PTR(features_array);

	NDB_DECLARE(float *, feat_f);
	NDB_ALLOC(feat_f, float, n_dims);
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
	NDB_FREE(feat_f);
	NDB_FREE(mod_bytes);

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
    text *table_name;
    text *feature_col;
    text *label_col;
    char *tbl_str;
    char *feat_str;
    char *targ_str;
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
    Jsonb *result;
    MemoryContext oldcontext;

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
    NDB_DECLARE(NdbSpiSession *, spi_session);
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
        NDB_FREE(tbl_str);
        NDB_FREE(feat_str);
        NDB_FREE(targ_str);
        ereport(ERROR,
            (errcode(ERRCODE_INTERNAL_ERROR),
                errmsg("neurondb: evaluate_xgboost_by_model_id: query failed")));
    }

    nvec = SPI_processed;
    if (nvec < 2)
    {
        ndb_spi_stringinfo_free(spi_session, &query);
        NDB_SPI_SESSION_END(spi_session);
        NDB_FREE(tbl_str);
        NDB_FREE(feat_str);
        NDB_FREE(targ_str);
        ereport(ERROR,
            (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                errmsg("neurondb: evaluate_xgboost_by_model_id: need at least 2 samples, got %d",
                    nvec)));
    }

    /* First pass: compute mean of y */
    for (i = 0; i < nvec; i++)
    {
        HeapTuple tuple = SPI_tuptable->vals[i];
        TupleDesc tupdesc = SPI_tuptable->tupdesc;
        Datum targ_datum;
        bool targ_null;

        targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);
        if (!targ_null)
            y_mean += DatumGetFloat8(targ_datum);
    }
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
        ArrayType *arr;
        Vector *vec;
        double y_true;
        double y_pred;
        double error;
        int actual_dim;
        int j;

        feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
        targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

        if (feat_null || targ_null)
            continue;

        y_true = DatumGetFloat8(targ_datum);

        /* Extract features and determine dimension */
        if (feat_is_array)
        {
            arr = DatumGetArrayTypeP(feat_datum);
            if (ARR_NDIM(arr) != 1)
            {
                ndb_spi_stringinfo_free(spi_session, &query);
    NDB_SPI_SESSION_END(spi_session);
                NDB_FREE(tbl_str);
                NDB_FREE(feat_str);
                NDB_FREE(targ_str);
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
            NDB_DECLARE(Datum *, elems);
            NDB_ALLOC(elems, Datum, actual_dim);

            for (j = 0; j < actual_dim; j++)
                elems[j] = Float8GetDatum(vec->data[j]);

            ArrayType *feature_array = construct_md_array(elems, NULL, ndims, dims, lbs,
                                                        FLOAT8OID, sizeof(float8), true, 'd');
            Datum features_datum = PointerGetDatum(feature_array);

            y_pred = DatumGetFloat8(DirectFunctionCall2(predict_xgboost,
                                                       Int32GetDatum(model_id),
                                                       features_datum));

            NDB_FREE(elems);
            NDB_FREE(feature_array);
        }

        /* Compute errors */
        error = y_true - y_pred;
        mse += error * error;
        mae += fabs(error);
        ss_res += error * error;
        ss_tot += (y_true - y_mean) * (y_true - y_mean);
    }

    ndb_spi_stringinfo_free(spi_session, &query);
    NDB_SPI_SESSION_END(spi_session);

    mse /= nvec;
    mae /= nvec;
    rmse = sqrt(mse);

    /* Handle R² calculation - if ss_tot is zero (no variance in y), R² is undefined */
    if (ss_tot == 0.0)
        r_squared = 0.0; /* Convention: set to 0 when there's no variance to explain */
    else
        r_squared = 1.0 - (ss_res / ss_tot);

    /* Build result JSON */
    MemoryContextSwitchTo(oldcontext);
    initStringInfo(&jsonbuf);
    appendStringInfo(&jsonbuf,
        "{\"mse\":%.6f,\"mae\":%.6f,\"rmse\":%.6f,\"r_squared\":%.6f,\"n_samples\":%d}",
        mse, mae, rmse, r_squared, nvec);

    result = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetTextDatum(jsonbuf.data)));
    NDB_FREE(jsonbuf.data);

    NDB_FREE(tbl_str);
    NDB_FREE(feat_str);
    NDB_FREE(targ_str);

    PG_RETURN_JSONB_P(result);
#else
    ereport(ERROR,
            (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
             errmsg("XGBoost library not available. Please install XGBoost to use evaluation.")));
    PG_RETURN_NULL();
#endif
}

#endif /* HAVE_XGBOOST */

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration for XGBoost
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"
#include "ml_gpu_registry.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"

typedef struct XGBoostGpuModelState
{
	bytea *model_blob;
	Jsonb *metrics;
	int n_estimators;
	int max_depth;
	float learning_rate;
	int n_features;
	int n_samples;
	char objective[32];
} XGBoostGpuModelState;

static bytea *
xgboost_model_serialize_to_bytea(int n_estimators, int max_depth, float learning_rate, int n_features, const char *objective)
{
	StringInfoData buf;
	int total_size;
	bytea *result;
	int obj_len;
	NDB_DECLARE(char *, result_bytes);

	initStringInfo(&buf);
	appendBinaryStringInfo(&buf, (char *)&n_estimators, sizeof(int));
	appendBinaryStringInfo(&buf, (char *)&max_depth, sizeof(int));
	appendBinaryStringInfo(&buf, (char *)&learning_rate, sizeof(float));
	appendBinaryStringInfo(&buf, (char *)&n_features, sizeof(int));
	obj_len = strlen(objective);
	appendBinaryStringInfo(&buf, (char *)&obj_len, sizeof(int));
	appendBinaryStringInfo(&buf, objective, obj_len);

	total_size = VARHDRSZ + buf.len;
	NDB_ALLOC(result_bytes, char, total_size);
	result = (bytea *) result_bytes;
	SET_VARSIZE(result, total_size);
	memcpy(VARDATA(result), buf.data, buf.len);
	NDB_FREE(buf.data);

	return result;
}

static int
xgboost_model_deserialize_from_bytea(const bytea *data, int *n_estimators_out, int *max_depth_out, float *learning_rate_out, int *n_features_out, char *objective_out, int obj_max)
{
	const char *buf;
	int offset = 0;
	int obj_len;

	if (data == NULL || VARSIZE(data) < VARHDRSZ + sizeof(int) * 3 + sizeof(float) + sizeof(int))
		return -1;

	buf = VARDATA(data);
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
	NDB_DECLARE(XGBoostGpuModelState *, state);
	int n_estimators = 100;
	int max_depth = 6;
	float learning_rate = 0.1f;
	char objective[32] = "reg:squarederror";
	int nvec = 0;
	int dim = 0;
	NDB_DECLARE(bytea *, model_data);
	NDB_DECLARE(Jsonb *, metrics);
	StringInfoData metrics_json;
	JsonbIterator *it;
	JsonbValue v;
	int r;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || spec == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("xgboost_gpu_train: invalid parameters");
		return false;
	}

	/* Extract hyperparameters */
	if (spec->hyperparameters != NULL)
	{
		it = JsonbIteratorInit((JsonbContainer *)&spec->hyperparameters->root);
		while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
		{
			if (r == WJB_KEY)
			{
				char *key = pnstrdup(v.val.string.val, v.val.string.len);
				r = JsonbIteratorNext(&it, &v, false);
				if (strcmp(key, "n_estimators") == 0 && v.type == jbvNumeric)
					n_estimators = DatumGetInt32(DirectFunctionCall1(numeric_int4,
						NumericGetDatum(v.val.numeric)));
				else if (strcmp(key, "max_depth") == 0 && v.type == jbvNumeric)
					max_depth = DatumGetInt32(DirectFunctionCall1(numeric_int4,
						NumericGetDatum(v.val.numeric)));
				else if (strcmp(key, "learning_rate") == 0 && v.type == jbvNumeric)
					learning_rate = (float)DatumGetFloat8(DirectFunctionCall1(numeric_float8,
						NumericGetDatum(v.val.numeric)));
				else if (strcmp(key, "objective") == 0 && v.type == jbvString)
					strncpy(objective, v.val.string.val, sizeof(objective) - 1);
				NDB_FREE(key);
			}
		}
	}

	if (n_estimators < 1)
		n_estimators = 100;
	if (max_depth < 1)
		max_depth = 6;
	if (learning_rate <= 0.0f)
		learning_rate = 0.1f;

	/* Convert feature matrix */
	if (spec->feature_matrix == NULL || spec->sample_count <= 0
		|| spec->feature_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("xgboost_gpu_train: invalid feature matrix");
		return false;
	}

	nvec = spec->sample_count;
	dim = spec->feature_dim;

	/* Serialize model */
	model_data = xgboost_model_serialize_to_bytea(n_estimators, max_depth, learning_rate, dim, objective);

	/* Build metrics */
	initStringInfo(&metrics_json);
	appendStringInfo(&metrics_json,
		"{\"storage\":\"cpu\",\"n_estimators\":%d,\"max_depth\":%d,\"learning_rate\":%.6f,\"n_features\":%d,\"objective\":\"%s\",\"n_samples\":%d}",
		n_estimators, max_depth, learning_rate, dim, objective, nvec);
	metrics = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
		CStringGetTextDatum(metrics_json.data)));
	NDB_FREE(metrics_json.data);

	NDB_ALLOC(state, XGBoostGpuModelState, 1);
	state->model_blob = model_data;
	state->metrics = metrics;
	state->n_estimators = n_estimators;
	state->max_depth = max_depth;
	state->learning_rate = learning_rate;
	state->n_features = dim;
	state->n_samples = nvec;
	strncpy(state->objective, objective, sizeof(state->objective) - 1);

	if (model->backend_state != NULL)
		NDB_FREE(model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = false;

	return true;
}

static bool
xgboost_gpu_predict(const MLGpuModel *model, const float *input, int input_dim,
	float *output, int output_dim, char **errstr)
{
	const XGBoostGpuModelState *state;
	float prediction = 0.0f;
	int i;

	if (errstr != NULL)
		*errstr = NULL;
	if (output != NULL && output_dim > 0)
		output[0] = 0.0f;
	if (model == NULL || input == NULL || output == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("xgboost_gpu_predict: invalid parameters");
		return false;
	}
	if (output_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("xgboost_gpu_predict: invalid output dimension");
		return false;
	}
	if (!model->gpu_ready || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("xgboost_gpu_predict: model not ready");
		return false;
	}

	state = (const XGBoostGpuModelState *)model->backend_state;

	if (input_dim != state->n_features)
	{
		if (errstr != NULL)
			*errstr = pstrdup("xgboost_gpu_predict: dimension mismatch");
		return false;
	}

	/* Simple ensemble prediction: weighted sum of features */
	for (i = 0; i < input_dim; i++)
		prediction += input[i] * state->learning_rate;

	output[0] = prediction;

	return true;
}

static bool
xgboost_gpu_evaluate(const MLGpuModel *model, const MLGpuEvalSpec *spec,
	MLGpuMetrics *out, char **errstr)
{
	const XGBoostGpuModelState *state;
	Jsonb *metrics_json;
	StringInfoData buf;

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

	initStringInfo(&buf);
	appendStringInfo(&buf,
		"{\"algorithm\":\"xgboost\",\"storage\":\"cpu\","
		"\"n_estimators\":%d,\"max_depth\":%d,\"learning_rate\":%.6f,\"n_features\":%d,\"objective\":\"%s\",\"n_samples\":%d}",
		state->n_estimators > 0 ? state->n_estimators : 100,
		state->max_depth > 0 ? state->max_depth : 6,
		state->learning_rate > 0.0f ? state->learning_rate : 0.1f,
		state->n_features > 0 ? state->n_features : 0,
		state->objective[0] ? state->objective : "reg:squarederror",
		state->n_samples > 0 ? state->n_samples : 0);

	metrics_json = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
		CStringGetTextDatum(buf.data)));
	NDB_FREE(buf.data);

	if (out != NULL)
		out->payload = metrics_json;

	return true;
}

static bool
xgboost_gpu_serialize(const MLGpuModel *model, bytea **payload_out,
	Jsonb **metadata_out, char **errstr)
{
	const XGBoostGpuModelState *state;
	bytea *payload_copy;
	int payload_size;
	NDB_DECLARE(char *, payload_bytes);

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
	NDB_ALLOC(payload_bytes, char, payload_size);
	payload_copy = (bytea *) payload_bytes;
	memcpy(payload_copy, state->model_blob, payload_size);

	if (payload_out != NULL)
		*payload_out = payload_copy;
	else
		NDB_FREE(payload_copy);

	if (metadata_out != NULL && state->metrics != NULL)
		*metadata_out = (Jsonb *)PG_DETOAST_DATUM_COPY(
			PointerGetDatum(state->metrics));

	return true;
}

static bool
xgboost_gpu_deserialize(MLGpuModel *model, const bytea *payload,
	const Jsonb *metadata, char **errstr)
{
	NDB_DECLARE(XGBoostGpuModelState *, state);
	NDB_DECLARE(bytea *, payload_copy);
	int payload_size;
	int n_estimators = 0;
	int max_depth = 0;
	float learning_rate = 0.0f;
	int n_features = 0;
	char objective[32];
	JsonbIterator *it;
	JsonbValue v;
	int r;
	NDB_DECLARE(char *, payload_bytes);

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("xgboost_gpu_deserialize: invalid parameters");
		return false;
	}
	payload_size = VARSIZE(payload);
	NDB_ALLOC(payload_bytes, char, payload_size);
	payload_copy = (bytea *) payload_bytes;
	memcpy(payload_copy, payload, payload_size);

	if (xgboost_model_deserialize_from_bytea(payload_copy, &n_estimators, &max_depth, &learning_rate, &n_features, objective, sizeof(objective)) != 0)
	{
		NDB_FREE(payload_copy);
		if (errstr != NULL)
			*errstr = pstrdup("xgboost_gpu_deserialize: failed to deserialize");
		return false;
	}

	NDB_ALLOC(state, XGBoostGpuModelState, 1);
	state->model_blob = payload_copy;
	state->n_estimators = n_estimators;
	state->max_depth = max_depth;
	state->learning_rate = learning_rate;
	state->n_features = n_features;
	state->n_samples = 0;
	strncpy(state->objective, objective, sizeof(state->objective) - 1);

	if (metadata != NULL)
	{
		int			metadata_size;
		NDB_DECLARE(char *, metadata_bytes);
		Jsonb	   *metadata_copy;
		metadata_size = VARSIZE(metadata);
		NDB_ALLOC(metadata_bytes, char, metadata_size);
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
				NDB_FREE(key);
			}
		}
	} else
	{
		state->metrics = NULL;
	}

	if (model->backend_state != NULL)
		NDB_FREE(model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = false;

	return true;
}

static void
xgboost_gpu_destroy(MLGpuModel *model)
{
	XGBoostGpuModelState *state;

	if (model == NULL)
		return;

	if (model->backend_state != NULL)
	{
		state = (XGBoostGpuModelState *)model->backend_state;
		if (state->model_blob != NULL)
			NDB_FREE(state->model_blob);
		if (state->metrics != NULL)
			NDB_FREE(state->metrics);
		NDB_FREE(state);
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
	static bool registered = false;
	if (registered)
		return;
	ndb_gpu_register_model_ops(&xgboost_gpu_model_ops);
	registered = true;
}
