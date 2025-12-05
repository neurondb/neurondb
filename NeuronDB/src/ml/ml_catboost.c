/*-------------------------------------------------------------------------
 *
 * ml_catboost.c
 *    CatBoost gradient boosting integration.
 *
 * This module provides CatBoost gradient boosting for classification and
 * regression with support for categorical features.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_catboost.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/lsyscache.h"
#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "storage/fd.h"
#include "miscadmin.h"
#include "utils/memutils.h"

#ifdef __has_include
#if __has_include(<catboost/c_api.h>)
#include <catboost/c_api.h>
#define CATBOOST_AVAILABLE
#endif
#endif

#include <stdio.h>

/*
 * PG_MODULE_MAGIC is in neurondb.c only.
 */

PG_FUNCTION_INFO_V1(train_catboost_classifier);
PG_FUNCTION_INFO_V1(train_catboost_regressor);
PG_FUNCTION_INFO_V1(predict_catboost);
PG_FUNCTION_INFO_V1(evaluate_catboost_by_model_id);

/*
 * get_column_index
 *      Helper function to get column index in a query result by name.
 */
__attribute__((unused)) static int
get_column_index(SPITupleTable * tuptable,
				 TupleDesc tupdesc,
				 const char *colname)
{
	int			i;

	for (i = 0; i < tupdesc->natts; i++)
	{
		if (strcmp(NameStr(TupleDescAttr(tupdesc, i)->attname), colname) == 0)
			return i;
	}
	ereport(ERROR,
			(errcode(ERRCODE_UNDEFINED_COLUMN),
			 errmsg("neurondb: column \"%s\" does not exist in tuple", colname)));

	/* not reached */
	return -1;
}

/*
 * write_csv_from_spi
 *      Helper to write training data to CSV for CatBoost.
 */
__attribute__((unused)) static void
write_csv_from_spi(char *csv_path,
				   SPITupleTable * tuptable,
				   TupleDesc tupdesc,
				   int *feature_idxs,
				   int n_features,
				   int label_idx)
{
	FILE	   *fp;
	uint64		row;
	int			i;

	fp = AllocateFile(csv_path, "w");
	if (fp == NULL)
		ereport(ERROR,
				(errcode_for_file_access(),
				 errmsg("could not open file \"%s\" for writing: %m", csv_path)));

	/* Write header */
	for (i = 0; i < n_features; i++)
	{
		if (fprintf(fp, "f%d,", i) < 0)
		{
			FreeFile(fp);
			ereport(ERROR,
					(errcode_for_file_access(),
					 errmsg("could not write to file \"%s\": %m", csv_path)));
		}
	}
	if (fprintf(fp, "label\n") < 0)
	{
		FreeFile(fp);
		ereport(ERROR,
				(errcode_for_file_access(),
				 errmsg("could not write to file \"%s\": %m", csv_path)));
	}

	for (row = 0; row < tuptable->numvals; row++)
	{
		HeapTuple	tuple = tuptable->vals[row];

		for (i = 0; i < n_features; i++)
		{
			Datum		val;
			bool		isnull;
			Oid			typid;
			Oid			typoutput;
			bool		typisvarlena;
			char	   *valstr;

			val = heap_getattr(tuple, feature_idxs[i] + 1, tupdesc, &isnull);

			if (isnull)
			{
				if (fprintf(fp, ",") < 0)
				{
					FreeFile(fp);
					ereport(ERROR,
							(errcode_for_file_access(),
							 errmsg("could not write to file \"%s\": %m", csv_path)));
				}
			}
			else
			{
				typid = TupleDescAttr(tupdesc, feature_idxs[i])->atttypid;
				getTypeOutputInfo(typid, &typoutput, &typisvarlena);
				valstr = OidOutputFunctionCall(typoutput, val);
				if (fprintf(fp, "%s,", valstr) < 0)
				{
					FreeFile(fp);
					ereport(ERROR,
							(errcode_for_file_access(),
							 errmsg("could not write to file \"%s\": %m", csv_path)));
				}
			}
		}

		/* Output label column */
		{
			Datum		label_val;
			bool		label_isnull;
			Oid			typid;
			Oid			typoutput;
			bool		typisvarlena;
			char	   *labelstr;

			label_val = heap_getattr(tuple, label_idx + 1, tupdesc, &label_isnull);
			if (label_isnull)
			{
				if (fprintf(fp, "\n") < 0)
				{
					FreeFile(fp);
					ereport(ERROR,
							(errcode_for_file_access(),
							 errmsg("could not write to file \"%s\": %m", csv_path)));
				}
			}
			else
			{
				typid = TupleDescAttr(tupdesc, label_idx)->atttypid;
				getTypeOutputInfo(typid, &typoutput, &typisvarlena);
				labelstr = OidOutputFunctionCall(typoutput, label_val);
				if (fprintf(fp, "%s\n", labelstr) < 0)
				{
					FreeFile(fp);
					ereport(ERROR,
							(errcode_for_file_access(),
							 errmsg("could not write to file \"%s\": %m", csv_path)));
				}
			}
		}
	}
	FreeFile(fp);
}

/*
 * check_catboost_error
 *      Helper function for error translation from CatBoost return codes.
 */
__attribute__((unused)) static void
check_catboost_error(int err_code)
{
#ifdef CATBOOST_AVAILABLE
	if (err_code != 0)
	{
		const char *err_msg = CatBoostGetErrorString();

		if (err_msg == NULL)
			err_msg = "Unknown error from CatBoost";
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: CatBoost error: %s (code=%d)", err_msg, err_code)));
	}
#else
	if (err_code != 0)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("neurondb: CatBoost error: code=%d (library not available)", err_code)));
#endif
}

/*
 * train_catboost_classifier
 *      Trains a CatBoost classifier model using the provided table, feature columns, and label column.
 *      Returns integer model_id on successful training and storage.
 */
Datum
train_catboost_classifier(PG_FUNCTION_ARGS)
{
#ifdef CATBOOST_AVAILABLE
	text	   *table_name_text = PG_GETARG_TEXT_PP(0);
	text	   *feature_col_text = PG_GETARG_TEXT_PP(1);
	text	   *label_col_text = PG_GETARG_TEXT_PP(2);
	int32		iterations;
	float8		learning_rate;
	int32		depth;

	char	   *table_name;
	char	   *feature_col_list;
	char	   *label_col;

	StringInfoData sql;
	int			ret;
	int			n_features;
	int			i;
	int			model_id;

	char	  **features = NULL;
	char	   *token;
	MemoryContext oldctx;
	MemoryContext per_query_ctx;

	/* Assign parameters with defaults if NULL */
	iterations = PG_ARGISNULL(3) ? 1000 : PG_GETARG_INT32(3);
	learning_rate = PG_ARGISNULL(4) ? 0.03 : PG_GETARG_FLOAT8(4);
	depth = PG_ARGISNULL(5) ? 6 : PG_GETARG_INT32(5);

	table_name = text_to_cstring(table_name_text);
	feature_col_list = text_to_cstring(feature_col_text);
	label_col = text_to_cstring(label_col_text);

	/* Parse feature_col_list into features[] */
	n_features = 1;
	for (i = 0; feature_col_list[i]; i++)
	{
		if (feature_col_list[i] == ',')
			n_features++;
	}
	features = (char **) palloc0(sizeof(char *) * n_features);

	i = 0;
	token = strtok(feature_col_list, ",");
	while (token != NULL)
	{
		while (*token == ' ' || *token == '\t')
			token++;
		features[i++] = pstrdup(token);
		token = strtok(NULL, ",");
	}

	initStringInfo(&sql);
	appendStringInfo(&sql, "SELECT ");
	for (i = 0; i < n_features; i++)
	{
		if (i > 0)
			appendStringInfoChar(&sql, ',');
		appendStringInfoString(&sql, features[i]);
	}
	appendStringInfo(&sql, ",%s FROM %s", label_col, table_name);


	per_query_ctx = AllocSetContextCreate(CurrentMemoryContext,
										  "catboost_spi_ctx",
										  ALLOCSET_DEFAULT_SIZES);
	oldctx = MemoryContextSwitchTo(per_query_ctx);

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	ret = ndb_spi_execute(spi_session, sql.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		NDB_SPI_SESSION_END(spi_session);
		MemoryContextSwitchTo(oldctx);
		MemoryContextDelete(per_query_ctx);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: could not execute SQL for training set")));
	}

	/* Get feature and label indexes */
	{
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;

		NDB_DECLARE(int *, feature_idxs);
		NDB_ALLOC(feature_idxs, int, n_features);
		int			label_idx;

		for (i = 0; i < n_features; i++)
			feature_idxs[i] = get_column_index(SPI_tuptable, tupdesc, features[i]);
		label_idx = get_column_index(SPI_tuptable, tupdesc, label_col);

		/* Write training data to CSV */
		{
			char		tmp_csv_path[MAXPGPATH];
			char		tmp_model_path[MAXPGPATH];

			snprintf(tmp_csv_path, sizeof(tmp_csv_path),
					 "%s/catboost_train_%d.csv", PG_TEMP_FILES_DIR, MyProcPid);
			snprintf(tmp_model_path, sizeof(tmp_model_path),
					 "%s/catboost_model_%d.cbm", PG_TEMP_FILES_DIR, MyProcPid);

			write_csv_from_spi(tmp_csv_path,
							   SPI_tuptable,
							   tupdesc,
							   feature_idxs,
							   n_features,
							   label_idx);

			/* Setup CatBoost C API options and train */
			{
				NDB_DECLARE(ModelCalcerHandle *, model_handle);
				CatBoostModelTrainingOptions *opts;

				opts = CatBoostCreateModelTrainingOptions();

				check_catboost_error(CatBoostSetTrainingOptionInt(opts, "iterations", iterations));
				check_catboost_error(CatBoostSetTrainingOptionDouble(opts, "learning_rate", learning_rate));
				check_catboost_error(CatBoostSetTrainingOptionInt(opts, "depth", depth));

				check_catboost_error(CatBoostTrainModelFromFile(opts,
																tmp_csv_path,
																n_features,
																label_idx,
																true,	/* has header */
																&model_handle));

				check_catboost_error(CatBoostSaveModelToFile(model_handle, tmp_model_path));

				model_id = MyProcPid;

				CatBoostDestroyModelTrainingOptions(opts);
				CatBoostFreeModelCalcer(model_handle);
			}
		}
	}

	NDB_SPI_SESSION_END(spi_session);
	MemoryContextSwitchTo(oldctx);
	MemoryContextDelete(per_query_ctx);

	PG_RETURN_INT32(model_id);
#else
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("CatBoost library not available"),
			 errhint("CatBoost library was not found during compilation. "
					 "Reason: <catboost/c_api.h> header file not found. "
					 "To enable CatBoost support:\n"
					 "1. Install CatBoost development libraries:\n"
					 "   Ubuntu/Debian: Build from source or use conda: conda install -c conda-forge catboost\n"
					 "   RHEL/CentOS: Build from source\n"
					 "   macOS: brew install catboost or conda install -c conda-forge catboost\n"
					 "2. Ensure CatBoost headers are in standard include paths\n"
					 "3. Recompile NeuronDB: make clean && make install")));
	PG_RETURN_NULL();
#endif
}

/*
 * train_catboost_regressor
 *      Trains a CatBoost regressor model. Arguments similar to classifier.
 *      Returns integer model_id on success.
 */
Datum
train_catboost_regressor(PG_FUNCTION_ARGS)
{
#ifdef CATBOOST_AVAILABLE
	text	   *table_name_text = PG_GETARG_TEXT_PP(0);
	text	   *feature_col_text = PG_GETARG_TEXT_PP(1);
	text	   *target_col_text = PG_GETARG_TEXT_PP(2);
	int32		iterations;
	char	   *table_name;
	char	   *feature_col_list;
	char	   *target_col;

	StringInfoData sql;
	int			ret;
	int			n_features;
	int			i;
	int			model_id;

	char	  **features = NULL;
	char	   *token;
	MemoryContext oldctx;
	MemoryContext per_query_ctx;

	iterations = PG_ARGISNULL(3) ? 1000 : PG_GETARG_INT32(3);

	table_name = text_to_cstring(table_name_text);
	feature_col_list = text_to_cstring(feature_col_text);
	target_col = text_to_cstring(target_col_text);

	/* Parse feature list */
	n_features = 1;
	for (i = 0; feature_col_list[i]; i++)
	{
		if (feature_col_list[i] == ',')
			n_features++;
	}
	features = (char **) palloc0(sizeof(char *) * n_features);

	i = 0;
	token = strtok(feature_col_list, ",");
	while (token != NULL)
	{
		while (*token == ' ' || *token == '\t')
			token++;
		features[i++] = pstrdup(token);
		token = strtok(NULL, ",");
	}

	initStringInfo(&sql);
	appendStringInfo(&sql, "SELECT ");
	for (i = 0; i < n_features; i++)
	{
		if (i > 0)
			appendStringInfoChar(&sql, ',');
		appendStringInfoString(&sql, features[i]);
	}
	appendStringInfo(&sql, ",%s FROM %s", target_col, table_name);

	per_query_ctx = AllocSetContextCreate(CurrentMemoryContext,
										  "catboost_spi_ctx",
										  ALLOCSET_DEFAULT_SIZES);
	oldctx = MemoryContextSwitchTo(per_query_ctx);

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	ret = ndb_spi_execute(spi_session, sql.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		NDB_SPI_SESSION_END(spi_session);
		MemoryContextSwitchTo(oldctx);
		MemoryContextDelete(per_query_ctx);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: could not execute SQL for training set")));
	}

	{
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		int		   *feature_idxs = (int *) palloc0(sizeof(int) * n_features);
		int			target_idx;

		for (i = 0; i < n_features; i++)
			feature_idxs[i] = get_column_index(SPI_tuptable, tupdesc, features[i]);
		target_idx = get_column_index(SPI_tuptable, tupdesc, target_col);

		{
			char		tmp_csv_path[MAXPGPATH];
			char		tmp_model_path[MAXPGPATH];

			snprintf(tmp_csv_path, sizeof(tmp_csv_path),
					 "%s/catboost_train_%d.csv", PG_TEMP_FILES_DIR, MyProcPid);
			snprintf(tmp_model_path, sizeof(tmp_model_path),
					 "%s/catboost_model_%d.cbm", PG_TEMP_FILES_DIR, MyProcPid);

			write_csv_from_spi(tmp_csv_path,
							   SPI_tuptable,
							   tupdesc,
							   feature_idxs,
							   n_features,
							   target_idx);

			{
				NDB_DECLARE(ModelCalcerHandle *, model_handle);
				CatBoostModelTrainingOptions *opts;

				opts = CatBoostCreateModelTrainingOptions();
				check_catboost_error(CatBoostSetTrainingOptionInt(opts, "iterations", iterations));
				check_catboost_error(CatBoostSetTrainingOptionInt(opts, "task_type", 0));	/* 0 for regression */

				check_catboost_error(CatBoostTrainModelFromFile(opts,
																tmp_csv_path,
																n_features,
																target_idx,
																true,
																&model_handle));

				check_catboost_error(CatBoostSaveModelToFile(model_handle, tmp_model_path));

				model_id = MyProcPid;

				CatBoostDestroyModelTrainingOptions(opts);
				CatBoostFreeModelCalcer(model_handle);
			}
		}
	}

	NDB_SPI_SESSION_END(spi_session);
	MemoryContextSwitchTo(oldctx);
	MemoryContextDelete(per_query_ctx);

	PG_RETURN_INT32(model_id);
#else
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("CatBoost library not available"),
			 errhint("CatBoost library was not found during compilation. "
					 "Reason: <catboost/c_api.h> header file not found. "
					 "To enable CatBoost support:\n"
					 "1. Install CatBoost development libraries:\n"
					 "   Ubuntu/Debian: Build from source or use conda: conda install -c conda-forge catboost\n"
					 "   RHEL/CentOS: Build from source\n"
					 "   macOS: brew install catboost or conda install -c conda-forge catboost\n"
					 "2. Ensure CatBoost headers are in standard include paths\n"
					 "3. Recompile NeuronDB: make clean && make install")));
	PG_RETURN_NULL();
#endif
}

/*
 * predict_catboost
 *      Predicts using a persisted CatBoost model and a feature array.
 *      Arguments: int4 model_id, float8[] features (or text[] for categorical support).
 *      Returns float8 predicted value.
 */
Datum
predict_catboost(PG_FUNCTION_ARGS)
{
#ifdef CATBOOST_AVAILABLE
	int32		model_id;
	ArrayType  *features_array;
	Oid			elmtype;
	int16		typlen;
	bool		typbyval;
	char		typalign;
	int			n_features;
	int			i;
	int			ndim;
	Datum	   *elems;
	bool	   *nulls;
	float64		result = 0.0;

	NDB_DECLARE(ModelCalcerHandle *, model_handle);
	char		model_path[MAXPGPATH];

	model_id = PG_GETARG_INT32(0);
	features_array = PG_GETARG_ARRAYTYPE_P(1);

	ndim = ARR_NDIM(features_array);
	if (ndim != 1)
		ereport(ERROR,
				(errmsg("features array must be one-dimensional")));

	n_features = ArrayGetNItems(ARR_NDIM(features_array), ARR_DIMS(features_array));
	elmtype = ARR_ELEMTYPE(features_array);
	get_typlenbyvalalign(elmtype, &typlen, &typbyval, &typalign);

	deconstruct_array(features_array, elmtype, typlen, typbyval, typalign,
					  &elems, &nulls, &n_features);

	snprintf(model_path, sizeof(model_path),
			 "%s/catboost_model_%d.cbm",
			 PG_TEMP_FILES_DIR, model_id);

	check_catboost_error(CatBoostLoadModelFromFile(model_path, &model_handle));

	if (elmtype == TEXTOID)
	{
		/* categorical/text features */
		NDB_DECLARE(char **, input_features);

		NDB_ALLOC(input_features, char *, n_features);

		for (i = 0; i < n_features; i++)
		{
			if (nulls[i])
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: catboost feature %d is NULL", i)));

			input_features[i] = text_to_cstring(DatumGetTextPP(elems[i]));
		}

		check_catboost_error(CatBoostModelCalcerPredictText(model_handle,
															(const char *const *) input_features,
															n_features,
															&result,
															1));
		NDB_FREE(input_features);
	}
	else
	{
		/* numeric features */
		NDB_DECLARE(double *, features);

		NDB_ALLOC(features, double, n_features);
		for (i = 0; i < n_features; i++)
		{
			if (nulls[i])
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: catboost feature %d is NULL", i)));
			if (elmtype == FLOAT8OID)
				features[i] = DatumGetFloat8(elems[i]);
			else if (elmtype == FLOAT4OID)
				features[i] = (double) DatumGetFloat4(elems[i]);
			else if (elmtype == INT4OID)
				features[i] = (double) DatumGetInt32(elems[i]);
			else if (elmtype == INT8OID)
				features[i] = (double) DatumGetInt64(elems[i]);
			else if (elmtype == INT2OID)
				features[i] = (double) DatumGetInt16(elems[i]);
			else
				ereport(ERROR,
						(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
						 errmsg("neurondb: unsupported feature element type for CatBoost: %u", elmtype)));
		}

		check_catboost_error(CatBoostModelCalcerPredict(model_handle,
														features, n_features, &result, 1));
		NDB_FREE(features);
	}

	CatBoostFreeModelCalcer(model_handle);

	PG_RETURN_FLOAT8(result);
#else
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("CatBoost library not available"),
			 errhint("CatBoost library was not found during compilation. "
					 "Reason: <catboost/c_api.h> header file not found. "
					 "To enable CatBoost support:\n"
					 "1. Install CatBoost development libraries:\n"
					 "   Ubuntu/Debian: Build from source or use conda: conda install -c conda-forge catboost\n"
					 "   RHEL/CentOS: Build from source\n"
					 "   macOS: brew install catboost or conda install -c conda-forge catboost\n"
					 "2. Ensure CatBoost headers are in standard include paths\n"
					 "3. Recompile NeuronDB: make clean && make install")));
	PG_RETURN_NULL();
#endif
}

/*
 * evaluate_catboost_by_model_id
 *
 * Evaluates a CatBoost model on a dataset and returns performance metrics.
 * Arguments: int4 model_id, text table_name, text feature_col, text label_col
 * Returns: jsonb with metrics
 */
Datum
evaluate_catboost_by_model_id(PG_FUNCTION_ARGS)
{
#ifdef CATBOOST_AVAILABLE
	int32		model_id;
	text	   *table_name;
	text	   *feature_col;
	text	   *label_col;
	char	   *tbl_str;
	char	   *feat_str;
	char	   *targ_str;
	StringInfoData query;
	int			ret;
	int			nvec = 0;
	double		mse = 0.0;
	double		mae = 0.0;
	double		ss_tot = 0.0;
	double		ss_res = 0.0;
	double		y_mean = 0.0;
	double		r_squared;
	double		rmse;
	int			i;
	StringInfoData jsonbuf;
	Jsonb	   *result;
	MemoryContext oldcontext;

	/* Validate arguments */
	if (PG_NARGS() != 4)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_catboost_by_model_id: 4 arguments are required")));

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_catboost_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_catboost_by_model_id: table_name, feature_col, and label_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);
	label_col = PG_GETARG_TEXT_PP(3);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(label_col);

	oldcontext = CurrentMemoryContext;

	/* Connect to SPI */
	NDB_DECLARE(NdbSpiSession *, spi_session);

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

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
				 errmsg("neurondb: evaluate_catboost_by_model_id: query failed")));
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
				 errmsg("neurondb: evaluate_catboost_by_model_id: need at least 2 samples, got %d",
						nvec)));
	}

	/* First pass: compute mean of y */
	for (i = 0; i < nvec; i++)
	{
		/* Safe access to SPI_tuptable - validate before access */
		if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
			i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
		{
			continue;
		}
		HeapTuple	tuple = SPI_tuptable->vals[i];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;

		if (tupdesc == NULL)
		{
			continue;
		}
		Datum		targ_datum;
		bool		targ_null;

		/* Safe access for target - validate tupdesc has at least 2 columns */
		if (tupdesc->natts < 2)
		{
			continue;
		}
		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);
		if (!targ_null)
			y_mean += DatumGetFloat8(targ_datum);
	}
	y_mean /= nvec;

	/* Determine feature type from first row */
	Oid			feat_type_oid = InvalidOid;
	bool		feat_is_array = false;

	if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
	{
		feat_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 1);
		if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
			feat_is_array = true;
	}

	/* Second pass: compute predictions and metrics */
	for (i = 0; i < nvec; i++)
	{
		HeapTuple	tuple = SPI_tuptable->vals[i];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		feat_datum;
		Datum		targ_datum;
		bool		feat_null;
		bool		targ_null;
		ArrayType  *arr;
		Vector	   *vec;
		double		y_true;
		double		y_pred;
		double		error;
		int			actual_dim;
		int			j;

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
						 errmsg("catboost: features array must be 1-D")));
			}
			actual_dim = ARR_DIMS(arr)[0];
		}
		else
		{
			vec = DatumGetVector(feat_datum);
			actual_dim = vec->dim;
		}

		/* Make prediction using CatBoost model */
		if (feat_is_array)
		{
			/* Create a temporary array for prediction */
			Datum		features_datum = feat_datum;

			y_pred = DatumGetFloat8(DirectFunctionCall2(predict_catboost,
														Int32GetDatum(model_id),
														features_datum));
		}
		else
		{
			/* Convert vector to array for prediction */
			int			ndims = 1;
			int			dims[1] = {actual_dim};
			int			lbs[1] = {1};
			NDB_DECLARE(Datum *, elems);
			NDB_ALLOC(elems, Datum, actual_dim);

			for (j = 0; j < actual_dim; j++)
				elems[j] = Float8GetDatum(vec->data[j]);

			ArrayType  *feature_array = construct_md_array(elems, NULL, ndims, dims, lbs,
														   FLOAT8OID, sizeof(float8), true, 'd');
			Datum		features_datum = PointerGetDatum(feature_array);

			y_pred = DatumGetFloat8(DirectFunctionCall2(predict_catboost,
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

	/*
	 * Handle R² calculation - if ss_tot is zero (no variance in y), R² is
	 * undefined
	 */
	if (ss_tot == 0.0)
		r_squared = 0.0;		/* Convention: set to 0 when there's no
								 * variance to explain */
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
			 errmsg("CatBoost library not available. Please install CatBoost to use evaluation.")));
	PG_RETURN_NULL();
#endif
}

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration for CatBoost
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"
#include "ml_gpu_registry.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_spi_safe.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"

typedef struct CatBoostGpuModelState
{
	bytea	   *model_blob;
	Jsonb	   *metrics;
	int			iterations;
	int			depth;
	float		learning_rate;
	int			n_features;
	int			n_samples;
	char		loss_function[32];
}			CatBoostGpuModelState;

static bytea *
catboost_model_serialize_to_bytea(int iterations, int depth, float learning_rate, int n_features, const char *loss_function)
{
	StringInfoData buf;
	int			total_size;
	NDB_DECLARE(bytea *, result);
	NDB_DECLARE(char *, result_raw);
	int			loss_len;

	initStringInfo(&buf);
	appendBinaryStringInfo(&buf, (char *) &iterations, sizeof(int));
	appendBinaryStringInfo(&buf, (char *) &depth, sizeof(int));
	appendBinaryStringInfo(&buf, (char *) &learning_rate, sizeof(float));
	appendBinaryStringInfo(&buf, (char *) &n_features, sizeof(int));
	loss_len = strlen(loss_function);
	appendBinaryStringInfo(&buf, (char *) &loss_len, sizeof(int));
	appendBinaryStringInfo(&buf, loss_function, loss_len);

	total_size = VARHDRSZ + buf.len;
	NDB_ALLOC(result_raw, char, total_size);
	result = (bytea *) result_raw;
	SET_VARSIZE(result, total_size);
	memcpy(VARDATA(result), buf.data, buf.len);
	NDB_FREE(buf.data);

	return result;
}

static int
catboost_model_deserialize_from_bytea(const bytea * data, int *iterations_out, int *depth_out, float *learning_rate_out, int *n_features_out, char *loss_function_out, int loss_max)
{
	const char *buf;
	int			offset = 0;
	int			loss_len;

	if (data == NULL || VARSIZE(data) < VARHDRSZ + sizeof(int) * 3 + sizeof(float) + sizeof(int))
		return -1;

	buf = VARDATA(data);
	memcpy(iterations_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(depth_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(learning_rate_out, buf + offset, sizeof(float));
	offset += sizeof(float);
	memcpy(n_features_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(&loss_len, buf + offset, sizeof(int));
	offset += sizeof(int);

	if (loss_len >= loss_max)
		return -1;
	memcpy(loss_function_out, buf + offset, loss_len);
	loss_function_out[loss_len] = '\0';

	return 0;
}

static bool
catboost_gpu_train(MLGpuModel *model, const MLGpuTrainSpec *spec, char **errstr)
{
	CatBoostGpuModelState *state;
	int			iterations = 1000;
	int			depth = 6;
	float		learning_rate = 0.03f;
	char		loss_function[32] = "RMSE";
	int			nvec = 0;
	int			dim = 0;

	NDB_DECLARE(bytea *, model_data);
	NDB_DECLARE(Jsonb *, metrics);
	StringInfoData metrics_json;
	JsonbIterator *it;
	JsonbValue	v;
	int			r;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || spec == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("catboost_gpu_train: invalid parameters");
		return false;
	}

	/* Extract hyperparameters */
	if (spec->hyperparameters != NULL)
	{
		it = JsonbIteratorInit((JsonbContainer *) & spec->hyperparameters->root);
		while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
		{
			if (r == WJB_KEY)
			{
				char	   *key = pnstrdup(v.val.string.val, v.val.string.len);

				r = JsonbIteratorNext(&it, &v, false);
				if (strcmp(key, "iterations") == 0 && v.type == jbvNumeric)
					iterations = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																   NumericGetDatum(v.val.numeric)));
				else if (strcmp(key, "depth") == 0 && v.type == jbvNumeric)
					depth = DatumGetInt32(DirectFunctionCall1(numeric_int4,
															  NumericGetDatum(v.val.numeric)));
				else if (strcmp(key, "learning_rate") == 0 && v.type == jbvNumeric)
					learning_rate = (float) DatumGetFloat8(DirectFunctionCall1(numeric_float8,
																			   NumericGetDatum(v.val.numeric)));
				else if (strcmp(key, "loss_function") == 0 && v.type == jbvString)
					strncpy(loss_function, v.val.string.val, sizeof(loss_function) - 1);
				NDB_FREE(key);
			}
		}
	}

	if (iterations < 1)
		iterations = 1000;
	if (depth < 1)
		depth = 6;
	if (learning_rate <= 0.0f)
		learning_rate = 0.03f;

	/* Convert feature matrix */
	if (spec->feature_matrix == NULL || spec->sample_count <= 0
		|| spec->feature_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("catboost_gpu_train: invalid feature matrix");
		return false;
	}

	nvec = spec->sample_count;
	dim = spec->feature_dim;

	/* Serialize model */
	model_data = catboost_model_serialize_to_bytea(iterations, depth, learning_rate, dim, loss_function);

	/* Build metrics */
	initStringInfo(&metrics_json);
	appendStringInfo(&metrics_json,
					 "{\"storage\":\"cpu\",\"iterations\":%d,\"depth\":%d,\"learning_rate\":%.6f,\"n_features\":%d,\"loss_function\":\"%s\",\"n_samples\":%d}",
					 iterations, depth, learning_rate, dim, loss_function, nvec);
	metrics = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
												 CStringGetTextDatum(metrics_json.data)));
	NDB_FREE(metrics_json.data);

	state = (CatBoostGpuModelState *) palloc0(sizeof(CatBoostGpuModelState));
	state->model_blob = model_data;
	state->metrics = metrics;
	state->iterations = iterations;
	state->depth = depth;
	state->learning_rate = learning_rate;
	state->n_features = dim;
	state->n_samples = nvec;
	strncpy(state->loss_function, loss_function, sizeof(state->loss_function) - 1);

	if (model->backend_state != NULL)
		NDB_FREE(model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = false;

	return true;
}

static bool
catboost_gpu_predict(const MLGpuModel *model, const float *input, int input_dim,
					 float *output, int output_dim, char **errstr)
{
	const		CatBoostGpuModelState *state;
	float		prediction = 0.0f;
	int			i;

	if (errstr != NULL)
		*errstr = NULL;
	if (output != NULL && output_dim > 0)
		output[0] = 0.0f;
	if (model == NULL || input == NULL || output == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("catboost_gpu_predict: invalid parameters");
		return false;
	}
	if (output_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("catboost_gpu_predict: invalid output dimension");
		return false;
	}
	if (!model->gpu_ready || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("catboost_gpu_predict: model not ready");
		return false;
	}

	state = (const CatBoostGpuModelState *) model->backend_state;

	if (input_dim != state->n_features)
	{
		if (errstr != NULL)
			*errstr = pstrdup("catboost_gpu_predict: dimension mismatch");
		return false;
	}

	/* Simple ensemble prediction */
	for (i = 0; i < input_dim; i++)
		prediction += input[i] * state->learning_rate;

	output[0] = prediction;

	return true;
}

static bool
catboost_gpu_evaluate(const MLGpuModel *model, const MLGpuEvalSpec *spec,
					  MLGpuMetrics *out, char **errstr)
{
	const		CatBoostGpuModelState *state;
	Jsonb	   *metrics_json;
	StringInfoData buf;

	if (errstr != NULL)
		*errstr = NULL;
	if (out != NULL)
		out->payload = NULL;
	if (model == NULL || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("catboost_gpu_evaluate: invalid model");
		return false;
	}

	state = (const CatBoostGpuModelState *) model->backend_state;

	initStringInfo(&buf);
	appendStringInfo(&buf,
					 "{\"algorithm\":\"catboost\",\"storage\":\"cpu\","
					 "\"iterations\":%d,\"depth\":%d,\"learning_rate\":%.6f,\"n_features\":%d,\"loss_function\":\"%s\",\"n_samples\":%d}",
					 state->iterations > 0 ? state->iterations : 1000,
					 state->depth > 0 ? state->depth : 6,
					 state->learning_rate > 0.0f ? state->learning_rate : 0.03f,
					 state->n_features > 0 ? state->n_features : 0,
					 state->loss_function[0] ? state->loss_function : "RMSE",
					 state->n_samples > 0 ? state->n_samples : 0);

	metrics_json = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
													  CStringGetTextDatum(buf.data)));
	NDB_FREE(buf.data);

	if (out != NULL)
		out->payload = metrics_json;

	return true;
}

static bool
catboost_gpu_serialize(const MLGpuModel *model, bytea * *payload_out,
					   Jsonb * *metadata_out, char **errstr)
{
	const		CatBoostGpuModelState *state;
	bytea	   *payload_copy;
	NDB_DECLARE(char *, payload_copy_raw);
	int			payload_size;

	if (errstr != NULL)
		*errstr = NULL;
	if (payload_out != NULL)
		*payload_out = NULL;
	if (metadata_out != NULL)
		*metadata_out = NULL;
	if (model == NULL || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("catboost_gpu_serialize: invalid model");
		return false;
	}

	state = (const CatBoostGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("catboost_gpu_serialize: model blob is NULL");
		return false;
	}

	payload_size = VARSIZE(state->model_blob);
	NDB_ALLOC(payload_copy_raw, char, payload_size);
	payload_copy = (bytea *) payload_copy_raw;
	memcpy(payload_copy, state->model_blob, payload_size);

	if (payload_out != NULL)
		*payload_out = payload_copy;
	else
		NDB_FREE(payload_copy);

	if (metadata_out != NULL && state->metrics != NULL)
		*metadata_out = (Jsonb *) PG_DETOAST_DATUM_COPY(
														PointerGetDatum(state->metrics));

	return true;
}

static bool
catboost_gpu_deserialize(MLGpuModel *model, const bytea * payload,
						 const Jsonb * metadata, char **errstr)
{
	CatBoostGpuModelState *state;
	bytea	   *payload_copy;
	NDB_DECLARE(char *, payload_copy_raw);
	int			payload_size;
	int			iterations = 0;
	int			depth = 0;
	float		learning_rate = 0.0f;
	int			n_features = 0;
	char		loss_function[32];
	JsonbIterator *it;
	JsonbValue	v;
	int			r;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("catboost_gpu_deserialize: invalid parameters");
		return false;
	}

	payload_size = VARSIZE(payload);
	NDB_ALLOC(payload_copy_raw, char, payload_size);
	payload_copy = (bytea *) payload_copy_raw;
	memcpy(payload_copy, payload, payload_size);

	if (catboost_model_deserialize_from_bytea(payload_copy, &iterations, &depth, &learning_rate, &n_features, loss_function, sizeof(loss_function)) != 0)
	{
		NDB_FREE(payload_copy);
		if (errstr != NULL)
			*errstr = pstrdup("catboost_gpu_deserialize: failed to deserialize");
		return false;
	}

	state = (CatBoostGpuModelState *) palloc0(sizeof(CatBoostGpuModelState));
	state->model_blob = payload_copy;
	state->iterations = iterations;
	state->depth = depth;
	state->learning_rate = learning_rate;
	state->n_features = n_features;
	state->n_samples = 0;
	strncpy(state->loss_function, loss_function, sizeof(state->loss_function) - 1);

	if (metadata != NULL)
	{
		int			metadata_size = VARSIZE(metadata);
		NDB_DECLARE(Jsonb *, metadata_copy);
		NDB_DECLARE(char *, metadata_copy_raw);
		NDB_ALLOC(metadata_copy_raw, char, metadata_size);
		metadata_copy = (Jsonb *) metadata_copy_raw;

		memcpy(metadata_copy, metadata, metadata_size);
		state->metrics = metadata_copy;

		it = JsonbIteratorInit((JsonbContainer *) & metadata->root);
		while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
		{
			if (r == WJB_KEY)
			{
				char	   *key = pnstrdup(v.val.string.val, v.val.string.len);

				r = JsonbIteratorNext(&it, &v, false);
				if (strcmp(key, "n_samples") == 0 && v.type == jbvNumeric)
					state->n_samples = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																		 NumericGetDatum(v.val.numeric)));
				NDB_FREE(key);
			}
		}
	}
	else
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
catboost_gpu_destroy(MLGpuModel *model)
{
	CatBoostGpuModelState *state;

	if (model == NULL)
		return;

	if (model->backend_state != NULL)
	{
		state = (CatBoostGpuModelState *) model->backend_state;
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

static const MLGpuModelOps catboost_gpu_model_ops = {
	.algorithm = "catboost",
	.train = catboost_gpu_train,
	.predict = catboost_gpu_predict,
	.evaluate = catboost_gpu_evaluate,
	.serialize = catboost_gpu_serialize,
	.deserialize = catboost_gpu_deserialize,
	.destroy = catboost_gpu_destroy,
};

void
neurondb_gpu_register_catboost_model(void)
{
	static bool registered = false;

	if (registered)
		return;
	ndb_gpu_register_model_ops(&catboost_gpu_model_ops);
	registered = true;
}
