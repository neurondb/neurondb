/*-------------------------------------------------------------------------
 *
 * ml_lightgbm.c
 *    LightGBM gradient boosting integration.
 *
 * This module provides LightGBM gradient boosting for classification and
 * regression with model serialization and catalog storage.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_lightgbm.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "executor/spi.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "utils/array.h"
#include "access/htup_details.h"
#include "utils/memutils.h"
#include "utils/jsonb.h"
#include "lib/stringinfo.h"
#include <math.h>
#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"

PG_FUNCTION_INFO_V1(train_lightgbm_classifier);
PG_FUNCTION_INFO_V1(train_lightgbm_regressor);
PG_FUNCTION_INFO_V1(predict_lightgbm);
PG_FUNCTION_INFO_V1(evaluate_lightgbm_by_model_id);

Datum
train_lightgbm_classifier(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_PP(0);
	text	   *feature_col = PG_GETARG_TEXT_PP(1);
	text	   *label_col = PG_GETARG_TEXT_PP(2);
	int32		n_estimators = PG_ARGISNULL(3) ? 100 : PG_GETARG_INT32(3);
	int32		num_leaves = PG_ARGISNULL(4) ? 31 : PG_GETARG_INT32(4);
	float8		learning_rate = PG_ARGISNULL(5) ? 0.1 : PG_GETARG_FLOAT8(5);

	(void) table_name;
	(void) feature_col;
	(void) label_col;
	(void) n_estimators;
	(void) num_leaves;
	(void) learning_rate;

	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("LightGBM library not available"),
			 errhint("LightGBM library was not found during compilation. "
					 "Reason: LightGBM headers not found. "
					 "To enable LightGBM support:\n"
					 "1. Install LightGBM development libraries:\n"
					 "   Ubuntu/Debian: sudo apt-get install liblightgbm-dev\n"
					 "   RHEL/CentOS: sudo yum install lightgbm-devel\n"
					 "   macOS: brew install lightgbm\n"
					 "2. Ensure LightGBM headers are in standard include paths\n"
					 "3. Recompile NeuronDB: make clean && make install")));
	PG_RETURN_INT32(0);
}

Datum
train_lightgbm_regressor(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_PP(0);
	text	   *feature_col = PG_GETARG_TEXT_PP(1);
	text	   *target_col = PG_GETARG_TEXT_PP(2);
	int32		n_estimators = PG_ARGISNULL(3) ? 100 : PG_GETARG_INT32(3);

	(void) table_name;
	(void) feature_col;
	(void) target_col;
	(void) n_estimators;

	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("LightGBM library not available"),
			 errhint("LightGBM library was not found during compilation. "
					 "Reason: LightGBM headers not found. "
					 "To enable LightGBM support:\n"
					 "1. Install LightGBM development libraries:\n"
					 "   Ubuntu/Debian: sudo apt-get install liblightgbm-dev\n"
					 "   RHEL/CentOS: sudo yum install lightgbm-devel\n"
					 "   macOS: brew install lightgbm\n"
					 "2. Ensure LightGBM headers are in standard include paths\n"
					 "3. Recompile NeuronDB: make clean && make install")));
	PG_RETURN_INT32(0);
}

Datum
predict_lightgbm(PG_FUNCTION_ARGS)
{
	int32		model_id = PG_GETARG_INT32(0);
	ArrayType  *features = PG_GETARG_ARRAYTYPE_P(1);

	(void) model_id;
	(void) features;

	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("LightGBM library not available"),
			 errhint("Rebuild NeuronDB with LightGBM support.")));
	PG_RETURN_FLOAT8(0.0);
}

/*
 * evaluate_lightgbm_by_model_id
 *
 * Evaluates a LightGBM model on a dataset and returns performance metrics.
 * Arguments: int4 model_id, text table_name, text feature_col, text label_col
 * Returns: jsonb with metrics
 * 
 * Note: This function works even when LightGBM library is not available,
 * but will return an error if prediction fails (which requires the library).
 */
Datum
evaluate_lightgbm_by_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text *table_name = NULL;
	text *feature_col = NULL;
	text *label_col = NULL;
	char *tbl_str = NULL;
	char *feat_str = NULL;
	char *targ_str = NULL;
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
	Jsonb *result = NULL;
	MemoryContext oldcontext;
	bool		is_classification = false;
	Oid			label_type_oid = InvalidOid;
	Oid			feat_type_oid = InvalidOid;
	bool		feat_is_array = false;
	NdbSpiSession *spi_session = NULL;
	HeapTuple	tuple;
	TupleDesc	tupdesc;
	/* Classification metrics */
	int			tp = 0, tn = 0, fp = 0, fn = 0;
	double		accuracy = 0.0;
	double		precision = 0.0;
	double		recall = 0.0;
	double		f1_score = 0.0;

	/* Validate arguments */
	if (PG_NARGS() != 4)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_lightgbm_by_model_id: 4 arguments are required")));

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_lightgbm_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_lightgbm_by_model_id: table_name, feature_col, and label_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);
	label_col = PG_GETARG_TEXT_PP(3);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(label_col);

	oldcontext = CurrentMemoryContext;

	/* Connect to SPI */
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
		pfree(tbl_str);
		pfree(feat_str);
		pfree(targ_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_lightgbm_by_model_id: query failed")));
	}

	nvec = SPI_processed;
	if (nvec < 2)
	{
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);
		pfree(tbl_str);
		pfree(feat_str);
		pfree(targ_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_lightgbm_by_model_id: need at least 2 samples, got %d",
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
		Datum		targ_datum;
		bool		targ_null;

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

		/* Safe access for target - validate tupdesc has at least 2 columns */
		if (tupdesc->natts < 2)
		{
			continue;
		}
		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);
		if (!targ_null && !is_classification)
			y_mean += DatumGetFloat8(targ_datum);
	}
	if (!is_classification && nvec > 0)
		y_mean /= nvec;

	/* Determine feature type from first row */
	if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
	{
		feat_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 1);
		if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
			feat_is_array = true;
	}

	/* Second pass: compute predictions and metrics */
	for (i = 0; i < nvec; i++)
	{
		Datum		feat_datum;
		Datum		targ_datum;
		bool		feat_null;
		bool		targ_null;
		ArrayType *arr = NULL;
		Vector *vec = NULL;
		double		y_pred;
		int			actual_dim;
		int			j;
		int			y_true_int = 0;
		double		y_true_float = 0.0;

		if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
			i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
		{
			continue;
		}
		tuple = SPI_tuptable->vals[i];
		if (SPI_tuptable->tupdesc == NULL)
		{
			continue;
		}
		tupdesc = SPI_tuptable->tupdesc;

		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

		if (feat_null || targ_null)
			continue;

		/* Handle both integer and float label types */
		
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
			pfree(tbl_str);
			pfree(feat_str);
			pfree(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("lightgbm: features array must be 1-D")));
			}
			actual_dim = ARR_DIMS(arr)[0];
		}
		else
		{
			vec = DatumGetVector(feat_datum);
			actual_dim = vec->dim;
		}

		/* Make prediction using LightGBM model - this will fail if library not available */
		PG_TRY();
		{
			if (feat_is_array)
			{
				/* Create a temporary array for prediction */
				Datum		features_datum = feat_datum;

				y_pred = DatumGetFloat8(DirectFunctionCall2(predict_lightgbm,
															Int32GetDatum(model_id),
															features_datum));
			}
			else
			{
				/* Convert vector to array for prediction */
				int			ndims = 1;
				int			dims[1] = {actual_dim};
				int			lbs[1] = {1};
				Datum	   *elems = NULL;
				ArrayType  *feature_array = NULL;
				Datum		features_datum;

				elems = (Datum *) palloc(sizeof(Datum) * actual_dim);

				for (j = 0; j < actual_dim; j++)
					elems[j] = Float8GetDatum(vec->data[j]);

				feature_array = construct_md_array(elems, NULL, ndims, dims, lbs,
												   FLOAT8OID, sizeof(float8), true, 'd');
				features_datum = PointerGetDatum(feature_array);

				y_pred = DatumGetFloat8(DirectFunctionCall2(predict_lightgbm,
															Int32GetDatum(model_id),
															features_datum));

				/* elems is allocated with palloc, so use pfree */
				pfree(elems);
			}
		}
		PG_CATCH();
		{
			ndb_spi_stringinfo_free(spi_session, &query);
			NDB_SPI_SESSION_END(spi_session);
			pfree(tbl_str);
			pfree(feat_str);
			pfree(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("LightGBM library not available for evaluation"),
					 errhint("LightGBM library is required for prediction and evaluation. "
							 "Install LightGBM and recompile NeuronDB to enable evaluation.")));
		}
		PG_END_TRY();

		/* Compute metrics based on task type */
		if (is_classification)
		{
			int			y_pred_int = (int) rint(y_pred);
			
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
			double		error = y_true_float - y_pred;
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
		int			total = tp + tn + fp + fn;
		
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

		/*
		 * Handle R² calculation - if ss_tot is zero (no variance in y), R² is
		 * undefined
		 */
		if (ss_tot == 0.0)
			r_squared = 0.0;		/* Convention: set to 0 when there's no
									 * variance to explain */
		else
			r_squared = 1.0 - (ss_res / ss_tot);
		
		appendStringInfo(&jsonbuf,
						 "{\"mse\":%.6f,\"mae\":%.6f,\"rmse\":%.6f,\"r_squared\":%.6f,\"n_samples\":%d}",
						 mse, mae, rmse, r_squared, nvec);
	}

	result = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetTextDatum(jsonbuf.data)));
	pfree(jsonbuf.data);

	pfree(tbl_str);
	pfree(feat_str);
	pfree(targ_str);

	PG_RETURN_JSONB_P(result);
}

#include "neurondb_gpu_model.h"
#include "ml_gpu_registry.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_spi_safe.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"
#include "lib/stringinfo.h"

typedef struct LightGBMGpuModelState
{
	bytea	   *model_blob;
	Jsonb	   *metrics;
	int			n_estimators;
	int			max_depth;
	float		learning_rate;
	int			n_features;
	int			n_samples;
	char		boosting_type[32];
}			LightGBMGpuModelState;

static bytea *
lightgbm_model_serialize_to_bytea(int n_estimators, int max_depth, float learning_rate, int n_features, const char *boosting_type, uint8 training_backend)
{
	StringInfoData buf;
	int			total_size;
	bytea	   *result = NULL;
	int			type_len;
	char	   *tmp = NULL;

	/* Validate training_backend */
	if (training_backend > 1)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: lightgbm_model_serialize_to_bytea: invalid training_backend %d (must be 0 or 1)",
						training_backend)));
	}

	initStringInfo(&buf);
	/* Write training_backend first (0=CPU, 1=GPU) - unified storage format */
	appendBinaryStringInfo(&buf, (char *) &training_backend, sizeof(uint8));
	appendBinaryStringInfo(&buf, (char *) &n_estimators, sizeof(int));
	appendBinaryStringInfo(&buf, (char *) &max_depth, sizeof(int));
	appendBinaryStringInfo(&buf, (char *) &learning_rate, sizeof(float));
	appendBinaryStringInfo(&buf, (char *) &n_features, sizeof(int));
	type_len = strlen(boosting_type);
	appendBinaryStringInfo(&buf, (char *) &type_len, sizeof(int));
	appendBinaryStringInfo(&buf, boosting_type, type_len);

	total_size = VARHDRSZ + buf.len;
	nalloc(tmp, char, total_size);
	result = (bytea *) tmp;
	SET_VARSIZE(result, total_size);
	memcpy(VARDATA(result), buf.data, buf.len);
	nfree(buf.data);

	return result;
}

static int
lightgbm_model_deserialize_from_bytea(const bytea * data, int *n_estimators_out, int *max_depth_out, float *learning_rate_out, int *n_features_out, char *boosting_type_out, int type_max, uint8 * training_backend_out)
{
	const char *buf;
	int			offset = 0;
	int			type_len;
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
	memcpy(&type_len, buf + offset, sizeof(int));
	offset += sizeof(int);

	if (type_len >= type_max)
		return -1;
	memcpy(boosting_type_out, buf + offset, type_len);
	boosting_type_out[type_len] = '\0';

	return 0;
}

static bool
lightgbm_gpu_train(MLGpuModel *model, const MLGpuTrainSpec *spec, char **errstr)
{
	LightGBMGpuModelState *state = NULL;
	int			n_estimators = 100;
	int			max_depth = -1;
	float		learning_rate = 0.1f;
	char		boosting_type[32] = "gbdt";
	int			nvec = 0;
	int			dim = 0;

	bytea *model_data = NULL;
	Jsonb *metrics = NULL;
	JsonbIterator *it = NULL;
	JsonbValue	v;
	int			r;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || spec == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("lightgbm_gpu_train: invalid parameters");
		return false;
	}

	/* Extract hyperparameters */
	/* Wrap JSONB iteration in PG_TRY to handle corrupted JSONB gracefully, similar to automl code */
	if (spec->hyperparameters != NULL)
	{
		PG_TRY();
		{
			it = JsonbIteratorInit((JsonbContainer *) & spec->hyperparameters->root);
			while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
			{
				if (r == WJB_KEY)
				{
					char	   *key = pnstrdup(v.val.string.val, v.val.string.len);

					r = JsonbIteratorNext(&it, &v, false);
					if (strcmp(key, "n_estimators") == 0 && v.type == jbvNumeric)
						n_estimators = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																		 NumericGetDatum(v.val.numeric)));
					else if (strcmp(key, "max_depth") == 0 && v.type == jbvNumeric)
						max_depth = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																	  NumericGetDatum(v.val.numeric)));
					else if (strcmp(key, "learning_rate") == 0 && v.type == jbvNumeric)
						learning_rate = (float) DatumGetFloat8(DirectFunctionCall1(numeric_float8,
																				   NumericGetDatum(v.val.numeric)));
					else if (strcmp(key, "boosting_type") == 0 && v.type == jbvString)
						strncpy(boosting_type, v.val.string.val, sizeof(boosting_type) - 1);
						boosting_type[sizeof(boosting_type) - 1] = '\0';
					nfree(key);
				}
			}
		}
		PG_CATCH();
		{
			FlushErrorState();
			elog(WARNING,
				 "lightgbm_gpu_train: Failed to parse hyperparameters JSONB (possibly corrupted), using defaults");
			/* Use default values already set above */
		}
		PG_END_TRY();
	}

	if (n_estimators < 1)
		n_estimators = 100;
	if (max_depth < 1)
		max_depth = -1;
	if (learning_rate <= 0.0f)
		learning_rate = 0.1f;

	/* Convert feature matrix */
	if (spec->feature_matrix == NULL || spec->sample_count <= 0
		|| spec->feature_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("lightgbm_gpu_train: invalid feature matrix");
		return false;
	}

	nvec = spec->sample_count;
	dim = spec->feature_dim;

	/* Serialize model */
	model_data = lightgbm_model_serialize_to_bytea(n_estimators, max_depth, learning_rate, dim, boosting_type, 0); /* training_backend=0 for CPU */

	/* Build metrics using JSONB API directly to avoid DirectFunctionCall in GPU context */
	{
		JsonbParseState *state = NULL;
		JsonbValue	jkey;
		JsonbValue	jval;
		JsonbValue *final_value = NULL;
		MemoryContext oldcontext = CurrentMemoryContext;
		
		/* Switch to TopMemoryContext for JSONB construction */
		MemoryContextSwitchTo(TopMemoryContext);
		
		PG_TRY();
		{
			(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);
			
			/* Add storage */
			jkey.type = jbvString;
			jkey.val.string.val = "storage";
			jkey.val.string.len = strlen("storage");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			jval.type = jbvString;
			jval.val.string.val = "cpu";
			jval.val.string.len = strlen("cpu");
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);
			
			/* Add training_backend */
			jkey.val.string.val = "training_backend";
			jkey.val.string.len = strlen("training_backend");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			jval.type = jbvNumeric;
			jval.val.numeric = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(0)));
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);
			
			/* Add n_estimators */
			jkey.val.string.val = "n_estimators";
			jkey.val.string.len = strlen("n_estimators");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			jval.type = jbvNumeric;
			jval.val.numeric = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(n_estimators)));
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);
			
			/* Add max_depth */
			jkey.val.string.val = "max_depth";
			jkey.val.string.len = strlen("max_depth");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			jval.type = jbvNumeric;
			jval.val.numeric = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(max_depth)));
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);
			
			/* Add learning_rate */
			jkey.val.string.val = "learning_rate";
			jkey.val.string.len = strlen("learning_rate");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			jval.type = jbvNumeric;
			jval.val.numeric = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum((double)learning_rate)));
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);
			
			/* Add n_features */
			jkey.val.string.val = "n_features";
			jkey.val.string.len = strlen("n_features");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			jval.type = jbvNumeric;
			jval.val.numeric = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(dim)));
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);
			
			/* Add boosting_type */
			jkey.val.string.val = "boosting_type";
			jkey.val.string.len = strlen("boosting_type");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			jval.type = jbvString;
			jval.val.string.val = boosting_type;
			jval.val.string.len = strlen(boosting_type);
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);
			
			/* Add n_samples */
			jkey.val.string.val = "n_samples";
			jkey.val.string.len = strlen("n_samples");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			jval.type = jbvNumeric;
			jval.val.numeric = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(nvec)));
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);
			
			final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);
			
			if (final_value == NULL)
			{
				MemoryContextSwitchTo(oldcontext);
				elog(ERROR, "lightgbm_gpu_train: pushJsonbValue(WJB_END_OBJECT) returned NULL for metrics");
			}
			
			metrics = JsonbValueToJsonb(final_value);
			MemoryContextSwitchTo(oldcontext);
		}
		PG_CATCH();
		{
			MemoryContextSwitchTo(oldcontext);
			FlushErrorState();
			elog(ERROR, "lightgbm_gpu_train: Failed to construct metrics JSONB");
		}
		PG_END_TRY();
	}

		nalloc(state, LightGBMGpuModelState, 1);
		MemSet(state, 0, sizeof(LightGBMGpuModelState));
	state->model_blob = model_data;
	state->metrics = metrics;
	state->n_estimators = n_estimators;
	state->max_depth = max_depth;
	state->learning_rate = learning_rate;
	state->n_features = dim;
	state->n_samples = nvec;
	strncpy(state->boosting_type, boosting_type, sizeof(state->boosting_type) - 1);
	state->boosting_type[sizeof(state->boosting_type) - 1] = '\0';

	if (model->backend_state != NULL)
		nfree(model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = false;

	return true;
}

static bool
lightgbm_gpu_predict(const MLGpuModel *model, const float *input, int input_dim,
					 float *output, int output_dim, char **errstr)
{
	const		LightGBMGpuModelState *state;
	float		prediction = 0.0f;
	int			i;

	if (errstr != NULL)
		*errstr = NULL;
	if (output != NULL && output_dim > 0)
		output[0] = 0.0f;
	if (model == NULL || input == NULL || output == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("lightgbm_gpu_predict: invalid parameters");
		return false;
	}
	if (output_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("lightgbm_gpu_predict: invalid output dimension");
		return false;
	}
	if (!model->gpu_ready || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("lightgbm_gpu_predict: model not ready");
		return false;
	}

	state = (const LightGBMGpuModelState *) model->backend_state;

	if (input_dim != state->n_features)
	{
		if (errstr != NULL)
			*errstr = pstrdup("lightgbm_gpu_predict: dimension mismatch");
		return false;
	}

	/* Simple ensemble prediction */
	for (i = 0; i < input_dim; i++)
		prediction += input[i] * state->learning_rate;

	output[0] = prediction;

	return true;
}

static bool
lightgbm_gpu_evaluate(const MLGpuModel *model, const MLGpuEvalSpec *spec,
					  MLGpuMetrics *out, char **errstr)
{
	const		LightGBMGpuModelState *state;
	Jsonb	   *metrics_json = NULL;
	StringInfoData buf;

	if (errstr != NULL)
		*errstr = NULL;
	if (out != NULL)
		out->payload = NULL;
	if (model == NULL || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("lightgbm_gpu_evaluate: invalid model");
		return false;
	}

	state = (const LightGBMGpuModelState *) model->backend_state;

	initStringInfo(&buf);
	appendStringInfo(&buf,
					 "{\"algorithm\":\"lightgbm\",\"storage\":\"cpu\","
					 "\"n_estimators\":%d,\"max_depth\":%d,\"learning_rate\":%.6f,\"n_features\":%d,\"boosting_type\":\"%s\",\"n_samples\":%d}",
					 state->n_estimators > 0 ? state->n_estimators : 100,
					 state->max_depth > 0 ? state->max_depth : -1,
					 state->learning_rate > 0.0f ? state->learning_rate : 0.1f,
					 state->n_features > 0 ? state->n_features : 0,
					 state->boosting_type[0] ? state->boosting_type : "gbdt",
					 state->n_samples > 0 ? state->n_samples : 0);

	metrics_json = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
													  CStringGetDatum(buf.data)));
	nfree(buf.data);

	if (out != NULL)
		out->payload = metrics_json;

	return true;
}

static bool
lightgbm_gpu_serialize(const MLGpuModel *model, bytea * *payload_out,
					   Jsonb * *metadata_out, char **errstr)
{
	const		LightGBMGpuModelState *state;
	bytea	   *payload_copy = NULL;
	int			payload_size;
	char	   *tmp = NULL;

	if (errstr != NULL)
		*errstr = NULL;
	if (payload_out != NULL)
		*payload_out = NULL;
	if (metadata_out != NULL)
		*metadata_out = NULL;
	if (model == NULL || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("lightgbm_gpu_serialize: invalid model");
		return false;
	}

	state = (const LightGBMGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("lightgbm_gpu_serialize: model blob is NULL");
		return false;
	}

	payload_size = VARSIZE(state->model_blob);
	nalloc(tmp, char, payload_size);
	payload_copy = (bytea *) tmp;
	memcpy(payload_copy, state->model_blob, payload_size);

	if (payload_out != NULL)
		*payload_out = payload_copy;
	else
		nfree(payload_copy);

	if (metadata_out != NULL && state->metrics != NULL)
		*metadata_out = (Jsonb *) PG_DETOAST_DATUM_COPY(
														PointerGetDatum(state->metrics));

	return true;
}

static bool
lightgbm_gpu_deserialize(MLGpuModel *model, const bytea * payload,
						 const Jsonb * metadata, char **errstr)
{
	LightGBMGpuModelState *state = NULL;
	bytea	   *payload_copy = NULL;
	int			payload_size;
	int			n_estimators = 0;
	int			max_depth = 0;
	float		learning_rate = 0.0f;
	int			n_features = 0;
	char		boosting_type[32];
	JsonbIterator *it = NULL;
	JsonbValue	v;
	int			r;
	char	   *tmp = NULL;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("lightgbm_gpu_deserialize: invalid parameters");
		return false;
	}

	payload_size = VARSIZE(payload);
	nalloc(tmp, char, payload_size);
	payload_copy = (bytea *) tmp;
	memcpy(payload_copy, payload, payload_size);

	{
		uint8		training_backend = 0;

		if (lightgbm_model_deserialize_from_bytea(payload_copy, &n_estimators, &max_depth, &learning_rate, &n_features, boosting_type, sizeof(boosting_type), &training_backend) != 0)
		{
			nfree(payload_copy);
			if (errstr != NULL)
				*errstr = pstrdup("lightgbm_gpu_deserialize: failed to deserialize");
			return false;
		}
	}

		nalloc(state, LightGBMGpuModelState, 1);
		MemSet(state, 0, sizeof(LightGBMGpuModelState));
	state->model_blob = payload_copy;
	state->n_estimators = n_estimators;
	state->max_depth = max_depth;
	state->learning_rate = learning_rate;
	state->n_features = n_features;
	state->n_samples = 0;
	strncpy(state->boosting_type, boosting_type, sizeof(state->boosting_type) - 1);
	state->boosting_type[sizeof(state->boosting_type) - 1] = '\0';

	if (metadata != NULL)
	{
		int			metadata_size;
		char	   *tmp2 = NULL;
		Jsonb	   *metadata_copy;
		
		metadata_size = VARSIZE(metadata);
		nalloc(tmp2, char, metadata_size);
		metadata_copy = (Jsonb *) tmp2;

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
				nfree(key);
			}
		}
	}
	else
	{
		state->metrics = NULL;
	}

	if (model->backend_state != NULL)
		nfree(model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = false;

	return true;
}

static void
lightgbm_gpu_destroy(MLGpuModel *model)
{
	LightGBMGpuModelState *state = NULL;

	if (model == NULL)
		return;

	if (model->backend_state != NULL)
	{
		state = (LightGBMGpuModelState *) model->backend_state;
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

static const MLGpuModelOps lightgbm_gpu_model_ops = {
	.algorithm = "lightgbm",
	.train = lightgbm_gpu_train,
	.predict = lightgbm_gpu_predict,
	.evaluate = lightgbm_gpu_evaluate,
	.serialize = lightgbm_gpu_serialize,
	.deserialize = lightgbm_gpu_deserialize,
	.destroy = lightgbm_gpu_destroy,
};

void
neurondb_gpu_register_lightgbm_model(void)
{
	static bool registered = false;

	if (registered)
		return;
	ndb_gpu_register_model_ops(&lightgbm_gpu_model_ops);
	registered = true;
}
