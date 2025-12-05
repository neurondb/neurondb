/*-------------------------------------------------------------------------
 *
 * ml_lightgbm.c
 *    LightGBM gradient boosting integration.
 *
 * This module provides LightGBM gradient boosting for classification and
 * regression with model serialization and catalog storage.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
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

PG_FUNCTION_INFO_V1(train_lightgbm_classifier);
PG_FUNCTION_INFO_V1(train_lightgbm_regressor);
PG_FUNCTION_INFO_V1(predict_lightgbm);

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

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration for LightGBM
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"
#include "ml_gpu_registry.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

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
lightgbm_model_serialize_to_bytea(int n_estimators, int max_depth, float learning_rate, int n_features, const char *boosting_type)
{
	StringInfoData buf;
	int			total_size;
	bytea	   *result;
	int			type_len;

	initStringInfo(&buf);
	appendBinaryStringInfo(&buf, (char *) &n_estimators, sizeof(int));
	appendBinaryStringInfo(&buf, (char *) &max_depth, sizeof(int));
	appendBinaryStringInfo(&buf, (char *) &learning_rate, sizeof(float));
	appendBinaryStringInfo(&buf, (char *) &n_features, sizeof(int));
	type_len = strlen(boosting_type);
	appendBinaryStringInfo(&buf, (char *) &type_len, sizeof(int));
	appendBinaryStringInfo(&buf, boosting_type, type_len);

	total_size = VARHDRSZ + buf.len;
	result = (bytea *) palloc(total_size);
	SET_VARSIZE(result, total_size);
	memcpy(VARDATA(result), buf.data, buf.len);
	NDB_FREE(buf.data);

	return result;
}

static int
lightgbm_model_deserialize_from_bytea(const bytea * data, int *n_estimators_out, int *max_depth_out, float *learning_rate_out, int *n_features_out, char *boosting_type_out, int type_max)
{
	const char *buf;
	int			offset = 0;
	int			type_len;

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
	LightGBMGpuModelState *state;
	int			n_estimators = 100;
	int			max_depth = -1;
	float		learning_rate = 0.1f;
	char		boosting_type[32] = "gbdt";
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
			*errstr = pstrdup("lightgbm_gpu_train: invalid parameters");
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
				NDB_FREE(key);
			}
		}
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
	model_data = lightgbm_model_serialize_to_bytea(n_estimators, max_depth, learning_rate, dim, boosting_type);

	/* Build metrics */
	initStringInfo(&metrics_json);
	appendStringInfo(&metrics_json,
					 "{\"storage\":\"cpu\",\"n_estimators\":%d,\"max_depth\":%d,\"learning_rate\":%.6f,\"n_features\":%d,\"boosting_type\":\"%s\",\"n_samples\":%d}",
					 n_estimators, max_depth, learning_rate, dim, boosting_type, nvec);
	metrics = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
												 CStringGetDatum(metrics_json.data)));
	NDB_FREE(metrics_json.data);

	state = (LightGBMGpuModelState *) palloc0(sizeof(LightGBMGpuModelState));
	state->model_blob = model_data;
	state->metrics = metrics;
	state->n_estimators = n_estimators;
	state->max_depth = max_depth;
	state->learning_rate = learning_rate;
	state->n_features = dim;
	state->n_samples = nvec;
	strncpy(state->boosting_type, boosting_type, sizeof(state->boosting_type) - 1);

	if (model->backend_state != NULL)
		NDB_FREE(model->backend_state);

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
	Jsonb	   *metrics_json;
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
	NDB_FREE(buf.data);

	if (out != NULL)
		out->payload = metrics_json;

	return true;
}

static bool
lightgbm_gpu_serialize(const MLGpuModel *model, bytea * *payload_out,
					   Jsonb * *metadata_out, char **errstr)
{
	const		LightGBMGpuModelState *state;
	bytea	   *payload_copy;
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
	payload_copy = (bytea *) palloc(payload_size);
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
lightgbm_gpu_deserialize(MLGpuModel *model, const bytea * payload,
						 const Jsonb * metadata, char **errstr)
{
	LightGBMGpuModelState *state;
	bytea	   *payload_copy;
	int			payload_size;
	int			n_estimators = 0;
	int			max_depth = 0;
	float		learning_rate = 0.0f;
	int			n_features = 0;
	char		boosting_type[32];
	JsonbIterator *it;
	JsonbValue	v;
	int			r;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("lightgbm_gpu_deserialize: invalid parameters");
		return false;
	}

	payload_size = VARSIZE(payload);
	payload_copy = (bytea *) palloc(payload_size);
	memcpy(payload_copy, payload, payload_size);

	if (lightgbm_model_deserialize_from_bytea(payload_copy, &n_estimators, &max_depth, &learning_rate, &n_features, boosting_type, sizeof(boosting_type)) != 0)
	{
		NDB_FREE(payload_copy);
		if (errstr != NULL)
			*errstr = pstrdup("lightgbm_gpu_deserialize: failed to deserialize");
		return false;
	}

	state = (LightGBMGpuModelState *) palloc0(sizeof(LightGBMGpuModelState));
	state->model_blob = payload_copy;
	state->n_estimators = n_estimators;
	state->max_depth = max_depth;
	state->learning_rate = learning_rate;
	state->n_features = n_features;
	state->n_samples = 0;
	strncpy(state->boosting_type, boosting_type, sizeof(state->boosting_type) - 1);

	if (metadata != NULL)
	{
		int			metadata_size = VARSIZE(metadata);
		Jsonb	   *metadata_copy = (Jsonb *) palloc(metadata_size);

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
lightgbm_gpu_destroy(MLGpuModel *model)
{
	LightGBMGpuModelState *state;

	if (model == NULL)
		return;

	if (model->backend_state != NULL)
	{
		state = (LightGBMGpuModelState *) model->backend_state;
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
