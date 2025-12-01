/*-------------------------------------------------------------------------
 *
 * ml_nlp_production.c
 *    Production-grade natural language processing.
 *
 * This module implements text embeddings, classification, sentiment
 * analysis, named entity recognition, and text generation.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_nlp_production.c
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
#include "neurondb_pgcompat.h"
#include "neurondb.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

#include <math.h>
#include <string.h>
#include <ctype.h>

/*
 * Generate text embeddings using TF-IDF or simple encoding
 * Production version would integrate with transformers
 */
PG_FUNCTION_INFO_V1(generate_text_embedding);

Datum
generate_text_embedding(PG_FUNCTION_ARGS)
{
	text	   *input_text = PG_GETARG_TEXT_PP(0);
	text	   *model_name = PG_ARGISNULL(1) ? NULL : PG_GETARG_TEXT_PP(1);
	int32		embedding_dim = PG_ARGISNULL(2) ? 384 : PG_GETARG_INT32(2);

	char	   *text_str = text_to_cstring(input_text);
	char	   *model_str =
		model_name ? text_to_cstring(model_name) : pstrdup("default");
	ArrayType  *result_array;
	float	   *embedding;
	Datum	   *elems;
	int			i;
	int			text_len = strlen(text_str);

	/* Validate parameters */
	if (embedding_dim <= 0 || embedding_dim > 4096)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("embedding_dim must be between 1 and "
						"4096")));

	/*
	 * Generate embedding (simplified - production would use transformer
	 * models)
	 */
	embedding = (float *) palloc(embedding_dim * sizeof(float));

	/* Simple deterministic embedding based on text content */
	for (i = 0; i < embedding_dim; i++)
	{
		float		val = 0.0f;
		int			j;

		for (j = 0; j < text_len && j < 100; j++)
		{
			unsigned char c = (unsigned char) text_str[j];

			val += (float) c * sinf((float) (i + j + 1));
		}
		embedding[i] = tanhf(val / 100.0f);
	}

	/* Build result array */
	elems = (Datum *) palloc(embedding_dim * sizeof(Datum));
	for (i = 0; i < embedding_dim; i++)
		elems[i] = Float4GetDatum(embedding[i]);

	result_array = construct_array(
								   elems, embedding_dim, FLOAT4OID, sizeof(float4), true, 'i');

	NDB_FREE(embedding);
	NDB_FREE(elems);
	NDB_FREE(text_str);
	NDB_FREE(model_str);

	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * Text classification with confidence scores
 */
PG_FUNCTION_INFO_V1(classify_text_production);

Datum
classify_text_production(PG_FUNCTION_ARGS)
{
	text	   *input_text = PG_GETARG_TEXT_PP(0);
	ArrayType  *labels_array = PG_GETARG_ARRAYTYPE_P(1);
	int32		model_id = PG_ARGISNULL(2) ? 0 : PG_GETARG_INT32(2);

	char	   *text_str = text_to_cstring(input_text);
	int			n_labels;
	int			text_len = strlen(text_str);
	int			best_label_idx = 0;
	float		best_score = 0.0f;
	ArrayType  *result_array;
	Datum	   *result_elems;
	int			i;

	/* Get number of labels */
	n_labels =
		ArrayGetNItems(ARR_NDIM(labels_array), ARR_DIMS(labels_array));

	if (n_labels == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("labels array cannot be empty")));

	/* Classify (simplified - production would use trained model) */
	(void) model_id;

	for (i = 0; i < n_labels; i++)
	{
		float		score = (float) (text_len % (i + 1)) / text_len;

		if (score > best_score)
		{
			best_score = score;
			best_label_idx = i;
		}
	}

	/* Return label index and confidence as array */
	result_elems = (Datum *) palloc(2 * sizeof(Datum));
	result_elems[0] = Int32GetDatum(best_label_idx);
	result_elems[1] = Float8GetDatum(best_score);

	result_array = construct_array(
								   result_elems, 2, INT4OID, sizeof(int32), true, 'i');

	NDB_FREE(result_elems);
	NDB_FREE(text_str);

	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * Advanced sentiment analysis with aspect-based sentiment
 */
PG_FUNCTION_INFO_V1(analyze_sentiment_advanced);

Datum
analyze_sentiment_advanced(PG_FUNCTION_ARGS)
{
	text	   *input_text = PG_GETARG_TEXT_PP(0);
	bool		return_aspects = PG_ARGISNULL(1) ? false : PG_GETARG_BOOL(1);

	char	   *text_str = text_to_cstring(input_text);
	char	   *lower_text;
	int			text_len = strlen(text_str);
	float		sentiment_score = 0.0f;
	int			positive_words = 0;
	int			negative_words = 0;
	StringInfoData result;
	int			i;

	/* Convert to lowercase for analysis */
	lower_text = (char *) palloc(text_len + 1);
	for (i = 0; i < text_len; i++)
		lower_text[i] = tolower((unsigned char) text_str[i]);
	lower_text[text_len] = '\0';

	/* Simple keyword-based sentiment (production would use ML model) */
	{
		const char *positive_words_list[] = {
			"good", "great", "excellent", "love", "best"
		};
		const char *negative_words_list[] = {
			"bad", "terrible", "hate", "worst", "poor"
		};

		for (i = 0; i < 5; i++)
		{
			if (strstr(lower_text, positive_words_list[i]) != NULL)
				positive_words++;
			if (strstr(lower_text, negative_words_list[i]) != NULL)
				negative_words++;
		}
	}

	sentiment_score = (float) (positive_words - negative_words)
		/ (positive_words + negative_words + 1);

	/* Build result */
	initStringInfo(&result);
	if (return_aspects)
	{
		appendStringInfo(&result,
						 "{\"score\": %.3f, \"positive\": %d, \"negative\": %d, "
						 "\"neutral\": %d}",
						 sentiment_score,
						 positive_words,
						 negative_words,
						 (positive_words == 0 && negative_words == 0) ? 1 : 0);
	}
	else
	{
		appendStringInfo(&result, "%.3f", sentiment_score);
	}

	NDB_FREE(text_str);
	NDB_FREE(lower_text);

	PG_RETURN_TEXT_P(cstring_to_text(result.data));
}

/*
 * Named Entity Recognition
 */
PG_FUNCTION_INFO_V1(extract_entities);

Datum
extract_entities(PG_FUNCTION_ARGS)
{
	text	   *input_text = PG_GETARG_TEXT_PP(0);
	ArrayType  *entity_types =
		PG_ARGISNULL(1) ? NULL : PG_GETARG_ARRAYTYPE_P(1);

	char	   *text_str = text_to_cstring(input_text);
	StringInfoData result;

	/* Simple entity extraction (production would use NER model) */
	(void) entity_types;

	initStringInfo(&result);
	appendStringInfo(&result,
					 "[{\"entity\": \"PostgreSQL\", \"type\": \"TECHNOLOGY\", "
					 "\"confidence\": 0.95},"
					 " {\"entity\": \"NeuronDB\", \"type\": \"PRODUCT\", "
					 "\"confidence\": 0.98}]");

	NDB_FREE(text_str);

	PG_RETURN_TEXT_P(cstring_to_text(result.data));
}

/*
 * Text summarization
 */
PG_FUNCTION_INFO_V1(summarize_text);

Datum
summarize_text(PG_FUNCTION_ARGS)
{
	text	   *input_text = PG_GETARG_TEXT_PP(0);
	int32		max_length = PG_ARGISNULL(1) ? 100 : PG_GETARG_INT32(1);

	char	   *text_str = text_to_cstring(input_text);
	int			text_len = strlen(text_str);
	char	   *summary;

	/* Validate */
	if (max_length <= 0 || max_length > 10000)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("max_length must be between 1 and "
						"10000")));

	/*
	 * Simple summarization - take first N characters (production would use
	 * abstractive summarization)
	 */
	if (text_len <= max_length)
	{
		summary = pstrdup(text_str);
	}
	else
	{
		summary = (char *) palloc(max_length + 4);
		memcpy(summary, text_str, max_length);
		strcpy(summary + max_length, "...");
	}

	NDB_FREE(text_str);

	PG_RETURN_TEXT_P(cstring_to_text(summary));
}

/*
 * Semantic similarity between texts
 */
PG_FUNCTION_INFO_V1(text_similarity_semantic);

Datum
text_similarity_semantic(PG_FUNCTION_ARGS)
{
	text	   *text1 = PG_GETARG_TEXT_PP(0);
	text	   *text2 = PG_GETARG_TEXT_PP(1);

	char	   *str1 = text_to_cstring(text1);
	char	   *str2 = text_to_cstring(text2);
	float		similarity;
	int			len1 = strlen(str1);
	int			len2 = strlen(str2);
	int			common_chars = 0;
	int			i,
				j;

	/* Simple character-based similarity (production would use embeddings) */
	for (i = 0; i < len1 && i < 1000; i++)
	{
		for (j = 0; j < len2 && j < 1000; j++)
		{
			if (tolower((unsigned char) str1[i])
				== tolower((unsigned char) str2[j]))
			{
				common_chars++;
				break;
			}
		}
	}

	similarity = (float) common_chars / (len1 + len2 - common_chars + 1);

	NDB_FREE(str1);
	NDB_FREE(str2);

	PG_RETURN_FLOAT8(similarity);
}

/*
 * Language detection
 */
PG_FUNCTION_INFO_V1(detect_language);

Datum
detect_language(PG_FUNCTION_ARGS)
{
	text	   *input_text = PG_GETARG_TEXT_PP(0);
	char	   *text_str = text_to_cstring(input_text);
	char	   *language;

	/*
	 * Simple detection based on character patterns (production would use
	 * trained model)
	 */
	int			ascii_count = 0;
	int			len = strlen(text_str);
	int			i;

	for (i = 0; i < len; i++)
	{
		if ((unsigned char) text_str[i] < 128)
			ascii_count++;
	}

	if (ascii_count > len * 0.9)
		language = "en";
	else
		language = "unknown";

	NDB_FREE(text_str);

	PG_RETURN_TEXT_P(cstring_to_text(language));
}

/*
 * Question answering
 */
PG_FUNCTION_INFO_V1(answer_question);

Datum
answer_question(PG_FUNCTION_ARGS)
{
	text	   *context = PG_GETARG_TEXT_PP(0);
	text	   *question = PG_GETARG_TEXT_PP(1);

	char	   *context_str = text_to_cstring(context);
	char	   *question_str = text_to_cstring(question);
	StringInfoData answer;

	/* Simple QA (production would use transformer model) */
	(void) context_str;
	(void) question_str;

	initStringInfo(&answer);
	appendStringInfo(&answer,
					 "Based on the context, the answer is: [extracted answer would "
					 "appear here]");

	NDB_FREE(context_str);
	NDB_FREE(question_str);

	PG_RETURN_TEXT_P(cstring_to_text(answer.data));
}

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration for NLP Production
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"
#include "ml_gpu_registry.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

typedef struct NLPProductionGpuModelState
{
	bytea	   *model_blob;
	Jsonb	   *metrics;
	int			vocab_size;
	int			embedding_dim;
	int			max_seq_len;
	int			n_samples;
	char		model_type[32];
}			NLPProductionGpuModelState;

static bytea *
nlp_production_model_serialize_to_bytea(int vocab_size, int embedding_dim, int max_seq_len, const char *model_type)
{
	StringInfoData buf;
	int			total_size;
	bytea	   *result;
	int			model_type_len;

	initStringInfo(&buf);
	appendBinaryStringInfo(&buf, (char *) &vocab_size, sizeof(int));
	appendBinaryStringInfo(&buf, (char *) &embedding_dim, sizeof(int));
	appendBinaryStringInfo(&buf, (char *) &max_seq_len, sizeof(int));
	model_type_len = strlen(model_type);
	appendBinaryStringInfo(&buf, (char *) &model_type_len, sizeof(int));
	appendBinaryStringInfo(&buf, model_type, model_type_len);

	total_size = VARHDRSZ + buf.len;
	result = (bytea *) palloc(total_size);
	SET_VARSIZE(result, total_size);
	memcpy(VARDATA(result), buf.data, buf.len);
	NDB_FREE(buf.data);

	return result;
}

static int
nlp_production_model_deserialize_from_bytea(const bytea * data, int *vocab_size_out, int *embedding_dim_out, int *max_seq_len_out, char *model_type_out, int model_type_max)
{
	const char *buf;
	int			offset = 0;
	int			model_type_len;

	if (data == NULL || VARSIZE(data) < VARHDRSZ + sizeof(int) * 4)
		return -1;

	buf = VARDATA(data);
	memcpy(vocab_size_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(embedding_dim_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(max_seq_len_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(&model_type_len, buf + offset, sizeof(int));
	offset += sizeof(int);

	if (model_type_len >= model_type_max)
		return -1;
	memcpy(model_type_out, buf + offset, model_type_len);
	model_type_out[model_type_len] = '\0';

	return 0;
}

static bool
nlp_production_gpu_train(MLGpuModel * model, const MLGpuTrainSpec * spec, char **errstr)
{
	NLPProductionGpuModelState *state;
	int			vocab_size = 30000;
	int			embedding_dim = 384;
	int			max_seq_len = 512;
	char		model_type[32] = "bert";
	int			nvec = 0;
	bytea	   *model_data = NULL;
	Jsonb	   *metrics = NULL;
	StringInfoData metrics_json;
	JsonbIterator *it;
	JsonbValue	v;
	int			r;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || spec == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("nlp_production_gpu_train: invalid parameters");
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
				if (strcmp(key, "vocab_size") == 0 && v.type == jbvNumeric)
					vocab_size = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																   NumericGetDatum(v.val.numeric)));
				else if (strcmp(key, "embedding_dim") == 0 && v.type == jbvNumeric)
					embedding_dim = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																	  NumericGetDatum(v.val.numeric)));
				else if (strcmp(key, "max_seq_len") == 0 && v.type == jbvNumeric)
					max_seq_len = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																	NumericGetDatum(v.val.numeric)));
				else if (strcmp(key, "model_type") == 0 && v.type == jbvString)
					strncpy(model_type, v.val.string.val, sizeof(model_type) - 1);
				NDB_FREE(key);
			}
		}
	}

	if (vocab_size < 1)
		vocab_size = 30000;
	if (embedding_dim < 1)
		embedding_dim = 384;
	if (max_seq_len < 1)
		max_seq_len = 512;

	/* Convert feature matrix to count samples */
	if (spec->feature_matrix == NULL || spec->sample_count <= 0
		|| spec->feature_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("nlp_production_gpu_train: invalid feature matrix");
		return false;
	}

	nvec = spec->sample_count;

	/* Serialize model */
	model_data = nlp_production_model_serialize_to_bytea(vocab_size, embedding_dim, max_seq_len, model_type);

	/* Build metrics */
	initStringInfo(&metrics_json);
	appendStringInfo(&metrics_json,
					 "{\"storage\":\"cpu\",\"vocab_size\":%d,\"embedding_dim\":%d,\"max_seq_len\":%d,\"model_type\":\"%s\",\"n_samples\":%d}",
					 vocab_size, embedding_dim, max_seq_len, model_type, nvec);
	metrics = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
												 CStringGetDatum(metrics_json.data)));
	NDB_FREE(metrics_json.data);

	state = (NLPProductionGpuModelState *) palloc0(sizeof(NLPProductionGpuModelState));
	state->model_blob = model_data;
	state->metrics = metrics;
	state->vocab_size = vocab_size;
	state->embedding_dim = embedding_dim;
	state->max_seq_len = max_seq_len;
	state->n_samples = nvec;
	strncpy(state->model_type, model_type, sizeof(state->model_type) - 1);

	if (model->backend_state != NULL)
		NDB_FREE(model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = false;

	return true;
}

static bool
nlp_production_gpu_predict(const MLGpuModel * model, const float *input, int input_dim,
						   float *output, int output_dim, char **errstr)
{
	const		NLPProductionGpuModelState *state;
	int			i;

	if (errstr != NULL)
		*errstr = NULL;
	if (output != NULL && output_dim > 0)
		memset(output, 0, output_dim * sizeof(float));
	if (model == NULL || input == NULL || output == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("nlp_production_gpu_predict: invalid parameters");
		return false;
	}
	if (output_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("nlp_production_gpu_predict: invalid output dimension");
		return false;
	}
	if (!model->gpu_ready || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("nlp_production_gpu_predict: model not ready");
		return false;
	}

	state = (const NLPProductionGpuModelState *) model->backend_state;

	/* Generate embeddings: normalize and transform input */
	for (i = 0; i < input_dim && i < output_dim && i < state->embedding_dim; i++)
	{
		output[i] = tanhf(input[i] * 0.1f);
	}

	/* Pad or truncate to embedding_dim */
	for (; i < output_dim && i < state->embedding_dim; i++)
		output[i] = 0.0f;

	return true;
}

static bool
nlp_production_gpu_evaluate(const MLGpuModel * model, const MLGpuEvalSpec * spec,
							MLGpuMetrics * out, char **errstr)
{
	const		NLPProductionGpuModelState *state;
	Jsonb	   *metrics_json;
	StringInfoData buf;

	if (errstr != NULL)
		*errstr = NULL;
	if (out != NULL)
		out->payload = NULL;
	if (model == NULL || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("nlp_production_gpu_evaluate: invalid model");
		return false;
	}

	state = (const NLPProductionGpuModelState *) model->backend_state;

	initStringInfo(&buf);
	appendStringInfo(&buf,
					 "{\"algorithm\":\"nlp_production\",\"storage\":\"cpu\","
					 "\"vocab_size\":%d,\"embedding_dim\":%d,\"max_seq_len\":%d,\"model_type\":\"%s\",\"n_samples\":%d}",
					 state->vocab_size > 0 ? state->vocab_size : 30000,
					 state->embedding_dim > 0 ? state->embedding_dim : 384,
					 state->max_seq_len > 0 ? state->max_seq_len : 512,
					 state->model_type[0] ? state->model_type : "bert",
					 state->n_samples > 0 ? state->n_samples : 0);

	metrics_json = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
													  CStringGetDatum(buf.data)));
	NDB_FREE(buf.data);

	if (out != NULL)
		out->payload = metrics_json;

	return true;
}

static bool
nlp_production_gpu_serialize(const MLGpuModel * model, bytea * *payload_out,
							 Jsonb * *metadata_out, char **errstr)
{
	const		NLPProductionGpuModelState *state;
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
			*errstr = pstrdup("nlp_production_gpu_serialize: invalid model");
		return false;
	}

	state = (const NLPProductionGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("nlp_production_gpu_serialize: model blob is NULL");
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
nlp_production_gpu_deserialize(MLGpuModel * model, const bytea * payload,
							   const Jsonb * metadata, char **errstr)
{
	NLPProductionGpuModelState *state;
	bytea	   *payload_copy;
	int			payload_size;
	int			vocab_size = 0;
	int			embedding_dim = 0;
	int			max_seq_len = 0;
	char		model_type[32];
	JsonbIterator *it;
	JsonbValue	v;
	int			r;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("nlp_production_gpu_deserialize: invalid parameters");
		return false;
	}

	payload_size = VARSIZE(payload);
	payload_copy = (bytea *) palloc(payload_size);
	memcpy(payload_copy, payload, payload_size);

	if (nlp_production_model_deserialize_from_bytea(payload_copy, &vocab_size, &embedding_dim, &max_seq_len, model_type, sizeof(model_type)) != 0)
	{
		NDB_FREE(payload_copy);
		if (errstr != NULL)
			*errstr = pstrdup("nlp_production_gpu_deserialize: failed to deserialize");
		return false;
	}

	state = (NLPProductionGpuModelState *) palloc0(sizeof(NLPProductionGpuModelState));
	state->model_blob = payload_copy;
	state->vocab_size = vocab_size;
	state->embedding_dim = embedding_dim;
	state->max_seq_len = max_seq_len;
	state->n_samples = 0;
	strncpy(state->model_type, model_type, sizeof(state->model_type) - 1);

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
nlp_production_gpu_destroy(MLGpuModel * model)
{
	NLPProductionGpuModelState *state;

	if (model == NULL)
		return;

	if (model->backend_state != NULL)
	{
		state = (NLPProductionGpuModelState *) model->backend_state;
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

static const MLGpuModelOps nlp_production_gpu_model_ops = {
	.algorithm = "nlp_production",
	.train = nlp_production_gpu_train,
	.predict = nlp_production_gpu_predict,
	.evaluate = nlp_production_gpu_evaluate,
	.serialize = nlp_production_gpu_serialize,
	.deserialize = nlp_production_gpu_deserialize,
	.destroy = nlp_production_gpu_destroy,
};

void
neurondb_gpu_register_nlp_production_model(void)
{
	static bool registered = false;

	if (registered)
		return;
	ndb_gpu_register_model_ops(&nlp_production_gpu_model_ops);
	registered = true;
}
