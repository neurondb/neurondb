/*-------------------------------------------------------------------------
 *
 * neurondb_hf.c
 *	  HuggingFace model SQL interface via ONNX Runtime
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 * SPDX-License-Identifier: PostgreSQL
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include "fmgr.h"
#include "funcapi.h"
#include "lib/stringinfo.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"

#include <math.h>

#include "neurondb_onnx.h"
#include "neurondb.h"
#include "neurondb_llm.h"
#include "utils/memutils.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_json.h"

extern char *neurondb_onnx_model_path;
extern bool neurondb_onnx_use_gpu;
extern int	neurondb_onnx_threads;
extern int	neurondb_onnx_cache_size;

/* Generation parameters structure */
typedef struct
{
	float		temperature;
	float		top_p;
	int			top_k;
	int			max_tokens;
	int			min_tokens;
	float		repetition_penalty;
	bool		do_sample;
	bool		return_prompt;
	int			seed;
}			ONNXGenParams;

/* Forward declarations */
/* parse_gen_params is now replaced by ndb_json_parse_gen_params from neurondb_json.h */
static int	sample_token_greedy(float *logits, int vocab_size);
static int
			sample_token_multinomial(float *logits, int vocab_size, float temperature);
static char *decode_tokens(int32_t * token_ids, int num_tokens);

PG_FUNCTION_INFO_V1(neurondb_hf_embedding);
PG_FUNCTION_INFO_V1(neurondb_hf_classify);
PG_FUNCTION_INFO_V1(neurondb_hf_ner);
PG_FUNCTION_INFO_V1(neurondb_hf_qa);
PG_FUNCTION_INFO_V1(neurondb_onnx_info);

Datum
neurondb_hf_embedding(PG_FUNCTION_ARGS)
{
	text	   *model_name_text;
	text	   *input_text;
	char	   *model_name;
	char	   *txt;
	ONNXModelSession *session;
	int32	   *token_ids;
	int32		token_length;
	ONNXTensor *input_tensor;
	ONNXTensor *output_tensor;
	Vector	   *result;
	int32		embedding_dim;
	int			i;
	int			j;
	int64		input_shape[2];
	float	   *input_data;
	float		sum;

	if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
		PG_RETURN_NULL();

	model_name_text = PG_GETARG_TEXT_P(0);
	input_text = PG_GETARG_TEXT_P(1);

	model_name = text_to_cstring(model_name_text);
	txt = text_to_cstring(input_text);

	/* Load or get cached model */
	session = neurondb_onnx_get_or_load_model(
											  model_name, ONNX_MODEL_EMBEDDING);

	/* Tokenize input text with model-specific tokenizer */
	token_ids = neurondb_tokenize_with_model(
											 txt, 128, &token_length, model_name);

	/* Convert token IDs to float array for ONNX */
	NDB_ALLOC(input_data, float, token_length);
	for (i = 0; i < token_length; i++)
		input_data[i] = (float) token_ids[i];

	/* Create input tensor */
	input_shape[0] = 1;
	input_shape[1] = token_length;

	input_tensor = neurondb_onnx_create_tensor(input_data, input_shape, 2);

	/* Run inference */
	PG_TRY();
	{
		output_tensor =
			neurondb_onnx_run_inference(session, input_tensor);
	}
	PG_CATCH();
	{
		neurondb_onnx_free_tensor(input_tensor);
		NDB_FREE(input_data);
		NDB_FREE(token_ids);
		NDB_FREE(model_name);
		NDB_FREE(txt);
		PG_RE_THROW();
	}
	PG_END_TRY();

	/* Calculate embedding dimension */
	embedding_dim = output_tensor->size / token_length;

	/* Pool embeddings (mean pooling across sequence dimension) */
	result = (Vector *) palloc0(VECTOR_SIZE(embedding_dim));
	SET_VARSIZE(result, VECTOR_SIZE(embedding_dim));
	result->dim = embedding_dim;

	for (i = 0; i < embedding_dim; i++)
	{
		sum = 0.0f;
		for (j = 0; j < token_length; j++)
			sum += output_tensor->data[j * embedding_dim + i];

		result->data[i] = sum / token_length;
	}

	neurondb_onnx_free_tensor(input_tensor);
	neurondb_onnx_free_tensor(output_tensor);
	NDB_FREE(input_data);
	NDB_FREE(token_ids);
	NDB_FREE(model_name);
	NDB_FREE(txt);

	PG_RETURN_POINTER(result);
}

Datum
neurondb_hf_classify(PG_FUNCTION_ARGS)
{
	text	   *model_name_text;
	text	   *input_text;
	char	   *model_name;
	char	   *txt;
	ONNXModelSession *session;
	int32	   *token_ids;
	int32		token_length;
	ONNXTensor *input_tensor;
	ONNXTensor *output_tensor;
	StringInfoData buf;
	int64		input_shape[2];
	float	   *input_data;
	float		positive_score;
	float		negative_score;
	int			i;
	char	   *predicted_label;

	if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
		PG_RETURN_NULL();

	model_name_text = PG_GETARG_TEXT_P(0);
	input_text = PG_GETARG_TEXT_P(1);

	model_name = text_to_cstring(model_name_text);
	txt = text_to_cstring(input_text);

	/* Load or get cached model */
	session = neurondb_onnx_get_or_load_model(
											  model_name, ONNX_MODEL_CLASSIFICATION);

	/* Tokenize input text with model-specific tokenizer */
	token_ids = neurondb_tokenize_with_model(
											 txt, 128, &token_length, model_name);

	/* Convert token IDs to float array for ONNX */
	NDB_ALLOC(input_data, float, token_length);
	for (i = 0; i < token_length; i++)
		input_data[i] = (float) token_ids[i];

	/* Create input tensor */
	input_shape[0] = 1;
	input_shape[1] = token_length;

	input_tensor = neurondb_onnx_create_tensor(input_data, input_shape, 2);

	/* Run inference */
	PG_TRY();
	{
		output_tensor =
			neurondb_onnx_run_inference(session, input_tensor);
	}
	PG_CATCH();
	{
		neurondb_onnx_free_tensor(input_tensor);
		NDB_FREE(input_data);
		NDB_FREE(token_ids);
		NDB_FREE(model_name);
		NDB_FREE(txt);
		PG_RE_THROW();
	}
	PG_END_TRY();

	/* Extract logits (assume binary classification: [negative, positive]) */
	if (output_tensor->size >= 2)
	{
		float		max_score;
		float		sum;

		negative_score = output_tensor->data[0];
		positive_score = output_tensor->data[1];

		/* Apply softmax */
		max_score = (positive_score > negative_score) ? positive_score
			: negative_score;
		negative_score = exp(negative_score - max_score);
		positive_score = exp(positive_score - max_score);
		sum = negative_score + positive_score;
		negative_score /= sum;
		positive_score /= sum;

		/* Determine predicted label */
		predicted_label = (positive_score > negative_score)
			? "POSITIVE"
			: "NEGATIVE";

		/* Build JSON result */
		initStringInfo(&buf);
		appendStringInfo(&buf,
						 "{\"label\": \"%s\", \"score\": %.4f}",
						 predicted_label,
						 (positive_score > negative_score) ? positive_score
						 : negative_score);
	}
	else
	{
		/* Fallback for unexpected output */
		initStringInfo(&buf);
		appendStringInfo(
						 &buf, "{\"label\": \"UNKNOWN\", \"score\": 0.0}");
	}

	neurondb_onnx_free_tensor(input_tensor);
	neurondb_onnx_free_tensor(output_tensor);
	NDB_FREE(input_data);
	NDB_FREE(token_ids);
	NDB_FREE(model_name);
	NDB_FREE(txt);

	PG_RETURN_TEXT_P(cstring_to_text(buf.data));
}

Datum
neurondb_hf_ner(PG_FUNCTION_ARGS)
{
	text	   *model_name_text;
	text	   *input_text;
	char	   *model_name;
	char	   *txt;
	ONNXModelSession *session;
	int32	   *token_ids;
	int32		token_length;
	ONNXTensor *input_tensor;
	ONNXTensor *output_tensor;
	StringInfoData buf;
	int64		input_shape[2];
	float	   *input_data;
	int			i;
	int			max_class;
	float		max_score;
	const char *entity_labels[] = {
		"O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"
	};
	const int	num_labels = 7;

	if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
		PG_RETURN_NULL();

	model_name_text = PG_GETARG_TEXT_P(0);
	input_text = PG_GETARG_TEXT_P(1);

	model_name = text_to_cstring(model_name_text);
	txt = text_to_cstring(input_text);

	/* Load or get cached model */
	session = neurondb_onnx_get_or_load_model(model_name, ONNX_MODEL_NER);

	/* Tokenize input text with model-specific tokenizer */
	token_ids = neurondb_tokenize_with_model(
											 txt, 128, &token_length, model_name);

	/* Convert token IDs to float array for ONNX */
	NDB_ALLOC(input_data, float, token_length);
	for (i = 0; i < token_length; i++)
		input_data[i] = (float) token_ids[i];

	/* Create input tensor */
	input_shape[0] = 1;
	input_shape[1] = token_length;

	input_tensor = neurondb_onnx_create_tensor(input_data, input_shape, 2);

	/* Run inference */
	PG_TRY();
	{
		output_tensor =
			neurondb_onnx_run_inference(session, input_tensor);
	}
	PG_CATCH();
	{
		neurondb_onnx_free_tensor(input_tensor);
		NDB_FREE(input_data);
		NDB_FREE(token_ids);
		NDB_FREE(model_name);
		NDB_FREE(txt);
		PG_RE_THROW();
	}
	PG_END_TRY();

	/* Build JSON result with entities */
	initStringInfo(&buf);
	appendStringInfo(&buf, "{\"entities\": [");

	/* Extract entities (simplified - just first token for demo) */
	if (output_tensor->size >= (size_t) num_labels)
	{
		max_class = 0;
		max_score = output_tensor->data[0];

		for (i = 1; i < num_labels && (size_t) i < output_tensor->size;
			 i++)
		{
			if (output_tensor->data[i] > max_score)
			{
				max_score = output_tensor->data[i];
				max_class = i;
			}
		}

		if (max_class > 0)		/* Not 'O' (outside) */
		{
			appendStringInfo(&buf,
							 "{\"entity\": \"%s\", \"label\": \"%s\", "
							 "\"score\": %.4f}",
							 txt,
							 entity_labels[max_class],
							 max_score);
		}
	}

	appendStringInfo(&buf, "]}");

	neurondb_onnx_free_tensor(input_tensor);
	neurondb_onnx_free_tensor(output_tensor);
	NDB_FREE(input_data);
	NDB_FREE(token_ids);
	NDB_FREE(model_name);
	NDB_FREE(txt);

	PG_RETURN_TEXT_P(cstring_to_text(buf.data));
}

Datum
neurondb_hf_qa(PG_FUNCTION_ARGS)
{
	text	   *model_name_text;
	text	   *question_text;
	text	   *context_text;
	char	   *model_name;
	char	   *question;
	char	   *context;
	ONNXModelSession *session;
	int32	   *token_ids;
	int32		token_length;
	ONNXTensor *input_tensor;
	ONNXTensor *output_tensor;
	StringInfoData buf;
	StringInfoData combined_text;
	int64		input_shape[2];
	float	   *input_data;
	volatile int start_idx = 0;
	volatile int end_idx = 0;
	volatile float start_score = -1000.0f;
	volatile float end_score = -1000.0f;
	int			i;

	if (PG_ARGISNULL(0) || PG_ARGISNULL(1) || PG_ARGISNULL(2))
		PG_RETURN_NULL();

	model_name_text = PG_GETARG_TEXT_P(0);
	question_text = PG_GETARG_TEXT_P(1);
	context_text = PG_GETARG_TEXT_P(2);

	model_name = text_to_cstring(model_name_text);
	question = text_to_cstring(question_text);
	context = text_to_cstring(context_text);

	/* Combine question and context for QA models */
	initStringInfo(&combined_text);
	appendStringInfo(&combined_text, "%s [SEP] %s", question, context);

	/* Load or get cached model */
	session = neurondb_onnx_get_or_load_model(model_name, ONNX_MODEL_QA);

	/* Tokenize input with model-specific tokenizer */
	token_ids = neurondb_tokenize_with_model(
											 combined_text.data, 256, &token_length, model_name);

	/* Convert token IDs to float array for ONNX */
	NDB_ALLOC(input_data, float, token_length);
	for (i = 0; i < token_length; i++)
		input_data[i] = (float) token_ids[i];

	/* Create input tensor */
	input_shape[0] = 1;
	input_shape[1] = token_length;

	input_tensor = neurondb_onnx_create_tensor(input_data, input_shape, 2);

	/* Run inference */
	PG_TRY();
	{
		output_tensor =
			neurondb_onnx_run_inference(session, input_tensor);
	}
	PG_CATCH();
	{
		neurondb_onnx_free_tensor(input_tensor);
		NDB_FREE(input_data);
		NDB_FREE(token_ids);
		NDB_FREE(model_name);
		NDB_FREE(question);
		NDB_FREE(context);
		PG_RE_THROW();
	}
	PG_END_TRY();

	/* Extract start and end positions (QA model outputs start/end logits) */
	/* Simplified: find max start and end positions */
	if (output_tensor->size >= (size_t) token_length * 2)
	{
		/* Find start position */
		for (i = 0; i < token_length; i++)
		{
			if (output_tensor->data[i] > start_score)
			{
				start_score = output_tensor->data[i];
				start_idx = i;
			}
		}

		/* Find end position */
		for (i = token_length;
			 i < token_length * 2 && (size_t) i < output_tensor->size;
			 i++)
		{
			if (output_tensor->data[i] > end_score)
			{
				end_score = output_tensor->data[i];
				end_idx = i - token_length;
			}
		}
	}

	/* Build JSON result */
	initStringInfo(&buf);
	appendStringInfo(&buf,
					 "{\"answer\": \"Answer extracted from position %d to %d\", "
					 "\"score\": %.4f, \"start\": %d, \"end\": %d}",
					 start_idx,
					 end_idx,
					 start_score,
					 start_idx,
					 end_idx);

	neurondb_onnx_free_tensor(input_tensor);
	neurondb_onnx_free_tensor(output_tensor);
	NDB_FREE(input_data);
	NDB_FREE(token_ids);
	NDB_FREE(model_name);
	NDB_FREE(question);
	NDB_FREE(context);

	PG_RETURN_TEXT_P(cstring_to_text(buf.data));
}

Datum
neurondb_onnx_info(PG_FUNCTION_ARGS)
{
	StringInfoData buf;

	initStringInfo(&buf);
	appendStringInfo(&buf,
					 "{\"available\": %s, \"version\": \"%s\"}",
					 neurondb_onnx_available() ? "true" : "false",
					 neurondb_onnx_version());

	PG_RETURN_TEXT_P(cstring_to_text(buf.data));
}

#ifdef HAVE_ONNX_RUNTIME

/*
 * parse_gen_params
 *	  Parse JSON parameters for text generation
 *
 * Expected JSON format:
 * {
 *   "temperature": 0.7,
 *   "top_p": 0.9,
 *   "top_k": 50,
 *   "max_tokens": 100,
 *   "min_tokens": 10,
 *   "repetition_penalty": 1.1,
 *   "do_sample": true,
 *   "return_prompt": false,
 *   "seed": 42
 * }
 */
/* parse_gen_params is now replaced by ndb_json_parse_gen_params from neurondb_json.h */

/*
 * sample_token_greedy
 *	  Sample token using greedy selection (argmax)
 */
static int
sample_token_greedy(float *logits, int vocab_size)
{
	int			max_idx = 0;
	float		max_val = logits[0];
	int			i;

	for (i = 1; i < vocab_size; i++)
	{
		if (logits[i] > max_val)
		{
			max_val = logits[i];
			max_idx = i;
		}
	}

	return max_idx;
}

/*
 * sample_token_multinomial
 *	  Sample token using multinomial sampling with temperature
 */
static int
sample_token_multinomial(float *logits, int vocab_size, float temperature)
{
	float	   *probs = NULL;
	float		sum = 0.0f;
	float		cumsum = 0.0f;
	float		r;
	int			i;
	int			selected = 0;

	/* Allocate probabilities array */
	NDB_ALLOC(probs, float, vocab_size);

	/* Compute softmax with temperature */
	for (i = 0; i < vocab_size; i++)
	{
		probs[i] = expf(logits[i] / temperature);
		sum += probs[i];
	}

	/* Normalize */
	if (sum > 0.0f)
	{
		for (i = 0; i < vocab_size; i++)
			probs[i] /= sum;
	}

	/* Sample from multinomial distribution */
	r = (float) rand() / (float) RAND_MAX;
	for (i = 0; i < vocab_size; i++)
	{
		cumsum += probs[i];
		if (r <= cumsum)
		{
			selected = i;
			break;
		}
	}

	NDB_FREE(probs);
	return selected;
}

/*
 * decode_tokens
 *	  Decode token IDs to text (simplified: just convert to string representation)
 */
static char *
decode_tokens(int32_t * token_ids, int num_tokens)
{
	StringInfoData buf;
	int			i;

	initStringInfo(&buf);

	for (i = 0; i < num_tokens; i++)
	{
		if (i > 0)
			appendStringInfoChar(&buf, ' ');
		appendStringInfo(&buf, "%d", token_ids[i]);
	}

	return buf.data;
}

/*
 * ndb_onnx_hf_complete
 *	  Generate text completion using ONNX Runtime
 *
 * This implements a simplified text generation pipeline:
 * 1. Parse generation parameters from JSON
 * 2. Load or find model in cache
 * 3. Tokenize input prompt
 * 4. Run inference in a loop (autoregressive generation):
 *    a. Run ONNX inference to get logits
 *    b. Apply temperature, top-k, top-p, repetition penalty
 *    c. Sample next token (greedy or multinomial)
 *    d. Append to input sequence
 *    e. Check for stop conditions
 * 5. Decode generated token IDs to text
 * 6. Return generated text
 */
int
ndb_onnx_hf_complete(const char *model_name,
					 const char *prompt,
					 const char *params_json,
					 char **text_out,
					 char **errstr)
{
	ONNXModelSession *session = NULL;
	ONNXGenParams gen_params;
	int32_t    *input_token_ids = NULL;
	int32		input_token_length = 0;
	int32	   *generated_token_ids = NULL;
	int32		generated_token_count = 0;
	int32		max_gen_tokens = 100;
	int32		min_gen_tokens = 0;
	int32		i;
	int			rc = -1;
	ONNXTensor *input_tensor = NULL;
	ONNXTensor *output_tensor = NULL;
	int64		input_shape[2];
	float	   *input_data = NULL;
	float	   *logits = NULL;
	int32		vocab_size = 0;
	int32		next_token_id = 0;
	char	   *generated_text = NULL;
	MemoryContext oldcontext;
	MemoryContext complete_context;

	if (errstr)
		*errstr = NULL;
	if (model_name == NULL || prompt == NULL || text_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup(
							  "invalid parameters for ONNX HF complete");
		return -1;
	}

	/* Create memory context for this operation */
	complete_context = AllocSetContextCreate(CurrentMemoryContext,
											 "onnx_hf_complete_ctx",
											 ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(complete_context);

	/* Parse generation parameters */
	if (params_json && strlen(params_json) > 0)
	{
		NdbGenParams ndb_params = {0};

		rc = ndb_json_parse_gen_params(params_json, &ndb_params, errstr);
		if (rc != 0)
		{
			/* Free any allocated resources */
			ndb_json_parse_gen_params_free(&ndb_params);
			MemoryContextSwitchTo(oldcontext);
			MemoryContextDelete(complete_context);
			return -1;
		}
		/* Map from NdbGenParams to ONNXGenParams */
		gen_params.temperature = ndb_params.temperature;
		gen_params.top_p = ndb_params.top_p;
		gen_params.top_k = ndb_params.top_k;
		gen_params.max_tokens = ndb_params.max_tokens;
		gen_params.min_tokens = ndb_params.min_tokens;
		gen_params.repetition_penalty = ndb_params.repetition_penalty;
		gen_params.do_sample = ndb_params.do_sample;
		gen_params.return_prompt = ndb_params.return_prompt;
		gen_params.seed = ndb_params.seed;
		/* Free allocated resources from NdbGenParams */
		ndb_json_parse_gen_params_free(&ndb_params);
	}
	else
	{
		/* Use defaults */
		memset(&gen_params, 0, sizeof(ONNXGenParams));
		gen_params.temperature = 1.0f;
		gen_params.top_p = 1.0f;
		gen_params.top_k = 0;
		gen_params.max_tokens = 100;
		gen_params.min_tokens = 0;
		gen_params.repetition_penalty = 1.0f;
		gen_params.do_sample = false;
		gen_params.return_prompt = false;
		gen_params.seed = 0;
	}

	max_gen_tokens = gen_params.max_tokens;
	min_gen_tokens = gen_params.min_tokens;

	/* Load or get cached model */
	PG_TRY();
	{
		session = neurondb_onnx_get_or_load_model(
												  model_name, ONNX_MODEL_GENERATION);
	}
	PG_CATCH();
	{
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(complete_context);
		PG_RE_THROW();
	}
	PG_END_TRY();

	/* Tokenize input prompt with model-specific tokenizer */
	input_token_ids = neurondb_tokenize_with_model(
												   prompt, 512, &input_token_length, model_name);
	if (input_token_ids == NULL || input_token_length == 0)
	{
		if (errstr)
			*errstr = pstrdup("failed to tokenize input prompt");
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(complete_context);
		return -1;
	}

	/* Allocate generated token IDs array */
	NDB_ALLOC(generated_token_ids, int32_t, max_gen_tokens);

	/* Set seed if provided */
	if (gen_params.seed != 0)
		srand(gen_params.seed);

	/* Generation loop */
	for (i = 0; i < max_gen_tokens; i++)
	{
		int32		current_seq_len;
		int32		j;

		NDB_DECLARE(float *, input_data);

		/* Prepare input sequence (prompt + generated tokens) */
		current_seq_len = input_token_length + generated_token_count;

		/* Allocate input data */
		if (input_data)
			NDB_FREE(input_data);
		NDB_ALLOC(input_data, float, current_seq_len);

		/* Copy prompt tokens */
		for (j = 0; j < input_token_length; j++)
			input_data[j] = (float) input_token_ids[j];

		/* Copy generated tokens */
		for (j = 0; j < generated_token_count; j++)
			input_data[input_token_length + j] =
				(float) generated_token_ids[j];

		/* Create input tensor */
		input_shape[0] = 1;
		input_shape[1] = current_seq_len;

		if (input_tensor)
			neurondb_onnx_free_tensor(input_tensor);
		input_tensor =
			neurondb_onnx_create_tensor(input_data, input_shape, 2);

		/* Run inference */
		PG_TRY();
		{
			if (output_tensor)
				neurondb_onnx_free_tensor(output_tensor);
			output_tensor = neurondb_onnx_run_inference(
														session, input_tensor);
		}
		PG_CATCH();
		{
			if (input_tensor)
				neurondb_onnx_free_tensor(input_tensor);
			if (input_data)
				NDB_FREE(input_data);
			if (generated_token_ids)
				NDB_FREE(generated_token_ids);
			if (input_token_ids)
				NDB_FREE(input_token_ids);
			MemoryContextSwitchTo(oldcontext);
			MemoryContextDelete(complete_context);
			PG_RE_THROW();
		}
		PG_END_TRY();

		/* Extract logits from output tensor */
		/* Assume output shape is [1, vocab_size] or [1, seq_len, vocab_size] */
		if (output_tensor->ndim == 2 && output_tensor->shape[0] == 1)
		{
			/* Output shape: [1, vocab_size] */
			vocab_size = output_tensor->shape[1];
			logits = output_tensor->data;
		}
		else if (output_tensor->ndim == 3
				 && output_tensor->shape[0] == 1)
		{
			/* Output shape: [1, seq_len, vocab_size] */
			/* Use last token's logits */
			int32		seq_len = output_tensor->shape[1];

			vocab_size = output_tensor->shape[2];
			logits = output_tensor->data
				+ (seq_len - 1) * vocab_size;
		}
		else
		{
			if (errstr)
				*errstr = psprintf(
								   "unexpected output tensor shape: [%lld",
								   (long long) output_tensor->shape[0]);
			if (input_tensor)
				neurondb_onnx_free_tensor(input_tensor);
			if (output_tensor)
				neurondb_onnx_free_tensor(output_tensor);
			if (input_data)
				NDB_FREE(input_data);
			if (generated_token_ids)
				NDB_FREE(generated_token_ids);
			if (input_token_ids)
				NDB_FREE(input_token_ids);
			MemoryContextSwitchTo(oldcontext);
			MemoryContextDelete(complete_context);
			return -1;
		}

		/* Apply temperature scaling */
		if (gen_params.temperature > 0.0f
			&& gen_params.temperature != 1.0f)
		{
			for (j = 0; j < vocab_size; j++)
				logits[j] /= gen_params.temperature;
		}

		/* Apply repetition penalty */
		if (gen_params.repetition_penalty != 1.0f)
		{
			for (j = 0; j < generated_token_count; j++)
			{
				int32		prev_token = generated_token_ids[j];

				if (prev_token >= 0 && prev_token < vocab_size)
				{
					if (logits[prev_token] > 0.0f)
						logits[prev_token] /=
							gen_params
							.repetition_penalty;
					else
						logits[prev_token] *=
							gen_params
							.repetition_penalty;
				}
			}
		}

		/* Sample next token */
		if (gen_params.do_sample && gen_params.temperature > 0.0f)
			next_token_id = sample_token_multinomial(
													 logits, vocab_size, gen_params.temperature);
		else
			next_token_id = sample_token_greedy(logits, vocab_size);

		/* Check for stop conditions (simplified: check for EOS token) */
		if (next_token_id == 102
			|| next_token_id == 0)	/* [SEP] or [PAD] */
		{
			if (generated_token_count >= min_gen_tokens)
				break;
		}

		/* Append to generated tokens */
		generated_token_ids[generated_token_count++] = next_token_id;
	}

	/* Decode generated tokens to text */
	generated_text =
		decode_tokens(generated_token_ids, generated_token_count);

	/* Copy to parent memory context */
	*text_out = MemoryContextStrdup(oldcontext, generated_text);

	if (input_tensor)
		neurondb_onnx_free_tensor(input_tensor);
	if (output_tensor)
		neurondb_onnx_free_tensor(output_tensor);
	if (input_data)
		NDB_FREE(input_data);
	if (generated_token_ids)
		NDB_FREE(generated_token_ids);
	if (input_token_ids)
		NDB_FREE(input_token_ids);

	MemoryContextSwitchTo(oldcontext);
	MemoryContextDelete(complete_context);

	return 0;
}

/*
 * ndb_onnx_hf_embed
 *	  Generate text embedding using ONNX model (C-callable)
 *
 * Returns 0 on success, -1 on error. vec_out and dim_out must be provided.
 */
PGDLLEXPORT int
ndb_onnx_hf_embed(const char *model_name,
				  const char *text,
				  float **vec_out,
				  int *dim_out,
				  char **errstr)
{
	ONNXModelSession *session = NULL;
	int32	   *token_ids = NULL;
	int32		token_length = 0;
	ONNXTensor *input_tensor = NULL;
	ONNXTensor *output_tensor = NULL;
	float	   *input_data = NULL;
	float	   *result_vec = NULL;
	int64		input_shape[2];
	int			embedding_dim = 0;
	int			i;
	int			j;
	float		sum;
	MemoryContext oldcontext;
	MemoryContext workcontext;

	/* Defensive parameter validation */
	if (model_name == NULL || strlen(model_name) == 0
		|| text == NULL || strlen(text) == 0
		|| vec_out == NULL || dim_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("Invalid parameters for text embedding");
		return -1;
	}

	/* Initialize outputs */
	*vec_out = NULL;
	*dim_out = 0;
	if (errstr)
		*errstr = NULL;

	/* Check ONNX availability */
	if (!neurondb_onnx_available())
	{
		if (errstr)
			*errstr = pstrdup("ONNX Runtime not available");
		return -1;
	}

	PG_TRY();
	{
		/* Create a work memory context for temporary allocations */
		workcontext = AllocSetContextCreate(CurrentMemoryContext,
											"ndb_onnx_hf_embed",
											ALLOCSET_DEFAULT_SIZES);
		oldcontext = MemoryContextSwitchTo(workcontext);

		/* Load or get cached model */
		session = neurondb_onnx_get_or_load_model(
												  model_name, ONNX_MODEL_EMBEDDING);

		if (session == NULL)
		{
			MemoryContextSwitchTo(oldcontext);
			MemoryContextDelete(workcontext);
			if (errstr)
				*errstr = psprintf("Failed to load ONNX model: %s", model_name);
			return -1;
		}

		/* Tokenize input text with model-specific tokenizer */
		token_ids = neurondb_tokenize_with_model(
												 text, 128, &token_length, model_name);

		NDB_DECLARE(float *, input_data);

		if (token_ids == NULL || token_length == 0)
		{
			MemoryContextSwitchTo(oldcontext);
			MemoryContextDelete(workcontext);
			if (errstr)
				*errstr = pstrdup("Failed to tokenize input text");
			return -1;
		}

		/* Convert token IDs to float array for ONNX */
		NDB_ALLOC(input_data, float, token_length);
		for (i = 0; i < token_length; i++)
			input_data[i] = (float) token_ids[i];

		/* Create input tensor */
		input_shape[0] = 1;
		input_shape[1] = token_length;

		input_tensor = neurondb_onnx_create_tensor(input_data, input_shape, 2);

		if (input_tensor == NULL)
		{
			MemoryContextSwitchTo(oldcontext);
			MemoryContextDelete(workcontext);
			if (errstr)
				*errstr = pstrdup("Failed to create input tensor");
			return -1;
		}

		/* Run inference */
		output_tensor = neurondb_onnx_run_inference(session, input_tensor);

		if (output_tensor == NULL || output_tensor->data == NULL)
		{
			if (input_tensor)
				neurondb_onnx_free_tensor(input_tensor);
			MemoryContextSwitchTo(oldcontext);
			MemoryContextDelete(workcontext);
			if (errstr)
				*errstr = pstrdup("ONNX inference failed");
			return -1;
		}

		/* Calculate embedding dimension */
		embedding_dim = output_tensor->size / token_length;

		NDB_DECLARE(float *, result_vec);

		if (embedding_dim <= 0)
		{
			neurondb_onnx_free_tensor(input_tensor);
			neurondb_onnx_free_tensor(output_tensor);
			MemoryContextSwitchTo(oldcontext);
			MemoryContextDelete(workcontext);
			if (errstr)
				*errstr = pstrdup("Invalid embedding dimension");
			return -1;
		}

		/* Allocate result vector in parent memory context */
		MemoryContextSwitchTo(oldcontext);
		NDB_ALLOC(result_vec, float, embedding_dim);

		/* Pool embeddings (mean pooling across sequence dimension) */
		for (i = 0; i < embedding_dim; i++)
		{
			sum = 0.0f;
			for (j = 0; j < token_length; j++)
				sum += output_tensor->data[j * embedding_dim + i];

			result_vec[i] = sum / token_length;
		}

		/* Copy results to output pointers */
		*vec_out = result_vec;
		*dim_out = embedding_dim;

		/* Cleanup work context */
		MemoryContextSwitchTo(workcontext);
		neurondb_onnx_free_tensor(input_tensor);
		neurondb_onnx_free_tensor(output_tensor);
		NDB_FREE(input_data);
		NDB_FREE(token_ids);
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(workcontext);
	}
	PG_CATCH();
	{
		/* Cleanup on error */
		if (result_vec)
			NDB_FREE(result_vec);
		if (input_tensor)
			neurondb_onnx_free_tensor(input_tensor);
		if (output_tensor)
			neurondb_onnx_free_tensor(output_tensor);
		if (input_data)
			NDB_FREE(input_data);
		if (token_ids)
			NDB_FREE(token_ids);
		if (workcontext)
		{
			MemoryContextSwitchTo(oldcontext);
			MemoryContextDelete(workcontext);
		}
		if (errstr)
			*errstr = pstrdup("Exception during ONNX text embedding");
		PG_RE_THROW();
	}
	PG_END_TRY();

	return 0;
}

/*
 * ndb_onnx_hf_image_embed
 *	  Generate image embedding using ONNX CLIP model
 *
 * Note: This is a placeholder implementation. Full image processing
 * requires image decoding libraries. This version works with
 * preprocessed image data or simplified preprocessing.
 */
int
ndb_onnx_hf_image_embed(const char *model_name,
						const unsigned char *image_data,
						size_t image_size,
						float **vec_out,
						int *dim_out,
						char **errstr)
{
	ONNXModelSession *session = NULL;
	ONNXTensor *input_tensor = NULL;
	ONNXTensor *output_tensor = NULL;
	float	   *input_data = NULL;
	int64		input_shape[4];
	int			rc = -1;
	int			i;
	float		sum;

	/* Defensive parameter validation */
	if (model_name == NULL || strlen(model_name) == 0
		|| image_data == NULL || image_size == 0
		|| vec_out == NULL || dim_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("Invalid parameters for image embedding");
		return -1;
	}

	/* Initialize outputs */
	*vec_out = NULL;
	*dim_out = 0;
	if (errstr)
		*errstr = NULL;

	/* Check ONNX availability */
	if (!neurondb_onnx_available())
	{
		if (errstr)
			*errstr = pstrdup("ONNX Runtime not available");
		return -1;
	}

	PG_TRY();
	{
		/* Load or get cached CLIP model */
		session = neurondb_onnx_get_or_load_model(
												  model_name, ONNX_MODEL_EMBEDDING);
		if (!session)
		{
			if (errstr)
				*errstr = pstrdup("Failed to load ONNX CLIP model");
			rc = -1;
		}
		else
		{
			/*
			 * CLIP image encoder expects preprocessed image: Shape: [1, 3,
			 * 224, 224] for RGB Values: Normalized [0, 1] or [-1, 1]
			 *
			 * Note: Full image preprocessing requires image decoding
			 * libraries (libjpeg, libpng, etc.). For now, we return an error
			 * indicating that HTTP API should be used for image embeddings,
			 * or image preprocessing needs to be implemented.
			 *
			 * Placeholder implementation - would need image libraries
			 */
			const int	img_size = 224;
			const int	channels = 3;
			const int	total_size = img_size * img_size * channels;

			NDB_DECLARE(float *, input_data);
			NDB_ALLOC(input_data, float, total_size);
			if (!input_data)
			{
				if (errstr)
					*errstr = pstrdup("Memory allocation failed");
				rc = -1;
			}
			else
			{
				/*
				 * Image preprocessing: For CLIP, we need to: 1. Decode image
				 * (JPEG/PNG) - requires image libraries 2. Resize to 224x224
				 * 3. Normalize pixel values to [0,1] or [-1,1] range 4.
				 * Convert to RGB format
				 *
				 * Since image libraries may not be available, we provide a
				 * basic implementation that expects pre-processed input. For
				 * production, integrate with libjpeg, libpng, or stb_image.
				 */

				/*
				 * Image preprocessing implementation: 1. Try to decode image
				 * if it's JPEG/PNG (basic detection) 2. Resize to 224x224 if
				 * needed 3. Normalize pixel values to [0,1] range 4. Convert
				 * to RGB format (NCHW: [1, 3, 224, 224])
				 *
				 * Note: Full JPEG/PNG decoding requires external libraries.
				 * This implementation handles raw RGB data and basic
				 * preprocessing.
				 */
				{
					/* Detect image format from magic bytes */
					bool		is_jpeg = (image_size >= 2 && image_data[0] == 0xFF && image_data[1] == 0xD8);
					bool		is_png = (image_size >= 8 && image_data[0] == 0x89 && image_data[1] == 0x50
										  && image_data[2] == 0x4E && image_data[3] == 0x47);
					bool		is_raw_rgb = false;
					int			src_width = 0;
					int			src_height = 0;
					int			src_channels = 3;
					unsigned char *decoded_data = NULL;

					if (is_jpeg || is_png)
					{
						/*
						 * Image format detected but decoding not available.
						 * For production, integrate libjpeg/libpng or
						 * stb_image.
						 */
						elog(WARNING,
							 "neurondb: JPEG/PNG image detected but decoding not available. "
							 "Please use pre-processed RGB data or integrate image decoding library.");
						/* Fall through to treat as raw RGB */
						is_raw_rgb = true;
					}
					else
					{
						/*
						 * Assume raw RGB data: [width * height * channels]
						 * bytes
						 */
						/* Try to infer dimensions from size */
						int			total_pixels = image_size / src_channels;

						if (total_pixels > 0)
						{
							/*
							 * Assume square image for simplicity, or use
							 * common aspect ratios
							 */
							src_width = (int) sqrt(total_pixels);
							src_height = total_pixels / src_width;
							if (src_width * src_height * src_channels == image_size)
							{
								is_raw_rgb = true;
								decoded_data = (unsigned char *) image_data;
							}
						}
					}

					if (is_raw_rgb && decoded_data != NULL && src_width > 0 && src_height > 0)
					{
						/* Resize and normalize raw RGB data */
						int			dst_width = img_size;
						int			dst_height = img_size;
						int			x,
									y,
									c;
						float		scale_x = (float) src_width / (float) dst_width;
						float		scale_y = (float) src_height / (float) dst_height;

						/* Resize using nearest-neighbor interpolation */
						for (y = 0; y < dst_height; y++)
						{
							for (x = 0; x < dst_width; x++)
							{
								int			src_x = (int) (x * scale_x);
								int			src_y = (int) (y * scale_y);

								if (src_x >= src_width)
									src_x = src_width - 1;
								if (src_y >= src_height)
									src_y = src_height - 1;

								for (c = 0; c < channels; c++)
								{
									/*
									 * NCHW format: [batch, channel, height,
									 * width]
									 */
									int			dst_idx = c * dst_height * dst_width + y * dst_width + x;
									int			src_idx = (src_y * src_width + src_x) * src_channels + c;
									unsigned char pixel_val = decoded_data[src_idx];

									/* Normalize to [0, 1] range */
									input_data[dst_idx] = (float) pixel_val / 255.0f;
								}
							}
						}
					}
					else
					{
						/* Fallback: zero-filled array if decoding fails */
						elog(WARNING,
							 "neurondb: Could not decode image data. Using zero-filled array. "
							 "Please provide pre-processed RGB data or integrate image decoding library.");
						for (int idx = 0; idx < total_size; idx++)
						{
							input_data[idx] = 0.0f;
						}
					}
				}

				/* CLIP input shape: [batch, channels, height, width] */
				input_shape[0] = 1;
				input_shape[1] = channels;
				input_shape[2] = img_size;
				input_shape[3] = img_size;

				input_tensor = neurondb_onnx_create_tensor(input_data, input_shape, 4);
				if (!input_tensor)
				{
					NDB_FREE(input_data);
					input_data = NULL;
					if (errstr)
						*errstr = pstrdup("Failed to create input tensor");
					rc = -1;
				}
				else
				{
					/* Run inference */
					output_tensor = neurondb_onnx_run_inference(session, input_tensor);
					if (!output_tensor || output_tensor->size <= 0)
					{
						neurondb_onnx_free_tensor(input_tensor);
						if (input_data)
							NDB_FREE(input_data);
						if (errstr)
							*errstr = pstrdup("ONNX inference failed");
						rc = -1;
					}
					else
					{
						NDB_DECLARE(float *, vec_out_local);

						/* Extract embedding */
						*dim_out = (int) output_tensor->size;
						NDB_ALLOC(vec_out_local, float, *dim_out);
						*vec_out = vec_out_local;
						if (!*vec_out)
						{
							neurondb_onnx_free_tensor(input_tensor);
							neurondb_onnx_free_tensor(output_tensor);
							if (input_data)
								NDB_FREE(input_data);
							if (errstr)
								*errstr = pstrdup("Memory allocation failed");
							*dim_out = 0;
							rc = -1;
						}
						else
						{
							memcpy(*vec_out, output_tensor->data,
								   *dim_out * sizeof(float));

							/* L2 normalization */
							sum = 0.0f;
							for (i = 0; i < *dim_out; i++)
								sum += (*vec_out)[i] * (*vec_out)[i];
							sum = sqrtf(sum);
							if (sum > 0.0f)
							{
								for (i = 0; i < *dim_out; i++)
									(*vec_out)[i] /= sum;
							}

							neurondb_onnx_free_tensor(input_tensor);
							neurondb_onnx_free_tensor(output_tensor);
							if (input_data)
								NDB_FREE(input_data);

							rc = 0;
						}
					}
				}
			}
		}
	}
	PG_CATCH();
	{
		/* Defensive cleanup on error */
		if (input_tensor)
			neurondb_onnx_free_tensor(input_tensor);
		if (output_tensor)
			neurondb_onnx_free_tensor(output_tensor);
		if (input_data)
			NDB_FREE(input_data);
		if (*vec_out)
		{
			NDB_FREE(*vec_out);
			*vec_out = NULL;
			*dim_out = 0;
		}
		if (errstr && !*errstr)
			*errstr = pstrdup("ONNX image embedding error");
		rc = -1;
	}
	PG_END_TRY();

	return rc;
}

/*
 * ndb_onnx_hf_multimodal_embed
 *	  Generate multimodal (text+image) embedding using ONNX CLIP
 */
int
ndb_onnx_hf_multimodal_embed(const char *model_name,
							 const char *text,
							 const unsigned char *image_data,
							 size_t image_size,
							 float **vec_out,
							 int *dim_out,
							 char **errstr)
{
	/* For CLIP multimodal, combine text and image encoders */
	float	   *text_vec = NULL;
	float	   *image_vec = NULL;
	int			text_dim = 0;
	int			image_dim = 0;
	int			rc = -1;
	int			i;
	float		sum;

	/* Defensive parameter validation */
	if (model_name == NULL || strlen(model_name) == 0
		|| text == NULL || strlen(text) == 0
		|| image_data == NULL || image_size == 0
		|| vec_out == NULL || dim_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("Invalid parameters for multimodal embedding");
		return -1;
	}

	/* Initialize outputs */
	*vec_out = NULL;
	*dim_out = 0;
	if (errstr)
		*errstr = NULL;

	/* Get text embedding */
	{
		char	   *text_err = NULL;

		if (ndb_onnx_hf_embed(model_name, text, &text_vec, &text_dim, &text_err) != 0)
		{
			if (errstr)
				*errstr = psprintf("Text embedding failed: %s",
								   text_err ? text_err : "unknown error");
			if (text_err)
				NDB_FREE(text_err);
			return -1;
		}
		if (text_err)
			NDB_FREE(text_err);
	}

	/* Get image embedding */
	{
		char	   *img_err = NULL;

		if (ndb_onnx_hf_image_embed(model_name, image_data, image_size,
									&image_vec, &image_dim, &img_err) != 0)
		{
			if (text_vec)
				NDB_FREE(text_vec);
			if (errstr)
				*errstr = psprintf("Image embedding failed: %s",
								   img_err ? img_err : "unknown error");
			if (img_err)
				NDB_FREE(img_err);
			return -1;
		}
		if (img_err)
			NDB_FREE(img_err);
	}

	/* Combine embeddings (CLIP uses shared space) */
	if (text_vec != NULL && image_vec != NULL && text_dim == image_dim && text_dim > 0)
	{
		NDB_DECLARE(float *, vec_out_local);

		/* Average pooling for fusion */
		*dim_out = text_dim;
		NDB_ALLOC(vec_out_local, float, *dim_out);
		*vec_out = vec_out_local;
		if (!*vec_out)
		{
			if (text_vec)
				NDB_FREE(text_vec);
			if (image_vec)
				NDB_FREE(image_vec);
			if (errstr)
				*errstr = pstrdup("Memory allocation failed");
			*dim_out = 0;
			return -1;
		}

		for (i = 0; i < *dim_out; i++)
			(*vec_out)[i] = (text_vec[i] + image_vec[i]) / 2.0f;

		/* L2 normalization */
		sum = 0.0f;
		for (i = 0; i < *dim_out; i++)
			sum += (*vec_out)[i] * (*vec_out)[i];
		sum = sqrtf(sum);
		if (sum > 0.0f)
		{
			for (i = 0; i < *dim_out; i++)
				(*vec_out)[i] /= sum;
		}

		if (text_vec)
			NDB_FREE(text_vec);
		if (image_vec)
			NDB_FREE(image_vec);

		rc = 0;
	}
	else
	{
		if (text_vec)
			NDB_FREE(text_vec);
		if (image_vec)
			NDB_FREE(image_vec);
		if (errstr)
			*errstr = pstrdup("Failed to combine embeddings (dimension mismatch)");
		rc = -1;
	}

	return rc;
}

/*
 * ndb_onnx_hf_rerank
 *	  Rerank documents using ONNX cross-encoder model
 *
 * Cross-encoder models take query+document pairs and output relevance scores.
 * This function processes each query-document pair through the model.
 */
int
ndb_onnx_hf_rerank(const char *model_name,
				   const char *query,
				   const char **docs,
				   int ndocs,
				   float **scores_out,
				   char **errstr)
{
	ONNXModelSession *session = NULL;
	int32	   *query_token_ids = NULL;
	int32		query_token_length = 0;
	int32	   *doc_token_ids = NULL;
	int32		doc_token_length = 0;
	int32	   *combined_token_ids = NULL;
	int32		combined_token_length = 0;
	ONNXTensor *input_tensor = NULL;
	ONNXTensor *output_tensor = NULL;
	float	   *input_data = NULL;
	int64		input_shape[2];
	float	   *scores = NULL;
	int			rc = -1;
	int			i;
	int			j;
	int			max_combined_length = 512;	/* Typical max for cross-encoders */

	/* Defensive parameter validation */
	if (model_name == NULL || strlen(model_name) == 0
		|| query == NULL || strlen(query) == 0
		|| docs == NULL || ndocs <= 0
		|| scores_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("Invalid parameters for ONNX reranking");
		return -1;
	}

	/* Initialize outputs */
	*scores_out = NULL;
	if (errstr)
		*errstr = NULL;

	/* Check ONNX availability */
	if (!neurondb_onnx_available())
	{
		if (errstr)
			*errstr = pstrdup("ONNX Runtime not available");
		return -1;
	}

	/* Allocate scores array */
	NDB_ALLOC(scores, float, ndocs);
	if (!scores)
	{
		if (errstr)
			*errstr = pstrdup("Memory allocation failed");
		return -1;
	}

	PG_TRY();
	{
		/* Load or get cached cross-encoder model */
		session = neurondb_onnx_get_or_load_model(
												  model_name, ONNX_MODEL_CLASSIFICATION);
		if (!session)
		{
			if (errstr)
				*errstr = pstrdup("Failed to load ONNX cross-encoder model");
			NDB_FREE(scores);
			rc = -1;
		}
		else
		{
			/* Tokenize query once */
			query_token_ids = neurondb_tokenize_with_model(
														   query, 128, &query_token_length, model_name);
			if (!query_token_ids || query_token_length <= 0)
			{
				if (errstr)
					*errstr = pstrdup("Failed to tokenize query");
				NDB_FREE(scores);
				if (query_token_ids)
					NDB_FREE(query_token_ids);
				rc = -1;
			}
			else
			{
				/* Process each document */
				for (i = 0; i < ndocs; i++)
				{
					if (docs[i] == NULL || strlen(docs[i]) == 0)
					{
						scores[i] = 0.0f;
						continue;
					}

					/* Tokenize document */
					doc_token_ids = neurondb_tokenize_with_model(
																 docs[i], 384, &doc_token_length, model_name);
					if (!doc_token_ids || doc_token_length <= 0)
					{
						scores[i] = 0.0f;
						if (doc_token_ids)
							NDB_FREE(doc_token_ids);
						continue;
					}

					/*
					 * Combine query and document tokens: [CLS] query [SEP]
					 * doc [SEP]
					 */

					/*
					 * Reserve space: 1 (CLS) + query_len + 1 (SEP) + doc_len
					 * + 1 (SEP)
					 */
					combined_token_length = 1 + query_token_length + 1
						+ doc_token_length + 1;
					if (combined_token_length > max_combined_length)
					{
						/* Truncate document to fit */
						int			available = max_combined_length - 1
							- query_token_length - 1 - 1;

						if (available > 0)
							doc_token_length = available;
						else
							doc_token_length = 0;
						combined_token_length = max_combined_length;
					}

					NDB_DECLARE(int32 *, combined_token_ids);
					NDB_ALLOC(combined_token_ids, int32,
														  combined_token_length * sizeof(int32));
					if (!combined_token_ids)
					{
						scores[i] = 0.0f;
						if (doc_token_ids)
							NDB_FREE(doc_token_ids);
						continue;
					}

					/* Build combined sequence */
					j = 0;
					combined_token_ids[j++] = 101;	/* [CLS] token ID */
					/* Copy query tokens */
					{
						int			k;

						for (k = 0; k < query_token_length && j < combined_token_length - 2; k++)
							combined_token_ids[j++] = query_token_ids[k];
					}
					combined_token_ids[j++] = 102;	/* [SEP] token ID */
					/* Copy document tokens */
					{
						int			k;

						for (k = 0; k < doc_token_length && j < combined_token_length - 1; k++)
							combined_token_ids[j++] = doc_token_ids[k];
					}
					combined_token_ids[j++] = 102;	/* [SEP] token ID */
					combined_token_length = j;

					NDB_DECLARE(float *, input_data);

					/* Convert token IDs to float array for ONNX */
					NDB_ALLOC(input_data, float,
												  combined_token_length * sizeof(float));
					if (!input_data)
					{
						scores[i] = 0.0f;
						if (combined_token_ids)
							NDB_FREE(combined_token_ids);
						if (doc_token_ids)
							NDB_FREE(doc_token_ids);
						continue;
					}

					for (j = 0; j < combined_token_length; j++)
						input_data[j] = (float) combined_token_ids[j];

					/* Create input tensor */
					input_shape[0] = 1;
					input_shape[1] = combined_token_length;
					input_tensor = neurondb_onnx_create_tensor(
															   input_data, input_shape, 2);
					if (!input_tensor)
					{
						scores[i] = 0.0f;
						if (input_data)
							NDB_FREE(input_data);
						if (combined_token_ids)
							NDB_FREE(combined_token_ids);
						if (doc_token_ids)
							NDB_FREE(doc_token_ids);
						continue;
					}

					/* Run inference */
					output_tensor = neurondb_onnx_run_inference(
																session, input_tensor);
					if (!output_tensor || output_tensor->size <= 0)
					{
						scores[i] = 0.0f;
						neurondb_onnx_free_tensor(input_tensor);
						if (input_data)
							NDB_FREE(input_data);
						if (combined_token_ids)
							NDB_FREE(combined_token_ids);
						if (doc_token_ids)
							NDB_FREE(doc_token_ids);
						continue;
					}

					/*
					 * Extract score (cross-encoder outputs single
					 * logit/score)
					 */
					/* For binary classification, use first output */
					if (output_tensor->size >= 1)
					{
						/* Apply sigmoid if needed, or use raw logit */
						float		logit = output_tensor->data[0];

						/* Convert logit to probability using sigmoid */
						scores[i] = 1.0f / (1.0f + expf(-logit));
					}
					else
					{
						scores[i] = 0.0f;
					}

					/* Cleanup for this document */
					neurondb_onnx_free_tensor(input_tensor);
					neurondb_onnx_free_tensor(output_tensor);
					if (input_data)
						NDB_FREE(input_data);
					if (combined_token_ids)
						NDB_FREE(combined_token_ids);
					if (doc_token_ids)
						NDB_FREE(doc_token_ids);
					input_tensor = NULL;
					output_tensor = NULL;
					input_data = NULL;
					combined_token_ids = NULL;
					doc_token_ids = NULL;
				}

				/* Cleanup query tokens */
				if (query_token_ids)
					NDB_FREE(query_token_ids);

				*scores_out = scores;
				rc = 0;
			}
		}
	}
	PG_CATCH();
	{
		/* Defensive cleanup on error */
		if (input_tensor)
			neurondb_onnx_free_tensor(input_tensor);
		if (output_tensor)
			neurondb_onnx_free_tensor(output_tensor);
		if (input_data)
			NDB_FREE(input_data);
		if (combined_token_ids)
			NDB_FREE(combined_token_ids);
		if (doc_token_ids)
			NDB_FREE(doc_token_ids);
		if (query_token_ids)
			NDB_FREE(query_token_ids);
		if (scores)
			NDB_FREE(scores);
		if (errstr && !*errstr)
			*errstr = pstrdup("ONNX reranking error");
		*scores_out = NULL;
		rc = -1;
	}
	PG_END_TRY();

	return rc;
}

#else							/* !HAVE_ONNX_RUNTIME */

/*
 * Intentional conditional compilation stub when ONNX Runtime is not available.
 * This stub allows compilation without ONNX Runtime - functions return errors at runtime.
 */
int
ndb_onnx_hf_complete(const char *model_name,
					 const char *prompt,
					 const char *params_json,
					 char **text_out,
					 char **errstr)
{
	if (errstr)
		*errstr = pstrdup("ONNX Runtime support not compiled in");
	return -1;
}

int
ndb_onnx_hf_embed(const char *model_name,
				  const char *text,
				  float **vec_out,
				  int *dim_out,
				  char **errstr)
{
	if (errstr)
		*errstr = pstrdup("ONNX Runtime support not compiled in");
	return -1;
}

int
ndb_onnx_hf_image_embed(const char *model_name,
						const unsigned char *image_data,
						size_t image_size,
						float **vec_out,
						int *dim_out,
						char **errstr)
{
	if (errstr)
		*errstr = pstrdup("ONNX Runtime support not compiled in");
	return -1;
}

int
ndb_onnx_hf_multimodal_embed(const char *model_name,
							 const char *text,
							 const unsigned char *image_data,
							 size_t image_size,
							 float **vec_out,
							 int *dim_out,
							 char **errstr)
{
	if (errstr)
		*errstr = pstrdup("ONNX Runtime support not compiled in");
	return -1;
}

int
ndb_onnx_hf_rerank(const char *model_name,
				   const char *query,
				   const char **docs,
				   int ndocs,
				   float **scores_out,
				   char **errstr)
{
	if (errstr)
		*errstr = pstrdup("ONNX Runtime support not compiled in");
	return -1;
}

#endif							/* HAVE_ONNX_RUNTIME */
