/*-------------------------------------------------------------------------
 *
 * ml_text.c
 *    Text machine learning functions.
 *
 * This module provides text processing and machine learning functions
 * for natural language processing tasks.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_text.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "utils/memutils.h"

#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"
#include "neurondb_spi_safe.h"

#include <ctype.h>
#include <string.h>
#include <math.h>

/* PG_MODULE_MAGIC is in neurondb.c only */

PG_FUNCTION_INFO_V1(neurondb_text_classify);
PG_FUNCTION_INFO_V1(neurondb_sentiment_analysis);
PG_FUNCTION_INFO_V1(neurondb_named_entity_recognition);
PG_FUNCTION_INFO_V1(neurondb_text_summarize);

#define MAX_TOKENS 4096
#define MAX_TOKEN_LEN 128
#define MAX_CATEGORY 128
#define MAX_CATEGORIES 16
#define MAX_ENTITIES 256
#define MAX_ENTITY_TYPE 32
#define MAX_ENTITY_LEN 128
#define MAX_SENTIMENT 12
#define MAX_SUMMARY 4096
#define MAX_SENTENCES 512

typedef struct ClassifyResult
{
	char		category[MAX_CATEGORY];
	float4		confidence;
}			ClassifyResult;

typedef struct NERResult
{
	char		entity[MAX_ENTITY_LEN];
	char		entity_type[MAX_ENTITY_TYPE];
	float4		confidence;
	int32		entity_position;
}			NERResult;

typedef struct SentimentResult
{
	float4		positive;
	float4		negative;
	float4		neutral;
	char		sentiment[MAX_SENTIMENT];
}			SentimentResult;

/*
 * simple_tokenize - Tokenize input into lowercase word tokens
 *
 * Tokenizes an input string into lowercase alphanumeric word tokens.
 * Skips non-alphanumeric characters and converts all tokens to lowercase.
 *
 * Parameters:
 *   input - Input string to tokenize
 *   tokens - Output array of token strings (allocated in CurrentMemoryContext)
 *   num_tokens - Output parameter to receive number of tokens found
 *
 * Notes:
 *   Memory for tokens is allocated in CurrentMemoryContext. The function
 *   limits the number of tokens to MAX_TOKENS and token length to MAX_TOKEN_LEN.
 *   Caller is responsible for freeing token memory.
 */
static void
simple_tokenize(const char *input, char **tokens, int *num_tokens)
{
	int			i = 0;
	int			input_len = (int) strlen(input);
	int			t = 0;

	while (i < input_len && t < MAX_TOKENS)
	{
		char		wordbuf[MAX_TOKEN_LEN];
		int			j = 0;

		/* Skip non-alphanumeric */
		while (i < input_len && !isalnum((unsigned char) input[i]))
			i++;
		if (i >= input_len)
			break;

		memset(wordbuf, 0, sizeof(wordbuf));
		while (i < input_len && isalnum((unsigned char) input[i])
			   && j < MAX_TOKEN_LEN - 1)
		{
			wordbuf[j++] = (char) tolower((unsigned char) input[i]);
			i++;
		}
		wordbuf[j] = '\0';
		tokens[t++] = pstrdup(wordbuf);
	}
	*num_tokens = t;
}

/*
 * Text Classification (Bag-of-words + SPI model table support)
 *
 * Args:
 *    INT4   model_id
 *    TEXT   input
 * Returns SETOF (category text, confidence float4)
 */
Datum
neurondb_text_classify(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	int32		model_id = PG_GETARG_INT32(0);
	text	   *input_text = PG_GETARG_TEXT_PP(1);

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		char	   *input_str;
		char	   *tokens[MAX_TOKENS];
		int			num_tokens = 0;

		NDB_DECLARE(ClassifyResult *, results);
		int			n_categories = 0;
		int			ret;
		char		qry[256];
		char	  **categories;
		int		   *category_counts;
		int			i,
					t,
					r;

		NDB_DECLARE(NdbSpiSession *, spi_session);

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext =
			MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		if (model_id <= 0)
			ereport(ERROR,
					(errmsg("model_id must be positive integer")));

		input_str = text_to_cstring(input_text);
		simple_tokenize(input_str, tokens, &num_tokens);

		snprintf(qry,
				 sizeof(qry),
				 "SELECT c.category, w.word "
				 "FROM neurondb_textclass_words w "
				 "JOIN neurondb_textclass_categories c ON (w.cat_id = "
				 "c.id) "
				 "WHERE w.model_id = %d",
				 model_id);

		oldcontext = CurrentMemoryContext;

		NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

		ret = ndb_spi_execute(spi_session, qry, true, 0);
		if (ret != SPI_OK_SELECT)
		{
			NDB_SPI_SESSION_END(spi_session);
			for (t = 0; t < num_tokens; t++)
				NDB_FREE(tokens[t]);
			NDB_FREE(input_str);
			ereport(ERROR,
					(errmsg("Could not fetch model word lists")));
		}

		/* Prepare category lists */
		categories = (char **) palloc0(sizeof(char *) * MAX_CATEGORIES);
		NDB_CHECK_ALLOC(categories, "categories");
		category_counts = (int *) palloc0(sizeof(int) * MAX_CATEGORIES);
		NDB_CHECK_ALLOC(category_counts, "category_counts");

		/* Fill category names (de-duplication) */
		for (r = 0; r < (int) SPI_processed; r++)
		{
			HeapTuple	tuple = SPI_tuptable->vals[r];
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;
			char	   *cat = TextDatumGetCString(
												  SPI_getbinval(tuple, tupdesc, 1, NULL));
			bool		found = false;

			for (i = 0; i < n_categories; i++)
			{
				if (strcmp(categories[i], cat) == 0)
				{
					found = true;
					break;
				}
			}
			if (!found && n_categories < MAX_CATEGORIES)
			{
				categories[n_categories++] = pstrdup(cat);
			}
			NDB_FREE(cat);
		}

		if (n_categories == 0)
		{
			NDB_SPI_SESSION_END(spi_session);
			for (t = 0; t < num_tokens; t++)
				NDB_FREE(tokens[t]);
			NDB_FREE(input_str);
			ereport(ERROR,
					(errmsg("No categories found for model_id %d",
							model_id)));
		}

		/* Tally up word matches for each category */
		for (r = 0; r < (int) SPI_processed; r++)
		{
			TupleDesc	tupdesc;
			text	   *cat_text;
			char	   *cat;
			text	   *word_text;
			char	   *word;

			/* Safe access to SPI_tuptable - validate before access */
			if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
				r >= (int) SPI_processed || SPI_tuptable->vals[r] == NULL)
			{
				continue;
			}
			tupdesc = SPI_tuptable->tupdesc;
			if (tupdesc == NULL)
			{
				continue;
			}
			/* Use safe function to get text */
			cat_text = ndb_spi_get_text(spi_session, r, 1, oldcontext);
			if (cat_text == NULL)
				continue;
			cat = text_to_cstring(cat_text);
			NDB_FREE(cat_text);

			/* Safe access for word - validate tupdesc has at least 2 columns */
			if (tupdesc->natts < 2)
			{
				NDB_FREE(cat);
				continue;
			}
			word_text = ndb_spi_get_text(spi_session, r, 2, oldcontext);
			if (word_text == NULL)
			{
				NDB_FREE(cat);
				continue;
			}
			word = text_to_cstring(word_text);
			NDB_FREE(word_text);

			for (t = 0; t < num_tokens; t++)
			{
				if (strcmp(tokens[t], word) == 0)
				{
					for (i = 0; i < n_categories; i++)
					{
						if (strcmp(categories[i], cat)
							== 0)
						{
							category_counts[i]++;
							break;
						}
					}
				}
			}
			NDB_FREE(cat);
			NDB_FREE(word);
		}

		for (t = 0; t < num_tokens; t++)
			NDB_FREE(tokens[t]);
		{
			/* Calculate confidences */
			int			total_count = 0;

			NDB_SPI_SESSION_END(spi_session);
			results = (ClassifyResult *) palloc0(
												 n_categories * sizeof(ClassifyResult));
			NDB_CHECK_ALLOC(results, "results");

			for (i = 0; i < n_categories; i++)
				total_count += category_counts[i];

			for (i = 0; i < n_categories; i++)
			{
				strlcpy(results[i].category,
						categories[i],
						MAX_CATEGORY);
				if (total_count)
					results[i].confidence =
						((float4) category_counts[i])
						/ total_count;
				else
					results[i].confidence =
						(1.0f / (float4) n_categories);
				NDB_FREE(categories[i]);
			}
			NDB_FREE(categories);
			NDB_FREE(category_counts);

			funcctx->user_fctx = results;
			funcctx->max_calls = n_categories;
			MemoryContextSwitchTo(oldcontext);
		}
	}

	funcctx = SRF_PERCALL_SETUP();
	{
		ClassifyResult *results = (ClassifyResult *) funcctx->user_fctx;

		if (funcctx->call_cntr < funcctx->max_calls)
		{
			Datum		values[2];
			bool		nulls[2] = {false, false};
			HeapTuple	tuple;

			values[0] = CStringGetTextDatum(
											results[funcctx->call_cntr].category);
			values[1] = Float4GetDatum(
									   results[funcctx->call_cntr].confidence);

			if (funcctx->tuple_desc == NULL)
			{
				TupleDesc	desc = CreateTemplateTupleDesc(2);

				TupleDescInitEntry(desc,
								   (AttrNumber) 1,
								   "category",
								   TEXTOID,
								   -1,
								   0);
				TupleDescInitEntry(desc,
								   (AttrNumber) 2,
								   "confidence",
								   FLOAT4OID,
								   -1,
								   0);
				funcctx->tuple_desc = BlessTupleDesc(desc);
			}
			tuple = heap_form_tuple(
									funcctx->tuple_desc, values, nulls);
			SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
		}
		else
		{
			if (funcctx->max_calls > 0 && funcctx->user_fctx)
				NDB_FREE(results);
			SRF_RETURN_DONE(funcctx);
		}
	}
}

/* Sentiment analysis based on VADER-like lexicon (from SPI table) */
Datum
neurondb_sentiment_analysis(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	text	   *input_text = PG_GETARG_TEXT_PP(0);
	text	   *model_text = PG_ARGISNULL(1) ? NULL : PG_GETARG_TEXT_PP(1);

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		char	   *input_str;

		NDB_DECLARE(char *, model_name);
		char	   *tokens[MAX_TOKENS];
		int			num_tokens = 0;
		int			pos = 0,
					neg = 0,
					neu = 0;
		SentimentResult *result;
		int			ret;
		char		qry[256];
		int			t,
					r;

		NDB_DECLARE(NdbSpiSession *, spi_session);

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext =
			MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		input_str = text_to_cstring(input_text);

		if (model_text)
			model_name = text_to_cstring(model_text);
		else
			model_name = pstrdup("vader");

		simple_tokenize(input_str, tokens, &num_tokens);

		snprintf(qry,
				 sizeof(qry),
				 "SELECT word, polarity FROM neurondb_sentiment_lexicon WHERE model = '%s'",
				 model_name);

		oldcontext = CurrentMemoryContext;

		NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

		ret = ndb_spi_execute(spi_session, qry, true, 0);
		if (ret != SPI_OK_SELECT)
		{
			NDB_SPI_SESSION_END(spi_session);
			for (t = 0; t < num_tokens; t++)
				NDB_FREE(tokens[t]);
			NDB_FREE(model_name);
			ereport(ERROR,
					(errmsg("Could not execute sentiment lexicon "
							"fetch")));
		}

		for (t = 0; t < num_tokens; t++)
		{
			bool		found = false;

			for (r = 0; r < (int) SPI_processed; r++)
			{
				HeapTuple	tuple = SPI_tuptable->vals[r];
				TupleDesc	tupdesc = SPI_tuptable->tupdesc;
				char	   *w = TextDatumGetCString(
													SPI_getbinval(tuple, tupdesc, 1, NULL));
				char	   *pol = TextDatumGetCString(
													  SPI_getbinval(tuple, tupdesc, 2, NULL));

				if (strcmp(tokens[t], w) == 0)
				{
					found = true;
					if (strcmp(pol, "positive") == 0)
						pos++;
					else if (strcmp(pol, "negative") == 0)
						neg++;
					else
						neu++;	/* treat unknown as neutral */
				}
				NDB_FREE(w);
				NDB_FREE(pol);
				if (found)
					break;
			}
			if (!found)
				neu++;
			NDB_FREE(tokens[t]);
		}
		NDB_FREE(model_name);

		NDB_SPI_SESSION_END(spi_session);

		if (num_tokens == 0)
			num_tokens = 1;

		result = (SentimentResult *) palloc0(sizeof(SentimentResult));
		NDB_CHECK_ALLOC(result, "result");
		result->positive = ((float4) pos) / num_tokens;
		result->negative = ((float4) neg) / num_tokens;
		result->neutral = ((float4) neu) / num_tokens;

		if (result->positive >= result->negative
			&& result->positive >= result->neutral)
		{
			strlcpy(result->sentiment, "positive", MAX_SENTIMENT);
		}
		else if (result->negative >= result->positive
				 && result->negative >= result->neutral)
		{
			strlcpy(result->sentiment, "negative", MAX_SENTIMENT);
		}
		else
		{
			strlcpy(result->sentiment, "neutral", MAX_SENTIMENT);
		}

		funcctx->user_fctx = result;
		funcctx->max_calls = 1;
		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();
	{
		SentimentResult *result = (SentimentResult *) funcctx->user_fctx;

		if (funcctx->call_cntr < funcctx->max_calls)
		{
			Datum		values[4];
			bool		nulls[4] = {false, false, false, false};
			HeapTuple	tuple;

			values[0] = CStringGetTextDatum(result->sentiment);
			values[1] = Float4GetDatum(result->positive);
			values[2] = Float4GetDatum(result->negative);
			values[3] = Float4GetDatum(result->neutral);

			if (funcctx->tuple_desc == NULL)
			{
				TupleDesc	desc = CreateTemplateTupleDesc(4);

				TupleDescInitEntry(desc,
								   (AttrNumber) 1,
								   "sentiment",
								   TEXTOID,
								   -1,
								   0);
				TupleDescInitEntry(desc,
								   (AttrNumber) 2,
								   "positive",
								   FLOAT4OID,
								   -1,
								   0);
				TupleDescInitEntry(desc,
								   (AttrNumber) 3,
								   "negative",
								   FLOAT4OID,
								   -1,
								   0);
				TupleDescInitEntry(desc,
								   (AttrNumber) 4,
								   "neutral",
								   FLOAT4OID,
								   -1,
								   0);
				funcctx->tuple_desc = BlessTupleDesc(desc);
			}
			tuple = heap_form_tuple(
									funcctx->tuple_desc, values, nulls);
			NDB_FREE(result);
			SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
		}
		else
		{
			SRF_RETURN_DONE(funcctx);
		}
	}
}

/*
 * Named Entity Recognition using SPI table rules
 * (Any entity in neurondb_ner_entities with matching tokens)
 */
Datum
neurondb_named_entity_recognition(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	text	   *input_text = PG_GETARG_TEXT_PP(0);
	ArrayType  *entity_types_array =
		PG_ARGISNULL(1) ? NULL : PG_GETARG_ARRAYTYPE_P(1);

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		char	   *input_str;
		char	   *tokens[MAX_TOKENS];
		int			num_tokens = 0;
		NERResult  *entities;
		int			n_entities = 0;
		int			ret,
					t,
					r;

		NDB_DECLARE(NdbSpiSession *, spi_session);

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext =
			MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		input_str = text_to_cstring(input_text);
		simple_tokenize(input_str, tokens, &num_tokens);

		oldcontext = CurrentMemoryContext;

		NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

		ret = ndb_spi_execute(spi_session, "SELECT entity, entity_type, confidence FROM neurondb_ner_entities", true, 0);
		if (ret != SPI_OK_SELECT)
		{
			NDB_SPI_SESSION_END(spi_session);
			for (t = 0; t < num_tokens; t++)
				NDB_FREE(tokens[t]);
			ereport(ERROR,
					(errmsg("NER entity table fetch failed")));
		}

		entities =
			(NERResult *) palloc0(MAX_ENTITIES * sizeof(NERResult));

		for (t = 0; t < num_tokens && n_entities < MAX_ENTITIES; t++)
		{
			for (r = 0; r < (int) SPI_processed
				 && n_entities < MAX_ENTITIES;
				 r++)
			{
				HeapTuple	tuple = SPI_tuptable->vals[r];
				TupleDesc	tupdesc = SPI_tuptable->tupdesc;
				char	   *dbent = TextDatumGetCString(
														SPI_getbinval(tuple, tupdesc, 1, NULL));
				char	   *type = TextDatumGetCString(
													   SPI_getbinval(tuple, tupdesc, 2, NULL));
				float4		conf = DatumGetFloat4(
												  SPI_getbinval(tuple, tupdesc, 3, NULL));

				if (strcmp(tokens[t], dbent) == 0)
				{
					strlcpy(entities[n_entities].entity,
							dbent,
							MAX_ENTITY_LEN);
					strlcpy(entities[n_entities]
							.entity_type,
							type,
							MAX_ENTITY_TYPE);
					entities[n_entities].confidence = conf;
					entities[n_entities].entity_position =
						t + 1;
					n_entities++;
				}
				NDB_FREE(dbent);
				NDB_FREE(type);
			}
			NDB_FREE(tokens[t]);
		}
		NDB_SPI_SESSION_END(spi_session);

		/* Optional: filter by entity_type list arg */
		if (entity_types_array != NULL && n_entities > 0)
		{
			Datum	   *datum_array;
			bool	   *nulls;
			int			n_types;
			int			k = 0,
						e,
						i;
			NERResult  *filtered;

			deconstruct_array(entity_types_array,
							  TEXTOID,
							  -1,
							  false,
							  'i',
							  &datum_array,
							  &nulls,
							  &n_types);
			filtered = (NERResult *) palloc0(
											 MAX_ENTITIES * sizeof(NERResult));
			NDB_CHECK_ALLOC(filtered, "filtered");
			for (e = 0; e < n_entities; e++)
			{
				bool		keep = false;

				for (i = 0; i < n_types; i++)
				{
					char	   *etype = TextDatumGetCString(
															datum_array[i]);

					if (pg_strcasecmp(etype,
									  entities[e].entity_type)
						== 0)
					{
						keep = true;
					}
					NDB_FREE(etype);
					if (keep)
						break;
				}
				if (keep && k < MAX_ENTITIES)
				{
					filtered[k++] = entities[e];
				}
			}
			NDB_FREE(entities);
			entities = filtered;
			n_entities = k;
		}

		funcctx->user_fctx = entities;
		funcctx->max_calls = n_entities;
		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();
	{
		NERResult  *entities = (NERResult *) funcctx->user_fctx;

		if (funcctx->call_cntr < funcctx->max_calls)
		{
			Datum		values[4];
			bool		nulls[4] = {false, false, false, false};
			HeapTuple	tuple;

			values[0] = CStringGetTextDatum(
											entities[funcctx->call_cntr].entity);
			values[1] = CStringGetTextDatum(
											entities[funcctx->call_cntr].entity_type);
			values[2] = Float4GetDatum(
									   entities[funcctx->call_cntr].confidence);
			values[3] = Int32GetDatum(
									  entities[funcctx->call_cntr].entity_position);

			if (funcctx->tuple_desc == NULL)
			{
				TupleDesc	desc = CreateTemplateTupleDesc(4);

				TupleDescInitEntry(desc,
								   (AttrNumber) 1,
								   "entity",
								   TEXTOID,
								   -1,
								   0);
				TupleDescInitEntry(desc,
								   (AttrNumber) 2,
								   "entity_type",
								   TEXTOID,
								   -1,
								   0);
				TupleDescInitEntry(desc,
								   (AttrNumber) 3,
								   "confidence",
								   FLOAT4OID,
								   -1,
								   0);
				TupleDescInitEntry(desc,
								   (AttrNumber) 4,
								   "entity_position",
								   INT4OID,
								   -1,
								   0);
				funcctx->tuple_desc = BlessTupleDesc(desc);
			}
			tuple = heap_form_tuple(
									funcctx->tuple_desc, values, nulls);
			SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
		}
		else
		{
			if (funcctx->max_calls > 0 && funcctx->user_fctx)
				NDB_FREE(entities);
			SRF_RETURN_DONE(funcctx);
		}
	}
}

/*
 * Text Summarization (extractive: top sentences by non-stopword count, uses SPI stopwords)
 * Args:
 *    text input
 *    int4 max_length (optional, default 128)
 *    text method (optional, default "extractive")
 */
Datum
neurondb_text_summarize(PG_FUNCTION_ARGS)
{
	text	   *input_text = PG_GETARG_TEXT_PP(0);
	int32		max_length = PG_ARGISNULL(1) ? 128 : PG_GETARG_INT32(1);
	text	   *method_text = PG_ARGISNULL(2) ? NULL : PG_GETARG_TEXT_PP(2);
	char	   *text_str;
	char	   *method;
	int			len;
	char		summary[MAX_SUMMARY];
	int			i;
	MemoryContext oldcontext;

	NDB_DECLARE(NdbSpiSession *, spi_session);

	text_str = text_to_cstring(input_text);
	len = (int) strlen(text_str);

	if (method_text)
		method = text_to_cstring(method_text);
	else
		method = pstrdup("extractive");

	memset(summary, 0, sizeof(summary));

	if (pg_strcasecmp(method, "extractive") == 0)
	{
		/* Split text to sentences (by '.', '?', '!') */
		int			sstart = 0,
					send = 0,
					slen;
		char	   *sentence_ptrs[MAX_SENTENCES];
		int			scores[MAX_SENTENCES];
		int			n_sentences = 0,
					written = 0;
		int			used[MAX_SENTENCES];
		int			ret,
					s;

		memset(used, 0, sizeof(used));
		while (send < len && n_sentences < MAX_SENTENCES)
		{
			sstart = send;
			while (send < len && text_str[send] != '.'
				   && text_str[send] != '?'
				   && text_str[send] != '!')
				send++;
			if (send < len)
				send++;
			slen = send - sstart;
			if (slen > 0 && n_sentences < MAX_SENTENCES)
			{
				sentence_ptrs[n_sentences] =
					pnstrdup(text_str + sstart, slen);
				scores[n_sentences] = 0;
				n_sentences++;
			}
			while (send < len
				   && isspace((unsigned char) text_str[send]))
				send++;
		}

		oldcontext = CurrentMemoryContext;

		NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

		ret = ndb_spi_execute(spi_session, "SELECT stopword FROM neurondb_summarizer_stopwords", true, 0);
		if (ret != SPI_OK_SELECT)
		{
			NDB_SPI_SESSION_END(spi_session);
			for (s = 0; s < n_sentences; s++)
				NDB_FREE(sentence_ptrs[s]);
			NDB_FREE(method);
			ereport(ERROR,
					(errmsg("could not query stopword table")));
		}
		{
			int			n_stopwords;
			char	  **stopwords;

			n_stopwords = SPI_processed;
			stopwords = (char **) palloc0(n_stopwords * sizeof(char *));
			NDB_CHECK_ALLOC(stopwords, "stopwords");
			for (i = 0; i < n_stopwords; i++)
			{
				HeapTuple	tup = SPI_tuptable->vals[i];
				TupleDesc	desc = SPI_tuptable->tupdesc;

				stopwords[i] = TextDatumGetCString(
												   SPI_getbinval(tup, desc, 1, NULL));
			}

			/* Score: count of non-stopword tokens per sentence */
			for (s = 0; s < n_sentences; s++)
			{
				char	   *sentence = sentence_ptrs[s];
				char	   *stoks[128];
				int			stok_ct = 0,
							tokidx;

				simple_tokenize(sentence, stoks, &stok_ct);
				for (tokidx = 0; tokidx < stok_ct; tokidx++)
				{
					bool		is_stop = false;
					int			sw;

					for (sw = 0; sw < n_stopwords; sw++)
					{
						if (strcmp(stoks[tokidx], stopwords[sw])
							== 0)
						{
							is_stop = true;
							break;
						}
					}
					if (!is_stop)
						scores[s]++;
					NDB_FREE(stoks[tokidx]);
				}
			}

			/* Assemble top sentences into summary */
			written = 0;
			while (written < max_length - 1)
			{
				int			maxscore = -1,
							maxi = -1;
				int			sl;
				int			tocopy;

				for (s = 0; s < n_sentences; s++)
				{
					if (!used[s] && scores[s] > maxscore)
					{
						maxscore = scores[s];
						maxi = s;
					}
				}
				if (maxi == -1 || maxscore == 0)
					break;
				sl = (int) strlen(sentence_ptrs[maxi]);
				tocopy = (sl > max_length - 1 - written)
					? (max_length - 1 - written)
					: sl;
				if (tocopy > 0)
				{
					memcpy(summary + written,
						   sentence_ptrs[maxi],
						   tocopy);
					written += tocopy;
					if (written < max_length - 1)
					{
						summary[written] = ' ';
						written++;
					}
				}
				used[maxi] = 1;
				if (written >= max_length - 1)
					break;
			}
			if (written > 0)
				summary[written - 1] = '\0';
			else
				summary[0] = '\0';

			for (i = 0; i < n_sentences; i++)
				NDB_FREE(sentence_ptrs[i]);
			for (i = 0; i < n_stopwords; i++)
				NDB_FREE(stopwords[i]);
			NDB_FREE(stopwords);
		}
		NDB_SPI_SESSION_END(spi_session);
	}
	else
	{
		/* Abstractive: copy first max_length-8 bytes, append marker */
		int			j = 0;

		for (i = 0; i < len && j < max_length - 8; i++)
		{
			summary[j++] = text_str[i];
		}
		summary[j] = '\0';
		if (j > 0 && summary[j - 1] == ' ')
			summary[j - 1] = '\0';
		strlcat(summary, " [abs]", sizeof(summary));
	}
	NDB_FREE(method);
	PG_RETURN_TEXT_P(cstring_to_text(summary));
}

#include "neurondb_gpu_model.h"
#include "ml_gpu_registry.h"

typedef struct TextGpuModelState
{
	bytea	   *model_blob;
	Jsonb	   *metrics;
	int			vocab_size;
	int			feature_dim;
	int			n_samples;
	char		task_type[32];
}			TextGpuModelState;

static bytea *
text_model_serialize_to_bytea(int vocab_size, int feature_dim, const char *task_type)
{
	StringInfoData buf;
	int			total_size;
	bytea	   *result;
	int			task_len;

	initStringInfo(&buf);
	appendBinaryStringInfo(&buf, (char *) &vocab_size, sizeof(int));
	appendBinaryStringInfo(&buf, (char *) &feature_dim, sizeof(int));
	task_len = strlen(task_type);
	appendBinaryStringInfo(&buf, (char *) &task_len, sizeof(int));
	appendBinaryStringInfo(&buf, task_type, task_len);

	total_size = VARHDRSZ + buf.len;
	result = (bytea *) palloc(total_size);
	NDB_CHECK_ALLOC(result, "result");
	SET_VARSIZE(result, total_size);
	memcpy(VARDATA(result), buf.data, buf.len);
	NDB_FREE(buf.data);

	return result;
}

static int
text_model_deserialize_from_bytea(const bytea * data, int *vocab_size_out, int *feature_dim_out, char *task_type_out, int task_max)
{
	const char *buf;
	int			offset = 0;
	int			task_len;

	if (data == NULL || VARSIZE(data) < VARHDRSZ + sizeof(int) * 3)
		return -1;

	buf = VARDATA(data);
	memcpy(vocab_size_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(feature_dim_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(&task_len, buf + offset, sizeof(int));
	offset += sizeof(int);

	if (task_len >= task_max)
		return -1;
	memcpy(task_type_out, buf + offset, task_len);
	task_type_out[task_len] = '\0';

	return 0;
}

static bool
text_gpu_train(MLGpuModel *model, const MLGpuTrainSpec *spec, char **errstr)
{
	TextGpuModelState *state;
	int			vocab_size = 1000;
	int			feature_dim = 128;
	char		task_type[32] = "classification";
	int			nvec = 0;

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
			*errstr = pstrdup("text_gpu_train: invalid parameters");
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
				else if (strcmp(key, "feature_dim") == 0 && v.type == jbvNumeric)
					feature_dim = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																	NumericGetDatum(v.val.numeric)));
				else if (strcmp(key, "task_type") == 0 && v.type == jbvString)
					strncpy(task_type, v.val.string.val, sizeof(task_type) - 1);
				NDB_FREE(key);
			}
		}
	}

	if (vocab_size < 1)
		vocab_size = 1000;
	if (feature_dim < 1)
		feature_dim = 128;

	/* Convert feature matrix to count samples */
	if (spec->feature_matrix == NULL || spec->sample_count <= 0
		|| spec->feature_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("text_gpu_train: invalid feature matrix");
		return false;
	}

	nvec = spec->sample_count;

	/* Serialize model */
	model_data = text_model_serialize_to_bytea(vocab_size, feature_dim, task_type);

	/* Build metrics */
	initStringInfo(&metrics_json);
	appendStringInfo(&metrics_json,
					 "{\"storage\":\"cpu\",\"vocab_size\":%d,\"feature_dim\":%d,\"task_type\":\"%s\",\"n_samples\":%d}",
					 vocab_size, feature_dim, task_type, nvec);
	metrics = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
												 CStringGetDatum(metrics_json.data)));
	NDB_FREE(metrics_json.data);

	state = (TextGpuModelState *) palloc0(sizeof(TextGpuModelState));
	NDB_CHECK_ALLOC(state, "state");
	state->model_blob = model_data;
	state->metrics = metrics;
	state->vocab_size = vocab_size;
	state->feature_dim = feature_dim;
	state->n_samples = nvec;
	strncpy(state->task_type, task_type, sizeof(state->task_type) - 1);

	if (model->backend_state != NULL)
		NDB_FREE(model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = false;

	return true;
}

static bool
text_gpu_predict(const MLGpuModel *model, const float *input, int input_dim,
				 float *output, int output_dim, char **errstr)
{
	const		TextGpuModelState *state;
	int			i;

	if (errstr != NULL)
		*errstr = NULL;
	if (output != NULL && output_dim > 0)
		memset(output, 0, output_dim * sizeof(float));
	if (model == NULL || input == NULL || output == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("text_gpu_predict: invalid parameters");
		return false;
	}
	if (output_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("text_gpu_predict: invalid output dimension");
		return false;
	}
	if (!model->gpu_ready || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("text_gpu_predict: model not ready");
		return false;
	}

	state = (const TextGpuModelState *) model->backend_state;

	/* Simple prediction: return normalized input features */
	if (strcmp(state->task_type, "classification") == 0)
	{
		float		sum = 0.0f;

		for (i = 0; i < input_dim && i < output_dim; i++)
		{
			output[i] = input[i];
			sum += input[i] * input[i];
		}
		if (sum > 0.0f)
		{
			sum = (float) sqrt((double) sum);
			for (i = 0; i < input_dim && i < output_dim; i++)
				output[i] /= sum;
		}
	}
	else
	{
		/* Regression or other tasks */
		for (i = 0; i < input_dim && i < output_dim; i++)
			output[i] = input[i];
	}

	return true;
}

static bool
text_gpu_evaluate(const MLGpuModel *model, const MLGpuEvalSpec *spec,
				  MLGpuMetrics *out, char **errstr)
{
	const		TextGpuModelState *state;
	Jsonb	   *metrics_json;
	StringInfoData buf;

	if (errstr != NULL)
		*errstr = NULL;
	if (out != NULL)
		out->payload = NULL;
	if (model == NULL || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("text_gpu_evaluate: invalid model");
		return false;
	}

	state = (const TextGpuModelState *) model->backend_state;

	initStringInfo(&buf);
	appendStringInfo(&buf,
					 "{\"algorithm\":\"text\",\"storage\":\"cpu\","
					 "\"vocab_size\":%d,\"feature_dim\":%d,\"task_type\":\"%s\",\"n_samples\":%d}",
					 state->vocab_size > 0 ? state->vocab_size : 1000,
					 state->feature_dim > 0 ? state->feature_dim : 128,
					 state->task_type[0] ? state->task_type : "classification",
					 state->n_samples > 0 ? state->n_samples : 0);

	metrics_json = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
													  CStringGetDatum(buf.data)));
	NDB_FREE(buf.data);

	if (out != NULL)
		out->payload = metrics_json;

	return true;
}

static bool
text_gpu_serialize(const MLGpuModel *model, bytea * *payload_out,
				   Jsonb * *metadata_out, char **errstr)
{
	const		TextGpuModelState *state;
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
			*errstr = pstrdup("text_gpu_serialize: invalid model");
		return false;
	}

	state = (const TextGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("text_gpu_serialize: model blob is NULL");
		return false;
	}

	payload_size = VARSIZE(state->model_blob);
	payload_copy = (bytea *) palloc(payload_size);
	NDB_CHECK_ALLOC(payload_copy, "payload_copy");
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
text_gpu_deserialize(MLGpuModel *model, const bytea * payload,
					 const Jsonb * metadata, char **errstr)
{
	TextGpuModelState *state;
	bytea	   *payload_copy;
	int			payload_size;
	int			vocab_size = 0;
	int			feature_dim = 0;
	char		task_type[32];
	JsonbIterator *it;
	JsonbValue	v;
	int			r;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("text_gpu_deserialize: invalid parameters");
		return false;
	}

	payload_size = VARSIZE(payload);
	payload_copy = (bytea *) palloc(payload_size);
	NDB_CHECK_ALLOC(payload_copy, "payload_copy");
	memcpy(payload_copy, payload, payload_size);

	if (text_model_deserialize_from_bytea(payload_copy, &vocab_size, &feature_dim, task_type, sizeof(task_type)) != 0)
	{
		NDB_FREE(payload_copy);
		if (errstr != NULL)
			*errstr = pstrdup("text_gpu_deserialize: failed to deserialize");
		return false;
	}

	state = (TextGpuModelState *) palloc0(sizeof(TextGpuModelState));
	NDB_CHECK_ALLOC(state, "state");
	state->model_blob = payload_copy;
	state->vocab_size = vocab_size;
	state->feature_dim = feature_dim;
	state->n_samples = 0;
	strncpy(state->task_type, task_type, sizeof(state->task_type) - 1);

	if (metadata != NULL)
	{
		int			metadata_size = VARSIZE(metadata);
		Jsonb	   *metadata_copy = (Jsonb *) palloc(metadata_size);

		NDB_CHECK_ALLOC(metadata_copy, "metadata_copy");
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
text_gpu_destroy(MLGpuModel *model)
{
	TextGpuModelState *state;

	if (model == NULL)
		return;

	if (model->backend_state != NULL)
	{
		state = (TextGpuModelState *) model->backend_state;
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

static const MLGpuModelOps text_gpu_model_ops = {
	.algorithm = "text",
	.train = text_gpu_train,
	.predict = text_gpu_predict,
	.evaluate = text_gpu_evaluate,
	.serialize = text_gpu_serialize,
	.deserialize = text_gpu_deserialize,
	.destroy = text_gpu_destroy,
};

void
neurondb_gpu_register_text_model(void)
{
	static bool registered = false;

	if (registered)
		return;
	ndb_gpu_register_model_ops(&text_gpu_model_ops);
	registered = true;
}
