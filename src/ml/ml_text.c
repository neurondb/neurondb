/*-------------------------------------------------------------------------
 *
 * ml_text.c
 *    Text machine learning functions.
 *
 * This module provides text processing and machine learning functions
 * for natural language processing tasks.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
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
	FuncCallContext *funcctx = NULL;
	int32		model_id = PG_GETARG_INT32(0);
	text	   *input_text = PG_GETARG_TEXT_PP(1);

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		char *input_str = NULL;
		char	   *tokens[MAX_TOKENS];
		int			num_tokens = 0;

		ClassifyResult *results = NULL;
		int			n_categories = 0;
		int			ret;
		char		qry[256];
		char	  **categories;
		int *category_counts = NULL;
		int			i,
					t,
					r;

		NdbSpiSession *spi_session = NULL;

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
				nfree(tokens[t]);
			nfree(input_str);
			ereport(ERROR,
					(errmsg("Could not fetch model word lists")));
		}

		/* Prepare category lists */
		nalloc(categories, char *, MAX_CATEGORIES);
		NDB_CHECK_ALLOC(categories, "categories");
		MemSet(categories, 0, sizeof(char *) * MAX_CATEGORIES);
		nalloc(category_counts, int, MAX_CATEGORIES);
		NDB_CHECK_ALLOC(category_counts, "category_counts");
		MemSet(category_counts, 0, sizeof(int) * MAX_CATEGORIES);
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
			nfree(cat);
		}

		if (n_categories == 0)
		{
			NDB_SPI_SESSION_END(spi_session);
			for (t = 0; t < num_tokens; t++)
				nfree(tokens[t]);
			nfree(input_str);
			ereport(ERROR,
					(errmsg("No categories found for model_id %d",
							model_id)));
		}

		/* Tally up word matches for each category */
		for (r = 0; r < (int) SPI_processed; r++)
		{
			TupleDesc	tupdesc;
			text *cat_text = NULL;
			char *cat = NULL;
			text *word_text = NULL;
			char *word = NULL;

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
			nfree(cat_text);

			/* Safe access for word - validate tupdesc has at least 2 columns */
			if (tupdesc->natts < 2)
			{
				nfree(cat);
				continue;
			}
			word_text = ndb_spi_get_text(spi_session, r, 2, oldcontext);
			if (word_text == NULL)
			{
				nfree(cat);
				continue;
			}
			word = text_to_cstring(word_text);
			nfree(word_text);

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
			nfree(cat);
			nfree(word);
		}

		for (t = 0; t < num_tokens; t++)
			nfree(tokens[t]);
		{
			/* Calculate confidences */
			int			total_count = 0;

			NDB_SPI_SESSION_END(spi_session);
			nalloc(results, ClassifyResult, n_categories);
			MemSet(results, 0, sizeof(ClassifyResult) * n_categories);
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
				nfree(categories[i]);
			}
			nfree(categories);
			nfree(category_counts);

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
				nfree(results);
			SRF_RETURN_DONE(funcctx);
		}
	}
}

/* Sentiment analysis based on VADER-like lexicon (from SPI table) */
Datum
neurondb_sentiment_analysis(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx = NULL;
	text	   *input_text = PG_GETARG_TEXT_PP(0);
	text	   *model_text = PG_ARGISNULL(1) ? NULL : PG_GETARG_TEXT_PP(1);

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		char *input_str = NULL;

		char *model_name = NULL;
		char	   *tokens[MAX_TOKENS];
		int			num_tokens = 0;
		int			pos = 0,
					neg = 0,
					neu = 0;
		SentimentResult *result = NULL;
		int			ret;
		char		qry[256];
		int			t,
					r;

		NdbSpiSession *spi_session = NULL;

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
				nfree(tokens[t]);
			nfree(model_name);
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
				nfree(w);
				nfree(pol);
				if (found)
					break;
			}
			if (!found)
				neu++;
			nfree(tokens[t]);
		}
		nfree(model_name);

		NDB_SPI_SESSION_END(spi_session);

		if (num_tokens == 0)
			num_tokens = 1;

		nalloc(result, SentimentResult, 1);
		MemSet(result, 0, sizeof(SentimentResult));
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
			nfree(result);
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
	FuncCallContext *funcctx = NULL;
	text	   *input_text = PG_GETARG_TEXT_PP(0);
	ArrayType  *entity_types_array =
		PG_ARGISNULL(1) ? NULL : PG_GETARG_ARRAYTYPE_P(1);

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		char *input_str = NULL;
		char	   *tokens[MAX_TOKENS];
		int			num_tokens = 0;
		NERResult *entities = NULL;
		int			n_entities = 0;
		int			ret,
					t,
					r;

		NdbSpiSession *spi_session = NULL;

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
				nfree(tokens[t]);
			ereport(ERROR,
					(errmsg("NER entity table fetch failed")));
		}

		nalloc(entities, NERResult, MAX_ENTITIES);
		MemSet(entities, 0, sizeof(NERResult) * MAX_ENTITIES);

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
				nfree(dbent);
				nfree(type);
			}
			nfree(tokens[t]);
		}
		NDB_SPI_SESSION_END(spi_session);

		/* Optional: filter by entity_type list arg */
		if (entity_types_array != NULL && n_entities > 0)
		{
			Datum *datum_array = NULL;
			bool *nulls = NULL;
			int			n_types;
			int			k = 0,
						e,
						i;
			NERResult *filtered = NULL;

			deconstruct_array(entity_types_array,
							  TEXTOID,
							  -1,
							  false,
							  'i',
							  &datum_array,
							  &nulls,
							  &n_types);
			nalloc(filtered, NERResult, MAX_ENTITIES);
			MemSet(filtered, 0, sizeof(NERResult) * MAX_ENTITIES);
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
					nfree(etype);
					if (keep)
						break;
				}
				if (keep && k < MAX_ENTITIES)
				{
					filtered[k++] = entities[e];
				}
			}
			nfree(entities);
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
				nfree(entities);
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
	char *text_str = NULL;
	char *method = NULL;
	int			len;
	char		summary[MAX_SUMMARY];
	int			i;
	MemoryContext oldcontext;

	NdbSpiSession *spi_session = NULL;

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
				nfree(sentence_ptrs[s]);
			nfree(method);
			ereport(ERROR,
					(errmsg("could not query stopword table")));
		}
		{
			int			n_stopwords;
			char	  **stopwords;

			n_stopwords = SPI_processed;
			nalloc(stopwords, char *, n_stopwords);
			MemSet(stopwords, 0, sizeof(char *) * n_stopwords);
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
					nfree(stoks[tokidx]);
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
				nfree(sentence_ptrs[i]);
			for (i = 0; i < n_stopwords; i++)
				nfree(stopwords[i]);
			nfree(stopwords);
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
	nfree(method);
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
	bytea	   *result = NULL;
	int			task_len;
	char	   *tmp = NULL;

	initStringInfo(&buf);
	appendBinaryStringInfo(&buf, (char *) &vocab_size, sizeof(int));
	appendBinaryStringInfo(&buf, (char *) &feature_dim, sizeof(int));
	task_len = strlen(task_type);
	appendBinaryStringInfo(&buf, (char *) &task_len, sizeof(int));
	appendBinaryStringInfo(&buf, task_type, task_len);

	total_size = VARHDRSZ + buf.len;
	nalloc(tmp, char, total_size);
	result = (bytea *) tmp;
	NDB_CHECK_ALLOC(result, "result");
	SET_VARSIZE(result, total_size);
	memcpy(VARDATA(result), buf.data, buf.len);
	nfree(buf.data);

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
	TextGpuModelState *state = NULL;
	int			vocab_size = 1000;
	int			feature_dim = 128;
	char		task_type[32] = "classification";
	int			nvec = 0;

	bytea *model_data = NULL;
	Jsonb *metrics = NULL;
	StringInfoData metrics_json;
	JsonbIterator *it = NULL;
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
				{
					strncpy(task_type, v.val.string.val, sizeof(task_type) - 1);
					task_type[sizeof(task_type) - 1] = '\0';
				}
				nfree(key);
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
	nfree(metrics_json.data);

		nalloc(state, TextGpuModelState, 1);
		MemSet(state, 0, sizeof(TextGpuModelState));
	NDB_CHECK_ALLOC(state, "state");
	state->model_blob = model_data;
	state->metrics = metrics;
	state->vocab_size = vocab_size;
	state->feature_dim = feature_dim;
	state->n_samples = nvec;
	strncpy(state->task_type, task_type, sizeof(state->task_type) - 1);
	state->task_type[sizeof(state->task_type) - 1] = '\0';

	if (model->backend_state != NULL)
		nfree(model->backend_state);

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
	Jsonb	   *metrics_json = NULL;
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
	nfree(buf.data);

	if (out != NULL)
		out->payload = metrics_json;

	return true;
}

static bool
text_gpu_serialize(const MLGpuModel *model, bytea * *payload_out,
				   Jsonb * *metadata_out, char **errstr)
{
	const		TextGpuModelState *state;
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
	nalloc(tmp, char, payload_size);
	payload_copy = (bytea *) tmp;
	NDB_CHECK_ALLOC(payload_copy, "payload_copy");
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
text_gpu_deserialize(MLGpuModel *model, const bytea * payload,
					 const Jsonb * metadata, char **errstr)
{
	TextGpuModelState *state = NULL;
	bytea	   *payload_copy = NULL;
	int			payload_size;
	int			vocab_size = 0;
	int			feature_dim = 0;
	char		task_type[32];
	JsonbIterator *it = NULL;
	JsonbValue	v;
	int			r;
	char	   *tmp = NULL;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("text_gpu_deserialize: invalid parameters");
		return false;
	}

	payload_size = VARSIZE(payload);
	nalloc(tmp, char, payload_size);
	payload_copy = (bytea *) tmp;
	NDB_CHECK_ALLOC(payload_copy, "payload_copy");
	memcpy(payload_copy, payload, payload_size);

	if (text_model_deserialize_from_bytea(payload_copy, &vocab_size, &feature_dim, task_type, sizeof(task_type)) != 0)
	{
		nfree(payload_copy);
		if (errstr != NULL)
			*errstr = pstrdup("text_gpu_deserialize: failed to deserialize");
		return false;
	}

		nalloc(state, TextGpuModelState, 1);
		MemSet(state, 0, sizeof(TextGpuModelState));
	NDB_CHECK_ALLOC(state, "state");
	state->model_blob = payload_copy;
	state->vocab_size = vocab_size;
	state->feature_dim = feature_dim;
	state->n_samples = 0;
	strncpy(state->task_type, task_type, sizeof(state->task_type) - 1);
	state->task_type[sizeof(state->task_type) - 1] = '\0';

	if (metadata != NULL)
	{
		int			metadata_size;
		char	   *tmp2 = NULL;
		Jsonb	   *metadata_copy;
		
		metadata_size = VARSIZE(metadata);
		nalloc(tmp2, char, metadata_size);
		metadata_copy = (Jsonb *) tmp2;

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
text_gpu_destroy(MLGpuModel *model)
{
	TextGpuModelState *state = NULL;

	if (model == NULL)
		return;

	if (model->backend_state != NULL)
	{
		state = (TextGpuModelState *) model->backend_state;
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
