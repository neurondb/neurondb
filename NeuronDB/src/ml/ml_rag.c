/*-------------------------------------------------------------------------
 *
 * ml_rag.c
 *    Retrieval-augmented generation pipeline.
 *
 * This module implements text chunking, embedding, ranking, and data
 * transformation for RAG support.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_rag.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/jsonb.h"
#include "utils/lsyscache.h"
#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "access/htup_details.h"
#include "utils/memutils.h"
#include <time.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <ctype.h>

#include "neurondb.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_llm.h"
#include "neurondb_gpu.h"

PG_FUNCTION_INFO_V1(neurondb_chunk_text);
PG_FUNCTION_INFO_V1(neurondb_embed_text);
PG_FUNCTION_INFO_V1(neurondb_rank_documents);
PG_FUNCTION_INFO_V1(neurondb_transform_data);

/*
 * neurondb_chunk_text
 *	  Chunk text for RAG with configurable size, overlap, and separator.
 */
Datum
neurondb_chunk_text(PG_FUNCTION_ARGS)
{
	text	   *input_text;
	int32		chunk_size;
	int32		overlap;
	text	   *separator_text;
	char	   *input_str;
	char	   *separator;
	int			input_len;
	int			chunk_count = 0;
	Datum	   *chunk_datums;
	ArrayType  *result_array;
	int			i,
				start,
				end,
				chunk_len;
	int			max_chunks;

	if (PG_NARGS() < 1 || PG_NARGS() > 4)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb_chunk_text: expected 1-4 arguments, got %d",
						PG_NARGS())));

	input_text = PG_GETARG_TEXT_PP(0);
	chunk_size = PG_ARGISNULL(1) ? 512 : PG_GETARG_INT32(1);
	overlap = PG_ARGISNULL(2) ? 128 : PG_GETARG_INT32(2);
	separator_text = PG_ARGISNULL(3) ? NULL : PG_GETARG_TEXT_PP(3);

	/* Convert input and separator to C string */
	input_str = text_to_cstring(input_text);
	input_len = strlen(input_str);
	separator = separator_text ? text_to_cstring(separator_text)
		: pstrdup("\n\n");

	if (chunk_size <= overlap)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("chunk_size must be greater than "
						"overlap")));

	/* Conservative estimate of number of chunks */
	if (input_len == 0)
	{
		chunk_datums = (Datum *) palloc(sizeof(Datum));
		chunk_datums[0] = CStringGetTextDatum("");
		result_array = construct_array(
									   chunk_datums, 1, TEXTOID, -1, false, TYPALIGN_INT);
		NDB_FREE(chunk_datums);
		PG_RETURN_ARRAYTYPE_P(result_array);
	}

	max_chunks =
		(input_len + chunk_size - overlap - 1) / (chunk_size - overlap);
	if (max_chunks < 1)
		max_chunks = 1;

	chunk_datums = (Datum *) palloc0(sizeof(Datum) * max_chunks);

	start = 0;
	while (start < input_len)
	{
		end = start + chunk_size;
		if (end > input_len)
			end = input_len;

		/* Try to split on the separator, if provided and found */
		if (separator && strlen(separator) > 0 && end < input_len)
		{
			int			sep_pos = -1;

			for (i = end; i > start; i--)
			{
				if (strncmp(input_str + i - strlen(separator),
							separator,
							strlen(separator))
					== 0)
				{
					sep_pos = i;
					break;
				}
			}
			if (sep_pos > start)
				end = sep_pos;
		}

		chunk_len = end - start;
		if (chunk_len <= 0)
			break;

		{
			char	   *chunk_buf = (char *) palloc(chunk_len + 1);

			memcpy(chunk_buf, input_str + start, chunk_len);
			chunk_buf[chunk_len] = '\0';
			chunk_datums[chunk_count++] =
				CStringGetTextDatum(chunk_buf);
			NDB_FREE(chunk_buf);
		}

		if (end == input_len)
			break;

		start = end - overlap;
		if (start < 0)
			start = 0;
	}
	result_array = construct_array(
								   chunk_datums, chunk_count, TEXTOID, -1, false, TYPALIGN_INT);
	NDB_FREE(chunk_datums);
	if (separator_text)
		NDB_FREE(separator);
	NDB_FREE(input_str);
	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * neurondb_embed_text
 *	  Generate vector embeddings for text using actual embedding API or GPU implementation.
 */
Datum
neurondb_embed_text(PG_FUNCTION_ARGS)
{
	text	   *model_text;
	text	   *input_text;
	bool		use_gpu;

	char	   *model_name;
	char	   *input_str;
	NdbLLMConfig cfg;
	NdbLLMCallOptions call_opts;
	NDB_DECLARE(float *, vec_data);
	NDB_DECLARE(Vector *, result);
	int			dim = 0;

	if (PG_NARGS() < 2 || PG_NARGS() > 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb_embed_text: expected 2-3 arguments, got %d",
						PG_NARGS())));

	model_text = PG_GETARG_TEXT_PP(0);
	input_text = PG_GETARG_TEXT_PP(1);
	use_gpu = PG_ARGISNULL(2) ? true : PG_GETARG_BOOL(2);

	model_name = text_to_cstring(model_text);
	input_str = text_to_cstring(input_text);

	elog(DEBUG1,
		 "neurondb_embed_text: model='%s', use_gpu=%d",
		 model_name,
		 use_gpu);

	/* Setup LLM config for embedding */
	memset(&cfg, 0, sizeof(cfg));
	cfg.provider = (neurondb_llm_provider != NULL) ? neurondb_llm_provider : "huggingface";
	cfg.endpoint = (neurondb_llm_endpoint != NULL) ?
		neurondb_llm_endpoint :
		"https://api-inference.huggingface.co";
	cfg.model = model_name != NULL ? model_name :
		(neurondb_llm_model != NULL ?
		 neurondb_llm_model :
		 "sentence-transformers/all-MiniLM-L6-v2");
	cfg.api_key = neurondb_llm_api_key;
	cfg.timeout_ms = neurondb_llm_timeout_ms;
	cfg.prefer_gpu = use_gpu && NDB_SHOULD_TRY_GPU();
	cfg.require_gpu = false;

	call_opts.task = "embed";
	call_opts.prefer_gpu = cfg.prefer_gpu;
	call_opts.require_gpu = cfg.require_gpu;
	call_opts.fail_open = neurondb_llm_fail_open;

	/* Call actual embedding API */
	if (ndb_llm_route_embed(&cfg, &call_opts, input_str, &vec_data, &dim) != 0 ||
		vec_data == NULL || dim <= 0)
	{
		NDB_FREE(model_name);
		NDB_FREE(input_str);
		ereport(ERROR,
				(errcode(ERRCODE_EXTERNAL_ROUTINE_INVOCATION_EXCEPTION),
				 errmsg("neurondb_embed_text: embedding generation failed"),
				 errdetail("Real model implementation required. Embedding generation failed for model '%s'", model_name)));
	}

	/* Create result vector */
	result = (Vector *) palloc(VARHDRSZ + sizeof(int16) * 2 + dim * sizeof(float));
	SET_VARSIZE(result, VARHDRSZ + sizeof(int16) * 2 + dim * sizeof(float));
	result->dim = dim;
	result->unused = 0;
	memcpy(result->data, vec_data, dim * sizeof(float));

	NDB_FREE(vec_data);
	NDB_FREE(model_name);
	NDB_FREE(input_str);

	PG_RETURN_POINTER(result);
}

/*
 * neurondb_rank_documents
 *	  Rank (rerank) array of documents based on query using simple heuristics
 *	  (supports: bm25, cosine, edit_distance ranking method names for demonstration).
 */
static float
simple_bm25(const char *query, const char *doc)
{
	/*
	 * Naive: score = matches of every query word in doc, normalized.
	 */
	float		score = 0.0f;
	int			total = 0;
	char	   *doc_lc = pstrdup(doc);
	char	   *query_lc = pstrdup(query);
	char	   *tok,
			   *saveptr_q;
	bool		found = false;

	/* Lowercase for case-insensitive matching */
	for (int i = 0; doc_lc[i]; i++)
		doc_lc[i] = tolower(doc_lc[i]);
	for (int i = 0; query_lc[i]; i++)
		query_lc[i] = tolower(query_lc[i]);

	for (tok = strtok_r(query_lc, " \t\r\n", &saveptr_q); tok != NULL;
		 tok = strtok_r(NULL, " \t\r\n", &saveptr_q))
	{
		total++;
		found = false;
		if (strstr(doc_lc, tok) != NULL)
			found = true;
		if (found)
			score += 1.0;
	}
	if (total > 0)
		score /= total;

	NDB_FREE(doc_lc);
	NDB_FREE(query_lc);
	return score + 0.5f;		/* Base for sort stability */
}

static float
simple_cosine(const char *query, const char *doc)
{
	/*
	 * Naive "cosine": overlap count for tokens.
	 */
	char	   *doc_lc = pstrdup(doc);
	char	   *query_lc = pstrdup(query);
	int			score = 0;
	int			qcount = 0;
	char	   *tok_q,
			   *saveptr_q;
	char	   *tok_d,
			   *saveptr_d;
	size_t		doc_len;

	for (int i = 0; doc_lc[i]; i++)
		doc_lc[i] = tolower(doc_lc[i]);
	for (int i = 0; query_lc[i]; i++)
		query_lc[i] = tolower(query_lc[i]);
	for (tok_q = strtok_r(query_lc, " \t\r\n", &saveptr_q); tok_q != NULL;
		 tok_q = strtok_r(NULL, " \t\r\n", &saveptr_q))
	{
		qcount++;
		/* Here, check if token appears in doc */
		for (tok_d = strtok_r(doc_lc, " \t\r\n", &saveptr_d);
			 tok_d != NULL;
			 tok_d = strtok_r(NULL, " \t\r\n", &saveptr_d))
		{
			if (strcmp(tok_d, tok_q) == 0)
			{
				score++;
				break;
			}
		}
		/* important: re-initialize for next q token */
		/* doc_lc was allocated with pstrdup(doc) so it has enough space */
		/* Restore original doc string (strtok_r modifies the string) */
		doc_len = strlen(doc);
		/* pstrdup allocates strlen(doc) + 1, so we have enough space */
		strncpy(doc_lc, doc, doc_len + 1);
		doc_lc[doc_len] = '\0';
		for (int i = 0; doc_lc[i]; i++)
			doc_lc[i] = tolower(doc_lc[i]);
	}
	if (qcount == 0)
		qcount = 1;
	NDB_FREE(doc_lc);
	NDB_FREE(query_lc);
	return (float) score / qcount + 0.5f;	/* stabilization */
}

static int
levenshtein(const char *s1, const char *s2)
{
	int			len1 = strlen(s1),
				len2 = strlen(s2);
	int		   *v0 = (int *) palloc(sizeof(int) * (len2 + 1));
	int		   *v1 = (int *) palloc(sizeof(int) * (len2 + 1));

	int			i,
				j,
				cost,
				min1,
				min2,
				min3;

	for (j = 0; j <= len2; j++)
		v0[j] = j;
	for (i = 0; i < len1; i++)
	{
		v1[0] = i + 1;
		for (j = 0; j < len2; j++)
		{
			cost = (tolower(s1[i]) == tolower(s2[j])) ? 0 : 1;
			min1 = v1[j] + 1;
			min2 = v0[j + 1] + 1;
			min3 = v0[j] + cost;
			if (min1 > min2)
				min1 = min2;
			if (min1 > min3)
				min1 = min3;
			v1[j + 1] = min1;
		}
		memcpy(v0, v1, sizeof(int) * (len2 + 1));
	}
	{
		int			result;

		result = v1[len2];
		NDB_FREE(v0);
		NDB_FREE(v1);
		return result;
	}
}

static float
simple_edit_distance(const char *query, const char *doc)
{
	/*
	 * Return normalized similarity based on edit distance: similarity = 1.0 -
	 * (edit_distance / max(len1, len2))
	 */
	int			lenq = strlen(query);
	int			lend;
	int			maxl;
	int			edit;

	lend = strlen(doc);
	maxl = (lenq > lend) ? lenq : lend;
	if (maxl == 0)
		return 1.0f;
	edit = levenshtein(query, doc);
	return 1.0f - ((float) edit / (float) maxl);
}

typedef struct
{
	char	   *doc;
	float		score;
	int			rawidx;
}			doc_with_score;

static int
doc_with_score_cmp(const void *a, const void *b)
{
	const		doc_with_score *da = (const doc_with_score *) a;
	const		doc_with_score *db = (const doc_with_score *) b;

	if (da->score > db->score)
		return -1;
	if (da->score < db->score)
		return 1;
	return da->rawidx - db->rawidx;
}

Datum
neurondb_rank_documents(PG_FUNCTION_ARGS)
{
	text	   *query_text;
	ArrayType  *documents_array;
	text	   *algorithm_text;

	char	   *query;
	char	   *algorithm;
	int			ndims,
				nelems,
			   *dims;
	Datum	   *elem_values;
	bool	   *elem_nulls;
	doc_with_score *ranklist;
	int			i,
				limit = 10,
				count = 0;
	StringInfoData json_result;

	if (PG_NARGS() < 2 || PG_NARGS() > 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb_rank_documents: expected 2-3 arguments, got %d",
						PG_NARGS())));

	query_text = PG_GETARG_TEXT_PP(0);
	documents_array = PG_GETARG_ARRAYTYPE_P(1);
	algorithm_text = PG_ARGISNULL(2) ? NULL : PG_GETARG_TEXT_PP(2);

	query = text_to_cstring(query_text);
	algorithm = algorithm_text ? text_to_cstring(algorithm_text)
		: pstrdup("bm25");

	ndims = ARR_NDIM(documents_array);
	dims = ARR_DIMS(documents_array);
	nelems = ArrayGetNItems(ndims, dims);

	deconstruct_array(documents_array,
					  TEXTOID,
					  -1,
					  false,
					  TYPALIGN_INT,
					  &elem_values,
					  &elem_nulls,
					  &nelems);
	ranklist = (doc_with_score *) palloc0(nelems * sizeof(doc_with_score));
	for (i = 0; i < nelems; i++)
	{
		if (elem_nulls[i])
		{
			ranklist[i].doc = NULL;
			ranklist[i].score = -FLT_MAX;
			ranklist[i].rawidx = i;
			continue;
		}
		ranklist[i].doc = TextDatumGetCString(elem_values[i]);
		if (strcmp(algorithm, "bm25") == 0)
			ranklist[i].score = simple_bm25(query, ranklist[i].doc);
		else if (strcmp(algorithm, "cosine") == 0)
			ranklist[i].score =
				simple_cosine(query, ranklist[i].doc);
		else if (strcmp(algorithm, "edit_distance") == 0)
			ranklist[i].score =
				simple_edit_distance(query, ranklist[i].doc);
		else
			ranklist[i].score = simple_bm25(query, ranklist[i].doc);
		ranklist[i].rawidx = i;
	}
	qsort(ranklist, nelems, sizeof(doc_with_score), doc_with_score_cmp);

	initStringInfo(&json_result);
	appendStringInfoString(&json_result, "{\"ranked\": [");

	count = 0;
	for (i = 0; i < nelems && count < limit; i++)
	{
		if (ranklist[i].doc)
		{
			if (count != 0)
				appendStringInfoString(&json_result, ", ");
			appendStringInfo(&json_result,
							 "{\"document\": %s, \"score\": %.4f, \"rank\": %d}",
							 quote_literal_cstr(ranklist[i].doc),
							 ranklist[i].score,
							 count + 1);
			count++;
		}
	}
	appendStringInfoString(&json_result, "]}");

	/* cleanup */
	for (i = 0; i < nelems; i++)
	{
		if (ranklist[i].doc)
			NDB_FREE(ranklist[i].doc);
	}
	NDB_FREE(ranklist);
	if (algorithm_text)
		NDB_FREE(algorithm);
	NDB_FREE(query);

	PG_RETURN_JSONB_P(DatumGetJsonbP(DirectFunctionCall1(
														 jsonb_in, CStringGetTextDatum(json_result.data))));
}

/*
 * neurondb_transform_data
 *	  Apply transformation (normalize, standardize, min_max) to float8 array.
 */

Datum
neurondb_transform_data(PG_FUNCTION_ARGS)
{
	text	   *pipeline_text;
	ArrayType  *input_array;

	char	   *pipeline_name;
	int			ndims,
				nelems,
			   *dims;
	Oid			element_type;
	NDB_DECLARE(Datum *, elem_values);
	NDB_DECLARE(bool *, elem_nulls);
	float8	   *transformed_data;
	Datum	   *result_datums;
	ArrayType  *result_array;
	int			i;
	float8		sum = 0.0,
				sum_sq = 0.0,
				min_v = DBL_MAX,
				max_v = -DBL_MAX,
				mean,
				stddev,
				range;

	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb_transform_data: expected 2 arguments, got %d",
						PG_NARGS())));

	pipeline_text = PG_GETARG_TEXT_PP(0);
	input_array = PG_GETARG_ARRAYTYPE_P(1);

	pipeline_name = text_to_cstring(pipeline_text);

	ndims = ARR_NDIM(input_array);
	dims = ARR_DIMS(input_array);
	nelems = ArrayGetNItems(ndims, dims);
	element_type = ARR_ELEMTYPE(input_array);

	deconstruct_array(input_array,
					  element_type,
					  sizeof(float8),
					  FLOAT8PASSBYVAL,
					  'd',
					  &elem_values,
					  &elem_nulls,
					  &nelems);

	if (nelems == 0)
	{
		result_array = construct_empty_array(FLOAT8OID);
		PG_RETURN_ARRAYTYPE_P(result_array);
	}

	/* Stats for transformation */
	for (i = 0; i < nelems; i++)
	{
		if (!elem_nulls[i])
		{
			float8		x = DatumGetFloat8(elem_values[i]);

			sum += x;
			sum_sq += x * x;
			if (x < min_v)
				min_v = x;
			if (x > max_v)
				max_v = x;
		}
	}
	mean = sum / nelems;
	stddev = sqrt((sum_sq / nelems) - (mean * mean));

	transformed_data = (float8 *) palloc0(sizeof(float8) * nelems);
	result_datums = (Datum *) palloc0(sizeof(Datum) * nelems);

	if (strcmp(pipeline_name, "normalize") == 0)
	{
		/* L2 normalization */
		float8		norm = sqrt(sum_sq);

		if (norm <= 0.0)
		{
			for (i = 0; i < nelems; i++)
				transformed_data[i] = 0.0;
		}
		else
		{
			for (i = 0; i < nelems; i++)
				transformed_data[i] = elem_nulls[i]
					? 0.0
					: DatumGetFloat8(elem_values[i]) / norm;
		}
	}
	else if (strcmp(pipeline_name, "standardize") == 0)
	{
		/* Z-score */
		if (stddev > 0.0)
		{
			for (i = 0; i < nelems; i++)
				transformed_data[i] = elem_nulls[i]
					? 0.0
					: (DatumGetFloat8(elem_values[i])
					   - mean)
					/ stddev;
		}
		else
		{
			for (i = 0; i < nelems; i++)
				transformed_data[i] = 0.0;
		}
	}
	else if (strcmp(pipeline_name, "min_max") == 0)
	{
		/* Min-max scaling [0, 1] */
		range = max_v - min_v;
		if (range > 0.0)
		{
			for (i = 0; i < nelems; i++)
				transformed_data[i] = elem_nulls[i]
					? 0.0
					: (DatumGetFloat8(elem_values[i])
					   - min_v)
					/ range;
		}
		else
		{
			for (i = 0; i < nelems; i++)
				transformed_data[i] = 0.5;
		}
	}
	else
	{
		NDB_FREE(transformed_data);
		NDB_FREE(result_datums);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("unsupported transformation pipeline: "
						"\"%s\"",
						pipeline_name),
				 errhint("Supported pipelines: normalize, "
						 "standardize, min_max")));
	}

	for (i = 0; i < nelems; i++)
		result_datums[i] = Float8GetDatum(transformed_data[i]);

	result_array = construct_array(result_datums,
								   nelems,
								   FLOAT8OID,
								   sizeof(float8),
								   FLOAT8PASSBYVAL,
								   'd');

	NDB_FREE(result_datums);
	NDB_FREE(transformed_data);
	NDB_FREE(pipeline_name);

	PG_RETURN_ARRAYTYPE_P(result_array);
}
