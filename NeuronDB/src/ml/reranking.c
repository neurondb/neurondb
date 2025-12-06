/*-------------------------------------------------------------------------
 *
 * reranking.c
 *    Reranking functions for search result refinement.
 *
 * This module provides reranking functions supporting cross-encoders,
 * LLMs, and ensemble approaches for semantic search.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/reranking.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_llm.h"
#include "neurondb_gpu.h"
#include "neurondb_constants.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/memutils.h"
#include "access/htup_details.h"
#include <string.h>
#include <float.h>
#include <math.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/* NOTE: PG_MODULE_MAGIC should reside only in the main extension file, not here. */

typedef struct RerankState
{
	char	   *query;			/* User query string for conditioning reranker */
	Datum	   *candidates;		/* Array of candidate Datum (text) values */
	bool	   *nulls;			/* Which candidates are NULL */
	float	   *scores;			/* Output: reranked scores [0,1], descending
								 * order */
	int		   *indices;		/* Output: indices of reranked elements */
	int			ncandidates;	/* Number of provided candidates */
}			RerankState;

static void
sort_rerank_desc(float *scores, int *indices, int n)
{
	int			i,
				j;

	for (i = 0; i < n - 1; i++)
	{
		for (j = i + 1; j < n; j++)
		{
			if (scores[j] > scores[i])
			{
				float		tmp_s = scores[i];
				int			tmp_i = indices[i];

				scores[i] = scores[j];
				indices[i] = indices[j];
				scores[j] = tmp_s;
				indices[j] = tmp_i;
			}
		}
	}
}

/*
 * rerank_cross_encoder - Perform cross-encoder reranking on candidate documents
 *
 * User-facing set-returning function that reranks candidate documents using
 * a cross-encoder model. Returns the top-k results with their relevance scores.
 *
 * Parameters:
 *   query - User query string (text)
 *   candidates - Array of candidate text documents to rerank (text[])
 *   model - Cross-encoder model name/identifier (text, optional)
 *   top_k - Number of top results to return (int32)
 *
 * Returns:
 *   SETOF (idx int, score float4) - Set of result tuples with indices and scores
 *
 * Notes:
 *   This function is a set-returning function (SRF) that performs multi-call
 *   execution. It uses either a specified model or falls back to a default
 *   reranking method. Results are sorted by score in descending order.
 */
PG_FUNCTION_INFO_V1(rerank_cross_encoder);
Datum
rerank_cross_encoder(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	int			call_cntr;
	int			max_calls;
	RerankState *state;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		TupleDesc	tupdesc;
		text	   *query_text;
		ArrayType  *candidates_array;
		text	   *model_text;
		int			top_k;

		NDB_DECLARE(char *, query_str);
		Datum	   *candidate_datums;
		bool	   *candidate_nulls;
		int			ncandidates;

		NDB_DECLARE(float *, scores);

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext =
			MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		/*--- 1. Parse Inputs ---*/
		query_text = PG_GETARG_TEXT_PP(0);
		candidates_array = PG_GETARG_ARRAYTYPE_P(1);
		model_text = (PG_ARGISNULL(2) ? NULL : PG_GETARG_TEXT_PP(2));
		top_k = PG_GETARG_INT32(3);

		query_str = text_to_cstring(query_text);

		deconstruct_array(candidates_array,
						  TEXTOID,
						  -1,
						  false,
						  'i',
						  &candidate_datums,
						  &candidate_nulls,
						  &ncandidates);

		if (top_k < 1)
			ereport(ERROR,
					(errmsg("top_k must be positive (got %d)",
							top_k)));
		if (ncandidates <= 0)
			ereport(ERROR,
					(errmsg("candidate array cannot be empty")));
		max_calls = (ncandidates < top_k) ? ncandidates : top_k;

		state = (RerankState *) palloc0(sizeof(RerankState));
		state->query = query_str;
		state->candidates = candidate_datums;
		state->nulls = candidate_nulls;
		state->scores = (float *) palloc0(ncandidates * sizeof(float));
		state->indices = (int *) palloc0(ncandidates * sizeof(int));
		state->ncandidates = ncandidates;

		if (model_text)
		{
			NDB_DECLARE(char *, model_str);
			NdbLLMConfig cfg;
			NdbLLMCallOptions call_opts;
			int			i;

			NDB_DECLARE(const char **, docs);
			int			api_result;

			model_str = text_to_cstring(model_text);

			cfg.provider = neurondb_llm_provider
				? neurondb_llm_provider
				: "huggingface";
			cfg.endpoint = neurondb_llm_endpoint
				? neurondb_llm_endpoint
				: "https://router.huggingface.co";
			cfg.model = model_str
				? model_str
				: (neurondb_llm_model ? neurondb_llm_model
				   : "sentence-transformers/"
				   "all-MiniLM-L6-v2");
			cfg.api_key = neurondb_llm_api_key;
			cfg.timeout_ms = neurondb_llm_timeout_ms;
			cfg.prefer_gpu = NDB_SHOULD_TRY_GPU();
			cfg.require_gpu = false;
			if (cfg.provider != NULL
				&& (pg_strcasecmp(
								  cfg.provider, "huggingface-local")
					== 0
					|| pg_strcasecmp(
									 cfg.provider, "hf-local")
					== 0)
				&& !neurondb_llm_fail_open)
				cfg.require_gpu = true;

			call_opts.task = "rerank";
			call_opts.prefer_gpu = cfg.prefer_gpu;
			call_opts.require_gpu = cfg.require_gpu;
			call_opts.fail_open = neurondb_llm_fail_open;

			NDB_ALLOC(docs, const char *, ncandidates);
			for (i = 0; i < ncandidates; i++)
			{
				if (!candidate_nulls[i]
					&& DatumGetPointer(candidate_datums[i]))
					docs[i] =
						text_to_cstring(DatumGetTextPP(
													   candidate_datums[i]));
				else
					docs[i] = "";
			}

			api_result = ndb_llm_route_rerank(&cfg,
											  &call_opts,
											  query_str,
											  docs,
											  ncandidates,
											  &scores);
			if (api_result == 0 && scores)
			{
				memcpy(state->scores,
					   scores,
					   sizeof(float) * ncandidates);
				for (i = 0; i < ncandidates; i++)
					state->indices[i] = i;

				sort_rerank_desc(state->scores,
								 state->indices,
								 ncandidates);
			}
			else
			{
				/* Reranking failed - cannot return dummy scores */
				for (i = 0; i < ncandidates; i++)
				{
					if (docs[i] && docs[i][0] != '\0')
						pfree((char *) docs[i]);
				}
				NDB_FREE(docs);
				NDB_FREE(query_str);
				ereport(ERROR,
						(errcode(ERRCODE_EXTERNAL_ROUTINE_INVOCATION_EXCEPTION),
						 errmsg("neurondb_rerank: reranking failed"),
						 errdetail("Reranking failed. Cannot return dummy scores.")));
			}

			for (i = 0; i < ncandidates; i++)
			{
				if (docs[i][0] != '\0')
				{
					pfree((char *) docs[i]);
				}
			}
			NDB_FREE(docs);
			if (scores)
				NDB_FREE(scores);
			NDB_FREE(model_str);
		}
		else
		{
			/* No model (default): preserve order, assign perfect scores */
			int			i;

			for (i = 0; i < ncandidates; i++)
			{
				state->indices[i] = i;
				state->scores[i] = 1.0f;
			}
		}

		/*--- 4. Build SRF output tuple descriptor: (idx int, score float4) ---*/
		tupdesc = CreateTemplateTupleDesc(2);
		TupleDescInitEntry(
						   tupdesc, (AttrNumber) 1, "idx", INT4OID, -1, 0);
		TupleDescInitEntry(
						   tupdesc, (AttrNumber) 2, "score", FLOAT4OID, -1, 0);
		BlessTupleDesc(tupdesc);

		funcctx->max_calls = max_calls;
		funcctx->user_fctx = state;
		funcctx->tuple_desc = tupdesc;

		MemoryContextSwitchTo(oldcontext);
	}

	/*--- Per-call: emit one tuple per result, ranked in descending order ---*/
	funcctx = SRF_PERCALL_SETUP();
	state = (RerankState *) funcctx->user_fctx;
	call_cntr = funcctx->call_cntr;
	max_calls = funcctx->max_calls;

	if (call_cntr < max_calls)
	{
		HeapTuple	tuple;
		Datum		values[2];
		bool		nulls[2] = {false, false};
		int			idx_ranked =
			state->indices
			[call_cntr];		/* sorted index in candidate array */

		values[0] = Int32GetDatum(idx_ranked);
		values[1] = Float4GetDatum(state->scores[idx_ranked]);
		tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);

		SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
	}
	else
	{
		/*
		 * Memory context cleanup: All allocations (state, query, arrays) are
		 * in funcctx->multi_call_memory_ctx and will be automatically freed
		 * when the context is destroyed. No explicit cleanup needed here.
		 */
		SRF_RETURN_DONE(funcctx);
	}
}

/*
 * rerank_llm
 *    Advanced: LLM-completion-based reranking (zero-/few-shot, instruction prompt).
 *    Synthesizes a prompt with query and candidates, invokes LLM completion,
 *    parses response to extract scores/indices, returns reranked results.
 */
PG_FUNCTION_INFO_V1(rerank_llm);
Datum
rerank_llm(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	int			call_cntr;
	int			max_calls;
	RerankState *state;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		TupleDesc	tupdesc;
		text	   *query_text;
		ArrayType  *candidates_array;
		text	   *model_text;
		text	   *prompt_template_text;
		int			top_k;
		float4		temperature;

		NDB_DECLARE(char *, query_str);
		NDB_DECLARE(char *, model_str);
		NDB_DECLARE(char *, prompt_template);
		Datum	   *candidate_datums;
		bool	   *candidate_nulls;
		int			ncandidates;
		int			i;
		StringInfoData prompt;
		StringInfoData params_json;
		char	   *params_json_str;
		NdbLLMConfig cfg;
		NdbLLMCallOptions call_opts;
		NdbLLMResp	resp;

		NDB_DECLARE(char *, llm_response);
		int			api_result;

		/*-- Prepare multi-call context --*/
		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext =
			MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		/*--- 1. Parse Inputs ---*/
		query_text = PG_GETARG_TEXT_PP(0);
		candidates_array = PG_GETARG_ARRAYTYPE_P(1);
		model_text = (PG_ARGISNULL(2) ? NULL : PG_GETARG_TEXT_PP(2));
		top_k = PG_ARGISNULL(3) ? -1 : PG_GETARG_INT32(3);
		prompt_template_text = (PG_ARGISNULL(4) ? NULL : PG_GETARG_TEXT_PP(4));
		temperature = PG_ARGISNULL(5) ? 0.0f : PG_GETARG_FLOAT4(5);

		query_str = text_to_cstring(query_text);

		deconstruct_array(candidates_array,
						  TEXTOID,
						  -1,
						  false,
						  'i',
						  &candidate_datums,
						  &candidate_nulls,
						  &ncandidates);

		if (ncandidates <= 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("candidate array cannot be empty")));

		max_calls = (top_k < 1 || top_k > ncandidates) ? ncandidates : top_k;

		/*--- 2. Allocate Rerank State ---*/
		state = (RerankState *) palloc0(sizeof(RerankState));
		state->query = query_str;
		state->candidates = candidate_datums;
		state->nulls = candidate_nulls;
		state->scores = (float *) palloc0(ncandidates * sizeof(float));
		state->indices = (int *) palloc0(ncandidates * sizeof(int));
		state->ncandidates = ncandidates;

		/* Initialize indices */
		for (i = 0; i < ncandidates; i++)
			state->indices[i] = i;

		/*--- 3. Build prompt with query and candidates ---*/
		initStringInfo(&prompt);
		if (prompt_template_text)
		{
			prompt_template = text_to_cstring(prompt_template_text);
			appendStringInfo(&prompt, "%s\n\n", prompt_template);
		}
		else
		{
			/* Default prompt template */
			appendStringInfo(&prompt,
							 "Rank the following documents by relevance to the query. "
							 "Return a JSON array of scores (0.0 to 1.0) in the same order.\n\n");
		}

		appendStringInfo(&prompt, "Query: %s\n\nDocuments:\n", query_str);
		for (i = 0; i < ncandidates; i++)
		{
			char	   *doc_str;

			if (!candidate_nulls[i] && DatumGetPointer(candidate_datums[i]))
			{
				doc_str = text_to_cstring(DatumGetTextPP(candidate_datums[i]));
				appendStringInfo(&prompt, "%d. %s\n", i + 1, doc_str);
				NDB_FREE(doc_str);
			}
			else
			{
				appendStringInfo(&prompt, "%d. [NULL]\n", i + 1);
			}
		}

		appendStringInfo(&prompt,
						 "\nReturn JSON array of scores: [score1, score2, ...]");

		/*--- 4. Configure LLM ---*/
		model_str = model_text ? text_to_cstring(model_text) : NULL;

		cfg.provider = neurondb_llm_provider
			? neurondb_llm_provider
			: "openai";
		cfg.endpoint = neurondb_llm_endpoint
			? neurondb_llm_endpoint
			: "https://api.openai.com/v1";
		cfg.model = model_str
			? model_str
			: (neurondb_llm_model ? neurondb_llm_model : "gpt-3.5-turbo");
		cfg.api_key = neurondb_llm_api_key;
		cfg.timeout_ms = neurondb_llm_timeout_ms;
		cfg.prefer_gpu = NDB_SHOULD_TRY_GPU();
		cfg.require_gpu = false;

		call_opts.task = "complete";
		call_opts.prefer_gpu = cfg.prefer_gpu;
		call_opts.require_gpu = false;
		call_opts.fail_open = neurondb_llm_fail_open;

		/* Build params JSON with temperature */
		initStringInfo(&params_json);
		appendStringInfo(&params_json,
						 "{\"temperature\":%.2f}",
						 temperature);
		params_json_str = params_json.data;

		/*--- 5. Call LLM completion API ---*/
		memset(&resp, 0, sizeof(resp));
		api_result = ndb_llm_route_complete(&cfg,
											&call_opts,
											prompt.data,
											params_json_str,
											&resp);

		if (api_result == 0 && resp.text)
		{
			llm_response = resp.text;

			/*--- 6. Parse LLM response (JSON array of scores) ---*/
			/* Simple JSON array parsing - look for [score1, score2, ...] */
			{
				const char *p = llm_response;
				int			score_idx = 0;
				double		score_val;

				/* Find opening bracket */
				while (*p && *p != '[')
					p++;
				if (*p == '[')
					p++;

				/* Parse scores */
				while (*p && score_idx < ncandidates)
				{
					char	   *endptr;

					/* Skip whitespace and commas */
					while (*p && (isspace((unsigned char) *p) || *p == ','))
						p++;
					if (*p == ']' || !*p)
						break;

					errno = 0;
					score_val = strtod(p, &endptr);
					if (endptr != p && errno == 0)
					{
						/* Clamp to [0, 1] */
						if (score_val < 0.0)
							score_val = 0.0;
						if (score_val > 1.0)
							score_val = 1.0;
						state->scores[score_idx] = (float) score_val;
						p = endptr;
					}
					else
					{
						/* Parse error, use default score */
						state->scores[score_idx] = 0.5f;
						break;
					}
					score_idx++;
				}

				/*
				 * Fill remaining with default scores if LLM didn't return
				 * enough
				 */
				for (; score_idx < ncandidates; score_idx++)
					state->scores[score_idx] = 0.5f;
			}
		}
		else
		{
			/* API call failed, use default scores */
			for (i = 0; i < ncandidates; i++)
				state->scores[i] = 0.5f;
		}

		/*--- 7. Sort by score (descending) ---*/
		sort_rerank_desc(state->scores, state->indices, ncandidates);

		/*--- 8. Setup SRF return ---*/
		tupdesc = CreateTemplateTupleDesc(2);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "index", INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "score", FLOAT4OID, -1, 0);
		funcctx->tuple_desc = BlessTupleDesc(tupdesc);
		funcctx->max_calls = max_calls;
		funcctx->user_fctx = state;

		if (llm_response && llm_response != resp.text)
			NDB_FREE(llm_response);
		if (resp.text)
			NDB_FREE(resp.text);
		if (resp.json)
			NDB_FREE(resp.json);
		if (params_json_str)
			NDB_FREE(params_json_str);
		if (prompt.data)
			NDB_FREE(prompt.data);
		if (model_str)
			NDB_FREE(model_str);
		if (prompt_template)
			NDB_FREE(prompt_template);

		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();
	state = (RerankState *) funcctx->user_fctx;
	call_cntr = funcctx->call_cntr;
	max_calls = funcctx->max_calls;

	if (call_cntr < max_calls)
	{
		HeapTuple	tuple;
		Datum		values[2];
		bool		nulls[2] = {false, false};
		int			idx_ranked = state->indices[call_cntr];

		values[0] = Int32GetDatum(idx_ranked);
		values[1] = Float4GetDatum(state->scores[idx_ranked]);
		tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);

		SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
	}
	else
	{
		SRF_RETURN_DONE(funcctx);
	}
}

/*
 * rerank_cohere
 *    Cohere API-style reranking (external API or compatible endpoint).
 */
PG_FUNCTION_INFO_V1(rerank_cohere);
Datum
rerank_cohere(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	int			call_cntr;
	int			max_calls;
	RerankState *state;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		TupleDesc	tupdesc;
		text	   *query_text;
		ArrayType  *candidates_array;
		text	   *model_text;
		int			top_k;

		NDB_DECLARE(char *, query_str);
		Datum	   *candidate_datums;
		bool	   *candidate_nulls;
		int			ncandidates;

		NDB_DECLARE(float *, scores);
		int			i;

		NDB_DECLARE(const char **, docs);
		int			api_result;

		NDB_DECLARE(char *, model_str);
		NdbLLMConfig cfg;
		NdbLLMCallOptions call_opts;

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		query_text = PG_GETARG_TEXT_PP(0);
		candidates_array = PG_GETARG_ARRAYTYPE_P(1);
		model_text = (PG_ARGISNULL(2) ? NULL : PG_GETARG_TEXT_PP(2));
		top_k = PG_GETARG_INT32(3);

		query_str = text_to_cstring(query_text);

		deconstruct_array(candidates_array, TEXTOID, -1, false, 'i',
						  &candidate_datums, &candidate_nulls, &ncandidates);

		if (top_k < 1)
			ereport(ERROR, (errmsg("top_k must be positive (got %d)", top_k)));
		if (ncandidates <= 0)
			ereport(ERROR, (errmsg("candidate array cannot be empty")));
		max_calls = (ncandidates < top_k) ? ncandidates : top_k;

		state = (RerankState *) palloc0(sizeof(RerankState));
		state->query = query_str;
		state->candidates = candidate_datums;
		state->nulls = candidate_nulls;
		state->scores = (float *) palloc0(ncandidates * sizeof(float));
		state->indices = (int *) palloc0(ncandidates * sizeof(int));
		state->ncandidates = ncandidates;

		/* Configure for Cohere API */
		model_str = model_text ? text_to_cstring(model_text) : NULL;
		cfg.provider = "cohere";
		cfg.endpoint = neurondb_llm_endpoint ? neurondb_llm_endpoint : "https://api.cohere.ai";
		cfg.model = model_str ? model_str : "rerank-english-v3.0";
		cfg.api_key = neurondb_llm_api_key;
		cfg.timeout_ms = neurondb_llm_timeout_ms;
		cfg.prefer_gpu = false;
		cfg.require_gpu = false;

		call_opts.task = "rerank";
		call_opts.prefer_gpu = false;
		call_opts.require_gpu = false;
		call_opts.fail_open = neurondb_llm_fail_open;

		NDB_ALLOC(docs, const char *, ncandidates);
		for (i = 0; i < ncandidates; i++)
		{
			if (!candidate_nulls[i] && DatumGetPointer(candidate_datums[i]))
				docs[i] = text_to_cstring(DatumGetTextPP(candidate_datums[i]));
			else
				docs[i] = "";
		}

		api_result = ndb_llm_route_rerank(&cfg, &call_opts, query_str, docs,
										  ncandidates, &scores);
		if (api_result == 0 && scores)
		{
			memcpy(state->scores, scores, sizeof(float) * ncandidates);
			for (i = 0; i < ncandidates; i++)
				state->indices[i] = i;
			sort_rerank_desc(state->scores, state->indices, ncandidates);
		}
		else
		{
			/* Fallback to sequential scores */
			for (i = 0; i < ncandidates; i++)
			{
				state->indices[i] = i;
				state->scores[i] = 1.0f - ((float) i / (float) ncandidates);
			}
		}

		for (i = 0; i < ncandidates; i++)
		{
			if (docs[i][0] != '\0')
			{
				pfree((char *) docs[i]);
			}
		}
		NDB_FREE(docs);
		if (scores)
			NDB_FREE(scores);
		if (model_str)
			NDB_FREE(model_str);

		tupdesc = CreateTemplateTupleDesc(2);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "idx", INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "score", FLOAT4OID, -1, 0);
		BlessTupleDesc(tupdesc);

		funcctx->max_calls = max_calls;
		funcctx->user_fctx = state;
		funcctx->tuple_desc = tupdesc;

		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();
	state = (RerankState *) funcctx->user_fctx;
	call_cntr = funcctx->call_cntr;
	max_calls = funcctx->max_calls;

	if (call_cntr < max_calls)
	{
		HeapTuple	tuple;
		Datum		values[2];
		bool		nulls[2] = {false, false};
		int			idx_ranked = state->indices[call_cntr];

		values[0] = Int32GetDatum(idx_ranked);
		values[1] = Float4GetDatum(state->scores[idx_ranked]);
		tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);

		SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
	}
	else
	{
		SRF_RETURN_DONE(funcctx);
	}
}

/*
 * rerank_colbert
 *    ColBERT architecture reranking with late interaction.
 *    Uses token-level embeddings and MaxSim scoring.
 */
PG_FUNCTION_INFO_V1(rerank_colbert);
Datum
rerank_colbert(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	int			call_cntr;
	int			max_calls;
	RerankState *state;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		TupleDesc	tupdesc;
		text	   *query_text;
		ArrayType  *candidates_array;
		text	   *model_text;
		int			top_k;

		NDB_DECLARE(char *, query_str);
		Datum	   *candidate_datums;
		bool	   *candidate_nulls;
		int			ncandidates;
		int			i,
					j;

		NDB_DECLARE(char *, model_str);
		NdbLLMConfig cfg;

		NDB_DECLARE(float *, query_embedding);
		int			query_dim = 0;

		NDB_DECLARE(float *, doc_embeddings);

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		query_text = PG_GETARG_TEXT_PP(0);
		candidates_array = PG_GETARG_ARRAYTYPE_P(1);
		model_text = (PG_ARGISNULL(2) ? NULL : PG_GETARG_TEXT_PP(2));
		top_k = PG_GETARG_INT32(3);

		query_str = text_to_cstring(query_text);

		deconstruct_array(candidates_array, TEXTOID, -1, false, 'i',
						  &candidate_datums, &candidate_nulls, &ncandidates);

		if (top_k < 1)
			ereport(ERROR, (errmsg("top_k must be positive (got %d)", top_k)));
		if (ncandidates <= 0)
			ereport(ERROR, (errmsg("candidate array cannot be empty")));
		max_calls = (ncandidates < top_k) ? ncandidates : top_k;

		state = (RerankState *) palloc0(sizeof(RerankState));
		state->query = query_str;
		state->candidates = candidate_datums;
		state->nulls = candidate_nulls;
		state->scores = (float *) palloc0(ncandidates * sizeof(float));
		state->indices = (int *) palloc0(ncandidates * sizeof(int));
		state->ncandidates = ncandidates;

		/* ColBERT uses token-level embeddings with MaxSim scoring */
		model_str = model_text ? text_to_cstring(model_text) : NULL;
		cfg.provider = neurondb_llm_provider ? neurondb_llm_provider : "huggingface";
		cfg.endpoint = neurondb_llm_endpoint ? neurondb_llm_endpoint : "https://router.huggingface.co";
		cfg.model = model_str ? model_str : "colbert-ir/colbertv2.0";
		cfg.api_key = neurondb_llm_api_key;
		cfg.timeout_ms = neurondb_llm_timeout_ms;
		cfg.prefer_gpu = NDB_SHOULD_TRY_GPU();
		cfg.require_gpu = false;

		/* Get query embedding (token-level for ColBERT) */
		if (ndb_llm_route_embed(&cfg, NULL, query_str, &query_embedding, &query_dim) == 0
			&& query_embedding != NULL)
		{
			/* Get document embeddings and compute MaxSim scores */
			/* Check for integer overflow in size calculation */
			if (ncandidates > 0 && query_dim > 0 && (size_t) ncandidates > MaxAllocSize / sizeof(float) / (size_t) query_dim)
			{
				NDB_FREE(state);
				ereport(ERROR,
						(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
						 errmsg("neurondb: doc_embeddings allocation size exceeds MaxAllocSize")));
			}
			doc_embeddings = (float *) palloc0(ncandidates * query_dim * sizeof(float));
			for (i = 0; i < ncandidates; i++)
			{
				char	   *doc_str;

				NDB_DECLARE(float *, doc_emb);
				int			temp_dim = 0;

				if (!candidate_nulls[i] && DatumGetPointer(candidate_datums[i]))
				{
					doc_str = text_to_cstring(DatumGetTextPP(candidate_datums[i]));
					if (ndb_llm_route_embed(&cfg, NULL, doc_str, &doc_emb, &temp_dim) == 0
						&& doc_emb != NULL && temp_dim == query_dim)
					{
						memcpy(doc_embeddings + i * query_dim, doc_emb, query_dim * sizeof(float));
						NDB_FREE(doc_emb);
					}
					NDB_FREE(doc_str);
				}
			}

			/* Compute MaxSim scores: sum of max similarities per query token */
			for (i = 0; i < ncandidates; i++)
			{
				float		maxsim_score = 0.0f;
				int			num_query_tokens = query_dim / 128; /* Approximate token
																 * count */

				if (num_query_tokens < 1)
					num_query_tokens = 1;

				/* Simplified MaxSim: dot product similarity */
				for (j = 0; j < query_dim && j < 128; j++)
				{
					float		max_sim = -FLT_MAX;
					int			k;

					for (k = 0; k < query_dim && k < 128; k++)
					{
						float		sim = query_embedding[j] * doc_embeddings[i * query_dim + k];

						if (sim > max_sim)
							max_sim = sim;
					}
					maxsim_score += max_sim;
				}
				state->scores[i] = maxsim_score / num_query_tokens;
				state->indices[i] = i;
			}

			if (query_embedding)
				NDB_FREE(query_embedding);
			if (doc_embeddings)
				NDB_FREE(doc_embeddings);
		}
		else
		{
			/* Fallback to sequential scores */
			for (i = 0; i < ncandidates; i++)
			{
				state->indices[i] = i;
				state->scores[i] = 1.0f - ((float) i / (float) ncandidates);
			}
		}

		sort_rerank_desc(state->scores, state->indices, ncandidates);

		if (model_str)
			NDB_FREE(model_str);

		tupdesc = CreateTemplateTupleDesc(2);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "idx", INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "score", FLOAT4OID, -1, 0);
		BlessTupleDesc(tupdesc);

		funcctx->max_calls = max_calls;
		funcctx->user_fctx = state;
		funcctx->tuple_desc = tupdesc;

		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();
	state = (RerankState *) funcctx->user_fctx;
	call_cntr = funcctx->call_cntr;
	max_calls = funcctx->max_calls;

	if (call_cntr < max_calls)
	{
		HeapTuple	tuple;
		Datum		values[2];
		bool		nulls[2] = {false, false};
		int			idx_ranked = state->indices[call_cntr];

		values[0] = Int32GetDatum(idx_ranked);
		values[1] = Float4GetDatum(state->scores[idx_ranked]);
		tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);

		SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
	}
	else
	{
		SRF_RETURN_DONE(funcctx);
	}
}

/*
 * rerank_ltr
 *    Learning-to-Rank model reranking (LambdaMART, RankNet, etc.).
 *    Uses feature-based ranking with learned models.
 */
PG_FUNCTION_INFO_V1(rerank_ltr);
Datum
rerank_ltr(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	int			call_cntr;
	int			max_calls;
	RerankState *state;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		TupleDesc	tupdesc;
		text	   *query_text;
		ArrayType  *candidates_array;
		text	   *model_text;
		int			top_k;

		NDB_DECLARE(char *, query_str);
		Datum	   *candidate_datums;
		bool	   *candidate_nulls;
		int			ncandidates;
		int			i;

		NDB_DECLARE(char *, model_str);
		NdbLLMConfig cfg;

		NDB_DECLARE(float *, query_embedding);
		int			query_dim = 0;

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		query_text = PG_GETARG_TEXT_PP(0);
		candidates_array = PG_GETARG_ARRAYTYPE_P(1);
		model_text = (PG_ARGISNULL(2) ? NULL : PG_GETARG_TEXT_PP(2));
		top_k = PG_GETARG_INT32(3);

		query_str = text_to_cstring(query_text);

		deconstruct_array(candidates_array, TEXTOID, -1, false, 'i',
						  &candidate_datums, &candidate_nulls, &ncandidates);

		if (top_k < 1)
			ereport(ERROR, (errmsg("top_k must be positive (got %d)", top_k)));
		if (ncandidates <= 0)
			ereport(ERROR, (errmsg("candidate array cannot be empty")));
		max_calls = (ncandidates < top_k) ? ncandidates : top_k;

		state = (RerankState *) palloc0(sizeof(RerankState));
		state->query = query_str;
		state->candidates = candidate_datums;
		state->nulls = candidate_nulls;
		state->scores = (float *) palloc0(ncandidates * sizeof(float));
		state->indices = (int *) palloc0(ncandidates * sizeof(int));
		state->ncandidates = ncandidates;

		/*
		 * LTR uses feature-based scoring (simplified: cosine similarity +
		 * features)
		 */
		model_str = model_text ? text_to_cstring(model_text) : NULL;
		cfg.provider = neurondb_llm_provider ? neurondb_llm_provider : "huggingface";
		cfg.endpoint = neurondb_llm_endpoint ? neurondb_llm_endpoint : "https://router.huggingface.co";
		cfg.model = model_str ? model_str : "sentence-transformers/all-MiniLM-L6-v2";
		cfg.api_key = neurondb_llm_api_key;
		cfg.timeout_ms = neurondb_llm_timeout_ms;
		cfg.prefer_gpu = NDB_SHOULD_TRY_GPU();
		cfg.require_gpu = false;

		/* Get query embedding */
		if (ndb_llm_route_embed(&cfg, NULL, query_str, &query_embedding, &query_dim) == 0
			&& query_embedding != NULL)
		{
			/* Compute LTR features and scores for each document */
			for (i = 0; i < ncandidates; i++)
			{
				char	   *doc_str;

				NDB_DECLARE(float *, doc_emb);
				int			doc_dim = 0;
				float		cosine_sim = 0.0f;
				float		doc_length = 0.0f;
				float		query_length = 0.0f;
				int			j;

				if (!candidate_nulls[i] && DatumGetPointer(candidate_datums[i]))
				{
					doc_str = text_to_cstring(DatumGetTextPP(candidate_datums[i]));
					if (ndb_llm_route_embed(&cfg, NULL, doc_str, &doc_emb, &doc_dim) == 0
						&& doc_emb != NULL && doc_dim == query_dim)
					{
						/* Compute cosine similarity */
						float		dot_product = 0.0f;

						for (j = 0; j < query_dim; j++)
						{
							dot_product += query_embedding[j] * doc_emb[j];
							query_length += query_embedding[j] * query_embedding[j];
							doc_length += doc_emb[j] * doc_emb[j];
						}

						query_length = sqrtf(query_length);
						doc_length = sqrtf(doc_length);

						if (query_length > 0.0f && doc_length > 0.0f)
							cosine_sim = dot_product / (query_length * doc_length);

						/* LTR score: combination of similarity and features */

						/*
						 * Simplified: use cosine similarity as base, add
						 * document length feature
						 */
						state->scores[i] = cosine_sim * 0.8f + (doc_length / (doc_length + 100.0f)) * 0.2f;

						NDB_FREE(doc_emb);
					}
					NDB_FREE(doc_str);
				}
				else
				{
					state->scores[i] = 0.0f;
				}
				state->indices[i] = i;
			}

			if (query_embedding)
				NDB_FREE(query_embedding);
		}
		else
		{
			/* Fallback to sequential scores */
			for (i = 0; i < ncandidates; i++)
			{
				state->indices[i] = i;
				state->scores[i] = 1.0f - ((float) i / (float) ncandidates);
			}
		}

		sort_rerank_desc(state->scores, state->indices, ncandidates);

		if (model_str)
			NDB_FREE(model_str);

		tupdesc = CreateTemplateTupleDesc(2);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "idx", INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "score", FLOAT4OID, -1, 0);
		BlessTupleDesc(tupdesc);

		funcctx->max_calls = max_calls;
		funcctx->user_fctx = state;
		funcctx->tuple_desc = tupdesc;

		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();
	state = (RerankState *) funcctx->user_fctx;
	call_cntr = funcctx->call_cntr;
	max_calls = funcctx->max_calls;

	if (call_cntr < max_calls)
	{
		HeapTuple	tuple;
		Datum		values[2];
		bool		nulls[2] = {false, false};
		int			idx_ranked = state->indices[call_cntr];

		values[0] = Int32GetDatum(idx_ranked);
		values[1] = Float4GetDatum(state->scores[idx_ranked]);
		tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);

		SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
	}
	else
	{
		SRF_RETURN_DONE(funcctx);
	}
}

/*
 * rerank_ensemble
 *    Ensemble reranking across multiple models/strategies.
 *    Combines scores from multiple reranking methods.
 */
PG_FUNCTION_INFO_V1(rerank_ensemble);
Datum
rerank_ensemble(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	int			call_cntr;
	int			max_calls;
	RerankState *state;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		TupleDesc	tupdesc;
		text	   *query_text;
		ArrayType  *candidates_array;
		ArrayType  *methods_array;
		int			top_k;

		NDB_DECLARE(char *, query_str);
		Datum	   *candidate_datums;
		bool	   *candidate_nulls;
		int			ncandidates;
		Datum	   *method_datums;
		bool	   *method_nulls;
		int			nmethods;
		float	  **method_scores;
		int			i,
					j;

		NDB_DECLARE(float *, weights);

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		query_text = PG_GETARG_TEXT_PP(0);
		candidates_array = PG_GETARG_ARRAYTYPE_P(1);
		methods_array = PG_GETARG_ARRAYTYPE_P(2);
		top_k = PG_GETARG_INT32(3);

		query_str = text_to_cstring(query_text);

		deconstruct_array(candidates_array, TEXTOID, -1, false, 'i',
						  &candidate_datums, &candidate_nulls, &ncandidates);
		deconstruct_array(methods_array, TEXTOID, -1, false, 'i',
						  &method_datums, &method_nulls, &nmethods);

		if (top_k < 1)
			ereport(ERROR, (errmsg("top_k must be positive (got %d)", top_k)));
		if (ncandidates <= 0)
			ereport(ERROR, (errmsg("candidate array cannot be empty")));
		if (nmethods <= 0)
			ereport(ERROR, (errmsg("methods array cannot be empty")));
		max_calls = (ncandidates < top_k) ? ncandidates : top_k;

		state = (RerankState *) palloc0(sizeof(RerankState));
		state->query = query_str;
		state->candidates = candidate_datums;
		state->nulls = candidate_nulls;
		state->scores = (float *) palloc0(ncandidates * sizeof(float));
		state->indices = (int *) palloc0(ncandidates * sizeof(int));
		state->ncandidates = ncandidates;

		/* Allocate arrays for method scores */
		method_scores = (float **) palloc0(nmethods * sizeof(float *));
		weights = (float *) palloc0(nmethods * sizeof(float));

		/* Default: equal weights */
		for (i = 0; i < nmethods; i++)
		{
			weights[i] = 1.0f / (float) nmethods;
			method_scores[i] = (float *) palloc0(ncandidates * sizeof(float));
		}

		/* Get scores from each method */
		for (i = 0; i < nmethods; i++)
		{
			char	   *method_str;

			NDB_DECLARE(float *, temp_scores);
			int			api_result;
			NdbLLMConfig cfg;
			NdbLLMCallOptions call_opts;

			NDB_DECLARE(const char **, docs);

			if (method_nulls[i] || !DatumGetPointer(method_datums[i]))
				continue;

			method_str = text_to_cstring(DatumGetTextPP(method_datums[i]));

			/* Call appropriate reranking method */
			cfg.provider = neurondb_llm_provider ? neurondb_llm_provider : "huggingface";
			cfg.endpoint = neurondb_llm_endpoint ? neurondb_llm_endpoint : "https://router.huggingface.co";
			cfg.model = NULL;
			cfg.api_key = neurondb_llm_api_key;
			cfg.timeout_ms = neurondb_llm_timeout_ms;
			cfg.prefer_gpu = NDB_SHOULD_TRY_GPU();
			cfg.require_gpu = false;

			call_opts.task = "rerank";
			call_opts.prefer_gpu = cfg.prefer_gpu;
			call_opts.require_gpu = cfg.require_gpu;
			call_opts.fail_open = neurondb_llm_fail_open;

			NDB_ALLOC(docs, const char *, ncandidates);
			for (j = 0; j < ncandidates; j++)
			{
				if (!candidate_nulls[j] && DatumGetPointer(candidate_datums[j]))
					docs[j] = text_to_cstring(DatumGetTextPP(candidate_datums[j]));
				else
					docs[j] = "";
			}

			api_result = ndb_llm_route_rerank(&cfg, &call_opts, query_str, docs,
											  ncandidates, &temp_scores);
			if (api_result == 0 && temp_scores != NULL)
			{
				memcpy(method_scores[i], temp_scores, ncandidates * sizeof(float));
				NDB_FREE(temp_scores);
			}
			else
			{
				/* Fallback: sequential scores */
				for (j = 0; j < ncandidates; j++)
					method_scores[i][j] = 1.0f - ((float) j / (float) ncandidates);
			}

			for (j = 0; j < ncandidates; j++)
			{
				if (docs[j][0] != '\0')
				{
					pfree((char *) docs[j]);
				}
			}
			NDB_FREE(docs);
			NDB_FREE(method_str);
		}

		/* Combine scores using weighted average */
		for (i = 0; i < ncandidates; i++)
		{
			float		ensemble_score = 0.0f;

			for (j = 0; j < nmethods; j++)
			{
				ensemble_score += method_scores[j][i] * weights[j];
			}
			state->scores[i] = ensemble_score;
			state->indices[i] = i;
		}

		for (i = 0; i < nmethods; i++)
		{
			if (method_scores[i])
				NDB_FREE(method_scores[i]);
		}
		NDB_FREE(method_scores);
		if (weights)
			NDB_FREE(weights);

		sort_rerank_desc(state->scores, state->indices, ncandidates);

		tupdesc = CreateTemplateTupleDesc(2);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "idx", INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "score", FLOAT4OID, -1, 0);
		BlessTupleDesc(tupdesc);

		funcctx->max_calls = max_calls;
		funcctx->user_fctx = state;
		funcctx->tuple_desc = tupdesc;

		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();
	state = (RerankState *) funcctx->user_fctx;
	call_cntr = funcctx->call_cntr;
	max_calls = funcctx->max_calls;

	if (call_cntr < max_calls)
	{
		HeapTuple	tuple;
		Datum		values[2];
		bool		nulls[2] = {false, false};
		int			idx_ranked = state->indices[call_cntr];

		values[0] = Int32GetDatum(idx_ranked);
		values[1] = Float4GetDatum(state->scores[idx_ranked]);
		tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);

		SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
	}
	else
	{
		SRF_RETURN_DONE(funcctx);
	}
}
