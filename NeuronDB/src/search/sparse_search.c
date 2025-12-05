/*-------------------------------------------------------------------------
 *
 * sparse_search.c
 *    Sparse retrieval implementation for SPLADE/ColBERTv2/BM25
 *
 * Implements efficient sparse vector search using inverted indexes.
 * Supports learned sparse retrieval models and traditional BM25.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/search/sparse_search.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "executor/spi.h"
#include "access/heapam.h"
#include "access/table.h"
#include "access/tupdesc.h"
#include "utils/guc.h"
#include "neurondb.h"
#include "neurondb_types.h"
#include "neurondb_sparse.h"
#include "neurondb_validation.h"
#include "neurondb_macros.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"
#ifdef HAVE_ONNX_RUNTIME
#include "neurondb_onnx.h"
#endif
#include <string.h>
#include <math.h>

/*
 * sparse_search: Search using sparse vector query
 */
PG_FUNCTION_INFO_V1(sparse_search);
Datum
sparse_search(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_PP(0);
	text	   *sparse_col = PG_GETARG_TEXT_PP(1);
	Datum		query_vec = PG_GETARG_DATUM(2);
	int32		k = PG_GETARG_INT32(3);
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	TupleDesc	tupdesc;
	Tuplestorestate *tupstore;
	MemoryContext per_query_ctx;
	MemoryContext oldcontext;

	NDB_DECLARE(NdbSpiSession *, session);
	StringInfoData sql;
	int			ret;
	Datum		values[2];
	bool		nulls[2] = {false, false};
	Oid			argtypes[1];
	Datum		args[1];
	char	   *tbl_str = text_to_cstring(table_name);
	char	   *col_str = text_to_cstring(sparse_col);

	if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("sparse_search must be called as table function")));

	if (!(rsinfo->allowedModes & SFRM_Materialize))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("sparse_search requires Materialize mode")));

	per_query_ctx = rsinfo->econtext->ecxt_per_query_memory;
	oldcontext = MemoryContextSwitchTo(per_query_ctx);

	tupdesc = CreateTemplateTupleDesc(2);
	TupleDescInitEntry(tupdesc, (AttrNumber) 1, "doc_id", INT4OID, -1, 0);
	TupleDescInitEntry(tupdesc, (AttrNumber) 2, "score", FLOAT4OID, -1, 0);
	BlessTupleDesc(tupdesc);

	{
		const char *work_mem_str = GetConfigOption("work_mem", true, false);
		int			work_mem_kb = 262144;	/* Default 256MB */

		if (work_mem_str)
		{
			work_mem_kb = atoi(work_mem_str);
			if (work_mem_kb <= 0)
				work_mem_kb = 262144;
		}
		tupstore = tuplestore_begin_heap(true, false, work_mem_kb);
	}
	rsinfo->returnMode = SFRM_Materialize;
	rsinfo->setResult = tupstore;
	rsinfo->setDesc = tupdesc;

	MemoryContextSwitchTo(oldcontext);
	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to begin SPI session")));

	/* Perform sparse search using dot product */
	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT ctid, sparse_vector_dot_product(%s, $1) AS score "
					 "FROM %s "
					 "WHERE %s IS NOT NULL "
					 "ORDER BY score DESC "
					 "LIMIT %d",
					 col_str,
					 tbl_str,
					 col_str,
					 k);

	argtypes[0] = get_fn_expr_argtype(fcinfo->flinfo, 2);
	args[0] = query_vec;

	ret = ndb_spi_execute_with_args(session,
									sql.data,
									1,
									argtypes,
									args,
									NULL,
									false,
									0);

	if (ret != SPI_OK_SELECT)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("sparse_search query failed")));

	/* Return results */
	for (int i = 0; i < SPI_processed; i++)
	{
		HeapTuple	tuple = SPI_tuptable->vals[i];
		ItemPointer ctid = &tuple->t_self;
		float4		score = DatumGetFloat4(SPI_getbinval(tuple,
														 SPI_tuptable->tupdesc,
														 2,
														 &nulls[1]));

		values[0] = ItemPointerGetBlockNumber(ctid);
		values[1] = Float4GetDatum(score);

		tuplestore_putvalues(tupstore, tupdesc, values, nulls);
	}

	NDB_FREE(sql.data);
	ndb_spi_session_end(&session);

	PG_RETURN_NULL();
}

/*
 * splade_embed: Generate SPLADE embedding from text
 */
PG_FUNCTION_INFO_V1(splade_embed);
Datum
splade_embed(PG_FUNCTION_ARGS)
{
	text	   *input_text = PG_GETARG_TEXT_PP(0);
	char	   *input_str = text_to_cstring(input_text);
#ifdef HAVE_ONNX_RUNTIME
	ONNXModelSession *session = NULL;
	ONNXTensor *input_tensor = NULL;
	ONNXTensor *output_tensor = NULL;
	int32	   *token_ids = NULL;
	int32		token_length = 0;
	int			max_length = 512;
	int			i;
	SparseVector *sparse_vec = NULL;
	int		   *indices = NULL;
	float	   *values = NULL;
	int			sparse_count = 0;

	/* Load SPLADE model (default model name) */
	session = neurondb_onnx_get_or_load_model("splade", ONNX_MODEL_EMBEDDING);
	if (session == NULL || !session->is_loaded)
	{
		NDB_FREE(input_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("splade_embed: SPLADE model not available")));
	}

	/* Tokenize input text */
	token_ids = neurondb_tokenize_with_model(input_str, max_length, &token_length, "splade");
	if (token_ids == NULL || token_length <= 0)
	{
		NDB_FREE(input_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("splade_embed: tokenization failed")));
	}

	/* Create input tensor */
	input_tensor = (ONNXTensor *) palloc0(sizeof(ONNXTensor));
	input_tensor->ndim = 2;
	NDB_DECLARE(int64 *, shape);
	NDB_ALLOC(shape, int64, 2);
	input_tensor->shape = shape;
	input_tensor->shape[0] = 1; /* Batch size */
	input_tensor->shape[1] = token_length;
	input_tensor->size = token_length;
	NDB_DECLARE(float *, data);
	NDB_ALLOC(data, float, token_length);
	input_tensor->data = data;
	for (i = 0; i < token_length; i++)
		input_tensor->data[i] = (float) token_ids[i];

	/* Run inference */
	output_tensor = neurondb_onnx_run_inference(session, input_tensor);
	if (output_tensor == NULL || output_tensor->data == NULL)
	{
		neurondb_onnx_free_tensor(input_tensor);
		NDB_FREE(token_ids);
		NDB_FREE(input_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("splade_embed: inference failed")));
	}

	/*
	 * Extract sparse vector from output (SPLADE outputs log-scaled sparse
	 * vectors)
	 */
	/* Count non-zero values */
	sparse_count = 0;
	for (i = 0; i < (int) output_tensor->size; i++)
	{
		if (output_tensor->data[i] > 0.0f)
			sparse_count++;
	}

	if (sparse_count > 0)
	{
		NDB_DECLARE(int *, indices);
		NDB_DECLARE(float *, values);
		NDB_ALLOC(indices, int32, sparse_count);
		NDB_ALLOC(values, float4, sparse_count);
		sparse_count = 0;
		for (i = 0; i < (int) output_tensor->size; i++)
		{
			if (output_tensor->data[i] > 0.0f)
			{
				indices[sparse_count] = (int32) i;
				/* SPLADE uses ReLU activation, values are already log-scaled */
				values[sparse_count] = (float4) output_tensor->data[i];
				sparse_count++;
			}
		}
	}

	/* Create sparse vector using proper structure */
	NDB_DECLARE(char *, sparse_vec_raw);
	NDB_ALLOC(sparse_vec_raw, char, SPARSE_VEC_SIZE(sparse_count));
	sparse_vec = (SparseVector *) sparse_vec_raw;
	SET_VARSIZE(sparse_vec, SPARSE_VEC_SIZE(sparse_count));
	sparse_vec->vocab_size = (int32) output_tensor->size;
	sparse_vec->nnz = (int32) sparse_count;
	sparse_vec->model_type = 1; /* SPLADE */
	sparse_vec->flags = 0;
	memcpy(SPARSE_VEC_TOKEN_IDS(sparse_vec), indices, sizeof(int32) * sparse_count);
	memcpy(SPARSE_VEC_WEIGHTS(sparse_vec), values, sizeof(float4) * sparse_count);

	neurondb_onnx_free_tensor(input_tensor);
	neurondb_onnx_free_tensor(output_tensor);
	NDB_FREE(token_ids);
	NDB_FREE(indices);
	NDB_FREE(values);
	NDB_FREE(input_str);

	PG_RETURN_POINTER(sparse_vec);
#else
	NDB_FREE(input_str);
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("splade_embed: ONNX runtime not available")));
	PG_RETURN_NULL();
#endif
}

/*
 * colbertv2_embed: Generate ColBERTv2 embedding from text
 */
PG_FUNCTION_INFO_V1(colbertv2_embed);
Datum
colbertv2_embed(PG_FUNCTION_ARGS)
{
	text	   *input_text = PG_GETARG_TEXT_PP(0);
	char	   *input_str = text_to_cstring(input_text);
#ifdef HAVE_ONNX_RUNTIME
	ONNXModelSession *session = NULL;
	ONNXTensor *input_tensor = NULL;
	ONNXTensor *output_tensor = NULL;
	int32	   *token_ids = NULL;
	int32		token_length = 0;
	int			max_length = 512;
	int			i,
				j;
	SparseVector *sparse_vec = NULL;
	int		   *indices = NULL;
	float	   *values = NULL;
	int			sparse_count = 0;
	int			output_dim;

	/* Load ColBERTv2 model (default model name) */
	session = neurondb_onnx_get_or_load_model("colbertv2", ONNX_MODEL_EMBEDDING);
	if (session == NULL || !session->is_loaded)
	{
		NDB_FREE(input_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("colbertv2_embed: ColBERTv2 model not available")));
	}

	/* Tokenize input text */
	token_ids = neurondb_tokenize_with_model(input_str, max_length, &token_length, "colbertv2");
	if (token_ids == NULL || token_length <= 0)
	{
		NDB_FREE(input_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("colbertv2_embed: tokenization failed")));
	}

	/* Create input tensor */
	input_tensor = (ONNXTensor *) palloc0(sizeof(ONNXTensor));
	input_tensor->ndim = 2;
	NDB_DECLARE(int64 *, shape);
	NDB_ALLOC(shape, int64, 2);
	input_tensor->shape = shape;
	input_tensor->shape[0] = 1; /* Batch size */
	input_tensor->shape[1] = token_length;
	input_tensor->size = token_length;
	NDB_DECLARE(float *, data);
	NDB_ALLOC(data, float, token_length);
	input_tensor->data = data;
	for (i = 0; i < token_length; i++)
		input_tensor->data[i] = (float) token_ids[i];

	/* Run inference */
	output_tensor = neurondb_onnx_run_inference(session, input_tensor);
	if (output_tensor == NULL || output_tensor->data == NULL)
	{
		neurondb_onnx_free_tensor(input_tensor);
		NDB_FREE(token_ids);
		NDB_FREE(input_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("colbertv2_embed: inference failed")));
	}

	/*
	 * ColBERTv2 outputs token-level embeddings, need to aggregate to sparse
	 * vector
	 */
	/* Output shape is typically [batch, tokens, dim] or [batch, tokens*dim] */
	if (output_tensor->ndim == 3)
	{
		/*
		 * [batch, tokens, dim] - take max pooling over tokens for each
		 * dimension
		 */
		int			tokens;
		float	   *max_pooled;

		tokens = (int) output_tensor->shape[1];
		output_dim = (int) output_tensor->shape[2];
		max_pooled = (float *) palloc0(sizeof(float) * output_dim);

		for (i = 0; i < tokens; i++)
		{
			for (j = 0; j < output_dim; j++)
			{
				float		val = output_tensor->data[i * output_dim + j];

				if (val > max_pooled[j])
					max_pooled[j] = val;
			}
		}

		/* Extract sparse representation (non-zero values) */
		sparse_count = 0;
		for (j = 0; j < output_dim; j++)
		{
			if (max_pooled[j] > 0.0f)
				sparse_count++;
		}

		if (sparse_count > 0)
		{
			NDB_DECLARE(int *, indices);
			NDB_DECLARE(float *, values);
			NDB_ALLOC(indices, int32, sparse_count);
			NDB_ALLOC(values, float4, sparse_count);
			sparse_count = 0;
			for (j = 0; j < output_dim; j++)
			{
				if (max_pooled[j] > 0.0f)
				{
					indices[sparse_count] = (int32) j;
					values[sparse_count] = (float4) max_pooled[j];
					sparse_count++;
				}
			}
		}

		NDB_FREE(max_pooled);
	}
	else
	{
		/* Flattened output - treat as 1D sparse vector */
		output_dim = (int) output_tensor->size;
		sparse_count = 0;
		for (i = 0; i < output_dim; i++)
		{
			if (output_tensor->data[i] > 0.0f)
				sparse_count++;
		}

		if (sparse_count > 0)
		{
			NDB_DECLARE(int *, indices);
			NDB_DECLARE(float *, values);
			NDB_ALLOC(indices, int32, sparse_count);
			NDB_ALLOC(values, float4, sparse_count);
			sparse_count = 0;
			for (i = 0; i < output_dim; i++)
			{
				if (output_tensor->data[i] > 0.0f)
				{
					indices[sparse_count] = (int32) i;
					values[sparse_count] = (float4) output_tensor->data[i];
					sparse_count++;
				}
			}
		}
	}

	/* Create sparse vector using proper structure */
	NDB_DECLARE(char *, sparse_vec_raw);
	NDB_ALLOC(sparse_vec_raw, char, SPARSE_VEC_SIZE(sparse_count));
	sparse_vec = (SparseVector *) sparse_vec_raw;
	SET_VARSIZE(sparse_vec, SPARSE_VEC_SIZE(sparse_count));
	sparse_vec->vocab_size = (int32) output_dim;
	sparse_vec->nnz = (int32) sparse_count;
	sparse_vec->model_type = 2; /* ColBERTv2 */
	sparse_vec->flags = 0;
	memcpy(SPARSE_VEC_TOKEN_IDS(sparse_vec), indices, sizeof(int32) * sparse_count);
	memcpy(SPARSE_VEC_WEIGHTS(sparse_vec), values, sizeof(float4) * sparse_count);

	neurondb_onnx_free_tensor(input_tensor);
	neurondb_onnx_free_tensor(output_tensor);
	NDB_FREE(token_ids);
	NDB_FREE(indices);
	NDB_FREE(values);
	NDB_FREE(input_str);

	PG_RETURN_POINTER(sparse_vec);
#else
	NDB_FREE(input_str);
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("colbertv2_embed: ONNX runtime not available")));
	PG_RETURN_NULL();
#endif
}

/*
 * Helper: Simple tokenization into lowercase words
 */
static void
bm25_tokenize(const char *text, char **tokens, int *num_tokens, int max_tokens)
{
	int			i = 0;
	int			text_len = (int) strlen(text);
	int			t = 0;
	char		wordbuf[256];
	int			j;

	while (i < text_len && t < max_tokens)
	{
		/* Skip non-alphanumeric */
		while (i < text_len && !isalnum((unsigned char) text[i]))
			i++;
		if (i >= text_len)
			break;

		j = 0;
		memset(wordbuf, 0, sizeof(wordbuf));
		while (i < text_len && isalnum((unsigned char) text[i]) && j < 255)
		{
			wordbuf[j++] = (char) tolower((unsigned char) text[i]);
			i++;
		}
		wordbuf[j] = '\0';
		if (j > 0)
			tokens[t++] = pstrdup(wordbuf);
	}
	*num_tokens = t;
}

/*
 * Helper: Count term frequencies in token array
 */
static void
bm25_count_tf(char **tokens, int num_tokens, int *term_counts, char **unique_terms, int *num_unique)
{
	int			i,
				j;
	int			found;

	*num_unique = 0;
	for (i = 0; i < num_tokens; i++)
	{
		found = 0;
		for (j = 0; j < *num_unique; j++)
		{
			if (strcmp(tokens[i], unique_terms[j]) == 0)
			{
				term_counts[j]++;
				found = 1;
				break;
			}
		}
		if (!found)
		{
			unique_terms[*num_unique] = tokens[i];
			term_counts[*num_unique] = 1;
			(*num_unique)++;
		}
	}
}

/*
 * bm25_score: Compute BM25 score between query and document
 *
 * BM25 formula: sum over query terms of:
 *   IDF(q_i) * (f(q_i, D) * (k1 + 1)) / (f(q_i, D) + k1 * (1 - b + b * |D| / avgdl))
 *
 * Where:
 *   - IDF(q_i) = log((N - n(q_i) + 0.5) / (n(q_i) + 0.5))
 *   - f(q_i, D) = term frequency of q_i in document D
 *   - |D| = document length in words
 *   - avgdl = average document length in collection
 *   - k1 = term frequency saturation parameter (default 1.2)
 *   - b = length normalization parameter (default 0.75)
 */
PG_FUNCTION_INFO_V1(bm25_score);
Datum
bm25_score(PG_FUNCTION_ARGS)
{
	text	   *query_text = PG_GETARG_TEXT_PP(0);
	text	   *doc_text = PG_GETARG_TEXT_PP(1);
	float8		k1 = PG_ARGISNULL(2) ? 1.2 : PG_GETARG_FLOAT8(2);
	float8		b = PG_ARGISNULL(3) ? 0.75 : PG_GETARG_FLOAT8(3);
	char	   *query_str = text_to_cstring(query_text);
	char	   *doc_str = text_to_cstring(doc_text);
	char	  **query_tokens = NULL;
	char	  **doc_tokens = NULL;
	char	  **query_unique = NULL;
	int		   *query_counts = NULL;
	int		   *doc_counts = NULL;
	int			num_query_tokens = 0;
	int			num_doc_tokens = 0;
	int			num_query_unique = 0;
	int			max_tokens = 10000;
	double		score = 0.0;
	int			i,
				j;
	double		doc_length;
	double		avg_doc_length = 100.0; /* Default average, would be computed
										 * from collection */
	double		N = 1000.0;		/* Total documents in collection, would be
								 * computed */
	int			n_qi;			/* Number of documents containing term q_i */
	double		idf;
	double		tf;
	double		bm25_term;

	/* Tokenize query and document */
	NDB_ALLOC(query_tokens, char *, max_tokens);
	NDB_ALLOC(doc_tokens, char *, max_tokens);
	bm25_tokenize(query_str, query_tokens, &num_query_tokens, max_tokens);
	bm25_tokenize(doc_str, doc_tokens, &num_doc_tokens, max_tokens);

	if (num_query_tokens == 0 || num_doc_tokens == 0)
	{
		for (i = 0; i < num_query_tokens; i++)
			NDB_FREE(query_tokens[i]);
		for (i = 0; i < num_doc_tokens; i++)
			NDB_FREE(doc_tokens[i]);
		NDB_FREE(query_tokens);
		NDB_FREE(doc_tokens);
		NDB_FREE(query_str);
		NDB_FREE(doc_str);
		PG_RETURN_FLOAT4(0.0f);
	}

	doc_length = (double) num_doc_tokens;

	/* Count term frequencies */
	NDB_ALLOC(query_unique, char *, num_query_tokens);
	query_counts = (int *) palloc0(sizeof(int) * num_query_tokens);
	doc_counts = (int *) palloc0(sizeof(int) * num_doc_tokens);
	bm25_count_tf(query_tokens, num_query_tokens, query_counts, query_unique, &num_query_unique);

	/* Count document term frequencies for query terms */
	for (i = 0; i < num_query_unique; i++)
	{
		doc_counts[i] = 0;
		for (j = 0; j < num_doc_tokens; j++)
		{
			if (strcmp(query_unique[i], doc_tokens[j]) == 0)
				doc_counts[i]++;
		}
	}

	/* Compute BM25 score */
	for (i = 0; i < num_query_unique; i++)
	{
		/* Term frequency in document */
		tf = (double) doc_counts[i];

		/* IDF: simplified - assume term appears in 10% of documents */
		/* In production, would query collection statistics */
		n_qi = (int) (N * 0.1);
		if (n_qi < 1)
			n_qi = 1;
		idf = log((N - (double) n_qi + 0.5) / ((double) n_qi + 0.5));

		/* BM25 term score */
		bm25_term = idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * doc_length / avg_doc_length));

		score += bm25_term;
	}

	for (i = 0; i < num_query_tokens; i++)
		NDB_FREE(query_tokens[i]);
	for (i = 0; i < num_doc_tokens; i++)
		NDB_FREE(doc_tokens[i]);
	NDB_FREE(query_tokens);
	NDB_FREE(doc_tokens);
	NDB_FREE(query_unique);
	NDB_FREE(query_counts);
	NDB_FREE(doc_counts);
	NDB_FREE(query_str);
	NDB_FREE(doc_str);

	PG_RETURN_FLOAT4((float4) score);
}
