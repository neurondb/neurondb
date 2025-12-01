/*-------------------------------------------------------------------------
 *
 * sparse_vector_types.c
 *    Learned sparse vector type for SPLADE/ColBERTv2 retrieval
 *
 * Implements sparse_vector type optimized for learned sparse retrieval
 * models like SPLADE and ColBERTv2. Stores sparse representations with
 * token IDs and learned weights.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/types/sparse_vector_types.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "lib/stringinfo.h"
#include "libpq/pqformat.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/varlena.h"
#include "neurondb.h"
#include "neurondb_types.h"
#include "neurondb_sparse.h"
#include <string.h>
#include <math.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_json.h"

/*
 * sparse_vector_in: Parse sparse vector from text
 * Format: "{vocab_size:30522, model:SPLADE, tokens:[100,200,300], weights:[0.5,0.8,0.3]}"
 */
PG_FUNCTION_INFO_V1(sparse_vector_in);
Datum
sparse_vector_in(PG_FUNCTION_ARGS)
{
	char	   *str = PG_GETARG_CSTRING(0);
	SparseVector *result;
	NdbSparseVectorParse parsed = {0};
	char	   *errstr = NULL;
	int			parse_result;

	/* Use centralized JSON parsing */
	parse_result = ndb_json_parse_sparse_vector(str, &parsed, &errstr);
	if (parse_result != 0)
	{
		if (errstr)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("sparse_vector parsing failed: %s", errstr)));
			pfree(errstr);
		}
		else
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("sparse_vector parsing failed")));
		}
		ndb_json_parse_sparse_vector_free(&parsed);
		PG_RETURN_NULL();
	}

	if (parsed.nnz == 0)
	{
		ndb_json_parse_sparse_vector_free(&parsed);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("sparse_vector must have at least one token")));
		PG_RETURN_NULL();
	}

	/* Build result */
	result = (SparseVector *) palloc0(SPARSE_VEC_SIZE(parsed.nnz));
	SET_VARSIZE(result, SPARSE_VEC_SIZE(parsed.nnz));
	result->vocab_size = parsed.vocab_size;
	result->nnz = parsed.nnz;
	result->model_type = parsed.model_type;

	memcpy(SPARSE_VEC_TOKEN_IDS(result), parsed.token_ids, sizeof(int32) * parsed.nnz);
	memcpy(SPARSE_VEC_WEIGHTS(result), parsed.weights, sizeof(float4) * parsed.nnz);

	/* Free parsed structure (but not the arrays - they're now in result) */
	parsed.token_ids = NULL;
	parsed.weights = NULL;
	ndb_json_parse_sparse_vector_free(&parsed);

	PG_RETURN_POINTER(result);
}

/*
 * sparse_vector_out: Convert sparse vector to text
 */
PG_FUNCTION_INFO_V1(sparse_vector_out);
Datum
sparse_vector_out(PG_FUNCTION_ARGS)
{
	SparseVector *sv = (SparseVector *) PG_GETARG_POINTER(0);
	StringInfoData buf;
	int32	   *token_ids;
	float4	   *weights;
	int			i;
	const char *model_name;

	if (sv == NULL)
		PG_RETURN_CSTRING(pstrdup("NULL"));

	token_ids = SPARSE_VEC_TOKEN_IDS(sv);
	weights = SPARSE_VEC_WEIGHTS(sv);

	switch (sv->model_type)
	{
		case 0:
			model_name = "BM25";
			break;
		case 1:
			model_name = "SPLADE";
			break;
		case 2:
			model_name = "ColBERTv2";
			break;
		default:
			model_name = "UNKNOWN";
			break;
	}

	initStringInfo(&buf);
	appendStringInfo(&buf,
					 "{vocab_size:%d, model:%s, tokens:[",
					 sv->vocab_size,
					 model_name);

	for (i = 0; i < sv->nnz; i++)
	{
		if (i > 0)
			appendStringInfoChar(&buf, ',');
		appendStringInfo(&buf, "%d", token_ids[i]);
	}

	appendStringInfoString(&buf, "], weights:[");

	for (i = 0; i < sv->nnz; i++)
	{
		if (i > 0)
			appendStringInfoChar(&buf, ',');
		appendStringInfo(&buf, "%g", weights[i]);
	}

	appendStringInfoChar(&buf, '}');

	PG_RETURN_CSTRING(buf.data);
}

/*
 * sparse_vector_recv: Binary receive
 */
PG_FUNCTION_INFO_V1(sparse_vector_recv);
Datum
sparse_vector_recv(PG_FUNCTION_ARGS)
{
	StringInfo	buf = (StringInfo) PG_GETARG_POINTER(0);
	SparseVector *result;
	int32		vocab_size;
	int32		nnz;
	uint16		model_type;
	int			i;

	vocab_size = pq_getmsgint(buf, sizeof(int32));
	nnz = pq_getmsgint(buf, sizeof(int32));
	model_type = pq_getmsgint(buf, sizeof(uint16));

	if (vocab_size <= 0 || vocab_size > 1000000)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_BINARY_REPRESENTATION),
				 errmsg("invalid sparse_vector vocab_size: %d",
						vocab_size)));

	if (nnz < 0 || nnz > 10000)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_BINARY_REPRESENTATION),
				 errmsg("invalid sparse_vector nnz: %d", nnz)));

	result = (SparseVector *) palloc0(SPARSE_VEC_SIZE(nnz));
	SET_VARSIZE(result, SPARSE_VEC_SIZE(nnz));
	result->vocab_size = vocab_size;
	result->nnz = nnz;
	result->model_type = model_type;

	for (i = 0; i < nnz; i++)
		SPARSE_VEC_TOKEN_IDS(result)[i] = pq_getmsgint(buf, sizeof(int32));

	for (i = 0; i < nnz; i++)
		SPARSE_VEC_WEIGHTS(result)[i] = pq_getmsgfloat4(buf);

	PG_RETURN_POINTER(result);
}

/*
 * sparse_vector_send: Binary send
 */
PG_FUNCTION_INFO_V1(sparse_vector_send);
Datum
sparse_vector_send(PG_FUNCTION_ARGS)
{
	SparseVector *sv = (SparseVector *) PG_GETARG_POINTER(0);
	StringInfoData buf;
	int32	   *token_ids;
	float4	   *weights;
	int			i;

	token_ids = SPARSE_VEC_TOKEN_IDS(sv);
	weights = SPARSE_VEC_WEIGHTS(sv);

	pq_begintypsend(&buf);
	pq_sendint(&buf, sv->vocab_size, sizeof(int32));
	pq_sendint(&buf, sv->nnz, sizeof(int32));
	pq_sendint(&buf, sv->model_type, sizeof(uint16));

	for (i = 0; i < sv->nnz; i++)
		pq_sendint(&buf, token_ids[i], sizeof(int32));

	for (i = 0; i < sv->nnz; i++)
		pq_sendfloat4(&buf, weights[i]);

	PG_RETURN_BYTEA_P(pq_endtypsend(&buf));
}

/*
 * sparse_vector_dot_product: Compute dot product between two sparse vectors
 */
PG_FUNCTION_INFO_V1(sparse_vector_dot_product);
Datum
sparse_vector_dot_product(PG_FUNCTION_ARGS)
{
	SparseVector *a = (SparseVector *) PG_GETARG_POINTER(0);
	SparseVector *b = (SparseVector *) PG_GETARG_POINTER(1);
	int32	   *a_tokens,
			   *b_tokens;
	float4	   *a_weights,
			   *b_weights;
	float4		result = 0.0;
	int			i,
				j;

	if (a == NULL || b == NULL)
		PG_RETURN_FLOAT4(0.0);

	a_tokens = SPARSE_VEC_TOKEN_IDS(a);
	b_tokens = SPARSE_VEC_TOKEN_IDS(b);
	a_weights = SPARSE_VEC_WEIGHTS(a);
	b_weights = SPARSE_VEC_WEIGHTS(b);

	/* Compute dot product by matching token IDs */
	for (i = 0; i < a->nnz; i++)
	{
		for (j = 0; j < b->nnz; j++)
		{
			if (a_tokens[i] == b_tokens[j])
			{
				result += a_weights[i] * b_weights[j];
				break;
			}
		}
	}

	PG_RETURN_FLOAT4(result);
}
