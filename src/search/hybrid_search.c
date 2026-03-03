/*-------------------------------------------------------------------------
 *
 * hybrid_search.c
 *    Hybrid search combining vector similarity with FTS and metadata.
 *
 * This file implements hybrid search capabilities that combine vector
 * similarity with full-text search (FTS), metadata filtering, keyword
 * matching, temporal awareness, and faceted search.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/search/hybrid_search.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "funcapi.h"
#include "executor/spi.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "miscadmin.h"
#include <math.h>
#include "access/htup_details.h"
#include "utils/lsyscache.h"
#include "utils/array.h"
#include "utils/varlena.h"
#include "utils/elog.h"
#include "utils/fmgrprotos.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_validation.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"

typedef struct mmr_cand_t
{
	char	   *id;
	Vector	   *vec;
	float		rel;
	bool		selected;
}			mmr_cand_t;

static inline char *
to_sql_literal(const char *val)
{
	StringInfoData buf;

	initStringInfo(&buf);
	appendStringInfoCharMacro(&buf, '\'');
	while (*val)
	{
		if (*val == '\'')
			appendStringInfoString(&buf, "''");
		else
			appendStringInfoChar(&buf, *val);
		val++;
	}
	appendStringInfoCharMacro(&buf, '\'');
	return buf.data;
}

static inline char *
plain_to_tsquery_format(const char *text)
{
	StringInfoData buf;
	const char *p = text;
	bool first = true;

	initStringInfo(&buf);
	
	/* Convert plain text to tsquery format: "word1 & word2 & word3" */
	while (*p)
	{
		/* Skip whitespace */
		while (*p && isspace((unsigned char) *p))
			p++;
		
		if (!*p)
			break;
		
		/* Add & operator between words */
		if (!first)
			appendStringInfoString(&buf, " & ");
		
		/* Add word (escape special tsquery chars) */
		while (*p && !isspace((unsigned char) *p))
		{
			char c = *p;
			/* Escape special tsquery characters: & | ! ( ) */
			if (c == '&' || c == '|' || c == '!' || c == '(' || c == ')')
				appendStringInfoChar(&buf, '\\');
			appendStringInfoChar(&buf, c);
			p++;
		}
		
		first = false;
	}
	
	return buf.data;
}

static void
__attribute__((unused))
vector_to_float_array(const Vector *vec, float *arr, int dim)
{
	int			i;

	for (i = 0; i < dim; i++)
		arr[i] = vec->data[i];
}

typedef struct HybridSearchState
{
	int			num_results;
	int			current_idx;
	int64	   *ids;
	float4	   *scores;
}			HybridSearchState;

PG_FUNCTION_INFO_V1(hybrid_search);
Datum
hybrid_search(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx = NULL;
	HybridSearchState *state = NULL;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		TupleDesc	tupdesc;
		text	   *table_name = NULL;
		Vector *query_vec = NULL;
		text	   *query_text = NULL;
		text	   *filters = NULL;
		float8		vector_weight;
		int32		limit;

	char	   *tbl_str = NULL;
	char	   *txt_str = NULL;
	char	   *filter_str = NULL;
	StringInfoData sql;
	StringInfoData vec_lit;
	int			spi_ret;
	int			i;
	int			proc;
	NdbSpiSession *session = NULL;
	text *query_type = NULL;
	char *qtype_str = NULL;
	const char *query_func = NULL;
	char *query_text_formatted = NULL;
	char *query_text_to_use = NULL;

		/* Validate argument count */
		if (PG_NARGS() < 6 || PG_NARGS() > 7)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: hybrid_search requires 6-7 arguments")));

		table_name = PG_GETARG_TEXT_PP(0);
		query_text = PG_GETARG_TEXT_PP(2);
		filters = PG_GETARG_TEXT_PP(3);
		vector_weight = PG_GETARG_FLOAT8(4);
		limit = PG_GETARG_INT32(5);
		
		/* Optional 7th parameter: query_type for FTS */
		if (PG_NARGS() >= 7 && !PG_ARGISNULL(6))
		{
			query_type = PG_GETARG_TEXT_PP(6);
			qtype_str = text_to_cstring(query_type);
		}
		else
		{
			qtype_str = pstrdup("plain");  /* Default to plainto_tsquery */
		}

		if (PG_ARGISNULL(1))
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("hybrid_search: query vector cannot be NULL")));

		query_vec = PG_GETARG_VECTOR_P(1);
		NDB_CHECK_VECTOR_VALID(query_vec);

		if (query_vec == NULL)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("hybrid_search: query vector is NULL")));

		if (VARSIZE(query_vec) < VARHDRSZ + sizeof(int16) * 2)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("hybrid_search: invalid vector size")));

		if (query_vec->dim <= 0 || query_vec->dim > 32767)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("hybrid_search: invalid vector dimension %d", query_vec->dim)));

		if (VARSIZE(query_vec) < VARHDRSZ + sizeof(int16) * 2 + sizeof(float4) * query_vec->dim)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("hybrid_search: vector size mismatch (dim=%d, size=%d)",
							query_vec->dim, VARSIZE(query_vec))));

		tbl_str = text_to_cstring(table_name);
		txt_str = text_to_cstring(query_text);
		filter_str = text_to_cstring(filters);


		session = ndb_spi_session_begin(CurrentMemoryContext, false);
		if (session == NULL)
			ereport(ERROR, (errmsg("hybrid_search: failed to begin SPI session")));

		initStringInfo(&vec_lit);
		appendStringInfoChar(&vec_lit, '{');
		for (i = 0; i < query_vec->dim; i++)
		{
			float4		val;

			if (i)
				appendStringInfoChar(&vec_lit, ',');

			val = query_vec->data[i];

			if (!isfinite(val))
			{
				pfree(vec_lit.data);
				ndb_spi_session_end(&session);
				pfree(tbl_str);
				pfree(txt_str);
				pfree(filter_str);
				pfree(qtype_str);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("hybrid_search: non-finite value in vector at index %d", i)));
			}
			appendStringInfo(&vec_lit, "%g", val);
		}
		appendStringInfoChar(&vec_lit, '}');

	/* Determine query function based on query_type */
	query_func = "plainto_tsquery";
	query_text_formatted = NULL;
	
	if (strcmp(qtype_str, "to") == 0 || strcmp(qtype_str, "to_tsquery") == 0)
		{
			query_func = "to_tsquery";
			/* Convert plain text to tsquery format (space-separated words -> word1 & word2) */
			query_text_formatted = plain_to_tsquery_format(txt_str);
		}
		else if (strcmp(qtype_str, "phrase") == 0 || strcmp(qtype_str, "phraseto_tsquery") == 0)
		{
			query_func = "phraseto_tsquery";
			query_text_formatted = NULL; /* Use original text */
		}
		else
		{
			query_text_formatted = NULL; /* Use original text for plainto_tsquery */
		}

	/* Use formatted text if available, otherwise use original */
	query_text_to_use = query_text_formatted ? query_text_formatted : txt_str;

	initStringInfo(&sql);

		appendStringInfo(&sql,
						 "WITH _hybrid_scores AS ("
						 " SELECT id,"
						 "        (1 - (embedding <-> '%s'::vector)) AS vector_score,"
						 "        ts_rank(fts_vector, %s('english', %s)) AS fts_score,"
						 "        metadata "
						 "   FROM %s "
						 "  WHERE metadata @> %s "
						 ") "
						 "SELECT id, hybrid_score "
						 " FROM (SELECT id, "
						 "              (%f * vector_score + (1 - %f) * fts_score) as "
						 "hybrid_score "
						 "         FROM _hybrid_scores) H "
						 " ORDER BY hybrid_score DESC "
						 " LIMIT %d;",
						 vec_lit.data,
						 query_func,
						 to_sql_literal(query_text_to_use),
						 tbl_str,
						 to_sql_literal(filter_str),
						 vector_weight,
						 vector_weight,
						 limit);

		spi_ret = ndb_spi_execute(session, sql.data, true, limit);
		if (spi_ret != SPI_OK_SELECT)
		{
			pfree(sql.data);
			pfree(vec_lit.data);
			ndb_spi_session_end(&session);
			pfree(tbl_str);
			pfree(txt_str);
			pfree(filter_str);
			if (query_text_formatted)
				pfree(query_text_formatted);
			pfree(qtype_str);
			ereport(ERROR, (errmsg("Failed to execute hybrid search SQL")));
		}
		
		/* Clean up formatted text if it was allocated */
		if (query_text_formatted)
			pfree(query_text_formatted);

		proc = SPI_processed;

		/* Initialize SRF context */
		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		/* Allocate state */
		nalloc(state, HybridSearchState, 1);
		NDB_CHECK_ALLOC(state, "state");

		if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("function returning record called in context that cannot accept type record")));
		funcctx->tuple_desc = BlessTupleDesc(tupdesc);

		if (proc == 0)
		{
			state->num_results = 0;
			state->ids = NULL;
			state->scores = NULL;
		}
		else
		{
			int64 *ids = NULL;
			float4 *scores = NULL;

			state->num_results = proc;
			NBP_ALLOC(ids, int64, proc);
			NDB_CHECK_ALLOC(ids, "state->ids");
			NBP_ALLOC(scores, float4, proc);
			NDB_CHECK_ALLOC(scores, "state->scores");
			state->ids = ids;
			state->scores = scores;

			for (i = 0; i < proc; i++)
			{
				bool		isnull_id,
							isnull_score;
				Datum		id_val,
							score_val;

				id_val = SPI_getbinval(SPI_tuptable->vals[i],
									   SPI_tuptable->tupdesc,
									   1,
									   &isnull_id);

				score_val = SPI_getbinval(SPI_tuptable->vals[i],
										  SPI_tuptable->tupdesc,
										  2,
										  &isnull_score);

				if (!isnull_id)
				{
					Oid			id_type = SPI_gettypeid(SPI_tuptable->tupdesc, 1);

					if (id_type == INT8OID)
						state->ids[i] = DatumGetInt64(id_val);
					else if (id_type == INT4OID)
						state->ids[i] = (int64) DatumGetInt32(id_val);
					else
						state->ids[i] = 0;
				}
				else
					state->ids[i] = 0;

				if (!isnull_score)
					state->scores[i] = DatumGetFloat4(score_val);
				else
					state->scores[i] = 0.0f;
			}
		}

		funcctx->user_fctx = state;
		funcctx->max_calls = proc;

		pfree(sql.data);
		pfree(vec_lit.data);
		pfree(tbl_str);
		pfree(txt_str);
		pfree(filter_str);
		if (query_text_formatted)
			pfree(query_text_formatted);
		pfree(qtype_str);
		ndb_spi_session_end(&session);
		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();
	state = (HybridSearchState *) funcctx->user_fctx;

	if (funcctx->call_cntr < funcctx->max_calls)
	{
		Datum		values[2];
		bool		nulls[2] = {false, false};
		HeapTuple	tuple;

		values[0] = Int64GetDatum(state->ids[funcctx->call_cntr]);
		values[1] = Float4GetDatum(state->scores[funcctx->call_cntr]);

		tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
		SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
	}
	else
	{
		SRF_RETURN_DONE(funcctx);
	}
}

typedef struct RrfItem
{
	const char *key;
	float8		score;
}			RrfItem;

static int
compare_rrf_items(const void *a, const void *b)
{
	const		RrfItem *item_a = (const RrfItem *) a;
	const		RrfItem *item_b = (const RrfItem *) b;

	if (item_b->score > item_a->score)
		return 1;
	if (item_b->score < item_a->score)
		return -1;
	return 0;
}

PG_FUNCTION_INFO_V1(reciprocal_rank_fusion);
Datum
reciprocal_rank_fusion(PG_FUNCTION_ARGS)
{
	ArrayType  *rankings = NULL;
	float8		k;
	int			n_rankers;
	int			i;
	int			j;
	int16		elmlen;
	bool		elmbyval;
	char		elmalign;
	ArrayType  **rank_arrays;
	int			item_count = 0;
	HTAB		   *item_hash = NULL;
	HASHCTL		info;
	Datum		   *result_datums = NULL;
	bool		   *result_nulls = NULL;
	ArrayType  *ret_array = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: reciprocal_rank_fusion requires 2 arguments")));

	rankings = PG_GETARG_ARRAYTYPE_P(0);
	k = PG_GETARG_FLOAT8(1);


	if (ARR_NDIM(rankings) != 1)
		ereport(ERROR,
				(errmsg("rankings argument must be 1-dimensional array "
						"of arrays")));

	n_rankers = ARR_DIMS(rankings)[0];

	deconstruct_array(rankings,
					  ANYARRAYOID,
					  -1,
					  false,
					  'd',
					  (Datum * *) & rank_arrays,
					  NULL,
					  &n_rankers);

	memset(&info, 0, sizeof(info));
	info.keysize = 512;
	info.entrysize = sizeof(RrfItem);
	item_hash =
		hash_create("RRFItems", 1024, &info, HASH_ELEM | HASH_BLOBS);

	for (i = 0; i < n_rankers; i++)
	{
		ArrayType  *ranker = rank_arrays[i];
		int			count = ArrayGetNItems(ARR_NDIM(ranker), ARR_DIMS(ranker));
		Oid			elemtype = ARR_ELEMTYPE(ranker);
		Datum *ids = NULL;
		bool *nulls = NULL;

		get_typlenbyvalalign(elemtype, &elmlen, &elmbyval, &elmalign);

		deconstruct_array(ranker,
						  elemtype,
						  elmlen,
						  elmbyval,
						  elmalign,
						  &ids,
						  &nulls,
						  &count);
		for (j = 0; j < count; j++)
		{
			char		id[512];

			if (nulls[j])
				continue;
			{
				text	   *t = DatumGetTextPP(ids[j]);
				int			len = VARSIZE_ANY_EXHDR(t);

				memcpy(id, VARDATA_ANY(t), len);
				id[len] = '\0';
			}
			{
				bool		found;
				struct
				{
					char		key[512];
					float8		score;
				}		   *entry;

				entry = hash_search(
									item_hash, id, HASH_ENTER, &found);
				if (!found)
					entry->score = 0;
				entry->score += 1.0 / (k + (double) j + 1.0);
			}
		}
		pfree(ids);
		pfree(nulls);
	}

	/* Output: sort the ids by score descending, return as text[] */
	item_count = hash_get_num_entries(item_hash);
	nalloc(result_datums, Datum, item_count);
	NDB_CHECK_ALLOC(result_datums, "allocation");
	MemSet(result_datums, 0, sizeof(Datum) * item_count);
	nalloc(result_nulls, bool, item_count);
	NDB_CHECK_ALLOC(result_nulls, "allocation");
	MemSet(result_nulls, 0, sizeof(bool) * item_count);
	NDB_CHECK_ALLOC(result_nulls, "allocation");

	{
		struct
		{
			char		key[512];
			float8		score;
		}		   *cur = NULL;
		struct HybridSearchItem
		{
			char *key;
			float8		score;
		}		   *items = NULL;
		HASH_SEQ_STATUS stat;
		int			idx = 0;
		int			idx_i,
					idx_j;

		nalloc(items, struct HybridSearchItem, item_count);
		NDB_CHECK_ALLOC(items, "allocation");
		hash_seq_init(&stat, item_hash);
		while ((cur = hash_seq_search(&stat)) != NULL)
		{
			items[idx].key = pstrdup(cur->key);
			items[idx].score = cur->score;
			idx++;
		}

		/*
		 * Sort by decreasing RRF score (bubble sort for small sets, qsort for
		 * larger)
		 */
		if (item_count <= 100)
		{
			/* Bubble sort for small sets */
			for (idx_i = 0; idx_i < item_count - 1; idx_i++)
			{
				for (idx_j = idx_i + 1; idx_j < item_count; idx_j++)
				{
					if (items[idx_j].score > items[idx_i].score)
					{
						char	   *temp_key = items[idx_i].key;
						float8		temp_score = items[idx_i].score;

						items[idx_i].key = items[idx_j].key;
						items[idx_i].score = items[idx_j].score;
						items[idx_j].key = temp_key;
						items[idx_j].score = temp_score;
					}
				}
			}
		}
		else
		{
			/* qsort for larger sets */
			qsort(items,
				  item_count,
				  sizeof(*items),
				  compare_rrf_items);
		}

		for (idx = 0; idx < item_count; idx++)
		{
			result_datums[idx] =
				PointerGetDatum(cstring_to_text(items[idx].key));
			result_nulls[idx] = false;
			pfree(items[idx].key);
		}

		pfree(items);
	}

	ret_array = construct_array(
								result_datums, item_count, TEXTOID, -1, false, 'i');

	hash_destroy(item_hash);
	pfree(result_datums);
	pfree(result_nulls);

	PG_RETURN_ARRAYTYPE_P(ret_array);
}

PG_FUNCTION_INFO_V1(semantic_keyword_search);
Datum
semantic_keyword_search(PG_FUNCTION_ARGS)
{
	text *table_name = NULL;
	Vector *semantic_query = NULL;
	text *keyword_query = NULL;
	int32		top_k;
	char *tbl_str = NULL;
	char *kw_str = NULL;
	StringInfoData sql;
	StringInfoData vec_lit;
	int			spi_ret;
	ArrayType *ret_array = NULL;
	int			proc;
	int			i;

	Datum *datums = NULL;
	bool *nulls = NULL;
	NdbSpiSession *session = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 4)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: semantic_keyword_search requires 4 arguments")));

	table_name = PG_GETARG_TEXT_PP(0);
	semantic_query = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(semantic_query);
	keyword_query = PG_GETARG_TEXT_PP(2);
	top_k = PG_GETARG_INT32(3);

	tbl_str = text_to_cstring(table_name);
	kw_str = text_to_cstring(keyword_query);

	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		ereport(ERROR, (errmsg("semantic_keyword_search: failed to begin SPI session")));

	initStringInfo(&vec_lit);
	appendStringInfoChar(&vec_lit, '{');
	for (i = 0; i < semantic_query->dim; i++)
	{
		if (i)
			appendStringInfoChar(&vec_lit, ',');
		appendStringInfo(
						 &vec_lit, "%g", ((float *) &semantic_query[1])[i]);
	}
	appendStringInfoChar(&vec_lit, '}');

	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT id FROM ("
					 " SELECT id,"
					 "        (1 - (embedding <-> '%s'::vector)) AS semantic_score,"
					 "        ts_rank_cd(fts_vector, plainto_tsquery('%s')) AS "
					 "bm25_score,"
					 "        ((1 - (embedding <-> '%s'::vector)) + "
					 "ts_rank_cd(fts_vector, plainto_tsquery('%s'))) AS "
					 "hybrid_score "
					 "   FROM %s "
					 "  WHERE fts_vector @@ plainto_tsquery('%s')"
					 ") scores "
					 "ORDER BY hybrid_score DESC "
					 "LIMIT %d;",
					 vec_lit.data,
					 kw_str,
					 vec_lit.data,
					 kw_str,
					 tbl_str,
					 kw_str,
					 top_k);

	spi_ret = ndb_spi_execute(session, sql.data, true, top_k);
	if (spi_ret != SPI_OK_SELECT)
	{
		pfree(sql.data);
		pfree(vec_lit.data);
		pfree(tbl_str);
		pfree(kw_str);
		ndb_spi_session_end(&session);
		ereport(ERROR,
				(errmsg("Failed to execute semantic_keyword_search "
						"SQL")));
	}

	proc = SPI_processed;
	if (proc == 0)
	{
		pfree(sql.data);
		pfree(vec_lit.data);
		pfree(tbl_str);
		pfree(kw_str);
		ndb_spi_session_end(&session);
		ret_array = construct_empty_array(TEXTOID);
		PG_RETURN_ARRAYTYPE_P(ret_array);
	}

	nalloc(datums, Datum, proc);
	NDB_CHECK_ALLOC(datums, "allocation");
	nalloc(nulls, bool, proc);
	NDB_CHECK_ALLOC(nulls, "allocation");
	for (i = 0; i < proc; i++)
	{
		bool		isnull;
		Datum		val = SPI_getbinval(SPI_tuptable->vals[i],
										SPI_tuptable->tupdesc,
										1,
										&isnull);

		if (!isnull)
			datums[i] = PointerGetDatum(
										cstring_to_text(DatumGetCString(val)));
		nulls[i] = isnull;
	}

	ret_array = construct_array(datums, proc, TEXTOID, -1, false, 'i');

	pfree(datums);
	pfree(nulls);
	pfree(sql.data);
	pfree(vec_lit.data);
	pfree(tbl_str);
	pfree(kw_str);
	ndb_spi_session_end(&session);

	PG_RETURN_ARRAYTYPE_P(ret_array);
}

PG_FUNCTION_INFO_V1(multi_vector_search);
Datum
multi_vector_search(PG_FUNCTION_ARGS)
{
	text	   *table_name = NULL;
	ArrayType  *query_vectors = NULL;
	text	   *agg_method = NULL;
	int32		top_k;

	/* Validate argument count */
	if (PG_NARGS() != 4)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: multi_vector_search requires 4 arguments")));

	{
		char		   *tbl_str = NULL;
		char		   *agg_str = NULL;
		int				nvecs;
		StringInfoData sql;
		StringInfoData subquery;
		Datum		   *vec_datums = NULL;
		bool		   *vec_nulls = NULL;
		Oid				vec_elemtype;
		int				i;
		int				spi_ret;
		int				proc;
		ArrayType	   *ret_array = NULL;
		Datum		   *datums = NULL;
		bool		   *nulls = NULL;
		NdbSpiSession  *session = NULL;

		table_name = PG_GETARG_TEXT_PP(0);
		query_vectors = PG_GETARG_ARRAYTYPE_P(1);
		agg_method = PG_GETARG_TEXT_PP(2);
		top_k = PG_GETARG_INT32(3);

		tbl_str = text_to_cstring(table_name);
		agg_str = text_to_cstring(agg_method);
		vec_elemtype = ARR_ELEMTYPE(query_vectors);

		nvecs = ArrayGetNItems(
							   ARR_NDIM(query_vectors), ARR_DIMS(query_vectors));
		session = ndb_spi_session_begin(CurrentMemoryContext, false);
		if (session == NULL)
			ereport(ERROR, (errmsg("multi_vector_search: failed to begin SPI session")));

		get_typlenbyvalalign(vec_elemtype, NULL, NULL, NULL);
		deconstruct_array(query_vectors,
						  vec_elemtype,
						  -1,
						  false,
						  'i',
						  &vec_datums,
						  &vec_nulls,
						  &nvecs);

		if (nvecs < 1)
		{
			pfree(tbl_str);
			pfree(agg_str);
			ndb_spi_session_end(&session);
			ereport(ERROR,
					(errmsg("multi_vector_search: at least one query "
							"vector required")));
		}

		{
			Vector	   *first_vec = (Vector *) DatumGetPointer(vec_datums[0]);

			if (first_vec->dim <= 0)
			{
				pfree(tbl_str);
				pfree(agg_str);
				pfree(vec_datums);
				pfree(vec_nulls);
				ndb_spi_session_end(&session);
				ereport(ERROR,
						(errmsg("query vectors must have positive "
								"dimension")));
			}
		}

		initStringInfo(&subquery);
		for (i = 0; i < nvecs; i++)
		{
			if (vec_nulls[i])
				continue;
			{
				Vector	   *qv = (Vector *) DatumGetPointer(vec_datums[i]);
				StringInfoData lit;
				int			j;

				initStringInfo(&lit);
				appendStringInfoChar(&lit, '{');
				for (j = 0; j < qv->dim; j++)
				{
					if (j)
						appendStringInfoChar(&lit, ',');
					appendStringInfo(
									 &lit, "%g", ((float *) &qv[1])[j]);
				}
				appendStringInfoChar(&lit, '}');
				if (i)
					appendStringInfoString(&subquery, ", ");
				appendStringInfo(&subquery, "'%s'::vector", lit.data);
				pfree(lit.data);
			}
		}
		pfree(vec_datums);
		pfree(vec_nulls);

		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "SELECT id FROM ("
						 "  SELECT id, "
						 "         GREATEST(%s) as max_score "
						 "    FROM ("
						 "      SELECT id, "
						 "             (1 - (embedding <-> ANY(ARRAY[%s]))) as "
						 "agg_score "
						 "        FROM %s"
						 "    ) _agg "
						 " ) z "
						 "ORDER BY max_score DESC LIMIT %d;",
						 strcmp(agg_str, "max") == 0 ? "agg_score" : "avg(agg_score)",
						 subquery.data,
						 tbl_str,
						 top_k);

		spi_ret = ndb_spi_execute(session, sql.data, true, top_k);
		if (spi_ret != SPI_OK_SELECT)
		{
			pfree(sql.data);
			pfree(subquery.data);
			pfree(tbl_str);
			pfree(agg_str);
			ndb_spi_session_end(&session);
			ereport(ERROR,
					(errmsg("Failed to execute multi_vector_search SQL")));
		}
		proc = SPI_processed;
		if (proc == 0)
		{
			pfree(sql.data);
			pfree(subquery.data);
			pfree(tbl_str);
			pfree(agg_str);
			ndb_spi_session_end(&session);
			ret_array = construct_empty_array(TEXTOID);
			PG_RETURN_ARRAYTYPE_P(ret_array);
		}
		datums = NULL;
		nulls = NULL;
		nalloc(datums, Datum, proc);
		NDB_CHECK_ALLOC(datums, "allocation");
		nalloc(nulls, bool, proc);
		NDB_CHECK_ALLOC(nulls, "allocation");
		for (i = 0; i < proc; i++)
		{
			bool		isnull;
			Datum		val = SPI_getbinval(SPI_tuptable->vals[i],
											SPI_tuptable->tupdesc,
											1,
											&isnull);

			if (!isnull)
				datums[i] = PointerGetDatum(
											cstring_to_text(DatumGetCString(val)));
			nulls[i] = isnull;
		}
		ret_array = construct_array(datums, proc, TEXTOID, -1, false, 'i');
		pfree(datums);
		pfree(nulls);
		pfree(sql.data);
		pfree(subquery.data);
		pfree(tbl_str);
		pfree(agg_str);
		ndb_spi_session_end(&session);

		PG_RETURN_ARRAYTYPE_P(ret_array);
	}
}

PG_FUNCTION_INFO_V1(faceted_vector_search);
Datum
faceted_vector_search(PG_FUNCTION_ARGS)
{
	text *table_name = NULL;
	Vector *query_vec = NULL;
	text *facet_column = NULL;
	int32		per_facet_limit;
	char *tbl_str = NULL;
	char *facet_str = NULL;
	StringInfoData sql;
	StringInfoData vec_lit;
	int			spi_ret;
	int			proc;
	ArrayType *ret_array = NULL;
	Datum *datums = NULL;
	bool *nulls = NULL;
	int			i;

	NdbSpiSession *session = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 4)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: faceted_vector_search requires 4 arguments")));

	table_name = PG_GETARG_TEXT_PP(0);
	query_vec = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(query_vec);
	facet_column = PG_GETARG_TEXT_PP(2);
	per_facet_limit = PG_GETARG_INT32(3);
	tbl_str = text_to_cstring(table_name);
	facet_str = text_to_cstring(facet_column);

	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		ereport(ERROR, (errmsg("faceted_vector_search: failed to begin SPI session")));

	initStringInfo(&vec_lit);
	appendStringInfoChar(&vec_lit, '{');
	for (i = 0; i < query_vec->dim; i++)
	{
		if (i)
			appendStringInfoChar(&vec_lit, ',');
		appendStringInfo(&vec_lit, "%g", ((float *) &query_vec[1])[i]);
	}
	appendStringInfoChar(&vec_lit, '}');

	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT id FROM ("
					 "   SELECT id,%s,(1 - (embedding <-> '%s'::vector)) AS vec_sim,"
					 "          ROW_NUMBER() OVER (PARTITION BY %s ORDER BY (1 - "
					 "(embedding <-> '%s'::vector)) DESC) AS rn "
					 "     FROM %s"
					 " ) faceted "
					 "WHERE rn <= %d "
					 "ORDER BY %s, vec_sim DESC;",
					 facet_str,
					 vec_lit.data,
					 facet_str,
					 vec_lit.data,
					 tbl_str,
					 per_facet_limit,
					 facet_str);

	spi_ret = ndb_spi_execute(session, sql.data, true, 0);
	if (spi_ret != SPI_OK_SELECT)
	{
		pfree(sql.data);
		pfree(vec_lit.data);
		pfree(tbl_str);
		pfree(facet_str);
		ndb_spi_session_end(&session);
		ereport(ERROR,
				(errmsg("Failed to execute faceted_vector_search "
						"SQL")));
	}
	proc = SPI_processed;
	if (proc == 0)
	{
		pfree(sql.data);
		pfree(vec_lit.data);
		pfree(tbl_str);
		pfree(facet_str);
		ndb_spi_session_end(&session);
		ret_array = construct_empty_array(TEXTOID);
		PG_RETURN_ARRAYTYPE_P(ret_array);
	}
	datums = NULL;
	nulls = NULL;
	nalloc(datums, Datum, proc);
	NDB_CHECK_ALLOC(datums, "allocation");
	nalloc(nulls, bool, proc);
	NDB_CHECK_ALLOC(nulls, "allocation");
	for (i = 0; i < proc; i++)
	{
		bool		isnull;
		Datum		val = SPI_getbinval(SPI_tuptable->vals[i],
										SPI_tuptable->tupdesc,
										1,
										&isnull);

		if (!isnull)
			datums[i] = PointerGetDatum(
										cstring_to_text(DatumGetCString(val)));
		nulls[i] = isnull;
	}
	ret_array = construct_array(datums, proc, TEXTOID, -1, false, 'i');
	pfree(datums);
	pfree(nulls);
	pfree(sql.data);
	pfree(vec_lit.data);
	pfree(tbl_str);
	pfree(facet_str);
	ndb_spi_session_end(&session);

	PG_RETURN_ARRAYTYPE_P(ret_array);
}

PG_FUNCTION_INFO_V1(temporal_vector_search);
Datum
temporal_vector_search(PG_FUNCTION_ARGS)
{
	text *table_name = NULL;
	Vector *query_vec = NULL;
	text *timestamp_col = NULL;
	float8		decay_rate;
	int32		top_k;
	char *tbl_str = NULL;
	char *ts_str = NULL;
	StringInfoData sql;
	StringInfoData vec_lit;
	int			spi_ret;
	int			proc;
	ArrayType *ret_array = NULL;
	Datum *datums = NULL;
	bool *nulls = NULL;
	int			i;

	NdbSpiSession *session = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 5)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: temporal_vector_search requires 5 arguments")));

	table_name = PG_GETARG_TEXT_PP(0);
	query_vec = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(query_vec);
	timestamp_col = PG_GETARG_TEXT_PP(2);
	decay_rate = PG_GETARG_FLOAT8(3);
	top_k = PG_GETARG_INT32(4);
	tbl_str = text_to_cstring(table_name);
	ts_str = text_to_cstring(timestamp_col);

	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		ereport(ERROR, (errmsg("temporal_vector_search: failed to begin SPI session")));

	initStringInfo(&vec_lit);
	appendStringInfoChar(&vec_lit, '{');
	for (i = 0; i < query_vec->dim; i++)
	{
		if (i)
			appendStringInfoChar(&vec_lit, ',');
		appendStringInfo(&vec_lit, "%g", ((float *) &query_vec[1])[i]);
	}
	appendStringInfoChar(&vec_lit, '}');

	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT id FROM ("
					 "  SELECT id,"
					 "         (1 - (embedding <-> '%s'::vector)) AS vec_score,"
					 "         EXTRACT(EPOCH FROM (now() - %s))/86400 AS age_days,"
					 "         ((1 - (embedding <-> '%s'::vector)) * exp(-(%f) * "
					 "(EXTRACT(EPOCH FROM (now() - %s))/86400))) AS tempo_score "
					 "    FROM %s"
					 "  ) temporal "
					 "ORDER BY tempo_score DESC "
					 "LIMIT %d;",
					 vec_lit.data,
					 ts_str,
					 vec_lit.data,
					 decay_rate,
					 ts_str,
					 tbl_str,
					 top_k);

	spi_ret = ndb_spi_execute(session, sql.data, true, top_k);
	if (spi_ret != SPI_OK_SELECT)
	{
		pfree(sql.data);
		pfree(vec_lit.data);
		pfree(tbl_str);
		pfree(ts_str);
		ndb_spi_session_end(&session);
		ereport(ERROR,
				(errmsg("Failed to execute temporal_vector_search "
						"SQL")));
	}

	proc = SPI_processed;
	if (proc == 0)
	{
		pfree(sql.data);
		pfree(vec_lit.data);
		pfree(tbl_str);
		pfree(ts_str);
		ndb_spi_session_end(&session);
		ret_array = construct_empty_array(TEXTOID);
		PG_RETURN_ARRAYTYPE_P(ret_array);
	}
	datums = NULL;
	nulls = NULL;
	nalloc(datums, Datum, proc);
	NDB_CHECK_ALLOC(datums, "allocation");
	nalloc(nulls, bool, proc);
	NDB_CHECK_ALLOC(nulls, "allocation");
	for (i = 0; i < proc; i++)
	{
		bool		isnull;
		Datum		val = SPI_getbinval(SPI_tuptable->vals[i],
										SPI_tuptable->tupdesc,
										1,
										&isnull);

		if (!isnull)
			datums[i] = PointerGetDatum(
										cstring_to_text(DatumGetCString(val)));
		nulls[i] = isnull;
	}
	ret_array = construct_array(datums, proc, TEXTOID, -1, false, 'i');
	pfree(datums);
	pfree(nulls);
	pfree(sql.data);
	pfree(vec_lit.data);
	pfree(tbl_str);
	pfree(ts_str);
	ndb_spi_session_end(&session);

	PG_RETURN_ARRAYTYPE_P(ret_array);
}

PG_FUNCTION_INFO_V1(diverse_vector_search);
Datum
diverse_vector_search(PG_FUNCTION_ARGS)
{
	text *table_name = NULL;
	Vector *query_vec = NULL;
	float8		lambda;
	int32		top_k;
	char *tbl_str = NULL;
	StringInfoData sql;
	StringInfoData vec_lit;
	int			spi_ret;
	int			proc;
	int			i,
				j;
	ArrayType *ret_array = NULL;
	Datum *datums = NULL;
	bool *nulls = NULL;
	int			n_candidates;
	int			select_count = 0;
	mmr_cand_t *cands = NULL;

	NdbSpiSession *session = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 4)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: diverse_vector_search requires 4 arguments")));

	table_name = PG_GETARG_TEXT_PP(0);
	query_vec = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(query_vec);
	lambda = PG_GETARG_FLOAT8(2);
	top_k = PG_GETARG_INT32(3);
	tbl_str = text_to_cstring(table_name);

	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		ereport(ERROR, (errmsg("diverse_vector_search: failed to begin SPI session")));

	initStringInfo(&vec_lit);
	appendStringInfoChar(&vec_lit, '{');
	for (i = 0; i < query_vec->dim; i++)
	{
		if (i)
			appendStringInfoChar(&vec_lit, ',');
		appendStringInfo(&vec_lit, "%g", ((float *) &query_vec[1])[i]);
	}
	appendStringInfoChar(&vec_lit, '}');

	/* First: get candidates with their relevance */
	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT id, embedding, (1 - (embedding <-> '%s'::vector)) AS "
					 "rel "
					 "FROM %s ORDER BY rel DESC LIMIT %d;",
					 vec_lit.data,
					 tbl_str,
					 top_k * 10);

	spi_ret = ndb_spi_execute(session, sql.data, true, top_k * 10);
	if (spi_ret != SPI_OK_SELECT)
	{
		pfree(sql.data);
		pfree(vec_lit.data);
		pfree(tbl_str);
		ndb_spi_session_end(&session);
		ereport(ERROR,
				(errmsg("Failed to execute diverse_vector_search "
						"SQL")));
	}

	proc = SPI_processed;
	if (proc == 0)
	{
		pfree(sql.data);
		pfree(vec_lit.data);
		pfree(tbl_str);
		ndb_spi_session_end(&session);
		ret_array = construct_empty_array(TEXTOID);
		PG_RETURN_ARRAYTYPE_P(ret_array);
	}

		n_candidates = proc;

	cands = NULL;
	nalloc(cands, mmr_cand_t, n_candidates);
	NDB_CHECK_ALLOC(cands, "allocation");

	for (i = 0; i < n_candidates; i++)
	{
		bool		isnull1;
		bool		isnull2;
		bool		isnull3;
		Datum		id_d = SPI_getbinval(SPI_tuptable->vals[i],
										 SPI_tuptable->tupdesc,
										 1,
										 &isnull1);
		Datum		vec_d = SPI_getbinval(SPI_tuptable->vals[i],
										  SPI_tuptable->tupdesc,
										  2,
										  &isnull2);
		Datum		rel_d = SPI_getbinval(SPI_tuptable->vals[i],
										  SPI_tuptable->tupdesc,
										  3,
										  &isnull3);

		cands[i].id = pstrdup(TextDatumGetCString(id_d));
		cands[i].vec = (Vector *) PG_DETOAST_DATUM(vec_d);
		cands[i].rel = DatumGetFloat4(rel_d);
		cands[i].selected = false;
	}

	nalloc(datums, Datum, top_k);
	NDB_CHECK_ALLOC(datums, "allocation");
	nalloc(nulls, bool, top_k);
	NDB_CHECK_ALLOC(nulls, "allocation");

	for (i = 0; i < top_k && select_count < n_candidates; i++)
	{
		float		best_score = -1e9f;
		int			best_idx = -1;

		for (j = 0; j < n_candidates; j++)
		{
			float		mmr_score;
			float		max_diverse = 0.0f;

			if (cands[j].selected)
				continue;

			if (i > 0)
			{
				int			s;

				for (s = 0; s < i; s++)
				{
					if (nulls[s])
						continue;
					{
						mmr_cand_t *sel = &cands[j];
						mmr_cand_t *other = &cands[s];
						float		dot = 0.0f;
						int			d;
						int			dim = sel->vec->dim;
						const float *x =
							(const float *) &sel
							->vec[1];
						const float *y =
							(const float *) &other
							->vec[1];

						for (d = 0; d < dim; d++)
							dot += x[d] * y[d];
						if (dot > max_diverse)
							max_diverse = dot;
					}
				}
			}

			mmr_score = lambda * cands[j].rel
				- (1.0f - lambda) * max_diverse;

			if (mmr_score > best_score)
			{
				best_score = mmr_score;
				best_idx = j;
			}
		}
		if (best_idx == -1)
			break;
		datums[i] =
			PointerGetDatum(cstring_to_text(cands[best_idx].id));
		nulls[i] = false;
		cands[best_idx].selected = true;
		select_count++;
	}

	ret_array =
		construct_array(datums, select_count, TEXTOID, -1, false, 'i');
	for (i = 0; i < n_candidates; i++)
	{
		pfree(cands[i].id);

		/*
		 * Only pfree detoasted vectors if address does not match the original
		 */
		{
			bool		dummy_isnull;
			Datum		orig = SPI_getbinval(SPI_tuptable->vals[i],
											 SPI_tuptable->tupdesc,
											 2,
											 &dummy_isnull);

			if ((void *) cands[i].vec
				!= (void *) DatumGetPointer(orig))
				pfree(cands[i].vec);
		}
	}
	pfree(datums);
	pfree(nulls);
	pfree(cands);
	pfree(sql.data);
	pfree(vec_lit.data);
	pfree(tbl_str);
	ndb_spi_session_end(&session);
	PG_RETURN_ARRAYTYPE_P(ret_array);
}

typedef struct FullTextSearchState
{
	int			num_results;
	int			current_idx;
	int64	   *ids;
	float4	   *scores;
}			FullTextSearchState;

PG_FUNCTION_INFO_V1(full_text_search);
Datum
full_text_search(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx = NULL;
	FullTextSearchState *state = NULL;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		TupleDesc	tupdesc;
		text	   *table_name = NULL;
		text	   *query_text = NULL;
		text	   *text_column = NULL;
		text	   *query_type = NULL;
		text	   *filters = NULL;
		int32		limit;

		char	   *tbl_str = NULL;
		char	   *txt_str = NULL;
		char	   *col_str = NULL;
		char	   *qtype_str = NULL;
		char	   *filter_str = NULL;
		StringInfoData sql;
		int			spi_ret;
		int			proc;
		NdbSpiSession *session = NULL;
		const char *query_func = NULL;

		/* Validate argument count */
		if (PG_NARGS() < 3 || PG_NARGS() > 6)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: full_text_search requires 3-6 arguments")));

		table_name = PG_GETARG_TEXT_PP(0);
		query_text = PG_GETARG_TEXT_PP(1);
		text_column = PG_ARGISNULL(2) ? NULL : PG_GETARG_TEXT_PP(2);
		query_type = PG_ARGISNULL(3) ? NULL : PG_GETARG_TEXT_PP(3);
		filters = PG_ARGISNULL(4) ? NULL : PG_GETARG_TEXT_PP(4);
		limit = PG_ARGISNULL(5) ? 10 : PG_GETARG_INT32(5);

		if (limit <= 0 || limit > 10000)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("full_text_search: limit must be between 1 and 10000")));

		tbl_str = text_to_cstring(table_name);
		txt_str = text_to_cstring(query_text);
		col_str = text_column ? text_to_cstring(text_column) : pstrdup("fts_vector");
		qtype_str = query_type ? text_to_cstring(query_type) : pstrdup("plain");
		filter_str = filters ? text_to_cstring(filters) : pstrdup("{}");

		session = ndb_spi_session_begin(CurrentMemoryContext, false);
		if (session == NULL)
			ereport(ERROR, (errmsg("full_text_search: failed to begin SPI session")));

		initStringInfo(&sql);

		/* Determine query function based on query_type */
		query_func = "plainto_tsquery";
		if (strcmp(qtype_str, "to") == 0 || strcmp(qtype_str, "to_tsquery") == 0)
			query_func = "to_tsquery";
		else if (strcmp(qtype_str, "phrase") == 0 || strcmp(qtype_str, "phraseto_tsquery") == 0)
			query_func = "phraseto_tsquery";
		/* else default to plainto_tsquery */

		appendStringInfo(&sql,
						 "SELECT id, ts_rank(%s, %s(%s)) AS score "
						 "  FROM %s "
						 " WHERE %s @@ %s(%s) ",
						 col_str,
						 query_func,
						 to_sql_literal(txt_str),
						 tbl_str,
						 col_str,
						 query_func,
						 to_sql_literal(txt_str));

		/* Add metadata filters if provided */
		if (strcmp(filter_str, "{}") != 0)
		{
			appendStringInfo(&sql, " AND metadata @> %s ", to_sql_literal(filter_str));
		}

		appendStringInfo(&sql,
						 " ORDER BY score DESC "
						 " LIMIT %d;",
						 limit);

		spi_ret = ndb_spi_execute(session, sql.data, true, limit);
		if (spi_ret != SPI_OK_SELECT)
		{
			pfree(sql.data);
			ndb_spi_session_end(&session);
			pfree(tbl_str);
			pfree(txt_str);
			pfree(col_str);
			pfree(qtype_str);
			pfree(filter_str);
			ereport(ERROR, (errmsg("Failed to execute full_text_search SQL")));
		}

		proc = SPI_processed;

		/* Initialize SRF context */
		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		/* Allocate state */
		nalloc(state, FullTextSearchState, 1);
		NDB_CHECK_ALLOC(state, "state");

		if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("function returning record called in context that cannot accept type record")));
		funcctx->tuple_desc = BlessTupleDesc(tupdesc);

		if (proc == 0)
		{
			state->num_results = 0;
			state->ids = NULL;
			state->scores = NULL;
		}
		else
		{
			int64 *ids = NULL;
			float4 *scores = NULL;

			state->num_results = proc;
			NBP_ALLOC(ids, int64, proc);
			NDB_CHECK_ALLOC(ids, "state->ids");
			NBP_ALLOC(scores, float4, proc);
			NDB_CHECK_ALLOC(scores, "state->scores");
			state->ids = ids;
			state->scores = scores;

			for (int i = 0; i < proc; i++)
			{
				bool		isnull_id,
							isnull_score;
				Datum		id_val,
							score_val;

				id_val = SPI_getbinval(SPI_tuptable->vals[i],
									   SPI_tuptable->tupdesc,
									   1,
									   &isnull_id);

				score_val = SPI_getbinval(SPI_tuptable->vals[i],
										  SPI_tuptable->tupdesc,
										  2,
										  &isnull_score);

				if (!isnull_id)
				{
					Oid			id_type = SPI_gettypeid(SPI_tuptable->tupdesc, 1);

					if (id_type == INT8OID)
						state->ids[i] = DatumGetInt64(id_val);
					else if (id_type == INT4OID)
						state->ids[i] = (int64) DatumGetInt32(id_val);
					else
						state->ids[i] = 0;
				}
				else
					state->ids[i] = 0;

				if (!isnull_score)
					state->scores[i] = DatumGetFloat4(score_val);
				else
					state->scores[i] = 0.0f;
			}
		}

		funcctx->user_fctx = state;
		funcctx->max_calls = proc;

		pfree(sql.data);
		pfree(tbl_str);
		pfree(txt_str);
		pfree(col_str);
		pfree(qtype_str);
		pfree(filter_str);
		ndb_spi_session_end(&session);
		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();
	state = (FullTextSearchState *) funcctx->user_fctx;

	if (funcctx->call_cntr < funcctx->max_calls)
	{
		Datum		values[2];
		bool		nulls[2] = {false, false};
		HeapTuple	tuple;

		values[0] = Int64GetDatum(state->ids[funcctx->call_cntr]);
		values[1] = Float4GetDatum(state->scores[funcctx->call_cntr]);

		tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
		SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
	}
	else
	{
		SRF_RETURN_DONE(funcctx);
	}
}
