/*-------------------------------------------------------------------------
 *
 * hybrid_search.c
 *    Hybrid search combining vector similarity with FTS and metadata.
 *
 * This file implements hybrid search capabilities that combine vector
 * similarity with full-text search (FTS), metadata filtering, keyword
 * matching, temporal awareness, and faceted search.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
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

/*
 * Helper: Build SQL safe literal for text input
 */
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

/*
 * Helper: Extract float array from Vector datum
 */
static void
__attribute__((unused))
vector_to_float_array(const Vector *vec, float *arr, int dim)
{
	int			i;

	for (i = 0; i < dim; i++)
		arr[i] = vec->data[i];
}

/*
 * Hybrid search: Vector + FTS + Metadata filters
 * Returns SRF: TABLE(id bigint, score real)
 */
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
	FuncCallContext *funcctx;
	HybridSearchState *state = NULL;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		TupleDesc	tupdesc;
		text	   *table_name = PG_GETARG_TEXT_PP(0);
		Vector	   *query_vec;
		text	   *query_text = PG_GETARG_TEXT_PP(2);
		text	   *filters = PG_GETARG_TEXT_PP(3);
		float8		vector_weight = PG_GETARG_FLOAT8(4);
		int32		limit = PG_GETARG_INT32(5);
		char	   *tbl_str;
		char	   *txt_str;
		char	   *filter_str;
		StringInfoData sql;
		StringInfoData vec_lit;
		int			spi_ret;
		int			i;
		int			proc;

		NDB_DECLARE(NdbSpiSession *, session);

		/* Get vector argument - handle NULL case */
		if (PG_ARGISNULL(1))
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("hybrid_search: query vector cannot be NULL")));

		/* PG_GETARG_VECTOR_P already handles detoasting */
		query_vec = PG_GETARG_VECTOR_P(1);
		NDB_CHECK_VECTOR_VALID(query_vec);

		/* Validate vector structure */
		if (query_vec == NULL)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("hybrid_search: query vector is NULL")));

		/* Check vector size */
		if (VARSIZE(query_vec) < VARHDRSZ + sizeof(int16) * 2)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("hybrid_search: invalid vector size")));

		if (query_vec->dim <= 0 || query_vec->dim > 32767)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("hybrid_search: invalid vector dimension %d", query_vec->dim)));

		/* Validate vector has enough data */
		if (VARSIZE(query_vec) < VARHDRSZ + sizeof(int16) * 2 + sizeof(float4) * query_vec->dim)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("hybrid_search: vector size mismatch (dim=%d, size=%d)",
							query_vec->dim, VARSIZE(query_vec))));

		tbl_str = text_to_cstring(table_name);
		txt_str = text_to_cstring(query_text);
		filter_str = text_to_cstring(filters);

		elog(DEBUG1,
			 "neurondb: Hybrid search on '%s' (query='%s', filters='%s', vec_dim=%d, weight=%.2f, limit=%d)",
			 tbl_str,
			 txt_str,
			 filter_str,
			 query_vec->dim,
			 vector_weight,
			 limit);

		session = ndb_spi_session_begin(CurrentMemoryContext, false);
		if (session == NULL)
			ereport(ERROR, (errmsg("hybrid_search: failed to begin SPI session")));

		/*
		 * Build vector literal representation for SQL (float array as
		 * '{...}')
		 */
		/* Vector data is a flexible array member, access directly */
		initStringInfo(&vec_lit);
		appendStringInfoChar(&vec_lit, '{');
		for (i = 0; i < query_vec->dim; i++)
		{
			float4		val;

			if (i)
				appendStringInfoChar(&vec_lit, ',');

			/* Access vector data safely - flexible array member */
			val = query_vec->data[i];

			/* Validate each value before formatting */
			if (!isfinite(val))
			{
				NDB_FREE(vec_lit.data);
				ndb_spi_session_end(&session);
				NDB_FREE(tbl_str);
				NDB_FREE(txt_str);
				NDB_FREE(filter_str);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("hybrid_search: non-finite value in vector at index %d", i)));
			}
			appendStringInfo(&vec_lit, "%g", val);
		}
		appendStringInfoChar(&vec_lit, '}');

		initStringInfo(&sql);

		/* Compose SQL and inject vector literal directly */
		appendStringInfo(&sql,
						 "WITH _hybrid_scores AS ("
						 " SELECT id,"
						 "        (1 - (embedding <-> '%s'::vector)) AS vector_score,"
						 "        ts_rank(fts_vector, plainto_tsquery(%s)) AS fts_score,"
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
						 to_sql_literal(txt_str),
						 tbl_str,
						 to_sql_literal(filter_str),
						 vector_weight,
						 vector_weight,
						 limit);

		spi_ret = ndb_spi_execute(session, sql.data, true, limit);
		if (spi_ret != SPI_OK_SELECT)
		{
			NDB_FREE(sql.data);
			NDB_FREE(vec_lit.data);
			ndb_spi_session_end(&session);
			NDB_FREE(tbl_str);
			NDB_FREE(txt_str);
			NDB_FREE(filter_str);
			ereport(ERROR, (errmsg("Failed to execute hybrid search SQL")));
		}

		proc = SPI_processed;

		/* Initialize SRF context */
		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		/* Allocate state */
		NDB_ALLOC(state, HybridSearchState, 1);
		NDB_CHECK_ALLOC(state, "state");

		/* Build tuple descriptor */
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
			/* Extract results from SPI */
			NDB_DECLARE(int64 *, ids);
			NDB_DECLARE(float4 *, scores);
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

				/* Get id (column 1) */
				id_val = SPI_getbinval(SPI_tuptable->vals[i],
									   SPI_tuptable->tupdesc,
									   1,
									   &isnull_id);

				/* Get score (column 2) */
				score_val = SPI_getbinval(SPI_tuptable->vals[i],
										  SPI_tuptable->tupdesc,
										  2,
										  &isnull_score);

				if (!isnull_id)
				{
					/* Convert id to int64 */
					Oid			id_type = SPI_gettypeid(SPI_tuptable->tupdesc, 1);

					if (id_type == INT8OID)
						state->ids[i] = DatumGetInt64(id_val);
					else if (id_type == INT4OID)
						state->ids[i] = (int64) DatumGetInt32(id_val);
					else
						state->ids[i] = 0;	/* fallback */
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

		NDB_FREE(sql.data);
		NDB_FREE(vec_lit.data);
		NDB_FREE(tbl_str);
		NDB_FREE(txt_str);
		NDB_FREE(filter_str);
		ndb_spi_session_end(&session);
		MemoryContextSwitchTo(oldcontext);
	}

	/* Subsequent calls */
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

/*
 * Comparison function for RRF items (for qsort)
 */
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

	/* Sort by decreasing score */
	if (item_b->score > item_a->score)
		return 1;
	if (item_b->score < item_a->score)
		return -1;
	return 0;
}

/*
 * Reciprocal Rank Fusion (RRF)
 */
PG_FUNCTION_INFO_V1(reciprocal_rank_fusion);
Datum
reciprocal_rank_fusion(PG_FUNCTION_ARGS)
{
	ArrayType  *rankings = PG_GETARG_ARRAYTYPE_P(0);
	float8		k = PG_GETARG_FLOAT8(1);
	int			n_rankers;
	int			i;
	int			j;
	int16		elmlen;
	bool		elmbyval;
	char		elmalign;
	ArrayType **rank_arrays;
	int			item_count = 0;
	HTAB	   *item_hash;
	HASHCTL		info;
	Datum	   *result_datums = NULL;
	bool	   *result_nulls = NULL;
	ArrayType  *ret_array;

	elog(DEBUG1,
		 "neurondb: Computing Reciprocal Rank Fusion with k=%.2f",
		 k);

	/*
	 * This implementation expects an array of text[] where each is an id
	 * order
	 */
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

	/* Build a hash table of document ids to cumulative RRF scores */
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
		Datum	   *ids;
		bool	   *nulls;

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
		NDB_FREE(ids);
		NDB_FREE(nulls);
	}

	/* Output: sort the ids by score descending, return as text[] */
	item_count = hash_get_num_entries(item_hash);
	result_datums = palloc0(sizeof(Datum) * item_count);
	NDB_CHECK_ALLOC(result_datums, "allocation");
	result_nulls = palloc0(sizeof(bool) * item_count);
	NDB_CHECK_ALLOC(result_nulls, "allocation");

	{
		struct
		{
			char		key[512];
			float8		score;
		}		   *cur;
		struct
		{
			char	   *key;
			float8		score;
		}		   *items;
		HASH_SEQ_STATUS stat;
		int			idx = 0;
		int			idx_i,
					idx_j;

		/* Collect items with scores */
		items = palloc(sizeof(*items) * item_count);
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

		/* Extract sorted keys */
		for (idx = 0; idx < item_count; idx++)
		{
			result_datums[idx] =
				PointerGetDatum(cstring_to_text(items[idx].key));
			result_nulls[idx] = false;
			NDB_FREE(items[idx].key);
		}

		NDB_FREE(items);
	}

	ret_array = construct_array(
								result_datums, item_count, TEXTOID, -1, false, 'i');

	hash_destroy(item_hash);
	NDB_FREE(result_datums);
	NDB_FREE(result_nulls);

	PG_RETURN_ARRAYTYPE_P(ret_array);
}

/*
 * Semantic + Keyword search with BM25
 */
PG_FUNCTION_INFO_V1(semantic_keyword_search);
Datum
semantic_keyword_search(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	Vector	   *semantic_query;
	text	   *keyword_query;
	int32		top_k;
	char	   *tbl_str;
	char	   *kw_str;
	StringInfoData sql;
	StringInfoData vec_lit;
	int			spi_ret;
	ArrayType  *ret_array;
	int			proc;
	int			i;

	NDB_DECLARE(Datum *, datums);
	NDB_DECLARE(bool *, nulls);
	NDB_DECLARE(NdbSpiSession *, session);

	table_name = PG_GETARG_TEXT_PP(0);
	semantic_query = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(semantic_query);
	keyword_query = PG_GETARG_TEXT_PP(2);
	top_k = PG_GETARG_INT32(3);

	tbl_str = text_to_cstring(table_name);
	kw_str = text_to_cstring(keyword_query);

	elog(DEBUG1,
		 "neurondb: Semantic + Keyword search on '%s' for '%s' "
		 "(vec_dim=%d), top_k=%d",
		 tbl_str,
		 kw_str,
		 semantic_query->dim,
		 top_k);
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
		NDB_FREE(sql.data);
		NDB_FREE(vec_lit.data);
		NDB_FREE(tbl_str);
		NDB_FREE(kw_str);
		ndb_spi_session_end(&session);
		ereport(ERROR,
				(errmsg("Failed to execute semantic_keyword_search "
						"SQL")));
	}

	proc = SPI_processed;
	if (proc == 0)
	{
		NDB_FREE(sql.data);
		NDB_FREE(vec_lit.data);
		NDB_FREE(tbl_str);
		NDB_FREE(kw_str);
		ndb_spi_session_end(&session);
		ret_array = construct_empty_array(TEXTOID);
		PG_RETURN_ARRAYTYPE_P(ret_array);
	}

	NDB_ALLOC(datums, Datum, proc);
	NDB_CHECK_ALLOC(datums, "allocation");
	NDB_ALLOC(nulls, bool, proc);
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

	NDB_FREE(datums);
	NDB_FREE(nulls);
	NDB_FREE(sql.data);
	NDB_FREE(vec_lit.data);
	NDB_FREE(tbl_str);
	NDB_FREE(kw_str);
	ndb_spi_session_end(&session);

	PG_RETURN_ARRAYTYPE_P(ret_array);
}

/*
 * Multi-vector search (ColBERT-style late interaction)
 */
PG_FUNCTION_INFO_V1(multi_vector_search);
Datum
multi_vector_search(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_PP(0);
	ArrayType  *query_vectors = PG_GETARG_ARRAYTYPE_P(1);
	text	   *agg_method = PG_GETARG_TEXT_PP(2);
	int32		top_k = PG_GETARG_INT32(3);
	char	   *tbl_str = text_to_cstring(table_name);
	char	   *agg_str = text_to_cstring(agg_method);
	int			nvecs;
	StringInfoData sql;
	StringInfoData subquery;
	Datum	   *vec_datums;
	bool	   *vec_nulls;
	Oid			vec_elemtype = ARR_ELEMTYPE(query_vectors);
	int			i;
	int			spi_ret;
	int			proc;
	ArrayType  *ret_array;
	Datum	   *datums;
	bool	   *nulls;

	NDB_DECLARE(NdbSpiSession *, session);

	nvecs = ArrayGetNItems(
						   ARR_NDIM(query_vectors), ARR_DIMS(query_vectors));
	elog(DEBUG1,
		 "neurondb: Multi-vector search on '%s' with %d queries, "
		 "agg=%s, top_k=%d",
		 tbl_str,
		 nvecs,
		 agg_str,
		 top_k);
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
		NDB_FREE(tbl_str);
		NDB_FREE(agg_str);
		ndb_spi_session_end(&session);
		ereport(ERROR,
				(errmsg("multi_vector_search: at least one query "
						"vector required")));
	}

	{
		/* Check first vector exists and validate */
		Vector	   *first_vec = (Vector *) DatumGetPointer(vec_datums[0]);

		if (first_vec->dim <= 0)
		{
			NDB_FREE(tbl_str);
			NDB_FREE(agg_str);
			NDB_FREE(vec_datums);
			NDB_FREE(vec_nulls);
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
			NDB_FREE(lit.data);
		}
	}
	NDB_FREE(vec_datums);
	NDB_FREE(vec_nulls);

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
		NDB_FREE(sql.data);
		NDB_FREE(subquery.data);
		NDB_FREE(tbl_str);
		NDB_FREE(agg_str);
		ndb_spi_session_end(&session);
		ereport(ERROR,
				(errmsg("Failed to execute multi_vector_search SQL")));
	}
	proc = SPI_processed;
	if (proc == 0)
	{
		NDB_FREE(sql.data);
		NDB_FREE(subquery.data);
		NDB_FREE(tbl_str);
		NDB_FREE(agg_str);
		ndb_spi_session_end(&session);
		ret_array = construct_empty_array(TEXTOID);
		PG_RETURN_ARRAYTYPE_P(ret_array);
	}
	datums = NULL;
	nulls = NULL;
	NDB_ALLOC(datums, Datum, proc);
	NDB_CHECK_ALLOC(datums, "allocation");
	NDB_ALLOC(nulls, bool, proc);
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
	NDB_FREE(datums);
	NDB_FREE(nulls);
	NDB_FREE(sql.data);
	NDB_FREE(subquery.data);
	NDB_FREE(tbl_str);
	NDB_FREE(agg_str);
	ndb_spi_session_end(&session);

	PG_RETURN_ARRAYTYPE_P(ret_array);
}

/*
 * Faceted search with vector similarity
 */
PG_FUNCTION_INFO_V1(faceted_vector_search);
Datum
faceted_vector_search(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	Vector	   *query_vec;
	text	   *facet_column;
	int32		per_facet_limit;
	char	   *tbl_str;
	char	   *facet_str;
	StringInfoData sql;
	StringInfoData vec_lit;
	int			spi_ret;
	int			proc;
	ArrayType  *ret_array;
	Datum	   *datums;
	bool	   *nulls;
	int			i;

	NDB_DECLARE(NdbSpiSession *, session);

	table_name = PG_GETARG_TEXT_PP(0);
	query_vec = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(query_vec);
	facet_column = PG_GETARG_TEXT_PP(2);
	per_facet_limit = PG_GETARG_INT32(3);
	tbl_str = text_to_cstring(table_name);
	facet_str = text_to_cstring(facet_column);

	elog(DEBUG1,
		 "neurondb: Faceted search on '%s' with facets='%s', limit=%d (vec_dim=%d)",
		 tbl_str,
		 facet_str,
		 per_facet_limit,
		 query_vec->dim);
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
		NDB_FREE(sql.data);
		NDB_FREE(vec_lit.data);
		NDB_FREE(tbl_str);
		NDB_FREE(facet_str);
		ndb_spi_session_end(&session);
		ereport(ERROR,
				(errmsg("Failed to execute faceted_vector_search "
						"SQL")));
	}
	proc = SPI_processed;
	if (proc == 0)
	{
		NDB_FREE(sql.data);
		NDB_FREE(vec_lit.data);
		NDB_FREE(tbl_str);
		NDB_FREE(facet_str);
		ndb_spi_session_end(&session);
		ret_array = construct_empty_array(TEXTOID);
		PG_RETURN_ARRAYTYPE_P(ret_array);
	}
	datums = NULL;
	nulls = NULL;
	NDB_ALLOC(datums, Datum, proc);
	NDB_CHECK_ALLOC(datums, "allocation");
	NDB_ALLOC(nulls, bool, proc);
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
	NDB_FREE(datums);
	NDB_FREE(nulls);
	NDB_FREE(sql.data);
	NDB_FREE(vec_lit.data);
	NDB_FREE(tbl_str);
	NDB_FREE(facet_str);
	ndb_spi_session_end(&session);

	PG_RETURN_ARRAYTYPE_P(ret_array);
}

/*
 * Temporal-aware vector search - boost recent documents with exponential decay
 */
PG_FUNCTION_INFO_V1(temporal_vector_search);
Datum
temporal_vector_search(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	Vector	   *query_vec;
	text	   *timestamp_col;
	float8		decay_rate;		/* e.g. 0.1 means decays by 1/exp(0.1) per day */
	int32		top_k;
	char	   *tbl_str;
	char	   *ts_str;
	StringInfoData sql;
	StringInfoData vec_lit;
	int			spi_ret;
	int			proc;
	ArrayType  *ret_array;
	Datum	   *datums;
	bool	   *nulls;
	int			i;

	NDB_DECLARE(NdbSpiSession *, session);

	table_name = PG_GETARG_TEXT_PP(0);
	query_vec = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(query_vec);
	timestamp_col = PG_GETARG_TEXT_PP(2);
	decay_rate = PG_GETARG_FLOAT8(3);
	top_k = PG_GETARG_INT32(4);
	tbl_str = text_to_cstring(table_name);
	ts_str = text_to_cstring(timestamp_col);

	elog(DEBUG1,
		 "neurondb: Temporal search on '%s'.%s with decay=%.4f, top_k=%d (vec_dim=%d)",
		 tbl_str,
		 ts_str,
		 decay_rate,
		 top_k,
		 query_vec->dim);
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
		NDB_FREE(sql.data);
		NDB_FREE(vec_lit.data);
		NDB_FREE(tbl_str);
		NDB_FREE(ts_str);
		ndb_spi_session_end(&session);
		ereport(ERROR,
				(errmsg("Failed to execute temporal_vector_search "
						"SQL")));
	}

	proc = SPI_processed;
	if (proc == 0)
	{
		NDB_FREE(sql.data);
		NDB_FREE(vec_lit.data);
		NDB_FREE(tbl_str);
		NDB_FREE(ts_str);
		ndb_spi_session_end(&session);
		ret_array = construct_empty_array(TEXTOID);
		PG_RETURN_ARRAYTYPE_P(ret_array);
	}
	datums = NULL;
	nulls = NULL;
	NDB_ALLOC(datums, Datum, proc);
	NDB_CHECK_ALLOC(datums, "allocation");
	NDB_ALLOC(nulls, bool, proc);
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
	NDB_FREE(datums);
	NDB_FREE(nulls);
	NDB_FREE(sql.data);
	NDB_FREE(vec_lit.data);
	NDB_FREE(tbl_str);
	NDB_FREE(ts_str);
	ndb_spi_session_end(&session);

	PG_RETURN_ARRAYTYPE_P(ret_array);
}

/*
 * Diversity-aware search (Maximal Marginal Relevance - MMR)
 * Returns top_k doc ids, maximizing both relevance and novelty.
 */
PG_FUNCTION_INFO_V1(diverse_vector_search);
Datum
diverse_vector_search(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	Vector	   *query_vec;
	float8		lambda;
	int32		top_k;
	char	   *tbl_str;
	StringInfoData sql;
	StringInfoData vec_lit;
	int			spi_ret;
	int			proc;
	int			i,
				j;
	ArrayType  *ret_array;
	Datum	   *datums = NULL;
	bool	   *nulls = NULL;
	int			n_candidates;
	int			select_count = 0;
	mmr_cand_t *cands;

	NDB_DECLARE(NdbSpiSession *, session);

	table_name = PG_GETARG_TEXT_PP(0);
	query_vec = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(query_vec);
	lambda = PG_GETARG_FLOAT8(2);
	top_k = PG_GETARG_INT32(3);
	tbl_str = text_to_cstring(table_name);

	elog(DEBUG1,
		 "neurondb: Diverse search on '%s' with lambda=%.2f, top_k=%d (vec_dim=%d)",
		 tbl_str,
		 lambda,
		 top_k,
		 query_vec->dim);
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
		NDB_FREE(sql.data);
		NDB_FREE(vec_lit.data);
		NDB_FREE(tbl_str);
		ndb_spi_session_end(&session);
		ereport(ERROR,
				(errmsg("Failed to execute diverse_vector_search "
						"SQL")));
	}

	proc = SPI_processed;
	if (proc == 0)
	{
		NDB_FREE(sql.data);
		NDB_FREE(vec_lit.data);
		NDB_FREE(tbl_str);
		ndb_spi_session_end(&session);
		ret_array = construct_empty_array(TEXTOID);
		PG_RETURN_ARRAYTYPE_P(ret_array);
	}

	n_candidates = proc;

	/* MMR greedy selection */

	cands = NULL;
	NDB_ALLOC(cands, mmr_cand_t, n_candidates);
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

	NDB_ALLOC(datums, Datum, top_k);
	NDB_CHECK_ALLOC(datums, "allocation");
	NDB_ALLOC(nulls, bool, top_k);
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

			/* Calculate max similarity to selected set */
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
		NDB_FREE(cands[i].id);

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
				NDB_FREE(cands[i].vec);
		}
	}
	NDB_FREE(datums);
	NDB_FREE(nulls);
	NDB_FREE(cands);
	NDB_FREE(sql.data);
	NDB_FREE(vec_lit.data);
	NDB_FREE(tbl_str);
	ndb_spi_session_end(&session);
	PG_RETURN_ARRAYTYPE_P(ret_array);
}
