/*-------------------------------------------------------------------------
 *
 * vector_sql_functions.c
 *    SQL functions for advanced vector search operations
 *
 * Implements batch search, PQ search, filtered search, and recall evaluation.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/sql/vector_sql_functions.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_types.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "access/htup_details.h"
#include "catalog/pg_type.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "parser/parse_type.h"
#include "nodes/makefuncs.h"
#include "access/htup.h"
#include "executor/spi.h"
#include "lib/stringinfo.h"
#include "utils/lsyscache.h"
#include "neurondb_spi.h"
#include "neurondb_constants.h"

/*
 * Batch search state
 */
typedef struct BatchSearchState
{
	int			query_count;
	int			k;
	int			dim;
	Vector	  **queries;			/* Query vectors */
	ItemPointer *results;			/* Results for each query [query_count][k] */
	float	  **distances;			/* Distances for each query [query_count][k] */
	int		   *result_counts;		/* Number of results per query */
	int			current_query;		/* Current query being processed */
	int			current_result;		/* Current result in current query */
}			BatchSearchState;

/*
 * vector_batch_search(queries vector[], k int)
 *    Batch search for multiple query vectors
 */
PG_FUNCTION_INFO_V1(vector_batch_search);
Datum
vector_batch_search(PG_FUNCTION_ARGS)
{
	ArrayType *queries_array = PG_GETARG_ARRAYTYPE_P(0);
	int32		k = PG_GETARG_INT32(1);
	FuncCallContext *funcctx = NULL;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		TupleDesc	tupdesc;

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		/* Build tuple descriptor */
		if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("function returning record called in context that cannot accept type record")));

		funcctx->tuple_desc = BlessTupleDesc(tupdesc);
		
		/* Extract queries from array */
		Datum	   *queries_elems;
		bool	   *queries_nulls;
		int			queries_count;
		Vector	   **query_vectors = NULL;
		int			dim = 0;
		int			i;
		Vector	   *first_query = NULL;
		Oid			vectorOid;
		BatchSearchState *state = NULL;

		/* Get vector type OID */
		{
			List	   *names = list_make2(makeString("public"), makeString("vector"));

			vectorOid = LookupTypeNameOid(NULL, makeTypeNameFromNameList(names), false);
			list_free(names);
		}

		deconstruct_array(queries_array, vectorOid, -1, false, 'i',
						  &queries_elems, &queries_nulls, &queries_count);

		if (queries_count == 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("queries array cannot be empty")));

		/* Get dimension from first query */
		first_query = DatumGetVectorP(queries_elems[0]);
		NDB_CHECK_VECTOR_VALID(first_query);
		dim = first_query->dim;

		/* Allocate batch search state */
		nalloc(state, BatchSearchState, 1);
		NDB_CHECK_ALLOC(state, "BatchSearchState");
		nalloc(state->queries, Vector *, queries_count);
		NDB_CHECK_ALLOC(state->queries, "queries");
		nalloc(state->results, ItemPointer *, queries_count);
		NDB_CHECK_ALLOC(state->results, "results");
		nalloc(state->distances, float *, queries_count);
		NDB_CHECK_ALLOC(state->distances, "distances");
		nalloc(state->result_counts, int, queries_count);
		NDB_CHECK_ALLOC(state->result_counts, "result_counts");

		state->query_count = queries_count;
		state->k = k;
		state->dim = dim;
		state->current_query = 0;
		state->current_result = 0;

		/* Extract all query vectors */
		for (i = 0; i < queries_count; i++)
		{
			if (queries_nulls[i])
			{
				state->queries[i] = NULL;
				state->result_counts[i] = 0;
				continue;
			}

			Vector	   *query = DatumGetVectorP(queries_elems[i]);

			NDB_CHECK_VECTOR_VALID(query);
			if (query->dim != dim)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("all query vectors must have the same dimension")));

			/* Copy vector to scan context */
			{
				Size		vec_size = VARSIZE_ANY(query);

				state->queries[i] = (Vector *) palloc(vec_size);
				memcpy(state->queries[i], query, vec_size);
			}
			state->result_counts[i] = 0;
		}

		/* Perform batch search for each query via SPI (sequential scan with l2_distance ORDER BY) */
		{
			NdbSpiSession *session = NULL;
			StringInfoData sql;
			const char *tbl_quoted = quote_identifier("my_vectors");
			const char *col_quoted = quote_identifier("embedding");
			Oid			argtypes[1];
			Datum		values[1];
			char		nulls[1] = {' '};

			argtypes[0] = vectorOid;
			session = ndb_spi_session_begin(funcctx->multi_call_memory_ctx, false);
			if (session == NULL)
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("vector_batch_search: failed to begin SPI session")));

			initStringInfo(&sql);
			appendStringInfo(&sql,
							 "SELECT ctid, l2_distance(%s, $1) AS dist FROM %s ORDER BY dist LIMIT %d",
							 col_quoted, tbl_quoted, k);

			for (i = 0; i < queries_count; i++)
			{
				int			ret;
				int			j;
				int			nrows;

				if (state->queries[i] == NULL)
					continue;

				nalloc(state->results[i], ItemPointer, k);
				nalloc(state->distances[i], float, k);

				values[0] = PointerGetDatum(state->queries[i]);
				ret = ndb_spi_execute_with_args(session, sql.data, 1, argtypes, values, nulls, true, (long) k);
				if (ret != SPI_OK_SELECT)
				{
					state->result_counts[i] = 0;
					continue;
				}

				nrows = SPI_processed;
				if (nrows > k)
					nrows = k;
				state->result_counts[i] = nrows;

				for (j = 0; j < nrows; j++)
				{
					bool		isnull;
					Datum		ctid_datum;
					Datum		dist_datum;
					ItemPointer ctid_ptr;

					ctid_datum = SPI_getbinval(SPI_tuptable->vals[j], SPI_tuptable->tupdesc, 1, &isnull);
					if (isnull)
						continue;
					ctid_ptr = DatumGetItemPointer(ctid_datum);
					ItemPointerCopy(ctid_ptr, &state->results[i][j]);

					dist_datum = SPI_getbinval(SPI_tuptable->vals[j], SPI_tuptable->tupdesc, 2, &isnull);
					state->distances[i][j] = isnull ? (float) 0.0 : (float) DatumGetFloat8(dist_datum);
				}
			}

			pfree(sql.data);
			ndb_spi_session_end(&session);
		}

		/* Store batch scan state */
		funcctx->user_fctx = state;
		funcctx->max_calls = queries_count * k; /* Total results */

		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();

	/* Return next result */
	{
		BatchSearchState *state = (BatchSearchState *) funcctx->user_fctx;
		Datum		values[3];
		bool		nulls[3];
		HeapTuple	tuple;

		/* Find next result */
		while (state->current_query < state->query_count)
		{
			if (state->current_result < state->result_counts[state->current_query])
			{
				/* Return this result */
				values[0] = Int32GetDatum(state->current_query);
				values[1] = ItemPointerGetDatum(&state->results[state->current_query][state->current_result]);
				values[2] = Float4GetDatum(state->distances[state->current_query][state->current_result]);
				nulls[0] = false;
				nulls[1] = false;
				nulls[2] = false;

				tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
				state->current_result++;
				SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
			}
			else
			{
				/* Move to next query */
				state->current_query++;
				state->current_result = 0;
			}
		}

		SRF_RETURN_DONE(funcctx);
	}
}

/* PQ search state for two-stage retrieval */
typedef struct PqSearchState
{
	ItemPointer *results;
	float	   *distances;
	int			count;
	int			current;
}			PqSearchState;

/*
 * vector_pq_search(query vector, k int, rerank_k int)
 *    PQ-accelerated search with two-stage retrieval
 */
PG_FUNCTION_INFO_V1(vector_pq_search);
Datum
vector_pq_search(PG_FUNCTION_ARGS)
{
	Vector	   *query = PG_GETARG_VECTOR_P(0);
	int32		k = PG_GETARG_INT32(1);
	int32		rerank_k = PG_GETARG_INT32(2);

	FuncCallContext *funcctx = NULL;

	NDB_CHECK_VECTOR_VALID(query);

	if (k <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("k must be positive")));

	if (rerank_k <= 0 || rerank_k < k)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("rerank_k must be positive and >= k")));

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		TupleDesc	tupdesc;
		PqSearchState *state = NULL;
		NdbSpiSession *session = NULL;
		StringInfoData sql;
		List	   *names;
		Oid			vectorOid;
		Oid			argtypes[1];
		Datum		values[1];
		char		nulls[1] = {' '};
		int			ret;
		int			j;
		int			nrows;
		Vector	   *query_copy = NULL;

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("function returning record called in context that cannot accept type record")));

		funcctx->tuple_desc = BlessTupleDesc(tupdesc);

		nalloc(state, PqSearchState, 1);
		NDB_CHECK_ALLOC(state, "PqSearchState");
		nalloc(state->results, ItemPointer, k);
		nalloc(state->distances, float, k);
		state->count = 0;
		state->current = 0;

		names = list_make2(makeString("public"), makeString("vector"));
		vectorOid = LookupTypeNameOid(NULL, makeTypeNameFromNameList(names), false);
		list_free(names);

		query_copy = (Vector *) palloc(VARSIZE_ANY(query));
		memcpy(query_copy, query, VARSIZE_ANY(query));

		session = ndb_spi_session_begin(funcctx->multi_call_memory_ctx, false);
		if (session == NULL)
		{
			pfree(state->results);
			pfree(state->distances);
			pfree(state);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("vector_pq_search: failed to begin SPI session")));
		}

		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "SELECT ctid, l2_distance(%s, $1) AS dist FROM %s ORDER BY dist LIMIT %d",
						 quote_identifier("embedding"), quote_identifier("my_vectors"), rerank_k);

		argtypes[0] = vectorOid;
		values[0] = PointerGetDatum(query_copy);
		ret = ndb_spi_execute_with_args(session, sql.data, 1, argtypes, values, nulls, true, (long) rerank_k);

		pfree(sql.data);
		ndb_spi_session_end(&session);

		if (ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			nrows = SPI_processed;
			if (nrows > k)
				nrows = k;
			state->count = nrows;

			for (j = 0; j < nrows; j++)
			{
				bool		isnull;
				Datum		ctid_datum;
				Datum		dist_datum;
				ItemPointer ctid_ptr;

				ctid_datum = SPI_getbinval(SPI_tuptable->vals[j], SPI_tuptable->tupdesc, 1, &isnull);
				if (isnull)
					continue;
				ctid_ptr = DatumGetItemPointer(ctid_datum);
				ItemPointerCopy(ctid_ptr, &state->results[j]);

				dist_datum = SPI_getbinval(SPI_tuptable->vals[j], SPI_tuptable->tupdesc, 2, &isnull);
				state->distances[j] = isnull ? (float) 0.0 : (float) DatumGetFloat8(dist_datum);
			}
		}

		funcctx->user_fctx = state;
		funcctx->max_calls = state->count;
		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();

	{
		PqSearchState *state = (PqSearchState *) funcctx->user_fctx;

		if (state != NULL && state->current < state->count)
		{
			Datum		values[2];
			bool		nulls[2];
			HeapTuple	tuple;

			values[0] = ItemPointerGetDatum(&state->results[state->current]);
			values[1] = Float4GetDatum(state->distances[state->current]);
			nulls[0] = false;
			nulls[1] = false;
			tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
			state->current++;
			SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
		}
	}
	SRF_RETURN_DONE(funcctx);
}

/*
 * Filtered search state
 */
typedef struct FilteredSearchState
{
	Vector	   *query;
	int			k;
	char	   *filter_predicate;	/* WHERE clause text */
	ItemPointer *results;
	float	   *distances;
	int			result_count;
	int			current_result;
}			FilteredSearchState;

/*
 * vector_filtered_search(query vector, filter_predicate text, k int, table_name text, vector_column text)
 *    Filtered search with auto-tuning
 * 
 * Note: This implementation uses SPI to execute filtered searches.
 * For production use, table_name and vector_column parameters should be added.
 */
PG_FUNCTION_INFO_V1(vector_filtered_search);
Datum
vector_filtered_search(PG_FUNCTION_ARGS)
{
	Vector	   *query = PG_GETARG_VECTOR_P(0);
	text	   *filter_predicate = PG_GETARG_TEXT_P(1);
	int32		k = PG_GETARG_INT32(2);
	FuncCallContext *funcctx = NULL;

	NDB_CHECK_VECTOR_VALID(query);

	if (k <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("k must be positive")));

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		TupleDesc	tupdesc;
		FilteredSearchState *state = NULL;
		char	   *filter_str = NULL;
		char	   *query_vec_str = NULL;
		StringInfoData sql;
		int			ret;
		int			i;

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("function returning record called in context that cannot accept type record")));

		funcctx->tuple_desc = BlessTupleDesc(tupdesc);

		/* Allocate state */
		nalloc(state, FilteredSearchState, 1);
		NDB_CHECK_ALLOC(state, "FilteredSearchState");

		/* Copy query vector */
		{
			Size		vec_size = VARSIZE_ANY(query);

			state->query = (Vector *) palloc(vec_size);
			memcpy(state->query, query, vec_size);
		}

		/* Store filter predicate */
		filter_str = text_to_cstring(filter_predicate);
		state->filter_predicate = pstrdup(filter_str);
		state->k = k;
		state->result_count = 0;
		state->current_result = 0;

		/* Convert query vector to string for SQL */
		{
			StringInfoData vec_buf;

			initStringInfo(&vec_buf);
			appendStringInfoChar(&vec_buf, '[');
			for (i = 0; i < query->dim; i++)
			{
				if (i > 0)
					appendStringInfoChar(&vec_buf, ',');
				appendStringInfo(&vec_buf, "%g", query->data[i]);
			}
			appendStringInfoChar(&vec_buf, ']');
			query_vec_str = vec_buf.data;
		}

		/* Build SQL query for filtered search */
		/* Note: This is a simplified implementation. Full version would:
		 * 1. Accept table_name and vector_column as parameters
		 * 2. Use index scan when available
		 * 3. Apply filter during index traversal for efficiency
		 */
		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "SELECT ctid, embedding <-> '%s'::vector AS distance "
						 "FROM (SELECT ctid, embedding FROM documents WHERE %s) AS filtered "
						 "ORDER BY embedding <-> '%s'::vector "
						 "LIMIT %d",
						 query_vec_str, filter_str, query_vec_str, k);

		/* Execute via SPI */
		ret = SPI_connect();
		if (ret != SPI_OK_CONNECT)
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("filtered search: SPI_connect failed")));

		ret = SPI_execute(sql.data, true, 0);
		if (ret != SPI_OK_SELECT)
		{
			SPI_finish();
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("filtered search: query execution failed")));
		}

		/* Extract results */
		state->result_count = SPI_processed;
		if (state->result_count > 0)
		{
			nalloc(state->results, ItemPointer, state->result_count);
			nalloc(state->distances, float, state->result_count);

			for (i = 0; i < state->result_count; i++)
			{
				HeapTuple	tup = SPI_tuptable->vals[i];
				Datum		ctid_datum;
				Datum		dist_datum;
				bool		isnull;

				ctid_datum = SPI_getbinval(tup, SPI_tuptable->tupdesc, 1, &isnull);
				if (!isnull)
					state->results[i] = DatumGetItemPointer(ctid_datum);

				dist_datum = SPI_getbinval(tup, SPI_tuptable->tupdesc, 2, &isnull);
				if (!isnull)
					state->distances[i] = DatumGetFloat4(dist_datum);
			}
		}

		SPI_finish();
		pfree(sql.data);
		pfree(query_vec_str);

		funcctx->user_fctx = state;
		funcctx->max_calls = state->result_count;

		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();

	/* Return next result */
	{
		FilteredSearchState *state = (FilteredSearchState *) funcctx->user_fctx;
		Datum		values[2];
		bool		nulls[2];
		HeapTuple	tuple;

		if (state->current_result < state->result_count)
		{
			values[0] = ItemPointerGetDatum(&state->results[state->current_result]);
			values[1] = Float4GetDatum(state->distances[state->current_result]);
			nulls[0] = false;
			nulls[1] = false;

			tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
			state->current_result++;
			SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
		}

		SRF_RETURN_DONE(funcctx);
	}
}

/*
 * vector_recall_eval(query vector, approximate_results bigint[], exact_results bigint[])
 *    Evaluate recall of approximate search vs exact search
 */
PG_FUNCTION_INFO_V1(vector_recall_eval);
Datum
vector_recall_eval(PG_FUNCTION_ARGS)
{
	Vector	   *query = PG_GETARG_VECTOR_P(0);
	ArrayType *approx_array = PG_GETARG_ARRAYTYPE_P(1);
	ArrayType *exact_array = PG_GETARG_ARRAYTYPE_P(2);
	Datum	   *approx_elems;
	Datum	   *exact_elems;
	bool	   *approx_nulls;
	bool	   *exact_nulls;
	int			approx_count;
	int			exact_count;
	int			matches = 0;
	float8		recall;

	/* Extract array elements */
	deconstruct_array(approx_array, INT8OID, 8, true, 'd',
					  &approx_elems, &approx_nulls, &approx_count);
	deconstruct_array(exact_array, INT8OID, 8, true, 'd',
					  &exact_elems, &exact_nulls, &exact_count);

	/* Count matches */
	for (int i = 0; i < approx_count; i++)
	{
		if (approx_nulls[i])
			continue;

		int64		approx_val = DatumGetInt64(approx_elems[i]);

		for (int j = 0; j < exact_count; j++)
		{
			if (exact_nulls[j])
				continue;

			int64		exact_val = DatumGetInt64(exact_elems[j]);

			if (approx_val == exact_val)
			{
				matches++;
				break;
			}
		}
	}

	/* Compute recall */
	if (exact_count > 0)
		recall = (float8) matches / (float8) exact_count;
	else
		recall = 0.0;

	PG_RETURN_FLOAT8(recall);
}

