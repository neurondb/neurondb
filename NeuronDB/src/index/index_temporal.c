/*-------------------------------------------------------------------------
 *
 * index_temporal.c
 *	  Temporal vector index with decay on insert time
 *
 * Implements TVX index that applies time-based decay to similarity
 * scores, enabling time-gated kNN queries where recent vectors
 * are automatically boosted.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/index/index_temporal.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_index.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/timestamp.h"
#include "executor/spi.h"
#include "utils/lsyscache.h"
#include "catalog/pg_type.h"
#include "funcapi.h"
#include "utils/memutils.h"
#include "utils/elog.h"
#include "lib/stringinfo.h"
#include <math.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"

/* Forward declarations */
static char *VectorToLiteral(Vector *v);

/*
 * get_temporal_index_table
 *	  Returns qualified TVX index table name for a table/col.
 */
static char *
get_temporal_index_table(const char *table, const char *col)
{
	char	   *buf;

	buf = palloc(strlen(table) + strlen(col) + 32);
	snprintf(buf,
			 strlen(table) + strlen(col) + 32,
			 "__tvx_index_%s_%s",
			 table,
			 col);
	return buf;
}

/*
 * temporal_index_create
 *	  Create temporal vector index table.
 *
 * Sets up a separate index table holding (id, vector, insert_ts) for the
 * specified table/column. Metadata could be extended to store decay rate
 * per index.
 */
PG_FUNCTION_INFO_V1(temporal_index_create);

Datum
temporal_index_create(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_PP(0);
	text	   *vector_col = PG_GETARG_TEXT_PP(1);
	text	   *timestamp_col = PG_GETARG_TEXT_PP(2);
	float8		decay_rate = PG_GETARG_FLOAT8(3);
	char	   *tbl_str;
	char	   *vec_str;
	char	   *ts_str;
	char	   *idx_tbl;
	StringInfoData sql;
	int			ret;
	NDB_DECLARE(NdbSpiSession *, session);

	tbl_str = text_to_cstring(table_name);
	vec_str = text_to_cstring(vector_col);
	ts_str = text_to_cstring(timestamp_col);
	idx_tbl = get_temporal_index_table(tbl_str, vec_str);

	elog(INFO,
		 "neurondb: Creating temporal index on %s.%s with timestamp %s "
		 "(decay=%.4f/day)",
		 tbl_str,
		 vec_str,
		 ts_str,
		 decay_rate);

	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		elog(ERROR, "failed to begin SPI session");

	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "CREATE TABLE IF NOT EXISTS %s ("
					 "id bigint PRIMARY KEY, "
					 "v vector, "
					 "insert_ts timestamptz"
					 ")",
					 idx_tbl);
	ret = ndb_spi_execute(session, sql.data, false, 0);
	if (ret != SPI_OK_UTILITY)
	{
		NDB_FREE(sql.data);
		ndb_spi_session_end(&session);
		elog(ERROR, "Failed to create TVX index table: %s", sql.data);
	}

	/* Use safe free/reinit to handle potential memory context changes */
	NDB_FREE(sql.data);
	initStringInfo(&sql);

	appendStringInfo(&sql, "TRUNCATE %s", idx_tbl);
	ret = ndb_spi_execute(session, sql.data, false, 0);
	if (ret != SPI_OK_UTILITY)
	{
		NDB_FREE(sql.data);
		ndb_spi_session_end(&session);
		elog(ERROR, "Failed to truncate TVX index table: %s", idx_tbl);
	}

	/* Use safe free/reinit to handle potential memory context changes */
	NDB_FREE(sql.data);
	initStringInfo(&sql);

	appendStringInfo(&sql,
					 "INSERT INTO %s (id, v, insert_ts) "
					 "SELECT id, %s, %s FROM %s",
					 idx_tbl,
					 vec_str,
					 ts_str,
					 tbl_str);

	ret = ndb_spi_execute(session, sql.data, false, 0);
	if (ret != SPI_OK_INSERT)
	{
		NDB_FREE(sql.data);
		ndb_spi_session_end(&session);
		elog(ERROR,
			 "Failed to bulk insert vectors into TVX index: %s",
			 sql.data);
	}

	NDB_FREE(sql.data);
	ndb_spi_session_end(&session);
	NDB_FREE(idx_tbl);

	PG_RETURN_BOOL(true);
}

/*
 * temporal_knn_search
 *	  Time-gated temporal kNN search.
 *
 * Args:
 *   query: vector (similarity, higher is better)
 *   k: number of results
 *   cutoff_time: cutoff on insert_ts (timestamptz); only entries inserted at or before this are considered
 *   table_name, vector_col (optional): CSTRING input for full implementation.
 *   decay_rate (optional): argument or from metadata
 *
 * RETURNS: SETOF (id bigint, score float4, insert_ts timestamptz)
 */
PG_FUNCTION_INFO_V1(temporal_knn_search);

Datum
temporal_knn_search(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	TupleDesc	tupdesc;
	MemoryContext oldcontext;
	NDB_DECLARE(NdbSpiSession *, session2);

	if (SRF_IS_FIRSTCALL())
	{
		Vector	   *query;
		int32		k;
		TimestampTz cutoff_time;
		text	   *table_name;
		text	   *vector_col;
		text	   *ts_col;
		float8		decay_rate;
		char	   *tbl_str;
		char	   *vec_str;
		char	   *idx_tbl;
		StringInfoData sql;
		int			ret;

		query = PG_GETARG_VECTOR_P(0);
		NDB_CHECK_VECTOR_VALID(query);
		k = PG_GETARG_INT32(1);
		cutoff_time = PG_GETARG_TIMESTAMPTZ(2);
		table_name = PG_GETARG_TEXT_PP(3);
		vector_col = PG_GETARG_TEXT_PP(4);
		ts_col = PG_GETARG_TEXT_PP(5);
		decay_rate = PG_GETARG_FLOAT8(6);

		(void) ts_col;			/* possibly unused */

		tbl_str = text_to_cstring(table_name);
		vec_str = text_to_cstring(vector_col);
		idx_tbl = get_temporal_index_table(tbl_str, vec_str);

		funcctx = SRF_FIRSTCALL_INIT();

		tupdesc = CreateTemplateTupleDesc(3);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "id", INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "score", FLOAT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "insert_ts", TIMESTAMPTZOID, -1, 0);
		funcctx->tuple_desc = BlessTupleDesc(tupdesc);

		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		session2 = ndb_spi_session_begin(funcctx->multi_call_memory_ctx, false);
		if (session2 == NULL)
			elog(ERROR, "failed to begin SPI session");

		initStringInfo(&sql);

		appendStringInfo(&sql,
						 "SELECT id, "
						 "temporal_score(-vector_l2(v, %s), insert_ts, now(), %f) AS score, "
						 "insert_ts "
						 "FROM %s "
						 "WHERE insert_ts <= to_timestamp(%lld) "
						 "ORDER BY score DESC "
						 "LIMIT %d",
						 VectorToLiteral(query),
						 decay_rate,
						 idx_tbl,
						 (long long) cutoff_time,
						 k);

		ret = ndb_spi_execute(session2, sql.data, true, 0);
		if (ret != SPI_OK_SELECT)
		{
			NDB_FREE(sql.data);
			ndb_spi_session_end(&session2);
			elog(ERROR, "Failed temporal index scan: %s", sql.data);
		}

		/* Store session to keep SPI connection alive for tuptable access */
		funcctx->user_fctx = session2;
		NDB_FREE(sql.data);

		MemoryContextSwitchTo(oldcontext);

		NDB_FREE(idx_tbl);
	}

	funcctx = SRF_PERCALL_SETUP();

	{
		uint64		call_cntr;
		uint64		max_calls;
		NdbSpiSession *session2;
		SPITupleTable *tuptable;

		call_cntr = funcctx->call_cntr;
		max_calls = funcctx->max_calls;
		session2 = (NdbSpiSession *) funcctx->user_fctx;
		tuptable = SPI_tuptable;  /* Access via global SPI_tuptable while session is active */

		if (call_cntr < max_calls)
		{
			HeapTuple	spi_tuple;
			Datum		values[3];
			bool		nulls[3] = {false, false, false};
			bool		isnull;
			HeapTuple	result_tuple;

			spi_tuple = tuptable->vals[call_cntr];
			values[0] = SPI_getbinval(spi_tuple, tuptable->tupdesc, 1, &isnull);
			nulls[0] = isnull;
			values[1] = SPI_getbinval(spi_tuple, tuptable->tupdesc, 2, &isnull);
			nulls[1] = isnull;
			values[2] = SPI_getbinval(spi_tuple, tuptable->tupdesc, 3, &isnull);
			nulls[2] = isnull;

			result_tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);

			SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(result_tuple));
		}
		else
		{
			if (session2 != NULL)
				ndb_spi_session_end(&session2);
			SRF_RETURN_DONE(funcctx);
		}
	}
}

/*
 * temporal_score
 *	  Compute time-decayed similarity score.
 *
 * base_score is typically similarity (e.g., negative L2 distance).
 * insert_time, current_time in microseconds.
 * decay_rate = in 1/days (e.g., 0.1 means about e^-0.1 decay per day).
 */
PG_FUNCTION_INFO_V1(temporal_score);

Datum
temporal_score(PG_FUNCTION_ARGS)
{
	float4		base_score = PG_GETARG_FLOAT4(0);
	TimestampTz insert_time = PG_GETARG_TIMESTAMPTZ(1);
	TimestampTz current_time = PG_GETARG_TIMESTAMPTZ(2);
	float8		decay_rate = PG_GETARG_FLOAT8(3);
	float8		age_days;
	float8		decay_factor;
	float4		final_score;

	age_days = (float8) (current_time - insert_time) / USECS_PER_DAY;
	decay_factor = exp(-decay_rate * age_days);
	final_score = base_score * decay_factor;

	PG_RETURN_FLOAT4(final_score);
}

/*
 * VectorToLiteral
 *	  Converts a Vector * to a single-quoted SQL literal, for use in SPI SQL.
 *	  Caller is responsible for pfree'ing result.
 */
static char *
VectorToLiteral(Vector *v)
{
	char	   *out;
	char	   *buf;

	out = vector_out_internal(v);
	buf = palloc(strlen(out) + 4);
	snprintf(buf, strlen(out) + 4, "'%s'", out);

	return buf;
}

/*
 * vector_l2
 *	  L2 distance function. Provide this or use an appropriate function from a vector extension.
 */
PG_FUNCTION_INFO_V1(vector_l2);

Datum
vector_l2(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;
	float4		sum = 0.0;
	int			i;

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);

	if (a->dim != b->dim)
		elog(ERROR, "vector_l2: dimensions do not match");

	for (i = 0; i < a->dim; i++)
	{
		float4		d = a->data[i] - b->data[i];

		sum += d * d;
	}

	PG_RETURN_FLOAT4(sqrt(sum));
}
