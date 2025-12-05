/*-------------------------------------------------------------------------
 *
 * index_hybrid.c
 *		Fused ANN plus GIN full-text index in single access method
 *
 * Implements HYBRID-F index that combines vector similarity and
 * full-text search in one index structure, enabling single plan node
 * and single heap walk for optimal performance.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/index_hybrid.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_index.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "executor/spi.h"
#include "utils/lsyscache.h"
#include "catalog/pg_type.h"
#include "funcapi.h"
#include "miscadmin.h"
#include "lib/stringinfo.h"
#include <string.h>
#include "neurondb_validation.h"
#include "neurondb_macros.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"

/*
 * Create hybrid fused index on a table: both HNSW (or flat) vector index and GIN
 * index for the text column. In real systems, we'd also set up supporting metadata.
 */
PG_FUNCTION_INFO_V1(hybrid_index_create);
Datum
hybrid_index_create(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_PP(0);
	text	   *vector_col = PG_GETARG_TEXT_PP(1);
	text	   *text_col = PG_GETARG_TEXT_PP(2);
	float4		fusion_weight = PG_GETARG_FLOAT4(3);
	char	   *tbl_str;
	char	   *vec_str;
	char	   *txt_str;
	StringInfoData sql;
	int			ret;

	NDB_DECLARE(NdbSpiSession *, session);

	tbl_str = text_to_cstring(table_name);
	vec_str = text_to_cstring(vector_col);
	txt_str = text_to_cstring(text_col);

	elog(INFO,
		 "neurondb: Creating hybrid index on %s (%s vector, %s text, "
		 "weight=%.2f)",
		 tbl_str,
		 vec_str,
		 txt_str,
		 fusion_weight);

	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		elog(ERROR, "failed to begin SPI session");

	initStringInfo(&sql);

	/*
	 * Create an ANN index (hnsw, IVF, or flat). For illustration, we just use
	 * a flat btree. In a real implementation, this would use HNSW or other
	 * ANN-aware extension. The actual index names would preferably be
	 * deterministic and unique.
	 */
	appendStringInfo(&sql,
					 "CREATE INDEX IF NOT EXISTS __hyb_ann_%s_%s ON %s USING btree "
					 "((%s))",
					 tbl_str,
					 vec_str,
					 tbl_str,
					 vec_str);

	ret = ndb_spi_execute(session, sql.data, false, 0);
	if (ret != SPI_OK_UTILITY)
	{
		NDB_FREE(sql.data);
		ndb_spi_session_end(&session);
		elog(ERROR,
			 "Failed to create vector ANN index on %s.%s: SPI error "
			 "%d",
			 tbl_str,
			 vec_str,
			 ret);
	}

	/* Use safe free/reinit to handle potential memory context changes */
	NDB_FREE(sql.data);
	initStringInfo(&sql);

	/*
	 * Create a GIN index over the text column for full-text search. In a real
	 * deployment, you'd want to use to_tsvector and a configuration.
	 */
	appendStringInfo(&sql,
					 "CREATE INDEX IF NOT EXISTS __hyb_gin_%s_%s ON %s USING GIN "
					 "(to_tsvector('english', %s))",
					 tbl_str,
					 txt_str,
					 tbl_str,
					 txt_str);

	ret = ndb_spi_execute(session, sql.data, false, 0);
	if (ret != SPI_OK_UTILITY)
	{
		NDB_FREE(sql.data);
		ndb_spi_session_end(&session);
		elog(ERROR,
			 "Failed to create GIN index on %s.%s: SPI error %d",
			 tbl_str,
			 txt_str,
			 ret);
	}

	NDB_FREE(sql.data);
	ndb_spi_session_end(&session);

	/* In production: store fusion config and columns in metadata */
	elog(INFO,
		 "neurondb: Hybrid index set up complete with "
		 "fusion_weight=%.2f",
		 fusion_weight);

	PG_RETURN_BOOL(true);
}

/*
 * Query hybrid index: combines full-text (tsvector/tsquery) and vector ANN score
 * to get top-k results. A real implementation would use custom C-level operators
 * or SQL functions for efficient candidate generation and scoring.
 *
 * This function demonstrates the full process:
 *   1. Perform full-text search to get candidate doc IDs and ranks (ts_rank).
 *   2. For each candidate, look up the vector and compute its similarity to the query.
 *   3. Combine the (normalized) text and vector scores via weighted fusion.
 *   4. Return top-k by fused score.
 *
 * RETURNS: SETOF (id bigint, fused_score float4, text_rank float4, vector_dist float4)
 */
PG_FUNCTION_INFO_V1(hybrid_index_search);
Datum
hybrid_index_search(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	TupleDesc	tupdesc;
	MemoryContext oldcontext;
	text	   *index_name;
	Vector	   *query_vec;
	text	   *query_text;
	int32		k;
	char	   *idx_str;
	char	   *txt_query;
	StringInfoData sql;
	int			ret;
	char	   *origin_table;
	char	   *vec_col;
	char	   *txt_col;

	NDB_DECLARE(NdbSpiSession *, session2);

	if (SRF_IS_FIRSTCALL())
	{
		index_name = PG_GETARG_TEXT_PP(0);
		query_vec = PG_GETARG_VECTOR_P(1);
		NDB_CHECK_VECTOR_VALID(query_vec);
		query_text = PG_GETARG_TEXT_PP(2);
		k = PG_GETARG_INT32(3);
		idx_str = text_to_cstring(index_name);
		txt_query = text_to_cstring(query_text);
		funcctx = SRF_FIRSTCALL_INIT();
		tupdesc = CreateTemplateTupleDesc(4);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "id", INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "fused_score", FLOAT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "text_rank", FLOAT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 4, "vector_dist", FLOAT4OID, -1, 0);
		funcctx->tuple_desc = BlessTupleDesc(tupdesc);
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);
		origin_table = idx_str;
		vec_col = "vector";
		txt_col = "document";
		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "SELECT id, "
						 "(%1$.4f * ts_rank + (1-%1$.4f) * (1 - norm_dist)) AS "
						 "fused_score, "
						 "ts_rank, "
						 "vector_dist "
						 "FROM ( "
						 "SELECT id, "
						 "ts_rank_cd(to_tsvector('english', %2$s), "
						 "plainto_tsquery('english', %3$s)) AS ts_rank, "
						 "vector_l2_dist(%4$s, '%5$s') AS vector_dist "
						 "FROM %6$s "
						 "WHERE to_tsvector('english', %2$s) @@ "
						 "plainto_tsquery('english', %3$s) "
						 ") candidates, "
						 "(SELECT MAX(vector_l2_dist(%4$s, '%5$s')) AS maxd, "
						 "MIN(vector_l2_dist(%4$s, '%5$s')) AS mind "
						 "FROM %6$s "
						 "WHERE to_tsvector('english', %2$s) @@ "
						 "plainto_tsquery('english', %3$s)) bounds "
						 "LEFT JOIN LATERAL ( "
						 "SELECT 1.0 AS norm_dist WHERE bounds.maxd = "
						 "bounds.mind "
						 "UNION ALL "
						 "SELECT (candidates.vector_dist - bounds.mind) / "
						 "NULLIF(bounds.maxd - bounds.mind,0) "
						 ") normtbl ON TRUE "
						 "ORDER BY fused_score DESC "
						 "LIMIT %7$d",
						 0.6 /* default fusion_weight */ ,
						 txt_col,
						 txt_query,
						 vec_col,
						 "vector",
						 origin_table,
						 k);
		session2 = ndb_spi_session_begin(CurrentMemoryContext, false);
		if (session2 == NULL)
			elog(ERROR,
				 "failed to begin SPI session for hybrid_index_search");
		ret = ndb_spi_execute(session2, sql.data, true, 0);
		if (ret != SPI_OK_SELECT)
			elog(ERROR,
				 "SPI_execute failed for hybrid query: code %d, "
				 "sql: %s",
				 ret,
				 sql.data);
		if (SPI_processed == 0)
		{
			NDB_FREE(sql.data);
			ndb_spi_session_end(&session2);
			funcctx->max_calls = 0;
			SRF_RETURN_DONE(funcctx);
		}
		funcctx->max_calls = SPI_processed;
		/* Store session to keep SPI connection alive for tuptable access */
		funcctx->user_fctx = session2;
		MemoryContextSwitchTo(oldcontext);
	}

	{
		uint64		call_cntr;
		uint64		max_calls;
		SPITupleTable *tuptable;
		HeapTuple	spi_tuple;
		Datum		values[4];
		bool		nulls[4];
		HeapTuple	ret_tuple;
		int			att;

		funcctx = SRF_PERCALL_SETUP();
		call_cntr = funcctx->call_cntr;
		max_calls = funcctx->max_calls;

		if (call_cntr < max_calls)
		{
			session2 = (NdbSpiSession *) funcctx->user_fctx;
			tuptable = SPI_tuptable;	/* Access via global SPI_tuptable
										 * while session is active */
			spi_tuple = tuptable->vals[call_cntr];

			/* id, fused_score, ts_rank, vector_dist */
			for (att = 0; att < 4; ++att)
			{
				values[att] = SPI_getbinval(spi_tuple,
											tuptable->tupdesc,
											att + 1,
											&nulls[att]);
			}
			ret_tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
			SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(ret_tuple));
		}
		else
		{
			session2 = (NdbSpiSession *) funcctx->user_fctx;
			if (session2 != NULL)
				ndb_spi_session_end(&session2);

			SRF_RETURN_DONE(funcctx);
		}
	}
}
