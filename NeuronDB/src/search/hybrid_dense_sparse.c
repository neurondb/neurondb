/*-------------------------------------------------------------------------
 *
 * hybrid_dense_sparse.c
 *    Hybrid dense + sparse search implementation
 *
 * Combines dense vector search (HNSW) with sparse vector search (inverted
 * index) for improved retrieval quality. Uses Reciprocal Rank Fusion (RRF)
 * or learned fusion weights.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/search/hybrid_dense_sparse.c
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
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"
#include "neurondb_macros.h"
#include <string.h>
#include <math.h>

/*
 * hybrid_dense_sparse_search: Combine dense and sparse search
 */
PG_FUNCTION_INFO_V1(hybrid_dense_sparse_search);
Datum
hybrid_dense_sparse_search(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_PP(0);
	text	   *dense_col = PG_GETARG_TEXT_PP(1);
	text	   *sparse_col = PG_GETARG_TEXT_PP(2);
	Datum		dense_query = PG_GETARG_DATUM(3);
	Datum		sparse_query = PG_GETARG_DATUM(4);
	int32		k = PG_GETARG_INT32(5);
	float4		dense_weight = PG_ARGISNULL(6) ? 0.5f : PG_GETARG_FLOAT4(6);
	float4		sparse_weight = PG_ARGISNULL(7) ? 0.5f : PG_GETARG_FLOAT4(7);
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
	Datum		args[2];
	Oid			argtypes[2];
	char	   *tbl_str = text_to_cstring(table_name);
	char	   *dense_str = text_to_cstring(dense_col);
	char	   *sparse_str = text_to_cstring(sparse_col);

	if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("hybrid_dense_sparse_search must be called as table function")));

	if (!(rsinfo->allowedModes & SFRM_Materialize))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("hybrid_dense_sparse_search requires Materialize mode")));

	per_query_ctx = rsinfo->econtext->ecxt_per_query_memory;
	oldcontext = MemoryContextSwitchTo(per_query_ctx);

	tupdesc = CreateTemplateTupleDesc(2);
	TupleDescInitEntry(tupdesc, (AttrNumber) 1, "doc_id", INT4OID, -1, 0);
	TupleDescInitEntry(tupdesc, (AttrNumber) 2, "fused_score", FLOAT4OID, -1, 0);
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

	/* Perform hybrid search using weighted fusion */
	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "WITH dense_scores AS ("
					 "  SELECT ctid, 1.0 / (1.0 + l2_distance(%s, $1)) AS score "
					 "  FROM %s "
					 "  WHERE %s IS NOT NULL "
					 "  ORDER BY l2_distance(%s, $1) "
					 "  LIMIT %d"
					 "), "
					 "sparse_scores AS ("
					 "  SELECT ctid, sparse_vector_dot_product(%s, $2) AS score "
					 "  FROM %s "
					 "  WHERE %s IS NOT NULL "
					 "  ORDER BY sparse_vector_dot_product(%s, $2) DESC "
					 "  LIMIT %d"
					 "), "
					 "fused AS ("
					 "  SELECT "
					 "    COALESCE(d.ctid, s.ctid) AS ctid, "
					 "    COALESCE(d.score, 0.0) * %.2f + "
					 "    COALESCE(s.score, 0.0) * %.2f AS fused_score "
					 "  FROM dense_scores d "
					 "  FULL OUTER JOIN sparse_scores s ON d.ctid = s.ctid"
					 ") "
					 "SELECT ctid, fused_score "
					 "FROM fused "
					 "ORDER BY fused_score DESC "
					 "LIMIT %d",
					 dense_str,
					 tbl_str,
					 dense_str,
					 dense_str,
					 k * 2,
					 sparse_str,
					 tbl_str,
					 sparse_str,
					 sparse_str,
					 k * 2,
					 dense_weight,
					 sparse_weight,
					 k);

	args[0] = dense_query;
	args[1] = sparse_query;
	argtypes[0] = get_fn_expr_argtype(fcinfo->flinfo, 3);
	argtypes[1] = get_fn_expr_argtype(fcinfo->flinfo, 4);

	ret = ndb_spi_execute_with_args(session,
									sql.data,
									2,
									argtypes,
									args,
									NULL,
									false,
									0);

	if (ret != SPI_OK_SELECT)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("hybrid_dense_sparse_search query failed")));

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
 * rrf_fusion: Reciprocal Rank Fusion for combining dense and sparse results
 */
PG_FUNCTION_INFO_V1(rrf_fusion);
Datum
rrf_fusion(PG_FUNCTION_ARGS)
{
	float4		dense_rank;
	float4		sparse_rank;
	float4		k_param;
	float4		rrf_score;

	PG_GETARG_INT32(0);			/* k - reserved for future use */
	dense_rank = PG_GETARG_FLOAT4(1);
	sparse_rank = PG_GETARG_FLOAT4(2);
	k_param = PG_ARGISNULL(3) ? 60.0f : PG_GETARG_FLOAT4(3);

	/* RRF formula: 1 / (k + rank) */
	rrf_score = 1.0f / (k_param + dense_rank) + 1.0f / (k_param + sparse_rank);

	PG_RETURN_FLOAT4(rrf_score);
}
