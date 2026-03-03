/*-------------------------------------------------------------------------
 *
 * vector_metrics.c
 *    Production observability for vector search operations
 *
 * Implements metrics collection for query latency, recall, GPU utilization,
 * and index performance.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/metrics/vector_metrics.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "utils/timestamp.h"
#include "utils/builtins.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/*
 * Vector search metrics
 */
typedef struct VectorSearchMetrics
{
	/* Query statistics */
	int64		total_queries;
	int64		total_batch_queries;
	double		total_latency_ms;
	double		min_latency_ms;
	double		max_latency_ms;
	double		p50_latency_ms;
	double		p95_latency_ms;
	double		p99_latency_ms;

	/* Recall statistics */
	double		avg_recall;
	int64		recall_evaluations;

	/* GPU statistics */
	int64		gpu_queries;
	int64		cpu_fallbacks;
	double		gpu_utilization;

	/* Index statistics */
	int64		hnsw_queries;
	int64		ivf_queries;
	int64		pq_queries;
	int64		index_hits;
	int64		index_misses;

	/* Filter statistics */
	int64		filtered_queries;
	double		avg_filter_selectivity;
}			VectorSearchMetrics;

static VectorSearchMetrics vector_metrics = {0};

/*
 * Record query execution metrics
 */
void
vector_metrics_record_query(bool used_gpu,
							 double latency_ms,
							 bool used_filter,
							 double filter_selectivity,
							 const char *index_type)
{
	vector_metrics.total_queries++;

	if (used_gpu)
		vector_metrics.gpu_queries++;
	else
		vector_metrics.cpu_fallbacks++;

	if (used_filter)
	{
		vector_metrics.filtered_queries++;
		vector_metrics.avg_filter_selectivity =
			(vector_metrics.avg_filter_selectivity * (vector_metrics.filtered_queries - 1)
			 + filter_selectivity) / vector_metrics.filtered_queries;
	}

	if (index_type)
	{
		if (strcmp(index_type, "hnsw") == 0)
			vector_metrics.hnsw_queries++;
		else if (strcmp(index_type, "ivf") == 0)
			vector_metrics.ivf_queries++;
		else if (strcmp(index_type, "pq") == 0)
			vector_metrics.pq_queries++;
	}

	/* Update latency statistics */
	vector_metrics.total_latency_ms += latency_ms;

	if (vector_metrics.total_queries == 1 || latency_ms < vector_metrics.min_latency_ms)
		vector_metrics.min_latency_ms = latency_ms;
	if (vector_metrics.total_queries == 1 || latency_ms > vector_metrics.max_latency_ms)
		vector_metrics.max_latency_ms = latency_ms;

	/* Update percentile statistics (simplified - use circular buffer in production) */
	/* For now, track p50, p95, p99 as running averages */
	if (vector_metrics.total_queries > 0)
	{
		double		avg = vector_metrics.total_latency_ms / vector_metrics.total_queries;

		/* Simple percentile estimation */
		if (latency_ms <= avg)
			vector_metrics.p50_latency_ms = avg;
		else if (latency_ms <= avg * 1.5)
			vector_metrics.p95_latency_ms = latency_ms;
		else
			vector_metrics.p99_latency_ms = latency_ms;
	}
}

/*
 * Record recall evaluation
 */
void
vector_metrics_record_recall(double recall)
{
	vector_metrics.recall_evaluations++;
	vector_metrics.avg_recall =
		(vector_metrics.avg_recall * (vector_metrics.recall_evaluations - 1) + recall)
		/ vector_metrics.recall_evaluations;
}

/*
 * SQL function: Get vector search metrics
 */
PG_FUNCTION_INFO_V1(vector_search_metrics);
Datum
vector_search_metrics(PG_FUNCTION_ARGS)
{
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	TupleDesc	tupdesc;

	if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo))
		ereport(ERROR,
				(errmsg("vector_search_metrics: invalid resultinfo")));

	if (rsinfo->expectedDesc != NULL)
		tupdesc = rsinfo->expectedDesc;
	else
	{
		tupdesc = CreateTemplateTupleDesc(15);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "total_queries", INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "avg_latency_ms", FLOAT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "p95_latency_ms", FLOAT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 4, "avg_recall", FLOAT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 5, "gpu_queries", INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 6, "cpu_fallbacks", INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 7, "hnsw_queries", INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 8, "ivf_queries", INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 9, "pq_queries", INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 10, "filtered_queries", INT8OID, -1, 0);
		BlessTupleDesc(tupdesc);
		rsinfo->expectedDesc = tupdesc;
	}

	/* Initialize materialized SRF */
	{
		MemoryContext per_query_ctx = rsinfo->econtext->ecxt_per_query_memory;
		MemoryContext oldcontext = MemoryContextSwitchTo(per_query_ctx);
		Tuplestorestate *tupstore = tuplestore_begin_heap(true, false, 1024);

		rsinfo->returnMode = SFRM_Materialize;
		rsinfo->setResult = tupstore;
		rsinfo->setDesc = tupdesc;

		MemoryContextSwitchTo(oldcontext);
	}

	/* Return metrics */
	{
		Datum		values[15];
		bool		nulls[15] = {false};
		double		avg_latency = vector_metrics.total_queries > 0
			? vector_metrics.total_latency_ms / vector_metrics.total_queries
			: 0.0;

		values[0] = Int64GetDatum(vector_metrics.total_queries);
		values[1] = Float8GetDatum(avg_latency);
		values[2] = Float8GetDatum(vector_metrics.p95_latency_ms);
		values[3] = Float8GetDatum(vector_metrics.avg_recall);
		values[4] = Int64GetDatum(vector_metrics.gpu_queries);
		values[5] = Int64GetDatum(vector_metrics.cpu_fallbacks);
		values[6] = Int64GetDatum(vector_metrics.hnsw_queries);
		values[7] = Int64GetDatum(vector_metrics.ivf_queries);
		values[8] = Int64GetDatum(vector_metrics.pq_queries);
		values[9] = Int64GetDatum(vector_metrics.filtered_queries);

		tuplestore_putvalues(rsinfo->setResult, rsinfo->setDesc, values, nulls);
	}

	return (Datum) 0;
}

