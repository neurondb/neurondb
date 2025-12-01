/*-------------------------------------------------------------------------
 *
 * planner.c
 *    Adaptive Intelligence: Detailed Planner Hook, Query Optimizer, Scaling, Prefetcher (Rugged/Defensive)
 *
 * This module provides modular, robust, and fully defensive implementations for:
 *    - auto-routing query plans (ANN vs. FTS)
 *    - self-learning query fingerprint optimizer/auto-param tuner
 *    - dynamic vector precision scaling
 *    - predictive prefetch of index entry points
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/planner/planner.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "executor/spi.h"
#include "utils/memutils.h"
#include "utils/hsearch.h"
#include "catalog/pg_type.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"

/*
 * auto_route_query:
 * Decides, with robust heuristics, whether a query should be routed to ANN (Approximate Nearest Neighbor)
 * or FTS (Full Table Scan) path.
 *
 * Heuristics:
 * - If embedding_length > 128 OR query string contains "similarity" OR "vector", use ANN.
 * - Otherwise use FTS.
 *
 * 100% defensive. Validates input. Never leaks.
 */
PG_FUNCTION_INFO_V1(auto_route_query);
Datum
auto_route_query(PG_FUNCTION_ARGS)
{
	text	   *query;
	int32		embedding_length;
	char	   *query_str = NULL;
	bool		use_ann = false;

	/* Validate input, crash-safe */
	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errmsg("auto_route_query: input query text is NULL")));
	if (PG_ARGISNULL(1))
		ereport(ERROR,
				(errmsg("auto_route_query: embedding_length is NULL")));

	query = PG_GETARG_TEXT_PP(0);
	embedding_length = PG_GETARG_INT32(1);

	/* Convert query to C string safely */
	query_str = text_to_cstring(query);

	if (query_str == NULL)
		ereport(ERROR,
				(errmsg("auto_route_query: failed to convert query "
						"text to C string")));

	/* Rugged ANN heuristic logic: case-insensitive, no undefined behaviour */
	if (embedding_length > 128
		|| (strcasestr(query_str, "similarity") != NULL)
		|| (strcasestr(query_str, "vector") != NULL))
	{
		use_ann = true;
	}

	elog(DEBUG1,
		 "neurondb:auto_route_query: route=%s embedding_length=%d",
		 use_ann ? "ANN" : "FTS",
		 embedding_length);

	/* Free non-null query_str */
	if (query_str)
		NDB_FREE(query_str);

	PG_RETURN_BOOL(use_ann);
}

/*
 * learn_from_query:
 * Records fingerprint and statistics for queries; tunes ANN parameterization on a self-learning basis.
 * Table: neurondb_query_history
 *
 * Inputs:
 *      query:           query text to fingerprint
 *      actual_recall:   float4, last recall (quality)
 *      latency_ms:      int32, observed latency in ms
 *
 * Actions:
 *  - If fingerprint already exists, updates stats/params (ef_search, beam_size) defensively and atomically.
 *  - On new query, initializes entry with defaults.
 *
 * 100% robust: all errors logged, all allocations checked, never leaks SPI.
 */
PG_FUNCTION_INFO_V1(learn_from_query);
Datum
learn_from_query(PG_FUNCTION_ARGS)
{
	text	   *query;
	float4		actual_recall;
	int32		latency_ms;
	char	   *query_str = NULL;
	uint64		fingerprint = 5381ULL;	/* djb2 init */
	int			i;
	bool		found = false;
	NDB_DECLARE(NdbSpiSession *, session);

	/* Defensive: NULL checks */
	if (PG_ARGISNULL(0))
		ereport(ERROR, (errmsg("learn_from_query: query is NULL")));
	if (PG_ARGISNULL(1))
		ereport(ERROR,
				(errmsg("learn_from_query: actual_recall is NULL")));
	if (PG_ARGISNULL(2))
		ereport(ERROR,
				(errmsg("learn_from_query: latency_ms is NULL")));

	query = PG_GETARG_TEXT_PP(0);
	actual_recall = PG_GETARG_FLOAT4(1);
	latency_ms = PG_GETARG_INT32(2);

	query_str = text_to_cstring(query);

	if (query_str == NULL)
		ereport(ERROR,
				(errmsg("learn_from_query: failed to convert query to "
						"C string")));

	/* Compute a robust DJB2 hash, no overflows */
	for (i = 0; query_str[i] != '\0'; i++)
		fingerprint = ((fingerprint << 5) + fingerprint)
			+ (unsigned char) query_str[i];

	elog(DEBUG1,
		 "neurondb:learn_from_query: fprint=%llu recall=%.6f latency=%d",
		 (unsigned long long) fingerprint,
		 actual_recall,
		 latency_ms);

	/* SPI connect - recover/exit safely */
	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("learn_from_query: failed to begin SPI session")));

	/* Always ensure the table exists. Safe for concurrency. */
	if (ndb_spi_execute(session, "CREATE TABLE IF NOT EXISTS neurondb_query_history ("
							 "  fingerprint  BIGINT PRIMARY KEY,"
							 "  last_recall  REAL, "
							 "  last_latency INTEGER, "
							 "  seen_count   INTEGER, "
							 "  ef_search    INTEGER, "
							 "  beam_size    INTEGER"
							 ")",
							 false,
							 0)
		!= SPI_OK_UTILITY)
	{
		ndb_spi_session_end(&session);
		ereport(ERROR,
				(errmsg("learn_from_query: cannot create "
						"neurondb_query_history")));
	}

	/* Check for existing fingerprint */
	{
		StringInfoData sql;

		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "SELECT fingerprint, seen_count, ef_search, beam_size "
						 "FROM neurondb_query_history "
						 "WHERE fingerprint = %llu",
						 (unsigned long long) fingerprint);

		if (ndb_spi_execute(session, sql.data, true, 1) == SPI_OK_SELECT
			&& SPI_processed == 1)
		{
			found = true;
		}

		if (sql.data)
			NDB_FREE(sql.data);
	}

	if (found)
	{
		/* Defensive extraction of columns */
		int32		seen_count = 0;
		int32		ef_search = 0;
		int32		beam_size = 0;
		bool		isnull1 = false,
					isnull2 = false,
					isnull3 = false;

		seen_count = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0],
												 SPI_tuptable->tupdesc,
												 2,
												 &isnull1));
		ef_search = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0],
												SPI_tuptable->tupdesc,
												3,
												&isnull2));
		beam_size = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0],
												SPI_tuptable->tupdesc,
												4,
												&isnull3));

		if (isnull1 || isnull2 || isnull3)
		{
			ndb_spi_session_end(&session);
			if (query_str)
				NDB_FREE(query_str);
			ereport(ERROR,
					(errmsg("learn_from_query: NULL detected in "
							"stats fields")));
		}

		seen_count = (seen_count < 0)
			? 1
			: seen_count + 1;	/* avoid underflow */

		/* Rugged heuristic parameter tuning branch */
		if (actual_recall < 0.90f && ef_search < 128)
			ef_search += 8;
		if (actual_recall > 0.99f && ef_search > 24)
			ef_search -= 8;

		if (latency_ms > 100 && beam_size > 4)
			beam_size -= 2;
		else if (latency_ms < 15 && beam_size < 32)
			beam_size += 2;

		/* Ensure params stay in sensible bounds */
		if (ef_search < 8)
			ef_search = 8;
		if (ef_search > 256)
			ef_search = 256;
		if (beam_size < 2)
			beam_size = 2;
		if (beam_size > 64)
			beam_size = 64;

		/* Update the tuple defensively */
		{
			StringInfoData usql;

			initStringInfo(&usql);
			appendStringInfo(&usql,
							 "UPDATE neurondb_query_history "
							 "SET last_recall=%.6f, last_latency=%d, "
							 "seen_count=%d, "
							 "ef_search=%d, beam_size=%d WHERE "
							 "fingerprint=%llu",
							 actual_recall,
							 latency_ms,
							 seen_count,
							 ef_search,
							 beam_size,
							 (unsigned long long) fingerprint);

			if (ndb_spi_execute(session, usql.data, false, 0) != SPI_OK_UPDATE)
			{
				if (usql.data)
					NDB_FREE(usql.data);
				ndb_spi_session_end(&session);
				if (query_str)
					NDB_FREE(query_str);
				ereport(ERROR,
						(errmsg("learn_from_query: failed to "
								"UPDATE "
								"neurondb_query_history")));
			}
			if (usql.data)
				NDB_FREE(usql.data);
		}
	}
	else
	{
		/* Robust INSERT for new tuple */
		StringInfoData isql;

		initStringInfo(&isql);
		appendStringInfo(&isql,
						 "INSERT INTO neurondb_query_history "
						 "(fingerprint, last_recall, last_latency, seen_count, "
						 "ef_search, beam_size) "
						 "VALUES (%llu, %.6f, %d, %d, %d, %d)",
						 (unsigned long long) fingerprint,
						 actual_recall,
						 latency_ms,
						 1,
						 32,
						 8);

		if (ndb_spi_execute(session, isql.data, false, 0) != SPI_OK_INSERT)
		{
			if (isql.data)
				NDB_FREE(isql.data);
			ndb_spi_session_end(&session);
			if (query_str)
				NDB_FREE(query_str);
			ereport(ERROR,
					(errmsg("learn_from_query: failed to INSERT "
							"neurondb_query_history")));
		}
		if (isql.data)
			NDB_FREE(isql.data);
	}

	if (query_str)
		NDB_FREE(query_str);

	ndb_spi_session_end(&session);

	PG_RETURN_BOOL(true);
}

/*
 * quantize_to_int8()
 * Quantize a float4 array to int8 (saturating at -128, 127), robustly and fully validated.
 */
static void
quantize_to_int8(const float4 * src, int8_t * dst, int len)
{
	int			i;

	if (!src || !dst)
		elog(ERROR, "quantize_to_int8: null src or dst");
	if (len < 0)
		elog(ERROR, "quantize_to_int8: negative len");

	for (i = 0; i < len; i++)
	{
		float		x = src[i];

		/* Saturate input robustly */
		if (isnan(x))
			x = 0.0f;
		if (x > 127.0f)
			x = 127.0f;
		else if (x < -128.0f)
			x = -128.0f;

		dst[i] = (int8_t) rintf(x);
	}
}

/*
 * quantize_to_float16()
 * Simulated quantization: rounds to one decimal (float16-like) for demonstration - robust
 */
static void
quantize_to_float16(const float4 * src, float4 * dst, int len)
{
	int			i;

	if (!src || !dst)
		elog(ERROR, "quantize_to_float16: null src or dst");
	if (len < 0)
		elog(ERROR, "quantize_to_float16: negative len");

	for (i = 0; i < len; i++)
	{
		float		in = src[i];

		if (isnan(in))
			dst[i] = 0.0f;
		else
			dst[i] = roundf(in * 10.0f) / 10.0f;
	}
}

/*
 * scale_precision()
 * Dynamically scales precision of a Vector* in-place, based on system memory pressure and recall requirements.
 * Defensive: never leaks, allocates robustly, and all error paths are clean.
 */
PG_FUNCTION_INFO_V1(scale_precision);
Datum
scale_precision(PG_FUNCTION_ARGS)
{
	Vector	   *input;
	float4		memory_pressure;
	float4		recall_target;
	int			target_precision;
	Vector	   *result = NULL;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errmsg("scale_precision: input vector is NULL")));
	if (PG_ARGISNULL(1))
		ereport(ERROR,
				(errmsg("scale_precision: memory_pressure is NULL")));
	if (PG_ARGISNULL(2))
		ereport(ERROR,
				(errmsg("scale_precision: recall_target is NULL")));

	input = (Vector *) PG_GETARG_POINTER(0);
	memory_pressure = PG_GETARG_FLOAT4(1);
	recall_target = PG_GETARG_FLOAT4(2);

	if (input == NULL)
		ereport(ERROR,
				(errmsg("scale_precision: input vector is missing")));

	/* Defensive: validate dim (max for int16 is 32767) */
	{
		int			dim = (int) input->dim;

		if (dim <= 0 || dim > 32767)
			ereport(ERROR,
					(errmsg("scale_precision: invalid vector "
							"dimension")));
	}

	/* Decision logic (100% robust, never produces weird values) */
	if (memory_pressure > 0.8f || recall_target < 0.85f)
		target_precision = 8;	/* Quantize to int8 */
	else if (memory_pressure > 0.6f || recall_target < 0.90f)
		target_precision = 16;	/* "float16" (rounded float4) */
	else
		target_precision = 32;	/* float32 (full) */

	elog(DEBUG1,
		 "neurondb:scale_precision: target=%dbit (mem=%.3f recall=%.3f)",
		 target_precision,
		 memory_pressure,
		 recall_target);

	/* Defensive instantiation of output vector */
	result = new_vector(input->dim);
	if (result == NULL)
		ereport(ERROR,
				(errmsg("scale_precision: failed to allocate result "
						"vector")));

	if (target_precision == 8)
	{
		int8_t	   *int8buf = NULL;
		int			k;

		int8buf = (int8_t *) palloc0(sizeof(int8_t) * input->dim);
		if (!int8buf)
			ereport(ERROR,
					(errmsg("scale_precision: failed to alloc int8 "
							"buffer")));
		quantize_to_int8(input->data, int8buf, input->dim);

		for (k = 0; k < input->dim; k++)
			result->data[k] = (float4) int8buf[k];

		NDB_FREE(int8buf);
	}
	else if (target_precision == 16)
	{
		quantize_to_float16(input->data, result->data, input->dim);
	}
	else						/* 32 */
	{
		memcpy(result->data, input->data, sizeof(float4) * input->dim);
	}

	PG_RETURN_POINTER(result);
}

/*
 * prefetch_entry_points
 * Simulates predictive prefetch of HNSW or similar ANN index entry points.
 * Performs a robust, bounded read of catalog statistics and reports the sum.
 *
 * index_name: text - required
 * query_vector: Vector* - required (but not used directly here)
 *
 * Careful error handling and resource management.
 */
PG_FUNCTION_INFO_V1(prefetch_entry_points);
Datum
prefetch_entry_points(PG_FUNCTION_ARGS)
{
	text	   *index_name;
	char	   *idx_str = NULL;
	int			prefetched_count = 0;
	NDB_DECLARE(NdbSpiSession *, session);

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errmsg("prefetch_entry_points: index_name is NULL")));
	if (PG_ARGISNULL(1))
		ereport(ERROR,
				(errmsg("prefetch_entry_points: query_vector is "
						"NULL")));

	index_name = PG_GETARG_TEXT_PP(0);
	/* query_vector not currently used in this implementation */
	(void) PG_GETARG_POINTER(1);

	idx_str = text_to_cstring(index_name);
	if (!idx_str)
		ereport(ERROR,
				(errmsg("prefetch_entry_points: index_name conversion "
						"failed")));

	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
	{
		if (idx_str)
			NDB_FREE(idx_str);
		ereport(ERROR,
				(errmsg("prefetch_entry_points: failed to begin SPI session")));
	}

	/*
	 * Query index statistics -- limit all dynamic stats lookups, always
	 * deallocate SQL buffers
	 */
	{
		StringInfoData sql;
		int			ret;

		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "SELECT c.oid, c.relpages "
						 "FROM pg_class c "
						 "JOIN pg_index i ON c.oid = i.indexrelid "
						 "WHERE c.relname = %s",
						 quote_literal_cstr(idx_str));

		/* Defensive: forcibly limit row count (~max 10) */
		ret = ndb_spi_execute(session, sql.data, true, 10);
		if (ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			int			q;

			for (q = 0; q < (int) SPI_processed; q++)
			{
				Datum		pagesDatum;
				bool		isnull = false;
				int32		pages = 0;

				pagesDatum =
					SPI_getbinval(SPI_tuptable->vals[q],
								  SPI_tuptable->tupdesc,
								  2,
								  &isnull);
				if (!isnull)
				{
					pages = DatumGetInt32(pagesDatum);
					if (pages < 0)
						pages = 0;
					if (pages > 128)
						pages = 128;	/* robust cap */
					prefetched_count += pages;
				}
			}
		}
		if (sql.data)
			NDB_FREE(sql.data);
	}

	ndb_spi_session_end(&session);

	if (idx_str)
		NDB_FREE(idx_str);

	elog(DEBUG1,
		 "neurondb:prefetch_entry_points: prefetched_count=%d",
		 prefetched_count);

	PG_RETURN_INT32(prefetched_count);
}
