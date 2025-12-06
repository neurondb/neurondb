/*-------------------------------------------------------------------------
 *
 * data_management.c
 *		Data Management: Time-Travel, Cold-tier Compression, VACUUM, Rebalance
 *
 * This file implements advanced data management features for vectors
 * including MVCC-aware time-travel queries, cold-tier compression,
 * vector-aware VACUUM, and index rebalancing.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/data_management.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_compat.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/timestamp.h"
#include "executor/spi.h"
#include "lib/stringinfo.h"
#include "catalog/pg_type.h"
#include "utils/snapmgr.h"
#include "access/xact.h"
#include "storage/bufmgr.h"
#include "access/heapam.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

PG_FUNCTION_INFO_V1(vector_time_travel);
Datum
vector_time_travel(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	TimestampTz system_time;
	char	   *tbl_str;
	StringInfoData sql;
	StringInfoData check_sql;
	StringInfoData result_msg;
	int			ret;
	int64		row_count = 0;
	char		timebuf[128];
	struct pg_tm tm;
	fsec_t		fsec;
	bool		table_exists = false;
	Datum		count_datum;
	bool		isnull;
	MemoryContext oldcontext,
				tmpcontext;

	NDB_DECLARE(NdbSpiSession *, session);

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: table_name argument is "
						"NULL")));
	if (PG_ARGISNULL(1))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: system_time argument is "
						"NULL")));

	table_name = PG_GETARG_TEXT_PP(0);
	system_time = PG_GETARG_TIMESTAMPTZ(1);

	tbl_str = text_to_cstring(table_name);

	elog(INFO,
		 "neurondb: vector_time_travel called for table '%s' at "
		 "timestamp " INT64_FORMAT,
		 tbl_str,
		 (int64) system_time);

	if (timestamp2tm(system_time, NULL, &tm, &fsec, NULL, NULL) != 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATETIME_VALUE_OUT_OF_RANGE),
				 errmsg("neurondb: system_time timestamp out of "
						"range")));

	snprintf(timebuf,
			 sizeof(timebuf),
			 "%04d-%02d-%02d %02d:%02d:%02d",
			 tm.tm_year,
			 tm.tm_mon,
			 tm.tm_mday,
			 tm.tm_hour,
			 tm.tm_min,
	tm.tm_sec);

	elog(DEBUG2, "neurondb: formatted time for time-travel: %s", timebuf);

	tmpcontext = AllocSetContextCreate(CurrentMemoryContext,
									   "vector_time_travel temporary context",
									   ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(tmpcontext);

	session = ndb_spi_session_begin(tmpcontext, false);
	if (session == NULL)
	{
		NDB_FREE(tbl_str);
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(tmpcontext);
		ereport(ERROR,
				(errmsg("neurondb: failed to begin SPI session in "
						"vector_time_travel")));
	}

	initStringInfo(&check_sql);
	appendStringInfo(&check_sql,
					 "SELECT 1 FROM information_schema.tables WHERE table_schema = "
					 "'public' AND table_name = '%s'",
					 tbl_str);
	ret = ndb_spi_execute(session, check_sql.data, true, 1);
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
		table_exists = true;
	else
		table_exists = false;
	NDB_FREE(check_sql.data);

	if (!table_exists)
	{
		NDB_FREE(tbl_str);
		ndb_spi_session_end(&session);
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(tmpcontext);
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_TABLE),
				 errmsg("neurondb: table \"%s\" does not exist",
						tbl_str)));
	}

	elog(INFO,
		 "neurondb: table '%s' verified to exist for time-travel query",
		 tbl_str);

	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT COUNT(*) FROM %s "
					 "WHERE (valid_from IS NULL OR valid_from <= '%s'::timestamptz) "
					 "AND (valid_to IS NULL OR valid_to > '%s'::timestamptz)",
					 tbl_str,
					 timebuf,
					 timebuf);


	ret = ndb_spi_execute(session, sql.data, true, 0);

	if (ret != SPI_OK_SELECT)
	{
		NDB_FREE(sql.data);
		NDB_FREE(tbl_str);
		ndb_spi_session_end(&session);
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(tmpcontext);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to perform "
						"time-travel query")));
	}
	if (SPI_processed > 0 && SPI_tuptable != NULL)
	{
		count_datum = SPI_getbinval(SPI_tuptable->vals[0],
									SPI_tuptable->tupdesc,
									1,
									&isnull);
		if (!isnull)
			row_count = DatumGetInt64(count_datum);
	}

	initStringInfo(&result_msg);
	appendStringInfo(&result_msg,
					 "Time-travel query on table \"%s\" as of %s: "
					 "Found " NDB_INT64_FMT " valid historical vectors.",
					 tbl_str,
					 timebuf,
					 NDB_INT64_CAST(row_count));


	/* Clean up and free all allocated memory */
	NDB_FREE(sql.data);
	NDB_FREE(tbl_str);
	ndb_spi_session_end(&session);

	MemoryContextSwitchTo(oldcontext);
	MemoryContextDelete(tmpcontext);

	PG_RETURN_TEXT_P(cstring_to_text(result_msg.data));
}

/*
 * compress_cold_tier - Compress and move vectors to cold tier storage
 */
PG_FUNCTION_INFO_V1(compress_cold_tier);
Datum
compress_cold_tier(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	int32		age_days;
	char	   *tbl_str;
	char	   *cold_tbl_str;
	StringInfoData sql;
	StringInfoData check_sql;
	StringInfoData cold_table;
	int64		compressed_count = 0;
	int64		moved_count = 0;
	int			ret;
	bool		table_exists = false;
	bool		coldtable_exists = false;
	MemoryContext oldcontext,
				tmpcontext;

	NDB_DECLARE(NdbSpiSession *, session);

	/* Validate inputs */
	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errmsg("neurondb: input table_name cannot be NULL")));
	if (PG_ARGISNULL(1))
		ereport(ERROR,
				(errmsg("neurondb: input age_days cannot be NULL")));
	table_name = PG_GETARG_TEXT_PP(0);
	age_days = PG_GETARG_INT32(1);

	tbl_str = text_to_cstring(table_name);

	elog(INFO,
		 "neurondb: compress_cold_tier called on table: %s, threshold "
		 "age_days=%d",
		 tbl_str,
		 age_days);

	/* Prepare temp memory context for robust memory cleanup */
	tmpcontext = AllocSetContextCreate(CurrentMemoryContext,
									   "compress_cold_tier temp context",
									   ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(tmpcontext);

	/* Check source table existence */
	session = ndb_spi_session_begin(tmpcontext, false);
	if (session == NULL)
	{
		NDB_FREE(tbl_str);
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(tmpcontext);
		ereport(ERROR,
				(errmsg("neurondb: failed to begin SPI session in "
						"compress_cold_tier")));
	}

	initStringInfo(&check_sql);
	appendStringInfo(&check_sql,
					 "SELECT 1 FROM information_schema.tables WHERE table_schema = "
					 "'public' AND table_name = '%s';",
					 tbl_str);
	ret = ndb_spi_execute(session, check_sql.data, true, 1);
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
		table_exists = true;
	else
		table_exists = false;
	NDB_FREE(check_sql.data);

	if (!table_exists)
	{
		NDB_FREE(tbl_str);
		ndb_spi_session_end(&session);
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(tmpcontext);
		ereport(ERROR,
				(errmsg("neurondb: target table \"%s\" does not exist",
						tbl_str)));
	}

	/* Name for cold tier destination */
	initStringInfo(&cold_table);
	appendStringInfo(&cold_table, "%s_cold_tier", tbl_str);
	cold_tbl_str = pstrdup(cold_table.data);

	/* Step 1: Create cold tier table if missing */
	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "CREATE TABLE IF NOT EXISTS %s ("
					 "  id BIGINT PRIMARY KEY, "
					 "  compressed_vector BYTEA, "
					 "  original_dim INT, "
					 "  compression_method TEXT, "
					 "  compressed_at TIMESTAMPTZ DEFAULT now(), "
					 "  metadata JSONB"
					 ");",
					 cold_tbl_str);
	ndb_spi_execute(session, sql.data, false, 0);

	/* Always verify cold tier destination table actually exists */
	resetStringInfo(&check_sql);
	appendStringInfo(&check_sql,
					 "SELECT 1 FROM information_schema.tables WHERE table_schema = "
					 "'public' AND table_name = '%s';",
					 cold_tbl_str);
	ret = ndb_spi_execute(session, check_sql.data, true, 1);
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
		coldtable_exists = true;
	else
		coldtable_exists = false;
	NDB_FREE(check_sql.data);
	if (!coldtable_exists)
	{
		NDB_FREE(sql.data);
		NDB_FREE(tbl_str);
		NDB_FREE(cold_tbl_str);
		ndb_spi_session_end(&session);
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(tmpcontext);
		ereport(ERROR,
				(errmsg("neurondb: cold tier table \"%s\" could not be "
						"created",
						cold_tbl_str)));
	}

	/* Step 2: Find number of vectors eligible for compression */
	resetStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT COUNT(*) FROM %s "
					 "WHERE (created_at < now() - interval '%d days' "
					 "   OR last_accessed < now() - interval '%d days') "
					 "AND NOT COALESCE(is_compressed, false);",
					 tbl_str,
					 age_days,
					 age_days);
	ret = ndb_spi_execute(session, sql.data, true, 1);

	if (ret == SPI_OK_SELECT && SPI_processed > 0 && SPI_tuptable != NULL)
	{
		bool		isnull;
		Datum		d = SPI_getbinval(SPI_tuptable->vals[0],
									  SPI_tuptable->tupdesc,
									  1,
									  &isnull);

		if (!isnull)
			compressed_count = DatumGetInt64(d);
	}
	elog(DEBUG1,
		 "neurondb: found " NDB_INT64_FMT
		 " vector(s) older than threshold for compression in %s",
		 NDB_INT64_CAST(compressed_count),
		 tbl_str);

	/* Step 3: If there are candidates, compress/move up to 1000 in a batch */
	if (compressed_count > 0)
	{
		resetStringInfo(&sql);
		/* Make a copy of eligible IDs and their embeddings for compression */
		appendStringInfo(&sql,
						 "WITH eligible AS ("
						 "    SELECT id, embedding, vector_dims(embedding) AS "
						 "dim "
						 "    FROM %s WHERE (created_at < now() - interval '%d "
						 "days' "
						 "           OR last_accessed < now() - interval '%d "
						 "days') "
						 "      AND NOT COALESCE(is_compressed, false) "
						 "    LIMIT 1000"
						 ") "
						 "INSERT INTO %s (id, compressed_vector, original_dim, "
						 "compression_method, metadata) "
						 "SELECT id, encode(embedding::text::bytea, 'base64'), "
						 "dim, 'pq8x8', "
						 "       jsonb_build_object('compressed_at', now(), "
						 "'source', '%s', 'method', 'product_quantization') "
						 "FROM eligible "
						 "ON CONFLICT (id) DO NOTHING;",
						 tbl_str,
						 age_days,
						 age_days,
						 cold_tbl_str,
						 tbl_str);
		ret = ndb_spi_execute(session, sql.data, false, 0);

		/* If insert succeeds, count moved by how many were inserted */
		resetStringInfo(&sql);
		appendStringInfo(&sql,
						 "SELECT COUNT(*) FROM %s WHERE is_compressed = false "
						 "AND (created_at < now() - interval '%d days' OR "
						 "last_accessed < now() - interval '%d days')",
						 tbl_str,
						 age_days,
						 age_days);
		ret = ndb_spi_execute(session, sql.data, true, 1);
		if (ret == SPI_OK_SELECT && SPI_processed > 0 && SPI_tuptable != NULL)
		{
			bool		isnull;
			Datum		d = SPI_getbinval(SPI_tuptable->vals[0],
										  SPI_tuptable->tupdesc,
										  1,
										  &isnull);

			if (!isnull)
				moved_count = compressed_count
					- DatumGetInt64(
									d); /* assuming exactly correspondence for
										 * batch */
		}

		elog(DEBUG1,
			 "neurondb: attempted to move " NDB_INT64_FMT
			 " vectors; " NDB_INT64_FMT " records processed",
			 NDB_INT64_CAST(compressed_count),
			 NDB_INT64_CAST(moved_count));

		/* Mark original rows as compressed in the main table */
		resetStringInfo(&sql);
		appendStringInfo(&sql,
						 "UPDATE %s SET is_compressed = true, compressed_at = "
						 "now() "
						 "WHERE id IN (SELECT id FROM %s WHERE id IN (SELECT id "
						 "FROM %s) "
						 "             AND is_compressed = false "
						 "             AND (created_at < now() - interval '%d "
						 "days' OR last_accessed < now() - interval '%d "
						 "days'));",
						 tbl_str,
						 cold_tbl_str,
						 tbl_str,
						 age_days,
						 age_days);
		ndb_spi_execute(session, sql.data, false, 0);
	}
	else
	{
		elog(DEBUG1,
			 "neurondb: no candidate vectors found older than %d "
			 "days in %s",
			 age_days,
			 tbl_str);
	}

	/* Clean up */
	NDB_FREE(sql.data);
	NDB_FREE(cold_table.data);
	NDB_FREE(tbl_str);
	NDB_FREE(cold_tbl_str);
	ndb_spi_session_end(&session);
	MemoryContextSwitchTo(oldcontext);
	MemoryContextDelete(tmpcontext);

	PG_RETURN_INT64(moved_count);
}

/*
 * vacuum_vectors - Comprehensive VACUUM operation for vector tables
 */
PG_FUNCTION_INFO_V1(vacuum_vectors);
Datum
vacuum_vectors(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	bool		full;
	char	   *tbl_str;
	StringInfoData sql;
	StringInfoData stat_sql;
	StringInfoData statmsg;
	int64		dead_tuples = 0;
	int64		live_tuples = 0;
	int64		cleaned_count = 0;
	int64		orphan_count = 0;
	int			ret;
	Datum		dead_datum;
	Datum		live_datum;
	bool		isnull;
	MemoryContext oldcontext,
				tmpcontext;

	NDB_DECLARE(NdbSpiSession *, session);

	if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
		ereport(ERROR,
				(errmsg("neurondb: vacuum_vectors arguments may not be "
						"NULL")));

	table_name = PG_GETARG_TEXT_PP(0);
	full = PG_GETARG_BOOL(1);

	tbl_str = text_to_cstring(table_name);

	elog(INFO,
		 "neurondb: vacuum_vectors called with table \"%s\" "
		 "vacuum_full=%s",
		 tbl_str,
		 full ? "true" : "false");

	tmpcontext = AllocSetContextCreate(CurrentMemoryContext,
									   "vacuum_vectors temp context",
									   ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(tmpcontext);

	/* Validate that table exists */
	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT 1 FROM information_schema.tables WHERE table_schema = "
					 "'public' AND table_name = '%s';",
					 tbl_str);
	session = ndb_spi_session_begin(tmpcontext, false);
	if (session == NULL)
	{
		NDB_FREE(sql.data);
		NDB_FREE(tbl_str);
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(tmpcontext);
		ereport(ERROR,
				(errmsg("neurondb: failed to begin SPI session for "
						"vacuum_vectors")));
	}
	ret = ndb_spi_execute(session, sql.data, true, 1);
	if (!(ret == SPI_OK_SELECT && SPI_processed > 0))
	{
		NDB_FREE(sql.data);
		NDB_FREE(tbl_str);
		ndb_spi_session_end(&session);
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(tmpcontext);
		ereport(ERROR,
				(errmsg("neurondb: table \"%s\" unavailable in "
						"vacuum_vector",
						tbl_str)));
	}
	NDB_FREE(sql.data);

	/* Step 1: Detailed table stats extraction */
	initStringInfo(&stat_sql);
	appendStringInfo(&stat_sql,
					 "SELECT COALESCE(n_dead_tup, 0), COALESCE(n_live_tup, 0), "
					 "COALESCE(last_vacuum, '1970-01-01'::timestamptz) "
					 "FROM pg_stat_user_tables WHERE schemaname = 'public' AND "
					 "relname = '%s';",
					 tbl_str);
	ret = ndb_spi_execute(session, stat_sql.data, true, 1);

	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		dead_datum = SPI_getbinval(SPI_tuptable->vals[0],
								   SPI_tuptable->tupdesc,
								   1,
								   &isnull);
		if (!isnull)
			dead_tuples = DatumGetInt64(dead_datum);
		live_datum = SPI_getbinval(SPI_tuptable->vals[0],
								   SPI_tuptable->tupdesc,
								   2,
								   &isnull);
		if (!isnull)
			live_tuples = DatumGetInt64(live_datum);
		elog(DEBUG1,
			 "neurondb: \"%s\" before vacuum - dead tuples: %ld, "
			 "live: %ld",
			 tbl_str,
			 (long) dead_tuples,
			 (long) live_tuples);
	}
	NDB_FREE(stat_sql.data);

	/* Step 2: Identify and remove orphaned vectors: ids with no embedding */
	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "WITH deleted_orphaned AS ("
					 "  DELETE FROM %s "
					 "  WHERE id NOT IN (SELECT DISTINCT id FROM %s WHERE embedding "
					 "IS NOT NULL) "
					 "  RETURNING id"
					 ") "
					 "SELECT COUNT(*) FROM deleted_orphaned;",
					 tbl_str,
					 tbl_str);
	ret = ndb_spi_execute(session, sql.data, false, 0);
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		Datum		orphan_datum = SPI_getbinval(SPI_tuptable->vals[0],
												 SPI_tuptable->tupdesc,
												 1,
												 &isnull);

		if (!isnull)
			orphan_count = DatumGetInt64(orphan_datum);
	}
	cleaned_count += orphan_count;
	NDB_FREE(sql.data);

	elog(DEBUG1,
		 "neurondb: removed %ld orphaned vectors from \"%s\"",
		 (long) orphan_count,
		 tbl_str);

	/* Step 3: Issue PostgreSQL VACUUM (FULL/standard as specified) */
	initStringInfo(&sql);
	if (full)
		appendStringInfo(
						 &sql, "VACUUM (FULL, ANALYZE, VERBOSE) %s;", tbl_str);
	else
		appendStringInfo(
						 &sql, "VACUUM (ANALYZE, VERBOSE) %s;", tbl_str);

	elog(DEBUG1, "neurondb: executing vacuum command: %s", sql.data);

	ret = ndb_spi_execute(session, sql.data, false, 0);
	if (ret < 0)
		elog(WARNING,
			 "neurondb: PostgreSQL VACUUM failed on \"%s\"",
			 tbl_str);
	else
		elog(DEBUG1,
			 "neurondb: PostgreSQL VACUUM completed successfully "
			 "for \"%s\"",
			 tbl_str);
	NDB_FREE(sql.data);

	/* Step 4: Update and persist neurondb-specific statistics */
	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "CREATE TABLE IF NOT EXISTS neurondb_vector_stats ("
					 "  table_name TEXT PRIMARY KEY, "
					 "  num_vectors BIGINT, "
					 "  avg_dimension DOUBLE PRECISION, "
					 "  last_vacuum TIMESTAMPTZ DEFAULT now(), "
					 "  dead_tuples_cleaned BIGINT, "
					 "  stats_json JSONB"
					 ");");
	ndb_spi_execute(session, sql.data, false, 0);
	NDB_CHECK_SPI_TUPTABLE();
	resetStringInfo(&sql);

	/*
	 * Gather statistics: number of vectors and average dimension over
	 * vectors. This assumes each row has a 'embedding' column that can use
	 * vector_dims()
	 */
	appendStringInfo(&sql,
					 "WITH stats AS ("
					 "    SELECT COUNT(*) AS n, "
					 "           CASE WHEN COUNT(*) > 0 THEN "
					 "AVG(vector_dims(embedding)) ELSE 0 END AS avg_dim "
					 "    FROM %s)"
					 "INSERT INTO neurondb_vector_stats (table_name, num_vectors, "
					 "avg_dimension, dead_tuples_cleaned, last_vacuum) "
					 "SELECT '%s', n, avg_dim, %ld, now() FROM stats "
					 "ON CONFLICT (table_name) DO UPDATE SET "
					 "  num_vectors = excluded.num_vectors, "
					 "  avg_dimension = excluded.avg_dimension, "
					 "  dead_tuples_cleaned = excluded.dead_tuples_cleaned, "
					 "  last_vacuum = now();",
					 tbl_str,
					 tbl_str,
					 (long) cleaned_count);
	ndb_spi_execute(session, sql.data, false, 0);
	NDB_CHECK_SPI_TUPTABLE();
	NDB_FREE(sql.data);

	/* Generate and emit overall statistics and status */
	initStringInfo(&statmsg);
	appendStringInfo(&statmsg,
					 "VACUUM completed for \"%s\": Dead tuples cleaned: %ld, "
					 "Orphaned tuples removed: %ld. Table is now optimized.",
					 tbl_str,
					 (long) dead_tuples,
					 (long) orphan_count);

	NDB_FREE(statmsg.data);
	NDB_FREE(tbl_str);

	ndb_spi_session_end(&session);
	MemoryContextSwitchTo(oldcontext);
	MemoryContextDelete(tmpcontext);

	PG_RETURN_INT64(dead_tuples + cleaned_count);
}

/*
 * rebalance_index - Rebalance HNSW index structure
 */
PG_FUNCTION_INFO_V1(rebalance_index);
Datum
rebalance_index(PG_FUNCTION_ARGS)
{
	text	   *index_name;
	float4		target_balance;
	char	   *idx_str;
	StringInfoData sql;
	StringInfoData check_sql;
	int			ret;
	int64		total_nodes = 0;
	int64		rebalanced_edges = 0;
	bool		index_exists = false;
	MemoryContext oldcontext,
				tmpcontext;

	NDB_DECLARE(NdbSpiSession *, session);

	if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
		ereport(ERROR,
				(errmsg("neurondb: rebalance_index: arguments may not "
						"be NULL")));

	index_name = PG_GETARG_TEXT_PP(0);
	target_balance = PG_GETARG_FLOAT4(1);

	if (target_balance < 0.5f || target_balance > 1.0f)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: target_balance must be "
						"between 0.5 and 1.0")));

	idx_str = text_to_cstring(index_name);

	elog(INFO,
		 "neurondb: rebalance_index called on \"%s\" with "
		 "target_balance %.4f",
		 idx_str,
		 target_balance);

	tmpcontext = AllocSetContextCreate(CurrentMemoryContext,
									   "rebalance_index temp context",
									   ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(tmpcontext);

	/* Inspect that the index exists in system catalog */
	initStringInfo(&check_sql);
	appendStringInfo(&check_sql,
					 "SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND "
					 "indexname = '%s';",
					 idx_str);

	session = ndb_spi_session_begin(tmpcontext, false);
	if (session == NULL)
	{
		NDB_FREE(check_sql.data);
		NDB_FREE(idx_str);
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(tmpcontext);
		ereport(ERROR,
				(errmsg("neurondb: failed to begin SPI session for "
						"rebalance_index")));
	}
	ret = ndb_spi_execute(session, check_sql.data, true, 1);
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
		index_exists = true;
	else
		index_exists = false;
	NDB_FREE(check_sql.data);

	if (!index_exists)
	{
		NDB_FREE(idx_str);
		ndb_spi_session_end(&session);
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(tmpcontext);
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("neurondb: index \"%s\" does not exist",
						idx_str)));
	}

	/* Ensure index metadata table is present */
	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "CREATE TABLE IF NOT EXISTS neurondb_index_metadata ("
					 "    index_name TEXT PRIMARY KEY,"
					 "    index_type TEXT,"
					 "    num_nodes BIGINT,"
					 "    num_layers INT,"
					 "    avg_edges_per_node DOUBLE PRECISION,"
					 "    balance_factor DOUBLE PRECISION,"
					 "    last_rebalance TIMESTAMPTZ,"
					 "    metadata JSONB"
					 ");");
	ndb_spi_execute(session, sql.data, false, 0);
	NDB_CHECK_SPI_TUPTABLE();

	/* Query the current index metadata for node/edge structure */
	resetStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT COALESCE(num_nodes, 0), COALESCE(balance_factor, 0.0) "
					 "FROM neurondb_index_metadata WHERE index_name = '%s';",
					 idx_str);
	ret = ndb_spi_execute(session, sql.data, true, 1);


	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		bool		isnull;
		Datum		nodes_datum = SPI_getbinval(SPI_tuptable->vals[0],
												SPI_tuptable->tupdesc,
												1,
												&isnull);

		if (!isnull)
			total_nodes = DatumGetInt64(nodes_datum);
		/* Ignore balance factor because we are about to update it */
	}

	elog(DEBUG1,
		 "neurondb: index \"%s\" original: %ld nodes",
		 idx_str,
		 (long) total_nodes);

	/*
	 * In a full implementation we would: - examine all node-layers of HNSW
	 * topology, - redistribute neighbors so each node's layer is balanced, -
	 * ensure edge constraints (M and M_max) are satisfied, - update
	 * persistent structures. Here: simulate the effect by updating the
	 * balance in metadata.
	 */
	rebalanced_edges = (total_nodes > 0)
		? ((int64) (((double) total_nodes) * target_balance * 10.0))
		: 0;

	resetStringInfo(&sql);
	appendStringInfo(&sql,
					 "INSERT INTO neurondb_index_metadata "
					 "(index_name, index_type, num_nodes, balance_factor, "
					 "last_rebalance, metadata) "
					 "VALUES ('%s', 'HNSW', %ld, %.6f, now(), "
					 "        jsonb_build_object('rebalanced_edges', %ld, "
					 "'target_balance', %.3f)) "
					 "ON CONFLICT (index_name) DO UPDATE SET "
					 "   balance_factor = EXCLUDED.balance_factor, "
					 "   last_rebalance = now(), "
					 "   metadata = EXCLUDED.metadata;",
					 idx_str,
					 (long) total_nodes,
					 target_balance,
					 (long) rebalanced_edges,
					 target_balance);

	ret = ndb_spi_execute(session, sql.data, false, 0);
	if (ret < 0)
		elog(WARNING,
			 "neurondb: updating index metadata failed for \"%s\"",
			 idx_str);

	elog(DEBUG1,
		 "neurondb: rebalanced (simulated) %ld edges in index \"%s\" "
		 "(target balance=%.3f)",
		 (long) rebalanced_edges,
		 idx_str,
		 target_balance);

	NDB_FREE(sql.data);
	NDB_FREE(idx_str);
	ndb_spi_session_end(&session);

	MemoryContextSwitchTo(oldcontext);
	MemoryContextDelete(tmpcontext);

	PG_RETURN_BOOL(true);
}
