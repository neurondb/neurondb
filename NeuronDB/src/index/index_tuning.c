/*-------------------------------------------------------------------------
 *
 * index_tuning.c
 *	  Automatic index parameter optimization and cost-based selection
 *
 * This file implements automatic tuning for vector indexes including:
 * - Parameter optimization (m, ef_construction for HNSW; lists for IVF)
 * - Automatic index type selection (HNSW vs IVF)
 * - Cost-based index selection
 * - Query pattern analysis for tuning
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/index/index_tuning.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "executor/spi.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "lib/stringinfo.h"
#include "access/reloptions.h"
#include "catalog/pg_class.h"
#include "catalog/pg_index.h"
#include "utils/rel.h"
#include "utils/lsyscache.h"
#include "utils/guc.h"
#include <math.h>
#include <stdlib.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"
#include "neurondb_constants.h"

/*
 * Automatically optimize HNSW parameters based on dataset characteristics
 * Returns JSONB with recommended parameters
 */
PG_FUNCTION_INFO_V1(index_tune_hnsw);
Datum
index_tune_hnsw(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_P(0);
	text	   *column_name = PG_GETARG_TEXT_P(1);
	char	   *tbl_name;
	char	   *col_name;
	StringInfoData sql;
	int			ret;
	int64		row_count = 0;
	int32		vector_dim = 0;
	double		memory_budget_mb = 0.0;
	int			recommended_m = 16;
	int			recommended_ef_construction = 64;
	StringInfoData json_buf;
	Jsonb	   *result_jsonb;
	NDB_DECLARE(NdbSpiSession *, session);

	tbl_name = text_to_cstring(table_name);
	col_name = text_to_cstring(column_name);

	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("index_tune_hnsw: failed to begin SPI session")));

	/* Get table row count */
	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT COUNT(*) FROM %s",
					 quote_identifier(tbl_name));

	ret = ndb_spi_execute(session, sql.data, true, 1);
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		bool		isnull;
		Datum		count_datum = SPI_getbinval(SPI_tuptable->vals[0],
												SPI_tuptable->tupdesc,
												1,
												&isnull);

		if (!isnull)
			row_count = DatumGetInt64(count_datum);
	}

	/* Get vector dimension */
	/* Use safe free/reinit to handle potential memory context changes */
	NDB_FREE(sql.data);
	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT vector_dims(%s) FROM %s LIMIT 1",
					 quote_identifier(col_name),
					 quote_identifier(tbl_name));

	ret = ndb_spi_execute_safe(sql.data, true, 1);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		bool		isnull;
		Datum		dim_datum = SPI_getbinval(SPI_tuptable->vals[0],
											  SPI_tuptable->tupdesc,
											  1,
											  &isnull);

		if (!isnull)
			vector_dim = DatumGetInt32(dim_datum);
	}

	/* Estimate memory budget (use work_mem if available) */
	{
		const char *work_mem_str = GetConfigOption("work_mem", true, false);

		if (work_mem_str)
		{
			memory_budget_mb = strtod(work_mem_str, NULL) / 1024.0; /* Convert KB to MB */
		}
		else
		{
			memory_budget_mb = 64.0;	/* Default 64MB */
		}
	}

	/* Parameter optimization heuristics */
	if (row_count < 10000)
	{
		/* Small dataset: prioritize speed */
		recommended_m = 12;
		recommended_ef_construction = 32;
	}
	else if (row_count < 100000)
	{
		/* Medium dataset: balanced */
		recommended_m = 16;
		recommended_ef_construction = 64;
	}
	else if (row_count < 1000000)
	{
		/* Large dataset: prioritize recall */
		recommended_m = 24;
		recommended_ef_construction = 128;
	}
	else
	{
		/* Very large dataset: high recall */
		recommended_m = 32;
		recommended_ef_construction = 200;
	}

	/* Adjust based on memory budget */
	{
		double		estimated_index_size_mb;
		double		bytes_per_vector;

		/* Rough estimate: m * ef_construction * dim * 4 bytes */
		bytes_per_vector = (double) recommended_m * (double) recommended_ef_construction *
			(double) vector_dim * 4.0;
		estimated_index_size_mb = (bytes_per_vector * (double) row_count) / (1024.0 * 1024.0);

		if (estimated_index_size_mb > memory_budget_mb * 0.8)
		{
			/* Reduce parameters to fit memory */
			recommended_m = (int) (recommended_m * 0.75);
			recommended_ef_construction = (int) (recommended_ef_construction * 0.75);
			if (recommended_m < 8)
				recommended_m = 8;
			if (recommended_ef_construction < 32)
				recommended_ef_construction = 32;
		}
	}

	/* Build JSONB result */
	initStringInfo(&json_buf);
	appendStringInfo(&json_buf,
					 "{\"index_type\":\"hnsw\","
					 "\"recommended_m\":%d,"
					 "\"recommended_ef_construction\":%d,"
					 "\"estimated_index_size_mb\":%.2f,"
					 "\"row_count\":%lld,"
					 "\"vector_dim\":%d,"
					 "\"memory_budget_mb\":%.2f}",
					 recommended_m,
					 recommended_ef_construction,
					 (double) recommended_m * (double) recommended_ef_construction *
					 (double) vector_dim * 4.0 * (double) row_count / (1024.0 * 1024.0),
					 (long long) row_count,
					 vector_dim,
					 memory_budget_mb);

	result_jsonb = DatumGetJsonbP(DirectFunctionCall1(
													  jsonb_in, CStringGetTextDatum(json_buf.data)));

	NDB_FREE(json_buf.data);
	NDB_FREE(tbl_name);
	NDB_FREE(col_name);
	NDB_FREE(sql.data);
	ndb_spi_session_end(&session);

	PG_RETURN_POINTER(result_jsonb);
}

/*
 * Automatically optimize IVF parameters based on dataset characteristics
 */
PG_FUNCTION_INFO_V1(index_tune_ivf);
Datum
index_tune_ivf(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_P(0);
	text	   *column_name = PG_GETARG_TEXT_P(1);
	char	   *tbl_name;
	char	   *col_name;
	StringInfoData sql;
	int			ret;
	int64		row_count = 0;
	int32		vector_dim = 0;
	int			recommended_lists = 100;
	StringInfoData json_buf;
	Jsonb	   *result_jsonb;
	NDB_DECLARE(NdbSpiSession *, session);

	tbl_name = text_to_cstring(table_name);
	col_name = text_to_cstring(column_name);

	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("index_tune_ivf: failed to begin SPI session")));

	/* Get table row count */
	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT COUNT(*) FROM %s",
					 quote_identifier(tbl_name));

	ret = ndb_spi_execute(session, sql.data, true, 1);
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		bool		isnull;
		Datum		count_datum = SPI_getbinval(SPI_tuptable->vals[0],
												SPI_tuptable->tupdesc,
												1,
												&isnull);

		if (!isnull)
			row_count = DatumGetInt64(count_datum);
	}

	/* Get vector dimension */
	/* Use safe free/reinit to handle potential memory context changes */
	NDB_FREE(sql.data);
	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT vector_dims(%s) FROM %s LIMIT 1",
					 quote_identifier(col_name),
					 quote_identifier(tbl_name));

	ret = ndb_spi_execute(session, sql.data, true, 1);
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		bool		isnull;
		Datum		dim_datum = SPI_getbinval(SPI_tuptable->vals[0],
											  SPI_tuptable->tupdesc,
											  1,
											  &isnull);

		if (!isnull)
			vector_dim = DatumGetInt32(dim_datum);
	}

	/* Parameter optimization: lists = sqrt(row_count) is a common heuristic */
	if (row_count > 0)
	{
		recommended_lists = (int) sqrt((double) row_count);
		/* Clamp to reasonable range */
		if (recommended_lists < 10)
			recommended_lists = 10;
		if (recommended_lists > 1000)
			recommended_lists = 1000;
		/* Round to nearest 10 */
		recommended_lists = ((recommended_lists + 5) / 10) * 10;
	}

	/* Build JSONB result */
	initStringInfo(&json_buf);
	appendStringInfo(&json_buf,
					 "{\"index_type\":\"ivf\","
					 "\"recommended_lists\":%d,"
					 "\"recommended_probes\":%d,"
					 "\"row_count\":%lld,"
					 "\"vector_dim\":%d}",
					 recommended_lists,
					 recommended_lists / 10,	/* probes = lists / 10 is
												 * common */
					 (long long) row_count,
					 vector_dim);

	result_jsonb = DatumGetJsonbP(DirectFunctionCall1(
													  jsonb_in, CStringGetTextDatum(json_buf.data)));

	NDB_FREE(json_buf.data);
	NDB_FREE(tbl_name);
	NDB_FREE(col_name);
	NDB_FREE(sql.data);
	ndb_spi_session_end(&session);

	PG_RETURN_POINTER(result_jsonb);
}

/*
 * Recommend index type (HNSW vs IVF) based on dataset and workload
 */
PG_FUNCTION_INFO_V1(index_recommend_type);
Datum
index_recommend_type(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_P(0);
	text	   *column_name = PG_GETARG_TEXT_P(1);
	char	   *tbl_name;
	char	   *col_name;
	StringInfoData sql;
	int			ret;
	int64		row_count = 0;
	int64		update_frequency = 0;
	double		memory_budget_mb = 0.0;
	char	   *recommended_type = "hnsw";
	StringInfoData json_buf;
	Jsonb	   *result_jsonb;
	Jsonb	   *hnsw_params;
	Jsonb	   *ivf_params;
	NDB_DECLARE(NdbSpiSession *, session);

	tbl_name = text_to_cstring(table_name);
	col_name = text_to_cstring(column_name);

	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("index_recommend_type: failed to begin SPI session")));

	/* Get table row count */
	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT COUNT(*) FROM %s",
					 quote_identifier(tbl_name));

	ret = ndb_spi_execute(session, sql.data, true, 1);
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		bool		isnull;
		Datum		count_datum = SPI_getbinval(SPI_tuptable->vals[0],
												SPI_tuptable->tupdesc,
												1,
												&isnull);

		if (!isnull)
			row_count = DatumGetInt64(count_datum);
	}

	/* Estimate update frequency (check for recent updates) */
	/* Use safe free/reinit to handle potential memory context changes */
	NDB_FREE(sql.data);
	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT COUNT(*) FROM %s WHERE "
					 "ctid >= (SELECT ctid FROM %s ORDER BY ctid DESC LIMIT 1 OFFSET %lld)",
					 quote_identifier(tbl_name),
					 quote_identifier(tbl_name),
					 (long long) (row_count / 10)); /* Check last 10% */

	ret = ndb_spi_execute(session, sql.data, true, 1);
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		bool		isnull;
		Datum		update_datum = SPI_getbinval(SPI_tuptable->vals[0],
												 SPI_tuptable->tupdesc,
												 1,
												 &isnull);

		if (!isnull)
			update_frequency = DatumGetInt64(update_datum);
	}

	{
		const char *work_mem_str = GetConfigOption("work_mem", true, false);

		if (work_mem_str)
		{
			memory_budget_mb = strtod(work_mem_str, NULL) / 1024.0; /* Convert KB to MB */
		}
		else
		{
			memory_budget_mb = 64.0;	/* Default 64MB */
		}
	}

	/* Decision logic */
	if (row_count > 10000000)
	{
		/* Very large dataset: prefer IVF */
		recommended_type = "ivf";
	}
	else if (update_frequency > row_count / 10)
	{
		/* High update frequency: prefer IVF (faster rebuild) */
		recommended_type = "ivf";
	}
	else
	{
		/* Default to HNSW for better recall and query performance */
		recommended_type = "hnsw";
	}

	/* Get recommended parameters for both types */
	hnsw_params = DatumGetJsonbP(DirectFunctionCall2(
													 index_tune_hnsw, PointerGetDatum(table_name), PointerGetDatum(column_name)));

	ivf_params = DatumGetJsonbP(DirectFunctionCall2(
													index_tune_ivf, PointerGetDatum(table_name), PointerGetDatum(column_name)));

	/* Build JSONB result */
	initStringInfo(&json_buf);
	appendStringInfo(&json_buf,
					 "{\"recommended_type\":\"%s\","
					 "\"row_count\":%lld,"
					 "\"update_frequency\":%lld,"
					 "\"memory_budget_mb\":%.2f,"
					 "\"hnsw_params\":%s,"
					 "\"ivf_params\":%s}",
					 recommended_type,
					 (long long) row_count,
					 (long long) update_frequency,
					 memory_budget_mb,
					 JsonbToCString(NULL, &hnsw_params->root, VARSIZE(hnsw_params)),
					 JsonbToCString(NULL, &ivf_params->root, VARSIZE(ivf_params)));

	result_jsonb = DatumGetJsonbP(DirectFunctionCall1(
													  jsonb_in, CStringGetTextDatum(json_buf.data)));

	NDB_FREE(json_buf.data);
	NDB_FREE(tbl_name);
	NDB_FREE(col_name);
	NDB_FREE(sql.data);
	ndb_spi_session_end(&session);

	PG_RETURN_POINTER(result_jsonb);
}

/*
 * Automatically tune ef_search/probes based on query performance history
 */
PG_FUNCTION_INFO_V1(index_tune_query_params);
Datum
index_tune_query_params(PG_FUNCTION_ARGS)
{
	text	   *index_name = PG_GETARG_TEXT_P(0);
	char	   *idx_name;
	StringInfoData sql;
	int			ret;
	double		avg_latency = 0.0;
	double		avg_recall = 0.0;
	int			recommended_ef_search = 64;
	int			recommended_probes = 10;
	char	   *index_type = "hnsw";
	StringInfoData json_buf;
	Jsonb	   *result_jsonb;
	NDB_DECLARE(NdbSpiSession *, session);

	idx_name = text_to_cstring(index_name);

	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("index_tune_query_params: failed to begin SPI session")));

	/* Determine index type */
	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT amname FROM pg_am am "
					 "JOIN pg_class c ON c.relam = am.oid "
					 "WHERE c.relname = %s",
					 quote_literal_cstr(idx_name));

	ret = ndb_spi_execute(session, sql.data, true, 1);
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		bool		isnull;
		Datum		amname_datum = SPI_getbinval(SPI_tuptable->vals[0],
												 SPI_tuptable->tupdesc,
												 1,
												 &isnull);

		if (!isnull)
		{
			char	   *amname = text_to_cstring(DatumGetTextP(amname_datum));

			if (strcmp(amname, "hnsw") == 0)
				index_type = "hnsw";
			else if (strcmp(amname, "ivfflat") == 0)
				index_type = "ivf";
			NDB_FREE(amname);
		}
	}

	/* Get query performance metrics */
	/* Use safe free/reinit to handle potential memory context changes */
	NDB_FREE(sql.data);
	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT AVG(latency_ms) as avg_latency, "
					 "       AVG(recall_at_k) as avg_recall "
					 "FROM %s.%s "
					 "WHERE query_timestamp > now() - interval '1 hour' "
					 "  AND recall_at_k IS NOT NULL",
					 quote_identifier(NDB_SCHEMA_NAME),
					 quote_identifier(NDB_TABLE_QUERY_METRICS));

	ret = ndb_spi_execute(session, sql.data, true, 1);
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		bool		isnull1,
					isnull2;
		Datum		latency_datum = SPI_getbinval(SPI_tuptable->vals[0],
												  SPI_tuptable->tupdesc,
												  1,
												  &isnull1);
		Datum		recall_datum = SPI_getbinval(SPI_tuptable->vals[0],
												 SPI_tuptable->tupdesc,
												 2,
												 &isnull2);

		if (!isnull1)
			avg_latency = DatumGetFloat8(latency_datum);
		if (!isnull2)
			avg_recall = DatumGetFloat8(recall_datum);
	}

	/* Tune parameters based on performance */
	if (strcmp(index_type, "hnsw") == 0)
	{
		recommended_ef_search = 64; /* default */

		if (avg_recall < 0.90 && avg_latency < 100.0)
		{
			/* Low recall, acceptable latency: increase ef_search */
			recommended_ef_search = 128;
		}
		else if (avg_recall >= 0.95 && avg_latency > 50.0)
		{
			/* Good recall, high latency: decrease ef_search */
			recommended_ef_search = 32;
		}
	}
	else if (strcmp(index_type, "ivf") == 0)
	{
		recommended_probes = 10;	/* default */

		if (avg_recall < 0.90 && avg_latency < 100.0)
		{
			/* Low recall, acceptable latency: increase probes */
			recommended_probes = 20;
		}
		else if (avg_recall >= 0.95 && avg_latency > 50.0)
		{
			/* Good recall, high latency: decrease probes */
			recommended_probes = 5;
		}
	}

	/* Build JSONB result */
	initStringInfo(&json_buf);
	if (strcmp(index_type, "hnsw") == 0)
	{
		appendStringInfo(&json_buf,
						 "{\"index_type\":\"hnsw\","
						 "\"recommended_ef_search\":%d,"
						 "\"avg_latency_ms\":%.2f,"
						 "\"avg_recall\":%.3f}",
						 recommended_ef_search,
						 avg_latency,
						 avg_recall);
	}
	else
	{
		appendStringInfo(&json_buf,
						 "{\"index_type\":\"ivf\","
						 "\"recommended_probes\":%d,"
						 "\"avg_latency_ms\":%.2f,"
						 "\"avg_recall\":%.3f}",
						 recommended_probes,
						 avg_latency,
						 avg_recall);
	}

	result_jsonb = DatumGetJsonbP(DirectFunctionCall1(
													  jsonb_in, CStringGetTextDatum(json_buf.data)));

	NDB_FREE(json_buf.data);
	NDB_FREE(idx_name);
	NDB_FREE(sql.data);
	ndb_spi_session_end(&session);

	PG_RETURN_POINTER(result_jsonb);
}
